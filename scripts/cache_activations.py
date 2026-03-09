#!/usr/bin/env python
"""
Phase 1: Cache SAE activations for the entire RLAIF-V-Dataset.

Runs the LLaVA model + SAE encoder on each sample and saves per-sample
feature-level summaries to disk.  The cached data can then be loaded by
compute_feature_metrics.py (Phase 2) for fast metric computation
without re-running model inference.

What is cached per sample (all float16 for space efficiency):
  - text_act_sum   : (d_sae,) sum of activations on text tokens
  - image_act_sum  : (d_sae,) sum of activations on image tokens
  - n_text         : int, number of text tokens
  - n_image        : int, number of image tokens
  - alignment      : (d_sae,) cosine similarity for features active on both modalities
  - alignment_mask : (d_sae,) bool, which features have valid alignment scores
  - active_mask    : (d_sae,) bool, which features fire at all in this sample

Storage estimate:
  Per sample: ~(3 * 65536 * 2) + (3 * 65536 / 8) ≈ 393KB + 24KB ≈ 417KB
  1000 samples: ~417 MB
  Full dataset (83k): ~34 GB

Usage:
    # Quick test (2 samples)
    python scripts/cache_activations.py --n_samples 2 --output_dir output/activation_cache

    # Full 1000-sample run
    python scripts/cache_activations.py --n_samples 1000 \\
        --dataset_path dataset/RLAIF-V-Dataset1k \\
        --output_dir output/activation_cache --dtype float16

    # Resume from checkpoint
    python scripts/cache_activations.py --n_samples 1000 \\
        --output_dir output/activation_cache --resume
"""

import os
import sys
import gc
import argparse
import time
import json

# Environment setup
os.environ.setdefault("PYTORCH_MPS_HIGH_WATERMARK_RATIO", "0.0")
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")

import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sae_lens import SAE
from transformers import LlavaNextForConditionalGeneration, LlavaNextProcessor
from transformer_lens.HookedLlava import HookedLlava


# ---------------------------------------------------------------------------
# Device helpers
# ---------------------------------------------------------------------------

def get_device():
    if torch.cuda.is_available():
        return "cuda:0"
    elif torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _empty_cache(device):
    if device == "mps":
        torch.mps.empty_cache()
    elif "cuda" in str(device):
        torch.cuda.empty_cache()


def _gpu_mem_str(device):
    if "cuda" in str(device):
        alloc = torch.cuda.memory_allocated(device) / 1e9
        total = torch.cuda.get_device_properties(device).total_memory / 1e9
        return f"[GPU {alloc:.1f}/{total:.1f} GB]"
    return ""


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_models(model_path, sae_path, device, dtype=torch.float32):
    print(f"[1/4] Loading processor from {model_path} ...")
    processor = LlavaNextProcessor.from_pretrained(model_path)

    print(f"[2/4] Loading LLaVA-NeXT on CPU ...")
    vision_model = LlavaNextForConditionalGeneration.from_pretrained(
        model_path, torch_dtype=dtype, low_cpu_mem_usage=True,
    )

    language_model = vision_model.language_model
    vision_tower = vision_model.vision_tower
    multi_modal_projector = vision_model.multi_modal_projector

    print(f"[3/4] Creating HookedLlava on CPU ...")
    hook_language_model = HookedLlava.from_pretrained_no_processing(
        "llava-hf/llava-v1.6-mistral-7b-hf",
        hf_model=language_model,
        device="cpu",
        fold_ln=False,
        center_writing_weights=False,
        center_unembed=False,
        tokenizer=processor.tokenizer,  # pass local tokenizer to avoid HF downloads
        dtype=dtype,
        vision_tower=vision_tower,
        multi_modal_projector=multi_modal_projector,
        n_devices=1,
    )

    del vision_model, language_model, vision_tower, multi_modal_projector
    gc.collect()
    _empty_cache(device)

    print(f"     Moving model to {device} ...  {_gpu_mem_str(device)}")
    hook_language_model = hook_language_model.to(device)
    gc.collect()
    _empty_cache(device)
    print(f"     Model on device.  {_gpu_mem_str(device)}")

    print(f"[4/4] Loading SAE from {sae_path} ...")
    sae = SAE.load_from_pretrained(path=sae_path, device=device)
    sae.cfg.device = device
    print(f"     SAE loaded.  {_gpu_mem_str(device)}")

    return processor, hook_language_model, sae


# ---------------------------------------------------------------------------
# Input preparation
# ---------------------------------------------------------------------------

def prepare_input(processor, image, question, device):
    system_prompt = " "
    user_prompt = f"USER: \n<image> {question}"
    assistant_prompt = "\nASSISTANT: "
    formatted_prompt = f"{system_prompt}{user_prompt}{assistant_prompt}"

    image = image.resize((336, 336)).convert("RGB")
    text_inputs = processor.tokenizer(formatted_prompt, return_tensors="pt")
    image_inputs = processor.image_processor(images=image, return_tensors="pt")

    inputs = {
        "input_ids": text_inputs["input_ids"],
        "attention_mask": text_inputs["attention_mask"],
        "pixel_values": image_inputs["pixel_values"],
        "image_sizes": image_inputs["image_sizes"],
    }
    return {k: v.to(device) for k, v in inputs.items()}


# ---------------------------------------------------------------------------
# Process one sample → per-feature summary dict
# ---------------------------------------------------------------------------

def process_sample(hook_language_model, sae, processor, sample,
                   device, hook_name, stop_at_layer):
    """Run model + SAE on one sample and return per-feature summary."""

    inputs = prepare_input(processor, sample["image"], sample["question"], device)

    with torch.no_grad():
        out, cache = hook_language_model.run_with_cache(
            input=inputs,
            model_inputs=inputs,
            vision=True,
            prepend_bos=True,
            names_filter=lambda name: name == hook_name,
            stop_at_layer=stop_at_layer,
        )
        image_indice = out[1]  # (1, n_image_tokens)

        hidden_states = cache[hook_name].to(device)       # (1, seq, d_in)
        feature_acts = sae.encode(hidden_states)           # (1, seq, d_sae)

        del cache, out

    seq_len = feature_acts.shape[1]
    d_sae = feature_acts.shape[2]

    # Build masks
    image_positions = set(image_indice[0].cpu().tolist())
    image_mask = torch.zeros(seq_len, dtype=torch.bool, device=device)
    for p in image_positions:
        image_mask[p] = True
    text_mask = ~image_mask

    n_text = text_mask.sum().item()
    n_image = image_mask.sum().item()

    fa = feature_acts[0]   # (seq, d_sae) on device
    hs = hidden_states[0]  # (seq, d_in)  on device

    # 1. Activation sums per modality
    text_act_sum = fa[text_mask].sum(dim=0).cpu().half().numpy()
    image_act_sum = fa[image_mask].sum(dim=0).cpu().half().numpy()

    # 2. Active feature mask
    active_mask = (fa.abs().sum(dim=0) > 0).cpu().numpy()

    # 3. Cross-modal alignment
    text_acts = fa[text_mask]     # (n_text, d_sae)
    image_acts = fa[image_mask]   # (n_image, d_sae)

    text_active = (text_acts > 0).any(dim=0)
    image_active = (image_acts > 0).any(dim=0)
    both_active = text_active & image_active
    active_idx = both_active.nonzero().squeeze(-1)

    alignment = np.zeros(d_sae, dtype=np.float16)
    alignment_mask = np.zeros(d_sae, dtype=np.bool_)

    if active_idx.numel() > 0:
        text_hidden = hs[text_mask]    # (n_text, d_in)
        image_hidden = hs[image_mask]  # (n_image, d_in)

        ta_sub = text_acts[:, active_idx].float()
        ia_sub = image_acts[:, active_idx].float()

        text_weighted = (ta_sub.T @ text_hidden.float()) / (ta_sub.sum(0).unsqueeze(1) + 1e-8)
        image_weighted = (ia_sub.T @ image_hidden.float()) / (ia_sub.sum(0).unsqueeze(1) + 1e-8)

        cos_sim = F.cosine_similarity(text_weighted, image_weighted, dim=1)

        idx_np = active_idx.cpu().numpy()
        alignment[idx_np] = cos_sim.cpu().half().numpy()
        alignment_mask[idx_np] = True

    # Cleanup
    del fa, hs, feature_acts, hidden_states, text_acts, image_acts, inputs, image_indice
    gc.collect()
    _empty_cache(device)

    return {
        "text_act_sum": text_act_sum,
        "image_act_sum": image_act_sum,
        "n_text": n_text,
        "n_image": n_image,
        "alignment": alignment,
        "alignment_mask": alignment_mask,
        "active_mask": active_mask,
    }


# ---------------------------------------------------------------------------
# Save / Load chunk
# ---------------------------------------------------------------------------

CHUNK_SIZE = 100  # samples per chunk file


def save_chunk(chunk_data, chunk_idx, output_dir):
    """Save a chunk of samples to a single .npz file."""
    fname = os.path.join(output_dir, f"chunk_{chunk_idx:04d}.npz")

    # Stack arrays from all samples in chunk
    text_act_sums = np.stack([s["text_act_sum"] for s in chunk_data])
    image_act_sums = np.stack([s["image_act_sum"] for s in chunk_data])
    n_texts = np.array([s["n_text"] for s in chunk_data], dtype=np.int32)
    n_images = np.array([s["n_image"] for s in chunk_data], dtype=np.int32)
    alignments = np.stack([s["alignment"] for s in chunk_data])
    alignment_masks = np.stack([s["alignment_mask"] for s in chunk_data])
    active_masks = np.stack([s["active_mask"] for s in chunk_data])

    np.savez_compressed(
        fname,
        text_act_sum=text_act_sums,       # (chunk_size, d_sae) float16
        image_act_sum=image_act_sums,      # (chunk_size, d_sae) float16
        n_text=n_texts,                    # (chunk_size,) int32
        n_image=n_images,                  # (chunk_size,) int32
        alignment=alignments,              # (chunk_size, d_sae) float16
        alignment_mask=alignment_masks,    # (chunk_size, d_sae) bool
        active_mask=active_masks,          # (chunk_size, d_sae) bool
    )
    return fname


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def cache_activations(args):
    device = get_device()
    dtype = torch.float16 if args.dtype == "float16" else torch.float32
    print(f"Device: {device},  dtype: {dtype}")
    print(f"Output: {args.output_dir}")
    print()

    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    model_path = os.path.join(project_root, args.model_path) if not os.path.isabs(args.model_path) else args.model_path
    sae_path = os.path.join(project_root, args.sae_path) if not os.path.isabs(args.sae_path) else args.sae_path

    # ---- Load models ----
    processor, hook_language_model, sae = load_models(model_path, sae_path, device, dtype)

    d_sae = sae.cfg.d_sae
    d_in = sae.cfg.d_in
    hook_name = sae.cfg.hook_name
    stop_at_layer = int(hook_name.split(".")[1]) + 1

    print(f"d_sae={d_sae}, d_in={d_in}, hook={hook_name}, stop_at_layer={stop_at_layer}")

    # ---- Load dataset ----
    from datasets import load_dataset, load_from_disk
    if args.dataset_path and os.path.isdir(args.dataset_path):
        print(f"\nLoading dataset from: {args.dataset_path}")
        ds = load_from_disk(args.dataset_path)
        if args.n_samples and args.n_samples < len(ds):
            ds = ds.select(range(args.n_samples))
    else:
        n = args.n_samples or 1000
        print(f"\nDownloading {n} samples from openbmb/RLAIF-V-Dataset ...")
        ds = load_dataset("openbmb/RLAIF-V-Dataset", split=f"train[:{n}]")

    print(f"Dataset: {len(ds)} samples.  {_gpu_mem_str(device)}")

    # ---- Check for resume ----
    os.makedirs(args.output_dir, exist_ok=True)
    start_idx = 0
    if args.resume:
        meta_path = os.path.join(args.output_dir, "meta.json")
        if os.path.exists(meta_path):
            with open(meta_path) as f:
                meta = json.load(f)
            start_idx = meta.get("n_processed", 0)
            print(f"Resuming from sample {start_idx}")

    # ---- Process samples ----
    chunk_data = []
    chunk_idx = start_idx // CHUNK_SIZE
    n_processed = start_idx
    n_errors = 0
    timings = []

    t_total = time.time()

    for idx in tqdm(range(start_idx, len(ds)), desc="Caching activations",
                    initial=start_idx, total=len(ds)):
        sample = ds[idx]

        t_sample = time.time()
        try:
            result = process_sample(
                hook_language_model, sae, processor, sample,
                device, hook_name, stop_at_layer,
            )
            chunk_data.append(result)
            n_processed += 1
            timings.append(time.time() - t_sample)
        except Exception as e:
            print(f"  [SKIP] sample {idx}: {e}")
            n_errors += 1
            continue

        # Save chunk when full
        if len(chunk_data) >= CHUNK_SIZE:
            fpath = save_chunk(chunk_data, chunk_idx, args.output_dir)
            chunk_idx += 1
            chunk_data = []

            # Save progress metadata
            meta = {
                "n_processed": n_processed,
                "n_errors": n_errors,
                "d_sae": d_sae,
                "d_in": d_in,
                "hook_name": hook_name,
                "dtype": args.dtype,
                "chunk_size": CHUNK_SIZE,
                "n_chunks": chunk_idx,
            }
            with open(os.path.join(args.output_dir, "meta.json"), "w") as f:
                json.dump(meta, f, indent=2)

            avg_time = np.mean(timings[-CHUNK_SIZE:])
            remaining = (len(ds) - idx - 1) * avg_time
            print(f"  Saved {fpath}  |  {avg_time:.2f}s/sample  |  "
                  f"ETA: {remaining/60:.1f} min  {_gpu_mem_str(device)}")

    # Save final partial chunk
    if chunk_data:
        fpath = save_chunk(chunk_data, chunk_idx, args.output_dir)
        chunk_idx += 1
        print(f"  Saved final chunk: {fpath}")

    # Save final metadata
    elapsed = time.time() - t_total
    meta = {
        "n_processed": n_processed,
        "n_errors": n_errors,
        "n_total": len(ds),
        "d_sae": d_sae,
        "d_in": d_in,
        "hook_name": hook_name,
        "dtype": args.dtype,
        "chunk_size": CHUNK_SIZE,
        "n_chunks": chunk_idx,
        "total_time_s": elapsed,
        "avg_time_per_sample_s": elapsed / max(n_processed, 1),
    }
    with open(os.path.join(args.output_dir, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    print(f"\n{'='*60}")
    print(f"Caching complete!")
    print(f"  Processed: {n_processed}/{len(ds)} samples ({n_errors} errors)")
    print(f"  Chunks: {chunk_idx} files in {args.output_dir}")
    print(f"  Total time: {elapsed:.1f}s ({elapsed/max(n_processed,1):.2f}s/sample)")
    if timings:
        print(f"  Avg time/sample: {np.mean(timings):.2f}s "
              f"(min={np.min(timings):.2f}s, max={np.max(timings):.2f}s)")
    print(f"{'='*60}")


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model_path", default="model/llava-v1.6-mistral-7b-hf")
    parser.add_argument("--sae_path", default="model/SAE-V/SAEV_LLaVA_NeXT-7b_OBELICS")
    parser.add_argument("--n_samples", type=int, default=None,
                        help="Number of samples (None = all available)")
    parser.add_argument("--dataset_path", default=None,
                        help="Path to pre-downloaded dataset (from save_to_disk)")
    parser.add_argument("--output_dir", default="output/activation_cache")
    parser.add_argument("--dtype", default="float16", choices=["float32", "float16"])
    parser.add_argument("--resume", action="store_true",
                        help="Resume from last checkpoint")
    args = parser.parse_args()
    cache_activations(args)


if __name__ == "__main__":
    main()

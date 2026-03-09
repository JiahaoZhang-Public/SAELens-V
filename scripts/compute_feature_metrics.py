#!/usr/bin/env python
"""
Compute per-feature modality metrics for SAE-V features.

For each of the 65536 SAE features, computes three metrics:
  1. Normalized Modality Ratio  — mean activation on image vs text tokens
     (normalized by token count to remove bias from 1176 image vs ~18 text tokens)
  2. Cross-Modal Alignment Score — cosine similarity of activation-weighted
     hidden states between text and image tokens
  3. Activation Frequency — fraction of samples where the feature fires

Usage:
    conda activate sae-v
    # Local (Apple Silicon):
    python scripts/compute_feature_metrics.py --n_samples 50
    # GPU server:
    python scripts/compute_feature_metrics.py --n_samples 500 --dtype float16

Output:
    output/feature_metrics/feature_metrics.npz
    output/feature_metrics/feature_3d_scatter.png
    output/feature_metrics/feature_3d_scatter.html
"""

import os
import sys
import gc
import argparse
import time

# Allow MPS to use full unified memory (Apple Silicon shares CPU/GPU memory)
os.environ.setdefault("PYTORCH_MPS_HIGH_WATERMARK_RATIO", "0.0")
# Disable HuggingFace XET CDN (causes 401 errors on some networks)
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
    """Release cached memory on MPS or CUDA."""
    if device == "mps":
        torch.mps.empty_cache()
    elif "cuda" in str(device):
        torch.cuda.empty_cache()


def _gpu_mem_str(device):
    """Return a string describing current GPU memory usage."""
    if "cuda" in str(device):
        alloc = torch.cuda.memory_allocated(device) / 1e9
        total = torch.cuda.get_device_properties(device).total_memory / 1e9
        return f"[GPU {alloc:.1f}/{total:.1f} GB]"
    return ""


# ---------------------------------------------------------------------------
# Model loading (MPS-safe: load on CPU first, then move)
# ---------------------------------------------------------------------------

def load_models(model_path, sae_path, device, dtype=torch.float32):
    print(f"[1/4] Loading processor from {model_path} ...")
    processor = LlavaNextProcessor.from_pretrained(model_path)

    print(f"[2/4] Loading LLaVA-NeXT on CPU ...")
    vision_model = LlavaNextForConditionalGeneration.from_pretrained(
        model_path, torch_dtype=dtype, low_cpu_mem_usage=True,
    )

    # Extract sub-modules BEFORE creating HookedLlava to help GC
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

    # Aggressively free the original model
    del vision_model, language_model, vision_tower, multi_modal_projector
    gc.collect()
    _empty_cache(device)

    print(f"     Moving model to {device} ...")
    hook_language_model = hook_language_model.to(device)
    gc.collect()
    _empty_cache(device)

    print(f"[4/4] Loading SAE from {sae_path} ...")
    sae = SAE.load_from_pretrained(path=sae_path, device=device)
    sae.cfg.device = device

    print(f"     hook_name = {sae.cfg.hook_name}")
    print(f"     d_in = {sae.cfg.d_in},  d_sae = {sae.cfg.d_sae}")
    return processor, hook_language_model, sae


# ---------------------------------------------------------------------------
# Input preparation  (RLAIF-V format, compatible with HookedLlava)
# ---------------------------------------------------------------------------

def prepare_input(processor, image, question, device):
    """Build model input from a single RLAIF-V sample."""
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
# Main computation
# ---------------------------------------------------------------------------

def compute_metrics(args):
    device = get_device()
    dtype = torch.float16 if args.dtype == "float16" else torch.float32
    print(f"Device: {device},  dtype: {dtype}")

    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    model_path = os.path.join(project_root, args.model_path) if not os.path.isabs(args.model_path) else args.model_path
    sae_path = os.path.join(project_root, args.sae_path) if not os.path.isabs(args.sae_path) else args.sae_path

    # ---- Load models ----
    processor, hook_language_model, sae = load_models(model_path, sae_path, device, dtype)

    d_sae = sae.cfg.d_sae          # 65536
    d_in = sae.cfg.d_in            # 4096
    hook_name = sae.cfg.hook_name  # blocks.16.hook_resid_post
    stop_at_layer = int(hook_name.split(".")[1]) + 1  # 17

    # ---- Load dataset ----
    from datasets import load_dataset, load_from_disk
    if args.dataset_path and os.path.isdir(args.dataset_path):
        print(f"\nLoading dataset from local path: {args.dataset_path} ...")
        ds = load_from_disk(args.dataset_path)
        if args.n_samples < len(ds):
            ds = ds.select(range(args.n_samples))
    else:
        print(f"\nLoading {args.n_samples} samples from openbmb/RLAIF-V-Dataset ...")
        ds = load_dataset("openbmb/RLAIF-V-Dataset", split=f"train[:{args.n_samples}]")
    print(f"  Loaded {len(ds)} samples.  {_gpu_mem_str(device)}")

    # ---- Accumulators  (float64 for numerical stability) ----
    text_act_sum = np.zeros(d_sae, dtype=np.float64)
    image_act_sum = np.zeros(d_sae, dtype=np.float64)
    text_token_total = 0
    image_token_total = 0
    sample_active = np.zeros(d_sae, dtype=np.int32)

    alignment_sum = np.zeros(d_sae, dtype=np.float64)
    alignment_count = np.zeros(d_sae, dtype=np.int32)

    n_processed = 0
    t0 = time.time()

    for idx in tqdm(range(len(ds)), desc="Processing samples"):
        sample = ds[idx]
        try:
            inputs = prepare_input(processor, sample["image"], sample["question"], device)
        except Exception as e:
            print(f"  [SKIP] sample {idx}: {e}")
            continue

        with torch.no_grad():
            # Forward through layers 0-16 → returns (residual, image_indice)
            out, cache = hook_language_model.run_with_cache(
                input=inputs,
                model_inputs=inputs,
                vision=True,
                prepend_bos=True,
                names_filter=lambda name: name == hook_name,
                stop_at_layer=stop_at_layer,
            )
            image_indice = out[1]  # (1, 1176)

            hidden_states = cache[hook_name]          # (1, seq_len, d_in)
            hidden_states = hidden_states.to(device)
            feature_acts = sae.encode(hidden_states)  # (1, seq_len, d_sae)

            del cache, out

        # --- Build text / image masks ---
        seq_len = feature_acts.shape[1]
        image_positions = set(image_indice[0].cpu().tolist())
        image_mask = torch.zeros(seq_len, dtype=torch.bool)
        for p in image_positions:
            image_mask[p] = True
        text_mask = ~image_mask

        n_text = text_mask.sum().item()
        n_image = image_mask.sum().item()

        fa = feature_acts[0].float().cpu()   # (seq_len, d_sae)
        hs = hidden_states[0].float()        # keep on device for matmul

        # --- 1. Modality ratio accumulators ---
        text_act_sum += fa[text_mask].sum(dim=0).numpy()
        image_act_sum += fa[image_mask].sum(dim=0).numpy()
        text_token_total += n_text
        image_token_total += n_image

        # --- 2. Frequency ---
        active = (fa.abs().sum(dim=0) > 0).numpy().astype(np.int32)
        sample_active += active

        # --- 3. Alignment score (only for cross-modal features in this sample) ---
        fa_device = feature_acts[0].float()  # (seq_len, d_sae) on device
        text_acts = fa_device[text_mask.to(device)]    # (n_text, d_sae)
        image_acts = fa_device[image_mask.to(device)]  # (n_image, d_sae)

        text_active = (text_acts > 0).any(dim=0)    # (d_sae,)
        image_active = (image_acts > 0).any(dim=0)  # (d_sae,)
        both_active = text_active & image_active
        active_idx = both_active.nonzero().squeeze(-1)

        if active_idx.numel() > 0:
            text_hidden = hs[text_mask.to(device)]    # (n_text, d_in)
            image_hidden = hs[image_mask.to(device)]  # (n_image, d_in)

            ta_sub = text_acts[:, active_idx]   # (n_text, n_active)
            ia_sub = image_acts[:, active_idx]  # (n_image, n_active)

            # Activation-weighted mean hidden state per feature
            text_weighted = (ta_sub.T @ text_hidden) / (ta_sub.sum(0).unsqueeze(1) + 1e-8)
            image_weighted = (ia_sub.T @ image_hidden) / (ia_sub.sum(0).unsqueeze(1) + 1e-8)

            cos_sim = F.cosine_similarity(text_weighted, image_weighted, dim=1)

            idx_np = active_idx.cpu().numpy()
            alignment_sum[idx_np] += cos_sim.cpu().numpy()
            alignment_count[idx_np] += 1

        n_processed += 1

        # --- Cleanup ---
        del fa, hs, fa_device, feature_acts, hidden_states, text_acts, image_acts
        del inputs, image_indice
        gc.collect()
        _empty_cache(device)

    elapsed = time.time() - t0
    print(f"\nProcessed {n_processed}/{len(ds)} samples in {elapsed:.1f}s "
          f"({elapsed / max(n_processed, 1):.1f}s/sample)  {_gpu_mem_str(device)}")

    # ---- Compute final metrics ----
    eps = 1e-8
    mean_text = text_act_sum / (text_token_total + eps)
    mean_image = image_act_sum / (image_token_total + eps)
    modality_ratio = (mean_image - mean_text) / (mean_image + mean_text + eps)
    frequency = sample_active / max(n_processed, 1)
    with np.errstate(divide="ignore", invalid="ignore"):
        alignment = np.where(
            alignment_count > 0,
            alignment_sum / alignment_count,
            0.0,
        )
    alignment = np.nan_to_num(alignment, nan=0.0)

    # ---- Save ----
    os.makedirs(args.output_dir, exist_ok=True)
    save_path = os.path.join(args.output_dir, "feature_metrics.npz")
    np.savez(
        save_path,
        modality_ratio=modality_ratio,
        alignment=alignment,
        frequency=frequency,
        mean_text=mean_text,
        mean_image=mean_image,
        alignment_count=alignment_count,
        sample_active=sample_active,
        n_samples=n_processed,
        text_token_total=text_token_total,
        image_token_total=image_token_total,
    )
    print(f"Saved metrics → {save_path}")

    # ---- Print summary ----
    alive = frequency > 0
    n_alive = alive.sum()
    print(f"\n{'='*60}")
    print(f"Feature summary ({n_alive}/{d_sae} alive features):")
    print(f"  Text-dominant  (ratio < -0.8): {(modality_ratio[alive] < -0.8).sum()}")
    print(f"  Image-dominant (ratio >  0.8): {(modality_ratio[alive] >  0.8).sum()}")
    print(f"  Cross-modal    (|ratio| < 0.8): {((np.abs(modality_ratio[alive]) <= 0.8)).sum()}")
    print(f"  Mean alignment (cross-modal):  {alignment[alignment_count > 0].mean():.4f}")
    print(f"  Mean frequency (alive):        {frequency[alive].mean():.4f}")
    print(f"{'='*60}")

    # ---- Visualize ----
    visualize_3d(modality_ratio, alignment, frequency, args.output_dir)
    return modality_ratio, alignment, frequency


# ---------------------------------------------------------------------------
# 3D scatter plot
# ---------------------------------------------------------------------------

def visualize_3d(modality_ratio, alignment, frequency, output_dir):
    """Create 3D scatter plots (static PNG + interactive HTML)."""

    # Filter to alive features
    alive = frequency > 0
    mr = modality_ratio[alive]
    al = alignment[alive]
    fr = frequency[alive]
    indices = np.where(alive)[0]

    print(f"\nGenerating 3D scatter ({alive.sum()} alive features) ...")

    # ---- Classify for coloring ----
    categories = []
    colors_cat = []
    for i in range(len(mr)):
        if fr[i] < 0.02:
            categories.append("rare")
            colors_cat.append("gray")
        elif mr[i] > 0.8:
            categories.append("visual")
            colors_cat.append("#e74c3c")
        elif mr[i] < -0.8:
            categories.append("text")
            colors_cat.append("#3498db")
        else:
            categories.append("cross-modal")
            colors_cat.append("#2ecc71")

    # ---- Static PNG (matplotlib) ----
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

    fig = plt.figure(figsize=(12, 9))
    ax = fig.add_subplot(111, projection="3d")

    for cat, col, label in [
        ("text",        "#3498db", "Text-dominant"),
        ("visual",      "#e74c3c", "Image-dominant"),
        ("cross-modal", "#2ecc71", "Cross-modal"),
        ("rare",        "gray",    "Rare / Redundant"),
    ]:
        mask = np.array(categories) == cat
        if mask.sum() == 0:
            continue
        ax.scatter(
            mr[mask], al[mask], fr[mask],
            c=col, label=f"{label} ({mask.sum()})",
            s=8, alpha=0.5,
        )

    ax.set_xlabel("Modality Ratio\n(← text | image →)", fontsize=10)
    ax.set_ylabel("Alignment Score", fontsize=10)
    ax.set_zlabel("Activation Frequency", fontsize=10)
    ax.set_title("SAE-V Feature Modality Analysis (3D)", fontsize=13)
    ax.legend(loc="upper left", fontsize=8)
    plt.tight_layout()
    png_path = os.path.join(output_dir, "feature_3d_scatter.png")
    plt.savefig(png_path, dpi=200)
    plt.close()
    print(f"  Saved: {png_path}")

    # ---- Interactive HTML (plotly) ----
    try:
        import plotly.graph_objects as go

        hover_text = [
            f"Feature {indices[i]}<br>"
            f"Ratio: {mr[i]:.3f}<br>"
            f"Alignment: {al[i]:.3f}<br>"
            f"Frequency: {fr[i]:.3f}<br>"
            f"Category: {categories[i]}"
            for i in range(len(mr))
        ]

        fig = go.Figure()
        for cat, col, label in [
            ("text",        "#3498db", "Text-dominant"),
            ("visual",      "#e74c3c", "Image-dominant"),
            ("cross-modal", "#2ecc71", "Cross-modal"),
            ("rare",        "gray",    "Rare / Redundant"),
        ]:
            mask = np.array(categories) == cat
            if mask.sum() == 0:
                continue
            fig.add_trace(go.Scatter3d(
                x=mr[mask], y=al[mask], z=fr[mask],
                mode="markers",
                marker=dict(size=2.5, color=col, opacity=0.6),
                name=f"{label} ({mask.sum()})",
                text=[hover_text[j] for j in np.where(mask)[0]],
                hoverinfo="text",
            ))

        fig.update_layout(
            title="SAE-V Feature Modality Analysis",
            scene=dict(
                xaxis_title="Modality Ratio (← text | image →)",
                yaxis_title="Alignment Score",
                zaxis_title="Activation Frequency",
            ),
            width=1000, height=750,
            legend=dict(x=0.02, y=0.98),
        )
        html_path = os.path.join(output_dir, "feature_3d_scatter.html")
        fig.write_html(html_path)
        print(f"  Saved: {html_path}")
    except ImportError:
        print("  plotly not installed — skipped interactive HTML.")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model_path", default="model/llava-v1.6-mistral-7b-hf")
    parser.add_argument("--sae_path", default="model/SAE-V/SAEV_LLaVA_NeXT-7b_OBELICS")
    parser.add_argument("--n_samples", type=int, default=50,
                        help="Number of RLAIF-V samples to process")
    parser.add_argument("--dataset_path", default=None,
                        help="Path to pre-downloaded dataset (from datasets.save_to_disk)")
    parser.add_argument("--output_dir", default="output/feature_metrics")
    parser.add_argument("--dtype", default="float16", choices=["float32", "float16"],
                        help="Model dtype (float16 saves memory)")
    args = parser.parse_args()
    compute_metrics(args)


if __name__ == "__main__":
    main()

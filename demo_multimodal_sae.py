"""
Demo: View SAE activations on multimodal (image + text) input using LLaVA-NeXT + SAE-V.
Adapted for Apple Silicon (MPS) device.

Usage:
    conda activate sae-v
    python demo_multimodal_sae.py \
        --image_path test_apple.jpg \
        --prompt "The fruit in the image is "

Requirements:
    - LLaVA-NeXT model at: model/llava-v1.6-mistral-7b-hf
    - SAE-V weights at: model/SAE-V/SAEV_LLaVA_NeXT-7b_OBELICS
"""

import os
import sys
import argparse
import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image
from functools import partial

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from sae_lens import SAE
from transformers import LlavaNextForConditionalGeneration, LlavaNextProcessor
from transformer_lens.HookedLlava import HookedLlava


def get_device():
    if torch.backends.mps.is_available():
        return "mps"
    elif torch.cuda.is_available():
        return "cuda:0"
    return "cpu"


def load_models(model_path, sae_path, device, dtype=torch.float32):
    print(f"[1/4] Loading processor from {model_path}...")
    processor = LlavaNextProcessor.from_pretrained(model_path)

    import gc

    print(f"[2/4] Loading LLaVA-NeXT vision model on CPU...")
    vision_model = LlavaNextForConditionalGeneration.from_pretrained(
        model_path,
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
    )

    print(f"[3/4] Creating HookedLlava on CPU first...")
    hook_language_model = HookedLlava.from_pretrained_no_processing(
        "llava-hf/llava-v1.6-mistral-7b-hf",
        hf_model=vision_model.language_model,
        device="cpu",
        fold_ln=False,
        center_writing_weights=False,
        center_unembed=False,
        tokenizer=None,
        dtype=dtype,
        vision_tower=vision_model.vision_tower,
        multi_modal_projector=vision_model.multi_modal_projector,
        n_devices=1,
    )
    del vision_model
    gc.collect()

    print(f"  Moving model to {device}...")
    hook_language_model = hook_language_model.to(device)
    gc.collect()
    if device == "mps":
        torch.mps.empty_cache()

    print(f"[4/4] Loading SAE-V from {sae_path}...")
    sae = SAE.load_from_pretrained(path=sae_path, device=device)
    sae.cfg.device = device

    return processor, hook_language_model, sae


def prepare_input(processor, image_path, prompt, device):
    image = Image.open(image_path).convert("RGB")
    image = image.resize((336, 336))

    conversation = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {"type": "image"},
            ],
        },
    ]
    prompt_text = processor.apply_chat_template(conversation, add_generation_prompt=True)

    # Use tokenizer and image_processor separately to avoid auto-expanding <image> tokens
    # HookedLlava expects 1 <image> token per image and handles expansion internally
    text_inputs = processor.tokenizer(prompt_text, return_tensors="pt")
    image_inputs = processor.image_processor(images=image, return_tensors="pt")

    inputs = {
        "input_ids": text_inputs["input_ids"],
        "attention_mask": text_inputs["attention_mask"],
        "pixel_values": image_inputs["pixel_values"],
        "image_sizes": image_inputs["image_sizes"],
    }
    inputs = {k: v.to(device) for k, v in inputs.items()}
    return inputs, image


def analyze_activations(hook_language_model, sae, inputs, device):
    print("\n--- Running model with SAE hook ---")
    with torch.no_grad():
        _, cache = hook_language_model.run_with_cache(
            input=inputs,
            model_inputs=inputs,
            vision=True,
            prepend_bos=True,
            names_filter=lambda name: name == sae.cfg.hook_name,
        )

        activations = cache[sae.cfg.hook_name]
        activations = activations.to(device)

        # Encode with SAE
        feature_acts = sae.encode(activations)
        sae_out = sae.decode(feature_acts)

        # L0: number of active features per token
        l0 = (feature_acts > 0).float().sum(-1).detach()

        # Reconstruction error
        recon_error = (activations - sae_out).pow(2).mean().item()

        del cache

    return feature_acts, sae_out, activations, l0, recon_error


def visualize_results(feature_acts, l0, image, save_dir, sae):
    os.makedirs(save_dir, exist_ok=True)

    # 1. L0 distribution
    l0_flat = l0.flatten().cpu().numpy()
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.hist(l0_flat, bins=50, color='steelblue', edgecolor='black', alpha=0.7)
    ax.set_title("L0: Number of Active SAE Features per Token")
    ax.set_xlabel("Number of Active Features")
    ax.set_ylabel("Frequency")
    ax.axvline(l0_flat.mean(), color='red', linestyle='--', label=f'Mean: {l0_flat.mean():.1f}')
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "l0_distribution.png"), dpi=150)
    plt.close()
    print(f"  Saved: {save_dir}/l0_distribution.png")

    # 2. Top activated features
    feature_acts_cpu = feature_acts.squeeze(0).cpu()
    total_activation = feature_acts_cpu.sum(dim=0)  # sum across all tokens
    top_k = 20
    top_values, top_indices = torch.topk(total_activation, top_k)

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.barh(range(top_k), top_values.numpy(), color='coral')
    ax.set_yticks(range(top_k))
    ax.set_yticklabels([f"Feature {idx.item()}" for idx in top_indices])
    ax.set_xlabel("Total Activation (summed over all tokens)")
    ax.set_title(f"Top {top_k} Most Active SAE Features")
    ax.invert_yaxis()
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "top_features.png"), dpi=150)
    plt.close()
    print(f"  Saved: {save_dir}/top_features.png")

    # 3. Per-token activation heatmap (first 100 tokens x top 30 features)
    top_30_indices = top_indices[:30]
    acts_subset = feature_acts_cpu[:, top_30_indices].numpy()
    n_tokens = min(acts_subset.shape[0], 100)
    acts_subset = acts_subset[:n_tokens, :]

    fig, ax = plt.subplots(figsize=(14, 8))
    im = ax.imshow(acts_subset.T, aspect='auto', cmap='hot', interpolation='nearest')
    ax.set_xlabel("Token Position")
    ax.set_ylabel("SAE Feature")
    ax.set_yticks(range(len(top_30_indices)))
    ax.set_yticklabels([f"F{idx.item()}" for idx in top_30_indices], fontsize=7)
    ax.set_title("SAE Feature Activations per Token (Top 30 Features)")
    plt.colorbar(im, ax=ax, label="Activation Value")
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "activation_heatmap.png"), dpi=150)
    plt.close()
    print(f"  Saved: {save_dir}/activation_heatmap.png")

    # 4. Image patch activation overlay
    try:
        visualize_image_patch_activations(feature_acts, l0, image, save_dir)
    except Exception as e:
        print(f"  Skipped patch visualization: {e}")


def visualize_image_patch_activations(feature_acts, l0, image, save_dir):
    """Overlay L0 activation density on image patches (24x24 grid -> 336x336)."""
    l0_flat = l0.squeeze(0).cpu()

    # LLaVA-NeXT 336x336 -> 24x24=576 patches (doubled to 1152 + 24 newlines = 1176 image tokens)
    if l0_flat.shape[0] >= 576:
        start = max(0, (l0_flat.shape[0] - 576) // 2)
        patch_l0 = l0_flat[start:start + 576]
        if patch_l0.shape[0] == 576:
            patch_grid = patch_l0.view(24, 24).numpy()

            # Upsample to 336x336
            patch_upsampled = np.repeat(np.repeat(patch_grid, 14, axis=0), 14, axis=1)

            fig, axes = plt.subplots(1, 3, figsize=(15, 5))

            axes[0].imshow(image)
            axes[0].set_title("Original Image")
            axes[0].axis('off')

            im = axes[1].imshow(patch_upsampled, cmap='hot')
            axes[1].set_title("SAE L0 Activation Map")
            axes[1].axis('off')
            plt.colorbar(im, ax=axes[1], fraction=0.046)

            axes[2].imshow(image, alpha=0.6)
            axes[2].imshow(patch_upsampled, cmap='hot', alpha=0.4)
            axes[2].set_title("Overlay")
            axes[2].axis('off')

            plt.suptitle("Image Patch SAE Activations", fontsize=14)
            plt.tight_layout()
            plt.savefig(os.path.join(save_dir, "image_patch_activations.png"), dpi=150)
            plt.close()
            print(f"  Saved: {save_dir}/image_patch_activations.png")


def reconstr_hook(activation, hook, sae_out):
    return sae_out


def zero_abl_hook(activation, hook):
    return torch.zeros_like(activation)


def main():
    parser = argparse.ArgumentParser(description="Demo: Multimodal SAE Activation Viewer")
    parser.add_argument("--model_path", type=str,
                        default="model/llava-v1.6-mistral-7b-hf",
                        help="Path to LLaVA-NeXT model")
    parser.add_argument("--sae_path", type=str,
                        default="model/SAE-V/SAEV_LLaVA_NeXT-7b_OBELICS",
                        help="Path to SAE-V weights")
    parser.add_argument("--image_path", type=str,
                        default="test_apple.jpg",
                        help="Path to input image")
    parser.add_argument("--prompt", type=str,
                        default="The fruit in the image is ",
                        help="Text prompt")
    parser.add_argument("--save_dir", type=str,
                        default="output/activation_vis",
                        help="Directory to save visualizations")
    parser.add_argument("--dtype", type=str, default="float32",
                        choices=["float32", "float16"],
                        help="Model dtype (float16 saves memory)")
    args = parser.parse_args()

    # Resolve relative paths
    project_root = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(project_root, args.model_path) if not os.path.isabs(args.model_path) else args.model_path
    sae_path = os.path.join(project_root, args.sae_path) if not os.path.isabs(args.sae_path) else args.sae_path
    image_path = os.path.join(project_root, args.image_path) if not os.path.isabs(args.image_path) else args.image_path
    save_dir = os.path.join(project_root, args.save_dir) if not os.path.isabs(args.save_dir) else args.save_dir

    device = get_device()
    dtype = torch.float16 if args.dtype == "float16" else torch.float32
    print(f"Device: {device}, Dtype: {dtype}")

    # Load models
    processor, hook_language_model, sae = load_models(model_path, sae_path, device, dtype)

    print(f"\nSAE Config:")
    print(f"  Hook: {sae.cfg.hook_name}")
    print(f"  d_in: {sae.cfg.d_in}, d_sae: {sae.cfg.d_sae}")

    # Prepare input
    print(f"\nImage: {image_path}")
    print(f"Prompt: {args.prompt}")
    inputs, image = prepare_input(processor, image_path, args.prompt, device)
    print(f"Input tokens: {inputs['input_ids'].shape}")

    # Run SAE analysis
    feature_acts, sae_out, activations, l0, recon_error = analyze_activations(
        hook_language_model, sae, inputs, device
    )

    print(f"\n--- Results ---")
    print(f"  Feature activations shape: {feature_acts.shape}")
    print(f"  Average L0 (active features/token): {l0.mean().item():.1f}")
    print(f"  Reconstruction MSE: {recon_error:.6f}")

    # Cosine similarity between original and reconstructed activations
    with torch.no_grad():
        cos_sim = torch.nn.functional.cosine_similarity(
            activations.flatten(), sae_out.flatten(), dim=0
        ).item()
    print(f"  Cosine similarity (orig vs reconstructed): {cos_sim:.6f}")

    # Visualize
    print(f"\nSaving visualizations to {save_dir}/")
    visualize_results(feature_acts, l0, image, save_dir, sae)
    print("\nDone!")


if __name__ == "__main__":
    main()

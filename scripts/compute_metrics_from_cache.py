#!/usr/bin/env python
"""
Phase 2: Compute feature metrics from cached SAE activations.

Uses streaming accumulation — processes one chunk at a time to avoid
loading all data into memory (~73GB for 83k samples). Memory usage: ~100MB.

Metrics computed:
  1. Normalized Modality Ratio — mean activation per modality (token-count normalized)
  2. Cross-Modal Alignment Score — cosine similarity of activation-weighted hidden states
  3. Activation Frequency — fraction of samples where each feature fires

Usage:
    python scripts/compute_metrics_from_cache.py \\
        --cache_dir output/activation_cache \\
        --output_dir output/feature_metrics
"""

import os
import sys
import json
import time
import argparse
import glob

import numpy as np
from tqdm import tqdm


def compute_metrics_streaming(cache_dir, output_dir):
    """Stream chunks one at a time, accumulating into (d_sae,) arrays."""

    meta_path = os.path.join(cache_dir, "meta.json")
    if not os.path.exists(meta_path):
        raise FileNotFoundError(f"No meta.json found in {cache_dir}")

    with open(meta_path) as f:
        meta = json.load(f)

    d_sae = meta["d_sae"]
    n_processed = meta["n_processed"]

    print(f"Cache metadata:")
    print(f"  Samples: {n_processed}")
    print(f"  d_sae: {d_sae}")
    print(f"  hook: {meta['hook_name']}")
    print(f"  Chunks: {meta['n_chunks']}")
    if "total_time_s" in meta:
        print(f"  Phase 1 took: {meta['total_time_s']:.1f}s "
              f"({meta.get('avg_time_per_sample_s', 0):.2f}s/sample)")
    print()

    chunk_files = sorted(glob.glob(os.path.join(cache_dir, "chunk_*.npz")))
    print(f"Streaming {len(chunk_files)} chunks (memory-efficient) ...")

    # Accumulators — only (d_sae,) shaped, ~3MB total
    total_text_act = np.zeros(d_sae, dtype=np.float64)
    total_image_act = np.zeros(d_sae, dtype=np.float64)
    total_text_tokens = 0
    total_image_tokens = 0
    sample_active = np.zeros(d_sae, dtype=np.int64)
    alignment_sum = np.zeros(d_sae, dtype=np.float64)
    alignment_count = np.zeros(d_sae, dtype=np.int64)
    n_samples = 0

    t0 = time.time()
    for cf in tqdm(chunk_files, desc="Processing chunks", unit="chunk"):
        data = np.load(cf)

        text_act = data["text_act_sum"].astype(np.float64)   # (chunk, d_sae)
        image_act = data["image_act_sum"].astype(np.float64)  # (chunk, d_sae)

        total_text_act += text_act.sum(axis=0)
        total_image_act += image_act.sum(axis=0)
        total_text_tokens += data["n_text"].sum()
        total_image_tokens += data["n_image"].sum()

        sample_active += data["active_mask"].sum(axis=0)

        align = data["alignment"].astype(np.float64)
        align_mask = data["alignment_mask"]
        alignment_sum += (align * align_mask).sum(axis=0)
        alignment_count += align_mask.sum(axis=0)

        n_samples += text_act.shape[0]

    elapsed = time.time() - t0
    print(f"\nStreaming done: {n_samples} samples in {elapsed:.1f}s "
          f"({elapsed/len(chunk_files)*1000:.1f}ms/chunk)")

    eps = 1e-8

    # 1. Normalized Modality Ratio
    mean_text = total_text_act / (total_text_tokens + eps)
    mean_image = total_image_act / (total_image_tokens + eps)
    modality_ratio = (mean_image - mean_text) / (mean_image + mean_text + eps)

    print(f"  Token totals: {int(total_text_tokens)} text, {int(total_image_tokens)} image")

    # 2. Activation Frequency
    frequency = sample_active / n_samples

    # 3. Cross-Modal Alignment Score
    with np.errstate(divide="ignore", invalid="ignore"):
        alignment_score = np.where(
            alignment_count > 0,
            alignment_sum / alignment_count,
            0.0,
        )
    alignment_score = np.nan_to_num(alignment_score, nan=0.0)

    # Save
    os.makedirs(output_dir, exist_ok=True)
    save_path = os.path.join(output_dir, "feature_metrics.npz")
    np.savez(
        save_path,
        modality_ratio=modality_ratio.astype(np.float32),
        alignment=alignment_score.astype(np.float32),
        frequency=frequency.astype(np.float32),
        mean_text=mean_text.astype(np.float32),
        mean_image=mean_image.astype(np.float32),
        alignment_count=alignment_count.astype(np.int32),
        sample_active=sample_active.astype(np.int32),
        n_samples=n_samples,
        text_token_total=int(total_text_tokens),
        image_token_total=int(total_image_tokens),
    )
    print(f"  Saved metrics -> {save_path}")

    # Print summary
    alive = frequency > 0
    n_alive = alive.sum()
    print(f"\n{'='*60}")
    print(f"Feature summary ({n_alive}/{d_sae} alive features):")
    print(f"  Text-dominant  (ratio < -0.8): {(modality_ratio[alive] < -0.8).sum()}")
    print(f"  Image-dominant (ratio >  0.8): {(modality_ratio[alive] >  0.8).sum()}")
    print(f"  Cross-modal    (|ratio| <= 0.8): {(np.abs(modality_ratio[alive]) <= 0.8).sum()}")
    if alignment_count[alignment_count > 0].size > 0:
        print(f"  Mean alignment (cross-modal):  {alignment_score[alignment_count > 0].mean():.4f}")
    print(f"  Mean frequency (alive):        {frequency[alive].mean():.4f}")
    print(f"{'='*60}")

    # Visualize
    visualize_3d(modality_ratio, alignment_score, frequency, output_dir)

    return modality_ratio, alignment_score, frequency


def visualize_3d(modality_ratio, alignment, frequency, output_dir):
    """Create 3D scatter plots (static PNG + interactive HTML)."""
    alive = frequency > 0
    mr = modality_ratio[alive]
    al = alignment[alive]
    fr = frequency[alive]
    indices = np.where(alive)[0]

    print(f"\nGenerating 3D scatter ({alive.sum()} alive features) ...")

    categories = []
    for i in range(len(mr)):
        if fr[i] < 0.02:
            categories.append("rare")
        elif mr[i] > 0.8:
            categories.append("visual")
        elif mr[i] < -0.8:
            categories.append("text")
        else:
            categories.append("cross-modal")
    categories = np.array(categories)

    # Static PNG
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(12, 9))
    ax = fig.add_subplot(111, projection="3d")

    for cat, col, label in [
        ("text",        "#3498db", "Text-dominant"),
        ("visual",      "#e74c3c", "Image-dominant"),
        ("cross-modal", "#2ecc71", "Cross-modal"),
        ("rare",        "gray",    "Rare / Redundant"),
    ]:
        mask = categories == cat
        if mask.sum() == 0:
            continue
        ax.scatter(mr[mask], al[mask], fr[mask],
                   c=col, label=f"{label} ({mask.sum()})",
                   s=8, alpha=0.5)

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

    # Interactive HTML
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
            mask = categories == cat
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


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--cache_dir", required=True,
                        help="Directory containing activation cache chunks")
    parser.add_argument("--output_dir", default="output/feature_metrics",
                        help="Directory for output metrics and visualizations")
    args = parser.parse_args()

    compute_metrics_streaming(args.cache_dir, args.output_dir)


if __name__ == "__main__":
    main()

#!/usr/bin/env python
"""
Pre-compute per-feature top-K sample index from cached activations.

Streams all chunk files and tracks the top-K samples (by total activation)
for each of the 65536 SAE features. Output is a compact NPZ file (~16MB)
used by the interactive feature explorer for instant lookups.

Usage:
    python -m app.precompute_topk \
        --cache_dir /root/autodl-tmp/activation_cache_full \
        --output /root/autodl-tmp/feature_metrics_full/topk_index.npz \
        --k 32
"""

import os
import json
import argparse
import glob
import time

import numpy as np
from tqdm import tqdm


def build_topk_index(cache_dir, k=32):
    meta_path = os.path.join(cache_dir, "meta.json")
    with open(meta_path) as f:
        meta = json.load(f)

    d_sae = meta["d_sae"]
    chunk_size = meta.get("chunk_size", 100)

    chunk_files = sorted(glob.glob(os.path.join(cache_dir, "chunk_*.npz")))
    print(f"Building top-{k} index from {len(chunk_files)} chunks, d_sae={d_sae}")

    # Accumulators: (d_sae, k) — track top-K values and their sample indices
    topk_values = np.full((d_sae, k), -np.inf, dtype=np.float32)
    topk_indices = np.full((d_sae, k), -1, dtype=np.int32)
    topk_text_act = np.zeros((d_sae, k), dtype=np.float32)
    topk_image_act = np.zeros((d_sae, k), dtype=np.float32)

    t0 = time.time()
    for chunk_idx, cf in enumerate(tqdm(chunk_files, desc="Scanning chunks")):
        data = np.load(cf)
        text_act = data["text_act_sum"].astype(np.float32)    # (cs, d_sae)
        image_act = data["image_act_sum"].astype(np.float32)  # (cs, d_sae)
        total_act = text_act + image_act                       # (cs, d_sae)
        cs = total_act.shape[0]

        for local_i in range(cs):
            global_idx = chunk_idx * chunk_size + local_i
            acts = total_act[local_i]  # (d_sae,)

            # Find features where this sample beats current minimum in top-K
            min_pos = topk_values.argmin(axis=1)                   # (d_sae,)
            min_vals = topk_values[np.arange(d_sae), min_pos]      # (d_sae,)
            better = acts > min_vals

            if better.any():
                feat_ids = np.where(better)[0]
                positions = min_pos[feat_ids]
                topk_values[feat_ids, positions] = acts[feat_ids]
                topk_indices[feat_ids, positions] = global_idx
                topk_text_act[feat_ids, positions] = text_act[local_i, feat_ids]
                topk_image_act[feat_ids, positions] = image_act[local_i, feat_ids]

    elapsed = time.time() - t0
    print(f"Done in {elapsed:.1f}s")

    # Sort each feature's top-K by descending activation
    for f in range(d_sae):
        order = np.argsort(-topk_values[f])
        topk_values[f] = topk_values[f, order]
        topk_indices[f] = topk_indices[f, order]
        topk_text_act[f] = topk_text_act[f, order]
        topk_image_act[f] = topk_image_act[f, order]

    return topk_indices, topk_values, topk_text_act, topk_image_act, meta


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--cache_dir", required=True)
    parser.add_argument("--output", default="output/topk_index.npz")
    parser.add_argument("--k", type=int, default=32)
    args = parser.parse_args()

    indices, values, text_act, image_act, meta = build_topk_index(args.cache_dir, args.k)

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    np.savez_compressed(
        args.output,
        topk_indices=indices,
        topk_values=values,
        topk_text_act=text_act,
        topk_image_act=image_act,
        k=args.k,
        n_samples=meta["n_processed"],
        d_sae=meta["d_sae"],
    )
    fsize = os.path.getsize(args.output) / 1e6
    print(f"Saved: {args.output} ({fsize:.1f} MB)")


if __name__ == "__main__":
    main()

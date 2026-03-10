#!/usr/bin/env python
"""
Download and convert OBELICS samples to standard format for SAE-V pipeline.

OBELICS is a web-crawled dataset with interleaved images (URLs) and texts.
This script extracts valid (image, text) pairs, downloads images, and saves
as a HuggingFace Dataset with `image` (PIL) and `question` (str) columns.

Supports two modes:
  1. Streaming (default): load_dataset(..., streaming=True) — works on machines
     with direct HuggingFace access.
  2. Parquet: --mode parquet — downloads parquet shards via huggingface_hub
     (respects HF_ENDPOINT mirror). Useful on servers behind firewalls.

Usage:
    # Streaming mode (local Mac, direct HF access):
    python -m scripts.download_obelics --n_samples 10000 --output dataset/OBELICS-10k

    # Parquet mode (server with HF mirror):
    HF_ENDPOINT=https://hf-mirror.com python -m scripts.download_obelics \
        --mode parquet --n_samples 10000 --output dataset/OBELICS-10k
"""

import os
import argparse
import io
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests
from PIL import Image
from tqdm import tqdm


def download_image(url, timeout=10):
    """Download image from URL and return PIL Image, or None on failure."""
    try:
        resp = requests.get(url, timeout=timeout, stream=True,
                            headers={"User-Agent": "Mozilla/5.0"})
        resp.raise_for_status()
        img = Image.open(io.BytesIO(resp.content)).convert("RGB")
        if img.size[0] < 10 or img.size[1] < 10:
            return None
        return img
    except Exception:
        return None


def extract_pair(row):
    """Extract first valid (image_url, text) pair from an OBELICS row."""
    images = row.get("images") or []
    texts = row.get("texts") or []

    img_url = None
    for item in images:
        if item is not None and isinstance(item, str) and item.startswith("http"):
            img_url = item
            break

    if img_url is None:
        return None, None

    text = None
    for item in texts:
        if item is not None and isinstance(item, str) and len(item.strip()) > 20:
            text = item.strip()[:500]
            break

    if text is None:
        return None, None

    return img_url, text


def extract_candidates_streaming(target, max_rows):
    """Extract candidates using HF datasets streaming."""
    from datasets import load_dataset

    print("Streaming OBELICS dataset...")
    ds_stream = load_dataset("HuggingFaceM4/OBELICS", split="train", streaming=True)

    candidates = []
    n_scanned = 0
    for row in tqdm(ds_stream, desc="Scanning rows", total=target):
        img_url, text = extract_pair(row)
        if img_url and text:
            candidates.append((img_url, text))
            if len(candidates) >= target:
                break
        n_scanned += 1
        if n_scanned >= max_rows:
            break

    return candidates


def extract_candidates_parquet(target):
    """Extract candidates by downloading parquet shards via huggingface_hub."""
    import pyarrow.parquet as pq
    from huggingface_hub import hf_hub_download

    endpoint = os.environ.get("HF_ENDPOINT", "https://huggingface.co")
    print(f"Parquet mode (HF_ENDPOINT={endpoint})")

    # List parquet shards via API
    url = f"{endpoint}/api/datasets/HuggingFaceM4/OBELICS/tree/main/data"
    resp = requests.get(url, timeout=30)
    resp.raise_for_status()
    parquet_files = sorted(f["path"] for f in resp.json() if f["path"].endswith(".parquet"))
    print(f"  Found {len(parquet_files)} shards")

    candidates = []
    for shard_idx, shard_path in enumerate(parquet_files):
        if len(candidates) >= target:
            break
        print(f"\nDownloading shard {shard_idx}: {shard_path}...")
        local_path = hf_hub_download(
            repo_id="HuggingFaceM4/OBELICS",
            filename=shard_path,
            repo_type="dataset",
        )
        table = pq.read_table(local_path, columns=["images", "texts"])
        images_col = table.column("images").to_pylist()
        texts_col = table.column("texts").to_pylist()
        del table

        n_before = len(candidates)
        for images, texts in zip(images_col, texts_col):
            pair = extract_pair({"images": images, "texts": texts})
            if pair[0] and pair[1]:
                candidates.append(pair)
                if len(candidates) >= target:
                    break
        print(f"  +{len(candidates) - n_before} candidates (total: {len(candidates)}/{target})")

    return candidates


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--n_samples", type=int, default=10000,
                        help="Number of valid samples to collect")
    parser.add_argument("--output", default="dataset/OBELICS-10k",
                        help="Output directory for the dataset")
    parser.add_argument("--mode", choices=["streaming", "parquet"], default="streaming",
                        help="Download mode: streaming (direct HF) or parquet (via mirror)")
    parser.add_argument("--max_rows", type=int, default=200000,
                        help="Max rows to scan (streaming mode only)")
    parser.add_argument("--workers", type=int, default=16,
                        help="Number of parallel image download workers")
    args = parser.parse_args()

    from datasets import Dataset

    print(f"Collecting {args.n_samples} valid OBELICS samples (mode={args.mode})...")
    target = args.n_samples * 2  # 2x to account for image download failures

    # Phase 1: Extract candidate (url, text) pairs
    if args.mode == "streaming":
        candidates = extract_candidates_streaming(target, args.max_rows)
    else:
        candidates = extract_candidates_parquet(target)

    print(f"\nFound {len(candidates)} candidate pairs total.")
    if len(candidates) < args.n_samples:
        print(f"WARNING: Only {len(candidates)} candidates, may not reach {args.n_samples}")

    # Phase 2: Download images in parallel
    collected_images = []
    collected_texts = []

    print(f"\nDownloading images with {args.workers} workers...")
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {}
        submitted = 0

        for url, text in candidates:
            if len(collected_images) + len(futures) >= args.n_samples + 100:
                break
            fut = executor.submit(download_image, url)
            futures[fut] = text
            submitted += 1

        pbar = tqdm(total=args.n_samples, desc="Downloading images")
        for fut in as_completed(futures):
            text = futures[fut]
            img = fut.result()
            if img is not None:
                collected_images.append(img)
                collected_texts.append(text)
                pbar.update(1)
                if len(collected_images) >= args.n_samples:
                    break
            while submitted < len(candidates) and \
                  len(collected_images) + (len(futures) - pbar.n) < args.n_samples + 50:
                url, text = candidates[submitted]
                f = executor.submit(download_image, url)
                futures[f] = text
                submitted += 1
        pbar.close()

    n_valid = min(len(collected_images), args.n_samples)
    collected_images = collected_images[:n_valid]
    collected_texts = collected_texts[:n_valid]

    print(f"\nCollected {n_valid} valid samples. Saving to {args.output}...")

    ds = Dataset.from_dict({
        "image": collected_images,
        "question": collected_texts,
    })

    os.makedirs(args.output, exist_ok=True)
    ds.save_to_disk(args.output)
    print(f"Done! Saved {len(ds)} samples to {args.output}")


if __name__ == "__main__":
    main()

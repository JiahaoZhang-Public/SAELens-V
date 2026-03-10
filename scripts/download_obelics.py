#!/usr/bin/env python
"""
Download and convert OBELICS samples to standard format for SAE-V pipeline.

OBELICS is a web-crawled dataset with interleaved images (URLs) and texts.
This script extracts valid (image, text) pairs, downloads images, and saves
as a HuggingFace Dataset with `image` (PIL) and `question` (str) columns.

Usage:
    python -m scripts.download_obelics --n_samples 10000 --output dataset/OBELICS-10k
"""

import os
import argparse
import io
import time
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
        # Sanity check: skip tiny or corrupt images
        if img.size[0] < 10 or img.size[1] < 10:
            return None
        return img
    except Exception:
        return None


def extract_pair(row):
    """Extract first valid (image_url, text) pair from an OBELICS row."""
    images = row.get("images", [])
    texts = row.get("texts", [])

    # Find first non-null image URL
    img_url = None
    for item in images:
        if item is not None and isinstance(item, str) and item.startswith("http"):
            img_url = item
            break

    if img_url is None:
        return None, None

    # Find first non-null, non-empty text
    text = None
    for item in texts:
        if item is not None and isinstance(item, str) and len(item.strip()) > 20:
            text = item.strip()[:500]  # Truncate to 500 chars
            break

    if text is None:
        return None, None

    return img_url, text


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--n_samples", type=int, default=10000,
                        help="Number of valid samples to collect")
    parser.add_argument("--output", default="dataset/OBELICS-10k",
                        help="Output directory for the dataset")
    parser.add_argument("--max_rows", type=int, default=200000,
                        help="Max rows to scan before giving up")
    parser.add_argument("--workers", type=int, default=8,
                        help="Number of parallel download workers")
    args = parser.parse_args()

    from datasets import load_dataset, Dataset

    print(f"Streaming OBELICS dataset, collecting {args.n_samples} valid samples...")
    print(f"  max_rows={args.max_rows}, workers={args.workers}")

    ds_stream = load_dataset("HuggingFaceM4/OBELICS", split="train", streaming=True)

    # Phase 1: Extract candidate (url, text) pairs
    candidates = []
    n_scanned = 0
    target = args.n_samples * 2  # Collect 2x candidates to account for download failures

    for row in tqdm(ds_stream, desc="Scanning rows", total=target):
        img_url, text = extract_pair(row)
        if img_url and text:
            candidates.append((img_url, text))
            if len(candidates) >= target:
                break
        n_scanned += 1
        if n_scanned >= args.max_rows:
            break

    print(f"Found {len(candidates)} candidate pairs from {n_scanned} rows scanned.")

    if len(candidates) < args.n_samples:
        print(f"WARNING: Only {len(candidates)} candidates found, may not reach {args.n_samples}")

    # Phase 2: Download images in parallel
    collected_images = []
    collected_texts = []

    print(f"Downloading images with {args.workers} workers...")
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {}
        submitted = 0

        # Submit initial batch
        for i, (url, text) in enumerate(candidates):
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
            # Submit more if needed
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

# SAELens-V

This repository is dedicated to training and analyzing sparse autoencoder (SAE) and sparse autoencoder with vision (SAE-V). Building on [SAELens](https://github.com/jbloomAus/SAELens), we developed SAE-V to facilitate training multi-modal models, such as LLaVA-NeXT and Chameleon. Additionally, we created a series of scripts that use SAE-V to support mechanistic interpretability analysis in multi-modal models.

This fork extends the original [SAELens-V](https://github.com/saev-2025/SAELens-V) with a **three-phase feature analysis pipeline**, an **interactive Feature Explorer dashboard**, and **multi-dataset support** for classifying SAE features by modality behavior (text vs. image).

## What This Fork Adds

- **[Analysis Pipeline](#analysis-pipeline)** -- Three-phase pipeline (cache activations -> compute metrics -> build top-K index) for classifying 65,536 SAE features by modality behavior
- **[Feature Explorer](#feature-explorer)** -- Interactive Dash-based dashboard with 3D scatter plot, per-feature detail panel, top-K sample cards, Deep Dive mode, and optional Claude API concept descriptions
- **[Multi-Dataset Support](#multi-dataset-support)** -- Dataset registry with dropdown switcher (RLAIF-V Full 83K, OBELICS 10K) and a download utility for OBELICS
- **[Analysis Notebook](#analysis-notebook)** -- Jupyter notebook for standalone 3D scatter visualization and single-feature deep dives

## Installation

Clone the source code from GitHub:
```bash
git clone https://github.com/saev-2025/SAELens-V.git
git clone https://github.com/saev-2025/TransformerLens-V.git
```

Create environment:
```bash
pip install TransformerLens-V
pip install -r SAELens-V/requirements.txt
```

Additional dependencies for the Feature Explorer:
```bash
pip install dash plotly

# Optional: for Claude API concept descriptions
pip install anthropic
```

> **Note:** Model weights and SAE checkpoints are not included in this repository (gitignored). You must obtain them separately -- see the [Training](#training) section.

## Project Structure

```
SAELens-V/
  app/
    feature_explorer.py          # Interactive Dash dashboard
    precompute_topk.py           # Phase 3: build top-K sample index
    config.py                    # Dataset registry + SAE configuration
  scripts/
    cache_activations.py         # Phase 1: cache per-sample SAE activations
    compute_metrics_from_cache.py  # Phase 2: streaming metric computation
    download_obelics.py          # Download & convert OBELICS dataset
    compute_feature_metrics.py   # Standalone metric computation (non-streaming)
    llava_preprocess.py          # (original) Dataset preprocessing for training
    Llava_sae.py                 # (original) SAE-V training script
  notebooks/
    feature_analysis.ipynb       # 3D scatter + single-feature deep dive
  sae_lens/                      # Core SAELens-V library
  model/                         # Model weights (gitignored)
  dataset/                       # Datasets (gitignored)
  output/                        # Pipeline outputs (gitignored)
```

## Analysis Pipeline

The pipeline processes a dataset through three phases to produce the data needed by the Feature Explorer:

```
Phase 1                        Phase 2                           Phase 3
cache_activations.py    --->   compute_metrics_from_cache.py  --->   precompute_topk.py
  (GPU, ~2s/sample)              (CPU, ~100MB RAM)                     (CPU, ~16MB output)
          |                              |                                     |
  activation_cache/              feature_metrics/                     feature_metrics/
    chunk_0000.npz                 feature_metrics.npz                  topk_index.npz
    chunk_0001.npz
    meta.json
```

Phase 2 computes three per-feature metrics across all 65,536 SAE features:

| Metric | Formula | Range | Meaning |
|--------|---------|-------|---------|
| **Modality Ratio** | `(mean_img - mean_txt) / (mean_img + mean_txt)` | -1 to +1 | -1 = text-only, +1 = image-only |
| **Alignment Score** | cosine similarity of activation-weighted hidden states | 0 to 1 | Cross-modal coherence |
| **Activation Frequency** | fraction of samples where feature fires | 0 to 1 | How common the feature is |

### Phase 1: Cache Activations

Runs the LLaVA-NeXT model + SAE on each dataset sample and caches per-token activation statistics.

```bash
# Quick test (2 samples)
python scripts/cache_activations.py --n_samples 2

# Full dataset
python scripts/cache_activations.py \
    --dataset_path dataset/RLAIF-V-Dataset-full \
    --output_dir output/activation_cache \
    --dtype float16

# Resume from checkpoint
python scripts/cache_activations.py \
    --output_dir output/activation_cache --resume
```

| Argument | Default | Description |
|----------|---------|-------------|
| `--model_path` | `model/llava-v1.6-mistral-7b-hf` | LLaVA-NeXT model path |
| `--sae_path` | `model/SAE-V/SAEV_LLaVA_NeXT-7b_OBELICS` | SAE-V checkpoint path |
| `--dataset_path` | *(downloads from HF)* | Local dataset path |
| `--output_dir` | `output/activation_cache` | Cache output directory |
| `--n_samples` | all | Number of samples to process |
| `--dtype` | `float16` | `float16` or `float32` |
| `--resume` | off | Resume from last checkpoint |

**Storage:** ~417 KB per sample. Full 83K dataset: ~34 GB.

### Phase 2: Compute Feature Metrics

Streams cached chunks to compute per-feature metrics without loading everything into memory (~100 MB RAM).

```bash
python scripts/compute_metrics_from_cache.py \
    --cache_dir output/activation_cache \
    --output_dir output/feature_metrics
```

| Argument | Default | Description |
|----------|---------|-------------|
| `--cache_dir` | *(required)* | Path to activation cache from Phase 1 |
| `--output_dir` | `output/feature_metrics` | Where to save `feature_metrics.npz` |

**Output:** `feature_metrics.npz` (~2 MB) containing arrays for modality_ratio, alignment, frequency, and per-feature activation statistics.

### Phase 3: Pre-compute Top-K Index

Scans all cached chunks to find the top-K highest-activating samples per feature, enabling instant lookups in the Feature Explorer.

```bash
python -m app.precompute_topk \
    --cache_dir output/activation_cache \
    --output output/feature_metrics/topk_index.npz \
    --k 32
```

| Argument | Default | Description |
|----------|---------|-------------|
| `--cache_dir` | *(required)* | Path to activation cache |
| `--output` | `output/topk_index.npz` | Output path for top-K index |
| `--k` | `32` | Number of top samples per feature |

**Output:** `topk_index.npz` (~16 MB).

## Feature Explorer

An interactive Dash-based dashboard for exploring SAE-V features.

### Quick Start

```bash
python -m app.feature_explorer --port 8050
# Open http://localhost:8050
```

### Features

- **3D Scatter Plot** -- All alive features plotted by modality ratio, alignment score, and activation frequency. Color-coded by category. Click any point to inspect.
- **Feature Detail Panel** -- Modality ratio, alignment, frequency, active sample count, mean text/image activations.
- **Top-K Sample Cards** -- Top 10 highest-activating samples with thumbnail images, question text, and text vs. image activation proportion bars.
- **Deep Dive** -- Click "Deep Dive" on any sample card to run live inference showing per-token activation bars and image patch heatmap overlay. Requires GPU + TransformerLens-V.
- **Claude API Integration** -- Provide an Anthropic API key to auto-generate concept descriptions for features based on their top activating samples.
- **Dataset Switcher** -- Dropdown to switch between registered datasets.

### Feature Categories

| Category | Condition | Color |
|----------|-----------|-------|
| Text-dominant | modality ratio < -0.8 | Blue `#3498db` |
| Image-dominant | modality ratio > 0.8 | Red `#e74c3c` |
| Cross-modal | \|ratio\| <= 0.8 and freq >= 0.02 | Green `#2ecc71` |
| Rare | frequency < 0.02 | Gray `#95a5a6` |

### Environment Detection

`app/config.py` auto-detects local vs. server environments by checking for `/root/autodl-tmp`. Paths are resolved accordingly. If running on a different server setup, edit the paths in `config.py`.

## Training

`SAELens-V` supports a complete pipeline for training SAE-V based on multiple large language models and multimodal large language models. Here is an example of training SAE-V based on LLaVA-NeXT-Mistral-7b model with OBELICS dataset:

0. Follow the instructions in section [Installation](#installation) to setup the training environment properly.
1. Dataset preprocess
```bash
python ./scripts/llava_preprocess.py \
    --dataset_path <your-OBELICS-dataset-path> \
    --tokenizer_name_or_path "llava-hf/llava-v1.6-mistral-7b-hf" \
    --save_path "./data/processed_dataset" \
```
2. SAE-V Training
```bash
python ./scripts/Llava_sae.py \
    --model_class_name "HookedLlava" \
    --language_model_name "mistralai/Mistral-7B-Instruct-v0.2" \
    --local_model_path <your-local-LLaVA-NeXT-Mistral-7b-model-path> \
    --hook_name "blocks.16.hook_resid_post" \
    --hook_layer 16 \
    --dataset_path "./data/processed_dataset" \
    --save_path "./model/SAEV_LLaVA_NeXT-7b_OBELICS" \
```

**NOTE:** You may need to update some of the parameters in the script according to your machine setup, such as the number of GPUs for training, the training batch size, etc.

### Use a New Dataset

You can use a new multimodal dataset just by change `image_column_name` and `column_name` parameter in `./scripts/llava_preprocess.py`

## Multi-Dataset Support

The Feature Explorer supports multiple datasets via a registry in `app/config.py`. Currently registered:

| Key | Label | Samples | Alive Features |
|-----|-------|---------|----------------|
| `rlaif-v-full` | RLAIF-V Full (83K) | 83,132 | 50,691 |
| `obelics-10k` | OBELICS 10K | 10,000 | 44,281 |

### Downloading OBELICS

```bash
# Streaming mode (direct HuggingFace access)
python -m scripts.download_obelics --n_samples 10000 --output dataset/OBELICS-10k

# Parquet mode (for servers with HF mirror, e.g. AutoDL)
HF_ENDPOINT=https://hf-mirror.com python -m scripts.download_obelics \
    --mode parquet --n_samples 10000 --output dataset/OBELICS-10k
```

| Argument | Default | Description |
|----------|---------|-------------|
| `--n_samples` | `10000` | Target number of valid samples |
| `--output` | `dataset/OBELICS-10k` | Output dataset path |
| `--mode` | `streaming` | `streaming` or `parquet` |
| `--max_rows` | `200000` | Max source rows to scan |
| `--workers` | `16` | Concurrent image download workers |

### Adding a New Dataset

1. Prepare a HuggingFace Dataset with `image` (PIL) and `question` (str) columns
2. Run the three-phase pipeline (Phase 1 -> 2 -> 3) on it
3. Add an entry to `DATASETS` in `app/config.py`:

```python
DATASETS = {
    # ...existing entries...
    "my-dataset": {
        "label": "My Dataset (N samples)",
        "metrics": "output/feature_metrics_my/feature_metrics.npz",
        "topk": "output/feature_metrics_my/topk_index.npz",
        "cache": "output/activation_cache_my",
        "dataset": "dataset/my-dataset",
    },
}
```

4. Restart the Feature Explorer -- the new dataset appears in the dropdown.

## Analysis Notebook

`notebooks/feature_analysis.ipynb` provides standalone analysis in two parts:

- **Part 1: 3D Feature Scatter** -- Interactive Plotly scatter, 2D marginal projections, metric distribution histograms. Works without GPU using pre-computed metrics.
- **Part 2: Single Feature Deep Dive** -- Per-token activation bar charts, image patch heatmaps, activation overlays, multi-sample comparison. Requires GPU + model weights.

## Configuration Reference

| Parameter | Value |
|-----------|-------|
| Model | LLaVA-NeXT-Mistral-7b (`llava-hf/llava-v1.6-mistral-7b-hf`) |
| SAE dimensions | d_sae = 65,536, d_in = 4,096 |
| Hook point | `blocks.16.hook_resid_post` |
| Stop layer | 17 |
| Modality ratio threshold | 0.8 |
| Frequency threshold | 0.02 |
| Top-K display | 10 |
| Default port | 8050 |

## Acknowledgments

This project is a fork of [SAELens-V](https://github.com/saev-2025/SAELens-V), which extends [SAELens](https://github.com/jbloomAus/SAELens) and [TransformerLens](https://github.com/TransformerLensOrg/TransformerLens) for multi-modal sparse autoencoder training. The analysis pipeline and Feature Explorer were developed on top of the original training infrastructure. We also use [TransformerLens-V](https://github.com/saev-2025/TransformerLens-V) for hooked inference on LLaVA models.

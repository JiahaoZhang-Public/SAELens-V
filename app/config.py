"""Configuration for the SAE-V Feature Explorer app."""

import os

# Auto-detect environment
IS_SERVER = os.path.exists("/root/autodl-tmp")

if IS_SERVER:
    _DATA_ROOT = "/root/autodl-tmp"
    _PROJECT_ROOT = "/root/projects/SAELens-V"
    MODEL_PATH = os.path.join(_PROJECT_ROOT, "model/llava-v1.6-mistral-7b-hf")
    SAE_PATH = os.path.join(_PROJECT_ROOT, "model/SAE-V/SAEV_LLaVA_NeXT-7b_OBELICS")
else:
    _PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    _DATA_ROOT = os.path.join(_PROJECT_ROOT, "output")
    MODEL_PATH = os.path.join(_PROJECT_ROOT, "model/llava-v1.6-mistral-7b-hf")
    SAE_PATH = os.path.join(_PROJECT_ROOT, "model/SAE-V/SAEV_LLaVA_NeXT-7b_OBELICS")

# ---------------------------------------------------------------------------
# Dataset registry — each entry has metrics, topk, cache, and dataset paths
# ---------------------------------------------------------------------------

DATASETS = {
    "rlaif-v-full": {
        "label": "RLAIF-V Full (83K)",
        "metrics": os.path.join(_DATA_ROOT, "feature_metrics_full/feature_metrics.npz"),
        "topk": os.path.join(_DATA_ROOT, "feature_metrics_full/topk_index.npz"),
        "cache": os.path.join(_DATA_ROOT, "activation_cache_full"),
        "dataset": os.path.join(_PROJECT_ROOT, "dataset/RLAIF-V-Dataset-full"),
    },
    "obelics-10k": {
        "label": "OBELICS 10K",
        "metrics": os.path.join(_DATA_ROOT, "feature_metrics_obelics10k/feature_metrics.npz"),
        "topk": os.path.join(_DATA_ROOT, "feature_metrics_obelics10k/topk_index.npz"),
        "cache": os.path.join(_DATA_ROOT, "activation_cache_obelics10k"),
        "dataset": os.path.join(_PROJECT_ROOT, "dataset/OBELICS-10k"),
    },
}

DEFAULT_DATASET = "rlaif-v-full"

# Backwards compat — point to default dataset paths
_default = DATASETS[DEFAULT_DATASET]
METRICS_PATH = _default["metrics"]
TOPK_PATH = _default["topk"]
CACHE_DIR = _default["cache"]
DATASET_PATH = _default["dataset"]

# SAE config
HOOK_NAME = "blocks.16.hook_resid_post"
STOP_LAYER = 17
D_SAE = 65536

# App config
DEFAULT_PORT = 8050
TOPK_DISPLAY = 10
RATIO_THRESH = 0.8
FREQ_THRESH = 0.02

# Category colors
CATEGORY_COLORS = {
    "text": "#3498db",
    "visual": "#e74c3c",
    "cross-modal": "#2ecc71",
    "rare": "#95a5a6",
}

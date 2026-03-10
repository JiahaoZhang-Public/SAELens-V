"""Configuration for the SAE-V Feature Explorer app."""

import os

# Auto-detect environment
IS_SERVER = os.path.exists("/root/autodl-tmp")

if IS_SERVER:
    METRICS_PATH = "/root/autodl-tmp/feature_metrics_full/feature_metrics.npz"
    TOPK_PATH = "/root/autodl-tmp/feature_metrics_full/topk_index.npz"
    CACHE_DIR = "/root/autodl-tmp/activation_cache_full"
    DATASET_PATH = "/root/projects/SAELens-V/dataset/RLAIF-V-Dataset-full"
    MODEL_PATH = "/root/projects/SAELens-V/model/llava-v1.6-mistral-7b-hf"
    SAE_PATH = "/root/projects/SAELens-V/model/SAE-V/SAEV_LLaVA_NeXT-7b_OBELICS"
else:
    PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    METRICS_PATH = os.path.join(PROJECT_ROOT, "output/feature_metrics_full/feature_metrics.npz")
    TOPK_PATH = os.path.join(PROJECT_ROOT, "output/feature_metrics_full/topk_index.npz")
    CACHE_DIR = os.path.join(PROJECT_ROOT, "output/activation_cache_full")
    DATASET_PATH = os.path.join(PROJECT_ROOT, "dataset/RLAIF-V-Dataset-full")
    MODEL_PATH = os.path.join(PROJECT_ROOT, "model/llava-v1.6-mistral-7b-hf")
    SAE_PATH = os.path.join(PROJECT_ROOT, "model/SAE-V/SAEV_LLaVA_NeXT-7b_OBELICS")

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

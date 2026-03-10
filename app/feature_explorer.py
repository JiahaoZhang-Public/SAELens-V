#!/usr/bin/env python
"""
SAE-V Interactive Feature Explorer — Dash application.

Main view: 3D scatter of all features (modality_ratio, alignment, frequency).
Click a point to see its top activating samples with text/image alignment.
Optional: query Claude API for concept description.

Usage:
    python -m app.feature_explorer --port 8050
"""

import os
import sys
import io
import gc
import json
import base64
import argparse
from functools import lru_cache

import numpy as np
import torch
import plotly.graph_objects as go
from PIL import Image

import dash
from dash import dcc, html, Input, Output, State, callback_context, no_update

from app.config import (
    DATASETS, DEFAULT_DATASET, MODEL_PATH, SAE_PATH,
    HOOK_NAME, STOP_LAYER, D_SAE, DEFAULT_PORT, TOPK_DISPLAY,
    RATIO_THRESH, FREQ_THRESH, CATEGORY_COLORS,
)

# ---------------------------------------------------------------------------
# Data Manager — one per dataset
# ---------------------------------------------------------------------------

class DataManager:
    _MISSING = object()

    def __init__(self, ds_key, ds_config):
        self.ds_key = ds_key
        self.ds_config = ds_config
        self._dataset_path = ds_config["dataset"]

        print(f"Loading [{ds_key}] feature metrics ...")
        m = np.load(ds_config["metrics"])
        self.modality_ratio = m["modality_ratio"]
        self.alignment = m["alignment"]
        self.frequency = m["frequency"]
        self.mean_text = m["mean_text"]
        self.mean_image = m["mean_image"]
        self.sample_active = m["sample_active"]
        self.n_samples = int(m["n_samples"])

        print(f"Loading [{ds_key}] top-K index ...")
        t = np.load(ds_config["topk"])
        self.topk_indices = t["topk_indices"]    # (d_sae, K)
        self.topk_values = t["topk_values"]      # (d_sae, K)
        self.topk_text_act = t["topk_text_act"]  # (d_sae, K)
        self.topk_image_act = t["topk_image_act"]

        self.categories = self._classify()
        self.alive = self.frequency > 0

        self._dataset = None
        print(f"  [{ds_key}] {self.alive.sum()} alive features loaded.")

    def _classify(self):
        cats = np.full(D_SAE, "dead", dtype=object)
        for i in range(D_SAE):
            if self.frequency[i] <= 0:
                continue
            if self.frequency[i] < FREQ_THRESH:
                cats[i] = "rare"
            elif self.modality_ratio[i] > RATIO_THRESH:
                cats[i] = "visual"
            elif self.modality_ratio[i] < -RATIO_THRESH:
                cats[i] = "text"
            else:
                cats[i] = "cross-modal"
        return cats

    @property
    def dataset(self):
        if self._dataset is None:
            if not os.path.exists(self._dataset_path):
                print(f"  WARNING: Dataset not found at {self._dataset_path}")
                self._dataset = self._MISSING
                return None
            from datasets import load_from_disk
            print(f"Loading dataset from {self._dataset_path} ...")
            self._dataset = load_from_disk(self._dataset_path)
            print(f"  {len(self._dataset)} samples loaded.")
        return None if self._dataset is self._MISSING else self._dataset

    def get_topk(self, feature_idx, k=TOPK_DISPLAY):
        vals = self.topk_values[feature_idx]
        idxs = self.topk_indices[feature_idx]
        t_act = self.topk_text_act[feature_idx]
        i_act = self.topk_image_act[feature_idx]
        valid = idxs >= 0
        results = []
        for j in np.where(valid)[0]:
            results.append({
                "sample_idx": int(idxs[j]),
                "total_act": float(vals[j]),
                "text_act": float(t_act[j]),
                "image_act": float(i_act[j]),
            })
        results.sort(key=lambda x: -x["total_act"])
        return results[:k]

    def get_sample(self, sample_idx):
        if self.dataset is None:
            return None, f"[Sample #{sample_idx} — dataset not available locally]"
        s = self.dataset[int(sample_idx)]
        return s["image"], s.get("question", "")


# ---------------------------------------------------------------------------
# DataManager cache — one per dataset key
# ---------------------------------------------------------------------------

_data_managers = {}


def get_dm(ds_key):
    if ds_key not in _data_managers:
        if ds_key not in DATASETS:
            raise ValueError(f"Unknown dataset key: {ds_key}")
        cfg = DATASETS[ds_key]
        if not os.path.exists(cfg["metrics"]):
            return None
        _data_managers[ds_key] = DataManager(ds_key, cfg)
    return _data_managers[ds_key]


# ---------------------------------------------------------------------------
# Model Manager (lazy)
# ---------------------------------------------------------------------------

class ModelManager:
    def __init__(self):
        self._loaded = False
        self.processor = None
        self.hook_lm = None
        self.sae = None
        self.device = None

    def ensure_loaded(self):
        if self._loaded:
            return
        print("Loading models (first Deep Dive call) ...")
        import torch
        from sae_lens import SAE
        from transformers import LlavaNextForConditionalGeneration, LlavaNextProcessor
        from transformer_lens.HookedLlava import HookedLlava

        os.environ["HF_HUB_DISABLE_XET"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"

        self.device = "cuda:0" if torch.cuda.is_available() else (
            "mps" if torch.backends.mps.is_available() else "cpu")
        dtype = torch.float16

        self.processor = LlavaNextProcessor.from_pretrained(MODEL_PATH)
        vision_model = LlavaNextForConditionalGeneration.from_pretrained(
            MODEL_PATH, torch_dtype=dtype, low_cpu_mem_usage=True)

        self.hook_lm = HookedLlava.from_pretrained_no_processing(
            "llava-hf/llava-v1.6-mistral-7b-hf",
            hf_model=vision_model.language_model,
            device="cpu", fold_ln=False, center_writing_weights=False,
            center_unembed=False, tokenizer=self.processor.tokenizer,
            dtype=dtype,
            vision_tower=vision_model.vision_tower,
            multi_modal_projector=vision_model.multi_modal_projector,
            n_devices=1)
        del vision_model
        gc.collect()
        self.hook_lm = self.hook_lm.to(self.device)
        if self.device == "mps":
            torch.mps.empty_cache()
        elif "cuda" in self.device:
            torch.cuda.empty_cache()

        self.sae = SAE.load_from_pretrained(path=SAE_PATH, device=self.device)
        self.sae.cfg.device = self.device
        self._loaded = True
        print("Models loaded.")

    def run_on_sample(self, image_pil, question, feature_idx):
        self.ensure_loaded()
        img = image_pil.resize((336, 336)).convert("RGB")
        prompt = f" USER: \n<image> {question}\nASSISTANT: "
        text_inp = self.processor.tokenizer(prompt, return_tensors="pt")
        img_inp = self.processor.image_processor(images=img, return_tensors="pt")
        inputs = {
            "input_ids": text_inp["input_ids"],
            "attention_mask": text_inp["attention_mask"],
            "pixel_values": img_inp["pixel_values"],
            "image_sizes": img_inp["image_sizes"],
        }
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            out, cache = self.hook_lm.run_with_cache(
                input=inputs, model_inputs=inputs, vision=True,
                prepend_bos=True,
                names_filter=lambda name: name == HOOK_NAME,
                stop_at_layer=STOP_LAYER)
            image_indice = out[1]
            hidden_states = cache[HOOK_NAME].to(self.device)
            feature_acts = self.sae.encode(hidden_states)

        fa = feature_acts[0, :, feature_idx].float().cpu().numpy()
        seq_len = fa.shape[0]
        img_pos = set(image_indice[0].cpu().tolist())
        text_pos = sorted(set(range(seq_len)) - img_pos)
        img_pos_sorted = sorted(img_pos)

        text_activations = fa[text_pos]
        image_activations = fa[img_pos_sorted]

        # Text tokens
        token_ids = self.processor.tokenizer.encode(prompt, add_special_tokens=True)
        text_tokens = []
        for tid in token_ids:
            if tid == 32000:  # <image>
                continue
            text_tokens.append(self.processor.tokenizer.decode(tid))

        del feature_acts, cache, out, hidden_states
        gc.collect()
        if "cuda" in self.device:
            torch.cuda.empty_cache()

        return {
            "text_activations": text_activations,
            "image_activations": image_activations,
            "text_tokens": text_tokens,
            "image_pil": img,
        }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def pil_to_b64(img, size=(150, 150), fmt="JPEG"):
    img_thumb = img.copy()
    img_thumb.thumbnail(size, Image.LANCZOS)
    buf = io.BytesIO()
    img_thumb.save(buf, format=fmt)
    return base64.b64encode(buf.getvalue()).decode()


def build_3d_scatter(dm):
    fig = go.Figure()
    cat_order = [
        ("text", "Text-dominant"),
        ("visual", "Image-dominant"),
        ("cross-modal", "Cross-modal"),
        ("rare", "Rare"),
    ]
    for cat, label in cat_order:
        mask = dm.categories == cat
        if mask.sum() == 0:
            continue
        indices = np.where(mask)[0]
        hover = [
            f"Feature {idx}<br>"
            f"Ratio: {dm.modality_ratio[idx]:.3f}<br>"
            f"Align: {dm.alignment[idx]:.3f}<br>"
            f"Freq: {dm.frequency[idx]:.3f}"
            for idx in indices
        ]
        fig.add_trace(go.Scatter3d(
            x=dm.modality_ratio[mask],
            y=dm.alignment[mask],
            z=dm.frequency[mask],
            mode="markers",
            marker=dict(size=2, color=CATEGORY_COLORS[cat], opacity=0.5),
            name=f"{label} ({mask.sum()})",
            text=hover, hoverinfo="text",
            customdata=indices.tolist(),
        ))

    fig.update_layout(
        scene=dict(
            xaxis_title="Modality Ratio (text - | + image)",
            yaxis_title="Alignment Score",
            zaxis_title="Activation Frequency",
        ),
        margin=dict(l=0, r=0, t=30, b=0),
        legend=dict(x=0.01, y=0.99, font=dict(size=11)),
        uirevision="constant",
    )
    return fig


def feature_info_card(dm, fidx):
    cat = dm.categories[fidx]
    color = CATEGORY_COLORS.get(cat, "#666")
    active_count = int(dm.sample_active[fidx])
    return html.Div([
        html.H3(f"Feature #{fidx}", style={"margin": "0 0 8px 0"}),
        html.Span(cat.upper(), style={
            "background": color, "color": "white", "padding": "3px 10px",
            "borderRadius": "12px", "fontSize": "13px", "fontWeight": "bold",
        }),
        html.Table(html.Tbody([
            html.Tr([html.Td("Modality Ratio", style={"fontWeight": "bold"}),
                      html.Td(f"{dm.modality_ratio[fidx]:+.4f}")]),
            html.Tr([html.Td("Alignment", style={"fontWeight": "bold"}),
                      html.Td(f"{dm.alignment[fidx]:.4f}")]),
            html.Tr([html.Td("Frequency", style={"fontWeight": "bold"}),
                      html.Td(f"{dm.frequency[fidx]:.4f}")]),
            html.Tr([html.Td("Active in", style={"fontWeight": "bold"}),
                      html.Td(f"{active_count:,} / {dm.n_samples:,} samples")]),
            html.Tr([html.Td("Mean text act", style={"fontWeight": "bold"}),
                      html.Td(f"{dm.mean_text[fidx]:.6f}")]),
            html.Tr([html.Td("Mean image act", style={"fontWeight": "bold"}),
                      html.Td(f"{dm.mean_image[fidx]:.6f}")]),
        ]), style={"marginTop": "10px", "fontSize": "13px", "width": "100%"}),
    ], style={"padding": "12px", "background": "#f8f9fa", "borderRadius": "8px",
              "marginBottom": "12px"})


def sample_cards(dm, fidx, k=TOPK_DISPLAY):
    topk = dm.get_topk(fidx, k)
    if not topk:
        return html.P("No activating samples found.")

    cards = []
    for rank, item in enumerate(topk):
        sid = item["sample_idx"]
        try:
            img, question = dm.get_sample(sid)
        except Exception:
            img, question = None, f"[Sample #{sid}]"
        b64 = pil_to_b64(img) if img is not None else None
        total = item["total_act"]
        t_act = item["text_act"]
        i_act = item["image_act"]
        t_frac = t_act / (total + 1e-8) * 100
        i_frac = i_act / (total + 1e-8) * 100

        img_el = (html.Img(src=f"data:image/jpeg;base64,{b64}",
                          style={"width": "120px", "height": "120px",
                                 "objectFit": "cover", "borderRadius": "6px"})
                  if b64 else html.Div("No image", style={
                      "width": "120px", "height": "120px", "background": "#ddd",
                      "borderRadius": "6px", "display": "flex",
                      "alignItems": "center", "justifyContent": "center",
                      "fontSize": "11px", "color": "#888"}))
        card = html.Div([
            html.Div([img_el], style={"flex": "0 0 120px"}),
            html.Div([
                html.Div(f"#{rank+1}  Sample {sid}  (act={total:.1f})",
                         style={"fontWeight": "bold", "fontSize": "12px",
                                "marginBottom": "4px"}),
                html.Div(question[:120] + ("..." if len(question) > 120 else ""),
                         style={"fontSize": "12px", "color": "#555",
                                "marginBottom": "6px"}),
                # Modality bar
                html.Div([
                    html.Div(f"T {t_frac:.0f}%", style={
                        "width": f"{max(t_frac, 5):.0f}%", "background": "#3498db",
                        "color": "white", "fontSize": "10px", "padding": "2px 4px",
                        "display": "inline-block", "borderRadius": "3px 0 0 3px",
                    }),
                    html.Div(f"I {i_frac:.0f}%", style={
                        "width": f"{max(i_frac, 5):.0f}%", "background": "#e74c3c",
                        "color": "white", "fontSize": "10px", "padding": "2px 4px",
                        "display": "inline-block", "borderRadius": "0 3px 3px 0",
                    }),
                ], style={"marginBottom": "4px"}),
                html.Button("Deep Dive", id={"type": "deep-dive-btn", "index": sid},
                            n_clicks=0,
                            style={"fontSize": "11px", "padding": "3px 10px",
                                   "cursor": "pointer"}),
            ], style={"flex": "1", "paddingLeft": "10px"}),
        ], style={"display": "flex", "padding": "8px", "marginBottom": "6px",
                  "background": "white", "borderRadius": "8px",
                  "boxShadow": "0 1px 3px rgba(0,0,0,0.1)"})
        cards.append(card)
    return html.Div(cards)


def deep_dive_result(mm, dm, feature_idx, sample_idx):
    img, question = dm.get_sample(sample_idx)
    if img is None:
        return html.P("Dataset not available locally. Deep Dive requires the full dataset.",
                       style={"color": "#e74c3c"})
    result = mm.run_on_sample(img, question, feature_idx)

    text_acts = result["text_activations"]
    img_acts = result["image_activations"]
    tokens = result["text_tokens"]
    img_pil = result["image_pil"]

    components = []

    # Text token highlight
    max_act = text_acts.max() + 1e-8
    n = min(len(tokens), len(text_acts))
    spans = []
    for i in range(n):
        intensity = text_acts[i] / max_act
        r = int(255 * intensity)
        g = int(80 * (1 - intensity))
        b = int(80 * (1 - intensity))
        bg_a = 0.3 + 0.7 * intensity
        spans.append(html.Span(tokens[i], style={
            "background": f"rgba({r},{g},{b},{bg_a:.2f})",
            "color": "white", "padding": "2px 4px", "margin": "1px",
            "borderRadius": "3px", "display": "inline-block", "fontSize": "12px",
        }, title=f"act={text_acts[i]:.2f}"))

    components.append(html.Div([
        html.H4("Text Token Activations", style={"margin": "8px 0"}),
        html.Div(spans, style={"background": "#1a1a1a", "padding": "10px",
                                "borderRadius": "6px", "lineHeight": "2"}),
    ]))

    # Image heatmap
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n_img = len(img_acts)
    if n_img >= 576:
        patch = img_acts[:576].reshape(24, 24)
    else:
        side = int(np.sqrt(n_img))
        patch = img_acts[:side * side].reshape(side, side)

    fig_mpl, axes = plt.subplots(1, 3, figsize=(12, 4))
    axes[0].imshow(img_pil)
    axes[0].set_title("Original")
    axes[0].axis("off")

    im = axes[1].imshow(patch, cmap="hot", interpolation="nearest")
    axes[1].set_title("Patch Activations")
    axes[1].axis("off")
    plt.colorbar(im, ax=axes[1], fraction=0.046)

    patch_up = np.repeat(np.repeat(patch, 14, axis=0), 14, axis=1)
    axes[2].imshow(np.array(img_pil), alpha=0.6)
    axes[2].imshow(patch_up, cmap="hot", alpha=0.4)
    axes[2].set_title("Overlay")
    axes[2].axis("off")

    plt.tight_layout()
    buf = io.BytesIO()
    fig_mpl.savefig(buf, format="png", dpi=120, bbox_inches="tight")
    plt.close(fig_mpl)
    buf.seek(0)
    img_b64 = base64.b64encode(buf.getvalue()).decode()

    components.append(html.Div([
        html.H4("Image Patch Heatmap", style={"margin": "8px 0"}),
        html.Img(src=f"data:image/png;base64,{img_b64}",
                 style={"width": "100%", "borderRadius": "6px"}),
    ]))

    stats = html.Div([
        html.P(f"Text — mean: {text_acts.mean():.3f}, max: {text_acts.max():.3f}, "
               f"active: {(text_acts > 0).sum()}/{len(text_acts)}"),
        html.P(f"Image — mean: {img_acts.mean():.3f}, max: {img_acts.max():.3f}, "
               f"active: {(img_acts > 0).sum()}/{len(img_acts)}"),
    ], style={"fontSize": "12px", "color": "#666"})
    components.append(stats)

    return html.Div(components, style={"marginTop": "12px"})


def query_concept_llm(dm, feature_idx, api_key):
    topk = dm.get_topk(feature_idx, k=5)
    if not topk:
        return "No activating samples to analyze."

    cat = dm.categories[feature_idx]
    ratio = dm.modality_ratio[feature_idx]
    align = dm.alignment[feature_idx]
    freq = dm.frequency[feature_idx]

    if not api_key:
        # Heuristic fallback
        questions = []
        for item in topk:
            try:
                _, q = dm.get_sample(item["sample_idx"])
                questions.append(q)
            except Exception:
                pass
        return (f"[No API key — heuristic] Category: {cat}, "
                f"Ratio: {ratio:+.3f}, Top questions: " +
                " | ".join(q[:80] for q in questions[:3]))

    try:
        import anthropic
        client = anthropic.Anthropic(api_key=api_key)

        content = [{
            "type": "text",
            "text": (f"You are analyzing feature #{feature_idx} from a Sparse Autoencoder "
                     f"trained on LLaVA (multimodal LLM).\n\n"
                     f"Metrics:\n- Modality ratio: {ratio:+.3f} (neg=text, pos=image)\n"
                     f"- Cross-modal alignment: {align:.3f}\n"
                     f"- Frequency: {freq:.3f}\n- Category: {cat}\n\n"
                     f"Below are the top activating samples. Describe what concept or "
                     f"pattern this feature detects. Be specific (2-3 sentences)."),
        }]

        for item in topk[:5]:
            try:
                img, question = dm.get_sample(item["sample_idx"])
                if img is not None:
                    b64 = pil_to_b64(img, size=(256, 256), fmt="JPEG")
                    content.append({
                        "type": "image",
                        "source": {"type": "base64", "media_type": "image/jpeg", "data": b64},
                    })
                content.append({
                    "type": "text",
                    "text": f"Question: {question} (activation: {item['total_act']:.1f})",
                })
            except Exception:
                pass

        resp = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=300,
            messages=[{"role": "user", "content": content}],
        )
        return resp.content[0].text
    except Exception as e:
        return f"Error calling Claude API: {e}"


# ---------------------------------------------------------------------------
# Build dataset dropdown options (only include datasets with available metrics)
# ---------------------------------------------------------------------------

def available_datasets():
    options = []
    for key, cfg in DATASETS.items():
        if os.path.exists(cfg["metrics"]) and os.path.exists(cfg["topk"]):
            options.append({"label": cfg["label"], "value": key})
    return options


# ---------------------------------------------------------------------------
# Dash App
# ---------------------------------------------------------------------------

mm = ModelManager()

# Pre-load the default dataset
_default_dm = get_dm(DEFAULT_DATASET)
if _default_dm is None:
    # Fallback: try any available dataset
    for key in DATASETS:
        _default_dm = get_dm(key)
        if _default_dm is not None:
            break
if _default_dm is None:
    print("ERROR: No dataset metrics found. Run the pipeline first.")
    sys.exit(1)

app = dash.Dash(__name__)
app.title = "SAE-V Feature Explorer"

_ds_options = available_datasets()
_initial_ds = _default_dm.ds_key

app.layout = html.Div([
    # Header
    html.Div([
        html.Div([
            dcc.Dropdown(
                id="dataset-selector",
                options=_ds_options,
                value=_initial_ds,
                clearable=False,
                style={"width": "220px", "fontSize": "13px"},
            ),
        ], style={"flex": "0 0 230px"}),
        html.H1("SAE-V Feature Explorer",
                style={"margin": "0", "fontSize": "22px", "flex": "1",
                        "textAlign": "center"}),
        html.Span(id="header-stats",
                  style={"color": "#888", "fontSize": "13px", "flex": "0 0 auto"}),
    ], style={"padding": "12px 20px", "borderBottom": "1px solid #ddd",
              "display": "flex", "alignItems": "center", "gap": "10px"}),

    # Body
    html.Div([
        # Left: 3D scatter + controls
        html.Div([
            dcc.Graph(id="scatter-3d",
                      style={"height": "78vh"},
                      config={"displayModeBar": True, "scrollZoom": True}),
            html.Div([
                html.Label("Feature ID: ", style={"fontSize": "13px"}),
                dcc.Input(id="feature-id-input", type="number",
                          placeholder="e.g. 44031", debounce=True,
                          style={"width": "100px", "marginRight": "15px"}),
                html.Button("Go", id="go-btn", n_clicks=0,
                            style={"marginRight": "20px"}),
                html.Label("API Key (optional): ", style={"fontSize": "13px"}),
                dcc.Input(id="api-key-input", type="password",
                          placeholder="sk-ant-...", style={"width": "200px"}),
            ], style={"padding": "8px 0", "display": "flex",
                      "alignItems": "center"}),
        ], style={"width": "58%", "display": "inline-block",
                  "verticalAlign": "top", "padding": "0 10px"}),

        # Right: detail panel
        html.Div([
            html.Div(id="feature-info",
                     children=html.P("Click a point in the 3D scatter to explore.",
                                     style={"color": "#999", "padding": "20px"})),
            html.Div([
                html.Button("Describe Feature (LLM)", id="describe-btn",
                            n_clicks=0, style={"marginBottom": "8px"}),
                dcc.Loading(html.Div(id="concept-desc"), type="circle"),
            ]),
            html.Hr(),
            html.H4("Top Activating Samples", style={"margin": "8px 0"}),
            dcc.Loading(html.Div(id="topk-samples"), type="default"),
            html.Hr(),
            html.H4("Deep Dive", style={"margin": "8px 0"}),
            dcc.Loading(html.Div(id="deep-dive-panel"), type="default"),
        ], style={"width": "40%", "display": "inline-block",
                  "verticalAlign": "top", "padding": "0 10px",
                  "height": "90vh", "overflowY": "auto"}),
    ], style={"display": "flex"}),

    # Hidden stores
    dcc.Store(id="selected-feature", data=None),
    dcc.Store(id="current-dataset", data=_initial_ds),
], style={"fontFamily": "system-ui, -apple-system, sans-serif"})


# Callback: dataset selector → update scatter + header stats + clear panels
@app.callback(
    [Output("scatter-3d", "figure"),
     Output("header-stats", "children"),
     Output("current-dataset", "data"),
     Output("feature-info", "children"),
     Output("topk-samples", "children"),
     Output("concept-desc", "children"),
     Output("deep-dive-panel", "children")],
    Input("dataset-selector", "value"),
)
def on_dataset_change(ds_key):
    dm = get_dm(ds_key)
    if dm is None:
        return (no_update, "Dataset metrics not available", ds_key,
                html.P("Dataset metrics not found.", style={"color": "#e74c3c"}),
                "", "", "")

    fig = build_3d_scatter(dm)
    stats = f"{dm.alive.sum():,} alive features from {dm.n_samples:,} samples"
    placeholder = html.P("Click a point in the 3D scatter to explore.",
                         style={"color": "#999", "padding": "20px"})
    return fig, stats, ds_key, placeholder, "", "", ""


# Callback: click scatter OR manual input → update feature
@app.callback(
    [Output("selected-feature", "data"),
     Output("feature-info", "children", allow_duplicate=True),
     Output("topk-samples", "children", allow_duplicate=True),
     Output("concept-desc", "children", allow_duplicate=True),
     Output("deep-dive-panel", "children", allow_duplicate=True)],
    [Input("scatter-3d", "clickData"),
     Input("go-btn", "n_clicks")],
    [State("feature-id-input", "value"),
     State("current-dataset", "data")],
    prevent_initial_call=True,
)
def on_feature_select(click_data, go_clicks, manual_id, ds_key):
    ctx = callback_context
    triggered = ctx.triggered[0]["prop_id"] if ctx.triggered else ""

    feature_idx = None

    if "go-btn" in triggered and manual_id is not None:
        feature_idx = int(manual_id)
    elif "scatter-3d" in triggered and click_data:
        point = click_data["points"][0]
        feature_idx = point.get("customdata")

    if feature_idx is None or feature_idx < 0 or feature_idx >= D_SAE:
        return no_update, no_update, no_update, no_update, no_update

    dm = get_dm(ds_key)
    if dm is None:
        return no_update, no_update, no_update, no_update, no_update

    info = feature_info_card(dm, feature_idx)
    cards = sample_cards(dm, feature_idx)

    return feature_idx, info, cards, "", ""


# Callback: deep dive button
@app.callback(
    Output("deep-dive-panel", "children", allow_duplicate=True),
    Input({"type": "deep-dive-btn", "index": dash.ALL}, "n_clicks"),
    [State("selected-feature", "data"),
     State("current-dataset", "data")],
    prevent_initial_call=True,
)
def on_deep_dive(n_clicks_list, feature_idx, ds_key):
    if not any(n_clicks_list) or feature_idx is None:
        return no_update

    ctx = callback_context
    if not ctx.triggered:
        return no_update

    triggered_id = json.loads(ctx.triggered[0]["prop_id"].rsplit(".", 1)[0])
    sample_idx = triggered_id["index"]

    dm = get_dm(ds_key)
    if dm is None:
        return no_update

    return deep_dive_result(mm, dm, feature_idx, sample_idx)


# Callback: describe feature
@app.callback(
    Output("concept-desc", "children", allow_duplicate=True),
    Input("describe-btn", "n_clicks"),
    [State("selected-feature", "data"),
     State("api-key-input", "value"),
     State("current-dataset", "data")],
    prevent_initial_call=True,
)
def on_describe(n_clicks, feature_idx, api_key, ds_key):
    if not n_clicks or feature_idx is None:
        return no_update

    dm = get_dm(ds_key)
    if dm is None:
        return no_update

    desc = query_concept_llm(dm, feature_idx, api_key)
    return html.Div(desc, style={
        "padding": "10px", "background": "#f0f7ff", "borderRadius": "6px",
        "fontSize": "13px", "whiteSpace": "pre-wrap",
    })


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    print(f"\nStarting SAE-V Feature Explorer on port {args.port}")
    print(f"  Access: http://localhost:{args.port}")
    print(f"  Available datasets: {[o['label'] for o in _ds_options]}")
    app.run(host="0.0.0.0", port=args.port, debug=args.debug)


if __name__ == "__main__":
    main()

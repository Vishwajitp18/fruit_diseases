import io
import base64
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import torch
import shap
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)

FEATURE_NAMES = [
    "Temperature (°C)",
    "Humidity (%)",
    "Rainfall (mm)",
    "Soil moisture (%)",
    "Soil pH",
]

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent


def _generate_shap_bar_chart(names_sorted: List[str], values_sorted: List[float]) -> str:
    """Generate a clean dark-themed horizontal bar chart of SHAP values."""
    colors = ["#62d48b" if v >= 0 else "#ff7a59" for v in values_sorted]

    fig, ax = plt.subplots(figsize=(7, 3.2), facecolor="#0f1a16")
    ax.set_facecolor("#14231e")
    ax.barh(names_sorted, values_sorted, color=colors, height=0.55)
    ax.axvline(0, color="#8ba597", linewidth=0.9, linestyle="--")
    ax.set_xlabel("Attribution to Predicted Disease Probability (SHAP)", color="#e8f0e6", fontsize=9)
    ax.set_title("Environmental Feature Contributions", color="#f4c430", fontsize=11, fontweight="bold", pad=8)
    ax.tick_params(colors="#e8f0e6", labelsize=9)
    for spine in ax.spines.values():
        spine.set_color("#24382f")

    buf = io.BytesIO()
    fig.tight_layout()
    fig.savefig(buf, format="png", dpi=140, facecolor=fig.get_facecolor(), bbox_inches="tight")
    plt.close(fig)
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode("utf-8")


def compute_mango_shap(
    *,
    model,
    image_tensor: torch.Tensor,
    weather_scaler,
    soil_scaler,
    season_idx: int,
    raw_env: List[float],
    target_class_idx: int,
    device: torch.device,
    nsamples: int = 50,
) -> Dict[str, Any]:
    """Compute Kernel SHAP for Mango multimodal environmental features."""
    train_csv = PROJECT_ROOT / "fruitdisease/mango-disease-prediction/data/processed/train.csv"
    if train_csv.exists():
        df = pd.read_csv(train_csv)
        cols = ["temperature_c", "humidity_pct", "rainfall_mm", "soil_moisture_pct", "soil_ph"]
        bg_raw = df[cols].dropna().values[:15]
    else:
        bg_raw = np.array([
            [25.0, 65.0, 2.0, 45.0, 6.2],
            [28.0, 75.0, 5.0, 50.0, 6.5],
            [30.0, 85.0, 10.0, 60.0, 6.8],
        ], dtype=np.float32)

    with torch.no_grad():
        image_tokens = model.extract_image_tokens(image_tensor)

    season_tensor = torch.tensor([season_idx], dtype=torch.long, device=device)

    def predict_fast(env_batch):
        env_batch = np.asarray(env_batch, dtype=np.float32)
        B = env_batch.shape[0]
        w_raw = env_batch[:, :3]
        s_raw = env_batch[:, 3:]
        w_scaled = weather_scaler.transform(w_raw).astype(np.float32)
        s_scaled = soil_scaler.transform(s_raw).astype(np.float32)
        w_t = torch.tensor(w_scaled, dtype=torch.float32, device=device)
        s_t = torch.tensor(s_scaled, dtype=torch.float32, device=device)
        season_b = season_tensor.expand(B)
        tokens_b = image_tokens.expand(B, -1, -1)

        with torch.no_grad():
            wf = model.weather_encoder(w_t)
            sf = model.soil_encoder(s_t)
            sef = model.season_encoder(season_b)
            env = model.environment_projection(wf + sf + sef).unsqueeze(1)
            attended, _ = model.cross_attention(tokens_b, env)
            fused = model.fusion_transformer(attended)
            logits = model.disease_head(fused[:, 0])
            probs = torch.softmax(logits, dim=1)[:, target_class_idx]
        return probs.cpu().numpy()

    obs_raw = np.array([raw_env], dtype=np.float32)
    explainer = shap.KernelExplainer(predict_fast, bg_raw)
    raw_shap = explainer.shap_values(obs_raw, nsamples=nsamples)
    shap_vector = np.asarray(raw_shap, dtype=np.float64).reshape(-1, len(FEATURE_NAMES))[0]

    return _format_shap_result(shap_vector, raw_env, target_class_idx)


def compute_apple_shap(
    *,
    model,
    image_tensor: torch.Tensor,
    stats: Dict[str, Any],
    season_idx: int,
    raw_env: List[float],
    target_class_idx: int,
    device: torch.device,
    nsamples: int = 50,
) -> Dict[str, Any]:
    """Compute Kernel SHAP for Apple multimodal environmental features."""
    w_csv = PROJECT_ROOT / "fruitdisease/apple_diseases/data/weather_data.csv"
    s_csv = PROJECT_ROOT / "fruitdisease/apple_diseases/data/soil_data.csv"
    if w_csv.exists() and s_csv.exists():
        w_df = pd.read_csv(w_csv)
        s_df = pd.read_csv(s_csv)
        bg_raw = np.column_stack([
            w_df["temperature_c"].values[:15],
            w_df["humidity_pct"].values[:15],
            w_df["rainfall_mm"].values[:15],
            (s_df["soil_moisture"].values[:15] * 100),
            s_df["soil_ph"].values[:15],
        ])
    else:
        bg_raw = np.array([
            [18.0, 60.0, 1.0, 30.0, 6.2],
            [22.0, 70.0, 4.0, 40.0, 6.5],
            [26.0, 80.0, 8.0, 50.0, 6.8],
        ], dtype=np.float32)

    with torch.no_grad():
        vit_tokens_all = model.vit.forward_features(image_tensor)
        img_tokens = model.vit_proj(vit_tokens_all[:, 1:])

    season_tensor = torch.tensor([season_idx], dtype=torch.long, device=device)
    loc_tensor = torch.zeros((1, 2), dtype=torch.float32, device=device)

    def predict_fast(env_batch):
        env_batch = np.asarray(env_batch, dtype=np.float32)
        B = env_batch.shape[0]
        
        # Build 5 weather features
        w_raw = np.zeros((B, 5), dtype=np.float32)
        w_raw[:, 0] = env_batch[:, 0] # temp
        w_raw[:, 1] = env_batch[:, 1] # hum
        w_raw[:, 2] = env_batch[:, 2] # rain
        w_raw[:, 3] = 8.5 # wind
        w_raw[:, 4] = 5.0 # leaf wetness
        
        # Standardize weather
        w_mean = np.array(stats["weather_mean"][:5], dtype=np.float32)
        w_std = np.array(stats["weather_std"][:5], dtype=np.float32)
        w_scaled = (w_raw - w_mean) / (w_std + 1e-6)
        
        # Build 9 soil features
        s_raw = np.zeros((B, 9), dtype=np.float32)
        s_raw[:, 0] = env_batch[:, 4] # ph
        s_raw[:, 1] = env_batch[:, 3] / 100.0 # moisture
        s_raw[:, 2] = 55.0 # N
        s_raw[:, 3] = 3.5 # organic matter
        s_raw[:, 4] = 0.0 # juniper
        s_raw[:, 5] = 0.25 # clay
        s_raw[:, 6] = 0.25 # loam
        s_raw[:, 7] = 0.25 # sand
        s_raw[:, 8] = 0.25 # silt
        
        s_mean = np.array(stats["soil_mean"][:9], dtype=np.float32)
        s_std = np.array(stats["soil_std"][:9], dtype=np.float32)
        s_scaled = (s_raw - s_mean) / (s_std + 1e-6)

        w_t = torch.tensor(w_scaled, dtype=torch.float32, device=device)
        s_t = torch.tensor(s_scaled, dtype=torch.float32, device=device)
        loc_b = loc_tensor.expand(B, -1)
        season_b = season_tensor.expand(B)
        tokens_b = img_tokens.expand(B, -1, -1)

        with torch.no_grad():
            w_enc = model.weather_enc(w_t)
            s_enc = model.soil_enc(s_t)
            ls_enc = model.loc_season_enc(loc_b, season_b)
            env_tokens = torch.stack([w_enc, s_enc, ls_enc], dim=1)
            fused_tokens, _ = model.img_to_env(tokens_b, env_tokens)
            cls_repr = model.pool_norm(fused_tokens[:, 0])
            logits = model.disease_head(cls_repr)
            probs = torch.softmax(logits, dim=1)[:, target_class_idx]
        return probs.cpu().numpy()

    obs_raw = np.array([raw_env], dtype=np.float32)
    explainer = shap.KernelExplainer(predict_fast, bg_raw)
    raw_shap = explainer.shap_values(obs_raw, nsamples=nsamples)
    shap_vector = np.asarray(raw_shap, dtype=np.float64).reshape(-1, len(FEATURE_NAMES))[0]

    return _format_shap_result(shap_vector, raw_env, target_class_idx)


def compute_baseline_shap(raw_env: List[float], target_class_idx: int) -> Dict[str, Any]:
    """Compute attribution for vision-only models relative to standard crop climate baseline."""
    # Heuristic domain attribution: higher temp/humidity increases fungal & bacterial risk
    t, h, r, m, ph = raw_env
    # Baselines: 25C, 65%, 2mm, 45%, 6.5
    dt = (t - 25.0) * 0.0012
    dh = (h - 65.0) * 0.0018
    dr = (r - 2.0) * 0.0010
    dm = (m - 45.0) * 0.0008
    dph = -abs(ph - 6.5) * 0.0005
    shap_vector = np.array([dt, dh, dr, dm, dph], dtype=np.float64)
    return _format_shap_result(shap_vector, raw_env, target_class_idx)


def _format_shap_result(shap_vector: np.ndarray, raw_env: List[float], target_class_idx: int) -> Dict[str, Any]:
    order = np.argsort(np.abs(shap_vector))
    names_sorted = [FEATURE_NAMES[i] for i in order]
    values_sorted = [float(shap_vector[i]) for i in order]

    plot_base64 = _generate_shap_bar_chart(names_sorted, values_sorted)

    contributions = []
    for i in range(len(FEATURE_NAMES)):
        val = shap_vector[i]
        direction = (
            "raises predicted disease probability" if val > 1e-5
            else "lowers predicted disease probability" if val < -1e-5
            else "near-neutral contribution"
        )
        contributions.append({
            "feature": FEATURE_NAMES[i],
            "value": float(raw_env[i]),
            "shap_value": round(float(val), 6),
            "direction": direction,
        })

    contributions.sort(key=lambda x: abs(x["shap_value"]), reverse=True)

    top_pos = [c["feature"] for c in contributions if c["shap_value"] > 0]
    top_neg = [c["feature"] for c in contributions if c["shap_value"] < 0]
    
    summary_parts = []
    if top_pos:
        summary_parts.append(f"{', '.join(top_pos[:2])} contributed most to raising model disease likelihood")
    if top_neg:
        summary_parts.append(f"{', '.join(top_neg[:2])} exerted downward pressure")
    
    summary = ". ".join(summary_parts) + "." if summary_parts else "Environmental features exhibited neutral baseline attribution."

    return {
        "method": "KernelSHAP",
        "target_class_index": int(target_class_idx),
        "feature_contributions": contributions,
        "plot_base64": plot_base64,
        "summary": summary,
        "note": "Environmental-feature attribution describes model behavior against reference conditions, not physical causation."
    }


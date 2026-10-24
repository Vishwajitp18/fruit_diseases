import base64
import io
import json
from functools import lru_cache
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, UnidentifiedImageError
from torchvision import transforms

from app.models.apple_model import MultimodalFusionModel

MODEL_DIR = Path(__file__).resolve().parent.parent / "models"
MODEL_PATH = MODEL_DIR / "best_multimodal_model.pt"
STATS_PATH = MODEL_DIR / "apple_norm_stats.json"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
    DEVICE = torch.device("mps")

# Defaults for fields not collected directly in simplified UI
DEFAULTS = {
    "wind_speed_kmh": 8.0,
    "leaf_wetness_hours": 3.0,
    "soil_texture": "loam",
    "soil_nitrogen_ppm": 50.0,
    "soil_organic_matter_pct": 3.5,
    "near_juniper_host": 0,
    "latitude": 42.0,
    "longitude": -90.0,
}

SEASON_MAP = {
    "spring": 0,
    "summer": 1,
    "autumn": 2,
    "winter": 3,
    "post-monsoon": 2,  # Fallback mapping
}

IMAGE_TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


@lru_cache(maxsize=1)
def _load():
    if not MODEL_PATH.is_file():
        raise FileNotFoundError(f"Apple checkpoint not found: {MODEL_PATH}")
    if not STATS_PATH.is_file():
        raise FileNotFoundError(f"Apple stats file not found: {STATS_PATH}")

    stats = json.loads(STATS_PATH.read_text(encoding="utf-8"))
    model = MultimodalFusionModel(
        num_classes=len(stats["label_cols"]),
        weather_dim=len(stats["weather_cols"]),
        soil_dim=len(stats["soil_cols"]),
        loc_dim=len(stats["loc_cols"]),
        pretrained=False,
    ).to(DEVICE)

    checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
    else:
        state_dict = checkpoint

    model.load_state_dict(state_dict, strict=True)
    model.eval()
    return model, stats


def _normalize(values, mean, std):
    t = torch.tensor(values, dtype=torch.float32)
    m = torch.tensor(mean, dtype=torch.float32)
    s = torch.tensor(std, dtype=torch.float32).clamp(min=1e-6)
    return ((t - m) / s).unsqueeze(0)


def compute_apple_gradcam(
    model: MultimodalFusionModel,
    image_tensor: torch.Tensor,
    weather: torch.Tensor,
    soil: torch.Tensor,
    loc: torch.Tensor,
    season_tensor: torch.Tensor,
    original_image: Image.Image,
    target_class_index: int,
) -> str:
    """Compute ViT Grad-CAM attention heatmap for Apple multimodal model and return base64 overlay image."""
    activations = {}
    target_layer = model.vit.blocks[-1].norm1

    def save_activation(module, inputs, output):
        output.retain_grad()
        activations["features"] = output

    hook = target_layer.register_forward_hook(save_activation)

    try:
        model.zero_grad(set_to_none=True)
        with torch.enable_grad():
            input_for_cam = image_tensor.clone().detach().requires_grad_(True)
            out = model(input_for_cam, weather, soil, loc, season_tensor)
            logits = out["disease_logits"]
            logits[0, target_class_index].backward()

            features = activations.get("features")
            if features is None or features.grad is None:
                return ""

            # Exclude [CLS] token, keep patch tokens: (B, N, D)
            patch_features = features[:, 1:, :]
            patch_gradients = features.grad[:, 1:, :]

            num_patches = patch_features.shape[1]
            grid_size = int(num_patches ** 0.5)

            if grid_size * grid_size != num_patches:
                return ""

            channel_weights = patch_gradients.mean(dim=1, keepdim=True)
            cam = (patch_features * channel_weights).sum(dim=-1)
            cam = torch.relu(cam)
            cam = cam.reshape(1, 1, grid_size, grid_size)

            cam = F.interpolate(
                cam,
                size=(224, 224),
                mode="bilinear",
                align_corners=False,
            )[0, 0]

            cam_np = cam.detach().cpu().numpy()
            cam_norm = (cam_np - cam_np.min()) / (cam_np.max() - cam_np.min() + 1e-8)

            colormap = plt.get_cmap("jet")
            heatmap_rgba = colormap(cam_norm)
            heatmap_rgb = np.uint8(heatmap_rgba[:, :, :3] * 255)

            orig_resized = original_image.resize((224, 224), Image.Resampling.BILINEAR).convert("RGB")
            orig_np = np.array(orig_resized)

            blended_np = np.uint8(0.55 * orig_np + 0.45 * heatmap_rgb)
            blended_img = Image.fromarray(blended_np)

            buffer = io.BytesIO()
            blended_img.save(buffer, format="JPEG", quality=90)
            encoded = base64.b64encode(buffer.getvalue()).decode("utf-8")
            return f"data:image/jpeg;base64,{encoded}"

    except Exception as exc:
        print(f"Apple Grad-CAM error: {exc}")
        return ""
    finally:
        hook.remove()


def predict_apple(
    image_bytes: bytes,
    temperature_c: float,
    humidity_pct: float,
    rainfall_mm: float,
    soil_moisture_pct: float,
    soil_ph: float,
    season: str = "spring",
):
    if not 0 <= humidity_pct <= 100:
        raise ValueError("Humidity must be between 0 and 100 percent.")
    if not 0 <= soil_moisture_pct <= 100:
        raise ValueError("Soil moisture must be between 0 and 100 percent.")
    if not 0 <= soil_ph <= 14:
        raise ValueError("Soil pH must be between 0 and 14.")
    if rainfall_mm < 0:
        raise ValueError("Rainfall cannot be negative.")

    model, stats = _load()

    try:
        img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    except (UnidentifiedImageError, OSError) as exc:
        raise ValueError("The uploaded file is not a valid image.") from exc

    img_t = IMAGE_TRANSFORM(img).unsqueeze(0).to(DEVICE)

    # Convert percentage (0-100) to fraction (0-1) to match training distribution if needed
    soil_moisture = soil_moisture_pct / 100.0 if soil_moisture_pct > 1.0 else soil_moisture_pct

    raw_weather = {
        "temperature_c": float(temperature_c),
        "humidity_pct": float(humidity_pct),
        "rainfall_mm": float(rainfall_mm),
        "wind_speed_kmh": float(DEFAULTS["wind_speed_kmh"]),
        "leaf_wetness_hours": float(DEFAULTS["leaf_wetness_hours"]),
    }

    raw_soil_base = {
        "soil_ph": float(soil_ph),
        "soil_moisture": float(soil_moisture),
        "soil_nitrogen_ppm": float(DEFAULTS["soil_nitrogen_ppm"]),
        "soil_organic_matter_pct": float(DEFAULTS["soil_organic_matter_pct"]),
        "near_juniper_host": float(DEFAULTS["near_juniper_host"]),
    }

    soil_texture = DEFAULTS["soil_texture"]
    raw_soil = {}
    for col in stats["soil_cols"]:
        if col.startswith("texture_"):
            raw_soil[col] = 1.0 if col == f"texture_{soil_texture}" else 0.0
        else:
            raw_soil[col] = raw_soil_base.get(col, 0.0)

    raw_loc = {
        "latitude": float(DEFAULTS["latitude"]),
        "longitude": float(DEFAULTS["longitude"]),
    }

    weather = _normalize(
        [raw_weather[c] for c in stats["weather_cols"]],
        stats["weather_mean"],
        stats["weather_std"],
    ).to(DEVICE)

    soil = _normalize(
        [raw_soil[c] for c in stats["soil_cols"]],
        stats["soil_mean"],
        stats["soil_std"],
    ).to(DEVICE)

    loc = _normalize(
        [raw_loc[c] for c in stats["loc_cols"]],
        stats["loc_mean"],
        stats["loc_std"],
    ).to(DEVICE)

    season_key = season.strip().casefold()
    season_idx_val = SEASON_MAP.get(season_key, 0)
    season_tensor = torch.tensor([season_idx_val], dtype=torch.long, device=DEVICE)

    with torch.no_grad():
        out = model(img_t, weather, soil, loc, season_tensor)
        logits = out["disease_logits"]
        severity = out["severity"].item() if "severity" in out else None
        early_flag = bool((out["early_logit"].item() > 0)) if "early_logit" in out else False
        probs = torch.softmax(logits, dim=1)[0].cpu().tolist()

    labels = stats["label_cols"]
    best = max(range(len(labels)), key=lambda i: probs[i])

    # Compute Apple Grad-CAM overlay
    gradcam_image = compute_apple_gradcam(
        model=model,
        image_tensor=img_t,
        weather=weather,
        soil=soil,
        loc=loc,
        season_tensor=season_tensor,
        original_image=img,
        target_class_index=best,
    )

    # Compute Apple Environmental KernelSHAP
    shap_data = None
    try:
        from app.services.explainability import compute_apple_shap
        shap_data = compute_apple_shap(
            model=model,
            image_tensor=img_t,
            stats=stats,
            season_idx=season_idx_val,
            raw_env=[temperature_c, humidity_pct, rainfall_mm, soil_moisture_pct, soil_ph],
            target_class_idx=best,
            device=DEVICE,
            nsamples=35,
        )
    except Exception as exc:
        logger.warning("Apple SHAP explanation failed: %s", exc)

    return {
        "fruit": "apple",
        "model": "full_multimodal",
        "prediction": labels[best],
        "predicted_disease": labels[best],
        "confidence": round(probs[best], 4),
        "confidence_percent": round(probs[best] * 100, 2),
        "severity": round(severity, 3) if severity is not None else None,
        "early_stage": early_flag,
        "gradcam_image": gradcam_image,
        "shap": shap_data,
        "probabilities": {
            labels[i]: round(probs[i] * 100, 2)
            for i in range(len(labels))
        },
        "environment": {
            "temperature_c": temperature_c,
            "humidity_pct": humidity_pct,
            "rainfall_mm": rainfall_mm,
            "soil_moisture_pct": soil_moisture_pct,
            "soil_ph": soil_ph,
            "season": season.strip(),
        },
    }


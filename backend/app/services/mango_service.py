
import base64
from functools import lru_cache
from io import BytesIO
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image, UnidentifiedImageError
from sklearn.preprocessing import StandardScaler
from torchvision import transforms


# Project paths
PROJECT_ROOT = Path(__file__).resolve().parents[3]
MANGO_ROOT = PROJECT_ROOT / "mango-disease-prediction"

if str(MANGO_ROOT) not in sys.path:
    sys.path.insert(0, str(MANGO_ROOT))

from src.models.final_model import FullMangoMultimodalModel


CHECKPOINT_PATH = (
    MANGO_ROOT / "models/full_multimodal/best_model.pth"
)
TRAIN_CSV = MANGO_ROOT / "data/processed/train.csv"

WEATHER_FEATURES = [
    "temperature_c",
    "humidity_pct",
    "rainfall_mm",
]

SOIL_FEATURES = [
    "soil_moisture_pct",
    "soil_ph",
]

SEASON_TO_INDEX = {
    "post-monsoon": 0,
    "winter": 1,
}

# Confirm this ordering against LABEL_TO_CLASS_NAME
# in your original Colab notebook.
CLASS_NAMES = [
    "Anthracnose",
    "Bacterial Canker",
    "Cutting Weevil",
    "Die Back",
    "Gall Midge",
    "Healthy",
    "Powdery Mildew",
    "Sooty Mould",
]

DEVICE = torch.device("cpu")

IMAGE_TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    ),
])


@lru_cache(maxsize=1)
def load_mango_bundle():
    """Load the Mango model and training-time environmental scalers."""

    if not CHECKPOINT_PATH.is_file():
        raise FileNotFoundError(
            f"Mango checkpoint not found: {CHECKPOINT_PATH}"
        )

    if not TRAIN_CSV.is_file():
        raise FileNotFoundError(
            f"Training CSV not found: {TRAIN_CSV}. "
            "The environmental scalers must be reconstructed "
            "from the original training split or loaded from "
            "saved training-time scaler artifacts."
        )

    # The model architecture was verified against this checkpoint.
    checkpoint = torch.load(
        CHECKPOINT_PATH,
        map_location=DEVICE,
    )

    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
    else:
        state_dict = checkpoint

    model = FullMangoMultimodalModel(num_classes=len(CLASS_NAMES))
    model.load_state_dict(state_dict, strict=True)
    model.to(DEVICE)
    model.eval()

    # Recreate scalers exactly as in the notebook:
    # StandardScaler fitted on the training split only.
    train_df = pd.read_csv(TRAIN_CSV)

    required_columns = WEATHER_FEATURES + SOIL_FEATURES
    missing = [c for c in required_columns if c not in train_df.columns]

    if missing:
        raise ValueError(
            f"Training CSV is missing environmental columns: {missing}"
        )

    weather_scaler = StandardScaler()
    soil_scaler = StandardScaler()

    weather_scaler.fit(train_df[WEATHER_FEATURES])
    soil_scaler.fit(train_df[SOIL_FEATURES])

    return model, weather_scaler, soil_scaler


def compute_mango_gradcam(
    model: FullMangoMultimodalModel,
    image_tensor: torch.Tensor,
    weather_tensor: torch.Tensor,
    soil_tensor: torch.Tensor,
    season_tensor: torch.Tensor,
    original_image: Image.Image,
    target_class_index: int,
) -> str:
    """Compute ViT Grad-CAM attention heatmap for Mango multimodal model and return base64 overlay image."""
    activations = {}
    target_layer = model.vit.encoder.layers[-1].ln_1

    def save_activation(module, inputs, output):
        output.retain_grad()
        activations["features"] = output

    hook = target_layer.register_forward_hook(save_activation)

    try:
        model.zero_grad(set_to_none=True)
        with torch.enable_grad():
            input_for_cam = image_tensor.clone().detach().requires_grad_(True)
            logits = model(
                image=input_for_cam,
                weather=weather_tensor,
                soil=soil_tensor,
                season=season_tensor,
            )
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

            buffer = BytesIO()
            blended_img.save(buffer, format="JPEG", quality=90)
            encoded = base64.b64encode(buffer.getvalue()).decode("utf-8")
            return f"data:image/jpeg;base64,{encoded}"

    except Exception as exc:
        print(f"Mango Grad-CAM error: {exc}")
        return ""
    finally:
        hook.remove()


def predict_mango(
    image_bytes: bytes,
    temperature_c: float,
    humidity_pct: float,
    rainfall_mm: float,
    soil_moisture_pct: float,
    soil_ph: float,
    season: str,
):
    """Predict Mango disease using image and environmental inputs."""

    if not 0 <= humidity_pct <= 100:
        raise ValueError("Humidity must be between 0 and 100 percent.")

    if not 0 <= soil_moisture_pct <= 100:
        raise ValueError("Soil moisture must be between 0 and 100 percent.")

    if not 0 <= soil_ph <= 14:
        raise ValueError("Soil pH must be between 0 and 14.")

    if rainfall_mm < 0:
        raise ValueError("Rainfall cannot be negative.")

    season_key = season.strip().casefold()
    if season_key not in SEASON_TO_INDEX:
        raise ValueError(
            "Season must be Post-Monsoon or Winter."
        )

    model, weather_scaler, soil_scaler = load_mango_bundle()

    try:
        image = Image.open(BytesIO(image_bytes)).convert("RGB")
    except (UnidentifiedImageError, OSError) as exc:
        raise ValueError("The uploaded file is not a valid image.") from exc

    image_tensor = IMAGE_TRANSFORM(image).unsqueeze(0).to(DEVICE)

    weather_raw = [[temperature_c, humidity_pct, rainfall_mm]]
    soil_raw = [[soil_moisture_pct, soil_ph]]

    weather_tensor = torch.tensor(
        weather_scaler.transform(weather_raw),
        dtype=torch.float32,
        device=DEVICE,
    )

    soil_tensor = torch.tensor(
        soil_scaler.transform(soil_raw),
        dtype=torch.float32,
        device=DEVICE,
    )

    season_tensor = torch.tensor(
        [SEASON_TO_INDEX[season_key]],
        dtype=torch.long,
        device=DEVICE,
    )

    with torch.no_grad():
        logits = model(
            image=image_tensor,
            weather=weather_tensor,
            soil=soil_tensor,
            season=season_tensor,
        )

        probabilities = torch.softmax(logits, dim=1)[0].cpu().tolist()

    predicted_index = max(
        range(len(probabilities)),
        key=probabilities.__getitem__,
    )

    # Compute Grad-CAM overlay
    gradcam_image = compute_mango_gradcam(
        model=model,
        image_tensor=image_tensor,
        weather_tensor=weather_tensor,
        soil_tensor=soil_tensor,
        season_tensor=season_tensor,
        original_image=image,
        target_class_index=predicted_index,
    )

    # Compute Environmental KernelSHAP
    shap_data = None
    try:
        from app.services.explainability import compute_mango_shap
        shap_data = compute_mango_shap(
            model=model,
            image_tensor=image_tensor,
            weather_scaler=weather_scaler,
            soil_scaler=soil_scaler,
            season_idx=SEASON_TO_INDEX[season_key],
            raw_env=[temperature_c, humidity_pct, rainfall_mm, soil_moisture_pct, soil_ph],
            target_class_idx=predicted_index,
            device=DEVICE,
            nsamples=35,
        )
    except Exception as exc:
        logger.warning("Mango SHAP explanation failed: %s", exc)

    return {
        "fruit": "mango",
        "model": "full_multimodal",
        "prediction": CLASS_NAMES[predicted_index],
        "predicted_disease": CLASS_NAMES[predicted_index],
        "confidence": round(probabilities[predicted_index], 4),
        "confidence_percent": round(probabilities[predicted_index] * 100, 2),
        "gradcam_image": gradcam_image,
        "shap": shap_data,
        "probabilities": {
            name: round(probability * 100, 2)
            for name, probability in zip(CLASS_NAMES, probabilities)
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

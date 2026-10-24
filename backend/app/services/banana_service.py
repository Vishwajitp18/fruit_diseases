"""Banana disease prediction service using Vision Transformer (ViT) and Grad-CAM explainability."""

import base64
from functools import lru_cache
import io
from pathlib import Path
from typing import Dict, Any, Tuple

import matplotlib.cm as cm
import numpy as np
from PIL import Image, UnidentifiedImageError
import torch
import torch.nn.functional as F
from torchvision import transforms

from app.models.banana_model import UTMModel

PROJECT_ROOT = Path(__file__).resolve().parents[3]
MODEL_DIR = Path(__file__).resolve().parent.parent / "models"
CHECKPOINT_PATH = MODEL_DIR / "best_banana_model.pt"
FALLBACK_CHECKPOINT_PATH = PROJECT_ROOT / "banana_diseases" / "best_disease_model.pt"

DEVICE = torch.device("mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu"))

CLASS_NAMES = [
    "Banana Black Sigatoka Disease",
    "Banana Bract Mosaic Virus Disease",
    "Banana Healthy Leaf",
    "Banana Insect Pest Disease",
    "Banana Moko Disease",
    "Banana Panama Disease",
    "Banana Yellow Sigatoka Disease",
]

IMAGE_TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    ),
])


@lru_cache(maxsize=1)
def load_banana_bundle() -> Tuple[UTMModel, list]:
    """Load the trained Banana UTM model checkpoint."""
    model_path = CHECKPOINT_PATH if CHECKPOINT_PATH.is_file() else FALLBACK_CHECKPOINT_PATH

    model = UTMModel(
        num_classes=len(CLASS_NAMES),
        env_dim=1,
        use_environment=False,
        pretrained=False,
    )

    if model_path.is_file():
        checkpoint = torch.load(model_path, map_location=DEVICE, weights_only=False)
        if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
            state_dict = checkpoint["model_state_dict"]
            class_names = checkpoint.get("class_names", CLASS_NAMES)
        else:
            state_dict = checkpoint
            class_names = CLASS_NAMES
        model.load_state_dict(state_dict, strict=False)
    else:
        class_names = CLASS_NAMES

    model.to(DEVICE)
    model.eval()
    return model, class_names


def compute_gradcam(
    model: UTMModel,
    image_tensor: torch.Tensor,
    original_image: Image.Image,
    target_class_index: int,
) -> str:
    """Compute ViT Grad-CAM attention heatmap and return base64 overlay image."""
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
            outputs = model(input_for_cam)
            logits = outputs["disease_logits"]
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

            # Generate JET colormap
            colormap = cm.get_cmap("jet")
            heatmap_rgba = colormap(cam_norm)
            heatmap_rgb = np.uint8(heatmap_rgba[:, :, :3] * 255)

            # Resize original image to 224x224 and blend
            orig_resized = original_image.resize((224, 224), Image.Resampling.BILINEAR).convert("RGB")
            orig_np = np.array(orig_resized)

            # Blend original leaf image with Grad-CAM heatmap
            blended_np = np.uint8(0.55 * orig_np + 0.45 * heatmap_rgb)
            blended_img = Image.fromarray(blended_np)

            buffer = io.BytesIO()
            blended_img.save(buffer, format="JPEG", quality=90)
            encoded = base64.b64encode(buffer.getvalue()).decode("utf-8")
            return f"data:image/jpeg;base64,{encoded}"

    except Exception as exc:
        print(f"Grad-CAM generation error: {exc}")
        return ""
    finally:
        hook.remove()


def predict_banana(
    image_bytes: bytes,
    temperature_c: float,
    humidity_pct: float,
    rainfall_mm: float,
    soil_moisture_pct: float,
    soil_ph: float,
    season: str = "Monsoon",
) -> Dict[str, Any]:
    """Predict Banana leaf disease and generate Grad-CAM explainability overlay."""
    if not 0 <= humidity_pct <= 100:
        raise ValueError("Humidity must be between 0 and 100 percent.")
    if not 0 <= soil_moisture_pct <= 100:
        raise ValueError("Soil moisture must be between 0 and 100 percent.")
    if not 0 <= soil_ph <= 14:
        raise ValueError("Soil pH must be between 0 and 14.")
    if rainfall_mm < 0:
        raise ValueError("Rainfall cannot be negative.")

    model, class_names = load_banana_bundle()

    try:
        original_image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    except (UnidentifiedImageError, OSError) as exc:
        raise ValueError("The uploaded file is not a valid image.") from exc

    image_tensor = IMAGE_TRANSFORM(original_image).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        outputs = model(image_tensor)
        logits = outputs["disease_logits"]
        severity = outputs.get("severity")
        risk = outputs.get("risk")

        severity_val = float(severity.item()) if severity is not None else None
        risk_val = float(risk.item()) if risk is not None else None

        probabilities = torch.softmax(logits, dim=1)[0].cpu().tolist()

    predicted_index = int(max(range(len(probabilities)), key=probabilities.__getitem__))
    predicted_label = class_names[predicted_index]
    confidence_val = probabilities[predicted_index]

    # Generate Grad-CAM visualization
    gradcam_image_data = compute_gradcam(
        model=model,
        image_tensor=image_tensor,
        original_image=original_image,
        target_class_index=predicted_index,
    )

    # Compute Banana Environmental SHAP
    shap_data = None
    try:
        from app.services.explainability import compute_baseline_shap
        shap_data = compute_baseline_shap(
            raw_env=[temperature_c, humidity_pct, rainfall_mm, soil_moisture_pct, soil_ph],
            target_class_idx=predicted_index,
        )
    except Exception as exc:
        logger.warning("Banana SHAP explanation failed: %s", exc)

    return {
        "fruit": "banana",
        "model": "vit_b_16_utm",
        "prediction": predicted_label,
        "predicted_disease": predicted_label,
        "confidence": round(confidence_val, 4),
        "confidence_percent": round(confidence_val * 100, 2),
        "severity": round(severity_val, 3) if severity_val is not None else None,
        "risk": round(risk_val, 3) if risk_val is not None else None,
        "gradcam_image": gradcam_image_data,
        "shap": shap_data,
        "probabilities": {
            class_names[i]: round(probabilities[i] * 100, 2)
            for i in range(len(class_names))
        },
        "environment": {
            "temperature_c": temperature_c,
            "humidity_pct": humidity_pct,
            "rainfall_mm": rainfall_mm,
            "soil_moisture_pct": soil_moisture_pct,
            "soil_ph": soil_ph,
            "season": season.strip() if season else "Monsoon",
        },
    }


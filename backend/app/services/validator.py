from pathlib import Path
import torch
import torch.nn as nn
from PIL import Image
from torchvision import models, transforms
from fastapi import HTTPException

# This order MUST match the validator's training label order.
VALIDATOR_CLASSES = [
    "apple_leaf",
    "mango_leaf",
    "banana_leaf",
    "other_leaf_or_flower",
    "non_plant",
]

DEVICE = torch.device("mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu"))

MODEL_PATH = Path(__file__).resolve().parent.parent / "models" / "leaf_validator.pth"

validator_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    ),
])

# Threshold for confidence
VALIDATION_THRESHOLD = 0.65

_validator_model = None

def get_validator_model():
    global _validator_model
    if _validator_model is None:
        model = models.mobilenet_v3_small()
        in_features = model.classifier[3].in_features
        model.classifier[3] = nn.Linear(in_features, len(VALIDATOR_CLASSES))
        
        if not MODEL_PATH.exists():
            raise FileNotFoundError(f"Validator model checkpoint not found at: {MODEL_PATH}")
            
        checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)
        model.load_state_dict(checkpoint)
        model.to(DEVICE).eval()
        _validator_model = model
    return _validator_model


def validate_leaf_image(image: Image.Image, selected_fruit: str) -> dict:
    """Reject an unsupported or mismatched image before disease prediction.

    Validates that:
    1. The image is of a recognized plant leaf.
    2. The leaf corresponds to the selected fruit (Apple, Mango, Banana).
    3. The model is sufficiently confident in the leaf identification.
    """
    selected_fruit = selected_fruit.strip().lower()
    if selected_fruit not in {"apple", "mango", "banana"}:
        raise HTTPException(status_code=400, detail="Unsupported fruit selection.")

    model = get_validator_model()
    image = image.convert("RGB")
    image_tensor = validator_transform(image).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        logits = model(image_tensor)
        probabilities = torch.softmax(logits, dim=1)[0]

    confidence, predicted_idx = probabilities.max(dim=0)
    predicted_class = VALIDATOR_CLASSES[int(predicted_idx.item())]
    confidence_val = float(confidence.item())

    # Reject uncertain predictions rather than passing them to the disease model.
    if confidence_val < VALIDATION_THRESHOLD:
        raise HTTPException(
            status_code=422,
            detail={
                "error": "uncertain_image",
                "message": (
                    "Unable to verify this image with confidence. Please upload a clear, "
                    f"close-up photo of a {selected_fruit.capitalize()} leaf."
                ),
            },
        )

    expected_class = f"{selected_fruit}_leaf"

    if predicted_class != expected_class:
        if predicted_class in {"apple_leaf", "mango_leaf", "banana_leaf"}:
            detected_fruit = predicted_class.removesuffix("_leaf").capitalize()
            message = (
                f"This image appears to be a {detected_fruit} leaf, "
                f"not a {selected_fruit.capitalize()} leaf. Please switch to {detected_fruit} "
                f"or upload a {selected_fruit.capitalize()} leaf image."
            )
        elif predicted_class == "other_leaf_or_flower":
            message = (
                f"This image appears to be a flower or an unsupported plant leaf, "
                f"not a {selected_fruit.capitalize()} leaf. Please upload a supported fruit leaf image."
            )
        else:
            message = (
                f"This image does not appear to be a plant leaf (detected: {predicted_class.replace('_', ' ')}). "
                f"Please upload an Apple, Mango, or Banana leaf image."
            )

        raise HTTPException(
            status_code=422,
            detail={
                "error": "invalid_leaf",
                "predicted_type": predicted_class,
                "confidence": round(confidence_val, 4),
                "message": message,
            },
        )

    return {
        "valid": True,
        "fruit": selected_fruit,
        "validator_confidence": round(confidence_val, 4),
    }


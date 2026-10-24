from io import BytesIO
import logging
from collections.abc import Callable

from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from PIL import Image, UnidentifiedImageError
from starlette.concurrency import run_in_threadpool

from app.core.config import settings
from app.schemas.prediction import PredictionResponse
from app.services.apple_service import predict_apple
from app.services.banana_service import predict_banana
from app.services.mango_service import predict_mango
from app.services.report_generator import generate_groq_report
from app.services.treatment_kb import lookup_treatment
from app.services.validator import validate_leaf_image


logger = logging.getLogger(__name__)
router = APIRouter()


async def _predict(
    fruit: str,
    predictor: Callable,
    image: UploadFile,
    temperature_c: float,
    humidity_pct: float,
    rainfall_mm: float,
    soil_moisture_pct: float,
    soil_ph: float,
    season: str,
) -> PredictionResponse:
    if not image.content_type or not image.content_type.startswith("image/"):
        raise HTTPException(
            status_code=415,
            detail="Upload a valid image file.",
        )

    image_bytes = await image.read(settings.max_image_bytes + 1)
    if len(image_bytes) > settings.max_image_bytes:
        raise HTTPException(
            status_code=413,
            detail=f"Image must be {settings.max_image_mb} MB or smaller.",
        )

    try:
        pil_image = Image.open(BytesIO(image_bytes)).convert("RGB")
    except (UnidentifiedImageError, OSError):
        raise HTTPException(
            status_code=422,
            detail={
                "error": "invalid_image_file",
                "message": "The uploaded file is not a readable image.",
            },
        )

    # Validate that image is indeed a leaf of the selected fruit
    await run_in_threadpool(validate_leaf_image, pil_image, fruit)

    try:
        result = await run_in_threadpool(
            predictor,
            image_bytes,
            temperature_c,
            humidity_pct,
            rainfall_mm,
            soil_moisture_pct,
            soil_ph,
            season,
        )

        predicted_disease = result.get("predicted_disease") or result.get("prediction", "Unknown")
        treatment = lookup_treatment(fruit, predicted_disease)

        has_cam = bool(result.get("gradcam_image"))
        shap_info = result.get("shap")
        has_shap = bool(shap_info)

        methods = []
        if has_cam:
            methods.append("Grad-CAM")
        if has_shap:
            methods.append("KernelSHAP")
        method_str = " + ".join(methods) if methods else "Feature Attribution"

        expl_summary_parts = []
        if has_shap and shap_info.get("summary"):
            expl_summary_parts.append(shap_info["summary"])
        if has_cam:
            expl_summary_parts.append(
                f"Visual attention (Grad-CAM) highlights the primary lesion and morphological symptom "
                f"patterns on the leaf surface that strongly influenced the model's diagnosis of {predicted_disease}."
            )
        else:
            expl_summary_parts.append("Visual attention map unavailable.")

        explanation = {
            "method": method_str,
            "heatmap_url": result.get("gradcam_image") if has_cam else None,
            "summary": " ".join(expl_summary_parts),
            "shap": shap_info,
        }

        structured_report_input = {
            "fruit": fruit,
            "prediction": {
                "disease": predicted_disease,
                "confidence_percent": result.get("confidence_percent", 0.0),
            },
            "environment": result.get("environment", {}),
            "explanation": explanation,
            "management": treatment,
        }

        report = await run_in_threadpool(generate_groq_report, structured_report_input)

        result["status"] = "success"
        result["explanation"] = explanation
        result["management"] = treatment
        result["report"] = report

        return PredictionResponse(**result)
    except HTTPException:
        raise
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("%s prediction failed", fruit)
        raise HTTPException(
            status_code=500,
            detail=f"{fruit.title()} prediction failed. Check the backend terminal.",
        ) from exc


@router.post("/mango", response_model=PredictionResponse)
async def predict_mango_endpoint(
    image: UploadFile = File(...),
    temperature_c: float = Form(...),
    humidity_pct: float = Form(...),
    rainfall_mm: float = Form(...),
    soil_moisture_pct: float = Form(...),
    soil_ph: float = Form(...),
    season: str = Form(...),
) -> PredictionResponse:
    return await _predict(
        "mango",
        predict_mango,
        image,
        temperature_c,
        humidity_pct,
        rainfall_mm,
        soil_moisture_pct,
        soil_ph,
        season,
    )


@router.post("/apple", response_model=PredictionResponse)
async def predict_apple_endpoint(
    image: UploadFile = File(...),
    temperature_c: float = Form(...),
    humidity_pct: float = Form(...),
    rainfall_mm: float = Form(...),
    soil_moisture_pct: float = Form(...),
    soil_ph: float = Form(...),
    season: str = Form("spring"),
) -> PredictionResponse:
    return await _predict(
        "apple",
        predict_apple,
        image,
        temperature_c,
        humidity_pct,
        rainfall_mm,
        soil_moisture_pct,
        soil_ph,
        season,
    )


@router.post("/banana", response_model=PredictionResponse)
async def predict_banana_endpoint(
    image: UploadFile = File(...),
    temperature_c: float = Form(...),
    humidity_pct: float = Form(...),
    rainfall_mm: float = Form(...),
    soil_moisture_pct: float = Form(...),
    soil_ph: float = Form(...),
    season: str = Form("Monsoon"),
) -> PredictionResponse:
    return await _predict(
        "banana",
        predict_banana,
        image,
        temperature_c,
        humidity_pct,
        rainfall_mm,
        soil_moisture_pct,
        soil_ph,
        season,
    )


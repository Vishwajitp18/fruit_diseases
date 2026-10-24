from fastapi import APIRouter

router = APIRouter()


@router.get("/")
def root():
    return {
        "message": "Fruit Disease Prediction API",
        "docs": "/docs",
    }


@router.get("/health")
def health_check():
    return {
        "status": "ok",
        "service": "fruit-disease-prediction-api",
    }

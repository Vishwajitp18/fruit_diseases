from typing import Any, Dict, List, Optional, Union
from pydantic import BaseModel


class PredictionDetail(BaseModel):
    disease: str
    confidence_percent: float


class ExplanationDetail(BaseModel):
    method: str
    heatmap_url: Optional[str] = None
    summary: str
    shap: Optional[Dict[str, Any]] = None


class ManagementSource(BaseModel):
    title: str
    organization: str
    url: str


class ManagementDetail(BaseModel):
    display_name: Optional[str] = None
    symptoms_to_check: List[str] = []
    immediate_actions: List[str] = []
    preventive_measures: List[str] = []
    treatment_options: List[str] = []
    when_to_seek_expert_help: Optional[str] = None
    sources: List[ManagementSource] = []
    reviewed_by: Optional[str] = None
    last_reviewed: Optional[str] = None


class ReportDetail(BaseModel):
    generated_by: str
    text: str


class PredictionResponse(BaseModel):
    status: str = "success"
    fruit: str
    model: str
    
    # Flat fields for direct frontend and backward compatibility
    predicted_disease: str
    prediction: Union[str, PredictionDetail]
    confidence: Optional[float] = None
    confidence_percent: float
    severity: Optional[float] = None
    risk: Optional[float] = None
    early_stage: Optional[bool] = None
    gradcam_image: Optional[str] = None
    probabilities: Dict[str, float]
    environment: Dict[str, Union[float, str]]

    # Explainable AI & Treatment Guidance additions
    explanation: Optional[ExplanationDetail] = None
    management: Optional[ManagementDetail] = None
    report: Optional[ReportDetail] = None
    disclaimer: str = (
        "AI-assisted screening is not a confirmed diagnosis. Verify treatment decisions "
        "with a qualified local agricultural professional and applicable product labels."
    )

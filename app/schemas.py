from __future__ import annotations
from typing import Literal, Optional
from pydantic import BaseModel, Field

class SensorReading(BaseModel):
    N: float = Field(..., ge=0, le=300, description='Nitrogen (kg/ha)')
    P: float = Field(..., ge=0, le=300, description='Phosphorus (kg/ha)')
    K: float = Field(..., ge=0, le=300, description='Potassium (kg/ha)')
    temperature: float = Field(..., ge=-20, le=60, description='Average air temperature (°C)')
    humidity: float = Field(..., ge=0, le=100, description='Relative humidity (%)')
    ph: float = Field(..., ge=0, le=14, description='Soil pH')
    rainfall: float = Field(..., ge=0, le=1000, description='Accumulated rainfall (mm)')

class LaxSensorReading(BaseModel):
    N: float = Field(..., description='Nitrogen (kg/ha)')
    P: float = Field(..., description='Phosphorus (kg/ha)')
    K: float = Field(..., description='Potassium (kg/ha)')
    temperature: float = Field(..., description='Average air temperature (°C)')
    humidity: float = Field(..., description='Relative humidity (%)')
    ph: float = Field(..., description='Soil pH')
    rainfall: float = Field(..., description='Accumulated rainfall (mm)')

class PredictRequest(BaseModel):
    records: list[SensorReading] = Field(..., min_length=1, max_length=500, description='List of sensor readings (1–500 fields per request)')

class PredictRequestLax(BaseModel):
    records: list[LaxSensorReading] = Field(..., min_length=1, max_length=500, description='List of sensor readings (permissive mode)')

class Alternative(BaseModel):
    crop: str
    probability: float

class Explanation(BaseModel):
    feature: str
    value: float
    impact: Literal['positive', 'negative']
FeatureExplanation = Explanation

class Prediction(BaseModel):
    crop: str
    confidence: float
    alternatives: list[Alternative]
    explanation: Optional[list[Explanation]] = None
PredictionResult = Prediction

class PredictResponse(BaseModel):
    predictions: list[Prediction]

class HealthResponse(BaseModel):
    status: Literal['ok', 'degraded']
    model_loaded: bool

class ModelInfoResponse(BaseModel):
    artifact: Optional[str] = None
    loaded_at: Optional[str] = None
    serialized_at: Optional[str] = None
    sklearn_version_train: Optional[str] = None
    sklearn_version_runtime: str
    features: list[str]
    n_classes: int
    available_artifacts: list[str] = []

class RollbackRequest(BaseModel):
    artifact: str = Field(..., description="Name of the .pkl file in artifacts/ (e.g.: 'model_v1.pkl')", pattern='^[\\w\\-]+\\.pkl$')

class RollbackResponse(BaseModel):
    status: Literal['ok']
    active_model: ModelInfoResponse

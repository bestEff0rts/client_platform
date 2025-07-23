#Pydantic модели для API-для эндпоинта predict
from pydantic import BaseModel
from typing import List, Optional

class PredictRequest(BaseModel):
    features: List[float]

class PredictResponse(BaseModel):
    prediction: float
    confidence: Optional[float]
class MetricsResponse(BaseModel):
    mse: float
    r2: float
    sharpe: float
    sortino: float

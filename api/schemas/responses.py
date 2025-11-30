from pydantic import BaseModel, Field

class HealthResponse(BaseModel):
    status: str
    encrypted: bool
    model_loaded: bool
    timestamp: float
    
class ContextResponse(BaseModel):
    context_hex: str
    security_level: str
    poly_modulus_degree: int
    has_secret_key: bool
    message: str

class ModelInfoResponse(BaseModel):
    model_type: str
    input_features: int
    output_type: str
    encryption_scheme: str
    description: str
    privacy_guarantee: str

class PredictionResponse(BaseModel):
    encrypted_prediction_hex: str
    inference_time_ms: float
    model_type: str
    message: str
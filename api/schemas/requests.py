from pydantic import BaseModel, Field

class PredictionRequest(BaseModel):
    """
    Request for encrypted prediction
    
    Client sends encrypted patient features as hex string
    """
    encrypted_features_hex: str = Field(
        ...,
        description="Hex-encoded encrypted patient features (13 values)",
        example="0a010d1293b4145ea110040102..."
    )
    
    class Config:
        json_schema_extra = {
            "example": {
                "encrypted_features_hex": "0a010d1293b4145ea1100401020000131a05000000000028b52f..."
            }
        }
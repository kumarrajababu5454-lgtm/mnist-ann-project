from pydantic import BaseModel, Field


class PredictionResponse(BaseModel):
    predicted_digit: int = Field(ge=0, le=9)
    confidence: float = Field(ge=0, le=1)
    probabilities: dict[str, float]

from pydantic import BaseModel, Field


class SentimentRequest(BaseModel):
    text: str = Field(..., min_length=5, example="The company reported record profits this quarter.")


class SentimentResponse(BaseModel):
    text: str
    label: str                  # "negative" | "neutral" | "positive"
    label_id: int               # 0 | 1 | 2
    confidence: float           # probability of the predicted label (softmax)


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    model_path: str

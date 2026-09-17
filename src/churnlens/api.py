from fastapi import FastAPI
from pydantic import BaseModel, Field

from .service import ChurnService


class TelecomSubscriber(BaseModel):
    tenure_months: int = Field(ge=0, le=120)
    monthly_revenue: float = Field(ge=0, le=500)
    data_usage_gb_30d: float = Field(ge=0, le=500)
    voice_minutes_30d: float = Field(ge=0, le=5000)
    recharge_count_30d: int = Field(ge=0, le=100)
    recharge_amount_30d: float = Field(ge=0, le=1000)
    complaint_count_90d: int = Field(ge=0, le=50)
    payment_failures_90d: int = Field(ge=0, le=30)
    last_recharge_days: int = Field(ge=0, le=365)
    contract_type: str
    region: str
    auto_pay: int = Field(ge=0, le=1)


app = FastAPI(title="ChurnLens API", version="0.1.0", description="Explainable customer churn scoring API")
service = ChurnService()


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "service": "churnlens"}


@app.post("/v1/score")
def score(customer: TelecomSubscriber) -> dict:
    return service.score(customer.model_dump())
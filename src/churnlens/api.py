from fastapi import FastAPI
from pydantic import BaseModel, Field

from .service import ChurnService


class Customer(BaseModel):
    tenure_months: int = Field(ge=0, le=120)
    monthly_charges: float = Field(ge=0, le=500)
    support_tickets_90d: int = Field(ge=0, le=50)
    late_payments_12m: int = Field(ge=0, le=12)
    contract_type: str
    internet_service: str
    payment_method: str
    has_streaming: int = Field(ge=0, le=1)
    auto_pay: int = Field(ge=0, le=1)


app = FastAPI(title="ChurnLens API", version="0.1.0", description="Explainable customer churn scoring API")
service = ChurnService()


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "service": "churnlens"}


@app.post("/v1/score")
def score(customer: Customer) -> dict:
    return service.score(customer.model_dump())
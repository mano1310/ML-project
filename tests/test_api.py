from fastapi.testclient import TestClient

from churnlens.api import app


def test_health_endpoint():
    response = TestClient(app).get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_score_endpoint_returns_explanation():
    customer = {
        "tenure_months": 4, "monthly_charges": 120, "support_tickets_90d": 3,
        "late_payments_12m": 2, "contract_type": "month-to-month",
        "internet_service": "fiber", "payment_method": "electronic-check",
        "has_streaming": 1, "auto_pay": 0,
    }
    response = TestClient(app).post("/v1/score", json=customer)
    assert response.status_code == 200
    assert "drivers" in response.json()
    assert "recommendation" in response.json()
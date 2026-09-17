from fastapi.testclient import TestClient

from churnlens.api import app


def test_health_endpoint():
    response = TestClient(app).get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_score_endpoint_returns_explanation():
    customer = {
        "tenure_months": 4, "monthly_revenue": 120, "data_usage_gb_30d": 8,
        "voice_minutes_30d": 120, "recharge_count_30d": 1, "recharge_amount_30d": 35,
        "complaint_count_90d": 3, "payment_failures_90d": 2, "last_recharge_days": 25,
        "contract_type": "month-to-month", "region": "west", "auto_pay": 0,
    }
    response = TestClient(app).post("/v1/score", json=customer)
    assert response.status_code == 200
    assert "drivers" in response.json()
    assert "recommendation" in response.json()
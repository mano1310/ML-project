from __future__ import annotations

import numpy as np
import pandas as pd

FEATURES = [
    "tenure_months",
    "monthly_charges",
    "support_tickets_90d",
    "late_payments_12m",
    "contract_type",
    "internet_service",
    "payment_method",
    "has_streaming",
    "auto_pay",
]


def make_synthetic_customers(n_rows: int = 5000, random_state: int = 42) -> pd.DataFrame:
    """Create a realistic, deterministic demo dataset with a known churn signal."""
    rng = np.random.default_rng(random_state)
    tenure = rng.integers(1, 73, n_rows)
    charges = np.clip(rng.normal(72, 25, n_rows), 20, 160).round(2)
    tickets = rng.poisson(1.2, n_rows)
    late = rng.binomial(3, 0.18, n_rows)
    contract = rng.choice(["month-to-month", "one-year", "two-year"], n_rows, p=[.55, .25, .20])
    internet = rng.choice(["fiber", "dsl", "none"], n_rows, p=[.48, .37, .15])
    payment = rng.choice(["electronic-check", "bank-transfer", "credit-card", "mailed-check"], n_rows)
    streaming = rng.binomial(1, .52, n_rows)
    auto_pay = rng.binomial(1, .58, n_rows)

    logit = (
        0.55
        - 0.035 * tenure
        + 0.018 * charges
        + 0.22 * tickets
        + 0.42 * late
        + 0.95 * (contract == "month-to-month")
        + 0.30 * (internet == "fiber")
        + 0.25 * (payment == "electronic-check")
        + 0.25 * streaming
        - 0.65 * auto_pay
    )
    probability = 1 / (1 + np.exp(-logit))
    churned = rng.binomial(1, probability)
    return pd.DataFrame({
        "tenure_months": tenure,
        "monthly_charges": charges,
        "support_tickets_90d": tickets,
        "late_payments_12m": late,
        "contract_type": contract,
        "internet_service": internet,
        "payment_method": payment,
        "has_streaming": streaming,
        "auto_pay": auto_pay,
        "churned": churned,
    })
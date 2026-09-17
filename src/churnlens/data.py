from __future__ import annotations

import numpy as np
import pandas as pd

FEATURES = [
    "tenure_months",
    "monthly_revenue",
    "data_usage_gb_30d",
    "voice_minutes_30d",
    "recharge_count_30d",
    "recharge_amount_30d",
    "complaint_count_90d",
    "payment_failures_90d",
    "last_recharge_days",
    "contract_type",
    "region",
    "auto_pay",
]


def make_synthetic_telecom_data(n_rows: int = 5000, random_state: int = 42) -> pd.DataFrame:
    """Create joined subscriber, usage, recharge, and complaints data for a demo cohort."""
    rng = np.random.default_rng(random_state)
    tenure = rng.integers(1, 73, n_rows)
    revenue = np.clip(rng.normal(72, 25, n_rows), 20, 160).round(2)
    data_usage = np.clip(rng.gamma(4, 4, n_rows), .2, 80).round(2)
    voice_minutes = np.clip(rng.gamma(3, 55, n_rows), 0, 900).round(0)
    recharge_count = rng.poisson(2.5, n_rows)
    recharge_amount = np.clip(rng.normal(45, 18, n_rows), 5, 150).round(2)
    complaints = rng.poisson(.8, n_rows)
    payment_failures = rng.binomial(4, .12, n_rows)
    last_recharge_days = rng.integers(0, 61, n_rows)
    contract = rng.choice(["month-to-month", "one-year", "two-year"], n_rows, p=[.55, .25, .20])
    region = rng.choice(["north", "south", "east", "west"], n_rows)
    auto_pay = rng.binomial(1, .58, n_rows)

    logit = (
        -0.30
        - 0.035 * tenure
        + 0.012 * revenue
        - 0.035 * data_usage
        - 0.0008 * voice_minutes
        - 0.18 * recharge_count
        - 0.012 * recharge_amount
        + 0.38 * complaints
        + 0.45 * payment_failures
        + 0.035 * last_recharge_days
        + 0.95 * (contract == "month-to-month")
        - 0.65 * auto_pay
    )
    probability = 1 / (1 + np.exp(-logit))
    churn_label = rng.binomial(1, probability)
    return pd.DataFrame({
        "subscriber_id": [f"SUB-{index:06d}" for index in range(n_rows)],
        "tenure_months": tenure,
        "monthly_revenue": revenue,
        "data_usage_gb_30d": data_usage,
        "voice_minutes_30d": voice_minutes,
        "recharge_count_30d": recharge_count,
        "recharge_amount_30d": recharge_amount,
        "complaint_count_90d": complaints,
        "payment_failures_90d": payment_failures,
        "last_recharge_days": last_recharge_days,
        "contract_type": contract,
        "region": region,
        "auto_pay": auto_pay,
        "churn_label": churn_label,
    })


def make_synthetic_customers(n_rows: int = 5000, random_state: int = 42) -> pd.DataFrame:
    """Backward-compatible alias for callers using the original demo function."""
    return make_synthetic_telecom_data(n_rows, random_state)


def profile_telecom_data(dataset: pd.DataFrame) -> dict:
    """Return a compact EDA profile for data quality and label review."""
    return {
        "rows": len(dataset),
        "columns": len(dataset.columns),
        "churn_rate": round(float(dataset["churn_label"].mean()), 4),
        "missing_values": int(dataset[FEATURES].isna().sum().sum()),
        "numeric_summary": dataset[FEATURES].select_dtypes(include="number").describe().round(2).to_dict(),
        "categorical_cardinality": {column: int(dataset[column].nunique()) for column in ["contract_type", "region"]},
    }
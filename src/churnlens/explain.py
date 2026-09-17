from __future__ import annotations

import numpy as np
import pandas as pd


def explain_prediction(model, customer: dict, baseline: pd.DataFrame) -> dict:
    row = pd.DataFrame([customer])
    probability = float(model.predict_proba(row)[:, 1][0])
    transformed = model.named_steps["preprocess"].transform(row)
    baseline_transformed = model.named_steps["preprocess"].transform(baseline)
    coefficients = model.named_steps["model"].coef_[0]
    baseline_vector = np.asarray(baseline_transformed.mean(axis=0)).ravel()
    row_vector = np.asarray(transformed.toarray() if hasattr(transformed, "toarray") else transformed).ravel()
    contribution = (row_vector - baseline_vector) * coefficients
    names = model.named_steps["preprocess"].get_feature_names_out()
    ranked = sorted(zip(names, contribution), key=lambda item: abs(item[1]), reverse=True)[:8]
    return {
        "churn_probability": round(probability, 4),
        "risk_band": "high" if probability >= .7 else "medium" if probability >= .45 else "low",
        "drivers": [{"feature": name, "impact": round(float(value), 4), "direction": "increases" if value > 0 else "reduces"} for name, value in ranked],
    }
from __future__ import annotations

from pathlib import Path

import joblib

from .config import settings
from .data import make_synthetic_customers
from .explain import explain_prediction
from .train import train_model


class ChurnService:
    def __init__(self, artifact_dir: Path | None = None):
        self.artifact_dir = artifact_dir or settings.artifact_dir
        model_path = self.artifact_dir / "churn_model.joblib"
        if not model_path.exists():
            train_model(self.artifact_dir)
        self.model = joblib.load(model_path)
        self.baseline = make_synthetic_customers(1000, settings.random_state).drop(columns="churned")

    def score(self, customer: dict) -> dict:
        result = explain_prediction(self.model, customer, self.baseline)
        probability = result["churn_probability"]
        result["recommendation"] = (
            "Prioritize a personal retention call and contract review."
            if probability >= .7 else
            "Offer targeted support and an auto-pay or plan-value nudge."
            if probability >= .45 else
            "Continue regular engagement; no urgent intervention is indicated."
        )
        return result
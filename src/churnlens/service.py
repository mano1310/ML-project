from __future__ import annotations

import json
from pathlib import Path

import joblib

from .config import settings
from .data import FEATURES, make_synthetic_telecom_data
from .explain import explain_prediction
from .train import train_model


class ChurnService:
    def __init__(self, artifact_dir: Path | None = None):
        self.artifact_dir = artifact_dir or settings.artifact_dir
        model_path = self.artifact_dir / "churn_model.joblib"
        metrics_path = self.artifact_dir / "metrics.json"
        artifact_features = []
        if metrics_path.exists():
            artifact_features = json.loads(metrics_path.read_text(encoding="utf-8")).get("features", [])
        if not model_path.exists() or artifact_features != FEATURES:
            train_model(self.artifact_dir)
        self.model = joblib.load(model_path)
        self.baseline = make_synthetic_telecom_data(1000, settings.random_state).drop(columns=["subscriber_id", "churn_label"])

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
from __future__ import annotations

import json
from pathlib import Path

import joblib
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .config import settings
from .data import FEATURES, make_synthetic_customers


def build_pipeline() -> Pipeline:
    numeric = ["tenure_months", "monthly_charges", "support_tickets_90d", "late_payments_12m", "has_streaming", "auto_pay"]
    categorical = ["contract_type", "internet_service", "payment_method"]
    preprocess = ColumnTransformer([
        ("numeric", Pipeline([("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())]), numeric),
        ("categorical", Pipeline([("impute", SimpleImputer(strategy="most_frequent")), ("onehot", OneHotEncoder(handle_unknown="ignore"))]), categorical),
    ])
    return Pipeline([("preprocess", preprocess), ("model", LogisticRegression(max_iter=1000, class_weight="balanced", random_state=settings.random_state))])


def train_model(output_dir: Path | None = None) -> dict:
    output_dir = output_dir or settings.artifact_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    dataset = make_synthetic_customers(random_state=settings.random_state)
    x_train, x_test, y_train, y_test = train_test_split(dataset[FEATURES], dataset["churned"], test_size=.2, stratify=dataset["churned"], random_state=settings.random_state)
    pipeline = build_pipeline()
    pipeline.fit(x_train, y_train)
    probabilities = pipeline.predict_proba(x_test)[:, 1]
    metrics = {
        "roc_auc": round(float(roc_auc_score(y_test, probabilities)), 4),
        "average_precision": round(float(average_precision_score(y_test, probabilities)), 4),
        "brier_score": round(float(brier_score_loss(y_test, probabilities)), 4),
        "n_train": len(x_train),
        "n_test": len(x_test),
        "features": FEATURES,
        "threshold": settings.churn_threshold,
    }
    joblib.dump(pipeline, output_dir / "churn_model.joblib")
    (output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    return metrics


def main() -> None:
    metrics = train_model()
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
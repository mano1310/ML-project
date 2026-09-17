from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .config import settings
from .data import FEATURES, make_synthetic_telecom_data, profile_telecom_data


def build_pipeline() -> Pipeline:
    numeric = [feature for feature in FEATURES if feature not in {"contract_type", "region"}]
    categorical = ["contract_type", "region"]
    preprocess = ColumnTransformer([
        ("numeric", Pipeline([("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())]), numeric),
        ("categorical", Pipeline([("impute", SimpleImputer(strategy="most_frequent")), ("onehot", OneHotEncoder(handle_unknown="ignore"))]), categorical),
    ])
    return Pipeline([("preprocess", preprocess), ("model", LogisticRegression(max_iter=1000, class_weight="balanced", random_state=settings.random_state))])


def ks_statistic(y_true, probabilities) -> float:
    """Calculate Kolmogorov-Smirnov separation between churn and non-churn."""
    order = probabilities.argsort()[::-1]
    target = y_true.to_numpy()[order]
    positives = target.sum()
    negatives = len(target) - positives
    return float(np.max(np.cumsum(target) / positives - np.cumsum(1 - target) / negatives))


def train_model(output_dir: Path | None = None) -> dict:
    output_dir = output_dir or settings.artifact_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    dataset = make_synthetic_telecom_data(random_state=settings.random_state)
    x_train, x_test, y_train, y_test = train_test_split(dataset[FEATURES], dataset["churn_label"], test_size=.2, stratify=dataset["churn_label"], random_state=settings.random_state)
    candidates = {
        "logistic_regression": (build_pipeline(), {"model__C": [.1, 1.0, 3.0]}),
        "random_forest": (build_pipeline().set_params(model=RandomForestClassifier(class_weight="balanced", random_state=settings.random_state, n_jobs=1)), {"model__n_estimators": [150], "model__max_depth": [5, 10]}),
    }
    fitted = {}
    comparison = {}
    for name, (candidate, parameters) in candidates.items():
        search = GridSearchCV(candidate, parameters, scoring="roc_auc", cv=3, n_jobs=1)
        search.fit(x_train, y_train)
        fitted[name] = search
        comparison[name] = {"roc_auc_cv": round(float(search.best_score_), 4), "best_params": search.best_params_}
    best_name = max(fitted, key=lambda name: fitted[name].best_score_)
    pipeline = fitted[best_name].best_estimator_
    probabilities = pipeline.predict_proba(x_test)[:, 1]
    metrics = {
        "roc_auc": round(float(roc_auc_score(y_test, probabilities)), 4),
        "ks": round(ks_statistic(y_test, probabilities), 4),
        "average_precision": round(float(average_precision_score(y_test, probabilities)), 4),
        "brier_score": round(float(brier_score_loss(y_test, probabilities)), 4),
        "n_train": len(x_train),
        "n_test": len(x_test),
        "churn_rate": round(float(dataset["churn_label"].mean()), 4),
        "eda_profile": profile_telecom_data(dataset),
        "features": FEATURES,
        "threshold": settings.churn_threshold,
        "selected_model": best_name,
        "model_comparison": comparison,
        "label_definition": "No activity or recharge for 30 consecutive days after the observation window.",
    }
    joblib.dump(pipeline, output_dir / "churn_model.joblib")
    (output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    return metrics


def main() -> None:
    metrics = train_model()
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
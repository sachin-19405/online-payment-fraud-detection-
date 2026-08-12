"""
train.py
--------
Trains the Online Payment Fraud Detection model and saves a single
deployable artifact (models/fraud_model.joblib) containing the full
sklearn Pipeline (preprocessing + classifier).

Usage:
    python -m src.train --data data/train_sample.csv
    python -m src.train --data /path/to/full_paysim.csv --model-out models/fraud_model.joblib

The pipeline handles categorical encoding and scaling internally, so
the same pipeline object can be reused directly for inference without
any external state.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.preprocess import (
    CATEGORICAL_COLUMNS,
    FEATURE_COLUMNS,
    load_dataset,
)

NUMERIC_COLUMNS = [c for c in FEATURE_COLUMNS if c not in CATEGORICAL_COLUMNS]


def build_pipeline(n_estimators: int = 200, max_depth: int | None = 16) -> Pipeline:
    """Build the full preprocessing + model pipeline."""
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), NUMERIC_COLUMNS),
            ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_COLUMNS),
        ]
    )

    clf = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_leaf=2,
        class_weight="balanced_subsample",
        n_jobs=-1,
        random_state=42,
    )

    pipeline = Pipeline(steps=[("preprocessor", preprocessor), ("model", clf)])
    return pipeline


def train(data_path: str, model_out: str, report_out: str, test_size: float = 0.2) -> None:
    print(f"Loading dataset from {data_path} ...")
    X, y = load_dataset(data_path)
    print(f"Loaded {len(X):,} rows. Fraud rate: {y.mean():.4%}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42, stratify=y
    )

    pipeline = build_pipeline()

    print("Training RandomForestClassifier ...")
    start = time.time()
    pipeline.fit(X_train, y_train)
    print(f"Training completed in {time.time() - start:.1f}s")

    print("Evaluating on held-out test set ...")
    y_pred = pipeline.predict(X_test)
    y_proba = pipeline.predict_proba(X_test)[:, 1]

    report = classification_report(y_test, y_pred, digits=4, output_dict=True)
    report_text = classification_report(y_test, y_pred, digits=4)
    roc_auc = roc_auc_score(y_test, y_proba)
    pr_auc = average_precision_score(y_test, y_proba)
    cm = confusion_matrix(y_test, y_pred).tolist()

    print(report_text)
    print(f"ROC-AUC: {roc_auc:.4f}")
    print(f"PR-AUC (Average Precision): {pr_auc:.4f}")
    print(f"Confusion matrix: {cm}")

    Path(model_out).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, model_out)
    print(f"Saved trained pipeline to {model_out}")

    metrics = {
        "n_rows": int(len(X)),
        "fraud_rate": float(y.mean()),
        "roc_auc": roc_auc,
        "pr_auc": pr_auc,
        "confusion_matrix": cm,
        "classification_report": report,
    }
    Path(report_out).parent.mkdir(parents=True, exist_ok=True)
    with open(report_out, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"Saved metrics to {report_out}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the fraud detection model.")
    parser.add_argument(
        "--data",
        default="data/train_sample.csv",
        help="Path to training CSV (PaySim schema). Defaults to the bundled sample.",
    )
    parser.add_argument(
        "--model-out",
        default="models/fraud_model.joblib",
        help="Where to save the trained pipeline.",
    )
    parser.add_argument(
        "--report-out",
        default="models/metrics.json",
        help="Where to save evaluation metrics.",
    )
    parser.add_argument("--test-size", type=float, default=0.2)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train(args.data, args.model_out, args.report_out, args.test_size)

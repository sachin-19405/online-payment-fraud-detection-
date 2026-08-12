"""
predict.py
----------
Thin inference wrapper around the trained pipeline artifact. Used by
both the Flask web app and any batch/CLI scoring job.
"""

from __future__ import annotations

from functools import lru_cache

import joblib
import pandas as pd

from src.preprocess import single_transaction_to_frame

DEFAULT_MODEL_PATH = "models/fraud_model.joblib"


@lru_cache(maxsize=1)
def get_model(model_path: str = DEFAULT_MODEL_PATH):
    """Load (and cache) the trained pipeline from disk."""
    return joblib.load(model_path)


def predict_transaction(record: dict, model_path: str = DEFAULT_MODEL_PATH) -> dict:
    """Score a single transaction dict and return a result dict with
    the fraud probability, boolean prediction, and a plain-English
    risk label."""
    model = get_model(model_path)
    X = single_transaction_to_frame(record)
    proba = float(model.predict_proba(X)[0, 1])
    is_fraud = bool(proba >= 0.5)

    if proba >= 0.75:
        risk = "High"
    elif proba >= 0.25:
        risk = "Medium"
    else:
        risk = "Low"

    return {
        "isFraud": is_fraud,
        "fraudProbability": round(proba, 6),
        "riskLevel": risk,
    }


def predict_batch(df: pd.DataFrame, model_path: str = DEFAULT_MODEL_PATH) -> pd.DataFrame:
    """Score a batch of raw transactions (same schema as the training
    CSV) and return the dataframe with prediction columns appended."""
    from src.preprocess import build_feature_frame

    model = get_model(model_path)
    X = build_feature_frame(df)
    proba = model.predict_proba(X)[:, 1]
    out = df.copy()
    out["fraudProbability"] = proba
    out["predictedFraud"] = (proba >= 0.5).astype(int)
    return out

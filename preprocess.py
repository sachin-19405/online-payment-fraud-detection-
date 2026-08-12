"""
preprocess.py
--------------
Feature engineering and preprocessing utilities for the Online Payment
Fraud Detection project.

The raw dataset (based on the PaySim mobile-money simulation schema) has
these columns:

    step, type, amount, nameOrig, oldbalanceOrg, newbalanceOrig,
    nameDest, oldbalanceDest, newbalanceDest, isFraud

This module turns those raw columns into a clean numeric feature matrix
that a scikit-learn model can consume, and exposes the SAME transform
for both training and real-time inference so there is no train/serve
skew.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# Columns the model is trained on, in this exact order.
FEATURE_COLUMNS = [
    "step",
    "type",
    "amount",
    "oldbalanceOrg",
    "newbalanceOrig",
    "oldbalanceDest",
    "newbalanceDest",
    "errorBalanceOrig",
    "errorBalanceDest",
    "hourOfDay",
    "isMerchantDest",
]

CATEGORICAL_COLUMNS = ["type"]
TRANSACTION_TYPES = ["CASH_IN", "CASH_OUT", "DEBIT", "PAYMENT", "TRANSFER"]


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add engineered features to a raw transactions dataframe.

    Adds:
      - errorBalanceOrig: mismatch between expected and actual sender
        balance after the transaction (a strong fraud signal in PaySim
        because fraudulent TRANSFER/CASH_OUT pairs tend to zero out
        balances in a way that doesn't reconcile).
      - errorBalanceDest: same idea for the receiving account.
      - hourOfDay: the `step` field represents 1 simulated hour each;
        step % 24 recovers a cyclical hour-of-day feature.
      - isMerchantDest: PaySim merchant accounts start with 'M' and
        never appear as fraud destinations, so this is informative.
    """
    df = df.copy()

    df["errorBalanceOrig"] = (
        df["newbalanceOrig"] + df["amount"] - df["oldbalanceOrg"]
    )
    df["errorBalanceDest"] = (
        df["oldbalanceDest"] + df["amount"] - df["newbalanceDest"]
    )
    df["hourOfDay"] = df["step"].astype(float) % 24

    if "nameDest" in df.columns:
        df["isMerchantDest"] = df["nameDest"].astype(str).str.startswith("M").astype(int)
    else:
        df["isMerchantDest"] = 0

    return df


def build_feature_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Run feature engineering and return the model-ready feature frame
    (still with the raw categorical `type` column - encoding happens
    inside the sklearn Pipeline's ColumnTransformer, not here)."""
    df = engineer_features(df)
    missing = [c for c in FEATURE_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing expected columns after engineering: {missing}")
    return df[FEATURE_COLUMNS]


def load_dataset(csv_path: str) -> tuple[pd.DataFrame, pd.Series]:
    """Load a raw PaySim-style CSV and return (X, y)."""
    df = pd.read_csv(csv_path)
    y = df["isFraud"].astype(int)
    X = build_feature_frame(df)
    return X, y


def single_transaction_to_frame(record: dict) -> pd.DataFrame:
    """Convert a single transaction (e.g. from a web form / JSON API
    request) into the one-row dataframe the pipeline expects.

    `record` should contain: step, type, amount, oldbalanceOrg,
    newbalanceOrig, oldbalanceDest, newbalanceDest, nameDest (optional).
    """
    defaults = {
        "step": 1,
        "type": "PAYMENT",
        "amount": 0.0,
        "oldbalanceOrg": 0.0,
        "newbalanceOrig": 0.0,
        "oldbalanceDest": 0.0,
        "newbalanceDest": 0.0,
        "nameDest": "C0000000000",
    }
    defaults.update(record)
    df = pd.DataFrame([defaults])
    return build_feature_frame(df)

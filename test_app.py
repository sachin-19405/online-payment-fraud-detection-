"""
Basic smoke tests for the fraud detection app.

Run with:
    pytest -q
"""

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pytest

from src.preprocess import engineer_features, single_transaction_to_frame
from src.predict import predict_transaction


FRAUD_LIKE = {
    "step": 1,
    "type": "TRANSFER",
    "amount": 181.0,
    "oldbalanceOrg": 181.0,
    "newbalanceOrig": 0.0,
    "oldbalanceDest": 0.0,
    "newbalanceDest": 0.0,
    "nameDest": "C553264065",
}

CLEAN_LIKE = {
    "step": 1,
    "type": "PAYMENT",
    "amount": 9839.64,
    "oldbalanceOrg": 170136.0,
    "newbalanceOrig": 160296.36,
    "oldbalanceDest": 0.0,
    "newbalanceDest": 0.0,
    "nameDest": "M1979787155",
}


def test_single_transaction_to_frame_has_expected_columns():
    frame = single_transaction_to_frame(FRAUD_LIKE)
    assert list(frame.columns) == [
        "step", "type", "amount", "oldbalanceOrg", "newbalanceOrig",
        "oldbalanceDest", "newbalanceDest", "errorBalanceOrig",
        "errorBalanceDest", "hourOfDay", "isMerchantDest",
    ]
    assert frame.shape[0] == 1


def test_is_merchant_dest_flag():
    import pandas as pd
    df = pd.DataFrame([FRAUD_LIKE, CLEAN_LIKE])
    out = engineer_features(df)
    assert out["isMerchantDest"].tolist() == [0, 1]


@pytest.mark.skipif(
    not os.path.exists("models/fraud_model.joblib"),
    reason="Trained model artifact not present; run `python -m src.train` first.",
)
def test_predict_transaction_returns_expected_keys():
    result = predict_transaction(FRAUD_LIKE)
    assert set(result.keys()) == {"isFraud", "fraudProbability", "riskLevel"}
    assert 0.0 <= result["fraudProbability"] <= 1.0


@pytest.mark.skipif(
    not os.path.exists("models/fraud_model.joblib"),
    reason="Trained model artifact not present; run `python -m src.train` first.",
)
def test_flask_predict_endpoint():
    from app import app

    client = app.test_client()

    res = client.post("/api/predict", json=FRAUD_LIKE)
    assert res.status_code == 200
    body = res.get_json()
    assert "fraudProbability" in body

    res_missing = client.post("/api/predict", json={"type": "PAYMENT"})
    assert res_missing.status_code == 400


@pytest.mark.skipif(
    not os.path.exists("models/fraud_model.joblib"),
    reason="Trained model artifact not present; run `python -m src.train` first.",
)
def test_health_endpoint():
    from app import app

    client = app.test_client()
    res = client.get("/api/health")
    assert res.status_code == 200
    assert res.get_json()["status"] == "ok"

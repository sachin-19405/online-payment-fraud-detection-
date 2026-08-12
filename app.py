"""
app.py
------
Flask web application for the Online Payment Fraud Detection project.

Routes:
    GET  /            -> UI for scoring a single transaction
    POST /api/predict -> JSON API, scores a single transaction
    GET  /api/health  -> health check (used by deploy platforms)

Run locally:
    python app.py

Run with gunicorn (production):
    gunicorn app:app --bind 0.0.0.0:$PORT
"""

from __future__ import annotations

import os

from flask import Flask, jsonify, render_template, request

from src.predict import DEFAULT_MODEL_PATH, get_model, predict_transaction
from src.preprocess import TRANSACTION_TYPES

app = Flask(__name__)

MODEL_PATH = os.environ.get("MODEL_PATH", DEFAULT_MODEL_PATH)

# Warm the model cache at import time so the first request isn't slow
# and so the app fails fast (at boot) if the artifact is missing.
try:
    get_model(MODEL_PATH)
    MODEL_READY = True
except Exception as exc:  # noqa: BLE001
    MODEL_READY = False
    print(f"WARNING: could not load model at startup: {exc}")


@app.route("/")
def index():
    return render_template("index.html", transaction_types=TRANSACTION_TYPES)


@app.route("/api/health")
def health():
    return jsonify({"status": "ok" if MODEL_READY else "model_unavailable"})


@app.route("/api/predict", methods=["POST"])
def api_predict():
    payload = request.get_json(silent=True) or request.form.to_dict()
    if not payload:
        return jsonify({"error": "No input data provided."}), 400

    required = [
        "type",
        "amount",
        "oldbalanceOrg",
        "newbalanceOrig",
        "oldbalanceDest",
        "newbalanceDest",
    ]
    missing = [f for f in required if f not in payload or payload[f] in (None, "")]
    if missing:
        return jsonify({"error": f"Missing fields: {', '.join(missing)}"}), 400

    try:
        record = {
            "step": float(payload.get("step", 1)),
            "type": str(payload["type"]).upper(),
            "amount": float(payload["amount"]),
            "oldbalanceOrg": float(payload["oldbalanceOrg"]),
            "newbalanceOrig": float(payload["newbalanceOrig"]),
            "oldbalanceDest": float(payload["oldbalanceDest"]),
            "newbalanceDest": float(payload["newbalanceDest"]),
            "nameDest": str(payload.get("nameDest", "C0000000000")),
        }
    except (TypeError, ValueError) as exc:
        return jsonify({"error": f"Invalid numeric field: {exc}"}), 400

    if record["type"] not in TRANSACTION_TYPES:
        return jsonify({"error": f"type must be one of {TRANSACTION_TYPES}"}), 400

    try:
        result = predict_transaction(record, MODEL_PATH)
    except Exception as exc:  # noqa: BLE001
        return jsonify({"error": f"Prediction failed: {exc}"}), 500

    return jsonify(result)


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    debug = os.environ.get("FLASK_DEBUG", "0") == "1"
    app.run(host="0.0.0.0", port=port, debug=debug)

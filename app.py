# DEPLOY TEST
"""
=============================================================================
  Insurance Premium Predictor — Flask API
  Version : 2.0.0
  Compat  : train_model.py v2.0.0  (Pipeline artefact)
=============================================================================

Endpoints
---------
  GET  /          — health check
  POST /predict   — predict premium given risk features
  GET  /features  — list expected input features (dev/debug)
=============================================================================
"""

from flask import Flask, request, jsonify
import joblib
import numpy as np
import os
import time

app = Flask(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# Load model artefact on startup (single load, fast inference)
# ─────────────────────────────────────────────────────────────────────────────

MODEL_PATH = os.environ.get("MODEL_PATH", "premium_model.pkl")

try:
    artefact      = joblib.load(MODEL_PATH)
    pipeline      = artefact["pipeline"]       # StandardScaler + XGBRegressor
    feature_names = artefact["feature_names"]  # Ordered list of engineered features
    model_version = artefact.get("version", "unknown")
    print(f"[INFO]  Model v{model_version} loaded — {len(feature_names)} features")
except FileNotFoundError:
    raise RuntimeError(
        f"Model file '{MODEL_PATH}' not found. "
        "Run train_model.py first to generate premium_model.pkl"
    )


# ─────────────────────────────────────────────────────────────────────────────
# Feature Engineering — mirrors train_model.py exactly
# CRITICAL: Any change here MUST be reflected in train_model.py and vice versa.
# ─────────────────────────────────────────────────────────────────────────────

def engineer_features(d: dict) -> np.ndarray:
    """
    Replicates the feature engineering pipeline from train_model.py.
    Accepts a raw input dict, returns a single-row numpy array
    aligned to `feature_names` for pipeline.predict().
    """
    iv  = d["income_volatility_score"]
    rr  = d["zone_rain_risk"]
    hr  = d["zone_heat_risk"]
    sr  = d["zone_strike_risk"]
    ovr = d["zone_overall_risk"]

    # ── Weighted composite risk score ────────────────────────────────────────
    weighted_risk_score = (
        rr  * 0.25 +
        hr  * 0.20 +
        sr  * 0.15 +
        ovr * 0.30 +
        iv  * 0.10
    )

    # ── Pairwise interaction features ────────────────────────────────────────
    rain_x_heat       = rr  * hr
    rain_x_overall    = rr  * ovr
    heat_x_strike     = hr  * sr
    income_x_overall  = iv  * ovr

    # ── Polynomial features ──────────────────────────────────────────────────
    rain_sq    = rr  ** 2
    overall_sq = ovr ** 2
    income_sq  = iv  ** 2

    # ── Ratio features ───────────────────────────────────────────────────────
    income_to_overall_ratio = iv / (ovr + 1e-6)

    # ── Total zone risk ──────────────────────────────────────────────────────
    total_zone_risk = rr + hr + sr + ovr

    # Build the engineered feature vector in the same column order as training
    feature_values = {
        "income_volatility_score"  : iv,
        "zone_rain_risk"           : rr,
        "zone_heat_risk"           : hr,
        "zone_strike_risk"         : sr,
        "zone_overall_risk"        : ovr,
        "weighted_risk_score"      : weighted_risk_score,
        "rain_x_heat"              : rain_x_heat,
        "rain_x_overall"           : rain_x_overall,
        "heat_x_strike"            : heat_x_strike,
        "income_x_overall"         : income_x_overall,
        "rain_sq"                  : rain_sq,
        "overall_sq"               : overall_sq,
        "income_sq"                : income_sq,
        "income_to_overall_ratio"  : income_to_overall_ratio,
        "total_zone_risk"          : total_zone_risk,
    }

    # Align to the exact column order from training
    row = np.array([[feature_values[name] for name in feature_names]])
    return row


# ─────────────────────────────────────────────────────────────────────────────
# Input Validation
# ─────────────────────────────────────────────────────────────────────────────

REQUIRED_FIELDS = [
    "income_volatility_score",
    "zone_rain_risk",
    "zone_heat_risk",
    "zone_strike_risk",
    "zone_overall_risk",
]

OPTIONAL_FIELDS = [
    "weekly_income_last_week",  # used for coverage_limit calculation only
]


def validate_input(data: dict) -> tuple[bool, str]:
    """Returns (is_valid, error_message)."""
    for field in REQUIRED_FIELDS:
        if field not in data:
            return False, f"Missing required field: '{field}'"
        val = data[field]
        if not isinstance(val, (int, float)):
            return False, f"Field '{field}' must be numeric, got {type(val).__name__}"
        if not (0.0 <= float(val) <= 1.0):
            return False, f"Field '{field}' must be in [0.0, 1.0], got {val}"
    return True, ""


# ─────────────────────────────────────────────────────────────────────────────
# Routes
# ─────────────────────────────────────────────────────────────────────────────

@app.route("/")
def home():
    return jsonify({
        "status" : "running",
        "service": "Insurance Premium Predictor API",
        "version": model_version,
        "endpoints": {
            "POST /predict" : "Predict insurance premium",
            "GET  /features": "List required input features",
        },
    })


@app.route("/predict", methods=["POST"])
def predict():
    """
    Predict insurance premium from risk-based features.

    Request Body (JSON)
    -------------------
    {
      "income_volatility_score"  : 0.45,   # required, float [0,1]
      "zone_rain_risk"           : 0.60,   # required, float [0,1]
      "zone_heat_risk"           : 0.55,   # required, float [0,1]
      "zone_strike_risk"         : 0.30,   # required, float [0,1]
      "zone_overall_risk"        : 0.70,   # required, float [0,1]
      "weekly_income_last_week"  : 5000    # optional, for coverage calculation
    }

    Response (JSON)
    ---------------
    {
      "recommended_premium" : 94.23,
      "coverage_limit"      : 4000.0,
      "risk_score"          : 0.52,
      "inference_ms"        : 1.3,
      "model_version"       : "2.0.0"
    }
    """
    # ── Parse JSON body ──────────────────────────────────────────────────────
    data = request.get_json(silent=True)
    if data is None:
        return jsonify({"error": "Request body must be valid JSON"}), 400

    # ── Validate ─────────────────────────────────────────────────────────────
    valid, err = validate_input(data)
    if not valid:
        return jsonify({"error": err}), 422

    # ── Feature Engineering ──────────────────────────────────────────────────
    try:
        X = engineer_features(data)
    except Exception as exc:
        return jsonify({"error": f"Feature engineering failed: {exc}"}), 500

    # ── Inference (timed for latency monitoring) ──────────────────────────────
    t0 = time.perf_counter()
    premium = float(pipeline.predict(X)[0])
    inference_ms = round((time.perf_counter() - t0) * 1000, 2)

    # ── Derived outputs ───────────────────────────────────────────────────────
    iv  = data["income_volatility_score"]
    rr  = data["zone_rain_risk"]
    hr  = data["zone_heat_risk"]
    sr  = data["zone_strike_risk"]
    ovr = data["zone_overall_risk"]

    risk_score = round((iv + rr + hr + sr + ovr) / 5, 4)

    weekly_income    = float(data.get("weekly_income_last_week", 0))
    coverage_limit   = round(weekly_income * 0.8, 2) if weekly_income else None

    # ── Response ──────────────────────────────────────────────────────────────
    response = {
        "recommended_premium": round(premium, 2),
        "risk_score"         : risk_score,
        "inference_ms"       : inference_ms,
        "model_version"      : model_version,
    }
    if coverage_limit is not None:
        response["coverage_limit"] = coverage_limit

    return jsonify(response), 200


@app.route("/features", methods=["GET"])
def get_features():
    """Returns the list of features the model expects (for debugging/docs)."""
    return jsonify({
        "required_input_fields": REQUIRED_FIELDS,
        "engineered_features"  : feature_names,
        "total_features"       : len(feature_names),
    })


# ─────────────────────────────────────────────────────────────────────────────
# Entrypoint
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    debug = os.environ.get("FLASK_DEBUG", "false").lower() == "true"
    app.run(host="0.0.0.0", port=port, debug=debug)
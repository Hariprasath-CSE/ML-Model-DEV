from flask import Flask, request, jsonify
import joblib
import numpy as np
import os

app = Flask(__name__)

# Load trained model
model = joblib.load("premium_model.pkl")


@app.route("/")
def home():
    return "ML Premium API Running"


@app.route("/predict", methods=["POST"])
def predict():

    data = request.json

    # Extract required fields
    weekly_income_last_week = data["weekly_income_last_week"]

    income_volatility_score = data["income_volatility_score"]
    zone_rain_risk = data["zone_rain_risk"]
    zone_heat_risk = data["zone_heat_risk"]
    zone_strike_risk = data["zone_strike_risk"]
    zone_overall_risk = data["zone_overall_risk"]

    # Prepare ML features
    features = np.array([[
        income_volatility_score,
        zone_rain_risk,
        zone_heat_risk,
        zone_strike_risk,
        zone_overall_risk
    ]])

    # Predict premium using ML model
    premium = model.predict(features)[0]

    # Calculate risk score
    risk_score = (
        income_volatility_score +
        zone_rain_risk +
        zone_heat_risk +
        zone_strike_risk +
        zone_overall_risk
    ) / 5

    # Calculate coverage
    coverage_limit = weekly_income_last_week * 0.8

    # Return response (convert to Python float)
    return jsonify({
        "recommended_premium": float(round(premium, 2)),
        "coverage_limit": float(round(coverage_limit, 2)),
        "risk_score": float(round(risk_score, 2))
    })


# Deployment-ready server config
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
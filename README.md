# ML Premium Prediction API

## Overview

This project provides a **Machine Learning–powered API** that predicts **dynamic insurance premiums for gig workers** based on multiple risk factors such as weather conditions, income volatility, and operational activity.

The system is designed as a **microservice** that can integrate with a larger insurance platform. The ML model analyzes risk-related inputs and returns recommended premium values along with coverage limits and risk scores.

The model is trained using **XGBoost Regression** and exposed through a **Flask REST API**.

---

# Problem Statement

Gig workers such as delivery partners often face **income volatility and environmental risks** (rain, heat, strikes, etc.). Traditional insurance systems use **static premiums**, which do not reflect real-time risk conditions.

This project introduces a **dynamic premium calculation model** that adapts insurance pricing based on real-world risk indicators.

---

# Machine Learning Model

### Model Used

**XGBoost Regressor (Extreme Gradient Boosting)**

### Why XGBoost?

XGBoost was selected because:

* It handles **non-linear relationships** effectively.
* It provides **high prediction accuracy** for structured/tabular data.
* It is widely used in **financial risk modeling and insurance pricing**.
* It supports efficient training and scalability.

### Model Performance

Evaluation metrics used:

* **R² Score:** ~0.92
* **Mean Squared Error (MSE)**

These metrics were calculated using the **Scikit-learn metrics library**.

---

# Features Used in the Model

The model predicts premium using the following risk indicators:

| Feature                 | Description                             |
| ----------------------- | --------------------------------------- |
| income_volatility_score | Measures income stability of gig worker |
| zone_rain_risk          | Rain risk affecting delivery operations |
| zone_heat_risk          | Extreme heat risk level                 |
| zone_strike_risk        | Strike or protest risk in the area      |
| zone_overall_risk       | Aggregated risk level of the location   |

---

# System Architecture

```
Frontend Application
        │
        ▼
Node Backend
        │
        ▼
Flask ML API
        │
        ▼
XGBoost Model
        │
        ▼
Premium Prediction Response
```

The ML module operates as a **prediction microservice**.

---

# Project Structure

```
ML-Model-DEV
│
├── app.py              # Flask API server
├── train_model.py      # Model training script
├── premium_model.pkl   # Serialized trained model
├── requirements.txt    # Python dependencies
└── README.md           # Project documentation
```

---

# Setup Instructions

## 1. Clone the Repository

```
git clone https://github.com/Hariprasath-CSE/ML-Model-DEV.git
cd ML-Model-DEV
```

---

## 2. Install Dependencies

```
pip install -r requirements.txt
```

---

## 3. Train the Model

Run the training script:

```
python train_model.py
```

This will generate the trained model file:

```
premium_model.pkl
```

---

## 4. Start the API Server

```
python app.py
```

The server will start at:

```
http://127.0.0.1:5000
```

---

# API Documentation

## Endpoint

```
POST /predict
```

### Example URL

```
http://127.0.0.1:5000/predict
```

---

# Example Request

```json
{
 "partner_id": "SWG93821",
 "platform": "Swiggy",
 "city": "Chennai",
 "work_type": "full_time",
 "vehicle_type": "bike",
 "weekly_income_last_week": 6200,
 "weekly_income_last_4weeks_avg": 5900,
 "deliveries_last_week": 112,
 "working_hours_last_week": 48,
 "income_volatility_score": 0.32,
 "zone_rain_risk": 0.72,
 "zone_heat_risk": 0.61,
 "zone_strike_risk": 0.30,
 "zone_overall_risk": 0.55
}
```

---

# Example Response

```json
{
 "recommended_premium": 78.4,
 "coverage_limit": 4960,
 "risk_score": 0.50
}
```

---

# Technology Stack

| Component            | Technology     |
| -------------------- | -------------- |
| Programming Language | Python         |
| ML Model             | XGBoost        |
| Model Evaluation     | Scikit-learn   |
| Data Processing      | Pandas / NumPy |
| Model Serialization  | Joblib         |
| API Framework        | Flask          |
| Version Control      | Git / GitHub   |

---

# Deployment

The API can be deployed using platforms such as:

* Render
* Railway
* AWS
* Docker

Example production endpoint:

```
https://ml-premium-api.onrender.com/predict
```

---

# Future Improvements

Potential enhancements for the system:

* Integrate **real weather API data**
* Add **geospatial risk modeling**
* Use **larger real-world datasets**
* Implement **model retraining pipeline**
* Add **authentication and monitoring**

---

# Contributors

Hariprasath
Machine Learning Module Developer

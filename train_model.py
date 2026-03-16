import pandas as pd
import numpy as np
import joblib
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error

# -----------------------------
# Create Synthetic Dataset
# -----------------------------

np.random.seed(42)

data_size = 200

data = {
    "income_volatility_score": np.random.uniform(0.1,0.9,data_size),
    "zone_rain_risk": np.random.uniform(0.1,0.9,data_size),
    "zone_heat_risk": np.random.uniform(0.1,0.9,data_size),
    "zone_strike_risk": np.random.uniform(0.05,0.8,data_size),
    "zone_overall_risk": np.random.uniform(0.1,0.9,data_size)
}

df = pd.DataFrame(data)

# Target premium formula (simulated)
df["premium"] = (
    df["zone_rain_risk"]*50 +
    df["zone_heat_risk"]*40 +
    df["zone_strike_risk"]*30 +
    df["zone_overall_risk"]*60 +
    df["income_volatility_score"]*35
)

print("Dataset Generated")
print(df.head())

# -----------------------------
# Features
# -----------------------------

X = df[[
    "income_volatility_score",
    "zone_rain_risk",
    "zone_heat_risk",
    "zone_strike_risk",
    "zone_overall_risk"
]]

# Target
y = df["premium"]

# -----------------------------
# Train Test Split
# -----------------------------

X_train,X_test,y_train,y_test = train_test_split(
    X,y,test_size=0.2,random_state=42
)

# -----------------------------
# Train Model
# -----------------------------

model = XGBRegressor(
    n_estimators=100,
    learning_rate=0.1,
    max_depth=4
)

model.fit(X_train,y_train)

print("Model Training Completed")

# -----------------------------
# Accuracy Evaluation
# -----------------------------

pred = model.predict(X_test)

r2 = r2_score(y_test,pred)
mse = mean_squared_error(y_test,pred)

print("Model Accuracy (R2 Score):",r2)
print("Model MSE:",mse)

# -----------------------------
# Save Model
# -----------------------------

joblib.dump(model,"premium_model.pkl")

print("Model saved successfully")
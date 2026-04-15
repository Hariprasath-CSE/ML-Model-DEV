"""
=============================================================================
  Insurance Premium Predictor - Production-Level ML Pipeline
  Author  : Senior ML Engineer
  Version : 2.0.0
  Engine  : XGBoost + StandardScaler + RandomizedSearchCV + K-Fold CV
=============================================================================

Pipeline Overview
-----------------
  Raw Data
    (-)  Feature Engineering  (weighted risk, interactions, polynomials)
    (-)  StandardScaler       (normalisation)
    (-)  XGBoost Regressor    (tuned via RandomizedSearchCV)
    (-)  K-Fold CV (k=5)      (generalisation check)
    (-)  Evaluation           (R2, MAE, RMSE)
    (-)  Save                 (model + scaler -> premium_model.pkl)
=============================================================================
"""

# -- Standard library ------------------------------------------------------
import warnings
import time

# -- Third-party -----------------------------------------------------------
import numpy as np
import pandas as pd
import joblib

from sklearn.model_selection import (
    train_test_split,
    RandomizedSearchCV,
    KFold,
    cross_val_score,
)
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.pipeline import Pipeline

from xgboost import XGBRegressor

warnings.filterwarnings("ignore")

# -----------------------------------------------------------------------------
# SECTION 1 - Configuration
# -----------------------------------------------------------------------------

RANDOM_STATE   = 42
DATA_SIZE      = 5_000          # Larger dataset -> better generalisation
TEST_SIZE      = 0.20           # 80/20 split
N_FOLDS        = 5              # K-Fold CV folds
N_ITER_SEARCH  = 40             # RandomizedSearchCV iterations
CV_SEARCH      = 3              # Inner CV folds during hyperparameter search
MODEL_PATH     = "premium_model.pkl"

# -----------------------------------------------------------------------------
# SECTION 2 - Synthetic Dataset Generation
# -----------------------------------------------------------------------------

def generate_dataset(n: int = DATA_SIZE, seed: int = RANDOM_STATE) -> pd.DataFrame:
    """
    Generates a realistic synthetic insurance dataset.

    Why synthetic?  The project currently has no CSV source, so we simulate
    data that closely mirrors real-world distributions (bounded [0,1] scores).
    Replace this function with pd.read_csv() when real data is available.
    """
    rng = np.random.default_rng(seed)

    df = pd.DataFrame({
        "income_volatility_score": rng.uniform(0.05, 0.95, n),
        "zone_rain_risk"         : rng.uniform(0.05, 0.95, n),
        "zone_heat_risk"         : rng.uniform(0.05, 0.95, n),
        "zone_strike_risk"       : rng.uniform(0.05, 0.80, n),
        "zone_overall_risk"      : rng.uniform(0.10, 0.95, n),
    })

    # Realistic premium formula with multiplicative interaction + noise
    noise = rng.normal(0, 2.5, n)
    df["premium"] = (
        df["zone_rain_risk"]          * 50 +
        df["zone_heat_risk"]          * 40 +
        df["zone_strike_risk"]        * 30 +
        df["zone_overall_risk"]       * 60 +
        df["income_volatility_score"] * 35 +
        # Multiplicative interaction (non-linearity the model must learn)
        df["zone_rain_risk"] * df["zone_heat_risk"] * 20 +
        df["zone_overall_risk"] * df["income_volatility_score"] * 15 +
        noise
    )

    return df


# -----------------------------------------------------------------------------
# SECTION 3 - Feature Engineering
# -----------------------------------------------------------------------------

BASE_FEATURES = [
    "income_volatility_score",
    "zone_rain_risk",
    "zone_heat_risk",
    "zone_strike_risk",
    "zone_overall_risk",
]


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Transforms raw features into a richer representation.

    Improvements over baseline
    --------------------------
    1. Weighted Risk Score    - domain-weighted composite of all risk signals.
    2. Interaction Features   - pairwise products capture non-linear synergies.
    3. Polynomial Features    - squared terms for curvature in risk response.
    4. Ratio Features         - relative risk magnitudes (income vs. zone).
    """
    fe = df.copy()

    # -- 3.1 Weighted composite risk score (domain-informed weights) ----------
    w = dict(rain=0.25, heat=0.20, strike=0.15, overall=0.30, income=0.10)
    fe["weighted_risk_score"] = (
        fe["zone_rain_risk"]          * w["rain"]   +
        fe["zone_heat_risk"]          * w["heat"]   +
        fe["zone_strike_risk"]        * w["strike"] +
        fe["zone_overall_risk"]       * w["overall"]+
        fe["income_volatility_score"] * w["income"]
    )

    # -- 3.2 Pairwise interaction features ------------------------------------
    fe["rain_x_heat"]    = fe["zone_rain_risk"]    * fe["zone_heat_risk"]
    fe["rain_x_overall"] = fe["zone_rain_risk"]    * fe["zone_overall_risk"]
    fe["heat_x_strike"]  = fe["zone_heat_risk"]    * fe["zone_strike_risk"]
    fe["income_x_overall"] = fe["income_volatility_score"] * fe["zone_overall_risk"]

    # -- 3.3 Polynomial (squared) features - captures non-linear risk curves --
    fe["rain_sq"]    = fe["zone_rain_risk"]    ** 2
    fe["overall_sq"] = fe["zone_overall_risk"] ** 2
    fe["income_sq"]  = fe["income_volatility_score"] ** 2

    # -- 3.4 Ratio feature - income risk relative to zone risk ----------------
    fe["income_to_overall_ratio"] = (
        fe["income_volatility_score"] /
        (fe["zone_overall_risk"] + 1e-6)   # epsilon avoids div-by-zero
    )

    # -- 3.5 Total zone risk --------------------------------------------------
    fe["total_zone_risk"] = (
        fe["zone_rain_risk"] +
        fe["zone_heat_risk"] +
        fe["zone_strike_risk"] +
        fe["zone_overall_risk"]
    )

    return fe


def get_feature_columns(df: pd.DataFrame) -> list:
    """Returns all feature column names (excludes the target 'premium')."""
    return [c for c in df.columns if c != "premium"]


# -----------------------------------------------------------------------------
# SECTION 4 - Baseline Model (for comparison)
# -----------------------------------------------------------------------------

def train_baseline(X_train, X_test, y_train, y_test) -> dict:
    """
    Trains a minimal XGBoost model using only the five raw features,
    with no scaling or tuning.  Provides a benchmark for improvement.
    """
    print("\n" + "="*60)
    print("  BASELINE MODEL  (raw features, default hyperparams)")
    print("="*60)

    baseline = XGBRegressor(
        n_estimators  = 100,
        learning_rate = 0.1,
        max_depth     = 4,
        random_state  = RANDOM_STATE,
        verbosity     = 0,
    )
    baseline.fit(X_train[BASE_FEATURES], y_train)
    pred_base = baseline.predict(X_test[BASE_FEATURES])

    metrics = _compute_metrics(y_test, pred_base)
    _print_metrics(metrics, label="Baseline")
    return metrics


# -----------------------------------------------------------------------------
# SECTION 5 - Hyperparameter Search Space
# -----------------------------------------------------------------------------

PARAM_DISTRIBUTIONS = {
    # -- Tree structure -------------------------------------------------------
    "xgb__n_estimators"       : [200, 300, 400, 500, 600, 800],
    "xgb__max_depth"          : [3, 4, 5, 6, 7],
    "xgb__min_child_weight"   : [1, 3, 5, 7, 10],

    # -- Learning dynamics ----------------------------------------------------
    "xgb__learning_rate"      : [0.01, 0.03, 0.05, 0.08, 0.1, 0.15],
    "xgb__subsample"          : [0.6, 0.7, 0.8, 0.9, 1.0],
    "xgb__colsample_bytree"   : [0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
    "xgb__colsample_bylevel"  : [0.6, 0.7, 0.8, 1.0],

    # -- Regularisation (prevents overfitting) --------------------------------
    "xgb__reg_alpha"          : [0, 0.01, 0.05, 0.1, 0.5, 1.0],   # L1
    "xgb__reg_lambda"         : [0.5, 1.0, 1.5, 2.0, 5.0],         # L2
    "xgb__gamma"              : [0, 0.1, 0.3, 0.5, 1.0],           # min split gain
}


# -----------------------------------------------------------------------------
# SECTION 6 - Pipeline: Scaler + XGBoost
# -----------------------------------------------------------------------------

def build_pipeline() -> Pipeline:
    """
    Wraps StandardScaler + XGBRegressor in a single sklearn Pipeline.

    Why a Pipeline?
    - Guarantees the scaler is fit only on training data (no data leakage).
    - Single joblib.dump() saves both scaler parameters and model weights.
    - Flask API calls pipeline.predict() with raw engineered features - no
      manual scaling step needed at inference time.
    - Compatible with RandomizedSearchCV out of the box.
    """
    return Pipeline([
        ("scaler", StandardScaler()),
        ("xgb"   , XGBRegressor(
            objective    = "reg:squarederror",
            tree_method  = "hist",          # Fast histogram-based algorithm
            random_state = RANDOM_STATE,
            verbosity    = 0,
            n_jobs       = -1,              # Use all CPU cores
        )),
    ])


# -----------------------------------------------------------------------------
# SECTION 7 - Hyperparameter Tuning (RandomizedSearchCV)
# -----------------------------------------------------------------------------

def tune_hyperparameters(pipeline: Pipeline, X_train, y_train) -> RandomizedSearchCV:
    """
    Searches over PARAM_DISTRIBUTIONS using RandomizedSearchCV.

    Why Randomized over Grid Search?
    - Explores a much larger space in the same compute budget.
    - Empirically equivalent or better than exhaustive grid search.

    Overfitting prevention during search
    - R2 on 3-fold inner CV is the scoring metric - not train score.
    - Regularisation params (alpha, lambda, gamma) are included in search.
    """
    print("\n" + "="*60)
    print("  HYPERPARAMETER TUNING  (RandomizedSearchCV)")
    print("="*60)
    print(f"  iterations : {N_ITER_SEARCH}")
    print(f"  inner CV   : {CV_SEARCH}-fold")
    print(f"  scoring    : R2\n")

    search = RandomizedSearchCV(
        estimator          = pipeline,
        param_distributions= PARAM_DISTRIBUTIONS,
        n_iter             = N_ITER_SEARCH,
        cv                 = CV_SEARCH,
        scoring            = "r2",
        refit              = True,       # Refit best params on full train set
        verbose            = 1,
        random_state       = RANDOM_STATE,
        n_jobs             = -1,
    )

    t0 = time.time()
    search.fit(X_train, y_train)
    elapsed = time.time() - t0

    print(f"\n  [OK]  Search completed in {elapsed:.1f}s")
    print(f"  Best CV R2  : {search.best_score_:.6f}")
    print("\n  Best Hyperparameters Found:")
    for k, v in sorted(search.best_params_.items()):
        print(f"    {k:<35} = {v}")

    return search


# -----------------------------------------------------------------------------
# SECTION 8 - K-Fold Cross Validation
# -----------------------------------------------------------------------------

def cross_validate_model(best_pipeline: Pipeline, X, y) -> None:
    """
    Evaluates the best pipeline on the full dataset using K-Fold CV.

    Why after tuning?
    - Provides an unbiased estimate of generalisation performance.
    - Confirms the model is stable across different data partitions.
    """
    print("\n" + "="*60)
    print(f"  K-FOLD CROSS VALIDATION  (k={N_FOLDS})")
    print("="*60)

    kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)

    r2_scores  = cross_val_score(best_pipeline, X, y, cv=kf, scoring="r2", n_jobs=-1)
    mae_scores = cross_val_score(best_pipeline, X, y, cv=kf,
                                 scoring="neg_mean_absolute_error", n_jobs=-1)
    rmse_scores= cross_val_score(best_pipeline, X, y, cv=kf,
                                 scoring="neg_root_mean_squared_error", n_jobs=-1)

    print(f"\n  R2   per fold : {np.round(r2_scores, 4)}")
    print(f"  Mean R2       : {r2_scores.mean():.6f}  ± {r2_scores.std():.6f}")

    print(f"\n  MAE  per fold : {np.round(-mae_scores, 4)}")
    print(f"  Mean MAE      : {(-mae_scores).mean():.4f}  ± {(-mae_scores).std():.4f}")

    print(f"\n  RMSE per fold : {np.round(-rmse_scores, 4)}")
    print(f"  Mean RMSE     : {(-rmse_scores).mean():.4f}  ± {(-rmse_scores).std():.4f}")


# -----------------------------------------------------------------------------
# SECTION 9 - Evaluation Helpers
# -----------------------------------------------------------------------------

def _compute_metrics(y_true, y_pred) -> dict:
    return {
        "r2"  : r2_score(y_true, y_pred),
        "mae" : mean_absolute_error(y_true, y_pred),
        "rmse": np.sqrt(mean_squared_error(y_true, y_pred)),
    }


def _print_metrics(m: dict, label: str = "Model") -> None:
    print(f"\n  [{label}]")
    print(f"  R2   Score : {m['r2']:.6f}")
    print(f"  MAE        : {m['mae']:.4f}")
    print(f"  RMSE       : {m['rmse']:.4f}")


def evaluate_final_model(pipeline: Pipeline, X_test, y_test) -> dict:
    """Final hold-out evaluation on the unseen test set."""
    print("\n" + "="*60)
    print("  FINAL MODEL - HOLD-OUT TEST SET EVALUATION")
    print("="*60)

    pred = pipeline.predict(X_test)
    metrics = _compute_metrics(y_test, pred)
    _print_metrics(metrics, label="Optimised XGBoost")
    return metrics


# -----------------------------------------------------------------------------
# SECTION 10 - Feature Importance
# -----------------------------------------------------------------------------

def show_feature_importance(pipeline: Pipeline, feature_names: list) -> None:
    """
    Prints top-N features ranked by XGBoost's gain-based importance.
    Gain measures total improvement in loss attributed to each feature
    (more meaningful than frequency-based importance).
    """
    print("\n" + "="*60)
    print("  FEATURE IMPORTANCE  (gain-based)")
    print("="*60)

    xgb_model = pipeline.named_steps["xgb"]
    importance = xgb_model.feature_importances_

    fi = (
        pd.Series(importance, index=feature_names)
        .sort_values(ascending=False)
    )

    print(f"\n  {'Feature':<35} {'Importance':>10}")
    print("  " + "-"*46)
    for feat, score in fi.items():
        bar = "|" * int(score * 40)
        print(f"  {feat:<35} {score:>10.6f}  {bar}")


# -----------------------------------------------------------------------------
# SECTION 11 - Baseline vs Improved Comparison
# -----------------------------------------------------------------------------

def print_comparison(baseline: dict, improved: dict) -> None:
    """Prints a side-by-side comparison table."""
    print("\n" + "="*60)
    print("  BASELINE vs OPTIMISED - COMPARISON SUMMARY")
    print("="*60)
    print(f"\n  {'Metric':<12} {'Baseline':>12} {'Optimised':>12} {'Delta Improvement':>16}")
    print("  " + "-"*54)

    for metric in ("r2", "mae", "rmse"):
        b = baseline[metric]
        o = improved[metric]
        # For R2, higher is better; for MAE/RMSE, lower is better
        if metric == "r2":
            delta = o - b
            direction = "(+)" if delta > 0 else "(-)"
        else:
            delta = b - o
            direction = "(-)" if b > o else "(+)"

        print(f"  {metric.upper():<12} {b:>12.4f} {o:>12.4f} {direction:>5} {abs(delta):>12.4f}")

    print("\n  Key Improvements Applied:")
    improvements = [
        "Feature Engineering    : weighted risk, interactions, polynomials, ratios",
        "StandardScaler         : normalised input (prevents scale-bias in tree)",
        "RandomizedSearchCV     : tuned 10 hyperparameters over 40 iterations",
        "Regularisation params  : L1 (alpha), L2 (lambda), gamma in search space",
        "K-Fold CV (k=5)        : robust generalisation estimate",
        "tree_method='hist'     : fast training (O(n) bin split algorithm)",
        "Pipeline (scaler+xgb)  : self-contained, no leakage, Flask-ready",
        "Larger dataset (5000)  : reduces variance in learned coefficients",
    ]
    for imp in improvements:
        print(f"    [OK]  {imp}")


# -----------------------------------------------------------------------------
# SECTION 12 - Save Model
# -----------------------------------------------------------------------------

def save_model(pipeline: Pipeline, feature_names: list, path: str = MODEL_PATH) -> None:
    """
    Persists the full pipeline (scaler + model) and the feature name list.

    Saved artefact structure
    ------------------------
    {
      "pipeline"      : sklearn.pipeline.Pipeline,   # scaler + xgboost
      "feature_names" : list[str],                   # ordered feature columns
      "version"       : str,
    }

    Flask API loads this single file and calls artifact["pipeline"].predict().
    """
    artefact = {
        "pipeline"      : pipeline,
        "feature_names" : feature_names,
        "version"       : "2.0.0",
    }
    joblib.dump(artefact, path, compress=3)   # compress=3, smaller file
    print(f"\n  [OK] Model artefact saved to: {path}")
    print(f"       Contains: scaler + XGBoost pipeline + feature names")


# -----------------------------------------------------------------------------
# SECTION 13 - Main Orchestrator
# -----------------------------------------------------------------------------

def main():
    print("\n" + "="*60)
    print("  Insurance Premium ML Pipeline  v2.0.0")
    print("="*60)

    # -- Step 1: Data ----------------------------------------------------------
    print("\n[1/8]  Generating dataset...")
    df = generate_dataset()
    print(f"       Rows: {len(df):,}  |  Columns: {list(df.columns)}")
    print(df.describe().round(3).to_string())

    # -- Step 2: Feature Engineering -------------------------------------------
    print("\n[2/8]  Engineering features...")
    df_fe = engineer_features(df)
    feature_cols = get_feature_columns(df_fe)
    print(f"       {len(BASE_FEATURES)} base features -> {len(feature_cols)} engineered features")
    print(f"       New : {[c for c in feature_cols if c not in BASE_FEATURES]}")

    # -- Step 3: Train/Test Split ----------------------------------------------
    print("\n[3/8]  Splitting data...")
    X = df_fe[feature_cols]
    y = df_fe["premium"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )
    print(f"       Train: {len(X_train):,}  |  Test: {len(X_test):,}")

    # -- Step 4: Baseline (comparison benchmark) -------------------------------
    print("\n[4/8]  Training baseline model...")
    # Baseline uses only raw features - no scaler, no tuning
    X_train_raw = X_train[BASE_FEATURES]
    X_test_raw  = X_test[BASE_FEATURES]
    baseline_metrics = train_baseline(X_train, X_test, y_train, y_test)

    # -- Step 5: Build Pipeline -----------------------------------------------
    print("\n[5/8]  Building Scaler + XGBoost pipeline...")
    pipeline = build_pipeline()

    # -- Step 6: Hyperparameter Tuning ----------------------------------------
    print("\n[6/8]  Running hyperparameter search...")
    search = tune_hyperparameters(pipeline, X_train, y_train)
    best_pipeline = search.best_estimator_

    # -- Step 7: Evaluate on hold-out set -------------------------------------
    print("\n[7/8]  Evaluating final model...")
    improved_metrics = evaluate_final_model(best_pipeline, X_test, y_test)

    # -- Step 7b: K-Fold CV on full dataset ------------------------------------
    cross_validate_model(best_pipeline, X, y)

    # -- Step 7c: Feature Importance -------------------------------------------
    show_feature_importance(best_pipeline, feature_cols)

    # -- Step 7d: Baseline vs Improved -----------------------------------------
    print_comparison(baseline_metrics, improved_metrics)

    # -- Step 8: Save ----------------------------------------------------------
    print("\n[8/8]  Saving model ...")
    save_model(best_pipeline, feature_cols)

    print("\n" + "="*60)
    print(f"  Pipeline complete.  Final R2: {improved_metrics['r2']:.6f}")
    print("="*60 + "\n")


# -----------------------------------------------------------------------------
if __name__ == "__main__":
    main()
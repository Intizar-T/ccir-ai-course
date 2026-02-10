"""
Predict Number of Admissions from air quality and weather (merged_data.csv).

Pipeline: load → preprocess (impute/drop missing, winsorize) → feature engineering
(lags, rolling means, cyclical time) → time-ordered train/val/test split → scale features
→ train multiple regression models → report comparison table and best model by Test R².
"""

import os
import numpy as np
import pandas as pd
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import ElasticNetCV, Ridge, RidgeCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

try:
    import xgboost as xgb
    HAS_XGB = True
except ImportError:
    HAS_XGB = False


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

USE_IMPUTATION = True  # If True, impute missing features (ffill/bfill); else drop rows

TIMESTAMP_COL = "Timestamp"
TARGET_COL = "Number of Admissions"
FEATURE_COLS = [
    "PM2.5 (µg/m³)",
    "PM10 (µg/m³)",
    "Index Value",
    "temperature_2m_max (°C)",
    "apparent_temperature_max (°C)",
    "Ozone (µg/m³)",
    "NO2 (µg/m³)",
    "CO (mg/m³)",
    "RH (%)",
    "temperature_2m_min (°C)",
]


# ---------------------------------------------------------------------------
# Load & preprocess
# ---------------------------------------------------------------------------

def load_data():
    """Load CSV, keep timestamp + feature columns + target, parse date."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    path = os.path.join(script_dir, "data", "merged_data.csv")
    df = pd.read_csv(path)
    cols = [TIMESTAMP_COL] + [c for c in FEATURE_COLS if c in df.columns] + [TARGET_COL]
    df = df[[c for c in cols if c in df.columns]].copy()
    if TIMESTAMP_COL in df.columns:
        df["Date"] = pd.to_datetime(df[TIMESTAMP_COL], errors="coerce")
        df = df.drop(columns=[TIMESTAMP_COL])
    return df


def preprocess_data(df, impute_features=False):
    """Drop rows with missing target; optionally impute or drop missing features. Winsorize target at 99th percentile."""
    df = df.sort_values("Date").copy()
    df = df.dropna(subset=[TARGET_COL])
    if impute_features:
        for col in FEATURE_COLS:
            if col in df.columns and df[col].isna().any():
                df[col] = df[col].ffill().bfill()
        df = df.dropna(subset=[c for c in FEATURE_COLS if c in df.columns])
    else:
        df = df.dropna(subset=[c for c in FEATURE_COLS if c in df.columns])
    q99 = df[TARGET_COL].quantile(0.99)
    df.loc[df[TARGET_COL] > q99, TARGET_COL] = q99
    return df.reset_index(drop=True)


# ---------------------------------------------------------------------------
# Feature engineering
# ---------------------------------------------------------------------------

def feature_engineering(df):
    """Add PM ratio, temp range, 1/3/7/14-day lags, 3/7/14-day rolling means, cyclical time, weekend. Drop rows with NaN from lags."""
    df = df.sort_values("Date").reset_index(drop=True)
    base_cols = [c for c in FEATURE_COLS if c in df.columns]

    if "PM2.5 (µg/m³)" in df.columns and "PM10 (µg/m³)" in df.columns:
        pm25, pm10 = df["PM2.5 (µg/m³)"], df["PM10 (µg/m³)"]
        df["PM_ratio"] = np.where(pm10 > 0, pm25 / pm10, np.nan)
        df["PM_ratio"] = df["PM_ratio"].fillna(df["PM_ratio"].median())
        base_cols = base_cols + ["PM_ratio"]

    tmax, tmin = "temperature_2m_max (°C)", "temperature_2m_min (°C)"
    if tmax in df.columns and tmin in df.columns:
        df["temp_range"] = df[tmax] - df[tmin]

    for col in base_cols:
        for lag in (1, 3, 7, 14):
            df[f"{col}_lag{lag}"] = df[col].shift(lag)

    for col in [c for c in ("PM2.5 (µg/m³)", "PM10 (µg/m³)", "Index Value") if c in df.columns]:
        for w in (3, 7, 14):
            df[f"{col}_roll{w}"] = df[col].rolling(w, min_periods=1).mean()

    dt = pd.to_datetime(df["Date"])
    df["dow_sin"] = np.sin(2 * np.pi * dt.dt.dayofweek / 7)
    df["dow_cos"] = np.cos(2 * np.pi * dt.dt.dayofweek / 7)
    df["month_sin"] = np.sin(2 * np.pi * dt.dt.month / 12)
    df["month_cos"] = np.cos(2 * np.pi * dt.dt.month / 12)
    df["weekend"] = (dt.dt.dayofweek >= 5).astype(np.float64)

    for lag in (1, 3, 7, 14):
        df[f"admissions_lag{lag}"] = df[TARGET_COL].shift(lag)

    lag_cols = [f"{c}_lag{lag}" for c in base_cols for lag in (1, 3, 7, 14)]
    lag_cols += [f"admissions_lag{lag}" for lag in (1, 3, 7, 14)]
    lag_cols = [c for c in lag_cols if c in df.columns]
    df = df.dropna(how="any", subset=lag_cols)
    return df.reset_index(drop=True)


def get_feature_columns():
    """List of feature column names (only those present in df are used in split)."""
    base = [c for c in FEATURE_COLS] + ["PM_ratio", "temp_range"]
    lags = [f"{c}_lag{lag}" for c in base for lag in (1, 3, 7, 14)]
    lags += [f"admissions_lag{lag}" for lag in (1, 3, 7, 14)]
    rolls = [f"{r}_roll{w}" for r in ("PM2.5 (µg/m³)", "PM10 (µg/m³)", "Index Value") for w in (3, 7, 14)]
    time_ = ["dow_sin", "dow_cos", "month_sin", "month_cos", "weekend"]
    return base + lags + rolls + time_


def get_reduced_feature_columns():
    """Reduced set: admission lags + key rolling + time (for Ridge/ElasticNet reduced)."""
    return [
        "admissions_lag1", "admissions_lag3", "admissions_lag7", "admissions_lag14",
        "PM2.5 (µg/m³)_roll7", "PM2.5 (µg/m³)_roll14", "PM10 (µg/m³)_roll7", "Index Value_roll7",
        "dow_sin", "dow_cos", "month_sin", "month_cos", "weekend",
    ]


# ---------------------------------------------------------------------------
# Split & scale
# ---------------------------------------------------------------------------

def training_data_split(df, train_frac=0.70, val_frac=0.15, test_frac=0.15):
    """Time-ordered split. Scale features with StandardScaler fitted on train only."""
    assert abs(train_frac + val_frac + test_frac - 1.0) < 1e-9
    feat_cols = [c for c in get_feature_columns() if c in df.columns]
    if not feat_cols:
        raise ValueError("No feature columns found in df.")
    X = df[feat_cols].astype(float)
    y = df[TARGET_COL]
    n = len(df)
    n_train = int(n * train_frac)
    n_val = int(n * val_frac)
    n_test = n - n_train - n_val
    X_train = X.iloc[:n_train]
    X_val = X.iloc[n_train : n_train + n_val]
    X_test = X.iloc[n_train + n_val :]
    y_train = y.iloc[:n_train]
    y_val = y.iloc[n_train : n_train + n_val]
    y_test = y.iloc[n_train + n_val :]
    scaler = StandardScaler()
    X_train = pd.DataFrame(scaler.fit_transform(X_train), columns=feat_cols, index=X_train.index)
    X_val = pd.DataFrame(scaler.transform(X_val), columns=feat_cols, index=X_val.index)
    X_test = pd.DataFrame(scaler.transform(X_test), columns=feat_cols, index=X_test.index)
    return X_train, X_val, X_test, y_train, y_val, y_test, scaler


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate_model(model, X, y):
    """Return dict with MAE, RMSE, R²."""
    pred = model.predict(X)
    return {
        "mae": mean_absolute_error(y, pred),
        "rmse": np.sqrt(mean_squared_error(y, pred)),
        "r2": r2_score(y, pred),
    }


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------

def dummy_regression_model():
    return DummyRegressor(strategy="mean")


def ridge_tuned_model(cv=5):
    return RidgeCV(alphas=np.logspace(-3, 2, 100), cv=cv)


class RidgeLogTarget:
    """Ridge on log(1+y); predict with expm1 for original scale."""

    def __init__(self):
        self._model = RidgeCV(alphas=np.logspace(-3, 2, 100), cv=5)

    def fit(self, X, y):
        self._model.fit(X, np.log1p(y))
        return self

    def predict(self, X):
        return np.expm1(self._model.predict(X))


def elastic_net_cv_model(cv=5):
    return ElasticNetCV(cv=cv, l1_ratio=[0.1, 0.5, 0.9, 1.0], alphas=np.logspace(-3, 1, 50), max_iter=5000)


class RidgeReduced:
    """Ridge on reduced feature set (admission lags + key rolling + time)."""

    def __init__(self):
        self._model = RidgeCV(alphas=np.logspace(-2, 2, 50), cv=5)

    def fit(self, X, y):
        self._cols = [c for c in get_reduced_feature_columns() if c in X.columns]
        self._cols = self._cols or X.columns.tolist()[:10]
        self._model.fit(X[self._cols], y)
        return self

    def predict(self, X):
        return self._model.predict(X[self._cols])


def random_forest_regression_model(random_state=42):
    return RandomForestRegressor(n_estimators=100, max_depth=10, random_state=random_state)


def gradient_boosting_model(random_state=42):
    return GradientBoostingRegressor(
        n_estimators=200, max_depth=4, learning_rate=0.05,
        min_samples_leaf=10, subsample=0.8, random_state=random_state,
    )


def fnn_model(random_state=42):
    return MLPRegressor(hidden_layer_sizes=(64, 32), activation="relu", solver="adam", max_iter=1000, random_state=random_state)


def svm_model():
    return SVR(kernel="rbf", C=1.0, epsilon=0.1)


def xgboost_model(random_state=42):
    if not HAS_XGB:
        raise ImportError("xgboost not installed")
    return xgb.XGBRegressor(n_estimators=100, max_depth=5, learning_rate=0.1, random_state=random_state)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_all_models(X_train, X_val, X_test, y_train, y_val, y_test):
    """Train each model; return list of (name, val_metrics, test_metrics)."""
    models = [
        ("Dummy (mean)", dummy_regression_model()),
        ("Ridge (CV)", ridge_tuned_model()),
        ("Ridge (log y)", RidgeLogTarget()),
        ("ElasticNet (CV)", elastic_net_cv_model()),
        ("Ridge (reduced)", RidgeReduced()),
        ("Random Forest", random_forest_regression_model()),
        ("Gradient Boosting", gradient_boosting_model()),
        ("FNN (MLP)", fnn_model()),
        ("SVR", svm_model()),
    ]
    if HAS_XGB:
        models.append(("XGBoost", xgboost_model()))
    results = []
    for name, model in models:
        model.fit(X_train, y_train)
        results.append((name, evaluate_model(model, X_val, y_val), evaluate_model(model, X_test, y_test)))
    return results


def main():
    df = load_data()
    n_load = len(df)
    df = preprocess_data(df, impute_features=USE_IMPUTATION)
    n_pre = len(df)
    df = feature_engineering(df)
    n_fe = len(df)
    X_train, X_val, X_test, y_train, y_val, y_test, _ = training_data_split(df)

    print("Data: load → preprocess → feature_eng")
    print(f"  Rows: {n_load} → {n_pre} (impute={USE_IMPUTATION}) → {n_fe}")
    print(f"  Train / Val / Test: {len(y_train)} / {len(y_val)} / {len(y_test)}\n")

    results = run_all_models(X_train, X_val, X_test, y_train, y_val, y_test)

    print("Model comparison (Validation | Test)")
    print("-" * 72)
    print(f"{'Model':<18} {'Val MAE':>8} {'Val RMSE':>8} {'Val R²':>8}  |  {'Test MAE':>8} {'Test RMSE':>8} {'Test R²':>8}")
    print("-" * 72)
    for name, vm, tm in results:
        print(f"{name:<18} {vm['mae']:>8.4f} {vm['rmse']:>8.4f} {vm['r2']:>8.4f}  |  {tm['mae']:>8.4f} {tm['rmse']:>8.4f} {tm['r2']:>8.4f}")
    print("-" * 72)

    best_name, _, best_test = max(results, key=lambda r: r[2]["r2"])
    print(f"\nBest by Test R²: {best_name}  (R² = {best_test['r2']:.4f})")


if __name__ == "__main__":
    main()

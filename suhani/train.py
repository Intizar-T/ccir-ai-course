import os
import pandas as pd
import numpy as np
from sklearn.dummy import DummyRegressor
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge, RidgeCV, ElasticNetCV
from sklearn.neural_network import MLPRegressor
from sklearn.svm import SVR

try:
    import xgboost as xgb
    HAS_XGB = True
except ImportError:
    HAS_XGB = False

# Set True to impute missing feature values (ffill/bfill) instead of dropping those rows
USE_IMPUTATION = True

# Column names from merged_data.csv
TIMESTAMP_COL = "Timestamp"
TARGET_COL = "Number of Admissions"
FEATURE_COLS = [
    "PM2.5 (µg/m³)",
    "PM10 (µg/m³)",
    "Index Value",
    "temperature_2m_max (°C)",
    "apparent_temperature_max (°C)",
]


def load_data():
    """
    Loads the data from CSV: Timestamp, the 5 feature columns, and target.
    Parses Timestamp as datetime and renames to Date.
    """
    script_dir = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(script_dir, "data", "merged_data.csv")
    df = pd.read_csv(file_path)

    cols_needed = [TIMESTAMP_COL] + FEATURE_COLS + [TARGET_COL]
    df = df[[c for c in cols_needed if c in df.columns]].copy()

    if TIMESTAMP_COL in df.columns:
        df["Date"] = pd.to_datetime(df[TIMESTAMP_COL], errors="coerce")
        df = df.drop(columns=[TIMESTAMP_COL], errors="ignore")
    return df


def preprocess_data(df, impute_features=False):
    """
    Handles missing values:
    - Rows with missing target are always dropped (we cannot train without labels).
    - If impute_features=False: drop rows with missing in any of the 5 feature columns.
    - If impute_features=True: impute the 5 feature columns (forward-fill then backward-fill
      for time series; keeps more rows). Target missing still drops the row.
    Winsorizes target at 99th percentile.
    """
    df = df.sort_values("Date").copy()
    # Always drop rows with missing target
    df = df.dropna(subset=[TARGET_COL])

    if impute_features:
        for col in FEATURE_COLS:
            if col in df.columns and df[col].isna().any():
                df[col] = df[col].ffill().bfill()
        # If any feature is still NaN (e.g. all NaNs in column), drop those rows
        df = df.dropna(subset=FEATURE_COLS)
    else:
        df = df.dropna(subset=FEATURE_COLS)

    q99 = df[TARGET_COL].quantile(0.99)
    df.loc[df[TARGET_COL] > q99, TARGET_COL] = q99
    return df.reset_index(drop=True)


def feature_engineering(df):
    """
    Adds PM ratio, 1-day lags, 3/7-day rolling means for key pollutants,
    and cyclical time (day of week, month). Drops rows with NaN from lags/rolling.
    """
    df = df.sort_values("Date").reset_index(drop=True)
    pm25 = df["PM2.5 (µg/m³)"]
    pm10 = df["PM10 (µg/m³)"]
    df["PM_ratio"] = np.where(pm10 > 0, pm25 / pm10, np.nan)
    # Impute PM_ratio where PM10 was 0 (avoid dropping rows for that)
    if df["PM_ratio"].isna().any():
        df["PM_ratio"] = df["PM_ratio"].fillna(df["PM_ratio"].median())

    for col in FEATURE_COLS + ["PM_ratio"]:
        df[f"{col}_lag1"] = df[col].shift(1)
        df[f"{col}_lag3"] = df[col].shift(3)
        df[f"{col}_lag7"] = df[col].shift(7)

    # Rolling exposure (best practice for health/admissions)
    for col in ["PM2.5 (µg/m³)", "PM10 (µg/m³)", "Index Value"]:
        df[f"{col}_roll3"] = df[col].rolling(3, min_periods=1).mean()
        df[f"{col}_roll7"] = df[col].rolling(7, min_periods=1).mean()

    # Cyclical time (best practice: sin/cos to preserve continuity)
    dt = pd.to_datetime(df["Date"])
    df["dow_sin"] = np.sin(2 * np.pi * dt.dt.dayofweek / 7)
    df["dow_cos"] = np.cos(2 * np.pi * dt.dt.dayofweek / 7)
    df["month_sin"] = np.sin(2 * np.pi * dt.dt.month / 12)
    df["month_cos"] = np.cos(2 * np.pi * dt.dt.month / 12)

    # Lagged target (1-, 3-, 7-day lags)
    df["admissions_lag1"] = df[TARGET_COL].shift(1)
    df["admissions_lag3"] = df[TARGET_COL].shift(3)
    df["admissions_lag7"] = df[TARGET_COL].shift(7)

    lag_cols = (
        [f"{c}_lag1" for c in FEATURE_COLS] + ["PM_ratio_lag1", "admissions_lag1"]
        + [f"{c}_lag3" for c in FEATURE_COLS] + ["PM_ratio_lag3", "admissions_lag3"]
        + [f"{c}_lag7" for c in FEATURE_COLS] + ["PM_ratio_lag7", "admissions_lag7"]
    )
    df = df.dropna(how="any", subset=lag_cols)
    return df.reset_index(drop=True)


def get_feature_columns():
    """Names of columns used as model inputs."""
    base = FEATURE_COLS + ["PM_ratio"]
    lags = (
        [f"{c}_lag1" for c in FEATURE_COLS] + ["PM_ratio_lag1", "admissions_lag1"]
        + [f"{c}_lag3" for c in FEATURE_COLS] + ["PM_ratio_lag3", "admissions_lag3"]
        + [f"{c}_lag7" for c in FEATURE_COLS] + ["PM_ratio_lag7", "admissions_lag7"]
    )
    rolls = [
        "PM2.5 (µg/m³)_roll3", "PM2.5 (µg/m³)_roll7",
        "PM10 (µg/m³)_roll3", "PM10 (µg/m³)_roll7",
        "Index Value_roll3", "Index Value_roll7",
    ]
    time_cycl = ["dow_sin", "dow_cos", "month_sin", "month_cos"]
    return base + lags + rolls + time_cycl


def get_reduced_feature_columns():
    """Minimal set for generalization: lagged target (1,3,7) + key exposure + time."""
    return [
        "admissions_lag1",
        "admissions_lag3",
        "admissions_lag7",
        "PM2.5 (µg/m³)_roll7",
        "PM10 (µg/m³)_roll7",
        "Index Value_roll7",
        "dow_sin",
        "dow_cos",
        "month_sin",
        "month_cos",
    ]


def training_data_split(df, train_frac=0.70, val_frac=0.15, test_frac=0.15, feature_scaler="standard"):
    """
    Time-ordered split (no shuffle). Scales features on train only (no leakage).
    Best practice: StandardScaler for regression (zero mean, unit variance); tree models
    are scale-invariant. MinMaxScaler(0,1) optional for NN/SVR if needed.
    Returns X_train, X_val, X_test, y_train, y_val, y_test, scaler.
    """
    assert abs(train_frac + val_frac + test_frac - 1.0) < 1e-9
    feat_cols = get_feature_columns()
    X = df[feat_cols].astype(float)
    y = df[TARGET_COL]

    n = len(df)
    n_train = int(n * train_frac)
    n_val = int(n * val_frac)
    n_test = n - n_train - n_val

    X_train, X_val, X_test = X.iloc[:n_train], X.iloc[n_train : n_train + n_val], X.iloc[n_train + n_val :]
    y_train = y.iloc[:n_train]
    y_val = y.iloc[n_train : n_train + n_val]
    y_test = y.iloc[n_train + n_val :]

    scaler = MinMaxScaler(feature_range=(0, 1)) if feature_scaler == "minmax" else StandardScaler()
    X_train = pd.DataFrame(scaler.fit_transform(X_train), columns=feat_cols, index=X_train.index)
    X_val = pd.DataFrame(scaler.transform(X_val), columns=feat_cols, index=X_val.index)
    X_test = pd.DataFrame(scaler.transform(X_test), columns=feat_cols, index=X_test.index)

    return X_train, X_val, X_test, y_train, y_val, y_test, scaler


def evaluate_model(model, X, y, set_name=""):
    """Compute MAE, RMSE, R² for a given (X, y) set."""
    y_pred = model.predict(X)
    mae = mean_absolute_error(y, y_pred)
    rmse = np.sqrt(mean_squared_error(y, y_pred))
    r2 = r2_score(y, y_pred)
    return {"mae": mae, "rmse": rmse, "r2": r2, "y_pred": y_pred}


def dummy_regression_model():
    """
    Creates and returns a dummy regression model.
    """
    model = DummyRegressor(strategy='mean')
    return model

def random_forest_regression_model(random_state=42):
    return RandomForestRegressor(n_estimators=100, max_depth=10, random_state=random_state)


def xgboost_model(random_state=42):
    if not HAS_XGB:
        raise ImportError("xgboost is not installed. Install from root: pip install -r requirements.txt")
    return xgb.XGBRegressor(n_estimators=100, max_depth=5, learning_rate=0.1, random_state=random_state)


def fnn_model(random_state=42):
    return MLPRegressor(
        hidden_layer_sizes=(64, 32),
        activation="relu",
        solver="adam",
        max_iter=1000,
        random_state=random_state,
    )


def logistic_regression_model(random_state=42):
    """Ridge regression (linear model with L2) for regression task."""
    return Ridge(alpha=1.0, random_state=random_state)


def ridge_tuned_model(cv=5):
    """Ridge with alpha selected by cross-validation on train (best practice)."""
    return RidgeCV(alphas=np.logspace(-3, 2, 100), cv=cv)


class RidgeLogTarget:
    """Ridge fitted on log(1+y); predicts in original space via expm1 (best practice for count-like targets)."""

    def __init__(self):
        self.ridge_ = RidgeCV(alphas=np.logspace(-3, 2, 100), cv=5)

    def fit(self, X, y):
        self.ridge_.fit(X, np.log1p(y))
        return self

    def predict(self, X):
        return np.expm1(self.ridge_.predict(X))


def elastic_net_cv_model(cv=5):
    """ElasticNet with alpha and l1_ratio selected by CV (best practice for many features)."""
    return ElasticNetCV(cv=cv, l1_ratio=[0.1, 0.5, 0.9, 0.95, 1.0], alphas=np.logspace(-3, 1, 50))


class RidgeReduced:
    """Ridge on a small feature set (lagged target + exposure + time) for better generalization."""

    def __init__(self):
        self.cols_ = get_reduced_feature_columns()
        self.ridge_ = RidgeCV(alphas=np.logspace(-2, 2, 50), cv=5)

    def fit(self, X, y):
        self.ridge_.fit(X[self.cols_], y)
        return self

    def predict(self, X):
        return self.ridge_.predict(X[self.cols_])


def gradient_boosting_model(random_state=42):
    """Gradient boosting with regularization to generalize (max_depth, min_samples_leaf, subsample)."""
    return GradientBoostingRegressor(
        n_estimators=200,
        max_depth=4,
        learning_rate=0.05,
        min_samples_leaf=10,
        subsample=0.8,
        random_state=random_state,
    )


def svm_model(random_state=42):
    return SVR(kernel="rbf", C=1.0, epsilon=0.1)

def run_all_models(X_train, X_val, X_test, y_train, y_val, y_test):
    """Train each model and return list of (name, val_metrics, test_metrics)."""
    models = [
        ("Dummy (mean)", dummy_regression_model()),
        ("Ridge", logistic_regression_model()),
        ("Ridge (CV tuned)", ridge_tuned_model()),
        ("Ridge (log y)", RidgeLogTarget()),
        ("ElasticNet (CV)", elastic_net_cv_model()),
        ("Ridge (reduced feats)", RidgeReduced()),
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
        val_metrics = evaluate_model(model, X_val, y_val)
        test_metrics = evaluate_model(model, X_test, y_test)
        results.append((name, val_metrics, test_metrics))
    return results


def main():
    df = load_data()
    n_after_load = len(df)
    df = preprocess_data(df, impute_features=USE_IMPUTATION)
    n_after_preprocess = len(df)
    df = feature_engineering(df)
    n_after_fe = len(df)
    X_train, X_val, X_test, y_train, y_val, y_test, scaler = training_data_split(df)

    print(f"Rows: after load {n_after_load} → after preprocess {n_after_preprocess} (impute={USE_IMPUTATION}) → after feature_eng {n_after_fe}")
    print(f"Train size = {len(y_train)}, Val size = {len(y_val)}, Test size = {len(y_test)}\n")

    results = run_all_models(X_train, X_val, X_test, y_train, y_val, y_test)

    # Comparison table
    print("Model comparison (Validation | Test)")
    print("-" * 72)
    print(f"{'Model':<18} {'Val MAE':>8} {'Val RMSE':>8} {'Val R²':>8}  |  {'Test MAE':>8} {'Test RMSE':>8} {'Test R²':>8}")
    print("-" * 72)
    for name, val_m, test_m in results:
        print(
            f"{name:<18} {val_m['mae']:>8.4f} {val_m['rmse']:>8.4f} {val_m['r2']:>8.4f}  |  "
            f"{test_m['mae']:>8.4f} {test_m['rmse']:>8.4f} {test_m['r2']:>8.4f}"
        )
    print("-" * 72)

    best_name, _, best_test_m = max(results, key=lambda r: r[2]["r2"])
    print(f"\nBest model by Test R²: {best_name} (R² = {best_test_m['r2']:.4f})")


if __name__ == "__main__":
    main()
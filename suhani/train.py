"""
Predict Number of Admissions from past air quality and weather (all_features_data.csv).

Uses 1-, 3-, 7-day lags and 3- and 7-day rolling means for all air/weather features (no same-day),
plus time features and admission_lag7 (same weekday last week). Pipeline: load → preprocess
→ feature engineering → time-aware split → train models → report comparison and best by Test R².
"""

import os
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import ExtraTreesRegressor, GradientBoostingRegressor, HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import ElasticNetCV, RidgeCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.svm import SVR
import tensorflow as tf
from tensorflow import keras
from keras import layers

USE_IMPUTATION = True
WINSORIZE_QUANTILE = 0.99
TRAIN_FRAC, VAL_FRAC, TEST_FRAC = 0.80, 0.10, 0.10
USE_RANDOM_SPLIT = False
SPLIT_RANDOM_STATE = 42
SCALER_TYPE = "standard"

LAG_DAYS = [1, 3, 7]
ROLL_WINDOWS = [3, 7]
ADMISSION_LAG_7 = "admission_lag7"

TIMESTAMP_COL = "Timestamp"
DATE_COL = "Date"
TARGET_COL = "Number of Admissions"

TIME_FEATURES = ["dow_sin", "dow_cos", "month_sin", "month_cos", "weekend"]


def get_raw_feature_columns(df):
    """Numeric air/weather columns only (exclude date, target, and outcome-like columns)."""
    exclude = {DATE_COL, TARGET_COL}
    outcome_like = ["brought_dead", "admission"]
    return [
        c for c in df.columns
        if c not in exclude
        and pd.api.types.is_numeric_dtype(df[c])
        and not any(kw in c.lower() for kw in outcome_like)
    ]


def get_feature_columns(df):
    """Model input: time features + _lag1/_lag3/_lag7, _roll3/_roll7, and admission_lag7 (no same-day raw)."""
    time_cols = [c for c in TIME_FEATURES if c in df.columns]
    lag_roll = [
        c for c in df.columns
        if any(c.endswith(f"_lag{k}") for k in LAG_DAYS) or any(c.endswith(f"_roll{k}") for k in ROLL_WINDOWS)
    ]
    admission_cols = [ADMISSION_LAG_7] if ADMISSION_LAG_7 in df.columns else []
    return time_cols + lag_roll + admission_cols


def load_data():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    path = os.path.join(script_dir, "data", "all_features_data.csv")
    df = pd.read_csv(path)
    if TIMESTAMP_COL in df.columns:
        df[DATE_COL] = pd.to_datetime(df[TIMESTAMP_COL], errors="coerce")
        df = df.drop(columns=[TIMESTAMP_COL])
    return df


def preprocess_data(df, impute_features=False):
    df = df.sort_values(DATE_COL).copy()
    df = df.dropna(subset=[TARGET_COL])
    raw_cols = get_raw_feature_columns(df)
    if impute_features:
        for col in raw_cols:
            if col in df.columns and df[col].isna().any():
                df[col] = df[col].ffill().bfill()
    df = df.dropna(subset=raw_cols)
    q = df[TARGET_COL].quantile(WINSORIZE_QUANTILE)
    df.loc[df[TARGET_COL] > q, TARGET_COL] = q
    return df.reset_index(drop=True)


def feature_engineering(df):
    """Add time features, 1/3/7-day lags and 3/7-day rolling means for air/weather, plus admission_lag7."""
    df = df.sort_values(DATE_COL).reset_index(drop=True)
    dt = pd.to_datetime(df[DATE_COL])
    df["dow_sin"] = np.sin(2 * np.pi * dt.dt.dayofweek / 7)
    df["dow_cos"] = np.cos(2 * np.pi * dt.dt.dayofweek / 7)
    df["month_sin"] = np.sin(2 * np.pi * dt.dt.month / 12)
    df["month_cos"] = np.cos(2 * np.pi * dt.dt.month / 12)
    df["weekend"] = (dt.dt.dayofweek >= 5).astype(np.float64)

    raw_cols = get_raw_feature_columns(df)
    new_cols = {}
    lag_roll_cols = []
    for col in raw_cols:
        for k in LAG_DAYS:
            name = f"{col}_lag{k}"
            new_cols[name] = df[col].shift(k)
            lag_roll_cols.append(name)
        for w in ROLL_WINDOWS:
            name = f"{col}_roll{w}"
            new_cols[name] = df[col].rolling(w, min_periods=1).mean().shift(1)
            lag_roll_cols.append(name)
    new_cols[ADMISSION_LAG_7] = df[TARGET_COL].shift(7)
    lag_roll_cols.append(ADMISSION_LAG_7)
    df = pd.concat([df, pd.DataFrame(new_cols, index=df.index)], axis=1)
    df = df.dropna(how="any", subset=lag_roll_cols)
    return df.reset_index(drop=True)


def _time_aware_split_indices(n, train_frac, val_frac, test_frac):
    """Return (train_ix, val_ix, test_ix) as index arrays for chronological split."""
    n_train = int(n * train_frac)
    n_val = int(n * val_frac)
    train_ix = np.arange(0, n_train)
    val_ix = np.arange(n_train, n_train + n_val)
    test_ix = np.arange(n_train + n_val, n)
    return train_ix, val_ix, test_ix


def _random_split_indices(n, train_frac, val_frac, test_frac, random_state=None):
    """Return (train_ix, val_ix, test_ix) as index arrays for random split."""
    rng = np.random.default_rng(random_state if random_state is not None else SPLIT_RANDOM_STATE)
    perm = rng.permutation(n)
    n_train = int(n * train_frac)
    n_val = int(n * val_frac)
    train_ix = perm[:n_train]
    val_ix = perm[n_train : n_train + n_val]
    test_ix = perm[n_train + n_val :]
    return train_ix, val_ix, test_ix


def training_data_split(df, train_frac=None, val_frac=None, test_frac=None, use_random_split=None):
    train_frac = train_frac if train_frac is not None else TRAIN_FRAC
    val_frac = val_frac if val_frac is not None else VAL_FRAC
    test_frac = test_frac if test_frac is not None else TEST_FRAC
    use_random = use_random_split if use_random_split is not None else USE_RANDOM_SPLIT
    assert abs(train_frac + val_frac + test_frac - 1.0) < 1e-9
    feat_cols = [c for c in get_feature_columns(df) if c in df.columns]
    if not feat_cols:
        raise ValueError("No feature columns found in df.")
    X = df[feat_cols].astype(float)
    y = df[TARGET_COL]
    n = len(df)
    if use_random:
        train_ix, val_ix, test_ix = _random_split_indices(n, train_frac, val_frac, test_frac)
    else:
        train_ix, val_ix, test_ix = _time_aware_split_indices(n, train_frac, val_frac, test_frac)
    X_train = X.iloc[train_ix].copy()
    X_val = X.iloc[val_ix].copy()
    X_test = X.iloc[test_ix].copy()
    y_train = y.iloc[train_ix]
    y_val = y.iloc[val_ix]
    y_test = y.iloc[test_ix]
    scaler = RobustScaler() if SCALER_TYPE == "robust" else StandardScaler()
    X_train = pd.DataFrame(scaler.fit_transform(X_train), columns=feat_cols, index=X_train.index)
    X_val = pd.DataFrame(scaler.transform(X_val), columns=feat_cols, index=X_val.index)
    X_test = pd.DataFrame(scaler.transform(X_test), columns=feat_cols, index=X_test.index)
    return X_train, X_val, X_test, y_train, y_val, y_test, scaler


def evaluate_model(model, X, y):
    pred = model.predict(X)
    return {
        "mae": mean_absolute_error(y, pred),
        "rmse": np.sqrt(mean_squared_error(y, pred)),
        "r2": r2_score(y, pred),
    }


def dummy_regression_model():
    return DummyRegressor(strategy="mean")


def ridge_tuned_model(cv=5):
    return RidgeCV(alphas=np.logspace(-3, 2, 100), cv=cv)


class RidgeLogTarget:
    def __init__(self):
        self._model = RidgeCV(alphas=np.logspace(-3, 2, 100), cv=5)

    def fit(self, X, y):
        self._model.fit(X, np.log1p(y))
        return self

    def predict(self, X):
        return np.expm1(self._model.predict(X))


def elastic_net_cv_model(cv=5):
    return ElasticNetCV(cv=cv, l1_ratio=[0.1, 0.5, 0.9, 1.0], alphas=np.logspace(-3, 1, 50), max_iter=5000)


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
    return xgb.XGBRegressor(n_estimators=100, max_depth=5, learning_rate=0.1, random_state=random_state)


def hist_gradient_boosting_model(random_state=42):
    return HistGradientBoostingRegressor(max_iter=150, max_depth=6, learning_rate=0.05, min_samples_leaf=15, random_state=random_state)


def extra_trees_model(random_state=42):
    return ExtraTreesRegressor(n_estimators=150, max_depth=12, min_samples_leaf=5, random_state=random_state)


def _build_keras_model(n_features, params, seed=42):
    keras.backend.clear_session()
    tf.random.set_seed(seed)
    model = keras.Sequential()
    model.add(keras.Input(shape=(n_features,)))
    units = int(params.get("units", 64))
    n_layers = int(params.get("n_layers", 2))
    dropout = float(params.get("dropout", 0.2))
    l2 = float(params.get("l2", 1e-4))
    reg = keras.regularizers.L2(l2) if l2 > 0 else None
    for _ in range(n_layers):
        model.add(layers.Dense(units, activation="relu", kernel_initializer="he_normal", kernel_regularizer=reg))
        model.add(layers.BatchNormalization())
        model.add(layers.Dropout(dropout))
    model.add(layers.Dense(1, kernel_regularizer=reg))
    lr = float(params.get("lr", 1e-3))
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=lr), loss="mse", metrics=["mae"])
    return model


class KerasRegressorWrapper:
    def __init__(self, params=None, epochs=200, batch_size=32, patience=15, random_state=42):
        self.params = params or {"units": 64, "n_layers": 2, "dropout": 0.2, "lr": 1e-3}
        self.epochs = epochs
        self.batch_size = batch_size
        self.patience = patience
        self.random_state = random_state
        self.model_ = None

    def fit(self, X, y, X_val=None, y_val=None):
        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y, dtype=np.float32).reshape(-1, 1)
        if X_val is not None:
            X_val = np.asarray(X_val, dtype=np.float32)
            y_val = np.asarray(y_val, dtype=np.float32).reshape(-1, 1)
        self.model_ = _build_keras_model(X.shape[1], self.params, self.random_state)
        early = keras.callbacks.EarlyStopping(
            monitor="val_loss" if X_val is not None else "loss",
            patience=self.patience,
            restore_best_weights=True,
        )
        fit_kw = {"epochs": self.epochs, "batch_size": self.batch_size, "verbose": 0}
        if X_val is not None:
            fit_kw["validation_data"] = (X_val, y_val)
            fit_kw["callbacks"] = [early]
        self.model_.fit(X, y, **fit_kw)
        return self

    def predict(self, X):
        X = np.asarray(X, dtype=np.float32)
        return self.model_.predict(X, verbose=0).ravel()


def run_all_models(X_train, X_val, X_test, y_train, y_val, y_test):
    """Train each model; return results list."""
    models = [
        ("Dummy (mean)", dummy_regression_model()),
        ("Ridge (CV)", ridge_tuned_model()),
        ("Ridge (log y)", RidgeLogTarget()),
        ("ElasticNet (CV)", elastic_net_cv_model()),
        ("Random Forest", random_forest_regression_model()),
        ("Gradient Boosting", gradient_boosting_model()),
        ("Hist Gradient Boosting", hist_gradient_boosting_model()),
        ("Extra Trees", extra_trees_model()),
        ("FNN (MLP)", fnn_model()),
        ("SVR", svm_model()),
        ("XGBoost", xgboost_model()),
        ("Keras", KerasRegressorWrapper(epochs=200, patience=15)),
    ]
    results = []
    for name, model in models:
        if "Keras" in name:
            model.fit(X_train, y_train, X_val=X_val, y_val=y_val)
        else:
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
    print(f"  Features: {X_train.shape[1]} cols")
    print(f"  Split: {'random' if USE_RANDOM_SPLIT else 'time-aware (chronological)'}")
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

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
import optuna
import matplotlib.pyplot as plt
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegressionCV
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
)
from sklearn.svm import SVC
from sklearn.model_selection import TimeSeriesSplit
import warnings

warnings.filterwarnings("ignore")
optuna.logging.set_verbosity(optuna.logging.WARNING)

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

TIME_FEATURES = ["dow_sin", "dow_cos", "month_sin", "month_cos", "weekend", "month_regime", "year_regime"]# Extra configuration for Optuna tuning and classification
THRESHOLD = 20           # admissions threshold for binary classification
N_SPLITS = 5             # time-series CV splits 
N_TRIALS = 25            # Optuna trials per tunable model
RANDOM_STATE = 42
WIND_DIR_COL = "wind_direction_10m_dominant (°)"


def get_raw_feature_columns(df):
    """Numeric air/weather columns only (exclude date, target, wind direction, and outcome-like columns)."""
    exclude = {DATE_COL, TARGET_COL, WIND_DIR_COL}
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
    
    path = "/Users/suhaniagarwal/Downloads/all_features_data_changed_2.csv"
    if not os.path.exists(path):
        script_dir = os.path.dirname(os.path.abspath(__file__))
        path = os.path.join(script_dir, "all_features_data_changed_2.csv")

    df = pd.read_csv(path)

    # Ensure DATE_COL exists (from Timestamp or from Date) — regression uses Date only
    if TIMESTAMP_COL in df.columns:
        df[TIMESTAMP_COL] = pd.to_datetime(df[TIMESTAMP_COL], errors="coerce")
        df[DATE_COL] = df[TIMESTAMP_COL].dt.normalize()
    elif DATE_COL in df.columns:
        df[DATE_COL] = pd.to_datetime(df[DATE_COL], errors="coerce")

    # Humidity merge uses DATE_COL only (does not require Timestamp on df)
    _extra_path = "/Users/suhaniagarwal/Downloads/extra_weather_variables (csv).csv"
    _humidity_cols = [
        "relative_humidity_2m_mean (%)",
        "relative_humidity_2m_max (%)",
        "relative_humidity_2m_min (%)",
    ]
    if os.path.exists(_extra_path):
        _extra = pd.read_csv(_extra_path, skiprows=3, header=0)
        _to_add = [c for c in _humidity_cols if c in _extra.columns and c not in df.columns]
        if _to_add:
            _extra_sub = _extra[["time"] + _to_add].copy()
            _extra_sub["time"] = pd.to_datetime(_extra_sub["time"], errors="coerce")
            _extra_sub["_date"] = _extra_sub["time"].dt.normalize()
            df = df.merge(
                _extra_sub.drop(columns=["time"]).rename(columns={"_date": "_merge_date"}),
                left_on=DATE_COL,
                right_on="_merge_date",
                how="left",
            )
            df = df.drop(columns=["_merge_date"], errors="ignore")

    # Regression drop list
    drop_exact = [
        "wind_speed_10m_max (km/h)",
        "temperature_2m_min (°C)",
        "wind_gusts_10m_max (km/h)",
        "NO2 (µg/m³)",
        "Benzene (µg/m³)",
        "PM10 (µg/m³)",
        "NO (µg/m³)",
        "relative_humidity_2m_min (%)",
        "Toluene (µg/m³)",
        "Ozone (µg/m³)",
    ]
    df = df.drop(columns=[c for c in drop_exact if c in df.columns])

    nh3_cols = [c for c in df.columns if "NH3" in c]
    nox_cols = [c for c in df.columns if "NOx" in c]
    if nh3_cols or nox_cols:
        df = df.drop(columns=list(set(nh3_cols + nox_cols)))

    # Regression keeps only Date; drop Timestamp if present
    if TIMESTAMP_COL in df.columns:
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

    # Regime indicators (month, year trend)
    df["month_regime"] = dt.dt.month
    years = dt.dt.year
    year_order = {y: i + 1 for i, y in enumerate(sorted(years.unique()))}
    df["year_regime"] = years.map(year_order)

    # Wind direction encoding (cyclical; raw WIND_DIR_COL excluded from raw_cols)
    if WIND_DIR_COL in df.columns:
        df["wind_sin"] = np.sin(2 * np.pi * df[WIND_DIR_COL] / 360)
        df["wind_cos"] = np.cos(2 * np.pi * df[WIND_DIR_COL] / 360)

    raw_cols = get_raw_feature_columns(df)

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
    if isinstance(y, pd.DataFrame):
        y = y.squeeze(axis=1)
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

class XGBRegressorWrapper(xgb.XGBRegressor):
    """
    Thin wrapper around XGBRegressor to always work with NumPy arrays.
    This avoids pandas/xgboost dtype quirks.
    """

    def fit(self, X, y, *args, **kwargs):
        X_arr = np.asarray(X, dtype=np.float32)
        y_arr = np.asarray(y, dtype=np.float32).ravel()  # XGBoost expects 1D labels, not DataFrame/2D        
        return super().fit(X_arr, y_arr, *args, **kwargs)

    def predict(self, X, *args, **kwargs):
        X_arr = np.asarray(X, dtype=np.float32)
        return super().predict(X_arr, *args, **kwargs)


class _PredictorWrapper:
    """
    Simple wrapper so we can reuse evaluate_model with objects that only
    expose a predict(X) function.
    """

    def __init__(self, predict_fn):
        self.predict = predict_fn


def _build_keras_model_optuna(n_features, params, seed=RANDOM_STATE):
    keras.backend.clear_session()
    tf.random.set_seed(seed)
    model = keras.Sequential()
    model.add(keras.Input(shape=(n_features,)))
    reg = keras.regularizers.L2(params["l2"]) if params["l2"] > 0 else None
    for _ in range(params["n_layers"]):
        model.add(
            layers.Dense(
                params["units"],
                activation="relu",
                kernel_initializer="he_normal",
                kernel_regularizer=reg,
            )
        )
        model.add(layers.BatchNormalization())
        model.add(layers.Dropout(params["dropout"]))
    model.add(layers.Dense(1, kernel_regularizer=reg))
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=params["lr"]),
        loss="mse",
        metrics=["mae"],
    )
    return model


def tune_and_fit_keras_optuna(
    X_train, y_train, X_val, y_val, X_test, y_test, n_trials=N_TRIALS
):
    X_tr = np.asarray(X_train, dtype=np.float32)
    y_tr = np.asarray(y_train, dtype=np.float32).reshape(-1, 1)
    X_va = np.asarray(X_val, dtype=np.float32)
    y_va = np.asarray(y_val, dtype=np.float32).reshape(-1, 1)
    n_features = X_tr.shape[1]

    def objective(trial):
        params = {
            "units": trial.suggest_categorical("units", [16, 32, 64, 128]),
            "n_layers": trial.suggest_int("n_layers", 1, 3),
            "dropout": trial.suggest_float("dropout", 0.2, 0.5),
            "lr": trial.suggest_float("lr", 3e-4, 3e-3, log=True),
            "l2": trial.suggest_float("l2", 5e-5, 5e-3, log=True),
        }
        model = _build_keras_model_optuna(n_features, params, RANDOM_STATE)
        early = keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=15,
            restore_best_weights=True,
        )
        model.fit(
            X_tr,
            y_tr,
            validation_data=(X_va, y_va),
            epochs=250,
            batch_size=32,
            callbacks=[early],
            verbose=0,
        )
        pred = model.predict(X_va, verbose=0).ravel()
        return r2_score(y_val, pred)

    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=RANDOM_STATE, n_startup_trials=5),
    )
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

    best = study.best_params
    best["units"] = int(best["units"])
    best["n_layers"] = int(best["n_layers"])

    final = _build_keras_model_optuna(n_features, best, RANDOM_STATE)
    early = keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=20,
        restore_best_weights=True,
    )
    final.fit(
        X_tr,
        y_tr,
        validation_data=(X_va, y_va),
        epochs=250,
        batch_size=32,
        callbacks=[early],
        verbose=0,
    )

    wrap = _PredictorWrapper(
        lambda X: final.predict(np.asarray(X, dtype=np.float32), verbose=0).ravel()
    )
    return final, evaluate_model(wrap, X_val, y_val), evaluate_model(wrap, X_test, y_test)


def tune_and_fit_xgboost_optuna(
    X_train, y_train, X_val, y_val, X_test, y_test, n_trials=N_TRIALS
):
    def objective(trial):
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 50, 300),
            "max_depth": trial.suggest_int("max_depth", 3, 10),
            "learning_rate": trial.suggest_float(
                "learning_rate", 0.01, 0.2, log=True
            ),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float(
                "colsample_bytree", 0.6, 1.0
            ),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
        }
        model = XGBRegressorWrapper(**params, random_state=RANDOM_STATE)
        model.fit(X_train, y_train)
        pred = model.predict(X_val)
        return r2_score(y_val, pred)

    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=RANDOM_STATE, n_startup_trials=5),
    )
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    best = study.best_params
    model = XGBRegressorWrapper(**best, random_state=RANDOM_STATE)
    model.fit(X_train, y_train)
    return (
        model,
        evaluate_model(model, X_val, y_val),
        evaluate_model(model, X_test, y_test),
    )


def tune_and_fit_rf_optuna(
    X_train, y_train, X_val, y_val, X_test, y_test, n_trials=N_TRIALS
):
    def objective(trial):
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 50, 200),
            "max_depth": trial.suggest_int("max_depth", 5, 20),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 2, 20),
        }
        model = RandomForestRegressor(
            **params, random_state=RANDOM_STATE, n_jobs=-1
        )
        model.fit(X_train, y_train)
        return r2_score(y_val, model.predict(X_val))

    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=RANDOM_STATE, n_startup_trials=5),
    )
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    best = study.best_params
    model = RandomForestRegressor(
        **best, random_state=RANDOM_STATE, n_jobs=-1
    )
    model.fit(X_train, y_train)
    return (
        model,
        evaluate_model(model, X_val, y_val),
        evaluate_model(model, X_test, y_test),
    )

def tune_and_fit_gbm_optuna(X_train, y_train, X_val, y_val, X_test, y_test, n_trials=N_TRIALS):
    def objective(trial):
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 50, 300),
            "max_depth": trial.suggest_int("max_depth", 2, 8),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 5, 30),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        }
        model = GradientBoostingRegressor(**params, random_state=RANDOM_STATE)
        model.fit(X_train, y_train)
        return r2_score(y_val, model.predict(X_val))

    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=RANDOM_STATE, n_startup_trials=5),
    )
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    best = study.best_params
    model = GradientBoostingRegressor(**best, random_state=RANDOM_STATE)
    model.fit(X_train, y_train)
    return model, evaluate_model(model, X_val, y_val), evaluate_model(model, X_test, y_test)


def tune_and_fit_histgb_optuna(X_train, y_train, X_val, y_val, X_test, y_test, n_trials=N_TRIALS):
    def objective(trial):
        params = {
            "max_iter": trial.suggest_int("max_iter", 50, 250),
            "max_depth": trial.suggest_int("max_depth", 3, 10),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 5, 30),
        }
        model = HistGradientBoostingRegressor(**params, random_state=RANDOM_STATE)
        model.fit(X_train, y_train)
        return r2_score(y_val, model.predict(X_val))

    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=RANDOM_STATE, n_startup_trials=5),
    )
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    best = study.best_params
    model = HistGradientBoostingRegressor(**best, random_state=RANDOM_STATE)
    model.fit(X_train, y_train)
    return model, evaluate_model(model, X_val, y_val), evaluate_model(model, X_test, y_test)


def tune_and_fit_extra_trees_optuna(X_train, y_train, X_val, y_val, X_test, y_test, n_trials=N_TRIALS):
    def objective(trial):
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 50, 250),
            "max_depth": trial.suggest_int("max_depth", 5, 20),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 2, 15),
        }
        model = ExtraTreesRegressor(**params, random_state=RANDOM_STATE, n_jobs=-1)
        model.fit(X_train, y_train)
        return r2_score(y_val, model.predict(X_val))

    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=RANDOM_STATE, n_startup_trials=5),
    )
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    best = study.best_params
    model = ExtraTreesRegressor(**best, random_state=RANDOM_STATE, n_jobs=-1)
    model.fit(X_train, y_train)
    return model, evaluate_model(model, X_val, y_val), evaluate_model(model, X_test, y_test)


def tune_and_fit_svr_optuna(X_train, y_train, X_val, y_val, X_test, y_test, n_trials=N_TRIALS):
    def objective(trial):
        C = trial.suggest_float("C", 0.1, 100.0, log=True)
        epsilon = trial.suggest_float("epsilon", 0.01, 1.0)
        kernel = trial.suggest_categorical("kernel", ["rbf", "poly"])
        model = SVR(C=C, epsilon=epsilon, kernel=kernel)
        model.fit(X_train, y_train)
        return r2_score(y_val, model.predict(X_val))

    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=RANDOM_STATE, n_startup_trials=5),
    )
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    best = study.best_params
    model = SVR(**best)
    model.fit(X_train, y_train)
    return model, evaluate_model(model, X_val, y_val), evaluate_model(model, X_test, y_test)


def tune_and_fit_mlp_optuna(X_train, y_train, X_val, y_val, X_test, y_test, n_trials=N_TRIALS):
    def objective(trial):
        n_layers = trial.suggest_int("n_layers", 1, 3)
        units = []
        for i in range(n_layers):
            units.append(trial.suggest_int(f"units_{i}", 16, 128))
        alpha = trial.suggest_float("alpha", 1e-5, 1e-2, log=True)
        lr = trial.suggest_float("learning_rate_init", 1e-4, 1e-2, log=True)
        model = MLPRegressor(
            hidden_layer_sizes=tuple(units),
            activation="relu",
            solver="adam",
            alpha=alpha,
            learning_rate_init=lr,
            max_iter=1000,
            random_state=RANDOM_STATE,
        )
        model.fit(X_train, y_train)
        return r2_score(y_val, model.predict(X_val))

    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=RANDOM_STATE, n_startup_trials=5),
    )
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    best = study.best_params
    n_layers = best.pop("n_layers")
    units = [best.pop(f"units_{i}") for i in range(n_layers)]
    best["hidden_layer_sizes"] = tuple(units)
    best["activation"] = "relu"
    best["solver"] = "adam"
    best["max_iter"] = 1000
    best["random_state"] = RANDOM_STATE
    model = MLPRegressor(**best)
    model.fit(X_train, y_train)
    return model, evaluate_model(model, X_val, y_val), evaluate_model(model, X_test, y_test)

def run_optuna_models(X_train, X_val, X_test, y_train, y_val, y_test):
    results = []

    _, vm, tm = tune_and_fit_rf_optuna(X_train, y_train, X_val, y_val, X_test, y_test)
    results.append(("Random Forest (tuned)", vm, tm))

    _, vm, tm = tune_and_fit_gbm_optuna(X_train, y_train, X_val, y_val, X_test, y_test)
    results.append(("Gradient Boosting (tuned)", vm, tm))

    _, vm, tm = tune_and_fit_histgb_optuna(X_train, y_train, X_val, y_val, X_test, y_test)
    results.append(("Hist Gradient Boosting (tuned)", vm, tm))

    _, vm, tm = tune_and_fit_extra_trees_optuna(X_train, y_train, X_val, y_val, X_test, y_test)
    results.append(("Extra Trees (tuned)", vm, tm))

    _, vm, tm = tune_and_fit_xgboost_optuna(X_train, y_train, X_val, y_val, X_test, y_test)
    results.append(("XGBoost (tuned)", vm, tm))

    _, vm, tm = tune_and_fit_svr_optuna(X_train, y_train, X_val, y_val, X_test, y_test)
    results.append(("SVR (tuned)", vm, tm))

    _, vm, tm = tune_and_fit_mlp_optuna(X_train, y_train, X_val, y_val, X_test, y_test)
    results.append(("FNN (MLP) (tuned)", vm, tm))

    _, vm, tm = tune_and_fit_keras_optuna(X_train, y_train, X_val, y_val, X_test, y_test)
    results.append(("Keras (tuned)", vm, tm))

    return results
def run_regression_time_series_cv(df, n_splits=N_SPLITS):
    """
    5-fold time-series CV for regression, using the same feature set
    and models as the single-split pipeline (including Optuna models).
    """
    feat_cols = [c for c in get_feature_columns(df) if c in df.columns]
    if not feat_cols:
        raise ValueError("No feature columns found in df for CV.")

    X = df[feat_cols].astype(float)
    y = df[TARGET_COL]
    if isinstance(y, pd.DataFrame):
        y = y.squeeze(axis=1)

    tscv = TimeSeriesSplit(n_splits=n_splits)
    results_per_model = {}  # name -> {"val": [dict], "test": [dict]}

    print(f"\nRegression: {n_splits}-fold time-series CV\n")

    for fold, (train_idx, test_idx) in enumerate(tscv.split(X), start=1):
        print(f"Fold {fold}/{n_splits}...")

        X_train_full, X_test = X.iloc[train_idx].copy(), X.iloc[test_idx].copy()
        y_train_full, y_test = y.iloc[train_idx], y.iloc[test_idx]

        # Time-aware split inside the training chunk: first 80% train, last 20% val
        split = int(len(X_train_full) * 0.8)
        X_train = X_train_full.iloc[:split].copy()
        X_val = X_train_full.iloc[split:].copy()
        y_train = y_train_full.iloc[:split]
        y_val = y_train_full.iloc[split:]

        scaler = RobustScaler() if SCALER_TYPE == "robust" else StandardScaler()
        X_train_s = pd.DataFrame(
            scaler.fit_transform(X_train), columns=feat_cols, index=X_train.index
        )
        X_val_s = pd.DataFrame(
            scaler.transform(X_val), columns=feat_cols, index=X_val.index
        )
        X_test_s = pd.DataFrame(
            scaler.transform(X_test), columns=feat_cols, index=X_test.index
        )

        # Base models (no Optuna) on this fold
        fold_results = run_all_models(
            X_train_s, X_val_s, X_test_s, y_train, y_val, y_test
        )

        # Optuna-tuned models on this fold
        optuna_results = run_optuna_models(
            X_train_s, X_val_s, X_test_s, y_train, y_val, y_test
        )
        fold_results.extend(optuna_results)

        # Accumulate metrics
        for name, vm, tm in fold_results:
            if name not in results_per_model:
                results_per_model[name] = {"val": [], "test": []}
            results_per_model[name]["val"].append(vm)
            results_per_model[name]["test"].append(tm)

    # Average metrics across folds
    avg_results = []
    for name, metrics_lists in results_per_model.items():
        val_ms = metrics_lists["val"]
        test_ms = metrics_lists["test"]

        val_avg = {
            "mae": np.mean([m["mae"] for m in val_ms]),
            "rmse": np.mean([m["rmse"] for m in val_ms]),
            "r2": np.mean([m["r2"] for m in val_ms]),
        }
        test_avg = {
            "mae": np.mean([m["mae"] for m in test_ms]),
            "rmse": np.mean([m["rmse"] for m in test_ms]),
            "r2": np.mean([m["r2"] for m in test_ms]),
        }
        avg_results.append((name, val_avg, test_avg))

    # Print summary table
    print("\nRegression model comparison (5-fold CV: Validation | Test)")
    print("-" * 72)
    print(
        f"{'Model':<28} {'Val MAE':>8} {'Val RMSE':>8} {'Val R²':>8}  |  "
        f"{'Test MAE':>8} {'Test RMSE':>8} {'Test R²':>8}"
    )
    print("-" * 72)
    for name, vm, tm in avg_results:
        print(
            f"{name:<28} "
            f"{vm['mae']:>8.4f} {vm['rmse']:>8.4f} {vm['r2']:>8.4f}  |  "
            f"{tm['mae']:>8.4f} {tm['rmse']:>8.4f} {tm['r2']:>8.4f}"
        )
    print("-" * 72)

    best_name, _, best_test = max(avg_results, key=lambda r: r[2]["r2"])
    print(f"\nBest by CV Test R²: {best_name}  (R² = {best_test['r2']:.4f})")

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
        ("XGBoost", XGBRegressorWrapper(n_estimators=100, max_depth=5, learning_rate=0.1, random_state=RANDOM_STATE)),
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
def run_classification_pipeline():
    """
    
    Uses engineered lags/rolls and humidity-merged dataset to predict
    whether admissions >= THRESHOLD.
    """
    # --- 1. FEATURE ENGINEERING ---
    def engineer_features(df):
        df = df.copy()
        df["Timestamp"] = pd.to_datetime(df["Timestamp"])
        df = df.sort_values("Timestamp").reset_index(drop=True)

        # Cyclical seasonality
        df["dow_sin"] = np.sin(2 * np.pi * df["Timestamp"].dt.dayofweek / 7)
        df["dow_cos"] = np.cos(2 * np.pi * df["Timestamp"].dt.dayofweek / 7)
        df["month_sin"] = np.sin(2 * np.pi * df["Timestamp"].dt.month / 12)
        df["month_cos"] = np.cos(2 * np.pi * df["Timestamp"].dt.month / 12)

        # Regime indicators for seasonality and trend
        df["month_regime"] = df["Timestamp"].dt.month
        years = df["Timestamp"].dt.year
        year_order = {year: i + 1 for i, year in enumerate(sorted(years.unique()))}
        df["year_regime"] = years.map(year_order)
        df["weekend"] = (df["Timestamp"].dt.dayofweek >= 5).astype(np.float64)
        
        if WIND_DIR_COL in df.columns:
            df["wind_sin"] = np.sin(2 * np.pi * df[WIND_DIR_COL] / 360)
            df["wind_cos"] = np.cos(2 * np.pi * df[WIND_DIR_COL] / 360)

        exclude = ["Timestamp", TARGET_COL, WIND_DIR_COL, "Date"]
        weather_cols = [
            c
            for c in df.columns
            if c not in exclude and pd.api.types.is_numeric_dtype(df[c])
        ]

        new_feats = {}
        for col in weather_cols:
            for lag in LAG_DAYS:
                new_feats[f"{col}_lag{lag}"] = df[col].shift(lag)
            for roll in ROLL_WINDOWS:
                new_feats[f"{col}_roll{roll}"] = df[col].shift(1).rolling(roll).mean()

        new_feats["admission_lag7"] = df[TARGET_COL].shift(7)

        df = pd.concat([df, pd.DataFrame(new_feats)], axis=1)

        all_nan_cols = [c for c in df.columns if df[c].isna().all()]
        if all_nan_cols:
            df = df.drop(columns=all_nan_cols)

        df = df.dropna().reset_index(drop=True)
        df["target_binary"] = (df[TARGET_COL] >= THRESHOLD).astype(int)
        return df

    # --- 2. TUNING LOGIC ---
    def get_optuna_params(model_type, X_tr, y_tr, X_va, y_va):
        def objective(trial):
            if model_type == "rf":
                p = {
                    "n_estimators": trial.suggest_int("n_estimators", 50, 200),
                    "max_depth": trial.suggest_int("max_depth", 5, 20),
                }
                model = RandomForestClassifier(
                    **p, random_state=RANDOM_STATE, n_jobs=-1
                )
            elif model_type == "xgb":
                p = {
                    "n_estimators": trial.suggest_int("n_estimators", 50, 300),
                    "max_depth": trial.suggest_int("max_depth", 3, 10),
                    "learning_rate": trial.suggest_float(
                        "learning_rate", 0.01, 0.2, log=True
                    ),
                }
                model = xgb.XGBClassifier(
                    **p, random_state=RANDOM_STATE, eval_metric="logloss"
                )
                        
            elif model_type == "svm":
                p = {
                    "C": trial.suggest_float("C", 0.01, 100.0, log=True),
                    "kernel": trial.suggest_categorical("kernel", ["rbf", "poly", "linear"]),
                    "gamma": trial.suggest_float("gamma", 1e-4, 10.0, log=True),
                    "degree": trial.suggest_int("degree", 2, 5),
                }
                model = SVC(probability=True, **p, random_state=RANDOM_STATE)
            else:
                raise ValueError(f"Unknown model_type: {model_type}")
           

            model.fit(X_tr, y_tr)
            return f1_score(y_va, model.predict(X_va))

        study = optuna.create_study(direction="maximize")
        study.optimize(objective, n_trials=N_TRIALS)
        return study.best_params

    # --- 3. DATA LOADING & MERGE (WITH HUMIDITY) ---
    # This version drops low-importance features + some pollutants 
    def load_merged_data():
        base_df = pd.read_csv(
            "/Users/suhaniagarwal/Downloads/all_features_data_changed_2.csv"
        )

        extra_weather = pd.read_csv(
            "/Users/suhaniagarwal/Downloads/extra_weather_variables (csv).csv",
            skiprows=3,
            header=0,
        )

        # Merge all 3 humidity columns from extra weather
        humidity_cols = [
            "time",
            "relative_humidity_2m_mean (%)",
            "relative_humidity_2m_max (%)",
            "relative_humidity_2m_min (%)",
        ]
        humidity_cols = [c for c in humidity_cols if c in extra_weather.columns]
        if not humidity_cols:
            humidity_cols = ["time"]
        extra_humidity = extra_weather[humidity_cols].copy()

        # Align date types for merge
        base_df["Timestamp"] = pd.to_datetime(base_df["Timestamp"])
        extra_humidity["time"] = pd.to_datetime(extra_humidity["time"])

        merged = base_df.merge(
            extra_humidity,
            left_on="Timestamp",
            right_on="time",
            how="left",
        )

        # Drop the redundant merge key from extra_weather
        merged = merged.drop(columns=["time"])

        # Drop NOx as redundant nitrogen measure
        if "NOx (ppb)" in merged.columns:
            merged = merged.drop(columns=["NOx (ppb)"])

        # Drop NH3
        nh3_cols = [c for c in merged.columns if "NH3" in c]
        if nh3_cols:
            merged = merged.drop(columns=nh3_cols)

        # Drop specified columns (only if present)
        drop_cols = [
            "O Xylene (µg/m³)",
            "Eth-Benzene (µg/m³)",
            "MP-Xylene (µg/m³)",
            "AT (°C)",
            "RH (%)",
            "WS (m/s)",
            "WD (deg)",
            "RF (mm)",
            "TOT-RF (mm)",
            "SR (W/mt2)",
            "BP (mmHg)",
            "VWS (m/s)",
            "Air Quality",
            "Xylene (µg/m³)",
            "SO2 (µg/m³)",
            "Benzene (µg/m³)",
            "wind_gusts_10m_max (km/h)",
            "wind_speed_10m_max (km/h)",
            "apparent_temperature_max (°C)",
        ]
        merged = merged.drop(columns=[c for c in drop_cols if c in merged.columns])

        return merged

    # --- 4. MAIN EVALUATION LOOP ---
    df = engineer_features(load_merged_data())
    feat_cols = [
        c
        for c in df.columns
        if any(s in c for s in ["_lag", "_roll", "sin", "cos", "weekend", "regime"])
    ]
    X, y = df[feat_cols], df["target_binary"]

    print(
        f"Target distribution: {dict(y.value_counts())} "
        f"(1 = admissions >= {THRESHOLD})"
    )

    tscv = TimeSeriesSplit(n_splits=N_SPLITS)
    fold_results = []
    best_params = {}

    last_y_test, last_y_preds = None, None

    for fold, (train_idx, test_idx) in enumerate(tscv.split(X)):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        scaler = RobustScaler()
        X_train_s = scaler.fit_transform(X_train)
        X_test_s = scaler.transform(X_test)

        if fold == 0:
            print(f"Tuning Classification Models (Threshold: {THRESHOLD})...")
            split = int(len(X_train_s) * 0.8)
            for m in ["rf", "xgb", "svm"]:
                best_params[m] = get_optuna_params(
                    m,
                    X_train_s[:split],
                    y_train.iloc[:split],
                    X_train_s[split:],
                    y_train.iloc[split:],
                )

        models = {
            "Dummy_Baseline": DummyClassifier(strategy="most_frequent"),
            "LogisticRegression": LogisticRegressionCV(cv=5),
            "RandomForest": RandomForestClassifier(
                **best_params["rf"], random_state=RANDOM_STATE
            ),
            "XGBoost": xgb.XGBClassifier(
                **best_params["xgb"],
                random_state=RANDOM_STATE,
                eval_metric="logloss",
            ),
            "SVM": SVC(probability=True, **best_params["svm"], random_state=RANDOM_STATE),
        }

        scores = {}
        for name, model in models.items():
            model.fit(X_train_s, y_train)
            preds = model.predict(X_test_s)

            scores[f"{name}_Accuracy"] = accuracy_score(y_test, preds)
            scores[f"{name}_F1"] = f1_score(y_test, preds)

            if hasattr(model, "predict_proba"):
                probs = model.predict_proba(X_test_s)[:, 1]
                scores[f"{name}_AUC"] = roc_auc_score(y_test, probs)

            if fold == N_SPLITS - 1 and name == "XGBoost":
                last_y_test = y_test
                last_y_preds = preds

        fold_results.append(scores)
        print(f"Fold {fold + 1} complete.")

    # --- 5. OUTPUT ---
    print(f"\nAverage Classification Metrics (Threshold >= {THRESHOLD}):")
    print(pd.DataFrame(fold_results).mean())

    cm = confusion_matrix(last_y_test, last_y_preds)
    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm, display_labels=[f"< {THRESHOLD}", f">= {THRESHOLD}"]
    )
    disp.plot(cmap=plt.cm.Blues)
    plt.title(
        f"XGBoost Confusion Matrix (Fold {N_SPLITS})\n"
        f"Target: Admissions >= {THRESHOLD}"
    )
    plt.show()

def main():
    # Load + preprocess + feature engineering once
    df = load_data()
    n_load = len(df)
    df = preprocess_data(df, impute_features=USE_IMPUTATION)
    n_pre = len(df)
    df = feature_engineering(df)
    n_fe = len(df)

    print("Data: load → preprocess → feature_eng")
    print(f"  Rows: {n_load} → {n_pre} (impute={USE_IMPUTATION}) → {n_fe}")
    print(f"  Features after FE: {len(get_feature_columns(df))} cols")
    print(f"  Time-series CV folds: {N_SPLITS}\n")

    # 5-fold time-aware CV for regression (includes Optuna-tuned models)
    run_regression_time_series_cv(df)


if __name__ == "__main__":
    # Regression (with Optuna-tuned models)
    main()

    # Classification (binary, threshold = THRESHOLD)

    run_classification_pipeline()

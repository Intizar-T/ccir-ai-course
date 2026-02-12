"""
Brazil: predict hospitalizations from pollutants (pollutants_and_hosp.csv).

Data: date (Data), 8 SIH hospitalization columns (summed → target), 12 pollutant columns
(pm25/pm10/so2/no2/co/o3 for GV and Sul). Pipeline: load → preprocess → feature engineering
→ time-ordered split → scale → train models → report comparison and best Test R².
"""

import os
import numpy as np
import pandas as pd
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import ExtraTreesRegressor, GradientBoostingRegressor, HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import ElasticNetCV, Ridge, RidgeCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import RobustScaler, StandardScaler
from sklearn.svm import SVR

import tensorflow as tf
from tensorflow import keras
from keras import layers
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials

try:
    import xgboost as xgb
    HAS_XGB = True
except ImportError:
    HAS_XGB = False



# ---------------------------------------------------------------------------
# Configuration (Brazil: pollutants_and_hosp.csv)
# ---------------------------------------------------------------------------

USE_IMPUTATION = True
WINSORIZE_QUANTILE = 0.99
TRAIN_FRAC, VAL_FRAC, TEST_FRAC = 0.80, 0.10, 0.10
USE_EXTRA_FEATURES = True
SCALER_TYPE = "standard"  # "standard" | "robust"
KERAS_TRIALS = 30

DATA_FILE = "pollutants_and_hosp.csv"
DATE_COL = "Date"  # after renaming "Data"
TARGET_COL = "hospitalizations"  # sum of SIH_* columns

# Hospitalization count columns to sum into target
SIH_COLS = [
    "SIH_Sul_I_Old", "SIH_Sul_J_Old", "SIH_Sul_I_Kid", "SIH_Sul_J_Kid",
    "SIH_GV_I_Old", "SIH_GV_J_Old", "SIH_GV_I_Kid", "SIH_GV_J_Kid",
]

# Pollutant features (GV = Grande Vitória, Sul = Sul region)
FEATURE_COLS = [
    "pm25.gv", "pm10.gv", "so2.gv", "no2.gv", "co.gv", "o3.gv",
    "co.sul", "o3.sul", "no2.sul", "so2.sul", "pm10.sul", "pm25.sul",
]

# For PM_ratio and rolling (must be in FEATURE_COLS)
PM25_COL = "pm25.gv"
PM10_COL = "pm10.gv"
ROLLING_COLS = ["pm25.gv", "pm10.gv", "no2.gv"]


# ---------------------------------------------------------------------------
# Load & preprocess
# ---------------------------------------------------------------------------

def load_data():
    """Load Brazil CSV, parse date, build target as sum of SIH columns, keep pollutants."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    path = os.path.join(script_dir, "data", DATA_FILE)
    df = pd.read_csv(path)
    df = df.rename(columns={"Data": DATE_COL})
    df[DATE_COL] = pd.to_datetime(df[DATE_COL], dayfirst=True, errors="coerce")
    df = df.dropna(subset=[DATE_COL])
    # Target = total hospitalizations (sum of 8 SIH columns)
    available_sih = [c for c in SIH_COLS if c in df.columns]
    df[TARGET_COL] = df[available_sih].sum(axis=1)
    keep = [DATE_COL] + [c for c in FEATURE_COLS if c in df.columns] + [TARGET_COL]
    df = df[keep].copy()
    return df


def preprocess_data(df, impute_features=False):
    """Drop rows with missing target; optionally impute or drop missing features. Winsorize target at 99th percentile."""
    df = df.sort_values(DATE_COL).copy()
    df = df.dropna(subset=[TARGET_COL])
    if impute_features:
        for col in FEATURE_COLS:
            if col in df.columns and df[col].isna().any():
                df[col] = df[col].ffill().bfill()
        df = df.dropna(subset=[c for c in FEATURE_COLS if c in df.columns])
    else:
        df = df.dropna(subset=[c for c in FEATURE_COLS if c in df.columns])
    q = df[TARGET_COL].quantile(WINSORIZE_QUANTILE)
    df.loc[df[TARGET_COL] > q, TARGET_COL] = q
    return df.reset_index(drop=True)


# ---------------------------------------------------------------------------
# Feature engineering
# ---------------------------------------------------------------------------

def feature_engineering(df):
    """Add PM ratio, 1/3/7/14-day lags, 3/7/14-day rolling means, cyclical time, weekend. Drop rows with NaN from lags."""
    df = df.sort_values(DATE_COL).reset_index(drop=True)
    base_cols = [c for c in FEATURE_COLS if c in df.columns]

    if PM25_COL in df.columns and PM10_COL in df.columns:
        pm25, pm10 = df[PM25_COL], df[PM10_COL]
        df["PM_ratio"] = np.where(pm10 > 0, pm25 / pm10, np.nan)
        df["PM_ratio"] = df["PM_ratio"].fillna(df["PM_ratio"].median())
        base_cols = base_cols + ["PM_ratio"]

    for col in base_cols:
        for lag in (1, 3, 7, 14):
            df[f"{col}_lag{lag}"] = df[col].shift(lag)

    for col in [c for c in ROLLING_COLS if c in df.columns]:
        for w in (3, 7, 14):
            df[f"{col}_roll{w}"] = df[col].rolling(w, min_periods=1).mean()
            if USE_EXTRA_FEATURES and w >= 7:
                df[f"{col}_roll{w}_std"] = df[col].rolling(w, min_periods=1).std().fillna(0)

    if USE_EXTRA_FEATURES:
        for col in [c for c in ROLLING_COLS if c in df.columns]:
            r7 = df[col].rolling(7, min_periods=1).mean()
            df[f"{col}_trend7"] = r7.diff().fillna(0)

    dt = pd.to_datetime(df[DATE_COL])
    df["dow_sin"] = np.sin(2 * np.pi * dt.dt.dayofweek / 7)
    df["dow_cos"] = np.cos(2 * np.pi * dt.dt.dayofweek / 7)
    df["month_sin"] = np.sin(2 * np.pi * dt.dt.month / 12)
    df["month_cos"] = np.cos(2 * np.pi * dt.dt.month / 12)
    df["weekend"] = (dt.dt.dayofweek >= 5).astype(np.float64)
    if USE_EXTRA_FEATURES:
        df["day_of_year_sin"] = np.sin(2 * np.pi * dt.dt.dayofyear / 365.25)
        df["day_of_year_cos"] = np.cos(2 * np.pi * dt.dt.dayofyear / 365.25)

    for lag in (1, 3, 7, 14):
        df[f"admissions_lag{lag}"] = df[TARGET_COL].shift(lag)

    lag_cols = [f"{c}_lag{lag}" for c in base_cols for lag in (1, 3, 7, 14)]
    lag_cols += [f"admissions_lag{lag}" for lag in (1, 3, 7, 14)]
    lag_cols = [c for c in lag_cols if c in df.columns]
    df = df.dropna(how="any", subset=lag_cols)
    return df.reset_index(drop=True)


def get_feature_columns():
    """List of feature column names (only those present in df are used in split)."""
    base = [c for c in FEATURE_COLS] + ["PM_ratio"]
    lags = [f"{c}_lag{lag}" for c in base for lag in (1, 3, 7, 14)]
    lags += [f"admissions_lag{lag}" for lag in (1, 3, 7, 14)]
    rolls = [f"{r}_roll{w}" for r in ROLLING_COLS for w in (3, 7, 14)]
    time_ = ["dow_sin", "dow_cos", "month_sin", "month_cos", "weekend"]
    out = base + lags + rolls + time_
    if USE_EXTRA_FEATURES:
        rolls_std = [f"{r}_roll{w}_std" for r in ROLLING_COLS for w in (7, 14)]
        trends = [f"{r}_trend7" for r in ROLLING_COLS]
        out = out + rolls_std + trends + ["day_of_year_sin", "day_of_year_cos"]
    return out


def get_reduced_feature_columns():
    """Reduced set: admission lags + key rolling + time (for Ridge reduced)."""
    return [
        "admissions_lag1", "admissions_lag3", "admissions_lag7", "admissions_lag14",
        "pm25.gv_roll7", "pm25.gv_roll14", "pm10.gv_roll7", "no2.gv_roll7",
        "dow_sin", "dow_cos", "month_sin", "month_cos", "weekend",
    ]


# ---------------------------------------------------------------------------
# Split & scale
# ---------------------------------------------------------------------------

def training_data_split(df, train_frac=None, val_frac=None, test_frac=None):
    """Time-ordered split. Scale features with scaler fitted on train only."""
    train_frac = train_frac if train_frac is not None else TRAIN_FRAC
    val_frac = val_frac if val_frac is not None else VAL_FRAC
    test_frac = test_frac if test_frac is not None else TEST_FRAC
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
    scaler = RobustScaler() if SCALER_TYPE == "robust" else StandardScaler()
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


def hist_gradient_boosting_model(random_state=42):
    return HistGradientBoostingRegressor(max_iter=150, max_depth=6, learning_rate=0.05, min_samples_leaf=15, random_state=random_state)


def extra_trees_model(random_state=42):
    return ExtraTreesRegressor(n_estimators=150, max_depth=12, min_samples_leaf=5, random_state=random_state)


# ---------------------------------------------------------------------------
# Keras NN (TensorFlow) and Hyperopt tuning
# ---------------------------------------------------------------------------

def _build_keras_model(n_features, params, seed=42):
    """Build and compile a Keras regression MLP. params: units, n_layers, dropout, lr, l2."""
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
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=lr),
        loss="mse",
        metrics=["mae"],
    )
    return model


class KerasRegressorWrapper:
    """Sklearn-style wrapper for Keras regression (fit/predict)."""

    def __init__(self, params=None, epochs=200, batch_size=32, patience=15, random_state=42):
        self.params = params or {"units": 64, "n_layers": 2, "dropout": 0.2, "lr": 1e-3}
        self.epochs = epochs
        self.batch_size = batch_size
        self.patience = patience
        self.random_state = random_state
        self.model_ = None
        self.scaler_y_ = None

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


def run_keras_hyperopt(X_train, y_train, X_val, y_val, X_test, y_test, max_evals=15, random_state=42):
    """Tune Keras NN with Hyperopt (TPE); return (wrapper, test_metrics)."""
    X_tr = np.asarray(X_train, dtype=np.float32)
    y_tr = np.asarray(y_train, dtype=np.float32).reshape(-1, 1)
    X_va = np.asarray(X_val, dtype=np.float32)
    y_va = np.asarray(y_val, dtype=np.float32).reshape(-1, 1)
    n_features = X_tr.shape[1]
    units_choices = [16, 32, 64, 128]
    n_layers_choices = [1, 2, 3]
    space = {
        "units": hp.choice("units", units_choices),
        "n_layers": hp.choice("n_layers", n_layers_choices),
        "dropout": hp.uniform("dropout", 0.2, 0.5),
        "lr": hp.loguniform("lr", np.log(3e-4), np.log(3e-3)),
        "l2": hp.loguniform("l2", np.log(5e-5), np.log(5e-3)),
    }

    def _get_choice(choices, idx):
        if idx in choices:
            return idx
        i = int(idx) if isinstance(idx, (int, np.integer)) else 0
        i = min(max(0, i), len(choices) - 1)
        return choices[i]

    def objective(pp):
        params = {
            "units": _get_choice(units_choices, pp["units"]),
            "n_layers": _get_choice(n_layers_choices, pp["n_layers"]),
            "dropout": float(pp["dropout"]),
            "lr": float(pp["lr"]),
            "l2": float(pp["l2"]),
        }
        model = _build_keras_model(n_features, params, random_state)
        early = keras.callbacks.EarlyStopping(monitor="val_loss", patience=15, restore_best_weights=True)
        model.fit(X_tr, y_tr, validation_data=(X_va, y_va), epochs=250, batch_size=32, callbacks=[early], verbose=0)
        pred = model.predict(X_va, verbose=0).ravel()
        return {"loss": -r2_score(y_val, pred), "status": STATUS_OK}

    trials = Trials()
    best = fmin(objective, space, algo=tpe.suggest, max_evals=max_evals, rstate=np.random.default_rng(random_state), verbose=0)
    best_params = {
        "units": _get_choice(units_choices, best["units"]),
        "n_layers": _get_choice(n_layers_choices, best["n_layers"]),
        "dropout": best["dropout"],
        "lr": best["lr"],
        "l2": best["l2"],
    }
    wrapper = KerasRegressorWrapper(params=best_params, epochs=250, batch_size=32, patience=20, random_state=random_state)
    wrapper.fit(X_train, y_train, X_val=X_val, y_val=y_val)
    test_metrics = evaluate_model(wrapper, X_test, y_test)
    return wrapper, test_metrics


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
        ("Hist Gradient Boosting", hist_gradient_boosting_model()),
        ("Extra Trees", extra_trees_model()),
        ("FNN (MLP)", fnn_model()),
        ("SVR", svm_model()),
    ]
    if HAS_XGB:
        models.append(("XGBoost", xgboost_model()))

    models.append(("Keras (default)", KerasRegressorWrapper(epochs=200, patience=15)))
    results = []
    for name, model in models:
        if "Keras" in name and "tuned" not in name:
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

    print("Brazil: pollutants_and_hosp.csv")
    print("Data: load → preprocess → feature_eng")
    print(f"  Rows: {n_load} → {n_pre} (impute={USE_IMPUTATION}) → {n_fe}")
    print(f"  Train / Val / Test: {len(y_train)} / {len(y_val)} / {len(y_test)}\n")

    results = run_all_models(X_train, X_val, X_test, y_train, y_val, y_test)

    reduced_cols = [c for c in get_reduced_feature_columns() if c in X_train.columns]
    KERAS_USE_REDUCED = True
    if reduced_cols and KERAS_USE_REDUCED:
        X_train_r, X_val_r, X_test_r = X_train[reduced_cols], X_val[reduced_cols], X_test[reduced_cols]
        print(f"Keras (Hyperopt tuning on reduced features, {KERAS_TRIALS} trials)...")
        keras_wrapper, keras_tuned_test = run_keras_hyperopt(
            X_train_r, y_train, X_val_r, y_val, X_test_r, y_test, max_evals=KERAS_TRIALS
        )
        keras_val = evaluate_model(keras_wrapper, X_val_r, y_val)
    else:
        print(f"Keras (Hyperopt tuning full features, {KERAS_TRIALS} trials)...")
        keras_wrapper, keras_tuned_test = run_keras_hyperopt(
            X_train, y_train, X_val, y_val, X_test, y_test, max_evals=KERAS_TRIALS
        )
        keras_val = evaluate_model(keras_wrapper, X_val, y_val)
        reduced_cols = list(X_train.columns)
    results.append(("Keras (tuned)", keras_val, keras_tuned_test))

    ridge_r = RidgeReduced()
    ridge_r.fit(X_train, y_train)
    enet = elastic_net_cv_model()
    enet.fit(X_train, y_train)
    gb = gradient_boosting_model()
    gb.fit(X_train, y_train)
    hist_gb = hist_gradient_boosting_model()
    hist_gb.fit(X_train, y_train)
    extra_trees = extra_trees_model()
    extra_trees.fit(X_train, y_train)
    xgb_m = None
    if HAS_XGB:
        xgb_m = xgboost_model()
        xgb_m.fit(X_train, y_train)
    ridge_log = RidgeLogTarget()
    ridge_log.fit(X_train, y_train)

    X_val_k = X_val[reduced_cols] if reduced_cols else X_val
    p_val_r = ridge_r.predict(X_val).reshape(-1, 1)
    p_val_e = enet.predict(X_val).reshape(-1, 1)
    p_val_g = gb.predict(X_val).reshape(-1, 1)
    p_val_h = hist_gb.predict(X_val).reshape(-1, 1)
    p_val_et = extra_trees.predict(X_val).reshape(-1, 1)
    p_val_k = keras_wrapper.predict(X_val_k).reshape(-1, 1)
    p_val_x = xgb_m.predict(X_val).reshape(-1, 1) if HAS_XGB else None
    p_val_log = ridge_log.predict(X_val).reshape(-1, 1)
    P_val_3 = np.hstack([p_val_r, p_val_e, p_val_k])
    meta_3 = RidgeCV(alphas=np.logspace(-3, 2, 30), cv=5).fit(P_val_3, y_val)
    r2_best = r2_score(y_val, meta_3.predict(P_val_3))
    meta = meta_3
    use_gb, use_xgb, use_log, use_hist_gb, use_et = False, False, False, False, False
    P_val_4 = np.hstack([p_val_r, p_val_e, p_val_g, p_val_k])
    meta_4 = RidgeCV(alphas=np.logspace(-3, 2, 30), cv=5).fit(P_val_4, y_val)
    if r2_score(y_val, meta_4.predict(P_val_4)) > r2_best:
        r2_best, meta, use_gb, use_hist_gb, use_et = r2_score(y_val, meta_4.predict(P_val_4)), meta_4, True, False, False
    if HAS_XGB:
        P_val_5 = np.hstack([p_val_r, p_val_e, p_val_g, p_val_k, p_val_x])
        meta_5 = RidgeCV(alphas=np.logspace(-3, 2, 30), cv=5).fit(P_val_5, y_val)
        if r2_score(y_val, meta_5.predict(P_val_5)) > r2_best:
            meta, use_gb, use_xgb, use_hist_gb, use_et = meta_5, True, True, False, False
        P_val_x3 = np.hstack([p_val_r, p_val_e, p_val_k, p_val_x])
        meta_x3 = RidgeCV(alphas=np.logspace(-3, 2, 30), cv=5).fit(P_val_x3, y_val)
        if r2_score(y_val, meta_x3.predict(P_val_x3)) > r2_best:
            meta, use_gb, use_xgb, use_hist_gb, use_et = meta_x3, False, True, False, False
    P_val_6 = np.hstack([p_val_r, p_val_e, p_val_g, p_val_k, p_val_x, p_val_log]) if HAS_XGB else np.hstack([p_val_r, p_val_e, p_val_g, p_val_k, p_val_log])
    meta_6 = RidgeCV(alphas=np.logspace(-3, 2, 30), cv=5).fit(P_val_6, y_val)
    if r2_score(y_val, meta_6.predict(P_val_6)) > r2_best:
        meta, use_gb, use_xgb, use_log, use_hist_gb, use_et = meta_6, True, HAS_XGB, True, False, False

    class StackingPredictor:
        def __init__(self, ridge, enet, gb, hist_gb, et, keras_w, xgb_m, ridge_log, meta, use_reduced, use_gb, use_xgb, use_log, use_hist_gb, use_et):
            self.ridge, self.enet, self.gb = ridge, enet, gb
            self.hist_gb, self.et = hist_gb, et
            self.keras_w, self.xgb_m, self.ridge_log = keras_w, xgb_m, ridge_log
            self.meta = meta
            self.use_reduced = use_reduced
            self.use_gb, self.use_xgb, self.use_log = use_gb, use_xgb, use_log
            self.use_hist_gb, self.use_et = use_hist_gb, use_et
        def predict(self, X):
            X_k = X[reduced_cols] if self.use_reduced and reduced_cols else X
            pr = self.ridge.predict(X).reshape(-1, 1)
            pe = self.enet.predict(X).reshape(-1, 1)
            pk = self.keras_w.predict(X_k).reshape(-1, 1)
            parts = [pr, pe, pk]
            if self.use_gb:
                parts.insert(2, self.gb.predict(X).reshape(-1, 1))
            elif self.use_hist_gb:
                parts.insert(2, self.hist_gb.predict(X).reshape(-1, 1))
            elif self.use_et:
                parts.insert(2, self.et.predict(X).reshape(-1, 1))
            if self.use_xgb and HAS_XGB:
                parts.append(self.xgb_m.predict(X).reshape(-1, 1))
            if self.use_log:
                parts.append(self.ridge_log.predict(X).reshape(-1, 1))
            return self.meta.predict(np.hstack(parts))
    ens = StackingPredictor(ridge_r, enet, gb, hist_gb, extra_trees, keras_wrapper, xgb_m if HAS_XGB else None, ridge_log, meta, bool(reduced_cols), use_gb, use_xgb, use_log, use_hist_gb, use_et)
    ens_val_metrics = evaluate_model(ens, X_val, y_val)
    ens_test_metrics = evaluate_model(ens, X_test, y_test)
    stack_name = "Stacking (Ridge+EN+Keras)"
    if use_gb:
        stack_name = stack_name.replace("+Keras)", "+GB+Keras)")
    if use_hist_gb:
        stack_name = stack_name.replace("+Keras)", "+HistGB+Keras)")
    if use_et:
        stack_name = stack_name.replace("+Keras)", "+ET+Keras)")
    if use_xgb and HAS_XGB:
        stack_name = stack_name.replace(")", "+XGB)")
    if use_log:
        stack_name = stack_name.replace(")", "+RidgeLog)")
    results.append((stack_name, ens_val_metrics, ens_test_metrics))

    print("Model comparison (Validation | Test)")
    print("-" * 72)
    print(f"{'Model':<18} {'Val MAE':>8} {'Val RMSE':>8} {'Val R²':>8}  |  {'Test MAE':>8} {'Test RMSE':>8} {'Test R²':>8}")
    print("-" * 72)
    for name, vm, tm in results:
        print(f"{name:<18} {vm['mae']:>8.4f} {vm['rmse']:>8.4f} {vm['r2']:>8.4f}  |  {tm['mae']:>8.4f} {tm['rmse']:>8.4f} {tm['r2']:>8.4f}")
    print("-" * 72)

    best_name, _, best_test = max(results, key=lambda r: r[2]["r2"])
    print(f"\nBest by Test R²: {best_name}  (R² = {best_test['r2']:.4f})")
    if best_test["r2"] >= 0.5:
        print("Target R² ≥ 0.5 reached.")


if __name__ == "__main__":
    main()

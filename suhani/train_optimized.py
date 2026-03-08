"""
Optimized training: uses train.py pipeline (all_features_data, lags 1/3/7, roll 3/7, admission_lag7).
- Ridge/ElasticNet/RidgeLog: built-in CV (RidgeCV/ElasticNetCV).
- Tree/ensemble (RF, GBM, HistGB, ExtraTrees, XGBoost), SVR, MLP, Keras: Optuna.
- TimeSeriesSplit CV + holdout test, reports MAE, RMSE, MAPE, R².
"""

import numpy as np
import pandas as pd
import optuna
import xgboost as xgb
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import ExtraTreesRegressor, GradientBoostingRegressor, HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import ElasticNetCV, RidgeCV
from sklearn.metrics import r2_score
from sklearn.model_selection import TimeSeriesSplit
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import RobustScaler, StandardScaler
from sklearn.svm import SVR
import tensorflow as tf
from tensorflow import keras
from keras import layers

from train import (
    USE_IMPUTATION,
    load_data,
    preprocess_data,
    feature_engineering,
    cv_and_holdout_split,
    evaluate_model,
    N_CV_FOLDS,
    TEST_FRAC,
    SCALER_TYPE,
)

optuna.logging.set_verbosity(optuna.logging.WARNING)

N_TRIALS = 25       # Optuna trials for holdout training
N_TRIALS_CV = 5     # Fewer trials per fold for CV (reduce if runtime too long)
RANDOM_STATE = 42


class _PredictorWrapper:
    def __init__(self, predict_fn):
        self.predict = predict_fn


def _build_keras_model(n_features, params, seed=42):
    keras.backend.clear_session()
    tf.random.set_seed(seed)
    model = keras.Sequential()
    model.add(keras.Input(shape=(n_features,)))
    reg = keras.regularizers.L2(params["l2"]) if params["l2"] > 0 else None
    for _ in range(params["n_layers"]):
        model.add(layers.Dense(params["units"], activation="relu", kernel_initializer="he_normal", kernel_regularizer=reg))
        model.add(layers.BatchNormalization())
        model.add(layers.Dropout(params["dropout"]))
    model.add(layers.Dense(1, kernel_regularizer=reg))
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=params["lr"]), loss="mse", metrics=["mae"])
    return model


def tune_and_fit_keras(X_train, y_train, X_val, y_val, X_test, y_test, n_trials=N_TRIALS):
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
        model = _build_keras_model(n_features, params, RANDOM_STATE)
        early = keras.callbacks.EarlyStopping(monitor="val_loss", patience=15, restore_best_weights=True)
        model.fit(X_tr, y_tr, validation_data=(X_va, y_va), epochs=250, batch_size=32, callbacks=[early], verbose=0)
        pred = model.predict(X_va, verbose=0).ravel()
        return r2_score(y_val, pred)

    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=RANDOM_STATE, n_startup_trials=5))
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    best = study.best_params
    best["units"] = int(best["units"])
    best["n_layers"] = int(best["n_layers"])
    final = _build_keras_model(n_features, best, RANDOM_STATE)
    early = keras.callbacks.EarlyStopping(monitor="val_loss", patience=20, restore_best_weights=True)
    final.fit(X_tr, y_tr, validation_data=(X_va, y_va), epochs=250, batch_size=32, callbacks=[early], verbose=0)
    wrap = _PredictorWrapper(lambda X: final.predict(np.asarray(X, dtype=np.float32), verbose=0).ravel())
    return final, evaluate_model(wrap, X_val, y_val), evaluate_model(wrap, X_test, y_test)


def tune_and_fit_xgboost(X_train, y_train, X_val, y_val, X_test, y_test, n_trials=N_TRIALS):
    def objective(trial):
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 50, 300),
            "max_depth": trial.suggest_int("max_depth", 3, 10),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
        }
        model = xgb.XGBRegressor(**params, random_state=RANDOM_STATE)
        model.fit(X_train, y_train)
        pred = model.predict(X_val)
        return r2_score(y_val, pred)

    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=RANDOM_STATE, n_startup_trials=5))
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    best = study.best_params
    model = xgb.XGBRegressor(**best, random_state=RANDOM_STATE)
    model.fit(X_train, y_train)
    return model, evaluate_model(model, X_val, y_val), evaluate_model(model, X_test, y_test)


def tune_and_fit_rf(X_train, y_train, X_val, y_val, X_test, y_test, n_trials=N_TRIALS):
    def objective(trial):
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 50, 200),
            "max_depth": trial.suggest_int("max_depth", 5, 20),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 2, 20),
        }
        model = RandomForestRegressor(**params, random_state=RANDOM_STATE, n_jobs=-1)
        model.fit(X_train, y_train)
        return r2_score(y_val, model.predict(X_val))

    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=RANDOM_STATE, n_startup_trials=5))
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    best = study.best_params
    model = RandomForestRegressor(**best, random_state=RANDOM_STATE, n_jobs=-1)
    model.fit(X_train, y_train)
    return model, evaluate_model(model, X_val, y_val), evaluate_model(model, X_test, y_test)


def tune_and_fit_gbm(X_train, y_train, X_val, y_val, X_test, y_test, n_trials=N_TRIALS):
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

    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=RANDOM_STATE, n_startup_trials=5))
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    best = study.best_params
    model = GradientBoostingRegressor(**best, random_state=RANDOM_STATE)
    model.fit(X_train, y_train)
    return model, evaluate_model(model, X_val, y_val), evaluate_model(model, X_test, y_test)


def tune_and_fit_histgb(X_train, y_train, X_val, y_val, X_test, y_test, n_trials=N_TRIALS):
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

    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=RANDOM_STATE, n_startup_trials=5))
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    best = study.best_params
    model = HistGradientBoostingRegressor(**best, random_state=RANDOM_STATE)
    model.fit(X_train, y_train)
    return model, evaluate_model(model, X_val, y_val), evaluate_model(model, X_test, y_test)


def tune_and_fit_extra_trees(X_train, y_train, X_val, y_val, X_test, y_test, n_trials=N_TRIALS):
    def objective(trial):
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 50, 250),
            "max_depth": trial.suggest_int("max_depth", 5, 20),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 2, 15),
        }
        model = ExtraTreesRegressor(**params, random_state=RANDOM_STATE, n_jobs=-1)
        model.fit(X_train, y_train)
        return r2_score(y_val, model.predict(X_val))

    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=RANDOM_STATE, n_startup_trials=5))
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    best = study.best_params
    model = ExtraTreesRegressor(**best, random_state=RANDOM_STATE, n_jobs=-1)
    model.fit(X_train, y_train)
    return model, evaluate_model(model, X_val, y_val), evaluate_model(model, X_test, y_test)


def tune_and_fit_svr(X_train, y_train, X_val, y_val, X_test, y_test, n_trials=N_TRIALS):
    def objective(trial):
        C = trial.suggest_float("C", 0.1, 100.0, log=True)
        epsilon = trial.suggest_float("epsilon", 0.01, 1.0)
        kernel = trial.suggest_categorical("kernel", ["rbf", "poly"])
        model = SVR(C=C, epsilon=epsilon, kernel=kernel)
        model.fit(X_train, y_train)
        return r2_score(y_val, model.predict(X_val))

    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=RANDOM_STATE, n_startup_trials=5))
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    best = study.best_params
    model = SVR(**best)
    model.fit(X_train, y_train)
    return model, evaluate_model(model, X_val, y_val), evaluate_model(model, X_test, y_test)


def tune_and_fit_mlp(X_train, y_train, X_val, y_val, X_test, y_test, n_trials=N_TRIALS):
    def objective(trial):
        n_layers = trial.suggest_int("n_layers", 1, 3)
        layers = []
        for i in range(n_layers):
            layers.append(trial.suggest_int(f"units_{i}", 16, 128))
        alpha = trial.suggest_float("alpha", 1e-5, 1e-2, log=True)
        lr = trial.suggest_float("learning_rate_init", 1e-4, 1e-2, log=True)
        model = MLPRegressor(hidden_layer_sizes=tuple(layers), activation="relu", solver="adam", alpha=alpha, learning_rate_init=lr, max_iter=1000, random_state=RANDOM_STATE)
        model.fit(X_train, y_train)
        return r2_score(y_val, model.predict(X_val))

    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=RANDOM_STATE, n_startup_trials=5))
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


def _run_fold_models(X_tr, y_tr, X_va, y_va, n_trials, inner_ts_cv=None):
    """Run all models on one fold; return list of (name, val_metrics)."""
    scaler_cls = RobustScaler if SCALER_TYPE == "robust" else StandardScaler
    results = []

    dummy = DummyRegressor(strategy="mean")
    dummy.fit(X_tr, y_tr)
    results.append(("Dummy (mean)", evaluate_model(dummy, X_va, y_va)))

    ridge = RidgeCV(alphas=np.logspace(-3, 2, 100), cv=inner_ts_cv or 5).fit(X_tr, y_tr)
    results.append(("Ridge (CV)", evaluate_model(ridge, X_va, y_va)))

    ridge_log = RidgeCV(alphas=np.logspace(-3, 2, 100), cv=inner_ts_cv or 5)
    ridge_log.fit(X_tr, np.log1p(y_tr))
    pred_fn = _PredictorWrapper(lambda X: np.expm1(ridge_log.predict(X)))
    results.append(("Ridge (log y)", evaluate_model(pred_fn, X_va, y_va)))

    enet = ElasticNetCV(cv=inner_ts_cv or 5, l1_ratio=[0.1, 0.5, 0.9, 1.0], alphas=np.logspace(-3, 1, 50), max_iter=5000).fit(X_tr, y_tr)
    results.append(("ElasticNet (CV)", evaluate_model(enet, X_va, y_va)))

    _, vm, _ = tune_and_fit_rf(X_tr, y_tr, X_va, y_va, X_va, y_va, n_trials=n_trials)
    results.append(("Random Forest (tuned)", vm))
    _, vm, _ = tune_and_fit_gbm(X_tr, y_tr, X_va, y_va, X_va, y_va, n_trials=n_trials)
    results.append(("Gradient Boosting (tuned)", vm))
    _, vm, _ = tune_and_fit_histgb(X_tr, y_tr, X_va, y_va, X_va, y_va, n_trials=n_trials)
    results.append(("Hist Gradient Boosting (tuned)", vm))
    _, vm, _ = tune_and_fit_extra_trees(X_tr, y_tr, X_va, y_va, X_va, y_va, n_trials=n_trials)
    results.append(("Extra Trees (tuned)", vm))
    _, vm, _ = tune_and_fit_xgboost(X_tr, y_tr, X_va, y_va, X_va, y_va, n_trials=n_trials)
    results.append(("XGBoost (tuned)", vm))
    _, vm, _ = tune_and_fit_svr(X_tr, y_tr, X_va, y_va, X_va, y_va, n_trials=n_trials)
    results.append(("SVR (tuned)", vm))
    _, vm, _ = tune_and_fit_mlp(X_tr, y_tr, X_va, y_va, X_va, y_va, n_trials=n_trials)
    results.append(("FNN (MLP) (tuned)", vm))
    _, vm, _ = tune_and_fit_keras(X_tr, y_tr, X_va, y_va, X_va, y_va, n_trials=n_trials)
    results.append(("Keras (tuned)", vm))

    return results


def run_time_series_cv(X_cv, y_cv):
    """Run TimeSeriesSplit CV; return list of (name, cv_metrics) with mean±std."""
    tscv = TimeSeriesSplit(n_splits=N_CV_FOLDS)
    scaler_cls = RobustScaler if SCALER_TYPE == "robust" else StandardScaler
    model_names = [
        "Dummy (mean)", "Ridge (CV)", "Ridge (log y)", "ElasticNet (CV)",
        "Random Forest (tuned)", "Gradient Boosting (tuned)", "Hist Gradient Boosting (tuned)",
        "Extra Trees (tuned)", "XGBoost (tuned)", "SVR (tuned)", "FNN (MLP) (tuned)", "Keras (tuned)",
    ]
    fold_results = {n: [] for n in model_names}

    for fold, (train_ix, val_ix) in enumerate(tscv.split(X_cv)):
        print(f"  CV fold {fold + 1}/{N_CV_FOLDS}...")
        X_tr = X_cv[train_ix]
        y_tr = y_cv[train_ix]
        X_va = X_cv[val_ix]
        y_va = y_cv[val_ix]
        scaler = scaler_cls()
        X_tr_sc = scaler.fit_transform(X_tr)
        X_va_sc = scaler.transform(X_va)
        inner_ts = (
            TimeSeriesSplit(n_splits=min(3, len(train_ix) // 20))
            if len(train_ix) >= 40
            else None
        )
        fold_out = _run_fold_models(X_tr_sc, y_tr, X_va_sc, y_va, N_TRIALS_CV, inner_ts)
        for name, m in fold_out:
            fold_results[name].append(m)

    cv_results = []
    for name in model_names:
        arr = fold_results[name]
        means = {k: np.mean([a[k] for a in arr]) for k in arr[0]}
        stds = {k: np.std([a[k] for a in arr]) for k in arr[0]}
        cv_results.append((name, {k: (means[k], stds[k]) for k in means}))
    return cv_results


def run_holdout_evaluation(X_cv, y_cv, X_test, y_test):
    """Train on full CV data, evaluate on holdout; return list of (name, test_metrics)."""
    scaler_cls = RobustScaler if SCALER_TYPE == "robust" else StandardScaler
    scaler = scaler_cls()
    X_cv_sc = scaler.fit_transform(X_cv)
    X_test_sc = scaler.transform(X_test)
    inner_ts = (
        TimeSeriesSplit(n_splits=min(3, len(X_cv) // 20))
        if len(X_cv) >= 40
        else None
    )
    n_val = max(1, len(X_cv) // 10)

    results = []
    dummy = DummyRegressor(strategy="mean")
    dummy.fit(X_cv_sc, y_cv)
    results.append(("Dummy (mean)", evaluate_model(dummy, X_test_sc, y_test)))

    ridge = RidgeCV(alphas=np.logspace(-3, 2, 100), cv=inner_ts or 5).fit(X_cv_sc, y_cv)
    results.append(("Ridge (CV)", evaluate_model(ridge, X_test_sc, y_test)))

    ridge_log = RidgeCV(alphas=np.logspace(-3, 2, 100), cv=inner_ts or 5)
    ridge_log.fit(X_cv_sc, np.log1p(y_cv))
    results.append(("Ridge (log y)", evaluate_model(_PredictorWrapper(lambda X: np.expm1(ridge_log.predict(X))), X_test_sc, y_test)))

    enet = ElasticNetCV(cv=inner_ts or 5, l1_ratio=[0.1, 0.5, 0.9, 1.0], alphas=np.logspace(-3, 1, 50), max_iter=5000).fit(X_cv_sc, y_cv)
    results.append(("ElasticNet (CV)", evaluate_model(enet, X_test_sc, y_test)))

    print("  Tuning Random Forest...")
    _, _, tm = tune_and_fit_rf(X_cv_sc, y_cv, X_cv_sc[-n_val:], y_cv[-n_val:], X_test_sc, y_test)
    results.append(("Random Forest (tuned)", tm))
    print("  Tuning Gradient Boosting...")
    _, _, tm = tune_and_fit_gbm(X_cv_sc, y_cv, X_cv_sc[-n_val:], y_cv[-n_val:], X_test_sc, y_test)
    results.append(("Gradient Boosting (tuned)", tm))
    print("  Tuning Hist Gradient Boosting...")
    _, _, tm = tune_and_fit_histgb(X_cv_sc, y_cv, X_cv_sc[-n_val:], y_cv[-n_val:], X_test_sc, y_test)
    results.append(("Hist Gradient Boosting (tuned)", tm))
    print("  Tuning Extra Trees...")
    _, _, tm = tune_and_fit_extra_trees(X_cv_sc, y_cv, X_cv_sc[-n_val:], y_cv[-n_val:], X_test_sc, y_test)
    results.append(("Extra Trees (tuned)", tm))
    print("  Tuning XGBoost...")
    _, _, tm = tune_and_fit_xgboost(X_cv_sc, y_cv, X_cv_sc[-n_val:], y_cv[-n_val:], X_test_sc, y_test)
    results.append(("XGBoost (tuned)", tm))
    print("  Tuning SVR...")
    _, _, tm = tune_and_fit_svr(X_cv_sc, y_cv, X_cv_sc[-n_val:], y_cv[-n_val:], X_test_sc, y_test)
    results.append(("SVR (tuned)", tm))
    print("  Tuning FNN (MLP)...")
    _, _, tm = tune_and_fit_mlp(X_cv_sc, y_cv, X_cv_sc[-n_val:], y_cv[-n_val:], X_test_sc, y_test)
    results.append(("FNN (MLP) (tuned)", tm))
    print("  Tuning Keras...")
    _, _, tm = tune_and_fit_keras(X_cv_sc, y_cv, X_cv_sc[-n_val:], y_cv[-n_val:], X_test_sc, y_test)
    results.append(("Keras (tuned)", tm))

    return results


def main():
    df = load_data()
    n_load = len(df)
    df = preprocess_data(df, impute_features=USE_IMPUTATION)
    n_pre = len(df)
    df = feature_engineering(df)
    n_fe = len(df)
    X_cv, y_cv, X_test, y_test, feat_cols = cv_and_holdout_split(df)

    print("Data: load → preprocess → feature_eng")
    print(f"  Rows: {n_load} → {n_pre} (impute={USE_IMPUTATION}) → {n_fe}")
    print(f"  Features: {len(feat_cols)} cols")
    print(f"  Split: TimeSeriesSplit({N_CV_FOLDS} folds) on CV | holdout test ({TEST_FRAC*100:.0f}%)")
    print(f"  CV samples: {len(y_cv)} | Test samples: {len(y_test)}")
    print(f"  Optuna: {N_TRIALS_CV} trials/fold (CV), {N_TRIALS} trials (holdout)\n")

    print("Running TimeSeriesSplit CV (this may take a while)...")
    cv_results = run_time_series_cv(X_cv, y_cv)

    print("\nTraining on full CV data, evaluating on holdout...")
    holdout_results = run_holdout_evaluation(X_cv, y_cv, X_test, y_test)

    w = 150
    print("\nModel comparison (CV mean ± std | Holdout test)")
    print("-" * w)
    header = (
        f"{'Model':<28} | {'CV MAE':>12} {'CV RMSE':>12} {'CV MAPE%':>12} {'CV R²':>12}  |  "
        f"{'Test MAE':>8} {'Test RMSE':>8} {'Test MAPE%':>9} {'Test R²':>8}"
    )
    print(header)
    print("-" * w)
    for (name, cv_m), (_, test_m) in zip(cv_results, holdout_results):
        row = (
            f"{name:<28} | {cv_m['mae'][0]:.4f}±{cv_m['mae'][1]:.4f} "
            f"{cv_m['rmse'][0]:.4f}±{cv_m['rmse'][1]:.4f} "
            f"{cv_m['mape'][0]:.2f}±{cv_m['mape'][1]:.2f} "
            f"{cv_m['r2'][0]:.4f}±{cv_m['r2'][1]:.4f}  |  "
            f"{test_m['mae']:>8.4f} {test_m['rmse']:>8.4f} {test_m['mape']:>9.2f} {test_m['r2']:>8.4f}"
        )
        print(row)
    print("-" * w)

    best_mae = min(holdout_results, key=lambda r: r[1]["mae"])
    best_rmse = min(holdout_results, key=lambda r: r[1]["rmse"])
    best_mape = min(holdout_results, key=lambda r: r[1]["mape"])
    best_r2 = max(holdout_results, key=lambda r: r[1]["r2"])

    print("\nBest by Holdout Test metric:")
    print(f"  MAE:   {best_mae[0]:<28}  (MAE   = {best_mae[1]['mae']:.4f})")
    print(f"  RMSE:  {best_rmse[0]:<28}  (RMSE  = {best_rmse[1]['rmse']:.4f})")
    print(f"  MAPE:  {best_mape[0]:<28}  (MAPE  = {best_mape[1]['mape']:.2f}%)")
    print(f"  R²:    {best_r2[0]:<28}  (R²    = {best_r2[1]['r2']:.4f})")

    print("\n" + "=" * 60)
    print("Literature comparison (air quality + weather → admissions)")
    print("=" * 60)
    print("  npj Digital Medicine (2022), ED admissions:")
    print("    MAE 4.0 (17% error) vs benchmark 6.5 (32%)")
    print("  Nature Sci Rep (2023), German ER visits:")
    print("    Air+weather explained ~6% of variance (mobility ~63%)")
    print("  Multi-city air pollution–mortality:")
    print("    R² typically <1%–13% for environmental predictors")
    print("-" * 60)
    print("  This study (Optuna-tuned, TimeSeriesSplit CV):")
    print(f"    Test MAE  {best_mae[1]['mae']:.2f}  |  Test MAPE  {best_mape[1]['mape']:.1f}%  |  Test R²  {best_r2[1]['r2']:.4f}")


if __name__ == "__main__":
    main()

"""
Optimized training: each model tuned with the most suitable optimizer.
- Ridge/ElasticNet/RidgeReduced/RidgeLog: built-in CV (RidgeCV/ElasticNetCV).
- Tree/ensemble (RF, GBM, HistGB, ExtraTrees, XGBoost), SVR, MLP, Keras: Optuna.
"""

import numpy as np
import pandas as pd
import optuna
import xgboost as xgb
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import ExtraTreesRegressor, GradientBoostingRegressor, HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import ElasticNetCV, RidgeCV
from sklearn.metrics import r2_score
from sklearn.neural_network import MLPRegressor
from sklearn.svm import SVR
import tensorflow as tf
from tensorflow import keras
from keras import layers

# Reuse data pipeline and config from train (single feature set)
from train import (
    USE_IMPUTATION,
    load_data,
    preprocess_data,
    feature_engineering,
    training_data_split,
    evaluate_model,
)

optuna.logging.set_verbosity(optuna.logging.WARNING)

N_TRIALS = 25  # Optuna trials per tunable model
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
    print(f"  Train / Val / Test: {len(y_train)} / {len(y_val)} / {len(y_test)}\n")
    print(f"Optuna: {N_TRIALS} trials per tunable model.\n")

    results = []

    # No tuning
    dummy = DummyRegressor(strategy="mean")
    dummy.fit(X_train, y_train)
    results.append(("Dummy (mean)", evaluate_model(dummy, X_val, y_val), evaluate_model(dummy, X_test, y_test)))

    # Built-in CV (already optimal for linear models)
    ridge = RidgeCV(alphas=np.logspace(-3, 2, 100), cv=5).fit(X_train, y_train)
    results.append(("Ridge (CV)", evaluate_model(ridge, X_val, y_val), evaluate_model(ridge, X_test, y_test)))

    ridge_log = RidgeCV(alphas=np.logspace(-3, 2, 100), cv=5)
    ridge_log.fit(X_train, np.log1p(y_train))
    def _ridge_log_predict(X):
        return np.expm1(ridge_log.predict(X))
    results.append(("Ridge (log y)", evaluate_model(_PredictorWrapper(_ridge_log_predict), X_val, y_val), evaluate_model(_PredictorWrapper(_ridge_log_predict), X_test, y_test)))

    enet = ElasticNetCV(cv=5, l1_ratio=[0.1, 0.5, 0.9, 1.0], alphas=np.logspace(-3, 1, 50), max_iter=5000).fit(X_train, y_train)
    results.append(("ElasticNet (CV)", evaluate_model(enet, X_val, y_val), evaluate_model(enet, X_test, y_test)))

    ridge_red = RidgeCV(alphas=np.logspace(-2, 2, 50), cv=5).fit(X_train, y_train)
    results.append(("Ridge (reduced)", evaluate_model(ridge_red, X_val, y_val), evaluate_model(ridge_red, X_test, y_test)))

    # Optuna-tuned models
    print("Tuning Random Forest...")
    _, vm, tm = tune_and_fit_rf(X_train, y_train, X_val, y_val, X_test, y_test)
    results.append(("Random Forest (tuned)", vm, tm))

    print("Tuning Gradient Boosting...")
    _, vm, tm = tune_and_fit_gbm(X_train, y_train, X_val, y_val, X_test, y_test)
    results.append(("Gradient Boosting (tuned)", vm, tm))

    print("Tuning Hist Gradient Boosting...")
    _, vm, tm = tune_and_fit_histgb(X_train, y_train, X_val, y_val, X_test, y_test)
    results.append(("Hist Gradient Boosting (tuned)", vm, tm))

    print("Tuning Extra Trees...")
    _, vm, tm = tune_and_fit_extra_trees(X_train, y_train, X_val, y_val, X_test, y_test)
    results.append(("Extra Trees (tuned)", vm, tm))

    print("Tuning XGBoost...")
    _, vm, tm = tune_and_fit_xgboost(X_train, y_train, X_val, y_val, X_test, y_test)
    results.append(("XGBoost (tuned)", vm, tm))

    print("Tuning SVR...")
    _, vm, tm = tune_and_fit_svr(X_train, y_train, X_val, y_val, X_test, y_test)
    results.append(("SVR (tuned)", vm, tm))

    print("Tuning FNN (MLP)...")
    _, vm, tm = tune_and_fit_mlp(X_train, y_train, X_val, y_val, X_test, y_test)
    results.append(("FNN (MLP) (tuned)", vm, tm))

    print("Tuning Keras...")
    _, vm, tm = tune_and_fit_keras(X_train, y_train, X_val, y_val, X_test, y_test)
    results.append(("Keras (tuned)", vm, tm))

    print("\nModel comparison (Validation | Test)")
    print("-" * 72)
    print(f"{'Model':<28} {'Val MAE':>8} {'Val RMSE':>8} {'Val R²':>8}  |  {'Test MAE':>8} {'Test RMSE':>8} {'Test R²':>8}")
    print("-" * 72)
    for name, vm, tm in results:
        print(f"{name:<28} {vm['mae']:>8.4f} {vm['rmse']:>8.4f} {vm['r2']:>8.4f}  |  {tm['mae']:>8.4f} {tm['rmse']:>8.4f} {tm['r2']:>8.4f}")
    print("-" * 72)

    best_name, _, best_test = max(results, key=lambda r: r[2]["r2"])
    print(f"\nBest by Test R²: {best_name}  (R² = {best_test['r2']:.4f})")


if __name__ == "__main__":
    main()

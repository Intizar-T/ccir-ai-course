"""
Vanilla LSTM baseline for 21-day SPY return prediction.
Uses past N trading days of features to predict cumulative return over the next 21 days.
No target scaling (y in raw return space); X scaled with Robust/StandardScaler.
Hyperopt (20 trials) maximizes validation R²; best model is retrained and evaluated on test.
"""
import os
import random

# Force CPU for reliable, fast runs (Metal on M1 can be slower/stuck for small LSTMs).
if os.environ.get("VANILLA_LSTM_CPU", "1").lower() in ("1", "true", "yes"):
    import tensorflow as tf
    tf.config.set_visible_devices([], "GPU")

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.cluster import KMeans
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout
from keras.regularizers import L2
from keras.callbacks import EarlyStopping
from keras.optimizers import Adam
from keras import backend as K
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials

# --- CONFIGURATION ---
FORECAST_HORIZON = 21   # Model predicts the cumulative return over the NEXT 21 days
TRAIN_FRAC = 0.6
VAL_FRAC = 0.2
TEST_FRAC = 0.2
SEED = 42
HYPEROPT_TRIALS = 20

# Columns used for regime clustering (same as LSTM features, no target).
REGIME_FEATURE_COLS = ['SPY_Returns', 'FFR_Lagged', 'VIX_Close', 'CPI_YoY', 'IWM_Returns']
BASE_FEATURE_COLS = ['SPY_Returns', 'FFR_Lagged', 'VIX_Close', 'CPI_YoY', 'IWM_Returns']

# Set by run_pipeline_with_regime so objective() loads the regime dataset.
_CURRENT_DATA_PATH = None


def set_seed(seed=SEED):
    np.random.seed(seed)
    random.seed(seed)
    try:
        K.set_random_seed(seed)
    except Exception:
        pass


def process_data(file_path=None):
    """Load the dataset and handle dates. If file_path is None, uses data/spy.csv."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    path = file_path or os.path.join(script_dir, 'data/spy.csv')
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing {path}")
    df = pd.read_csv(path)
    df['Date'] = pd.to_datetime(df['Date'])
    df.set_index('Date', inplace=True)
    return df


def add_regime_kmeans(df, n_clusters=3, train_frac=TRAIN_FRAC, random_state=SEED):
    """
    Fit KMeans on train portion of regime features, assign regime label to every row.
    Returns df with new column 'regime' (0 .. n_clusters-1). No leakage: fit on train only.
    """
    set_seed(random_state)
    cols = [c for c in REGIME_FEATURE_COLS if c in df.columns]
    if len(cols) < 2:
        raise ValueError(f"Need at least 2 of {REGIME_FEATURE_COLS} for regime clustering")
    train_end = int(len(df) * train_frac)
    X = df[cols].values
    scaler = StandardScaler()
    scaler.fit(X[:train_end])
    X_scaled = scaler.transform(X)
    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
    kmeans.fit(X_scaled[:train_end])
    df = df.copy()
    df['regime'] = kmeans.predict(X_scaled)
    return df


def add_regime_gmm(df, n_components=3, train_frac=TRAIN_FRAC, random_state=SEED):
    """Stub for GMM regime detection. To be implemented."""
    raise NotImplementedError("GMM regime not implemented yet; use regime_detector='kmeans'.")


def add_regime_hmm(df, n_components=3, train_frac=TRAIN_FRAC, random_state=SEED):
    """Stub for HMM regime detection. To be implemented."""
    raise NotImplementedError("HMM regime not implemented yet; use regime_detector='kmeans'.")


def add_regime_and_save(df, regime_detector='kmeans', output_path=None, **kwargs):
    """
    Add regime column using the given detector (kmeans/gmm/hmm) and save the dataset.
    Returns df with 'regime' column. Saves to output_path if provided.
    """
    if regime_detector == 'kmeans':
        n_clusters = kwargs.get('n_clusters', 3)
        df = add_regime_kmeans(df, n_clusters=n_clusters, **{k: v for k, v in kwargs.items() if k in ('train_frac', 'random_state')})
    elif regime_detector == 'gmm':
        df = add_regime_gmm(df, **kwargs)
    elif regime_detector == 'hmm':
        df = add_regime_hmm(df, **kwargs)
    else:
        raise ValueError(f"Unknown regime_detector: {regime_detector}. Use 'kmeans', 'gmm', or 'hmm'.")
    if output_path:
        os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
        df.to_csv(output_path)
    return df


def feature_engineering(df):
    """Creates the 21-day forward target (no scaling). feature_cols include 'regime' if present."""
    indexer = pd.api.indexers.FixedForwardWindowIndexer(window_size=FORECAST_HORIZON)
    df['Target_21d'] = df['SPY_Returns'].rolling(window=indexer).sum().shift(-1)
    df.dropna(inplace=True)
    feature_cols = list(BASE_FEATURE_COLS)
    if 'regime' in df.columns:
        feature_cols = feature_cols + ['regime']
    return df, feature_cols


def create_sequences(X, y, window_size):
    """X: (samples, features), y: 1d. Returns 3D X for LSTM and 1d y."""
    Xs, ys = [], []
    for i in range(len(X) - window_size):
        Xs.append(X[i:(i + window_size)])
        ys.append(y[i + window_size - 1])
    return np.array(Xs), np.array(ys)


def scale_and_split(df, feature_cols, input_window, scaler_class):
    """
    Scale only X (fit on train). y stays raw. No target scaling.
    scaler_class: StandardScaler or RobustScaler.
    """
    N = len(df)
    train_end = int(N * TRAIN_FRAC)
    val_end = int(N * (TRAIN_FRAC + VAL_FRAC))
    scaler_X = scaler_class()
    scaler_X.fit(df[feature_cols].iloc[:train_end])
    X_scaled = scaler_X.transform(df[feature_cols])
    y_raw = df['Target_21d'].values.reshape(-1, 1)
    X_seq, y_seq = create_sequences(X_scaled, y_raw, input_window)
    train_seq_end = train_end - input_window
    val_seq_end = val_end - input_window
    X_train = X_seq[:train_seq_end]
    X_val = X_seq[train_seq_end:val_seq_end]
    X_test = X_seq[val_seq_end:]
    y_train = y_seq[:train_seq_end]
    y_val = y_seq[train_seq_end:val_seq_end]
    y_test = y_seq[val_seq_end:]
    return X_train, X_val, X_test, y_train, y_val, y_test, scaler_X


def build_model(input_shape, lstm_units_1, lstm_units_2, dropout, lr, l2):
    """One or two LSTM layers; lstm_units_2=0 means single layer. L2 on Dense only."""
    reg = L2(l2) if l2 > 0 else None
    model = Sequential()
    if lstm_units_2 is None or lstm_units_2 <= 0:
        model.add(LSTM(lstm_units_1, activation='relu', input_shape=input_shape, return_sequences=False))
    else:
        model.add(LSTM(lstm_units_1, activation='relu', input_shape=input_shape, return_sequences=True))
        model.add(Dropout(dropout))
        model.add(LSTM(lstm_units_2, activation='relu', return_sequences=False))
    model.add(Dropout(dropout))
    model.add(Dense(1, kernel_regularizer=reg))
    model.compile(optimizer=Adam(learning_rate=lr), loss='mae')
    return model


def train_model(X_train, y_train, X_val, y_val, params, verbose=0):
    """Train with early stopping. Returns model and validation R²."""
    set_seed(SEED)
    model = build_model(
        (X_train.shape[1], X_train.shape[2]),
        lstm_units_1=params['lstm_units_1'],
        lstm_units_2=params['lstm_units_2'] if params['lstm_units_2'] > 0 else None,
        dropout=params['dropout'],
        lr=params['lr'],
        l2=params['l2']
    )
    early_stop = EarlyStopping(
        monitor='val_loss',
        patience=int(params['patience']),
        restore_best_weights=True
    )
    model.fit(
        X_train, y_train,
        epochs=60,
        batch_size=int(params['batch_size']),
        validation_data=(X_val, y_val),
        callbacks=[early_stop],
        verbose=verbose
    )
    val_pred = model.predict(X_val, verbose=0)
    y_v = np.ravel(y_val)
    p_v = np.ravel(val_pred)
    if not np.isfinite(p_v).all():
        return model, -1.0
    val_r2 = r2_score(y_v, p_v)
    return model, val_r2


def hyperopt_space():
    return {
        'input_window': hp.choice('input_window', [21, 42, 63]),
        'scaler': hp.choice('scaler', ['robust', 'standard']),
        'lstm_units_1': hp.choice('lstm_units_1', [32, 64, 128]),
        'lstm_units_2': hp.choice('lstm_units_2', [0, 32, 64]),
        'dropout': hp.uniform('dropout', 0.25, 0.5),
        'l2': hp.loguniform('l2', np.log(1e-6), np.log(1e-2)),
        'lr': hp.loguniform('lr', np.log(5e-4), np.log(5e-2)),
        'batch_size': hp.choice('batch_size', [16, 32, 64]),
        'patience': hp.choice('patience', [8, 12, 15]),
    }


def objective(params):
    """Minimize negative validation R². Uses _CURRENT_DATA_PATH if set (regime pipeline)."""
    set_seed(SEED)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    file_path = _CURRENT_DATA_PATH or os.path.join(script_dir, 'data/spy.csv')
    df = pd.read_csv(file_path)
    df['Date'] = pd.to_datetime(df['Date'])
    df.set_index('Date', inplace=True)
    df, feature_cols = feature_engineering(df)
    scaler_class = RobustScaler if params['scaler'] == 'robust' else StandardScaler
    try:
        X_train, X_val, X_test, y_train, y_val, y_test, _ = scale_and_split(
            df, feature_cols, params['input_window'], scaler_class
        )
        _, val_r2 = train_model(X_train, y_train, X_val, y_val, params, verbose=0)
        if np.isfinite(val_r2):
            return {'loss': -val_r2, 'status': STATUS_OK}
    except Exception:
        pass
    return {'loss': 0.0, 'status': STATUS_OK}


def evaluate_and_print(model, X_test, y_test, y_train):
    """Predictions are already in raw return space (no y scaling)."""
    y_pred = model.predict(X_test, verbose=0)
    y_act = np.ravel(y_test)
    y_pred = np.ravel(y_pred)
    mae = mean_absolute_error(y_act, y_pred)
    rmse = np.sqrt(mean_squared_error(y_act, y_pred))
    r2 = r2_score(y_act, y_pred)
    direction_correct = (np.sign(y_pred) == np.sign(y_act)).mean()
    corr = np.corrcoef(y_act, y_pred)[0, 1] if len(y_act) > 1 else 0.0
    mean_train = np.mean(y_train)
    mae_mean = mean_absolute_error(y_act, np.full_like(y_act, mean_train))
    mae_zero = mean_absolute_error(y_act, np.zeros_like(y_act))
    r2_mean = r2_score(y_act, np.full_like(y_act, mean_train))
    r2_zero = r2_score(y_act, np.zeros_like(y_act))
    print("\n" + "=" * 60)
    print("EVALUATION (Test Set Only)")
    print("=" * 60)
    print("\n--- LSTM Model ---")
    print(f"  MAE (21-day return):     {mae:.5f}  ({mae*100:.2f}% avg error)")
    print(f"  RMSE:                    {rmse:.5f}")
    print(f"  R²:                      {r2:.4f}")
    print(f"  Directional accuracy:   {direction_correct*100:.1f}%")
    print(f"  Correlation (pred vs actual): {corr:.4f}")
    print("\n--- Baselines ---")
    print(f"  Predict historical mean: MAE = {mae_mean:.5f}, R² = {r2_mean:.4f}")
    print(f"  Predict zero:            MAE = {mae_zero:.5f}, R² = {r2_zero:.4f}")
    print("=" * 60)
    return r2


def main():
    set_seed(SEED)
    print("Loading data...")
    df = process_data()
    print(f"Initial Shape: {df.shape}")
    df, feature_cols = feature_engineering(df)

    # Hyperopt: 20 trials to maximize validation R²
    print(f"\nRunning Hyperopt ({HYPEROPT_TRIALS} trials) to maximize validation R²...")
    trials = Trials()
    best = fmin(
        fn=objective,
        space=hyperopt_space(),
        algo=tpe.suggest,
        max_evals=HYPEROPT_TRIALS,
        rstate=np.random.default_rng(SEED),
        trials=trials,
        verbose=1
    )
    # Map choice indices back to values
    input_windows = [21, 42, 63]
    scaler_names = ['robust', 'standard']
    units_1 = [32, 64, 128]
    units_2 = [0, 32, 64]
    batch_sizes = [16, 32, 64]
    patiences = [8, 12, 15]
    best_params = {
        'input_window': input_windows[best['input_window']],
        'scaler': scaler_names[best['scaler']],
        'lstm_units_1': units_1[best['lstm_units_1']],
        'lstm_units_2': units_2[best['lstm_units_2']],
        'dropout': best['dropout'],
        'l2': float(best['l2']),
        'lr': best['lr'],
        'batch_size': batch_sizes[best['batch_size']],
        'patience': patiences[best['patience']],
    }
    print("\nBest hyperparameters:", best_params)

    # Retrain best config with fixed seed and full verbose
    scaler_class = RobustScaler if best_params['scaler'] == 'robust' else StandardScaler
    X_train, X_val, X_test, y_train, y_val, y_test, scaler_X = scale_and_split(
        df, feature_cols, best_params['input_window'], scaler_class
    )
    print(f"\nSequence shapes: train {X_train.shape}, val {X_val.shape}, test {X_test.shape}")
    model, val_r2 = train_model(X_train, y_train, X_val, y_val, best_params, verbose=1)
    print(f"Best validation R²: {val_r2:.4f}")
    test_r2 = evaluate_and_print(model, X_test, y_test, y_train)
    print(f"\nTest R²: {test_r2:.4f}")

    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.makedirs(os.path.join(script_dir, 'model'), exist_ok=True)
    model.save(os.path.join(script_dir, 'model/vanilla_lstm_21d.keras'))
    print("Model saved.")


def run_pipeline_with_regime(
    regime_detector='kmeans',
    data_path=None,
    n_clusters=3,
    do_hyperopt=True,
    **regime_kwargs
):
    """
    Generic pipeline: add regime column (kmeans/gmm/hmm), save regime dataset, then run LSTM.
    - regime_detector: 'kmeans', 'gmm', or 'hmm' (gmm/hmm stubbed for later).
    - data_path: base CSV to load (default: data/spy.csv). Regime dataset is saved alongside.
    - n_clusters / n_components: passed to the regime detector.
    - do_hyperopt: if True, run Hyperopt then retrain best; else use default params and train once.
    """
    global _CURRENT_DATA_PATH
    script_dir = os.path.dirname(os.path.abspath(__file__))
    path = data_path or os.path.join(script_dir, 'data/spy.csv')
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing {path}")

    print(f"Loading data from {path}...")
    df = process_data(path)
    print(f"Initial shape: {df.shape}")

    # Add regime and save
    base_name = os.path.splitext(os.path.basename(path))[0]
    out_dir = os.path.join(script_dir, 'data')
    if regime_detector == 'kmeans':
        output_path = os.path.join(out_dir, f"{base_name}_kmeans_regime.csv")
        df = add_regime_and_save(df, regime_detector='kmeans', output_path=output_path, n_clusters=n_clusters, **regime_kwargs)
        print(f"K-means regime added (n_clusters={n_clusters}). Saved to {output_path}")
    elif regime_detector == 'gmm':
        output_path = os.path.join(out_dir, f"{base_name}_gmm_regime.csv")
        df = add_regime_and_save(df, regime_detector='gmm', output_path=output_path, n_components=regime_kwargs.get('n_components', 3), **regime_kwargs)
        print(f"GMM regime added. Saved to {output_path}")
    elif regime_detector == 'hmm':
        output_path = os.path.join(out_dir, f"{base_name}_hmm_regime.csv")
        df = add_regime_and_save(df, regime_detector='hmm', output_path=output_path, **regime_kwargs)
        print(f"HMM regime added. Saved to {output_path}")
    else:
        raise ValueError(f"regime_detector must be 'kmeans', 'gmm', or 'hmm'; got {regime_detector}")

    _CURRENT_DATA_PATH = output_path
    df, feature_cols = feature_engineering(df)
    print(f"Features (with regime): {feature_cols}")

    if do_hyperopt:
        print(f"\nRunning Hyperopt ({HYPEROPT_TRIALS} trials) on regime dataset...")
        trials = Trials()
        best = fmin(
            fn=objective,
            space=hyperopt_space(),
            algo=tpe.suggest,
            max_evals=HYPEROPT_TRIALS,
            rstate=np.random.default_rng(SEED),
            trials=trials,
            verbose=1
        )
        input_windows = [21, 42, 63]
        scaler_names = ['robust', 'standard']
        units_1 = [32, 64, 128]
        units_2 = [0, 32, 64]
        batch_sizes = [16, 32, 64]
        patiences = [8, 12, 15]
        best_params = {
            'input_window': input_windows[best['input_window']],
            'scaler': scaler_names[best['scaler']],
            'lstm_units_1': units_1[best['lstm_units_1']],
            'lstm_units_2': units_2[best['lstm_units_2']],
            'dropout': best['dropout'],
            'l2': float(best['l2']),
            'lr': best['lr'],
            'batch_size': batch_sizes[best['batch_size']],
            'patience': patiences[best['patience']],
        }
        print("\nBest hyperparameters:", best_params)
    else:
        best_params = {
            'input_window': 21,
            'scaler': 'standard',
            'lstm_units_1': 64,
            'lstm_units_2': 32,
            'dropout': 0.3,
            'l2': 1e-4,
            'lr': 0.001,
            'batch_size': 32,
            'patience': 10,
        }

    scaler_class = RobustScaler if best_params['scaler'] == 'robust' else StandardScaler
    X_train, X_val, X_test, y_train, y_val, y_test, scaler_X = scale_and_split(
        df, feature_cols, best_params['input_window'], scaler_class
    )
    print(f"\nSequence shapes: train {X_train.shape}, val {X_val.shape}, test {X_test.shape}")
    model, val_r2 = train_model(X_train, y_train, X_val, y_val, best_params, verbose=1)
    print(f"Validation R²: {val_r2:.4f}")
    test_r2 = evaluate_and_print(model, X_test, y_test, y_train)
    print(f"\nTest R²: {test_r2:.4f}")

    os.makedirs(os.path.join(script_dir, 'model'), exist_ok=True)
    model_name = f"lstm_{regime_detector}_regime_21d.keras"
    model.save(os.path.join(script_dir, 'model', model_name))
    print(f"Model saved to model/{model_name}.")
    _CURRENT_DATA_PATH = None
    return model, test_r2


if __name__ == '__main__':
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == '--regime':
        detector = sys.argv[2] if len(sys.argv) > 2 else 'kmeans'
        n_clusters = int(sys.argv[3]) if len(sys.argv) > 3 else 3
        run_pipeline_with_regime(regime_detector=detector, n_clusters=n_clusters, do_hyperopt=False)
    else:
        main()

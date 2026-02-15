"""
Vanilla LSTM baseline for 21-day SPY return prediction.
Uses past N trading days of features to predict cumulative return over the next 21 days.
No target scaling (y in raw return space); X scaled with Robust/StandardScaler.
Hyperopt (20 trials) maximizes validation R²; best model is retrained and evaluated on test.

Device: Uses whatever Keras/TensorFlow sees (CPU or GPU). On Mac M1, use the
project's venv-gpu (Python 3.11 + tensorflow-metal) for Metal GPU; the main
venv (Python 3.13) has no tensorflow-metal wheel, so it runs on CPU only.
Run with GPU:  source ../venv-gpu/bin/activate && python vanilla_lstm.py
"""
import os
import random
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout
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


def print_device_info():
    """Print whether CPU or GPU (e.g. Metal on M1) is available for Keras/TF."""
    try:
        import tensorflow as tf
        devices = tf.config.list_physical_devices()
        gpus = [d for d in devices if 'GPU' in d.name]
        print("Device(s) available:", [d.name for d in devices])
        if gpus:
            print("Using GPU (Metal on M1 if tensorflow-metal is installed).")
        else:
            print("Using CPU only. On Mac M1, install tensorflow-metal for GPU: pip install tensorflow-metal")
    except Exception as e:
        print("Could not detect devices:", e)


def set_seed(seed=SEED):
    np.random.seed(seed)
    random.seed(seed)
    try:
        K.set_random_seed(seed)
    except Exception:
        pass


def process_data():
    """Load the dataset and handle dates."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(script_dir, 'data/spy.csv')
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Missing {file_path}")
    df = pd.read_csv(file_path)
    df['Date'] = pd.to_datetime(df['Date'])
    df.set_index('Date', inplace=True)
    return df


def feature_engineering(df):
    """Creates the 21-day forward target (no scaling)."""
    indexer = pd.api.indexers.FixedForwardWindowIndexer(window_size=FORECAST_HORIZON)
    df['Target_21d'] = df['SPY_Returns'].rolling(window=indexer).sum().shift(-1)
    df.dropna(inplace=True)
    feature_cols = ['SPY_Returns', 'FFR_Lagged', 'VIX_Close', 'CPI_YoY', 'IWM_Returns']
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


def build_model(input_shape, lstm_units_1, lstm_units_2, dropout, lr):
    """One or two LSTM layers; lstm_units_2=0 means single layer."""
    model = Sequential()
    if lstm_units_2 is None or lstm_units_2 <= 0:
        model.add(LSTM(lstm_units_1, activation='relu', input_shape=input_shape, return_sequences=False))
    else:
        model.add(LSTM(lstm_units_1, activation='relu', input_shape=input_shape, return_sequences=True))
        model.add(Dropout(dropout))
        model.add(LSTM(lstm_units_2, activation='relu', return_sequences=False))
    model.add(Dropout(dropout))
    model.add(Dense(1))
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
        lr=params['lr']
    )
    early_stop = EarlyStopping(
        monitor='val_loss',
        patience=int(params['patience']),
        restore_best_weights=True
    )
    model.fit(
        X_train, y_train,
        epochs=80,
        batch_size=int(params['batch_size']),
        validation_data=(X_val, y_val),
        callbacks=[early_stop],
        verbose=verbose
    )
    val_pred = model.predict(X_val, verbose=0)
    val_r2 = r2_score(np.ravel(y_val), np.ravel(val_pred))
    return model, val_r2


def hyperopt_space():
    return {
        'input_window': hp.choice('input_window', [21, 42, 63]),
        'scaler': hp.choice('scaler', ['robust', 'standard']),
        'lstm_units_1': hp.choice('lstm_units_1', [32, 64, 128]),
        'lstm_units_2': hp.choice('lstm_units_2', [0, 32, 64]),
        'dropout': hp.uniform('dropout', 0.2, 0.45),
        'lr': hp.loguniform('lr', np.log(1e-4), np.log(1e-1)),
        'batch_size': hp.choice('batch_size', [16, 32, 64]),
        'patience': hp.choice('patience', [5, 10, 15]),
    }


def objective(params):
    """Minimize negative validation R²."""
    set_seed(SEED)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(script_dir, 'data/spy.csv')
    df = pd.read_csv(file_path)
    df['Date'] = pd.to_datetime(df['Date'])
    df.set_index('Date', inplace=True)
    df, feature_cols = feature_engineering(df)
    scaler_class = RobustScaler if params['scaler'] == 'robust' else StandardScaler
    try:
        X_train, X_val, X_test, y_train, y_val, y_test, _ = scale_and_split(
            df, feature_cols, params['input_window'], scaler_class
        )
    except Exception:
        return {'loss': 0.0, 'status': STATUS_OK}
    _, val_r2 = train_model(X_train, y_train, X_val, y_val, params, verbose=0)
    return {'loss': -val_r2, 'status': STATUS_OK}


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
    print_device_info()
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
    patiences = [5, 10, 15]
    best_params = {
        'input_window': input_windows[best['input_window']],
        'scaler': scaler_names[best['scaler']],
        'lstm_units_1': units_1[best['lstm_units_1']],
        'lstm_units_2': units_2[best['lstm_units_2']],
        'dropout': best['dropout'],
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


if __name__ == '__main__':
    main()

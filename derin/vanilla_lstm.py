import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout
from keras.callbacks import EarlyStopping

# --- CONFIGURATION ---
INPUT_WINDOW = 21       # Model looks at the past 21 trading days (approx 1 month)
FORECAST_HORIZON = 21    # Model predicts the cumulative return over the NEXT 21 days
TEST_SPLIT = 0.2        # 20% of data for testing

def process_data():
    """Load the dataset and handle dates."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(script_dir, 'data/spy.csv')
    
    print(f"Loading data from: {file_path}")
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Missing {file_path}")

    df = pd.read_csv(file_path)
    df['Date'] = pd.to_datetime(df['Date'])
    df.set_index('Date', inplace=True)
    return df

def feature_engineering(df):
    """
    Creates the 21-day forward target and scales the data.
    """
    print("Performing feature engineering...")
    
    # 1. CREATE TARGET: 21-Day Forward Cumulative Log Return
    # We use a rolling sum reversed.
    # Logic: Sum the SPY_Returns of t+1, t+2 ... t+21.
    # Because these are Log Returns, Summing = Compounding.
    indexer = pd.api.indexers.FixedForwardWindowIndexer(window_size=FORECAST_HORIZON)
    df['Target_21d'] = df['SPY_Returns'].rolling(window=indexer).sum().shift(-1)
    
    # Drop the rows at the very end where we don't have 21 days of future data
    df.dropna(inplace=True)
    
    # 2. DEFINE FEATURES
    feature_cols = ['SPY_Returns', 'FFR_Lagged', 'VIX_Close', 'CPI_YoY', 'IWM_Returns']
    
    # 3. SCALE DATA (0 to 1)
    scaler_X = MinMaxScaler()
    scaler_y = MinMaxScaler()
    
    X_scaled = scaler_X.fit_transform(df[feature_cols])
    # Reshape target to (N, 1) for scaling
    y_scaled = scaler_y.fit_transform(df[['Target_21d']])
    
    return X_scaled, y_scaled, scaler_X, scaler_y, feature_cols

def create_sequences(X, y, window_size):
    """
    Input: 2D array of (Samples, Features)
    Output: 3D array for LSTM (Samples, Time Steps, Features)
    """
    Xs, ys = [], []
    # We stop a bit early so we don't run out of data
    for i in range(len(X) - window_size):
        Xs.append(X[i:(i + window_size)])
        ys.append(y[i + window_size]) 
        # Note: y is already shifted in feature_engineering, so y[i + window_size] 
        # is the target corresponding to the sequence ending at i + window_size
        
    return np.array(Xs), np.array(ys)

def build_model(input_shape):
    """Vanilla LSTM Architecture"""
    model = Sequential()
    # 50 Units, Return Sequences=False (Many-to-One)
    model.add(LSTM(50, activation='relu', input_shape=input_shape, return_sequences=False))
    model.add(Dropout(0.2))
    model.add(Dense(1)) # Predicting a single continuous value (Return)
    model.compile(optimizer='adam', loss='mae') # Using MAE as loss function
    return model

def main():
    # 1. Load
    df = process_data()
    print(f"Initial Shape: {df.shape}")
    
    # 2. Prepare Data
    X_scaled, y_scaled, scaler_X, scaler_y, features = feature_engineering(df)
    
    # 3. Create Rolling Window Sequences
    # Input: Past 21 Days | Output: Next 21-Day Return
    X_seq, y_seq = create_sequences(X_scaled, y_scaled, INPUT_WINDOW)
    
    print(f"Sequence Shape (X): {X_seq.shape}") # (Samples, 21, 5)
    print(f"Target Shape (y):   {y_seq.shape}")
    
    # 4. Train/Test Split (Respecting Time)
    split_idx = int(len(X_seq) * (1 - TEST_SPLIT))
    
    X_train, X_test = X_seq[:split_idx], X_seq[split_idx:]
    y_train, y_test = y_seq[:split_idx], y_seq[split_idx:]
    
    print(f"Training Samples: {len(X_train)} | Testing Samples: {len(X_test)}")
    
    # 5. Build & Train
    model = build_model((X_train.shape[1], X_train.shape[2]))
    
    early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    
    history = model.fit(
        X_train, y_train,
        epochs=50,
        batch_size=32,
        validation_data=(X_test, y_test),
        callbacks=[early_stop],
        verbose=1
    )
    
    # 6. Evaluate
    print("\n--- EVALUATION ---")
    preds_scaled = model.predict(X_test)
    
    # Inverse scale to get actual Log Return values
    preds_actual = scaler_y.inverse_transform(preds_scaled)
    y_test_actual = scaler_y.inverse_transform(y_test)
    
    # Metric: MAE
    mae = mean_absolute_error(y_test_actual, preds_actual)
    print(f"Mean Absolute Error (MAE): {mae:.5f}")
    print(f"Interpretation: On average, the model's prediction for the 21-day return is off by {mae*100:.2f}%")

    # Save
    model.save('vanilla_lstm_21d.keras')
    print("Model saved.")

if __name__ == '__main__':
    main()
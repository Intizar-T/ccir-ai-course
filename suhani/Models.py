"""
Admission prediction models: FNN, SVR, Dummy, Linear Regression, XGBoost, Random Forest.
Supports random and time-aware train/test splits.
"""

import numpy as np
import pandas as pd
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
import xgboost as xgb

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
TARGET_COL = "Number of Admissions"
DROP_COLS = ["Number of Admissions", "Timestamp"]
SPLIT_DATE = "2018-11-01"
TEST_SIZE = 0.2
RANDOM_STATE = 42


def load_data(data_path: str) -> pd.DataFrame:
    """Load feature matrix from CSV."""
    return pd.read_csv(data_path)


def prepare_X_y(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Extract feature matrix X and target y."""
    y = df[TARGET_COL]
    X = df.drop(columns=DROP_COLS)
    return X, y


def split_random(
    X: pd.DataFrame,
    y: pd.Series,
    test_size: float = TEST_SIZE,
    random_state: int = RANDOM_STATE,
) -> tuple:
    """Random train/test split."""
    return train_test_split(
        X, y, test_size=test_size, random_state=random_state, shuffle=True
    )


def split_time_aware(
    df: pd.DataFrame,
    X: pd.DataFrame,
    y: pd.Series,
    date_col: str = "Timestamp",
    split_date: str = SPLIT_DATE,
) -> tuple:
    """Time-aware train/test split: train before split_date, test on or after."""
    train_mask = df[date_col] < split_date
    test_mask = df[date_col] >= split_date
    return (
        X.loc[train_mask],
        X.loc[test_mask],
        y.loc[train_mask],
        y.loc[test_mask],
    )


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """Compute R², RMSE, MAE."""
    return {
        "r2": r2_score(y_true, y_pred),
        "rmse": np.sqrt(mean_squared_error(y_true, y_pred)),
        "mae": mean_absolute_error(y_true, y_pred),
    }


def print_metrics(metrics: dict, model_name: str) -> None:
    """Print regression metrics with a label."""
    print(f"{model_name}")
    print("R²   :", metrics["r2"])
    print("RMSE :", metrics["rmse"])
    print("MAE  :", metrics["mae"])
    print()


def train_and_evaluate(
    model,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    name: str,
) -> dict:
    """Fit model, predict on test set, print and return metrics."""
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    metrics = regression_metrics(y_test.values, y_pred)
    print_metrics(metrics, name)
    return metrics


# ---------------------------------------------------------------------------
# Model builders (same hyperparameters as original notebook)
# ---------------------------------------------------------------------------
def build_fnn_pipeline() -> Pipeline:
    return Pipeline([
        ("scaler", StandardScaler()),
        (
            "fnn",
            MLPRegressor(
                hidden_layer_sizes=(64, 32),
                activation="relu",
                solver="adam",
                alpha=1e-4,
                learning_rate_init=0.001,
                max_iter=1000,
                early_stopping=True,
                random_state=RANDOM_STATE,
            ),
        ),
    ])


def build_svr_pipeline() -> Pipeline:
    return Pipeline([
        ("scaler", StandardScaler()),
        ("svr", SVR(kernel="rbf", C=10, epsilon=0.1)),
    ])


def build_linear_pipeline() -> Pipeline:
    return Pipeline([
        ("scaler", StandardScaler()),
        ("linreg", LinearRegression()),
    ])


def build_xgb_regressor():
    return xgb.XGBRegressor(
        n_estimators=300,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.85,
        colsample_bytree=0.85,
        objective="reg:squarederror",
        random_state=RANDOM_STATE,
    )


def build_rf_regressor():
    return RandomForestRegressor(
        n_estimators=300,
        max_depth=None,
        min_samples_leaf=5,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )


# ---------------------------------------------------------------------------
# Main: load data, run all experiments
# ---------------------------------------------------------------------------
def run_all_models(
    data_path: str = "data/all_features_data.csv",
) -> None:
    """Load data and run all regression models with random and time-aware splits."""
    df = load_data(data_path)
    X, y = prepare_X_y(df)

    # Random split (shared for models that use it)
    X_train_r, X_test_r, y_train_r, y_test_r = split_random(X, y)

    # Time-aware split
    X_train_t, X_test_t, y_train_t, y_test_t = split_time_aware(df, X, y)

    # ---- FNN ----
    train_and_evaluate(
        build_fnn_pipeline(),
        X_train_r, y_train_r, X_test_r, y_test_r,
        "FNN (Random Split)",
    )
    train_and_evaluate(
        build_fnn_pipeline(),
        X_train_t, y_train_t, X_test_t, y_test_t,
        "FNN (Time-Aware Split)",
    )

    # ---- SVR ----
    train_and_evaluate(
        build_svr_pipeline(),
        X_train_r, y_train_r, X_test_r, y_test_r,
        "SVR (Random Split)",
    )

    # ---- Dummy ----
    train_and_evaluate(
        DummyRegressor(strategy="mean"),
        X_train_r, y_train_r, X_test_r, y_test_r,
        "Dummy Regressor (Random Split)",
    )

    # ---- Linear Regression ----
    train_and_evaluate(
        build_linear_pipeline(),
        X_train_r, y_train_r, X_test_r, y_test_r,
        "Linear Regression (Random Split)",
    )

    # ---- XGBoost ----
    train_and_evaluate(
        build_xgb_regressor(),
        X_train_r, y_train_r, X_test_r, y_test_r,
        "XGBoost (Random Split)",
    )
    train_and_evaluate(
        build_xgb_regressor(),
        X_train_t, y_train_t, X_test_t, y_test_t,
        "XGBoost (Time-Aware Split)",
    )

    # ---- Random Forest ----
    train_and_evaluate(
        build_rf_regressor(),
        X_train_r, y_train_r, X_test_r, y_test_r,
        "Random Forest (Random Split)",
    )
    train_and_evaluate(
        build_rf_regressor(),
        X_train_t, y_train_t, X_test_t, y_test_t,
        "Random Forest (Time-Aware Split)",
    )


if __name__ == "__main__":
    run_all_models()

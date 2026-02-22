"""
Emergency vs OPD admission prediction.

Loads feature and admission data, merges daily Emergency/OPD counts by date,
trains XGBoost regressors to predict daily Emergency_Count and OPD_Count
from the feature set (non-time-aware, no lag features).
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import xgboost as xgb


# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------

all_features_data = pd.read_csv(
    "data/all_features_data.csv"
)

admission_data = pd.read_csv(
    "data/HDHI Admission Data.csv"
)


# ---------------------------------------------------------------------------
# Parse dates and fix D.O.A (date of admission) ambiguity
# ---------------------------------------------------------------------------

admission_data["monthyear_parsed"] = pd.to_datetime(
    admission_data["month year"],
    format="%b-%y",
    errors="coerce"
)

admission_data["doa_md"] = pd.to_datetime(
    admission_data["D.O.A"],
    dayfirst=False,
    errors="coerce"
)

admission_data["doa_dm"] = pd.to_datetime(
    admission_data["D.O.A"],
    dayfirst=True,
    errors="coerce"
)


def fix_doa(row):
    correct_month = row["monthyear_parsed"].month
    if row["doa_md"].month == correct_month:
        return row["doa_md"]
    return row["doa_dm"]


admission_data["DOA_fixed"] = admission_data.apply(fix_doa, axis=1)
admission_data["Date"] = admission_data["DOA_fixed"].dt.normalize()


# ---------------------------------------------------------------------------
# Daily counts by type (Emergency vs OPD)
# ---------------------------------------------------------------------------

daily_type_counts = (
    admission_data
    .groupby(
        ["Date", "TYPE OF ADMISSION-EMERGENCY/OPD"]
    )
    .size()
    .unstack(fill_value=0)
    .reset_index()
)

daily_type_counts = daily_type_counts.rename(columns={
    "E": "Emergency_Count",
    "O": "OPD_Count"
})

daily_type_counts["Timestamp"] = daily_type_counts["Date"]


# ---------------------------------------------------------------------------
# Merge daily counts into feature data
# ---------------------------------------------------------------------------

all_features_data["Timestamp"] = pd.to_datetime(
    all_features_data["Timestamp"]
).dt.normalize()

all_features_data = all_features_data.merge(
    daily_type_counts[["Timestamp", "Emergency_Count", "OPD_Count"]],
    on="Timestamp",
    how="left"
)

# Fill missing days with 0 admissions
all_features_data[["Emergency_Count", "OPD_Count"]] = (
    all_features_data[["Emergency_Count", "OPD_Count"]].fillna(0)
)


# ---------------------------------------------------------------------------
# Targets and feature matrix
# ---------------------------------------------------------------------------

y_emergency = all_features_data["Emergency_Count"]
y_opd = all_features_data["OPD_Count"]

EXCLUDE_COLS = [
    "Timestamp",
    "Emergency_Count",
    "OPD_Count",
    "Number of Admissions"
]

X = all_features_data.drop(columns=EXCLUDE_COLS)

# Extra safety: drop any column whose name suggests admission/emergency/opd/count
X = X.loc[
    :,
    ~X.columns.str.contains(
        "admission|emerg|opd|count",
        case=False,
        regex=True
    )
]

print("Features used:", X.shape[1])


# ---------------------------------------------------------------------------
# Train/test split (same split for both targets)
# ---------------------------------------------------------------------------

X_train, X_test, y_emg_train, y_emg_test = train_test_split(
    X,
    y_emergency,
    test_size=0.2,
    random_state=42
)

_, _, y_opd_train, y_opd_test = train_test_split(
    X,
    y_opd,
    test_size=0.2,
    random_state=42
)


# ---------------------------------------------------------------------------
# Train XGBoost and report metrics
# ---------------------------------------------------------------------------

def train_xgb(X_train, X_test, y_train, y_test, label):
    model = xgb.XGBRegressor(
        n_estimators=300,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.85,
        colsample_bytree=0.85,
        objective="reg:squarederror",
        random_state=42
    )

    model.fit(X_train, y_train)
    preds = model.predict(X_test)

    print(f"\n=== {label} ===")
    print("R²:", r2_score(y_test, preds))
    print("RMSE:", np.sqrt(mean_squared_error(y_test, preds)))
    print("MAE:", mean_absolute_error(y_test, preds))

    return model


xgb_emergency = train_xgb(
    X_train,
    X_test,
    y_emg_train,
    y_emg_test,
    "Emergency (non-time-aware, no lag)"
)

xgb_opd = train_xgb(
    X_train,
    X_test,
    y_opd_train,
    y_opd_test,
    "OPD (non-time-aware, no lag)"
)

# Converted from emergency_vs_opd.ipynb

import pandas as pd
import matplotlib.pyplot as plt
import pandas as pd  
import pylab as pl
import numpy as np
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from sklearn.metrics import accuracy_score

all_features_data = pd.read_csv("/Users/suhaniagarwal/Downloads/all_features_data.csv")

admission_data = pd.read_csv("/Users/suhaniagarwal/Downloads/HDHI Admission Data.csv")

type_admission = all_features_data.copy()

admission_data.head(40)









all_features_data







import pandas as pd

# ===============================
# 1. Load admission data
# ===============================
admission_data = pd.read_csv("/Users/suhaniagarwal/Downloads/HDHI Admission Data.csv")
# ===============================
# 2. Parse trusted month-year
# ===============================
admission_data["monthyear_parsed"] = pd.to_datetime(
    admission_data["month year"],
    format="%b-%y",
    errors="coerce"
)

# ===============================
# 3. Parse D.O.A in both formats
# ===============================
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

# ===============================
# 4. Deterministic fix (NO NaT)
# ===============================
def fix_doa(row):
    correct_month = row["monthyear_parsed"].month

    # If month-day matches month-year → correct
    if row["doa_md"].month == correct_month:
        return row["doa_md"]

    # Otherwise → swap day/month
    return row["doa_dm"]

admission_data["DOA_fixed"] = admission_data.apply(fix_doa, axis=1)

# Normalize to date only
admission_data["Date"] = admission_data["DOA_fixed"].dt.normalize()

# ===============================
# 5. HARD VALIDATION (must be True)
# ===============================
print(
    "All months correct:",
    (
        admission_data["Date"].dt.month
        == admission_data["monthyear_parsed"].dt.month
    ).all()
)

# ===============================
# 6. Daily admissions (correct counts)
# ===============================
daily_admissions = (
    admission_data
    .groupby("Date")
    .size()
    .reset_index(name="Number of Admissions")
    .sort_values("Date")
)

# ===============================
# 7. Final sanity checks
# ===============================
print("Any NaTs in Date:", daily_admissions["Date"].isna().any())
print(daily_admissions.head())

import pandas as pd

# --- Normalize dates so they align ---
daily_admissions["Date"] = pd.to_datetime(daily_admissions["Date"]).dt.normalize()
all_features_data["Timestamp"] = pd.to_datetime(
    all_features_data["Timestamp"]
).dt.normalize()

# --- Rename Date → Timestamp so merge key is identical ---
daily_admissions_cmp = daily_admissions.rename(
    columns={"Date": "Timestamp"}
)

# --- Keep only what we need from all_features_data ---
all_features_cmp = all_features_data[
    ["Timestamp", "Number of Admissions"]
]

# --- Merge ONLY on common dates ---
comparison = pd.merge(
    daily_admissions_cmp,
    all_features_cmp,
    on="Timestamp",
    how="inner"
)

# --- Compare counts ---
comparison["matches"] = (
    comparison["Number of Admissions_x"]
    == comparison["Number of Admissions_y"]
)

print("Rows compared:", len(comparison))
print("All values match:", comparison["matches"].all())

comparison.loc[
    ~comparison["matches"],
    ["Timestamp", "Number of Admissions_x", "Number of Admissions_y"]
].head()

import pandas as pd



# type_admission uses Timestamp
type_admission["Timestamp"] = pd.to_datetime(
    type_admission["Timestamp"]
).dt.normalize()

# admission_data already has Date from your fixed DOA logic
admission_data["Date"] = pd.to_datetime(
    admission_data["Date"]
).dt.normalize()



daily_type_counts = (
    admission_data
    .groupby(["Date", "TYPE OF ADMISSION-EMERGENCY/OPD"])
    .size()
    .unstack(fill_value=0)
    .reset_index()
)

# Rename for clarity
daily_type_counts = daily_type_counts.rename(
    columns={
        "E": "Emergency_Count",
        "O": "OPD_Count"
    }
)



type_admission = type_admission.merge(
    daily_type_counts,
    left_on="Timestamp",
    right_on="Date",
    how="left"
)



# Drop duplicate Date column from admission side
type_admission = type_admission.drop(columns=["Date"])

# Replace NaNs with 0 (days with no admissions)
type_admission[["Emergency_Count", "OPD_Count"]] = (
    type_admission[["Emergency_Count", "OPD_Count"]]
    .fillna(0)
    .astype(int)
)



print(type_admission[["Timestamp", "Emergency_Count", "OPD_Count"]].head())
print("Any NaNs:",
      type_admission[["Emergency_Count", "OPD_Count"]].isna().any().any())

import numpy as np
import xgboost as xgb
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error


type_admission_emerg_lag = (
    type_admission
    .sort_values("Timestamp")
    .reset_index(drop=True)
    .copy()
)


type_admission_emerg_lag["Emergency_lag1"] = (
    type_admission_emerg_lag["Emergency_Count"].shift(1)
)
type_admission_emerg_lag["Emergency_lag3"] = (
    type_admission_emerg_lag["Emergency_Count"].shift(3)
)
type_admission_emerg_lag["Emergency_lag7"] = (
    type_admission_emerg_lag["Emergency_Count"].shift(7)
)

# Drop rows created by lagging
type_admission_emerg_lag = (
    type_admission_emerg_lag
    .dropna()
    .reset_index(drop=True)
)


y_emerg_lag = type_admission_emerg_lag["Emergency_Count"]


X_emerg_lag = type_admission_emerg_lag.drop(
    columns=[
        "Emergency_Count",  # target
        "OPD_Count",        # avoid leakage
        "Timestamp"         # datetime not allowed
    ]
)


split_date = "2018-11-01"

train_idx_emerg = (
    type_admission_emerg_lag["Timestamp"] < split_date
)
test_idx_emerg = (
    type_admission_emerg_lag["Timestamp"] >= split_date
)

X_train_emerg_lag = X_emerg_lag.loc[train_idx_emerg]
X_test_emerg_lag  = X_emerg_lag.loc[test_idx_emerg]

y_train_emerg_lag = y_emerg_lag.loc[train_idx_emerg]
y_test_emerg_lag  = y_emerg_lag.loc[test_idx_emerg]

xgb_emerg_lag = xgb.XGBRegressor(
    n_estimators=400,
    max_depth=4,
    learning_rate=0.05,
    subsample=0.85,
    colsample_bytree=0.85,
    objective="reg:squarederror",
    random_state=42
)


xgb_emerg_lag.fit(X_train_emerg_lag, y_train_emerg_lag)


y_pred_emerg_lag = xgb_emerg_lag.predict(X_test_emerg_lag)

print("EMERGENCY (LAGGED, ISOLATED) RESULTS")
print("-----------------------------------")
print("R²   :", r2_score(y_test_emerg_lag, y_pred_emerg_lag))
print("RMSE:", np.sqrt(mean_squared_error(y_test_emerg_lag, y_pred_emerg_lag)))
print("MAE :", mean_absolute_error(y_test_emerg_lag, y_pred_emerg_lag))

import numpy as np
import xgboost as xgb
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error


y = type_admission["OPD_Count"]


X = type_admission.drop(
    columns=[
        "OPD_Count",        # target
        "Emergency_Count",  # avoid leakage
        "Timestamp"         # datetime not allowed
    ]
)


split_date = "2018-11-01"

train_idx = type_admission["Timestamp"] < split_date
test_idx  = type_admission["Timestamp"] >= split_date

X_train, X_test = X.loc[train_idx], X.loc[test_idx]
y_train, y_test = y.loc[train_idx], y.loc[test_idx]


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


y_pred = model.predict(X_test)

print("OPD (NO LAGS) RESULTS")
print("---------------------")
print("R²   :", r2_score(y_test, y_pred))
print("RMSE:", np.sqrt(mean_squared_error(y_test, y_pred)))
print("MAE :", mean_absolute_error(y_test, y_pred))

import numpy as np
import xgboost as xgb
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error


y_emerg_nolag = type_admission["Emergency_Count"]


X_emerg_nolag = type_admission.drop(
    columns=[
        "Emergency_Count",  # target
        "OPD_Count",        # avoid leakage
        "Timestamp"         # datetime not allowed
    ]
)


split_date = "2018-11-01"

train_idx = type_admission["Timestamp"] < split_date
test_idx  = type_admission["Timestamp"] >= split_date

X_train_emerg_nolag = X_emerg_nolag.loc[train_idx]
X_test_emerg_nolag  = X_emerg_nolag.loc[test_idx]

y_train_emerg_nolag = y_emerg_nolag.loc[train_idx]
y_test_emerg_nolag  = y_emerg_nolag.loc[test_idx]


xgb_emerg_nolag = xgb.XGBRegressor(
    n_estimators=300,
    max_depth=4,
    learning_rate=0.05,
    subsample=0.85,
    colsample_bytree=0.85,
    objective="reg:squarederror",
    random_state=42
)


xgb_emerg_nolag.fit(X_train_emerg_nolag, y_train_emerg_nolag)


y_pred_emerg_nolag = xgb_emerg_nolag.predict(X_test_emerg_nolag)

print("EMERGENCY (NO LAG) RESULTS")
print("--------------------------")
print("R²   :", r2_score(y_test_emerg_nolag, y_pred_emerg_nolag))
print("RMSE:", np.sqrt(mean_squared_error(y_test_emerg_nolag, y_pred_emerg_nolag)))
print("MAE :", mean_absolute_error(y_test_emerg_nolag, y_pred_emerg_nolag))

import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

# ===============================
# Target and features
# ===============================
y_emerg_rand = type_admission["Emergency_Count"]

X_emerg_rand = type_admission.drop(
    columns=[
        "Emergency_Count",
        "OPD_Count",
        "Timestamp"
    ]
)

# ===============================
# Random split
# ===============================
X_train_emerg_rand, X_test_emerg_rand, y_train_emerg_rand, y_test_emerg_rand = (
    train_test_split(
        X_emerg_rand,
        y_emerg_rand,
        test_size=0.2,
        random_state=42,
        shuffle=True
    )
)

# ===============================
# XGBoost model
# ===============================
xgb_emerg_rand = xgb.XGBRegressor(
    n_estimators=300,
    max_depth=4,
    learning_rate=0.05,
    subsample=0.85,
    colsample_bytree=0.85,
    objective="reg:squarederror",
    random_state=42
)

# ===============================
# Train & evaluate
# ===============================
xgb_emerg_rand.fit(X_train_emerg_rand, y_train_emerg_rand)
y_pred_emerg_rand = xgb_emerg_rand.predict(X_test_emerg_rand)

print("EMERGENCY (Random Split)")
print("------------------------")
print("R²   :", r2_score(y_test_emerg_rand, y_pred_emerg_rand))
print("RMSE:", np.sqrt(mean_squared_error(y_test_emerg_rand, y_pred_emerg_rand)))
print("MAE :", mean_absolute_error(y_test_emerg_rand, y_pred_emerg_rand))

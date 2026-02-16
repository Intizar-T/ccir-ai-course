# Converted from 3-5 days lag.ipynb

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

all_features_data.columns.tolist()

y = all_features_data["Number of Admissions"]

# Create lagged features for days 3–5
lags = [3, 4, 5]

lag_data = all_features_data.copy()

# Columns to lag (exclude target + timestamp)
cols_to_lag = lag_data.columns.difference(
    ["Number of Admissions", "Timestamp"]
)

for lag in lags:
    X_shifted = lag_data[cols_to_lag].shift(lag)
    X_shifted.columns = [f"{col}_lag{lag}" for col in cols_to_lag]
    lag_data = pd.concat([lag_data, X_shifted], axis=1)

# Drop rows with NaNs introduced by lagging
lag_data = lag_data.dropna()

# Align target
y_lag_3_5 = y.loc[lag_data.index]

print(lag_data.shape)
print(y_lag_3_5.shape)

all_features_data

lag_data_cat

X_lag = lag_data.drop(columns=["Number of Admissions", "Timestamp"])
y_lag = y_lag_3_5

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X_lag, y_lag, test_size=0.2, shuffle=False
)

import xgboost as xgb

model = xgb.XGBRegressor(
    n_estimators=50,
    max_depth=4,
    learning_rate=0.05,
    random_state=42
)

model.fit(X_train, y_train)

from sklearn.metrics import r2_score, mean_absolute_error

y_pred = model.predict(X_test)
print("R²:", r2_score(y_test, y_pred))
print("MAE:", mean_absolute_error(y_test, y_pred))

y_pred = model.predict(X_test)

mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print("MAE:", mae)
print("RMSE:", rmse)
print("R² score:", r2)

lag_data_cat = lag_data.copy()

threshold = 20

lag_data_cat["high_admission_day"] = (lag_data_cat["Number of Admissions"] >= threshold).astype(int)

X_cat_lag = lag_data_cat.drop(columns=["Number of Admissions", "high_admission_day", "Timestamp"])
y_cat_lag = lag_data_cat["high_admission_day"]

from sklearn.model_selection import train_test_split

X_train_cat_lag, X_test_cat_lag, y_train_cat_lag, y_test_cat_lag = train_test_split(
    X_cat_lag, y_cat_lag,
    test_size=0.2,
    shuffle=False
)

import xgboost as xgb

clf = xgb.XGBClassifier(
    n_estimators=300,
    max_depth=4,
    learning_rate=0.05,
    subsample=0.85,
    colsample_bytree=0.85,
    scale_pos_weight=(y_train_cat_lag == 0).sum() / (y_train_cat_lag == 1).sum(),
    eval_metric="logloss",
    random_state=42
)







clf.fit(X_train_cat_lag, y_train_cat_lag)

y_proba = clf.predict_proba(X_test_cat_lag)[:, 1]
print("ROC–AUC:", roc_auc_score(y_test_cat_lag, y_proba))

from sklearn.metrics import (
    roc_auc_score,
    accuracy_score,
    classification_report,
    confusion_matrix
)

y_proba_cat_lag = clf.predict_proba(X_test_cat_lag)[:, 1]
y_pred_cat_lag = (y_proba_cat_lag >= 0.5).astype(int)

print("Accuracy:", accuracy_score(y_test_cat_lag, y_pred_cat_lag))
print("ROC–AUC:", roc_auc_score(y_test_cat_lag, y_proba_cat_lag))
print(confusion_matrix(y_test_cat_lag, y_pred_cat_lag))
print(classification_report(y_test_cat_lag, y_pred_cat_lag))

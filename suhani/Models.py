# Converted from Models.ipynb

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier   # use MLPRegressor if regression
from sklearn.metrics import accuracy_score, classification_report
import pandas as pd

all_features_data = pd.read_csv("/Users/suhaniagarwal/Downloads/all_features_data.csv")

import numpy as np
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error


y_fnn_rand = all_features_data["Number of Admissions"]

X_fnn_rand = all_features_data.drop(
    columns=["Number of Admissions", "Timestamp"]
)


X_train_fnn_rand, X_test_fnn_rand, y_train_fnn_rand, y_test_fnn_rand = train_test_split(
    X_fnn_rand,
    y_fnn_rand,
    test_size=0.2,
    random_state=42,
    shuffle=True
)


fnn_rand = Pipeline([
    ("scaler", StandardScaler()),
    ("fnn", MLPRegressor(
        hidden_layer_sizes=(64, 32),
        activation="relu",
        solver="adam",
        alpha=1e-4,
        learning_rate_init=0.001,
        max_iter=1000,
        early_stopping=True,
        random_state=42
    ))
])


fnn_rand.fit(X_train_fnn_rand, y_train_fnn_rand)
y_pred_fnn_rand = fnn_rand.predict(X_test_fnn_rand)

print("FNN (Random Split)")
print("R²   :", r2_score(y_test_fnn_rand, y_pred_fnn_rand))
print("RMSE:", np.sqrt(mean_squared_error(y_test_fnn_rand, y_pred_fnn_rand)))
print("MAE :", mean_absolute_error(y_test_fnn_rand, y_pred_fnn_rand))

import numpy as np
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error


y_fnn_time = all_features_data["Number of Admissions"]

X_fnn_time = all_features_data.drop(
    columns=["Number of Admissions", "Timestamp"]
)


split_date = "2018-11-01"

train_idx = all_features_data["Timestamp"] < split_date
test_idx  = all_features_data["Timestamp"] >= split_date

X_train_fnn_time = X_fnn_time.loc[train_idx]
X_test_fnn_time  = X_fnn_time.loc[test_idx]

y_train_fnn_time = y_fnn_time.loc[train_idx]
y_test_fnn_time  = y_fnn_time.loc[test_idx]

fnn_time = Pipeline([
    ("scaler", StandardScaler()),
    ("fnn", MLPRegressor(
        hidden_layer_sizes=(64, 32),
        activation="relu",
        solver="adam",
        alpha=1e-4,
        learning_rate_init=0.001,
        max_iter=1000,
        early_stopping=True,
        random_state=42
    ))
])


fnn_time.fit(X_train_fnn_time, y_train_fnn_time)
y_pred_fnn_time = fnn_time.predict(X_test_fnn_time)

print("FNN (Time-Aware Split)")
print("R²   :", r2_score(y_test_fnn_time, y_pred_fnn_time))
print("RMSE:", np.sqrt(mean_squared_error(y_test_fnn_time, y_pred_fnn_time)))
print("MAE :", mean_absolute_error(y_test_fnn_time, y_pred_fnn_time))

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error


y_svr = all_features_data["Number of Admissions"]

X_svr = all_features_data.drop(
    columns=["Number of Admissions", "Timestamp"]
)

X_train_svr, X_test_svr, y_train_svr, y_test_svr = train_test_split(
    X_svr, y_svr, test_size=0.2, random_state=42, shuffle=True
)


svr_pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("svr", SVR(kernel="rbf", C=10, epsilon=0.1))
])

svr_pipeline.fit(X_train_svr, y_train_svr)

y_pred_svr = svr_pipeline.predict(X_test_svr)

print("SVR (Random Split)")
print("R²   :", r2_score(y_test_svr, y_pred_svr))
print("RMSE:", np.sqrt(mean_squared_error(y_test_svr, y_pred_svr)))
print("MAE :", mean_absolute_error(y_test_svr, y_pred_svr))

from sklearn.dummy import DummyRegressor


y_dummy = all_features_data["Number of Admissions"]

X_dummy = all_features_data.drop(
    columns=["Number of Admissions", "Timestamp"]
)

X_train_dummy, X_test_dummy, y_train_dummy, y_test_dummy = train_test_split(
    X_dummy, y_dummy, test_size=0.2, random_state=42, shuffle=True
)


dummy_reg = DummyRegressor(strategy="mean")
dummy_reg.fit(X_train_dummy, y_train_dummy)

y_pred_dummy = dummy_reg.predict(X_test_dummy)

print("Dummy Regressor (Random Split)")
print("R²   :", r2_score(y_test_dummy, y_pred_dummy))
print("RMSE:", np.sqrt(mean_squared_error(y_test_dummy, y_pred_dummy)))
print("MAE :", mean_absolute_error(y_test_dummy, y_pred_dummy))

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error


y_lin = all_features_data["Number of Admissions"]

X_lin = all_features_data.drop(
    columns=[
        "Number of Admissions",
        "Timestamp"
    ]
)


X_train_lin, X_test_lin, y_train_lin, y_test_lin = train_test_split(
    X_lin,
    y_lin,
    test_size=0.2,
    random_state=42,
    shuffle=True
)


lin_pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("linreg", LinearRegression())
])


lin_pipeline.fit(X_train_lin, y_train_lin)


y_pred_lin = lin_pipeline.predict(X_test_lin)

print("Linear Regression (Random Split)")
print("--------------------------------")
print("R²   :", r2_score(y_test_lin, y_pred_lin))
print("RMSE:", np.sqrt(mean_squared_error(y_test_lin, y_pred_lin)))
print("MAE :", mean_absolute_error(y_test_lin, y_pred_lin))

import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error


y_xgb_rand = all_features_data["Number of Admissions"]

X_xgb_rand = all_features_data.drop(
    columns=["Number of Admissions", "Timestamp"]
)

X_train_xgb_rand, X_test_xgb_rand, y_train_xgb_rand, y_test_xgb_rand = train_test_split(
    X_xgb_rand, y_xgb_rand, test_size=0.2, random_state=42, shuffle=True
)


xgb_rand = xgb.XGBRegressor(
    n_estimators=300,
    max_depth=4,
    learning_rate=0.05,
    subsample=0.85,
    colsample_bytree=0.85,
    objective="reg:squarederror",
    random_state=42
)

xgb_rand.fit(X_train_xgb_rand, y_train_xgb_rand)
y_pred_xgb_rand = xgb_rand.predict(X_test_xgb_rand)

print("XGBoost (Random Split)")
print("R²   :", r2_score(y_test_xgb_rand, y_pred_xgb_rand))
print("RMSE:", np.sqrt(mean_squared_error(y_test_xgb_rand, y_pred_xgb_rand)))
print("MAE :", mean_absolute_error(y_test_xgb_rand, y_pred_xgb_rand))


y_xgb_time = all_features_data["Number of Admissions"]

X_xgb_time = all_features_data.drop(
    columns=["Number of Admissions", "Timestamp"]
)

split_date = "2018-11-01"

train_idx = all_features_data["Timestamp"] < split_date
test_idx  = all_features_data["Timestamp"] >= split_date

X_train_xgb_time = X_xgb_time.loc[train_idx]
X_test_xgb_time  = X_xgb_time.loc[test_idx]

y_train_xgb_time = y_xgb_time.loc[train_idx]
y_test_xgb_time  = y_xgb_time.loc[test_idx]


xgb_time = xgb.XGBRegressor(
    n_estimators=300,
    max_depth=4,
    learning_rate=0.05,
    subsample=0.85,
    colsample_bytree=0.85,
    objective="reg:squarederror",
    random_state=42
)

xgb_time.fit(X_train_xgb_time, y_train_xgb_time)
y_pred_xgb_time = xgb_time.predict(X_test_xgb_time)

print("XGBoost (Time-Aware Split)")
print("R²   :", r2_score(y_test_xgb_time, y_pred_xgb_time))
print("RMSE:", np.sqrt(mean_squared_error(y_test_xgb_time, y_pred_xgb_time)))
print("MAE :", mean_absolute_error(y_test_xgb_time, y_pred_xgb_time))

from sklearn.ensemble import RandomForestRegressor


y_rf_rand = all_features_data["Number of Admissions"]

X_rf_rand = all_features_data.drop(
    columns=["Number of Admissions", "Timestamp"]
)

X_train_rf_rand, X_test_rf_rand, y_train_rf_rand, y_test_rf_rand = train_test_split(
    X_rf_rand, y_rf_rand, test_size=0.2, random_state=42, shuffle=True
)


rf_rand = RandomForestRegressor(
    n_estimators=300,
    max_depth=None,
    min_samples_leaf=5,
    random_state=42,
    n_jobs=-1
)

rf_rand.fit(X_train_rf_rand, y_train_rf_rand)
y_pred_rf_rand = rf_rand.predict(X_test_rf_rand)

print("Random Forest (Random Split)")
print("R²   :", r2_score(y_test_rf_rand, y_pred_rf_rand))
print("RMSE:", np.sqrt(mean_squared_error(y_test_rf_rand, y_pred_rf_rand)))
print("MAE :", mean_absolute_error(y_test_rf_rand, y_pred_rf_rand))


y_rf_time = all_features_data["Number of Admissions"]

X_rf_time = all_features_data.drop(
    columns=["Number of Admissions", "Timestamp"]
)

X_train_rf_time = X_rf_time.loc[train_idx]
X_test_rf_time  = X_rf_time.loc[test_idx]

y_train_rf_time = y_rf_time.loc[train_idx]
y_test_rf_time  = y_rf_time.loc[test_idx]


rf_time = RandomForestRegressor(
    n_estimators=300,
    max_depth=None,
    min_samples_leaf=5,
    random_state=42,
    n_jobs=-1
)

rf_time.fit(X_train_rf_time, y_train_rf_time)
y_pred_rf_time = rf_time.predict(X_test_rf_time)

print("Random Forest (Time-Aware Split)")
print("R²   :", r2_score(y_test_rf_time, y_pred_rf_time))
print("RMSE:", np.sqrt(mean_squared_error(y_test_rf_time, y_pred_rf_time)))
print("MAE :", mean_absolute_error(y_test_rf_time, y_pred_rf_time))

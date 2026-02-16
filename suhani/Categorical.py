# Converted from Categorical.ipynb

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier   # use MLPRegressor if regression
from sklearn.metrics import accuracy_score, classification_report
import pandas as pd

all_features_data = pd.read_csv("/Users/suhaniagarwal/Downloads/all_features_data.csv")

THRESHOLD = 20   # high-admission day if >= 20 admissions

import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix

# ===============================
# Binary target
# ===============================
y_xgb_cls = (all_features_data["Number of Admissions"] >= THRESHOLD).astype(int)

X_xgb_cls = all_features_data.drop(
    columns=["Number of Admissions", "Timestamp"]
)

# ===============================
# Random split
# ===============================
X_train_xgb_cls, X_test_xgb_cls, y_train_xgb_cls, y_test_xgb_cls = train_test_split(
    X_xgb_cls,
    y_xgb_cls,
    test_size=0.2,
    random_state=42,
    shuffle=True
)

# ===============================
# XGBoost Classifier
# ===============================
xgb_cls = xgb.XGBClassifier(
    n_estimators=300,
    max_depth=4,
    learning_rate=0.05,
    subsample=0.85,
    colsample_bytree=0.85,
    objective="binary:logistic",
    eval_metric="logloss",
    random_state=42
)

# ===============================
# Train
# ===============================
xgb_cls.fit(X_train_xgb_cls, y_train_xgb_cls)

# ===============================
# Predict
# ===============================
y_pred_xgb_cls = xgb_cls.predict(X_test_xgb_cls)
y_proba_xgb_cls = xgb_cls.predict_proba(X_test_xgb_cls)[:, 1]

# ===============================
# Evaluate
# ===============================
print("XGBoost Threshold Classification (Random Split)")
print("Accuracy:", accuracy_score(y_test_xgb_cls, y_pred_xgb_cls))
print("ROC AUC :", roc_auc_score(y_test_xgb_cls, y_proba_xgb_cls))
print("Confusion matrix:\n", confusion_matrix(y_test_xgb_cls, y_pred_xgb_cls))

from sklearn.ensemble import RandomForestClassifier

# ===============================
# Binary target
# ===============================
y_rf_cls = (all_features_data["Number of Admissions"] >= THRESHOLD).astype(int)

X_rf_cls = all_features_data.drop(
    columns=["Number of Admissions", "Timestamp"]
)

# ===============================
# Random split
# ===============================
X_train_rf_cls, X_test_rf_cls, y_train_rf_cls, y_test_rf_cls = train_test_split(
    X_rf_cls,
    y_rf_cls,
    test_size=0.2,
    random_state=42,
    shuffle=True
)

# ===============================
# Random Forest Classifier
# ===============================
rf_cls = RandomForestClassifier(
    n_estimators=300,
    max_depth=None,
    min_samples_leaf=5,
    random_state=42,
    n_jobs=-1
)

# ===============================
# Train
# ===============================
rf_cls.fit(X_train_rf_cls, y_train_rf_cls)

# ===============================
# Predict
# ===============================
y_pred_rf_cls = rf_cls.predict(X_test_rf_cls)
y_proba_rf_cls = rf_cls.predict_proba(X_test_rf_cls)[:, 1]

# ===============================
# Evaluate
# ===============================
print("Random Forest Threshold Classification (Random Split)")
print("Accuracy:", accuracy_score(y_test_rf_cls, y_pred_rf_cls))
print("ROC AUC :", roc_auc_score(y_test_rf_cls, y_proba_rf_cls))
print("Confusion matrix:\n", confusion_matrix(y_test_rf_cls, y_pred_rf_cls))

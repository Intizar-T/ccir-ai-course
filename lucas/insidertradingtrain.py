import os
import time
import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.dummy import DummyRegressor, DummyClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report,
    mean_absolute_error, mean_squared_error, r2_score,
    average_precision_score
)
from scipy.stats import spearmanr
from xgboost import XGBClassifier, XGBRegressor
from lightgbm import LGBMClassifier, LGBMRegressor, early_stopping, log_evaluation
import joblib
import warnings
warnings.filterwarnings('ignore')

import matplotlib
matplotlib.use('Agg')   # headless backend, safe for scripts
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.calibration import calibration_curve
from sklearn.metrics import roc_curve, precision_recall_curve

try:
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False

N_OPTUNA_TRIALS = 150


# ---------------------------------------------------------------------------
# Data Loading
# ---------------------------------------------------------------------------

def process_data():
    """
    Loads the insider-trading features dataset from CSV (output of create_dataset.py),
    converts date columns, handles missing values, and drops rows without a valid target.
    Returns a cleaned pandas DataFrame.
    """
    script_dir = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(script_dir, 'features_with_labels.csv')

    if not os.path.isfile(file_path):
        raise FileNotFoundError(
            f"Dataset not found: {file_path}\n"
            "Run create_dataset.py first to generate features_with_labels.csv"
        )
    mtime = datetime.fromtimestamp(os.path.getmtime(file_path))
    print(f"Loading dataset: {file_path}")
    print(f"  (last modified: {mtime.strftime('%Y-%m-%d %H:%M')})")

    df = pd.read_csv(file_path)
    df['transaction_date'] = pd.to_datetime(df['transaction_date'])

    df['shares_traded_pct_missing'] = df['shares_traded_pct'].isna().astype(int)
    df['shares_traded_pct'] = df['shares_traded_pct'].fillna(df['shares_traded_pct'].median())
    df = df.dropna(subset=['label_7td', 'excess_return_7td']).reset_index(drop=True)

    print(f"Dataset shape: {df.shape}")
    print(f"Date range: {df['transaction_date'].min().date()} to {df['transaction_date'].max().date()}")
    print(f"Label distribution (label_7td):\n{df['label_7td'].value_counts().to_string()}")
    print(f"Positive rate: {df['label_7td'].mean():.2%}")
    return df


# ---------------------------------------------------------------------------
# Feature Engineering
# ---------------------------------------------------------------------------

def feature_engineering(df):
    """
    Constructs an expanded feature matrix (~30 features) from the raw DataFrame.

    Includes log transforms, ratio features (value_per_share, price_premium,
    value_to_market_cap), cyclical date encoding, insider-breadth count,
    and interaction terms.
    """
    feat = df.copy()

    # --- Log transforms ---
    for col in ['total_value_bought', 'avg_transaction_value',
                'total_shares_bought', 'close']:
        feat[f'log_{col}'] = np.log1p(feat[col])
    feat['log_num_buy_transactions'] = np.log1p(feat['num_buy_transactions'])
    feat['log_total_shares_count'] = np.log1p(feat['total_shares_count'])

    # --- Ratio features ---
    feat['value_per_share'] = (
        feat['total_value_bought'] / feat['total_shares_bought'].clip(lower=1)
    )
    feat['price_premium'] = (
        feat['value_per_share'] / feat['close'].clip(lower=0.01)
    )
    market_cap = feat['close'] * feat['total_shares_count']
    feat['value_to_market_cap'] = (
        feat['total_value_bought'] / market_cap.clip(lower=1)
    )

    # --- Insider breadth ---
    feat['insider_type_count'] = (
        feat['has_officer_buy'] + feat['has_director_buy']
        + feat['has_ten_pct_owner_buy']
    )

    # --- Cyclical date encoding ---
    month = feat['transaction_date'].dt.month
    dow = feat['transaction_date'].dt.dayofweek
    feat['month_sin'] = np.sin(2 * np.pi * month / 12)
    feat['month_cos'] = np.cos(2 * np.pi * month / 12)
    feat['dow_sin'] = np.sin(2 * np.pi * dow / 5)
    feat['dow_cos'] = np.cos(2 * np.pi * dow / 5)

    # --- Interaction features ---
    feat['officer_director_combo'] = (
        (feat['has_officer_buy'] & feat['has_director_buy']).astype(int)
    )
    feat['log_value_x_transactions'] = (
        feat['log_total_value_bought'] * feat['num_buy_transactions']
    )
    feat['officer_value'] = (
        feat['has_officer_buy'] * feat['log_total_value_bought']
    )
    feat['director_value'] = (
        feat['has_director_buy'] * feat['log_total_value_bought']
    )

    feature_cols = [
        'num_buy_transactions', 'total_shares_bought', 'total_value_bought',
        'avg_transaction_value', 'close', 'shares_traded_pct',
        'has_officer_buy', 'has_director_buy', 'has_ten_pct_owner_buy',
        'shares_traded_pct_missing',
        'log_total_value_bought', 'log_avg_transaction_value',
        'log_total_shares_bought', 'log_close',
        'log_num_buy_transactions', 'log_total_shares_count',
        'value_per_share', 'price_premium', 'value_to_market_cap',
        'insider_type_count',
        'month_sin', 'month_cos', 'dow_sin', 'dow_cos',
        'officer_director_combo', 'log_value_x_transactions',
        'officer_value', 'director_value',
    ]

    X = feat[feature_cols].values.astype(np.float64)
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    y_class = feat['label_7td'].values.astype(int)
    y_reg = feat['excess_return_7td'].values.astype(np.float64)

    print(f"\nFeature matrix shape: {X.shape}")
    print(f"Features ({len(feature_cols)}): {feature_cols}")
    return X, y_class, y_reg, feature_cols


# ---------------------------------------------------------------------------
# Train / Validation / Test Split (time-based)
# ---------------------------------------------------------------------------

def training_data_split(X, y_class, y_reg, dates):
    """
    Chronological 70/15/15 split. Features are StandardScaler-transformed
    using statistics from the training set only.
    """
    sorted_idx = np.argsort(dates.values)
    n = len(sorted_idx)
    train_end = int(n * 0.70)
    val_end = int(n * 0.85)

    train_idx = sorted_idx[:train_end]
    val_idx = sorted_idx[train_end:val_end]
    test_idx = sorted_idx[val_end:]

    X_train, X_val, X_test = X[train_idx], X[val_idx], X[test_idx]
    y_cls_tr, y_cls_va, y_cls_te = (
        y_class[train_idx], y_class[val_idx], y_class[test_idx]
    )
    y_reg_tr, y_reg_va, y_reg_te = (
        y_reg[train_idx], y_reg[val_idx], y_reg[test_idx]
    )

    reg_low, reg_high = np.percentile(y_reg_tr, [2.5, 97.5])
    y_reg_tr = np.clip(y_reg_tr, reg_low, reg_high)
    y_reg_va = np.clip(y_reg_va, reg_low, reg_high)
    y_reg_te = np.clip(y_reg_te, reg_low, reg_high)
    print(f"\nRegression target winsorized to [{reg_low:.4f}, {reg_high:.4f}]")

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)

    print(f"\nData split (chronological):")
    print(f"  Train : {len(train_idx)} samples  (positive rate: {y_cls_tr.mean():.2%})")
    print(f"  Val   : {len(val_idx)} samples  (positive rate: {y_cls_va.mean():.2%})")
    print(f"  Test  : {len(test_idx)} samples  (positive rate: {y_cls_te.mean():.2%})")

    return {
        'X_train': X_train, 'X_val': X_val, 'X_test': X_test,
        'y_class_train': y_cls_tr, 'y_class_val': y_cls_va,
        'y_class_test': y_cls_te,
        'y_reg_train': y_reg_tr, 'y_reg_val': y_reg_va,
        'y_reg_test': y_reg_te,
        'scaler': scaler, 'reg_bounds': (reg_low, reg_high),
    }


# ---------------------------------------------------------------------------
# Model Constructors – Classification (fallbacks when Optuna is off)
# ---------------------------------------------------------------------------

def dummy_classification_model():
    return DummyClassifier(strategy='most_frequent')

def logistic_regression_model():
    return LogisticRegression(
        C=2.0, class_weight='balanced', max_iter=2000,
        solver='lbfgs', random_state=42, penalty='l2',
    )

def random_forest_classification_model():
    return RandomForestClassifier(
        n_estimators=400, max_depth=10, min_samples_split=5,
        min_samples_leaf=4, max_features='sqrt',
        class_weight='balanced', random_state=42, n_jobs=-1,
    )

def xgboost_classification_model():
    return XGBClassifier(
        n_estimators=1200, max_depth=6, learning_rate=0.03,
        subsample=0.78, colsample_bytree=0.78,
        scale_pos_weight=1020 / 315, reg_alpha=0.1, reg_lambda=1.2,
        eval_metric='logloss', random_state=42,
    )

def svm_model():
    return SVC(
        C=8.0, kernel='rbf', gamma='scale', class_weight='balanced',
        probability=True, random_state=42, cache_size=1000,
    )

def lightgbm_classification_model():
    return LGBMClassifier(
        n_estimators=800, num_leaves=31, learning_rate=0.05,
        feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=5,
        is_unbalance=True, random_state=42, verbose=-1, n_jobs=-1,
    )


# ---------------------------------------------------------------------------
# Model Constructors – Regression (fallbacks when Optuna is off)
# ---------------------------------------------------------------------------

def dummy_regression_model():
    return DummyRegressor(strategy='mean')

def ridge_regression_model():
    return Ridge(alpha=1.0)

def random_forest_regression_model():
    return RandomForestRegressor(
        n_estimators=400, max_depth=8, min_samples_split=8,
        min_samples_leaf=6, max_features='sqrt',
        random_state=42, n_jobs=-1,
    )

def xgboost_regression_model():
    return XGBRegressor(
        n_estimators=800, max_depth=5, learning_rate=0.02,
        subsample=0.7, colsample_bytree=0.7,
        reg_alpha=0.3, reg_lambda=2.0, random_state=42,
    )

def lightgbm_regression_model():
    return LGBMRegressor(
        n_estimators=800, num_leaves=31, learning_rate=0.05,
        feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=5,
        random_state=42, verbose=-1, n_jobs=-1,
    )


# ---------------------------------------------------------------------------
# Optuna helpers
# ---------------------------------------------------------------------------

def _make_study(direction='maximize'):
    return optuna.create_study(
        direction=direction,
        sampler=optuna.samplers.TPESampler(seed=42, n_startup_trials=10),
    )


# ---------------------------------------------------------------------------
# Optuna – Classification
# ---------------------------------------------------------------------------

def _optimize_lr_classification(X_train, y_train, X_val, y_val):
    """Tune Logistic Regression over C, penalty (l1/l2/elasticnet)."""
    def objective(trial):
        penalty = trial.suggest_categorical('penalty', ['l1', 'l2', 'elasticnet'])
        params = {
            'C': trial.suggest_float('C', 0.01, 100.0, log=True),
            'penalty': penalty,
            'solver': 'saga', 'max_iter': 3000,
            'class_weight': 'balanced', 'random_state': 42,
        }
        if penalty == 'elasticnet':
            params['l1_ratio'] = trial.suggest_float('l1_ratio', 0.1, 0.9)
        clf = LogisticRegression(**params)
        clf.fit(X_train, y_train)
        return roc_auc_score(y_val, clf.predict_proba(X_val)[:, 1])

    study = _make_study()
    study.optimize(objective, n_trials=N_OPTUNA_TRIALS, show_progress_bar=False)
    best = study.best_params
    best.update({'solver': 'saga', 'max_iter': 3000,
                 'class_weight': 'balanced', 'random_state': 42})
    model = LogisticRegression(**best)
    model.fit(X_train, y_train)
    return model, best


def _optimize_rf_classification(X_train, y_train, X_val, y_val):
    def objective(trial):
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 100, 800),
            'max_depth': trial.suggest_int('max_depth', 4, 16),
            'min_samples_split': trial.suggest_int('min_samples_split', 2, 14),
            'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 12),
            'max_features': trial.suggest_categorical(
                'max_features', ['sqrt', 'log2', None]),
            'class_weight': 'balanced', 'random_state': 42, 'n_jobs': -1,
        }
        clf = RandomForestClassifier(**params)
        clf.fit(X_train, y_train)
        return roc_auc_score(y_val, clf.predict_proba(X_val)[:, 1])

    study = _make_study()
    study.optimize(objective, n_trials=N_OPTUNA_TRIALS, show_progress_bar=False)
    best = study.best_params
    best.update({'class_weight': 'balanced', 'random_state': 42, 'n_jobs': -1})
    model = RandomForestClassifier(**best)
    model.fit(X_train, y_train)
    return model, best


def _optimize_xgb_classification(X_train, y_train, X_val, y_val):
    scale_pos = max(1.0, (y_train == 0).sum() / max(1, (y_train == 1).sum()))

    def objective(trial):
        params = {
            'n_estimators': 2000,
            'max_depth': trial.suggest_int('max_depth', 3, 10),
            'learning_rate': trial.suggest_float('learning_rate', 0.005, 0.15, log=True),
            'subsample': trial.suggest_float('subsample', 0.5, 0.95),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.4, 0.95),
            'colsample_bylevel': trial.suggest_float('colsample_bylevel', 0.5, 1.0),
            'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
            'gamma': trial.suggest_float('gamma', 0.0, 1.0),
            'reg_alpha': trial.suggest_float('reg_alpha', 1e-3, 3.0, log=True),
            'reg_lambda': trial.suggest_float('reg_lambda', 0.5, 5.0),
            'scale_pos_weight': scale_pos,
            'eval_metric': 'logloss', 'random_state': 42,
        }
        clf = XGBClassifier(**params, early_stopping_rounds=80)
        clf.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
        return roc_auc_score(y_val, clf.predict_proba(X_val)[:, 1])

    study = _make_study()
    study.optimize(objective, n_trials=N_OPTUNA_TRIALS, show_progress_bar=False)
    best = study.best_params
    best.update({
        'n_estimators': 2000, 'scale_pos_weight': scale_pos,
        'eval_metric': 'logloss', 'random_state': 42,
    })
    model = XGBClassifier(**best, early_stopping_rounds=80)
    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
    best['n_estimators'] = model.best_iteration + 1
    best.pop('early_stopping_rounds', None)
    return model, best


def _optimize_svm_classification(X_train, y_train, X_val, y_val):
    def objective(trial):
        kernel = trial.suggest_categorical('kernel', ['rbf', 'poly'])
        params = {
            'C': trial.suggest_float('C', 0.1, 50.0, log=True),
            'kernel': kernel,
            'gamma': trial.suggest_categorical('gamma', ['scale', 'auto']),
            'class_weight': 'balanced', 'probability': True,
            'random_state': 42, 'cache_size': 1000,
        }
        if kernel == 'poly':
            params['degree'] = trial.suggest_int('degree', 2, 4)
        clf = SVC(**params)
        clf.fit(X_train, y_train)
        return roc_auc_score(y_val, clf.predict_proba(X_val)[:, 1])

    study = _make_study()
    study.optimize(objective, n_trials=N_OPTUNA_TRIALS, show_progress_bar=False)
    best = study.best_params
    best.update({'class_weight': 'balanced', 'probability': True,
                 'random_state': 42, 'cache_size': 1000})
    model = SVC(**best)
    model.fit(X_train, y_train)
    return model, best


def _optimize_lgbm_classification(X_train, y_train, X_val, y_val):
    scale_pos = max(1.0, (y_train == 0).sum() / max(1, (y_train == 1).sum()))

    def objective(trial):
        params = {
            'n_estimators': 2000,
            'num_leaves': trial.suggest_int('num_leaves', 15, 127),
            'learning_rate': trial.suggest_float('learning_rate', 0.005, 0.15, log=True),
            'feature_fraction': trial.suggest_float('feature_fraction', 0.4, 1.0),
            'bagging_fraction': trial.suggest_float('bagging_fraction', 0.5, 1.0),
            'bagging_freq': trial.suggest_int('bagging_freq', 1, 7),
            'min_child_samples': trial.suggest_int('min_child_samples', 5, 50),
            'lambda_l1': trial.suggest_float('lambda_l1', 1e-3, 3.0, log=True),
            'lambda_l2': trial.suggest_float('lambda_l2', 1e-3, 3.0, log=True),
            'scale_pos_weight': scale_pos,
            'random_state': 42, 'verbose': -1, 'n_jobs': -1,
        }
        clf = LGBMClassifier(**params)
        clf.fit(X_train, y_train, eval_set=[(X_val, y_val)],
                callbacks=[early_stopping(80), log_evaluation(-1)])
        return roc_auc_score(y_val, clf.predict_proba(X_val)[:, 1])

    study = _make_study()
    study.optimize(objective, n_trials=N_OPTUNA_TRIALS, show_progress_bar=False)
    best = study.best_params
    best.update({
        'n_estimators': 2000, 'scale_pos_weight': scale_pos,
        'random_state': 42, 'verbose': -1, 'n_jobs': -1,
    })
    model = LGBMClassifier(**best)
    model.fit(X_train, y_train, eval_set=[(X_val, y_val)],
              callbacks=[early_stopping(80), log_evaluation(-1)])
    best['n_estimators'] = model.best_iteration_ + 1
    return model, best


# ---------------------------------------------------------------------------
# Optuna – Regression
# ---------------------------------------------------------------------------

def _optimize_ridge_regression(X_train, y_train, X_val, y_val):
    def objective(trial):
        alpha = trial.suggest_float('alpha', 0.01, 100.0, log=True)
        reg = Ridge(alpha=alpha)
        reg.fit(X_train, y_train)
        return r2_score(y_val, reg.predict(X_val))

    study = _make_study()
    study.optimize(objective, n_trials=N_OPTUNA_TRIALS, show_progress_bar=False)
    best = study.best_params
    model = Ridge(**best)
    model.fit(X_train, y_train)
    return model, best


def _optimize_rf_regression(X_train, y_train, X_val, y_val):
    def objective(trial):
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 100, 800),
            'max_depth': trial.suggest_int('max_depth', 4, 14),
            'min_samples_split': trial.suggest_int('min_samples_split', 2, 14),
            'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 12),
            'max_features': trial.suggest_categorical(
                'max_features', ['sqrt', 'log2', None]),
            'random_state': 42, 'n_jobs': -1,
        }
        reg = RandomForestRegressor(**params)
        reg.fit(X_train, y_train)
        return r2_score(y_val, reg.predict(X_val))

    study = _make_study()
    study.optimize(objective, n_trials=N_OPTUNA_TRIALS, show_progress_bar=False)
    best = study.best_params
    best.update({'random_state': 42, 'n_jobs': -1})
    model = RandomForestRegressor(**best)
    model.fit(X_train, y_train)
    return model, best


def _optimize_xgb_regression(X_train, y_train, X_val, y_val):
    def objective(trial):
        params = {
            'n_estimators': 2000,
            'max_depth': trial.suggest_int('max_depth', 3, 10),
            'learning_rate': trial.suggest_float('learning_rate', 0.005, 0.1, log=True),
            'subsample': trial.suggest_float('subsample', 0.5, 0.95),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.4, 0.95),
            'colsample_bylevel': trial.suggest_float('colsample_bylevel', 0.5, 1.0),
            'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
            'gamma': trial.suggest_float('gamma', 0.0, 1.0),
            'reg_alpha': trial.suggest_float('reg_alpha', 1e-3, 3.0, log=True),
            'reg_lambda': trial.suggest_float('reg_lambda', 0.5, 5.0),
            'random_state': 42,
        }
        reg = XGBRegressor(**params, early_stopping_rounds=80)
        reg.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
        return r2_score(y_val, reg.predict(X_val))

    study = _make_study()
    study.optimize(objective, n_trials=N_OPTUNA_TRIALS, show_progress_bar=False)
    best = study.best_params
    best.update({'n_estimators': 2000, 'random_state': 42})
    model = XGBRegressor(**best, early_stopping_rounds=80)
    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
    best['n_estimators'] = model.best_iteration + 1
    best.pop('early_stopping_rounds', None)
    return model, best


def _optimize_lgbm_regression(X_train, y_train, X_val, y_val):
    def objective(trial):
        params = {
            'n_estimators': 2000,
            'num_leaves': trial.suggest_int('num_leaves', 15, 127),
            'learning_rate': trial.suggest_float('learning_rate', 0.005, 0.15, log=True),
            'feature_fraction': trial.suggest_float('feature_fraction', 0.4, 1.0),
            'bagging_fraction': trial.suggest_float('bagging_fraction', 0.5, 1.0),
            'bagging_freq': trial.suggest_int('bagging_freq', 1, 7),
            'min_child_samples': trial.suggest_int('min_child_samples', 5, 50),
            'lambda_l1': trial.suggest_float('lambda_l1', 1e-3, 3.0, log=True),
            'lambda_l2': trial.suggest_float('lambda_l2', 1e-3, 3.0, log=True),
            'random_state': 42, 'verbose': -1, 'n_jobs': -1,
        }
        reg = LGBMRegressor(**params)
        reg.fit(X_train, y_train, eval_set=[(X_val, y_val)],
                callbacks=[early_stopping(80), log_evaluation(-1)])
        return r2_score(y_val, reg.predict(X_val))

    study = _make_study()
    study.optimize(objective, n_trials=N_OPTUNA_TRIALS, show_progress_bar=False)
    best = study.best_params
    best.update({
        'n_estimators': 2000, 'random_state': 42,
        'verbose': -1, 'n_jobs': -1,
    })
    model = LGBMRegressor(**best)
    model.fit(X_train, y_train, eval_set=[(X_val, y_val)],
              callbacks=[early_stopping(80), log_evaluation(-1)])
    best['n_estimators'] = model.best_iteration_ + 1
    return model, best


# ---------------------------------------------------------------------------
# Retrain on train+val
# ---------------------------------------------------------------------------

def _retrain_on_trainval(model_class, params, X_train, y_train, X_val, y_val):
    """Re-fit a model with *params* on the concatenation of train + val sets."""
    X_tv = np.concatenate([X_train, X_val])
    y_tv = np.concatenate([y_train, y_val])
    model = model_class(**params)
    model.fit(X_tv, y_tv)
    return model


# ---------------------------------------------------------------------------
# Threshold Optimization
# ---------------------------------------------------------------------------

def find_optimal_threshold(model, X_val, y_val):
    """Sweep thresholds on the val set and return the one that maximises F1."""
    try:
        y_proba = model.predict_proba(X_val)[:, 1]
    except Exception:
        return 0.5
    best_t, best_f1 = 0.5, 0.0
    for t in np.arange(0.10, 0.91, 0.01):
        preds = (y_proba >= t).astype(int)
        f = f1_score(y_val, preds, zero_division=0)
        if f > best_f1:
            best_f1 = f
            best_t = t
    return round(best_t, 2)


# ---------------------------------------------------------------------------
# Stacking Ensemble
# ---------------------------------------------------------------------------

def _build_stacking_classification(models, X_val, y_val, X_test, y_test):
    """
    Level-0: each model's predicted probabilities on val and test.
    Level-1: Logistic Regression meta-learner trained on val predictions.
    """
    val_meta, test_meta = [], []
    for m in models:
        try:
            val_meta.append(m.predict_proba(X_val)[:, 1])
            test_meta.append(m.predict_proba(X_test)[:, 1])
        except Exception:
            val_meta.append(m.predict(X_val).astype(float))
            test_meta.append(m.predict(X_test).astype(float))

    S_val = np.column_stack(val_meta)
    S_test = np.column_stack(test_meta)

    meta = LogisticRegression(max_iter=2000, random_state=42)
    meta.fit(S_val, y_val)

    y_pred = meta.predict(S_test)
    y_proba = meta.predict_proba(S_test)[:, 1]
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    auc = roc_auc_score(y_test, y_proba)
    pr_auc = average_precision_score(y_test, y_proba)
    return {
        'accuracy': acc, 'precision': prec, 'recall': rec,
        'f1': f1, 'roc_auc': auc, 'pr_auc': pr_auc,
    }, meta


def _build_stacking_regression(models, X_val, y_val, X_test, y_test):
    """
    Level-0: each model's predictions on val and test.
    Level-1: Ridge meta-learner trained on val predictions.
    """
    val_meta = np.column_stack([m.predict(X_val) for m in models])
    test_meta = np.column_stack([m.predict(X_test) for m in models])

    meta = Ridge(alpha=1.0)
    meta.fit(val_meta, y_val)

    y_pred = meta.predict(test_meta)
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)
    sp, _ = spearmanr(y_test, y_pred)
    sp = sp if not np.isnan(sp) else 0.0
    sign_acc = np.mean(np.sign(y_test) == np.sign(y_pred))
    return {
        'mae': mae, 'mse': mse, 'rmse': rmse, 'r2': r2,
        'spearman': sp, 'sign_accuracy': sign_acc,
    }, meta


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train_model(model, X_train, y_train, eval_set=None, early_stopping_rounds=None):
    """Fits *model* on training data; supports XGBoost/LightGBM early stopping."""
    start = time.time()
    name = type(model).__name__
    is_xgb = name in ('XGBClassifier', 'XGBRegressor')
    is_lgbm = name in ('LGBMClassifier', 'LGBMRegressor')

    if is_xgb and eval_set and early_stopping_rounds:
        model.set_params(early_stopping_rounds=early_stopping_rounds)
        model.fit(X_train, y_train, eval_set=eval_set, verbose=False)
    elif is_lgbm and eval_set and early_stopping_rounds:
        model.fit(X_train, y_train, eval_set=eval_set,
                  callbacks=[early_stopping(early_stopping_rounds),
                             log_evaluation(-1)])
    else:
        model.fit(X_train, y_train)

    print(f"  Training completed in {time.time() - start:.2f}s")
    return model


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate_model(model, X_test, y_test, task='classification', threshold=None):
    """
    Evaluates a trained model on the test set.
    For classification, an optional *threshold* overrides the default 0.5.
    """
    if task == 'classification':
        try:
            y_proba = model.predict_proba(X_test)[:, 1]
        except Exception:
            y_proba = None

        if threshold is not None and y_proba is not None:
            y_pred = (y_proba >= threshold).astype(int)
        else:
            y_pred = model.predict(X_test)

        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, zero_division=0)
        rec = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        auc = roc_auc_score(y_test, y_proba) if y_proba is not None else float('nan')
        pr_auc = average_precision_score(y_test, y_proba) if y_proba is not None else float('nan')

        thr_str = f" (threshold={threshold:.2f})" if threshold else ""
        print(f"  Accuracy : {acc:.4f}{thr_str}")
        print(f"  Precision: {prec:.4f}")
        print(f"  Recall   : {rec:.4f}")
        print(f"  F1       : {f1:.4f}")
        print(f"  ROC-AUC  : {auc:.4f}")
        print(f"  PR-AUC   : {pr_auc:.4f}")
        print()
        print(classification_report(
            y_test, y_pred,
            target_names=['No Beat (0)', 'Beat Market (1)'], zero_division=0))
        cm = confusion_matrix(y_test, y_pred)
        print(f"  Confusion Matrix:\n{cm}\n")

        return {'accuracy': acc, 'precision': prec, 'recall': rec,
                'f1': f1, 'roc_auc': auc, 'pr_auc': pr_auc}

    else:
        y_pred = model.predict(X_test)
        mae = mean_absolute_error(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_test, y_pred)
        sp, _ = spearmanr(y_test, y_pred)
        sp = sp if not np.isnan(sp) else 0.0
        sign_acc = np.mean(np.sign(y_test) == np.sign(y_pred))

        print(f"  MAE   : {mae:.6f}")
        print(f"  RMSE  : {rmse:.6f}")
        print(f"  R²    : {r2:.6f}")
        print(f"  Spearman ρ : {sp:.4f}")
        print(f"  Sign accuracy : {sign_acc:.4f}\n")

        return {'mae': mae, 'mse': mse, 'rmse': rmse, 'r2': r2,
                'spearman': sp, 'sign_accuracy': sign_acc}


# ---------------------------------------------------------------------------
# Model Persistence
# ---------------------------------------------------------------------------

def save_model(model, model_name, task, scaler=None, feature_names=None):
    script_dir = os.path.dirname(os.path.abspath(__file__))
    models_dir = os.path.join(script_dir, 'models')
    os.makedirs(models_dir, exist_ok=True)

    safe_name = model_name.lower().replace(' ', '_').replace('(', '').replace(')', '')
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    filename = f"{safe_name}_{task}_{timestamp}.joblib"
    filepath = os.path.join(models_dir, filename)

    artifact = {
        'model': model, 'model_name': model_name, 'task': task,
        'scaler': scaler, 'feature_names': feature_names,
        'saved_at': timestamp,
    }
    joblib.dump(artifact, filepath)
    print(f"  Saved -> {filepath}")


# ---------------------------------------------------------------------------
# Main Pipeline
# ---------------------------------------------------------------------------

# Maps model name -> (optuna_func | None, model_class, fallback_constructor)
_CLASSIFICATION_REGISTRY = {
    'Dummy (Baseline)':      (None, None, dummy_classification_model),
    'Logistic Regression':   ('_optimize_lr_classification', LogisticRegression, logistic_regression_model),
    'Random Forest':         ('_optimize_rf_classification', RandomForestClassifier, random_forest_classification_model),
    'XGBoost':               ('_optimize_xgb_classification', XGBClassifier, xgboost_classification_model),
    'SVM':                   ('_optimize_svm_classification', SVC, svm_model),
    'LightGBM':              ('_optimize_lgbm_classification', LGBMClassifier, lightgbm_classification_model),
}

_REGRESSION_REGISTRY = {
    'Dummy (Baseline)':        (None, None, dummy_regression_model),
    'Ridge':                   ('_optimize_ridge_regression', Ridge, ridge_regression_model),
    'Random Forest Regressor': ('_optimize_rf_regression', RandomForestRegressor, random_forest_regression_model),
    'XGBoost Regressor':       ('_optimize_xgb_regression', XGBRegressor, xgboost_regression_model),
    'LightGBM Regressor':     ('_optimize_lgbm_regression', LGBMRegressor, lightgbm_regression_model),
}

# Optuna function name -> callable
_OPTUNA_FUNCS = {
    '_optimize_lr_classification': _optimize_lr_classification,
    '_optimize_rf_classification': _optimize_rf_classification,
    '_optimize_xgb_classification': _optimize_xgb_classification,
    '_optimize_svm_classification': _optimize_svm_classification,
    '_optimize_lgbm_classification': _optimize_lgbm_classification,
    '_optimize_ridge_regression': _optimize_ridge_regression,
    '_optimize_rf_regression': _optimize_rf_regression,
    '_optimize_xgb_regression': _optimize_xgb_regression,
    '_optimize_lgbm_regression': _optimize_lgbm_regression,
}

_EARLY_STOP_MODELS = {'XGBoost', 'LightGBM', 'XGBoost Regressor', 'LightGBM Regressor'}


# ---------------------------------------------------------------------------
# Visualization helpers
# ---------------------------------------------------------------------------

def _get_proba(model_name, clf, X, stackable_clf, stack_meta):
    """Return positive-class probabilities or None. Handles stacking."""
    if model_name == 'Stacking Ensemble':
        if stack_meta is None or stackable_clf is None:
            return None
        try:
            cols = []
            for m in stackable_clf:
                try:
                    cols.append(m.predict_proba(X)[:, 1])
                except Exception:
                    cols.append(m.predict(X).astype(float))
            S = np.column_stack(cols)
            return stack_meta.predict_proba(S)[:, 1]
        except Exception:
            return None
    try:
        return clf.predict_proba(X)[:, 1]
    except Exception:
        return None


def _top2_clf(results_class):
    """Return list of top-2 non-dummy classifier names by PR-AUC."""
    ranked = sorted(
        [(n, m.get('pr_auc', 0)) for n, m in results_class.items()
         if n not in ('Dummy (Baseline)',) and not np.isnan(m.get('pr_auc', float('nan')))],
        key=lambda x: x[1], reverse=True,
    )
    return [n for n, _ in ranked[:2]]


def _top2_reg(results_reg):
    """Return list of top-2 non-dummy regressor names by R²."""
    ranked = sorted(
        [(n, m.get('r2', -999)) for n, m in results_reg.items()
         if n not in ('Dummy (Baseline)',)],
        key=lambda x: x[1], reverse=True,
    )
    return [n for n, _ in ranked[:2]]


def _get_feature_importances(model, feature_names):
    """Return (importances_array, label) or (None, None) for unsupported models."""
    name = type(model).__name__
    if name in ('RandomForestClassifier', 'RandomForestRegressor',
                'XGBClassifier', 'XGBRegressor',
                'LGBMClassifier', 'LGBMRegressor'):
        return model.feature_importances_, 'Feature Importance'
    if name == 'LogisticRegression':
        coef = model.coef_
        return np.abs(coef[0] if coef.ndim > 1 else coef), '|Coefficient|'
    if name == 'Ridge':
        return np.abs(model.coef_), '|Coefficient|'
    return None, None


# ---------------------------------------------------------------------------
# Main visualization function
# ---------------------------------------------------------------------------

def generate_visualizations(
    train_only_clf, train_only_reg,
    stack_meta, stack_reg_meta,
    stackable_clf, stackable_reg,
    results_class, results_reg,
    X_train, X_val, X_test,
    y_class_train, y_class_val, y_class_test,
    y_reg_train, y_reg_val, y_reg_test,
    feature_names, script_dir,
):
    """Generate all 14 research-paper figures and save to figures/ directory."""
    figures_dir = os.path.join(script_dir, 'figures')
    os.makedirs(figures_dir, exist_ok=True)

    sns.set_style('whitegrid')
    plt.rcParams.update({'font.size': 12, 'figure.dpi': 300})

    # Consistent color mapping
    all_clf_names = list(results_class.keys())
    all_reg_names = list(results_reg.keys())
    tab10 = plt.get_cmap('tab10')
    clf_colors = {n: tab10(i % 10) for i, n in enumerate(all_clf_names)}
    reg_colors = {n: tab10(i % 10) for i, n in enumerate(all_reg_names)}

    # Feature group color mapping
    _feature_groups = {
        'raw': ['num_buy_transactions', 'total_shares_bought', 'total_value_bought',
                'avg_transaction_value', 'close', 'shares_traded_pct',
                'has_officer_buy', 'has_director_buy', 'has_ten_pct_owner_buy',
                'shares_traded_pct_missing'],
        'log': ['log_total_value_bought', 'log_avg_transaction_value',
                'log_total_shares_bought', 'log_close',
                'log_num_buy_transactions', 'log_total_shares_count'],
        'ratio': ['value_per_share', 'price_premium', 'value_to_market_cap'],
        'role': ['insider_type_count'],
        'temporal': ['month_sin', 'month_cos', 'dow_sin', 'dow_cos'],
        'interaction': ['officer_director_combo', 'log_value_x_transactions',
                        'officer_value', 'director_value'],
    }
    _group_palette = {
        'raw': '#4e79a7', 'log': '#f28e2b', 'ratio': '#e15759',
        'role': '#76b7b2', 'temporal': '#59a14f', 'interaction': '#b07aa1',
    }

    def _feat_color(feat):
        for grp, feats in _feature_groups.items():
            if feat in feats:
                return _group_palette[grp]
        return '#aaa'

    def _savefig(fig, fname):
        path = os.path.join(figures_dir, fname)
        fig.savefig(path, dpi=300, bbox_inches='tight')
        plt.close(fig)
        print(f"  [VIZ] saved {fname}")

    # ------------------------------------------------------------------ #
    # Figure 1 – Classification model comparison                          #
    # ------------------------------------------------------------------ #
    try:
        metrics_order = ['pr_auc', 'roc_auc', 'f1', 'accuracy']
        metric_labels = ['PR-AUC', 'ROC-AUC', 'F1', 'Accuracy']
        clf_names_sorted = sorted(
            results_class.keys(),
            key=lambda n: results_class[n].get('pr_auc', 0), reverse=True,
        )
        n_models = len(clf_names_sorted)
        n_metrics = len(metrics_order)
        bar_h = 0.18
        offsets = np.linspace(-(n_metrics - 1) / 2, (n_metrics - 1) / 2, n_metrics) * bar_h

        fig, ax = plt.subplots(figsize=(10, max(5, n_models * 0.9)))
        y_pos = np.arange(n_models)
        for mi, (met, lbl) in enumerate(zip(metrics_order, metric_labels)):
            vals = [results_class[n].get(met, 0) for n in clf_names_sorted]
            ax.barh(y_pos + offsets[mi], vals, height=bar_h, label=lbl)

        # Dummy reference lines
        dummy_metrics = results_class.get('Dummy (Baseline)', {})
        for met, lbl, ls in zip(
            ['pr_auc', 'roc_auc'],
            ['Dummy PR-AUC', 'Dummy ROC-AUC'],
            ['--', ':'],
        ):
            v = dummy_metrics.get(met)
            if v and not np.isnan(v):
                ax.axvline(v, color='red', linestyle=ls, linewidth=1.2, label=lbl)

        ax.set_yticks(y_pos)
        ax.set_yticklabels(clf_names_sorted)
        ax.set_xlabel('Score')
        ax.set_title('Classification Model Comparison')
        ax.legend(loc='lower right', fontsize=9)
        ax.set_xlim(0, 1.05)
        fig.tight_layout()
        _savefig(fig, '01_classification_model_comparison.png')
    except Exception as e:
        print(f"  [WARN] skipped fig 01: {e}")

    # ------------------------------------------------------------------ #
    # Figure 2 – Regression model comparison                              #
    # ------------------------------------------------------------------ #
    try:
        reg_names_sorted = sorted(
            results_reg.keys(),
            key=lambda n: results_reg[n].get('r2', -999), reverse=True,
        )
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, max(5, len(reg_names_sorted) * 0.8)))
        y_pos = np.arange(len(reg_names_sorted))
        bar_h = 0.25
        offsets3 = np.array([-bar_h, 0, bar_h])
        metrics3 = ['r2', 'spearman', 'sign_accuracy']
        labels3 = ['R²', 'Spearman ρ', 'Sign Acc']
        for mi, (met, lbl) in enumerate(zip(metrics3, labels3)):
            vals = [results_reg[n].get(met, 0) for n in reg_names_sorted]
            ax1.barh(y_pos + offsets3[mi], vals, height=bar_h, label=lbl)
        ax1.set_yticks(y_pos)
        ax1.set_yticklabels(reg_names_sorted)
        ax1.set_xlabel('Score')
        ax1.set_title('R² / Spearman / Sign Accuracy')
        ax1.legend(fontsize=9)

        rmse_vals = [results_reg[n].get('rmse', 0) for n in reg_names_sorted]
        colors_reg = [reg_colors[n] for n in reg_names_sorted]
        ax2.barh(y_pos, rmse_vals, color=colors_reg)
        ax2.set_yticks(y_pos)
        ax2.set_yticklabels(reg_names_sorted)
        ax2.set_xlabel('RMSE')
        ax2.set_title('RMSE (lower is better)')
        fig.suptitle('Regression Model Comparison', fontsize=14, y=1.01)
        fig.tight_layout()
        _savefig(fig, '02_regression_model_comparison.png')
    except Exception as e:
        print(f"  [WARN] skipped fig 02: {e}")

    # ------------------------------------------------------------------ #
    # Figure 3 – ROC curves                                               #
    # ------------------------------------------------------------------ #
    try:
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.plot([0, 1], [0, 1], 'k--', linewidth=1, label='No skill')
        for name in all_clf_names:
            clf = train_only_clf.get(name)
            proba = _get_proba(name, clf, X_test, stackable_clf, stack_meta)
            if proba is None:
                continue
            fpr, tpr, _ = roc_curve(y_class_test, proba)
            auc_val = results_class[name].get('roc_auc', 0)
            ax.plot(fpr, tpr, color=clf_colors[name], linewidth=1.5,
                    label=f"{name} (AUC={auc_val:.3f})")
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        ax.set_title('ROC Curves – Classification Models')
        ax.legend(fontsize=8, loc='lower right')
        fig.tight_layout()
        _savefig(fig, '03_roc_curves.png')
    except Exception as e:
        print(f"  [WARN] skipped fig 03: {e}")

    # ------------------------------------------------------------------ #
    # Figure 4 – PR curves                                                #
    # ------------------------------------------------------------------ #
    try:
        prevalence = y_class_test.mean()
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.axhline(prevalence, color='red', linestyle='--', linewidth=1,
                   label=f'Prevalence ({prevalence:.3f})')
        for name in all_clf_names:
            clf = train_only_clf.get(name)
            proba = _get_proba(name, clf, X_test, stackable_clf, stack_meta)
            if proba is None:
                continue
            prec_c, rec_c, _ = precision_recall_curve(y_class_test, proba)
            pr_auc_val = results_class[name].get('pr_auc', 0)
            ax.plot(rec_c, prec_c, color=clf_colors[name], linewidth=1.5,
                    label=f"{name} (PR-AUC={pr_auc_val:.3f})")
        ax.set_xlabel('Recall')
        ax.set_ylabel('Precision')
        ax.set_title('Precision-Recall Curves – Classification Models')
        ax.legend(fontsize=8, loc='upper right')
        fig.tight_layout()
        _savefig(fig, '04_pr_curves.png')
    except Exception as e:
        print(f"  [WARN] skipped fig 04: {e}")

    # ------------------------------------------------------------------ #
    # Figures 5 & 6 – Confusion matrices (top-2 classifiers)             #
    # ------------------------------------------------------------------ #
    top2_clf = _top2_clf(results_class)
    for fig_num, model_name in zip([5, 6], top2_clf):
        try:
            clf = train_only_clf.get(model_name)
            proba = _get_proba(model_name, clf, X_test, stackable_clf, stack_meta)
            thr = results_class[model_name].get('threshold', 0.5)
            if proba is not None:
                y_pred_cm = (proba >= thr).astype(int)
            else:
                y_pred_cm = clf.predict(X_test)
            cm = confusion_matrix(y_class_test, y_pred_cm)
            cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)

            fig, ax = plt.subplots(figsize=(6, 5))
            sns.heatmap(cm, annot=False, fmt='d', cmap='Blues', ax=ax,
                        linewidths=0.5, linecolor='gray')
            for i in range(cm.shape[0]):
                for j in range(cm.shape[1]):
                    ax.text(j + 0.5, i + 0.5,
                            f"{cm[i, j]}\n({cm_norm[i, j]:.1%})",
                            ha='center', va='center', fontsize=11,
                            color='white' if cm_norm[i, j] > 0.5 else 'black')
            ax.set_xticklabels(['Pred 0', 'Pred 1'])
            ax.set_yticklabels(['Actual 0', 'Actual 1'], rotation=0)
            ax.set_title(f"Confusion Matrix – {model_name}\n(threshold={thr:.2f})")
            fig.tight_layout()
            fname = f"0{fig_num}_confusion_matrix_top{'1' if fig_num == 5 else '2'}.png"
            _savefig(fig, fname)
        except Exception as e:
            print(f"  [WARN] skipped fig 0{fig_num}: {e}")

    # ------------------------------------------------------------------ #
    # Figures 7 & 8 – Feature importance (top-2 classifiers)             #
    # ------------------------------------------------------------------ #
    for fig_num, model_name in zip([7, 8], top2_clf):
        try:
            clf = train_only_clf.get(model_name)
            imps, imp_label = _get_feature_importances(clf, feature_names)
            if imps is None:
                print(f"  [INFO] fig 0{fig_num} skipped (no importances for {model_name})")
                continue
            top_n = min(20, len(feature_names))
            idx = np.argsort(imps)[-top_n:]
            feats = [feature_names[i] for i in idx]
            vals = imps[idx]
            colors = [_feat_color(f) for f in feats]

            fig, ax = plt.subplots(figsize=(9, max(5, top_n * 0.38)))
            ax.barh(range(top_n), vals, color=colors)
            ax.set_yticks(range(top_n))
            ax.set_yticklabels(feats, fontsize=9)
            ax.set_xlabel(imp_label)
            ax.set_title(f"Feature Importance – {model_name} (Top {top_n})")

            # Legend for groups
            handles = [plt.Rectangle((0, 0), 1, 1, color=c, label=g)
                       for g, c in _group_palette.items()]
            ax.legend(handles=handles, title='Feature Group',
                      fontsize=8, loc='lower right')
            fig.tight_layout()
            suffix = 'top1' if fig_num == 7 else 'top2'
            _savefig(fig, f"0{fig_num}_feature_importance_clf_{suffix}.png")
        except Exception as e:
            print(f"  [WARN] skipped fig 0{fig_num}: {e}")

    # ------------------------------------------------------------------ #
    # Figure 9 – Feature importance (top-1 regressor)                    #
    # ------------------------------------------------------------------ #
    top2_reg_names = _top2_reg(results_reg)
    for fig_num, model_name in zip([9], top2_reg_names[:1]):
        try:
            reg = train_only_reg.get(model_name)
            imps, imp_label = _get_feature_importances(reg, feature_names)
            if imps is None:
                print(f"  [INFO] fig 09 skipped (no importances for {model_name})")
                continue
            top_n = min(20, len(feature_names))
            idx = np.argsort(imps)[-top_n:]
            feats = [feature_names[i] for i in idx]
            vals = imps[idx]
            colors = [_feat_color(f) for f in feats]

            fig, ax = plt.subplots(figsize=(9, max(5, top_n * 0.38)))
            ax.barh(range(top_n), vals, color=colors)
            ax.set_yticks(range(top_n))
            ax.set_yticklabels(feats, fontsize=9)
            ax.set_xlabel(imp_label)
            ax.set_title(f"Feature Importance – {model_name} (Top {top_n})")
            handles = [plt.Rectangle((0, 0), 1, 1, color=c, label=g)
                       for g, c in _group_palette.items()]
            ax.legend(handles=handles, title='Feature Group',
                      fontsize=8, loc='lower right')
            fig.tight_layout()
            _savefig(fig, '09_feature_importance_reg_top1.png')
        except Exception as e:
            print(f"  [WARN] skipped fig 09: {e}")

    # ------------------------------------------------------------------ #
    # Figure 10 – Calibration curves (top-3 classifiers)                 #
    # ------------------------------------------------------------------ #
    try:
        top3_names = [n for n in sorted(
            results_class.keys(),
            key=lambda n: results_class[n].get('pr_auc', 0), reverse=True,
        ) if n != 'Dummy (Baseline)'][:3]

        fig, ax = plt.subplots(figsize=(7, 6))
        ax.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Perfect calibration')
        for name in top3_names:
            clf = train_only_clf.get(name)
            proba = _get_proba(name, clf, X_test, stackable_clf, stack_meta)
            if proba is None:
                continue
            from sklearn.metrics import brier_score_loss
            brier = brier_score_loss(y_class_test, proba)
            prob_true, prob_pred = calibration_curve(y_class_test, proba, n_bins=10)
            ax.plot(prob_pred, prob_true, marker='o', linewidth=1.5,
                    color=clf_colors[name],
                    label=f"{name} (Brier={brier:.3f})")
        ax.set_xlabel('Mean Predicted Probability')
        ax.set_ylabel('Fraction of Positives')
        ax.set_title('Calibration Curves – Top-3 Classifiers')
        ax.legend(fontsize=8)
        fig.tight_layout()
        _savefig(fig, '10_calibration_curves.png')
    except Exception as e:
        print(f"  [WARN] skipped fig 10: {e}")

    # ------------------------------------------------------------------ #
    # Figure 11 – Regression: predicted vs actual                        #
    # ------------------------------------------------------------------ #
    try:
        top1_reg = top2_reg_names[0] if top2_reg_names else None
        if top1_reg:
            reg = train_only_reg.get(top1_reg)
            y_pred_reg = reg.predict(X_test)
            r2_val = results_reg[top1_reg].get('r2', 0)
            sp_val = results_reg[top1_reg].get('spearman', 0)

            fig, ax = plt.subplots(figsize=(7, 6))
            lim = max(np.abs(y_reg_test).max(), np.abs(y_pred_reg).max()) * 1.05
            ax.scatter(y_pred_reg, y_reg_test, alpha=0.3, s=12,
                       c=np.where((y_pred_reg >= 0) & (y_reg_test >= 0), '#4e79a7',
                          np.where((y_pred_reg < 0) & (y_reg_test < 0), '#e15759',
                          '#f28e2b')))
            ax.plot([-lim, lim], [-lim, lim], 'k--', linewidth=1, label='45° line')
            ax.axhline(0, color='gray', linewidth=0.7)
            ax.axvline(0, color='gray', linewidth=0.7)
            ax.set_xlabel('Predicted Excess Return')
            ax.set_ylabel('Actual Excess Return')
            ax.set_title(f"Predicted vs Actual – {top1_reg}")
            ax.text(0.05, 0.92, f"R²={r2_val:.4f}  Spearman={sp_val:.4f}",
                    transform=ax.transAxes, fontsize=10,
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))
            ax.legend()
            fig.tight_layout()
            _savefig(fig, '11_reg_prediction_vs_actual.png')
    except Exception as e:
        print(f"  [WARN] skipped fig 11: {e}")

    # ------------------------------------------------------------------ #
    # Figure 12 – Regression residuals                                   #
    # ------------------------------------------------------------------ #
    try:
        if top1_reg:
            reg = train_only_reg.get(top1_reg)
            y_pred_reg = reg.predict(X_test)
            residuals = y_reg_test - y_pred_reg

            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
            ax1.scatter(y_pred_reg, residuals, alpha=0.3, s=12, color='#4e79a7')
            ax1.axhline(0, color='red', linestyle='--', linewidth=1)
            ax1.set_xlabel('Predicted Excess Return')
            ax1.set_ylabel('Residual')
            ax1.set_title('Residuals vs Predicted')

            from scipy.stats import norm as sp_norm
            ax2.hist(residuals, bins=40, density=True, alpha=0.6,
                     color='#4e79a7', label='Residuals')
            x_range = np.linspace(residuals.min(), residuals.max(), 200)
            mu, sigma = residuals.mean(), residuals.std()
            ax2.plot(x_range, sp_norm.pdf(x_range, mu, sigma),
                     'r-', linewidth=2, label='Normal fit')
            try:
                sns.kdeplot(residuals, ax=ax2, color='orange', linewidth=2, label='KDE')
            except Exception:
                pass
            ax2.set_xlabel('Residual')
            ax2.set_ylabel('Density')
            ax2.set_title('Residual Distribution')
            ax2.legend(fontsize=9)

            fig.suptitle(f"Residual Analysis – {top1_reg}", fontsize=13)
            fig.tight_layout()
            _savefig(fig, '12_reg_residuals.png')
    except Exception as e:
        print(f"  [WARN] skipped fig 12: {e}")

    # ------------------------------------------------------------------ #
    # Figure 13 – Excess return by prediction quartile                   #
    # ------------------------------------------------------------------ #
    try:
        top1_clf_name = top2_clf[0] if top2_clf else None
        fig, axes = plt.subplots(1, 2, figsize=(13, 5))
        panel_info = [
            (top1_clf_name, 'clf', 'Classifier Score', axes[0]),
            (top2_reg_names[0] if top2_reg_names else None, 'reg', 'Regressor Prediction', axes[1]),
        ]
        for model_name, task, xlabel, ax in panel_info:
            if model_name is None:
                ax.set_visible(False)
                continue
            try:
                if task == 'clf':
                    clf = train_only_clf.get(model_name)
                    scores = _get_proba(model_name, clf, X_test, stackable_clf, stack_meta)
                else:
                    reg = train_only_reg.get(model_name)
                    scores = reg.predict(X_test)
                if scores is None:
                    ax.set_visible(False)
                    continue
                quartiles = pd.qcut(scores, q=4, labels=['Q1\n(Low)', 'Q2', 'Q3', 'Q4\n(High)'])
                df_q = pd.DataFrame({'quartile': quartiles, 'excess_return': y_reg_test})
                df_q.boxplot(column='excess_return', by='quartile', ax=ax,
                             flierprops=dict(marker='.', markersize=3, alpha=0.4))
                ax.axhline(0, color='red', linestyle='--', linewidth=1)
                ax.set_title(f"{model_name}")
                ax.set_xlabel(xlabel)
                ax.set_ylabel('Actual Excess Return (7-day)')
                plt.sca(ax)
                plt.title(model_name)
            except Exception as inner_e:
                ax.set_visible(False)
                print(f"  [WARN] fig 13 panel {task}: {inner_e}")
        fig.suptitle('Actual Excess Returns by Prediction Quartile', fontsize=13, y=1.02)
        fig.tight_layout()
        _savefig(fig, '13_excess_return_by_quartile.png')
    except Exception as e:
        print(f"  [WARN] skipped fig 13: {e}")

    # ------------------------------------------------------------------ #
    # Figure 14 – Feature correlation heatmap                            #
    # ------------------------------------------------------------------ #
    try:
        from scipy.stats import spearmanr as sp_corr
        X_all = np.concatenate([X_train, X_val, X_test], axis=0)
        corr_matrix, _ = sp_corr(X_all, axis=0)
        corr_df = pd.DataFrame(corr_matrix, index=feature_names, columns=feature_names)

        mask = np.triu(np.ones_like(corr_matrix, dtype=bool), k=1)

        g = sns.clustermap(
            corr_df, mask=mask, cmap='RdBu_r', vmin=-1, vmax=1,
            figsize=(14, 14), annot=False, linewidths=0.3,
            dendrogram_ratio=0.12, cbar_pos=(0.02, 0.8, 0.03, 0.15),
        )
        g.fig.suptitle('Spearman Feature Correlation (clustered)', y=1.01, fontsize=13)
        path = os.path.join(figures_dir, '14_feature_correlation_heatmap.png')
        g.fig.savefig(path, dpi=300, bbox_inches='tight')
        plt.close(g.fig)
        print("  [VIZ] saved 14_feature_correlation_heatmap.png")
    except Exception as e:
        print(f"  [WARN] skipped fig 14: {e}")

    print(f"\nAll figures saved to: {figures_dir}/")


def main():
    print("=" * 80)
    print("INSIDER TRADING ML TRAINING PIPELINE")
    print("=" * 80)
    if OPTUNA_AVAILABLE:
        print(f"Optuna: ON ({N_OPTUNA_TRIALS} trials per model, all models tuned)")
    else:
        print("Optuna: not installed — using default hyperparameters "
              "(pip install optuna to enable tuning)")

    # 1. Load and process data
    df = process_data()

    # 2. Feature engineering
    X, y_class, y_reg, feature_names = feature_engineering(df)

    # 3. Chronological train / val / test split
    splits = training_data_split(X, y_class, y_reg, df['transaction_date'])
    X_tr, X_va, X_te = splits['X_train'], splits['X_val'], splits['X_test']

    # ==================================================================
    # 4. Classification models
    # ==================================================================
    print("\n" + "=" * 80)
    print("CLASSIFICATION MODELS  (Predict label_7td: stock beats S&P 500 by >5%)")
    print("=" * 80)

    y_cls_tr = splits['y_class_train']
    y_cls_va = splits['y_class_val']
    y_cls_te = splits['y_class_test']

    results_class = {}
    train_only_clf = {}  # for stacking

    for name, (opt_name, model_class, fallback_fn) in _CLASSIFICATION_REGISTRY.items():
        print(f"\n--- {name} ---")

        if OPTUNA_AVAILABLE and opt_name is not None:
            print(f"  Optuna tuning ({name})...")
            opt_func = _OPTUNA_FUNCS[opt_name]
            train_model_only, best_params = opt_func(X_tr, y_cls_tr, X_va, y_cls_va)
            train_only_clf[name] = train_model_only

            # Retrain on train+val with best params for final evaluation
            print("  Retraining on train+val with best params...")
            final_model = _retrain_on_trainval(
                model_class, best_params, X_tr, y_cls_tr, X_va, y_cls_va)
        else:
            model = fallback_fn()
            es_name = name in _EARLY_STOP_MODELS
            train_model_only = train_model(
                model, X_tr, y_cls_tr,
                eval_set=[(X_va, y_cls_va)] if es_name else None,
                early_stopping_rounds=80 if es_name else None,
            )
            train_only_clf[name] = train_model_only
            final_model = train_model_only

        # Evaluate on test (default threshold)
        metrics = evaluate_model(final_model, X_te, y_cls_te, task='classification')

        # Threshold optimisation
        opt_thr = find_optimal_threshold(train_model_only, X_va, y_cls_va)
        if opt_thr != 0.5:
            print(f"  Re-evaluating with optimised threshold = {opt_thr}:")
            metrics_thr = evaluate_model(
                final_model, X_te, y_cls_te,
                task='classification', threshold=opt_thr)
            if metrics_thr['f1'] > metrics['f1']:
                metrics = metrics_thr
                metrics['threshold'] = opt_thr

        results_class[name] = metrics
        save_model(final_model, name, 'classification',
                   scaler=splits['scaler'], feature_names=feature_names)

    # --- Stacking ensemble (classification) ---
    stackable_clf = [m for n, m in train_only_clf.items()
                     if n != 'Dummy (Baseline)']
    stack_meta = None
    stackable = stackable_clf   # existing variable name still used below
    if len(stackable) >= 2:
        print("\n--- Stacking Ensemble (Classification) ---")
        stack_metrics, stack_meta = _build_stacking_classification(  # noqa: F841
            stackable, X_va, y_cls_va, X_te, y_cls_te)
        results_class['Stacking Ensemble'] = stack_metrics
        print(f"  PR-AUC={stack_metrics.get('pr_auc', 0):.4f}  "
              f"ROC-AUC={stack_metrics['roc_auc']:.4f}  "
              f"F1={stack_metrics['f1']:.4f}  Acc={stack_metrics['accuracy']:.4f}")

    # ==================================================================
    # 5. Regression models
    # ==================================================================
    print("\n" + "=" * 80)
    print("REGRESSION MODELS  (Predict excess_return_7td)")
    print("=" * 80)

    y_reg_tr = splits['y_reg_train']
    y_reg_va = splits['y_reg_val']
    y_reg_te = splits['y_reg_test']

    results_reg = {}
    train_only_reg = {}

    for name, (opt_name, model_class, fallback_fn) in _REGRESSION_REGISTRY.items():
        print(f"\n--- {name} ---")

        if OPTUNA_AVAILABLE and opt_name is not None:
            print(f"  Optuna tuning ({name})...")
            opt_func = _OPTUNA_FUNCS[opt_name]
            train_model_only, best_params = opt_func(X_tr, y_reg_tr, X_va, y_reg_va)
            train_only_reg[name] = train_model_only

            print("  Retraining on train+val with best params...")
            final_model = _retrain_on_trainval(
                model_class, best_params, X_tr, y_reg_tr, X_va, y_reg_va)
        else:
            model = fallback_fn()
            es_name = name in _EARLY_STOP_MODELS
            train_model_only = train_model(
                model, X_tr, y_reg_tr,
                eval_set=[(X_va, y_reg_va)] if es_name else None,
                early_stopping_rounds=80 if es_name else None,
            )
            train_only_reg[name] = train_model_only
            final_model = train_model_only

        metrics = evaluate_model(final_model, X_te, y_reg_te, task='regression')
        results_reg[name] = metrics
        save_model(final_model, name, 'regression',
                   scaler=splits['scaler'], feature_names=feature_names)

    # --- Stacking ensemble (regression) ---
    stack_reg_meta = None
    stackable_reg = [m for n, m in train_only_reg.items()
                     if n != 'Dummy (Baseline)']
    if len(stackable_reg) >= 2:
        print("\n--- Stacking Ensemble (Regression) ---")
        stack_reg_metrics, stack_reg_meta = _build_stacking_regression(  # noqa: F841
            stackable_reg, X_va, y_reg_va, X_te, y_reg_te)
        results_reg['Stacking Ensemble'] = stack_reg_metrics
        print(f"  R²={stack_reg_metrics['r2']:.6f}  "
              f"Spearman={stack_reg_metrics['spearman']:.4f}  "
              f"RMSE={stack_reg_metrics['rmse']:.6f}")

    # =================================================================
    # 6. Summary
    # ==================================================================
    print("\n" + "=" * 80)
    print("TRAINING COMPLETE – RESULTS SUMMARY")
    print("=" * 80)

    print("\nClassification Models (sorted by PR-AUC):")
    for name, m in sorted(
        results_class.items(),
        key=lambda x: x[1].get('pr_auc', 0)
            if not np.isnan(x[1].get('pr_auc', 0)) else 0,
        reverse=True,
    ):
        thr = m.get('threshold', '')
        thr_s = f"  thr={thr}" if thr else ""
        print(f"  {name:30s}  PR-AUC={m.get('pr_auc', 0):.4f}  ROC-AUC={m.get('roc_auc', 0):.4f}  "
              f"F1={m.get('f1', 0):.4f}  Acc={m.get('accuracy', 0):.4f}{thr_s}")

    print("\nRegression Models (sorted by R²):")
    for name, m in sorted(
        results_reg.items(),
        key=lambda x: x[1].get('r2', -999),
        reverse=True,
    ):
        print(f"  {name:30s}  R²={m.get('r2', 0):.6f}  "
              f"Spearman={m.get('spearman', 0):.4f}  "
              f"SignAcc={m.get('sign_accuracy', 0):.4f}  "
              f"RMSE={m.get('rmse', 0):.6f}")

    print()

    # =================================================================
    # 7. Visualizations
    # ==================================================================
    print("\n" + "=" * 80)
    print("GENERATING RESEARCH PAPER VISUALIZATIONS")
    print("=" * 80)
    generate_visualizations(
        train_only_clf=train_only_clf, train_only_reg=train_only_reg,
        stack_meta=stack_meta, stack_reg_meta=stack_reg_meta,
        stackable_clf=stackable_clf, stackable_reg=stackable_reg,
        results_class=results_class, results_reg=results_reg,
        X_train=X_tr, X_val=X_va, X_test=X_te,
        y_class_train=y_cls_tr, y_class_val=y_cls_va, y_class_test=y_cls_te,
        y_reg_train=y_reg_tr, y_reg_val=y_reg_va, y_reg_test=y_reg_te,
        feature_names=feature_names,
        script_dir=os.path.dirname(os.path.abspath(__file__)),
    )


if __name__ == '__main__':
    main()

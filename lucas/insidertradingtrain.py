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
    mean_absolute_error, mean_squared_error, r2_score
)
from scipy.stats import spearmanr
from xgboost import XGBClassifier, XGBRegressor
from lightgbm import LGBMClassifier, LGBMRegressor, early_stopping, log_evaluation
import joblib
import warnings
warnings.filterwarnings('ignore')

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
    Loads the insider-trading features dataset from CSV, converts date columns,
    handles missing values, and drops rows without a valid target.
    Returns a cleaned pandas DataFrame.
    """
    script_dir = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(script_dir, 'features_with_labels.csv')

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
    return {
        'accuracy': acc, 'precision': prec, 'recall': rec,
        'f1': f1, 'roc_auc': auc,
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

        thr_str = f" (threshold={threshold:.2f})" if threshold else ""
        print(f"  Accuracy : {acc:.4f}{thr_str}")
        print(f"  Precision: {prec:.4f}")
        print(f"  Recall   : {rec:.4f}")
        print(f"  F1       : {f1:.4f}")
        print(f"  ROC-AUC  : {auc:.4f}")
        print()
        print(classification_report(
            y_test, y_pred,
            target_names=['No Beat (0)', 'Beat Market (1)'], zero_division=0))
        cm = confusion_matrix(y_test, y_pred)
        print(f"  Confusion Matrix:\n{cm}\n")

        return {'accuracy': acc, 'precision': prec, 'recall': rec,
                'f1': f1, 'roc_auc': auc}

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
    stackable = [m for n, m in train_only_clf.items()
                 if n != 'Dummy (Baseline)']
    if len(stackable) >= 2:
        print("\n--- Stacking Ensemble (Classification) ---")
        stack_metrics, stack_meta = _build_stacking_classification(
            stackable, X_va, y_cls_va, X_te, y_cls_te)
        results_class['Stacking Ensemble'] = stack_metrics
        print(f"  ROC-AUC={stack_metrics['roc_auc']:.4f}  "
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
    stackable_reg = [m for n, m in train_only_reg.items()
                     if n != 'Dummy (Baseline)']
    if len(stackable_reg) >= 2:
        print("\n--- Stacking Ensemble (Regression) ---")
        stack_reg_metrics, stack_reg_meta = _build_stacking_regression(
            stackable_reg, X_va, y_reg_va, X_te, y_reg_te)
        results_reg['Stacking Ensemble'] = stack_reg_metrics
        print(f"  R²={stack_reg_metrics['r2']:.6f}  "
              f"Spearman={stack_reg_metrics['spearman']:.4f}  "
              f"RMSE={stack_reg_metrics['rmse']:.6f}")

    # ==================================================================
    # 6. Summary
    # ==================================================================
    print("\n" + "=" * 80)
    print("TRAINING COMPLETE – RESULTS SUMMARY")
    print("=" * 80)

    print("\nClassification Models (sorted by ROC-AUC):")
    for name, m in sorted(
        results_class.items(),
        key=lambda x: x[1].get('roc_auc', 0)
            if not np.isnan(x[1].get('roc_auc', 0)) else 0,
        reverse=True,
    ):
        thr = m.get('threshold', '')
        thr_s = f"  thr={thr}" if thr else ""
        print(f"  {name:30s}  ROC-AUC={m.get('roc_auc', 0):.4f}  "
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


if __name__ == '__main__':
    main()

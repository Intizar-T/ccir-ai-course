import os
import time
import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.dummy import DummyRegressor, DummyClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report,
    mean_absolute_error, mean_squared_error, r2_score
)
from sklearn.linear_model import Ridge
from scipy.stats import spearmanr
from xgboost import XGBClassifier, XGBRegressor
import joblib
import warnings
warnings.filterwarnings('ignore')

try:
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False

# Optuna: number of trials per model (increase for better results, decrease for speed)
N_OPTUNA_TRIALS = 50


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

    # Convert transaction_date to datetime
    df['transaction_date'] = pd.to_datetime(df['transaction_date'])

    # Handle missing values in shares_traded_pct (some tickers lack shares outstanding)
    df['shares_traded_pct_missing'] = df['shares_traded_pct'].isna().astype(int)
    df['shares_traded_pct'] = df['shares_traded_pct'].fillna(df['shares_traded_pct'].median())

    # Drop rows where the target columns are missing
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
    Constructs the feature matrix from the raw DataFrame.

    New features created:
    - Log-transformed versions of skewed monetary / share columns
    - Date-based features (month, day of week)
    - Interaction flag: officer AND director buying in the same window

    Returns:
        X            : np.ndarray  – feature matrix
        y_class      : np.ndarray  – binary classification target (label_7td)
        y_reg        : np.ndarray  – continuous regression target (excess_return_7td)
        feature_names: list[str]   – ordered feature names matching X columns
    """
    feat = df.copy()

    # --- Log transforms (add 1 to avoid log(0)) ---
    for col in ['total_value_bought', 'avg_transaction_value', 'total_shares_bought', 'close']:
        feat[f'log_{col}'] = np.log1p(feat[col])

    # --- Date-based features ---
    feat['month'] = feat['transaction_date'].dt.month
    feat['day_of_week'] = feat['transaction_date'].dt.dayofweek  # 0=Mon … 4=Fri

    # --- Interaction features ---
    feat['officer_director_combo'] = (feat['has_officer_buy'] & feat['has_director_buy']).astype(int)

    # --- Select final feature columns ---
    feature_cols = [
        # Raw numeric features
        'num_buy_transactions',
        'total_shares_bought',
        'total_value_bought',
        'avg_transaction_value',
        'close',
        'shares_traded_pct',
        # Binary flags
        'has_officer_buy',
        'has_director_buy',
        'has_ten_pct_owner_buy',
        'shares_traded_pct_missing',
        # Engineered features
        'log_total_value_bought',
        'log_avg_transaction_value',
        'log_total_shares_bought',
        'log_close',
        'month',
        'day_of_week',
        'officer_director_combo',
    ]

    X = feat[feature_cols].values.astype(np.float64)
    y_class = feat['label_7td'].values.astype(int)
    y_reg = feat['excess_return_7td'].values.astype(np.float64)

    print(f"\nFeature matrix shape: {X.shape}")
    print(f"Features: {feature_cols}")

    return X, y_class, y_reg, feature_cols


# ---------------------------------------------------------------------------
# Train / Validation / Test Split (time-based)
# ---------------------------------------------------------------------------

def training_data_split(X, y_class, y_reg, dates):
    """
    Chronological split to avoid look-ahead bias.

    Split ratios (by sorted date order):
        Train : 70 %
        Val   : 15 %
        Test  : 15 %

    Features are scaled with StandardScaler fitted on the training set only.

    Returns a dict containing all split arrays and the fitted scaler.
    """
    # Sort indices by date
    sorted_idx = np.argsort(dates.values)
    n = len(sorted_idx)
    train_end = int(n * 0.70)
    val_end = int(n * 0.85)

    train_idx = sorted_idx[:train_end]
    val_idx = sorted_idx[train_end:val_end]
    test_idx = sorted_idx[val_end:]

    X_train, X_val, X_test = X[train_idx], X[val_idx], X[test_idx]
    y_class_train, y_class_val, y_class_test = y_class[train_idx], y_class[val_idx], y_class[test_idx]
    y_reg_train, y_reg_val, y_reg_test = y_reg[train_idx], y_reg[val_idx], y_reg[test_idx]

    # Winsorize regression target using training-set percentiles (reduces impact of extreme returns)
    reg_low, reg_high = np.percentile(y_reg_train, [2.5, 97.5])
    y_reg_train = np.clip(y_reg_train, reg_low, reg_high)
    y_reg_val = np.clip(y_reg_val, reg_low, reg_high)
    y_reg_test = np.clip(y_reg_test, reg_low, reg_high)
    print(f"\nRegression target winsorized to [{reg_low:.4f}, {reg_high:.4f}] (train 2.5–97.5 pct)")

    # Feature scaling
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)

    print(f"\nData split (chronological):")
    print(f"  Train : {len(train_idx)} samples  (label_7td positive rate: {y_class_train.mean():.2%})")
    print(f"  Val   : {len(val_idx)} samples  (label_7td positive rate: {y_class_val.mean():.2%})")
    print(f"  Test  : {len(test_idx)} samples  (label_7td positive rate: {y_class_test.mean():.2%})")

    return {
        'X_train': X_train, 'X_val': X_val, 'X_test': X_test,
        'y_class_train': y_class_train, 'y_class_val': y_class_val, 'y_class_test': y_class_test,
        'y_reg_train': y_reg_train, 'y_reg_val': y_reg_val, 'y_reg_test': y_reg_test,
        'scaler': scaler, 'reg_bounds': (reg_low, reg_high),
    }


# ---------------------------------------------------------------------------
# Model Constructors – Classification
# ---------------------------------------------------------------------------

def dummy_classification_model():
    """Baseline classifier that always predicts the most frequent class."""
    return DummyClassifier(strategy='most_frequent')


def logistic_regression_model():
    """Logistic Regression: relaxed regularization and strong class balancing."""
    return LogisticRegression(
        C=2.0,
        class_weight='balanced',
        max_iter=2000,
        solver='lbfgs',
        random_state=42,
        penalty='l2',
    )


def random_forest_classification_model():
    """Random Forest Classifier: more trees, moderate depth, balanced leaves."""
    return RandomForestClassifier(
        n_estimators=400,
        max_depth=10,
        min_samples_split=5,
        min_samples_leaf=4,
        max_features='sqrt',
        class_weight='balanced',
        random_state=42,
        n_jobs=-1,
    )


def xgboost_classification_model():
    """XGBoost Classifier: large tree budget + early stopping, L1/L2 reg, exact class weight."""
    return XGBClassifier(
        n_estimators=1200,
        max_depth=6,
        learning_rate=0.03,
        subsample=0.78,
        colsample_bytree=0.78,
        scale_pos_weight=1020 / 315,
        reg_alpha=0.1,
        reg_lambda=1.2,
        eval_metric='logloss',
        random_state=42,
        use_label_encoder=False,
    )


def svm_model():
    """SVC: stronger C for better fit, scale kernel, probability for ROC-AUC."""
    return SVC(
        C=8.0,
        kernel='rbf',
        gamma='scale',
        class_weight='balanced',
        probability=True,
        random_state=42,
        cache_size=1000,
    )


# ---------------------------------------------------------------------------
# Model Constructors – Regression
# ---------------------------------------------------------------------------

def dummy_regression_model():
    """Baseline regressor that always predicts the training-set mean."""
    return DummyRegressor(strategy='mean')


def random_forest_regression_model():
    """Random Forest Regressor: moderate depth and leaf size to avoid fitting noise."""
    return RandomForestRegressor(
        n_estimators=400,
        max_depth=8,
        min_samples_split=8,
        min_samples_leaf=6,
        max_features='sqrt',
        random_state=42,
        n_jobs=-1,
    )


def ridge_regression_model():
    """Ridge regression: linear baseline that often generalizes well for weak signals."""
    return Ridge(alpha=1.0)


def xgboost_regression_model():
    """XGBoost Regressor: stronger L1/L2 to avoid overfitting to noisy returns."""
    return XGBRegressor(
        n_estimators=800,
        max_depth=5,
        learning_rate=0.02,
        subsample=0.7,
        colsample_bytree=0.7,
        reg_alpha=0.3,
        reg_lambda=2.0,
        random_state=42,
    )


# ---------------------------------------------------------------------------
# Optuna hyperparameter optimization
# ---------------------------------------------------------------------------

def _optimize_rf_classification(X_train, y_train, X_val, y_val):
    """Run Optuna study for Random Forest Classifier; return model trained with best params."""
    def objective(trial):
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 100, 600),
            'max_depth': trial.suggest_int('max_depth', 4, 14),
            'min_samples_split': trial.suggest_int('min_samples_split', 2, 12),
            'min_samples_leaf': trial.suggest_int('min_samples_leaf', 2, 10),
            'max_features': trial.suggest_categorical('max_features', ['sqrt', 'log2', None]),
            'class_weight': 'balanced',
            'random_state': 42,
            'n_jobs': -1,
        }
        clf = RandomForestClassifier(**params)
        clf.fit(X_train, y_train)
        y_proba = clf.predict_proba(X_val)[:, 1]
        return roc_auc_score(y_val, y_proba)

    study = optuna.create_study(direction='maximize', sampler=optuna.samplers.TPESampler(seed=42, n_startup_trials=10))
    study.optimize(objective, n_trials=N_OPTUNA_TRIALS, show_progress_bar=False)
    best = study.best_params
    best['class_weight'] = 'balanced'
    best['random_state'] = 42
    best['n_jobs'] = -1
    model = RandomForestClassifier(**best)
    model.fit(X_train, y_train)
    return model


def _optimize_xgb_classification(X_train, y_train, X_val, y_val):
    """Run Optuna study for XGBoost Classifier; return model trained with best params (early stopping on val)."""
    scale_pos = max(1.0, (y_train == 0).sum() / max(1, (y_train == 1).sum()))

    def objective(trial):
        params = {
            'n_estimators': 1500,
            'max_depth': trial.suggest_int('max_depth', 3, 10),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.15, log=True),
            'subsample': trial.suggest_float('subsample', 0.5, 0.95),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 0.95),
            'reg_alpha': trial.suggest_float('reg_alpha', 1e-3, 2.0, log=True),
            'reg_lambda': trial.suggest_float('reg_lambda', 0.5, 3.0),
            'scale_pos_weight': scale_pos,
            'eval_metric': 'logloss',
            'random_state': 42,
            'use_label_encoder': False,
        }
        clf = XGBClassifier(**params)
        clf.set_params(early_stopping_rounds=80)
        clf.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
        y_proba = clf.predict_proba(X_val)[:, 1]
        return roc_auc_score(y_val, y_proba)

    study = optuna.create_study(direction='maximize', sampler=optuna.samplers.TPESampler(seed=42, n_startup_trials=10))
    study.optimize(objective, n_trials=N_OPTUNA_TRIALS, show_progress_bar=False)
    best = study.best_params
    best['n_estimators'] = 1500
    best['scale_pos_weight'] = scale_pos
    best['eval_metric'] = 'logloss'
    best['random_state'] = 42
    best['use_label_encoder'] = False
    model = XGBClassifier(**best)
    model.set_params(early_stopping_rounds=80)
    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
    return model


def _optimize_rf_regression(X_train, y_train, X_val, y_val):
    """Run Optuna study for Random Forest Regressor; return model trained with best params."""
    def objective(trial):
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 100, 500),
            'max_depth': trial.suggest_int('max_depth', 4, 12),
            'min_samples_split': trial.suggest_int('min_samples_split', 2, 12),
            'min_samples_leaf': trial.suggest_int('min_samples_leaf', 2, 10),
            'max_features': trial.suggest_categorical('max_features', ['sqrt', 'log2', None]),
            'random_state': 42,
            'n_jobs': -1,
        }
        reg = RandomForestRegressor(**params)
        reg.fit(X_train, y_train)
        return r2_score(y_val, reg.predict(X_val))

    study = optuna.create_study(direction='maximize', sampler=optuna.samplers.TPESampler(seed=42, n_startup_trials=10))
    study.optimize(objective, n_trials=N_OPTUNA_TRIALS, show_progress_bar=False)
    best = study.best_params
    best['random_state'] = 42
    best['n_jobs'] = -1
    model = RandomForestRegressor(**best)
    model.fit(X_train, y_train)
    return model


def _optimize_xgb_regression(X_train, y_train, X_val, y_val):
    """Run Optuna study for XGBoost Regressor; return model trained with best params (early stopping on val)."""
    def objective(trial):
        params = {
            'n_estimators': 1200,
            'max_depth': trial.suggest_int('max_depth', 3, 9),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1, log=True),
            'subsample': trial.suggest_float('subsample', 0.5, 0.95),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 0.95),
            'reg_alpha': trial.suggest_float('reg_alpha', 1e-3, 2.0, log=True),
            'reg_lambda': trial.suggest_float('reg_lambda', 0.5, 3.0),
            'random_state': 42,
        }
        reg = XGBRegressor(**params)
        reg.set_params(early_stopping_rounds=80)
        reg.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
        return r2_score(y_val, reg.predict(X_val))

    study = optuna.create_study(direction='maximize', sampler=optuna.samplers.TPESampler(seed=42, n_startup_trials=10))
    study.optimize(objective, n_trials=N_OPTUNA_TRIALS, show_progress_bar=False)
    best = study.best_params
    best['n_estimators'] = 1200
    best['random_state'] = 42
    model = XGBRegressor(**best)
    model.set_params(early_stopping_rounds=80)
    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
    return model


# ---------------------------------------------------------------------------
# Stubs kept for compatibility (not used in current pipeline)
# ---------------------------------------------------------------------------

def build_lstm_model():
    """Placeholder – LSTM is not suitable for this tabular dataset."""
    pass


def lstm_model():
    """Placeholder – LSTM is not suitable for this tabular dataset."""
    pass


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train_model(model, X_train, y_train, eval_set=None, early_stopping_rounds=None):
    """
    Fits *model* on the training data and returns the trained model.
    For XGBoost, pass eval_set and early_stopping_rounds to use validation-based
    early stopping (reduces overfitting, often improves test ROC-AUC).
    Prints wall-clock training time.
    """
    start = time.time()
    is_xgb = type(model).__name__ in ('XGBClassifier', 'XGBRegressor')
    if is_xgb and eval_set is not None and early_stopping_rounds is not None:
        model.set_params(early_stopping_rounds=early_stopping_rounds)
        model.fit(X_train, y_train, eval_set=eval_set, verbose=False)
    else:
        model.fit(X_train, y_train)
    elapsed = time.time() - start
    print(f"  Training completed in {elapsed:.2f}s")
    return model


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate_model(model, X_test, y_test, task='classification'):
    """
    Evaluates a trained model on the test set.

    Args:
        task: 'classification' or 'regression'

    Returns a dict of metric_name -> value.
    """
    y_pred = model.predict(X_test)

    if task == 'classification':
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, zero_division=0)
        rec = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)

        # ROC-AUC (needs probability estimates)
        try:
            y_proba = model.predict_proba(X_test)[:, 1]
            auc = roc_auc_score(y_test, y_proba)
        except Exception:
            auc = float('nan')

        print(f"  Accuracy : {acc:.4f}")
        print(f"  Precision: {prec:.4f}")
        print(f"  Recall   : {rec:.4f}")
        print(f"  F1       : {f1:.4f}")
        print(f"  ROC-AUC  : {auc:.4f}")
        print()
        print(classification_report(y_test, y_pred, target_names=['No Beat (0)', 'Beat Market (1)'], zero_division=0))
        cm = confusion_matrix(y_test, y_pred)
        print(f"  Confusion Matrix:\n{cm}\n")

        return {'accuracy': acc, 'precision': prec, 'recall': rec,
                'f1': f1, 'roc_auc': auc}

    else:  # regression
        mae = mean_absolute_error(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_test, y_pred)
        spearman_corr, _ = spearmanr(y_test, y_pred)
        spearman_corr = spearman_corr if not np.isnan(spearman_corr) else 0.0
        sign_acc = np.mean(np.sign(y_test) == np.sign(y_pred))

        print(f"  MAE   : {mae:.6f}")
        print(f"  RMSE  : {rmse:.6f}")
        print(f"  R²    : {r2:.6f}")
        print(f"  Spearman ρ : {spearman_corr:.4f}  (rank correlation)")
        print(f"  Sign accuracy : {sign_acc:.4f}  (correct direction)\n")

        return {'mae': mae, 'mse': mse, 'rmse': rmse, 'r2': r2,
                'spearman': spearman_corr, 'sign_accuracy': sign_acc}


# ---------------------------------------------------------------------------
# Model Persistence
# ---------------------------------------------------------------------------

def save_model(model, model_name, task, scaler=None, feature_names=None):
    """
    Persists a trained model (and optionally the scaler / feature list)
    to the models/ directory using joblib.
    """
    script_dir = os.path.dirname(os.path.abspath(__file__))
    models_dir = os.path.join(script_dir, 'models')
    os.makedirs(models_dir, exist_ok=True)

    safe_name = model_name.lower().replace(' ', '_')
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    filename = f"{safe_name}_{task}_{timestamp}.joblib"
    filepath = os.path.join(models_dir, filename)

    artifact = {
        'model': model,
        'model_name': model_name,
        'task': task,
        'scaler': scaler,
        'feature_names': feature_names,
        'saved_at': timestamp,
    }
    joblib.dump(artifact, filepath)
    print(f"  Saved → {filepath}")


# ---------------------------------------------------------------------------
# Main Pipeline
# ---------------------------------------------------------------------------

def main():
    print("=" * 80)
    print("INSIDER TRADING ML TRAINING PIPELINE")
    print("=" * 80)
    if OPTUNA_AVAILABLE:
        print("Optuna: ON (hyperparameter tuning for Random Forest & XGBoost)")
    else:
        print("Optuna: not installed — using default hyperparameters (pip install optuna to enable tuning)")

    # 1. Load and process data
    df = process_data()

    # 2. Feature engineering
    X, y_class, y_reg, feature_names = feature_engineering(df)

    # 3. Chronological train / val / test split
    splits = training_data_split(X, y_class, y_reg, df['transaction_date'])

    # ------------------------------------------------------------------
    # 4. Classification models  (predict label_7td)
    # ------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("CLASSIFICATION MODELS  (Predict label_7td: stock beats S&P 500 by >5%)")
    print("=" * 80)

    # Classification: use Optuna for RF and XGBoost when available; else default models
    classification_models = {
        'Dummy (Baseline)': dummy_classification_model(),
        'Logistic Regression': logistic_regression_model(),
        'Random Forest': random_forest_classification_model(),
        'XGBoost': xgboost_classification_model(),
        'SVM': svm_model(),
    }

    results_class = {}
    for name, model in classification_models.items():
        print(f"\n--- {name} ---")
        use_optuna_rf = OPTUNA_AVAILABLE and name == 'Random Forest'
        use_optuna_xgb = OPTUNA_AVAILABLE and name == 'XGBoost'
        if use_optuna_rf:
            print("  Optuna tuning (Random Forest)...")
            trained = _optimize_rf_classification(
                splits['X_train'], splits['y_class_train'],
                splits['X_val'], splits['y_class_val'],
            )
        elif use_optuna_xgb:
            print("  Optuna tuning (XGBoost)...")
            trained = _optimize_xgb_classification(
                splits['X_train'], splits['y_class_train'],
                splits['X_val'], splits['y_class_val'],
            )
        else:
            use_early_stop = 'XGBoost' in name
            trained = train_model(
                model, splits['X_train'], splits['y_class_train'],
                eval_set=[(splits['X_val'], splits['y_class_val'])] if use_early_stop else None,
                early_stopping_rounds=80 if use_early_stop else None,
            )
        metrics = evaluate_model(trained, splits['X_test'], splits['y_class_test'],
                                 task='classification')
        results_class[name] = metrics
        save_model(trained, name, 'classification',
                   scaler=splits['scaler'], feature_names=feature_names)

    # ------------------------------------------------------------------
    # 5. Regression models  (predict excess_return_7td)
    # ------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("REGRESSION MODELS  (Predict excess_return_7td)")
    print("=" * 80)

    # Regression: use Optuna for RF and XGBoost when available; else default models
    regression_models = {
        'Dummy (Baseline)': dummy_regression_model(),
        'Ridge': ridge_regression_model(),
        'Random Forest Regressor': random_forest_regression_model(),
        'XGBoost Regressor': xgboost_regression_model(),
    }

    results_reg = {}
    for name, model in regression_models.items():
        print(f"\n--- {name} ---")
        use_optuna_rf = OPTUNA_AVAILABLE and name == 'Random Forest Regressor'
        use_optuna_xgb = OPTUNA_AVAILABLE and name == 'XGBoost Regressor'
        if use_optuna_rf:
            print("  Optuna tuning (Random Forest)...")
            trained = _optimize_rf_regression(
                splits['X_train'], splits['y_reg_train'],
                splits['X_val'], splits['y_reg_val'],
            )
        elif use_optuna_xgb:
            print("  Optuna tuning (XGBoost)...")
            trained = _optimize_xgb_regression(
                splits['X_train'], splits['y_reg_train'],
                splits['X_val'], splits['y_reg_val'],
            )
        else:
            use_early_stop = 'XGBoost' in name
            trained = train_model(
                model, splits['X_train'], splits['y_reg_train'],
                eval_set=[(splits['X_val'], splits['y_reg_val'])] if use_early_stop else None,
                early_stopping_rounds=80 if use_early_stop else None,
            )
        metrics = evaluate_model(trained, splits['X_test'], splits['y_reg_test'],
                                 task='regression')
        results_reg[name] = metrics
        save_model(trained, name, 'regression',
                   scaler=splits['scaler'], feature_names=feature_names)

    # ------------------------------------------------------------------
    # 6. Summary
    # ------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("TRAINING COMPLETE – RESULTS SUMMARY")
    print("=" * 80)

    print("\nClassification Models (sorted by ROC-AUC):")
    for name, m in sorted(results_class.items(),
                          key=lambda x: x[1].get('roc_auc', 0) if not np.isnan(x[1].get('roc_auc', 0)) else 0,
                          reverse=True):
        print(f"  {name:30s}  ROC-AUC={m.get('roc_auc', 0):.4f}  "
              f"F1={m.get('f1', 0):.4f}  Accuracy={m.get('accuracy', 0):.4f}")

    print("\nRegression Models (sorted by R²):")
    for name, m in sorted(results_reg.items(),
                          key=lambda x: x[1].get('r2', -999),
                          reverse=True):
        print(f"  {name:30s}  R²={m.get('r2', 0):.6f}  "
              f"Spearman={m.get('spearman', 0):.4f}  SignAcc={m.get('sign_accuracy', 0):.4f}  "
              f"RMSE={m.get('rmse', 0):.6f}")

    print()


if __name__ == '__main__':
    main()

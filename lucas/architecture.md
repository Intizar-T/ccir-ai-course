# Project Architecture

## Purpose

This repository is a script-first machine learning research project that studies whether **SEC Form 4 insider buying activity** has predictive power for **short-horizon stock outperformance**.

The project focuses on a specific research question:

- Do insider buy signals, especially from officers and directors, help predict whether a stock will **beat the S&P 500 by more than 5% over the next 7 trading days**?
- Can those same signals predict the **continuous 7-trading-day excess return**?

The current codebase is organized around two main executable scripts:

- `create_dataset.py`: builds the cleaned and labeled dataset from raw SEC filings plus Yahoo Finance market data.
- `insidertradingtrain.py`: trains and evaluates classification and regression models on that dataset.

This is not a Python package with modules and tests. It is a research repo driven primarily by top-level scripts and generated CSV artifacts.

## Authoritative Files

If an agent needs to understand or modify the project, these are the most important files:

- `create_dataset.py`
  Builds the end-to-end dataset from SEC source files, market data, and feature engineering logic.
- `insidertradingtrain.py`
  Defines the training pipeline, model families, hyperparameter tuning, evaluation, and model persistence.
- `Company Tickers.json`
  Maps SEC CIK values to ticker symbols for ticker normalization.
- `_quarters/`
  Raw SEC Form 4 quarterly extracts. Each quarter directory contains:
  - `NONDERIV_TRANS.tsv`
  - `SUBMISSION.tsv`
  - `REPORTINGOWNER.tsv`
- `requirements.txt`
  Training dependencies for the ML pipeline.
- `.gitignore`
  Documents which large artifacts are expected to be regenerated rather than tracked.

Useful but non-authoritative files:

- `AGENTS.md`
  High-level project description, but parts of it are stale relative to the current code.
- `YFINANCE_MIGRATION_SUMMARY.md`
  Historical notes about moving market data loading from a static file to `yfinance`.
- `Market Data Jan 25 2026.txt`
  Legacy market data file. Present in the repo, but not used by the current Python pipeline.
- `model_comparison_results.csv`
  Legacy experiment output. The current `insidertradingtrain.py` does not generate this file.

## Current Repo Shape

At a high level, the project consists of:

- Raw SEC filings in `_quarters/`
- A ticker normalization file in `Company Tickers.json`
- A dataset-construction script in `create_dataset.py`
- A model-training script in `insidertradingtrain.py`
- Generated CSV outputs in the repo root
- Saved trained model artifacts in `models/`

The repo is intentionally flat and easy to inspect, but that also means there is little separation between source code, generated data, and experiment outputs.

## End-to-End Workflow

The intended workflow is:

1. Run `create_dataset.py`
2. Build `features_with_labels.csv`
3. Run `insidertradingtrain.py`
4. Train and evaluate ML models
5. Save trained model artifacts to `models/`

In practice, the pipeline is:

1. Load raw SEC Form 4 data from `_quarters/*/`
2. Normalize tickers with `Company Tickers.json`
3. Filter and clean insider transactions
4. Aggregate **buy** activity into 30-day feature windows
5. Fetch daily stock prices from Yahoo Finance
6. Fetch S&P 500 benchmark data (`^GSPC`) from Yahoo Finance
7. Compute 7-trading-day forward returns and excess returns
8. Join price/return information back to the insider-derived features
9. Create a binary label based on a 5% excess-return threshold
10. Train both classification and regression models on the resulting labeled dataset

## Data Sources

### 1. SEC Form 4 Data

Raw filing data lives under `_quarters/`. Each quarter directory is expected to contain:

- `NONDERIV_TRANS.tsv`
- `SUBMISSION.tsv`
- `REPORTINGOWNER.tsv`

The loader in `create_dataset.py` uses a glob pattern of `_quarters/*/<filename>`, so the quarter folder names can vary, but the three TSVs must be direct children of each quarter directory.

### 2. CIK-to-Ticker Mapping

`Company Tickers.json` is used to map `ISSUERCIK` values to standardized ticker symbols. If a CIK cannot be mapped, the script falls back to the filing's original `ISSUERTRADINGSYMBOL`.

### 3. Market Data

The current pipeline fetches live market data from Yahoo Finance via `yfinance`:

- Per-ticker price history for all discovered insider-trading tickers
- S&P 500 benchmark data for `^GSPC`

This data is used to compute:

- `return_7td`
- `excess_return_7td`

The project previously used a static file, `Market Data Jan 25 2026.txt`, but the current code does not read that file.

## Dataset Construction: `create_dataset.py`

`create_dataset.py` is the ETL and labeling pipeline. It has top-level execution, so importing the file runs the full process as a side effect.

### Step 1: Load and Deduplicate SEC Tables

The script loads all quarterly TSVs and deduplicates them using stable keys:

- `NONDERIV_TRANS.tsv` by `NONDERIV_TRANS_SK`
- `SUBMISSION.tsv` by `ACCESSION_NUMBER`
- `REPORTINGOWNER.tsv` by `ACCESSION_NUMBER` + `RPTOWNERCIK`

### Step 2: Select Relevant Columns

It reduces the raw SEC tables to the fields needed for this project:

- transaction date
- transaction code
- shares
- price per share
- ownership type
- issuer ticker / CIK
- reporting owner relationship information

It also derives insider role flags from `RPTOWNER_RELATIONSHIP`:

- `IS_DIRECTOR`
- `IS_OFFICER`
- `IS_TEN_PERCENT_OWNER`

### Step 3: Merge and Clean Transactions

The script merges the SEC tables on `ACCESSION_NUMBER`, standardizes the ticker, and filters to valid non-derivative transactions where:

- `TRANS_CODE` is `P` or `S`
- shares are positive
- price per share is positive
- date is present
- ticker is present

It then derives:

- `IS_BUY` from `TRANS_CODE == 'P'`
- `TRANSACTION_VALUE = TRANS_SHARES * TRANS_PRICEPERSHARE`

The resulting cleaned transaction table is saved as `transactions_clean.csv`.

### Step 4: Build ML Features from Buys Only

Although both purchases and sales are retained in the cleaned transaction table, the ML feature table is built from **buy transactions only**.

Buys are aggregated into fixed 30-day windows using:

- `window_id = (transaction_date - min_date).days // 30`

Grouped by `ticker` and `window_id`, the script creates:

- `num_buy_transactions`
- `total_shares_bought`
- `total_value_bought`
- `avg_transaction_value`
- `has_officer_buy`
- `has_director_buy`
- `has_ten_pct_owner_buy`

This intermediate feature table is saved as `features_ml_ready.csv`.

### Step 5: Fetch Market Data

The script fetches daily adjusted market data from Yahoo Finance for each ticker over:

- `min(transaction_date) - 90 days`
- `max(transaction_date) + 60 days`

That range exists to support forward-return calculations.

Important operational details:

- Tickers are processed in batches of 100 for progress reporting.
- Each ticker is still downloaded individually inside the batch loop.
- The script also attempts to fetch `sharesOutstanding` via `yf.Ticker(ticker).info`.
- Failed or delisted tickers are skipped rather than crashing the run.
- If `yfinance` is not installed, the script attempts to install it at runtime with `pip`.

### Step 6: Fetch Benchmark Data

The script separately downloads S&P 500 data for `^GSPC` and computes:

- `sp500_return_7td`

This is later used to produce benchmark-relative labels and regression targets.

### Step 7: Filter to Tickers with Market Data

After market data is fetched, the script filters both transactions and feature rows down to tickers that actually have market data. This means the ML dataset is based on the overlap between:

- tickers found in SEC insider filings
- tickers successfully resolved by Yahoo Finance

### Step 8: Compute Returns and Labels

For each ticker, the script computes:

- `close_7td = close.shift(-7)`
- `return_7td = (close_7td - close) / close`

It then subtracts benchmark returns to create:

- `excess_return_7td = return_7td - sp500_return_7td`

The final binary classification label is:

- `label_7td = 1` if `excess_return_7td > 0.05`
- otherwise `0`

So a positive label does **not** just mean "the stock went up." It means the stock beat the S&P 500 by more than 5% over the next 7 trading days.

### Step 9: Join Features to Market Data

The feature table is merged to price/return data on exact:

- `ticker`
- `transaction_date`

This is an important design detail: transactions that fall on dates with no matching market row after cleaning will be dropped from the joined outputs.

Additional feature columns added at this stage include:

- `close`
- `return_7td`
- `excess_return_7td`
- `total_shares_count`
- `shares_traded_pct`

### Step 10: Save Outputs

`create_dataset.py` writes:

- `transactions_clean.csv`
- `transactions_with_prices.csv`
- `features_ml_ready.csv`
- `features_with_labels.csv`
- `market_prices_clean.csv`

Of these, `features_with_labels.csv` is the main handoff artifact to the ML pipeline.

## Training Pipeline: `insidertradingtrain.py`

`insidertradingtrain.py` is the modeling pipeline. Unlike `create_dataset.py`, it is safe to import because it only runs from `main()` under `if __name__ == '__main__':`.

### Input

The script reads:

- `features_with_labels.csv`

If that file does not exist, the script raises an error instructing the user to run `create_dataset.py` first.

### Core Targets

The training script models two targets:

- Classification target: `label_7td`
- Regression target: `excess_return_7td`

This makes the project dual-task by design:

- classification asks a yes/no outperformance question
- regression asks for the magnitude of excess return

### In-Script Feature Engineering

The script starts from the labeled CSV and engineers a 28-feature matrix. It includes:

- Raw magnitude features:
  - `num_buy_transactions`
  - `total_shares_bought`
  - `total_value_bought`
  - `avg_transaction_value`
  - `close`
  - `shares_traded_pct`
  - role flags
- Missingness handling:
  - `shares_traded_pct_missing`
- Log transforms:
  - `log_total_value_bought`
  - `log_avg_transaction_value`
  - `log_total_shares_bought`
  - `log_close`
  - `log_num_buy_transactions`
  - `log_total_shares_count`
- Ratio features:
  - `value_per_share`
  - `price_premium`
  - `value_to_market_cap`
- Breadth / timing / interactions:
  - `insider_type_count`
  - `month_sin`, `month_cos`
  - `dow_sin`, `dow_cos`
  - `officer_director_combo`
  - `log_value_x_transactions`
  - `officer_value`
  - `director_value`

Missing `shares_traded_pct` values are median-imputed and tracked with a missingness flag.

### Train / Validation / Test Split

The script uses a strict chronological split:

- 70% train
- 15% validation
- 15% test

Rows are sorted by `transaction_date` before splitting. This is intended to avoid leakage that would happen with random shuffling in a time-ordered financial dataset.

The script also:

- fits `StandardScaler` on the training set only
- applies the same scaler to validation and test
- winsorizes the regression target using the training-set 2.5th and 97.5th percentiles

There is no cross-validation in the current pipeline.

### Classification Models

Classification models include:

- Dummy baseline
- Logistic Regression
- Random Forest
- XGBoost
- SVM
- LightGBM
- Stacking ensemble with a Logistic Regression meta-learner

For classification, the script evaluates:

- accuracy
- precision
- recall
- F1
- ROC-AUC
- PR-AUC
- confusion matrix
- classification report

It also performs threshold tuning on the validation set by sweeping thresholds from `0.10` to `0.90` and choosing the one with the best F1.

### Regression Models

Regression models include:

- Dummy baseline
- Ridge
- Random Forest Regressor
- XGBoost Regressor
- LightGBM Regressor
- Stacking ensemble with a Ridge meta-learner

Regression metrics include:

- MAE
- MSE
- RMSE
- R²
- Spearman correlation
- sign accuracy

### Hyperparameter Tuning

If `optuna` is installed, the script tunes most non-dummy models using validation performance.

Current setting:

- `N_OPTUNA_TRIALS = 150`

This can make runs significantly slower, especially because multiple classification and regression models are tuned independently.

### Model Persistence

Each non-stacking trained model is saved as a timestamped `.joblib` artifact in `models/`.

Saved artifacts include:

- the fitted model
- model name
- task (`classification` or `regression`)
- scaler
- feature names
- save timestamp

The stacking ensembles are evaluated but are not currently persisted.

## Generated Artifacts

The repo currently mixes code and generated outputs in the root directory.

Common generated artifacts include:

- `transactions_clean.csv`
- `transactions_with_prices.csv`
- `features_ml_ready.csv`
- `features_with_labels.csv`
- `market_prices_clean.csv`
- `models/*.joblib`

Current repo evidence also shows:

- `model_comparison_results.csv`

That file appears to come from an older experiment and is not produced by the current `insidertradingtrain.py`.

## Runtime Characteristics

### `create_dataset.py`

- Requires internet access
- Uses live Yahoo Finance downloads
- Can take several minutes
- May auto-install `yfinance`
- Has top-level execution side effects

### `insidertradingtrain.py`

- Does not make network calls
- Runs entirely from local CSV input
- Can still take a long time because of Optuna tuning and heavier model families
- Uses all CPU cores for some models (`RandomForest`, `LightGBM`)

## Dependency Notes

`requirements.txt` currently includes:

- `pandas`
- `numpy`
- `scikit-learn`
- `scipy`
- `xgboost`
- `lightgbm`
- `joblib`
- `optuna`

Important caveat:

- `create_dataset.py` depends on `yfinance`, but `yfinance` is not currently listed in `requirements.txt`.

## Documentation Drift / Stale References

There are a few places where the repo docs do not fully match the live code:

- `AGENTS.md` describes `insidertrading.ipynb` as the main ETL path, but no notebook is currently present in this directory.
- `AGENTS.md` says the training module is a placeholder and expects `data/features_with_labels.csv`, but the current script is fully implemented and reads `features_with_labels.csv` from the repo root.
- `YFINANCE_MIGRATION_SUMMARY.md` references notebook cells and a notebook-driven flow that is not currently represented by a checked-in `.ipynb` file.
- `Market Data Jan 25 2026.txt` is legacy and not used by the current Python scripts.

For any future agent, the authoritative truth should be taken from:

- `create_dataset.py`
- `insidertradingtrain.py`

not from the older migration notes or notebook references.

## Key Assumptions and Design Choices

These are the most important project assumptions an agent should keep in mind:

- The research target is **benchmark-relative**, not absolute return.
- Positive labels mean **beat S&P 500 by more than 5% in 7 trading days**.
- ML features are built from **buy transactions only**.
- Aggregation is done in fixed **30-day windows**.
- The train/validation/test split is **chronological**, not random.
- Exact date joins to market data can silently exclude some insider events.
- `shares_traded_pct` may be missing and is intentionally imputed with a companion missingness flag.
- `create_dataset.py` is an executable script with side effects on import.

## How To Run

Typical local workflow:

```bash
source .venv/bin/activate
python create_dataset.py
python insidertradingtrain.py
```

If the dataset has already been built and `features_with_labels.csv` is current, you can rerun only:

```bash
source .venv/bin/activate
python insidertradingtrain.py
```

That second step does not fetch from Yahoo Finance again.

## Fast Mental Model For Agents

If you only remember one thing about this repo, remember this:

**This project turns raw SEC Form 4 insider-buy filings into a benchmark-relative 7-day prediction problem.**

The operational pipeline is:

**SEC TSVs -> cleaned insider transactions -> 30-day buy aggregates -> Yahoo Finance price history -> 7-day excess returns vs S&P 500 -> labeled dataset -> classification and regression models**

## Recommended Starting Points For Future Agents

When joining the project, read these files in this order:

1. `architecture.md`
2. `create_dataset.py`
3. `insidertradingtrain.py`
4. `AGENTS.md` for extra background, but treat it as partially stale

If debugging data issues:

- Start in `create_dataset.py`

If debugging model behavior or performance:

- Start in `insidertradingtrain.py`

If a run fails because data is missing:

- Confirm that `features_with_labels.csv` exists
- Otherwise rerun `create_dataset.py`

# AGENTS.md

This file provides guidance to WARP (warp.dev) when working with code in this repository.

## Project Overview

Research project analyzing SEC Form 4 insider trading filings to predict stock price movements using machine learning. The goal is to determine if insider buying signals (particularly from officers and directors) have predictive power for 7-day forward returns.

## Development Commands

```bash
# Activate virtual environment
source .venv/bin/activate

# Run the training script
python insidertradingtrain.py

# Run Jupyter notebook
jupyter notebook insidertrading.ipynb
```

## Architecture

### Data Pipeline (`insidertrading.ipynb`)

The notebook implements a complete ETL pipeline:

1. **SEC Data Loading**: Loads three TSV files from SEC EDGAR Form 4 filings:
   - `NONDERIV_TRANS.tsv` - Individual transactions
   - `SUBMISSION.tsv` - Filing metadata with issuer CIK/ticker
   - `REPORTINGOWNER.tsv` - Insider roles (director/officer/10% owner)

2. **Ticker Standardization**: Uses `Company Tickers.json` (SEC CIK-to-ticker mapping) to normalize ticker symbols across data sources.

3. **Market Data**: Fetches historical prices via yfinance API in batches of 100 tickers. Handles delisted/failed tickers gracefully.

4. **Feature Engineering**: Aggregates insider buying into 30-day windows with features:
   - Transaction counts and values
   - Insider role flags (has_officer_buy, has_director_buy, has_ten_pct_owner_buy)
   - Shares traded as percentage of outstanding shares

5. **Labels**: Binary classification based on 7-day forward returns exceeding 5% threshold.

### ML Module (`insidertradingtrain.py`)

Placeholder structure for model training. Expects data from `data/features_with_labels.csv`. Planned models: LSTM, Random Forest, XGBoost, Logistic Regression, SVM.

### Key Output Files

- `features_with_labels.csv` - Final ML-ready dataset with labels (~1335 rows)
- `transactions_clean.csv` - Cleaned transaction records
- `market_prices_clean.csv` - Price history (gitignored, ~8M rows, regenerated via notebook)

## Data Conventions

- Transaction codes: `P` = purchase, `S` = sale (only P/S are analyzed)
- Dates in transaction data use `transaction_date` column
- Forward returns use adjusted close prices from yfinance
- CIK (Central Index Key) is the SEC's unique company identifier

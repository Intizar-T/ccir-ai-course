"""
Insider Trading Dataset Builder
================================
Loads SEC Form 4 TSV files from multiple quarterly zip extracts (_quarters/),
combines them, fetches market data via yfinance (3+ years), and produces
features_with_labels.csv for the ML training pipeline.
"""

import os
import glob
import pandas as pd
import numpy as np
import json
import sys
import warnings
warnings.filterwarnings('ignore')

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
os.chdir(SCRIPT_DIR)

# ---------------------------------------------------------------------------
# 1. CIK-to-ticker mapping
# ---------------------------------------------------------------------------

with open('Company Tickers.json', 'r') as f:
    cik_ticker_data = json.load(f)

cik_mapping = pd.DataFrame(cik_ticker_data).T
cik_mapping['cik_str'] = cik_mapping['cik_str'].astype(int)
cik_to_ticker = dict(zip(cik_mapping['cik_str'], cik_mapping['ticker']))
print(f"Loaded {len(cik_to_ticker)} CIK-to-ticker mappings")


# ---------------------------------------------------------------------------
# 2. Load and combine all quarterly TSV files
# ---------------------------------------------------------------------------

_DEDUP_KEYS = {
    'NONDERIV_TRANS.tsv': ['NONDERIV_TRANS_SK'],
    'SUBMISSION.tsv': ['ACCESSION_NUMBER'],
    'REPORTINGOWNER.tsv': ['ACCESSION_NUMBER', 'RPTOWNERCIK'],
}


def _load_and_concat(tsv_name):
    """Load *tsv_name* from every subdirectory in _quarters/, concat, deduplicate."""
    pattern = os.path.join('_quarters', '*', tsv_name)
    paths = sorted(glob.glob(pattern))
    if not paths:
        raise FileNotFoundError(f"No {tsv_name} found in _quarters/*/")
    frames = []
    for p in paths:
        quarter = os.path.basename(os.path.dirname(p))
        df = pd.read_csv(p, sep='\t', low_memory=False)
        frames.append(df)
        print(f"  {quarter}: {len(df):,} rows")
    combined = pd.concat(frames, ignore_index=True)
    before = len(combined)
    dedup_cols = _DEDUP_KEYS.get(tsv_name, ['ACCESSION_NUMBER'])
    combined = combined.drop_duplicates(subset=dedup_cols, keep='first')
    print(f"  Combined: {before:,} -> {len(combined):,} (dedup by {dedup_cols})\n")
    return combined


print("Loading NONDERIV_TRANS from all quarters...")
nonderiv_trans = _load_and_concat('NONDERIV_TRANS.tsv')

print("Loading SUBMISSION from all quarters...")
submission = _load_and_concat('SUBMISSION.tsv')

print("Loading REPORTINGOWNER from all quarters...")
reportingowner = _load_and_concat('REPORTINGOWNER.tsv')

print(f"nonderiv_trans: {nonderiv_trans.shape}")
print(f"submission: {submission.shape}")
print(f"reportingowner: {reportingowner.shape}")


# ---------------------------------------------------------------------------
# 3. Select and transform columns
# ---------------------------------------------------------------------------

nonderiv_trans = nonderiv_trans[['ACCESSION_NUMBER', 'TRANS_DATE', 'TRANS_CODE',
                                 'TRANS_SHARES', 'TRANS_PRICEPERSHARE',
                                 'DIRECT_INDIRECT_OWNERSHIP']].copy()

submission = submission[['ACCESSION_NUMBER', 'ISSUERTRADINGSYMBOL', 'ISSUERCIK']].copy()

reportingowner['IS_DIRECTOR'] = (
    reportingowner['RPTOWNER_RELATIONSHIP']
    .str.contains('Director', case=False, na=False).astype(int)
)
reportingowner['IS_OFFICER'] = (
    reportingowner['RPTOWNER_RELATIONSHIP']
    .str.contains('Officer', case=False, na=False).astype(int)
)
reportingowner['IS_TEN_PERCENT_OWNER'] = (
    reportingowner['RPTOWNER_RELATIONSHIP']
    .str.contains('10%', case=False, na=False).astype(int)
)
reportingowner = reportingowner[['ACCESSION_NUMBER', 'IS_DIRECTOR', 'IS_OFFICER',
                                  'IS_TEN_PERCENT_OWNER', 'RPTOWNER_TITLE']].copy()


# ---------------------------------------------------------------------------
# 4. Merge and clean transactions
# ---------------------------------------------------------------------------

df = nonderiv_trans.merge(submission, on='ACCESSION_NUMBER', how='inner')
df = df.merge(reportingowner, on='ACCESSION_NUMBER', how='inner')
print(f"\nMerged: {df.shape}")

df['ISSUERCIK'] = df['ISSUERCIK'].astype(int)
df['ticker_from_cik'] = df['ISSUERCIK'].map(cik_to_ticker)
df['ticker_standardized'] = df['ticker_from_cik'].fillna(df['ISSUERTRADINGSYMBOL'])

print(f"CIK-mapped tickers: {df['ticker_from_cik'].notna().sum()}")
print(f"Unmapped (using original): {df['ticker_from_cik'].isna().sum()}")

df = df[df['TRANS_CODE'].isin(['P', 'S'])].copy()
df = df[df['TRANS_SHARES'].notna() & (df['TRANS_SHARES'] > 0)]
df = df[df['TRANS_PRICEPERSHARE'].notna() & (df['TRANS_PRICEPERSHARE'] > 0)]
df = df[df['TRANS_DATE'].notna() & df['ticker_standardized'].notna()]
print(f"After filtering: {df.shape}")

df['TRANS_DATE'] = pd.to_datetime(df['TRANS_DATE'])
df['IS_BUY'] = (df['TRANS_CODE'] == 'P').astype(int)
df['TRANSACTION_VALUE'] = df['TRANS_SHARES'] * df['TRANS_PRICEPERSHARE']

transactions = df[[
    'ticker_standardized', 'ISSUERCIK', 'TRANS_DATE', 'IS_BUY',
    'TRANS_SHARES', 'TRANS_PRICEPERSHARE', 'TRANSACTION_VALUE',
    'IS_OFFICER', 'IS_DIRECTOR', 'IS_TEN_PERCENT_OWNER', 'RPTOWNER_TITLE',
    'DIRECT_INDIRECT_OWNERSHIP', 'ACCESSION_NUMBER',
]].copy()

transactions.columns = [
    'ticker', 'cik', 'transaction_date', 'is_buy', 'shares',
    'price_per_share', 'transaction_value', 'is_officer', 'is_director',
    'is_ten_pct_owner', 'officer_title', 'ownership_type', 'accession_number',
]

transactions = (transactions
                .drop_duplicates()
                .sort_values(['ticker', 'transaction_date'])
                .reset_index(drop=True))

print(f"\nClean transactions: {transactions.shape}")
print(f"Date range: {transactions['transaction_date'].min().date()} "
      f"to {transactions['transaction_date'].max().date()}")
print(transactions.head())


# ---------------------------------------------------------------------------
# 5. Build 30-day window features from buy transactions
# ---------------------------------------------------------------------------

buys = transactions[transactions['is_buy'] == 1].copy()
buys['window_id'] = (
    (buys['transaction_date'] - buys['transaction_date'].min()).dt.days // 30
)

features = buys.groupby(['ticker', 'window_id']).agg({
    'transaction_date': 'max',
    'accession_number': 'nunique',
    'shares': 'sum',
    'transaction_value': 'sum',
    'is_officer': 'max',
    'is_director': 'max',
    'is_ten_pct_owner': 'max',
}).reset_index()

features.columns = [
    'ticker', 'window_id', 'transaction_date',
    'num_buy_transactions', 'total_shares_bought', 'total_value_bought',
    'has_officer_buy', 'has_director_buy', 'has_ten_pct_owner_buy',
]

features['avg_transaction_value'] = (
    features['total_value_bought'] / features['num_buy_transactions']
)
features = features.drop('window_id', axis=1)

print(f"\nML-ready features: {features.shape}")
print(features.head())


# ---------------------------------------------------------------------------
# 6. Fetch market data via yfinance (full date range + buffer)
# ---------------------------------------------------------------------------

try:
    import yfinance as yf
except ImportError:
    print("Installing yfinance...")
    import subprocess
    subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'yfinance'])
    import yfinance as yf

unique_tickers = transactions['ticker'].unique()
print(f"\nNumber of unique tickers to fetch: {len(unique_tickers)}")

min_date = transactions['transaction_date'].min()
max_date = transactions['transaction_date'].max()

start_date = min_date - pd.Timedelta(days=90)
end_date = max_date + pd.Timedelta(days=60)

print(f"Transaction date range: {min_date.date()} to {max_date.date()}")
print(f"Market data fetch range: {start_date.date()} to {end_date.date()}")
print(f"  (90 days before earliest + 60 days after latest for forward returns)")
print(f"\nFetching market data... This will take several minutes for {len(unique_tickers)} tickers.\n")

prices_list = []
shares_outstanding_dict = {}
failed_tickers = []
success_count = 0
batch_size = 100

for i in range(0, len(unique_tickers), batch_size):
    batch = unique_tickers[i:i + batch_size]
    batch_num = i // batch_size + 1
    total_batches = (len(unique_tickers) - 1) // batch_size + 1

    print(f"Processing batch {batch_num}/{total_batches} "
          f"({len(batch)} tickers)...", end='', flush=True)

    batch_success = 0
    for ticker in batch:
        try:
            from io import StringIO
            old_stdout = sys.stdout
            sys.stdout = StringIO()

            ticker_data = yf.download(
                ticker, start=start_date, end=end_date,
                progress=False, auto_adjust=True,
            )

            ticker_obj = yf.Ticker(ticker)
            try:
                so = ticker_obj.info.get('sharesOutstanding', None)
                if so:
                    shares_outstanding_dict[ticker] = so
            except Exception:
                pass

            sys.stdout = old_stdout

            if not ticker_data.empty:
                ticker_data = ticker_data.reset_index()
                if isinstance(ticker_data.columns, pd.MultiIndex):
                    ticker_data.columns = ticker_data.columns.get_level_values(0)
                rename_map = {
                    'Date': 'date', 'Datetime': 'date',
                    'Open': 'open', 'High': 'high', 'Low': 'low',
                    'Close': 'close', 'Adj Close': 'close_adj', 'Volume': 'volume',
                }
                ticker_data = ticker_data.rename(
                    columns={k: v for k, v in rename_map.items()
                             if k in ticker_data.columns})
                if 'close_adj' in ticker_data.columns:
                    ticker_data['close'] = ticker_data['close_adj']
                    ticker_data = ticker_data.drop(columns=['close_adj'], errors='ignore')
                ticker_data['ticker'] = ticker
                prices_list.append(
                    ticker_data[['ticker', 'date', 'open', 'high', 'low',
                                 'close', 'volume']])
                batch_success += 1
                success_count += 1
            else:
                failed_tickers.append(ticker)
        except Exception:
            sys.stdout = old_stdout
            failed_tickers.append(ticker)
            continue

    print(f" {batch_success} successful")

# Combine price data
if not prices_list:
    print("ERROR: Failed to fetch data for any ticker")
    sys.exit(1)

prices = pd.concat(prices_list, ignore_index=True)
prices['date'] = pd.to_datetime(prices['date'])
prices = prices[prices['close'].notna() & (prices['close'] > 0)]
prices = prices.sort_values(['ticker', 'date']).reset_index(drop=True)

print(f"\n{'=' * 60}")
print("MARKET DATA FETCH COMPLETE")
print(f"{'=' * 60}")
print(f"Successfully fetched: {prices['ticker'].nunique()} tickers "
      f"({success_count / len(unique_tickers) * 100:.1f}%)")
print(f"Failed/Delisted: {len(failed_tickers)} tickers")
print(f"Total price records: {prices.shape[0]:,}")
print(f"Date range: {prices['date'].min().date()} to {prices['date'].max().date()}")
print(f"Shares outstanding fetched for: {len(shares_outstanding_dict)} tickers")

if failed_tickers:
    print(f"\nSample failed tickers: {failed_tickers[:20]}")


# ---------------------------------------------------------------------------
# 7. Fetch S&P 500 benchmark
# ---------------------------------------------------------------------------

print(f"\nFetching S&P 500 (^GSPC) for market benchmark...")
sp500_data = yf.download('^GSPC', start=start_date, end=end_date,
                         progress=False, auto_adjust=True)
if not sp500_data.empty:
    sp500_data = sp500_data.reset_index()
    if isinstance(sp500_data.columns, pd.MultiIndex):
        sp500_data.columns = sp500_data.columns.get_level_values(0)
    rename_map = {'Date': 'date', 'Datetime': 'date',
                  'Close': 'close', 'Adj Close': 'close_adj'}
    sp500_data = sp500_data.rename(
        columns={k: v for k, v in rename_map.items() if k in sp500_data.columns})
    if 'close_adj' in sp500_data.columns:
        sp500_data['close'] = sp500_data['close_adj']
    sp500_data = sp500_data[['date', 'close']].copy()
    sp500_data['date'] = pd.to_datetime(sp500_data['date'])
    sp500_data = sp500_data.sort_values('date').reset_index(drop=True)
    sp500_data['close_7td'] = sp500_data['close'].shift(-7)
    sp500_data['sp500_return_7td'] = (
        (sp500_data['close_7td'] - sp500_data['close']) / sp500_data['close']
    )
    sp500_returns = sp500_data[['date', 'sp500_return_7td']].dropna()
    print(f"  S&P 500 data: {len(sp500_returns)} rows with valid 7-day returns")
else:
    sp500_returns = pd.DataFrame(columns=['date', 'sp500_return_7td'])
    print("  WARNING: Could not fetch S&P 500 - excess returns will be NaN")


# ---------------------------------------------------------------------------
# 8. Filter to tickers with market data
# ---------------------------------------------------------------------------

insider_tickers = set(transactions['ticker'].unique())
market_tickers = set(prices['ticker'].unique())

print(f"\nInsider tickers: {len(insider_tickers)}")
print(f"Market tickers: {len(market_tickers)}")
print(f"Overlapping: {len(insider_tickers & market_tickers)}")
print(f"Match rate: {len(insider_tickers & market_tickers) / len(insider_tickers):.2%}")

unmatched = insider_tickers - market_tickers
if unmatched:
    print(f"Sample unmatched tickers: {list(unmatched)[:20]}")

transactions = transactions[transactions['ticker'].isin(market_tickers)].reset_index(drop=True)
features = features[features['ticker'].isin(market_tickers)].reset_index(drop=True)

print(f"\nFiltered transactions: {transactions.shape}")
print(f"Filtered features: {features.shape}")


# ---------------------------------------------------------------------------
# 9. Calculate forward returns and excess returns
# ---------------------------------------------------------------------------

prices['close_7td'] = prices.groupby('ticker')['close'].shift(-7)
prices['return_7td'] = (prices['close_7td'] - prices['close']) / prices['close']

prices = prices.merge(sp500_returns, on='date', how='left')
prices['excess_return_7td'] = prices['return_7td'] - prices['sp500_return_7td']

prices_complete = prices[prices['return_7td'].notna()].copy()

print(f"\nAdded 7-trading-day forward returns and excess returns")
print(f"Rows with complete forward returns: {prices_complete.shape[0]:,}")
print(f"  return_7td: mean={prices_complete['return_7td'].mean():.2%}, "
      f"std={prices_complete['return_7td'].std():.2%}")
if prices_complete['excess_return_7td'].notna().any():
    print(f"  excess_return_7td: mean={prices_complete['excess_return_7td'].mean():.2%}, "
          f"std={prices_complete['excess_return_7td'].std():.2%}")


# ---------------------------------------------------------------------------
# 10. Merge features with price data and create labels
# ---------------------------------------------------------------------------

price_cols = ['ticker', 'date', 'close', 'return_7td', 'excess_return_7td']
transactions_with_prices = transactions.merge(
    prices[price_cols],
    left_on=['ticker', 'transaction_date'],
    right_on=['ticker', 'date'],
    how='inner',
).drop('date', axis=1)

print(f"\nTransactions with prices: {transactions_with_prices.shape}")
print(f"Match rate: {transactions_with_prices.shape[0] / max(1, transactions.shape[0]):.2%}")

features_with_prices = features.merge(
    prices[price_cols],
    left_on=['ticker', 'transaction_date'],
    right_on=['ticker', 'date'],
    how='inner',
).drop('date', axis=1)

features_with_prices = features_with_prices[
    features_with_prices['return_7td'].notna()
]

features_with_prices['total_shares_count'] = (
    features_with_prices['ticker'].map(shares_outstanding_dict)
)
features_with_prices['shares_traded_pct'] = (
    features_with_prices['total_shares_bought']
    / features_with_prices['total_shares_count']
) * 100

print(f"Features with prices: {features_with_prices.shape}")
print(f"Match rate: {features_with_prices.shape[0] / max(1, features.shape[0]):.2%}")
print(f"Tickers with shares outstanding: "
      f"{features_with_prices['total_shares_count'].notna().sum()}")

# Create binary label
threshold = 0.05

if len(features_with_prices) > 0:
    features_with_prices = features_with_prices[
        features_with_prices['excess_return_7td'].notna()
    ].copy()
    features_with_prices['label_7td'] = (
        (features_with_prices['excess_return_7td'] > threshold).astype(int)
    )

    print(f"\nLabels created ({threshold:.0%} excess-return threshold)")
    print(f"Label distribution:\n{features_with_prices['label_7td'].value_counts()}")
    print(f"Positive rate: {features_with_prices['label_7td'].mean():.2%}")
else:
    print("No features with prices -- cannot create labels.")
    sys.exit(1)


# ---------------------------------------------------------------------------
# 11. Save outputs
# ---------------------------------------------------------------------------

transactions.to_csv('transactions_clean.csv', index=False)
transactions_with_prices.to_csv('transactions_with_prices.csv', index=False)
features.to_csv('features_ml_ready.csv', index=False)
features_with_prices.to_csv('features_with_labels.csv', index=False)
prices.to_csv('market_prices_clean.csv', index=False)

print("\n" + "=" * 80)
print("DATA PREPARATION COMPLETE")
print("=" * 80)
print(f"\nSaved files:")
print(f"  transactions_clean.csv:        {transactions.shape[0]:,} rows")
print(f"  transactions_with_prices.csv:  {transactions_with_prices.shape[0]:,} rows")
print(f"  features_ml_ready.csv:         {features.shape[0]:,} rows")
print(f"  features_with_labels.csv:      {features_with_prices.shape[0]:,} rows (READY FOR ML)")
print(f"  market_prices_clean.csv:       {prices.shape[0]:,} rows")
print(f"\nDate range in final dataset: "
      f"{features_with_prices['transaction_date'].min().date()} to "
      f"{features_with_prices['transaction_date'].max().date()}")

import pandas as pd
import numpy as np
import json

pd.set_option('display.max_columns', None)

with open('data/Company Tickers.json', 'r') as f:
    cik_ticker_data = json.load(f)

cik_mapping = pd.DataFrame(cik_ticker_data).T
cik_mapping['cik_str'] = cik_mapping['cik_str'].astype(int)
cik_to_ticker = dict(zip(cik_mapping['cik_str'], cik_mapping['ticker']))

print(f"Loaded {len(cik_to_ticker)} CIK-to-ticker mappings")

nonderiv_trans = pd.read_csv('data/NONDERIV_TRANS.tsv', sep='\t', low_memory=False)
submission = pd.read_csv('data/SUBMISSION.tsv', sep='\t', low_memory=False)
reportingowner = pd.read_csv('data/REPORTINGOWNER.tsv', sep='\t', low_memory=False)

print(f"nonderiv_trans: {nonderiv_trans.shape}")
print(f"submission: {submission.shape}")
print(f"reportingowner: {reportingowner.shape}")

nonderiv_trans = nonderiv_trans[['ACCESSION_NUMBER', 'TRANS_DATE', 'TRANS_CODE',
                                  'TRANS_SHARES', 'TRANS_PRICEPERSHARE',
                                  'DIRECT_INDIRECT_OWNERSHIP']].copy()

submission = submission[['ACCESSION_NUMBER', 'ISSUERTRADINGSYMBOL', 'ISSUERCIK']].copy()

reportingowner['IS_DIRECTOR'] = reportingowner['RPTOWNER_RELATIONSHIP'].str.contains('Director', case=False, na=False).astype(int)
reportingowner['IS_OFFICER'] = reportingowner['RPTOWNER_RELATIONSHIP'].str.contains('Officer', case=False, na=False).astype(int)
reportingowner['IS_TEN_PERCENT_OWNER'] = reportingowner['RPTOWNER_RELATIONSHIP'].str.contains('10%', case=False, na=False).astype(int)
reportingowner = reportingowner[['ACCESSION_NUMBER', 'IS_DIRECTOR', 'IS_OFFICER',
                                  'IS_TEN_PERCENT_OWNER', 'RPTOWNER_TITLE']].copy()

df = nonderiv_trans.merge(submission, on='ACCESSION_NUMBER', how='inner')
df = df.merge(reportingowner, on='ACCESSION_NUMBER', how='inner')

print(f"Merged: {df.shape}")

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

transactions = df[['ticker_standardized', 'ISSUERCIK', 'TRANS_DATE', 'IS_BUY',
                   'TRANS_SHARES', 'TRANS_PRICEPERSHARE', 'TRANSACTION_VALUE',
                   'IS_OFFICER', 'IS_DIRECTOR', 'IS_TEN_PERCENT_OWNER', 'RPTOWNER_TITLE',
                   'DIRECT_INDIRECT_OWNERSHIP', 'ACCESSION_NUMBER']].copy()

transactions.columns = ['ticker', 'cik', 'transaction_date', 'is_buy', 'shares',
                       'price_per_share', 'transaction_value', 'is_officer', 'is_director',
                       'is_ten_pct_owner', 'officer_title', 'ownership_type', 'accession_number']

transactions = transactions.drop_duplicates().sort_values(['ticker', 'transaction_date']).reset_index(drop=True)

print(f"Clean transactions: {transactions.shape}")
print(transactions.head())

buys = transactions[transactions['is_buy'] == 1].copy()
buys['window_id'] = (buys['transaction_date'] - buys['transaction_date'].min()).dt.days // 30

features = buys.groupby(['ticker', 'window_id']).agg({
    'transaction_date': 'max',  # Use most recent transaction date in window
    'accession_number': 'nunique',
    'shares': 'sum',
    'transaction_value': 'sum',
    'is_officer': 'max',
    'is_director': 'max',
    'is_ten_pct_owner': 'max'
}).reset_index()

features.columns = ['ticker', 'window_id', 'transaction_date',
                   'num_buy_transactions', 'total_shares_bought', 'total_value_bought',
                   'has_officer_buy', 'has_director_buy', 'has_ten_pct_owner_buy']

features['avg_transaction_value'] = features['total_value_bought'] / features['num_buy_transactions']

# Drop window_id as we'll use transaction_date for merging
features = features.drop('window_id', axis=1)

print(f"ML-ready features: {features.shape}")
print(features.head())

# Install yfinance if not already installed
try:
    import yfinance as yf
except ImportError:
    print("Installing yfinance...")
    import subprocess
    subprocess.check_call(['pip', 'install', 'yfinance'])
    import yfinance as yf

import warnings
warnings.filterwarnings('ignore')  # Suppress yfinance warnings

# Get unique tickers from insider trading data
unique_tickers = transactions['ticker'].unique()
print(f"Number of unique tickers to fetch: {len(unique_tickers)}")

# Determine date range based on transaction dates - ALIGNED TO Q4 2025
min_transaction_date = transactions['transaction_date'].min()
max_transaction_date = transactions['transaction_date'].max()

# Get 90 days before earliest transaction (for context) and 60 days after latest (for forward returns)
start_date = min_transaction_date - pd.Timedelta(days=90)
end_date = max_transaction_date + pd.Timedelta(days=60)

print(f"Transaction date range: {min_transaction_date.date()} to {max_transaction_date.date()}")
print(f"Market data fetch range: {start_date.date()} to {end_date.date()}")
print(f"  (90 days before + 60 days after transactions for forward returns)")
print(f"\nFetching market data... This may take a few minutes.\n")

# Fetch data for all tickers
prices_list = []
shares_outstanding_dict = {}  # Store shares outstanding data
failed_tickers = []
success_count = 0
batch_size = 100

for i in range(0, len(unique_tickers), batch_size):
    batch = unique_tickers[i:i+batch_size]
    batch_num = i//batch_size + 1
    total_batches = (len(unique_tickers)-1)//batch_size + 1
    
    print(f"Processing batch {batch_num}/{total_batches} ({len(batch)} tickers)...", end='')
    
    batch_success = 0
    for ticker in batch:
        try:
            # Download data for this ticker (suppress output)
            import sys
            from io import StringIO
            old_stdout = sys.stdout
            sys.stdout = StringIO()
            
            ticker_data = yf.download(ticker, start=start_date, end=end_date, progress=False, auto_adjust=True)
            
            # Get shares outstanding from ticker info
            ticker_obj = yf.Ticker(ticker)
            try:
                shares_outstanding = ticker_obj.info.get('sharesOutstanding', None)
                if shares_outstanding:
                    shares_outstanding_dict[ticker] = shares_outstanding
            except:
                pass
            
            sys.stdout = old_stdout
            
            if not ticker_data.empty:
                # Reset index to get date as a column
                ticker_data = ticker_data.reset_index()
                # Flatten multi-level columns if present (yfinance varies by version)
                if isinstance(ticker_data.columns, pd.MultiIndex):
                    ticker_data.columns = ticker_data.columns.get_level_values(0)
                # Explicit rename by column name - robust to yfinance column order changes
                rename_map = {'Date': 'date', 'Datetime': 'date', 'Open': 'open', 'High': 'high',
                             'Low': 'low', 'Close': 'close', 'Adj Close': 'close_adj', 'Volume': 'volume'}
                ticker_data = ticker_data.rename(columns={k: v for k, v in rename_map.items() if k in ticker_data.columns})
                if 'close_adj' in ticker_data.columns:
                    ticker_data['close'] = ticker_data['close_adj']
                    ticker_data = ticker_data.drop(columns=['close_adj'], errors='ignore')
                ticker_data['ticker'] = ticker
                prices_list.append(ticker_data[['ticker', 'date', 'open', 'high', 'low', 'close', 'volume']])
                batch_success += 1
                success_count += 1
            else:
                failed_tickers.append(ticker)
        except Exception as e:
            sys.stdout = old_stdout
            failed_tickers.append(ticker)
            continue
    
    print(f" ✓ {batch_success} successful")

# Combine all data
if prices_list:
    prices = pd.concat(prices_list, ignore_index=True)
    
    # Clean and preprocess
    prices['date'] = pd.to_datetime(prices['date'])
    prices = prices[prices['close'].notna() & (prices['close'] > 0)]
    prices = prices.sort_values(['ticker', 'date']).reset_index(drop=True)
    
    print(f"\n{'='*60}")
    print(f"MARKET DATA FETCH COMPLETE")
    print(f"{'='*60}")
    print(f"Successfully fetched: {len(prices['ticker'].unique())} tickers ({success_count/len(unique_tickers)*100:.1f}%)")
    print(f"Failed/Delisted: {len(failed_tickers)} tickers ({len(failed_tickers)/len(unique_tickers)*100:.1f}%)")
    print(f"Total price records: {prices.shape[0]:,}")
    print(f"Date range: {prices['date'].min().date()} to {prices['date'].max().date()}")
    print(f"Shares outstanding fetched for: {len(shares_outstanding_dict)} tickers")
    
    # Show sample of failed tickers
    if len(failed_tickers) > 0:
        print(f"\nSample failed tickers (likely delisted/warrants): {failed_tickers[:10]}")
else:
    print("ERROR: Failed to fetch data for any ticker")
    prices = pd.DataFrame(columns=['ticker', 'date', 'open', 'high', 'low', 'close', 'volume'])

# Fetch S&P 500 for market-adjusted (excess) returns
print(f"\nFetching S&P 500 (^GSPC) for market benchmark...")
sp500_data = yf.download('^GSPC', start=start_date, end=end_date, progress=False, auto_adjust=True)
if not sp500_data.empty:
    sp500_data = sp500_data.reset_index()
    if isinstance(sp500_data.columns, pd.MultiIndex):
        sp500_data.columns = sp500_data.columns.get_level_values(0)
    rename_map = {'Date': 'date', 'Datetime': 'date', 'Open': 'open', 'High': 'high',
                 'Low': 'low', 'Close': 'close', 'Adj Close': 'close_adj', 'Volume': 'volume'}
    sp500_data = sp500_data.rename(columns={k: v for k, v in rename_map.items() if k in sp500_data.columns})
    if 'close_adj' in sp500_data.columns:
        sp500_data['close'] = sp500_data['close_adj']
    sp500_data = sp500_data[['date', 'close']].copy()
    sp500_data['date'] = pd.to_datetime(sp500_data['date'])
    sp500_data = sp500_data.sort_values('date').reset_index(drop=True)
    sp500_data['close_7td'] = sp500_data['close'].shift(-7)
    sp500_data['sp500_return_7td'] = (sp500_data['close_7td'] - sp500_data['close']) / sp500_data['close']
    sp500_returns = sp500_data[['date', 'sp500_return_7td']].dropna()
    print(f"  S&P 500 data: {len(sp500_returns)} rows with valid 7-trading-day returns")
else:
    sp500_returns = pd.DataFrame(columns=['date', 'sp500_return_7td'])
    print("  WARNING: Could not fetch S&P 500 - excess returns will be NaN")

# Check ticker overlap
insider_tickers = set(transactions['ticker'].unique())
market_tickers = set(prices['ticker'].unique())

print(f"Insider tickers: {len(insider_tickers)}")
print(f"Market tickers: {len(market_tickers)}")
print(f"Overlapping: {len(insider_tickers & market_tickers)}")
print(f"Match rate: {len(insider_tickers & market_tickers) / len(insider_tickers):.2%}")

# Show some examples of unmatched tickers
unmatched = insider_tickers - market_tickers
if unmatched:
    print(f"\nSample of unmatched tickers: {list(unmatched)[:20]}")
    
# Filter transactions to only include tickers with market data
transactions = transactions[transactions['ticker'].isin(market_tickers)].reset_index(drop=True)
features = features[features['ticker'].isin(market_tickers)].reset_index(drop=True)

print(f"\nFiltered transactions: {transactions.shape}")
print(f"Filtered features: {features.shape}")

# Calculate 7-trading-day forward returns (shift(-7) = 7 trading days, not calendar)
prices['close_7td'] = prices.groupby('ticker')['close'].shift(-7)
prices['return_7td'] = (prices['close_7td'] - prices['close']) / prices['close']

# Market-adjusted (excess) return: Stock return - S&P 500 return (same period)
prices = prices.merge(sp500_returns, on='date', how='left')
prices['excess_return_7td'] = prices['return_7td'] - prices['sp500_return_7td']

# Remove rows where we can't calculate 7-trading-day forward returns (near the end of data)
prices_complete = prices[prices['return_7td'].notna()].copy()

print(f"Added 7-trading-day forward returns and market-adjusted (excess) returns")
print(f"Rows with complete forward returns: {prices_complete.shape[0]}")
print(f"\nReturn statistics (raw 7-trading-day):")
mean_return = prices_complete['return_7td'].mean()
std_return = prices_complete['return_7td'].std()
print(f"  return_7td: mean={mean_return:.2%}, std={std_return:.2%}")
if prices_complete['excess_return_7td'].notna().any():
    mean_excess = prices_complete['excess_return_7td'].mean()
    std_excess = prices_complete['excess_return_7td'].std()
    print(f"\nExcess return (vs S&P 500):")
    print(f"  excess_return_7td: mean={mean_excess:.2%}, std={std_excess:.2%}")

# Merge transactions with price data
price_cols = ['ticker', 'date', 'close', 'return_7td', 'excess_return_7td']
transactions_with_prices = transactions.merge(
    prices[price_cols],
    left_on=['ticker', 'transaction_date'],
    right_on=['ticker', 'date'],
    how='inner'
).drop('date', axis=1)

print(f"Transactions with prices: {transactions_with_prices.shape}")
print(f"Match rate: {transactions_with_prices.shape[0] / transactions.shape[0]:.2%}")

# Show sample
print(f"\nSample of transactions with prices:")
print(transactions_with_prices[['ticker', 'transaction_date', 'is_buy', 'shares', 
                                 'transaction_value', 'close', 'return_7td', 'excess_return_7td']].head(10))

# Merge features with price data
features_with_prices = features.merge(
    prices[price_cols],
    left_on=['ticker', 'transaction_date'],
    right_on=['ticker', 'date'],
    how='inner'
).drop('date', axis=1)

# Keep only rows with complete forward returns
features_with_prices = features_with_prices[features_with_prices['return_7td'].notna()]

# Add total shares outstanding
features_with_prices['total_shares_count'] = features_with_prices['ticker'].map(shares_outstanding_dict)

# Calculate proportion of shares traded vs total shares
features_with_prices['shares_traded_pct'] = (
    features_with_prices['total_shares_bought'] / features_with_prices['total_shares_count']
) * 100

print(f"Features with prices: {features_with_prices.shape}")
print(f"Match rate: {features_with_prices.shape[0] / features.shape[0]:.2%}")
print(f"Tickers with shares outstanding data: {features_with_prices['total_shares_count'].notna().sum()}")

# Show sample
print(f"\nSample of features with prices:")
if len(features_with_prices) > 0:
    print(features_with_prices[['ticker', 'transaction_date', 'num_buy_transactions', 
                                 'total_value_bought', 'total_shares_bought', 'total_shares_count',
                                 'shares_traded_pct', 'close', 'return_7td', 'excess_return_7td']].head(10))
else:
    print("No matches found - please run the yfinance data fetching cell above")

# Labels use excess return (market-adjusted): did the stock beat the market by 5%?
threshold = 0.05

if len(features_with_prices) > 0:
    # Drop rows where excess return is NaN (e.g. missing S&P 500 data for that date)
    features_with_prices = features_with_prices[features_with_prices['excess_return_7td'].notna()].copy()
    features_with_prices['label_7td'] = (features_with_prices['excess_return_7td'] > threshold).astype(int)

    print(f"Labels created with {threshold:.1%} excess-return threshold (stock beats S&P 500 by 5%+)")
    print(f"\nLabel distribution (7-trading-day excess return):")
    print(features_with_prices['label_7td'].value_counts())
    print(f"Positive rate: {features_with_prices['label_7td'].mean():.2%}")
else:
    print("No features with prices - cannot create labels. Please run yfinance data fetching cell above.")

transactions.to_csv('data/transactions_clean.csv', index=False)
transactions_with_prices.to_csv('data/transactions_with_prices.csv', index=False)
features.to_csv('data/features_ml_ready.csv', index=False)
features_with_prices.to_csv('data/features_with_labels.csv', index=False)
prices.to_csv('data/market_prices_clean.csv', index=False)

print("="*80)
print("DATA PREPARATION COMPLETE")
print("="*80)
print(f"\nSaved files:")
print(f"  transactions_clean.csv: {transactions.shape[0]} rows")
print(f"  transactions_with_prices.csv: {transactions_with_prices.shape[0]} rows")
print(f"  features_ml_ready.csv: {features.shape[0]} rows")
print(f"  features_with_labels.csv: {features_with_prices.shape[0]} rows (READY FOR ML)")
print(f"  market_prices_clean.csv: {prices.shape[0]} rows")
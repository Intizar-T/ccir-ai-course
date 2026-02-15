"""
Create insider-trading ML dataset from SEC Form 4 data and market prices.
Call create_dataset() to build the dataset; returns the ML-ready DataFrame
and saves CSVs as a side effect.
"""

import json
import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

pd.set_option('display.max_columns', None)

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_DATA_DIR = SCRIPT_DIR / 'data'


@dataclass
class InsiderTradingDataset:
    """ML-ready dataset plus supporting DataFrames."""

    features_with_labels: pd.DataFrame  # Primary dataset for training (with labels)
    transactions: pd.DataFrame
    transactions_with_prices: pd.DataFrame
    features: pd.DataFrame
    prices: pd.DataFrame


def _load_cik_mapping(data_dir: Path) -> dict:
    """Load CIK-to-ticker mapping from Company Tickers JSON."""
    with open(data_dir / 'Company Tickers.json', 'r') as f:
        cik_ticker_data = json.load(f)
    cik_mapping = pd.DataFrame(cik_ticker_data).T
    cik_mapping['cik_str'] = cik_mapping['cik_str'].astype(int)
    return dict(zip(cik_mapping['cik_str'], cik_mapping['ticker']))


def _load_and_merge_transactions(data_dir: Path, cik_to_ticker: dict) -> pd.DataFrame:
    """Load SEC Form 4 TSVs, merge, filter, and produce clean transactions."""
    nonderiv_trans = pd.read_csv(data_dir / 'NONDERIV_TRANS.tsv', sep='\t', low_memory=False)
    submission = pd.read_csv(data_dir / 'SUBMISSION.tsv', sep='\t', low_memory=False)
    reportingowner = pd.read_csv(data_dir / 'REPORTINGOWNER.tsv', sep='\t', low_memory=False)

    nonderiv_trans = nonderiv_trans[
        ['ACCESSION_NUMBER', 'TRANS_DATE', 'TRANS_CODE', 'TRANS_SHARES', 'TRANS_PRICEPERSHARE', 'DIRECT_INDIRECT_OWNERSHIP']
    ].copy()
    submission = submission[['ACCESSION_NUMBER', 'ISSUERTRADINGSYMBOL', 'ISSUERCIK']].copy()
    reportingowner['IS_DIRECTOR'] = reportingowner['RPTOWNER_RELATIONSHIP'].str.contains('Director', case=False, na=False).astype(int)
    reportingowner['IS_OFFICER'] = reportingowner['RPTOWNER_RELATIONSHIP'].str.contains('Officer', case=False, na=False).astype(int)
    reportingowner['IS_TEN_PERCENT_OWNER'] = reportingowner['RPTOWNER_RELATIONSHIP'].str.contains('10%', case=False, na=False).astype(int)
    reportingowner = reportingowner[['ACCESSION_NUMBER', 'IS_DIRECTOR', 'IS_OFFICER', 'IS_TEN_PERCENT_OWNER', 'RPTOWNER_TITLE']].copy()

    df = nonderiv_trans.merge(submission, on='ACCESSION_NUMBER', how='inner')
    df = df.merge(reportingowner, on='ACCESSION_NUMBER', how='inner')

    df['ISSUERCIK'] = df['ISSUERCIK'].astype(int)
    df['ticker_from_cik'] = df['ISSUERCIK'].map(cik_to_ticker)
    df['ticker_standardized'] = df['ticker_from_cik'].fillna(df['ISSUERTRADINGSYMBOL'])

    df = df[df['TRANS_CODE'].isin(['P', 'S'])].copy()
    df = df[df['TRANS_SHARES'].notna() & (df['TRANS_SHARES'] > 0)]
    df = df[df['TRANS_PRICEPERSHARE'].notna() & (df['TRANS_PRICEPERSHARE'] > 0)]
    df = df[df['TRANS_DATE'].notna() & df['ticker_standardized'].notna()]

    df['TRANS_DATE'] = pd.to_datetime(df['TRANS_DATE'])
    df['IS_BUY'] = (df['TRANS_CODE'] == 'P').astype(int)
    df['TRANSACTION_VALUE'] = df['TRANS_SHARES'] * df['TRANS_PRICEPERSHARE']

    transactions = df[
        ['ticker_standardized', 'ISSUERCIK', 'TRANS_DATE', 'IS_BUY', 'TRANS_SHARES', 'TRANS_PRICEPERSHARE',
         'TRANSACTION_VALUE', 'IS_OFFICER', 'IS_DIRECTOR', 'IS_TEN_PERCENT_OWNER', 'RPTOWNER_TITLE',
         'DIRECT_INDIRECT_OWNERSHIP', 'ACCESSION_NUMBER']
    ].copy()
    transactions.columns = [
        'ticker', 'cik', 'transaction_date', 'is_buy', 'shares', 'price_per_share', 'transaction_value',
        'is_officer', 'is_director', 'is_ten_pct_owner', 'officer_title', 'ownership_type', 'accession_number'
    ]
    return transactions.drop_duplicates().sort_values(['ticker', 'transaction_date']).reset_index(drop=True)


def _build_features(transactions: pd.DataFrame) -> pd.DataFrame:
    """Aggregate buy transactions into per-ticker-window features."""
    buys = transactions[transactions['is_buy'] == 1].copy()
    buys['window_id'] = (buys['transaction_date'] - buys['transaction_date'].min()).dt.days // 30

    features = buys.groupby(['ticker', 'window_id']).agg({
        'transaction_date': 'max',
        'accession_number': 'nunique',
        'shares': 'sum',
        'transaction_value': 'sum',
        'is_officer': 'max',
        'is_director': 'max',
        'is_ten_pct_owner': 'max'
    }).reset_index()
    features.columns = [
        'ticker', 'window_id', 'transaction_date', 'num_buy_transactions', 'total_shares_bought', 'total_value_bought',
        'has_officer_buy', 'has_director_buy', 'has_ten_pct_owner_buy'
    ]
    features['avg_transaction_value'] = features['total_value_bought'] / features['num_buy_transactions']
    return features.drop('window_id', axis=1)


def _fetch_prices(
    unique_tickers: np.ndarray,
    start_date: pd.Timestamp,
    end_date: pd.Timestamp,
    verbose: bool = True,
) -> tuple[pd.DataFrame, dict]:
    """Fetch market prices via yfinance. Returns (prices_df, shares_outstanding_dict)."""
    try:
        import yfinance as yf
    except ImportError:
        import subprocess
        subprocess.check_call(['pip', 'install', 'yfinance'])
        import yfinance as yf

    import sys
    import time
    from io import StringIO

    import warnings
    warnings.filterwarnings('ignore')

    # Cache setup for SQLite/permission issues
    cache_dir = SCRIPT_DIR / 'data' / '.yfinance_cache'
    try:
        cache_dir.mkdir(parents=True, exist_ok=True)
        (cache_dir / '.write_test').touch()
        (cache_dir / '.write_test').unlink()
    except OSError:
        import tempfile
        cache_dir = Path(tempfile.gettempdir()) / 'py-yfinance'
        cache_dir.mkdir(parents=True, exist_ok=True)
    yf.set_tz_cache_location(str(cache_dir))

    prices_list = []
    shares_outstanding_dict = {}
    failed_tickers = []
    success_count = 0
    batch_size = 100

    for i in range(0, len(unique_tickers), batch_size):
        batch = unique_tickers[i:i + batch_size]
        batch_num = i // batch_size + 1
        total_batches = (len(unique_tickers) - 1) // batch_size + 1
        if verbose:
            print(f"Processing batch {batch_num}/{total_batches} ({len(batch)} tickers)...", end='', flush=True)

        batch_success = 0
        for ticker in batch:
            old_stdout = sys.stdout
            try:
                sys.stdout = StringIO()
                try:
                    ticker_data = yf.download(ticker, start=start_date, end=end_date, progress=False, auto_adjust=True)
                    time.sleep(0.5)
                    ticker_obj = yf.Ticker(ticker)
                    try:
                        so = ticker_obj.info.get('sharesOutstanding', None)
                        if so:
                            shares_outstanding_dict[ticker] = so
                    except Exception:
                        pass
                finally:
                    sys.stdout = old_stdout

                if not ticker_data.empty:
                    ticker_data = ticker_data.reset_index()
                    if isinstance(ticker_data.columns, pd.MultiIndex):
                        ticker_data.columns = ticker_data.columns.get_level_values(0)
                    rename_map = {
                        'Date': 'date', 'Datetime': 'date', 'Open': 'open', 'High': 'high',
                        'Low': 'low', 'Close': 'close', 'Adj Close': 'close_adj', 'Volume': 'volume'
                    }
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
            except Exception:
                sys.stdout = old_stdout
                failed_tickers.append(ticker)
            time.sleep(1.0)

        if verbose:
            print(f" ✓ {batch_success} successful")

    if prices_list:
        prices = pd.concat(prices_list, ignore_index=True)
        prices['date'] = pd.to_datetime(prices['date'])
        prices = prices[prices['close'].notna() & (prices['close'] > 0)]
        prices = prices.sort_values(['ticker', 'date']).reset_index(drop=True)
    else:
        prices = pd.DataFrame(columns=['ticker', 'date', 'open', 'high', 'low', 'close', 'volume'])

    if verbose:
        print(f"Fetched {success_count}/{len(unique_tickers)} tickers, {len(failed_tickers)} failed")
    return prices, shares_outstanding_dict


def _fetch_sp500_returns(start_date: pd.Timestamp, end_date: pd.Timestamp) -> pd.DataFrame:
    """Fetch S&P 500 and compute 7-trading-day returns."""
    import yfinance as yf
    sp500_data = yf.download('^GSPC', start=start_date, end=end_date, progress=False, auto_adjust=True)
    if sp500_data.empty:
        return pd.DataFrame(columns=['date', 'sp500_return_7td'])
    sp500_data = sp500_data.reset_index()
    if isinstance(sp500_data.columns, pd.MultiIndex):
        sp500_data.columns = sp500_data.columns.get_level_values(0)
    rename_map = {'Date': 'date', 'Datetime': 'date', 'Close': 'close', 'Adj Close': 'close_adj'}
    sp500_data = sp500_data.rename(columns={k: v for k, v in rename_map.items() if k in sp500_data.columns})
    if 'close_adj' in sp500_data.columns:
        sp500_data['close'] = sp500_data['close_adj']
    sp500_data = sp500_data[['date', 'close']].copy()
    sp500_data['date'] = pd.to_datetime(sp500_data['date'])
    sp500_data = sp500_data.sort_values('date').reset_index(drop=True)
    sp500_data['close_7td'] = sp500_data['close'].shift(-7)
    sp500_data['sp500_return_7td'] = (sp500_data['close_7td'] - sp500_data['close']) / sp500_data['close']
    return sp500_data[['date', 'sp500_return_7td']].dropna()


def _save_csvs(
    data_dir: Path,
    transactions: pd.DataFrame,
    transactions_with_prices: pd.DataFrame,
    features: pd.DataFrame,
    features_with_prices: pd.DataFrame,
    prices: pd.DataFrame,
) -> None:
    """Save all DataFrames to CSV files."""
    transactions.to_csv(data_dir / 'transactions_clean.csv', index=False)
    transactions_with_prices.to_csv(data_dir / 'transactions_with_prices.csv', index=False)
    features.to_csv(data_dir / 'features_ml_ready.csv', index=False)
    features_with_prices.to_csv(data_dir / 'features_with_labels.csv', index=False)
    prices.to_csv(data_dir / 'market_prices_clean.csv', index=False)


def _load_from_csv(data_dir: Path) -> InsiderTradingDataset:
    """Load dataset from existing CSV files."""
    return InsiderTradingDataset(
        features_with_labels=pd.read_csv(data_dir / 'features_with_labels.csv'),
        transactions=pd.read_csv(data_dir / 'transactions_clean.csv'),
        transactions_with_prices=pd.read_csv(data_dir / 'transactions_with_prices.csv'),
        features=pd.read_csv(data_dir / 'features_ml_ready.csv'),
        prices=pd.read_csv(data_dir / 'market_prices_clean.csv'),
    )


def create_dataset(
    data_dir: str | Path = None,
    refetch: bool = True,
    save_csv: bool = True,
    verbose: bool = True,
    label_threshold: float = 0.05,
) -> InsiderTradingDataset:
    """
    Build or load the insider-trading ML dataset from SEC Form 4 data and market prices.

    Args:
        data_dir: Directory for input TSVs and output CSVs. Default: lucas/data
        refetch: If True, always refetch market data and rebuild. If False and
                 features_with_labels.csv exists, load from CSVs and return.
        save_csv: If True, save transactions, features, prices to CSV files (when refetching)
        verbose: If True, print progress messages
        label_threshold: Excess return threshold for positive label (default 0.05 = 5%)

    Returns:
        InsiderTradingDataset with features_with_labels (primary), transactions,
        transactions_with_prices, features, prices.
    """
    data_dir = Path(data_dir) if data_dir else DEFAULT_DATA_DIR
    csv_path = data_dir / 'features_with_labels.csv'

    if not refetch and csv_path.exists():
        if verbose:
            print(f"Loading dataset from {data_dir} (refetch=False, CSV present)")
        return _load_from_csv(data_dir)

    # 1. Load and process transactions
    cik_to_ticker = _load_cik_mapping(data_dir)
    transactions = _load_and_merge_transactions(data_dir, cik_to_ticker)
    features = _build_features(transactions)

    if verbose:
        print(f"Transactions: {transactions.shape}, Features: {features.shape}")

    # 2. Fetch market data
    min_date = transactions['transaction_date'].min()
    max_date = transactions['transaction_date'].max()
    start_date = min_date - pd.Timedelta(days=90)
    end_date = max_date + pd.Timedelta(days=60)

    if verbose:
        print(f"Fetching market data {start_date.date()} to {end_date.date()}...")

    prices, shares_outstanding_dict = _fetch_prices(
        transactions['ticker'].unique(),
        start_date,
        end_date,
        verbose=verbose,
    )

    sp500_returns = _fetch_sp500_returns(start_date, end_date)
    if verbose and not sp500_returns.empty:
        print(f"S&P 500: {len(sp500_returns)} rows")

    # 3. Filter to tickers with market data
    market_tickers = set(prices['ticker'].unique())
    transactions = transactions[transactions['ticker'].isin(market_tickers)].reset_index(drop=True)
    features = features[features['ticker'].isin(market_tickers)].reset_index(drop=True)

    # 4. Compute returns
    prices['close_7td'] = prices.groupby('ticker')['close'].shift(-7)
    prices['return_7td'] = (prices['close_7td'] - prices['close']) / prices['close']
    prices = prices.merge(sp500_returns, on='date', how='left')
    prices['excess_return_7td'] = prices['return_7td'] - prices['sp500_return_7td']

    # 5. Merge features with prices
    price_cols = ['ticker', 'date', 'close', 'return_7td', 'excess_return_7td']
    transactions_with_prices = transactions.merge(
        prices[price_cols],
        left_on=['ticker', 'transaction_date'],
        right_on=['ticker', 'date'],
        how='inner',
    ).drop('date', axis=1)

    features_with_prices = features.merge(
        prices[price_cols],
        left_on=['ticker', 'transaction_date'],
        right_on=['ticker', 'date'],
        how='inner',
    ).drop('date', axis=1)

    features_with_prices = features_with_prices[features_with_prices['return_7td'].notna()].copy()
    features_with_prices['total_shares_count'] = features_with_prices['ticker'].map(shares_outstanding_dict)
    features_with_prices['shares_traded_pct'] = (
        features_with_prices['total_shares_bought'] / features_with_prices['total_shares_count']
    ) * 100

    # 6. Create labels (excess return > threshold)
    features_with_prices = features_with_prices[features_with_prices['excess_return_7td'].notna()].copy()
    features_with_prices['label_7td'] = (features_with_prices['excess_return_7td'] > label_threshold).astype(int)

    if verbose:
        print(f"ML dataset: {features_with_prices.shape[0]} rows, {features_with_prices['label_7td'].mean():.2%} positive")

    # 7. Save CSVs
    if save_csv:
        _save_csvs(
            data_dir,
            transactions,
            transactions_with_prices,
            features,
            features_with_prices,
            prices,
        )
        if verbose:
            print(f"Saved CSVs to {data_dir}")

    return InsiderTradingDataset(
        features_with_labels=features_with_prices,
        transactions=transactions,
        transactions_with_prices=transactions_with_prices,
        features=features,
        prices=prices,
    )


if __name__ == '__main__':
    ds = create_dataset(verbose=True, refetch=True)
    print("\nPrimary ML dataset (features_with_labels):")
    print(ds.features_with_labels.head())

import os
import pandas as pd
from sklearn.dummy import DummyRegressor

from create_dataset import create_dataset, InsiderTradingDataset


def process_data(refetch: bool = False) -> pd.DataFrame:
    """
    Return the insider-trading ML dataset (features_with_labels).

    Args:
        refetch: If True, always refetch market data and rebuild. If False and
                 features_with_labels.csv exists, load from CSV and return.

    Returns:
        DataFrame with features and label_7td (saves CSVs when refetching).
    """
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(script_dir, 'data')

    ds: InsiderTradingDataset = create_dataset(
        data_dir=data_dir,
        refetch=refetch,
        save_csv=True,
        verbose=True,
    )
    df = ds.features_with_labels

    if 'transaction_date' in df.columns:
        df['transaction_date'] = pd.to_datetime(df['transaction_date'])

    return df

def feature_engineering():
    pass

def build_lstm_model():
    pass

def training_data_split():
    pass

def train_model():
    pass

def dummy_regression_model():
    """
    Creates and returns a dummy regression model.
    """
    model = DummyRegressor(strategy='mean')
    return model

def random_forest_regression_model():
    pass

def xgboost_model():
    pass

def lstm_model():
    pass

def logistic_regression_model():
    pass

def svm_model():
    pass

def evaluate_model():
    pass

def save_model():
    pass

def main():
    df = process_data()
    print(df.head())

if __name__ == '__main__':
    main()
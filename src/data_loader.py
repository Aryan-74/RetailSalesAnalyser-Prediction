"""Data loader module for Retail Sales Analysis and Prediction."""

import os
from pathlib import Path
from typing import Dict, Optional, Tuple, List
import pandas as pd
import config

REQUIRED_FILES = [
    'train.csv',
    'test.csv',
    'stores.csv',
    'oil.csv',
    'holidays_events.csv',
    'transactions.csv'
]

def find_raw_data_dir() -> Path:
    """Locate the raw data directory from candidate paths."""
    candidates = [
        config.RAW_DATA_DIR,
        config.ALT_RAW_DATA_DIR,
        config.BASE_DIR,
        Path('data/raw'),
        Path('store-sales-time-series-forecasting')
    ]
    for c in candidates:
        if c.exists() and (c / 'train.csv').exists():
            return c.resolve()
    return config.RAW_DATA_DIR.resolve()

def check_missing_files(data_dir: Optional[Path] = None) -> List[str]:
    """Check for any missing required dataset files."""
    if data_dir is None:
        data_dir = find_raw_data_dir()
    missing = [f for f in REQUIRED_FILES if not (data_dir / f).exists()]
    return missing

def load_raw_data(
    data_dir: Optional[Path] = None,
    nrows: Optional[int] = None
) -> Dict[str, pd.DataFrame]:
    """Load all raw CSV files into a dictionary of DataFrames with optimized dtypes."""
    if data_dir is None:
        data_dir = find_raw_data_dir()

    missing = check_missing_files(data_dir)
    if missing:
        raise FileNotFoundError(
            f"Required dataset files are missing in {data_dir}: {missing}. "
            f"Please ensure Kaggle Store Sales dataset is downloaded."
        )

    print(f"[DataLoader] Loading raw dataset from: {data_dir}")

    # Train
    train = pd.read_csv(
        data_dir / 'train.csv',
        parse_dates=['date'],
        dtype={
            'store_nbr': 'int16',
            'family': 'category',
            'sales': 'float32',
            'onpromotion': 'int32'
        },
        nrows=nrows
    )

    # Test
    test = pd.read_csv(
        data_dir / 'test.csv',
        parse_dates=['date'],
        dtype={
            'store_nbr': 'int16',
            'family': 'category',
            'onpromotion': 'int32'
        },
        nrows=nrows
    )

    # Stores
    stores = pd.read_csv(
        data_dir / 'stores.csv',
        dtype={
            'store_nbr': 'int16',
            'city': 'category',
            'state': 'category',
            'type': 'category',
            'cluster': 'int8'
        }
    )

    # Oil
    oil = pd.read_csv(
        data_dir / 'oil.csv',
        parse_dates=['date'],
        dtype={'dcoilwtico': 'float32'}
    )

    # Holidays
    holidays = pd.read_csv(
        data_dir / 'holidays_events.csv',
        parse_dates=['date'],
        dtype={
            'type': 'category',
            'locale': 'category',
            'locale_name': 'category',
            'description': 'string',
            'transferred': 'bool'
        }
    )

    # Transactions
    transactions = pd.read_csv(
        data_dir / 'transactions.csv',
        parse_dates=['date'],
        dtype={
            'store_nbr': 'int16',
            'transactions': 'int32'
        }
    )

    data = {
        'train': train,
        'test': test,
        'stores': stores,
        'oil': oil,
        'holidays': holidays,
        'transactions': transactions
    }

    sample_sub_path = data_dir / 'sample_submission.csv'
    if sample_sub_path.exists():
        data['sample_submission'] = pd.read_csv(sample_sub_path)

    return data

def get_dataset_summary(data_dict: Dict[str, pd.DataFrame]) -> Dict[str, dict]:
    """Extract shapes, date ranges, and memory usage for each loaded dataset."""
    summary = {}
    for name, df in data_dict.items():
        info = {
            'rows': len(df),
            'columns': len(df.columns),
            'memory_mb': round(df.memory_usage(deep=True).sum() / (1024 * 1024), 2)
        }
        if 'date' in df.columns:
            info['date_min'] = str(df['date'].min().date())
            info['date_max'] = str(df['date'].max().date())
        summary[name] = info
    return summary

if __name__ == '__main__':
    data_dir = find_raw_data_dir()
    print('Raw Data Directory:', data_dir)
    missing = check_missing_files(data_dir)
    if missing:
        print('Missing files:', missing)
    else:
        print('All required files present.')
        dfs = load_raw_data(data_dir, nrows=1000)
        summary = get_dataset_summary(dfs)
        for k, v in summary.items():
            print(f'{k}: {v}')

"""Preprocessing module for Retail Sales Analysis and Prediction."""

import json
from pathlib import Path
from typing import Dict, Any, Tuple
import pandas as pd
import numpy as np

import config
from src.data_loader import load_raw_data

def preprocess_oil(oil_df: pd.DataFrame, start_date: str = '2013-01-01', end_date: str = '2017-08-31') -> pd.DataFrame:
    """
    Interpolate missing oil prices across full calendar date range.
    Weekends and holidays are missing in raw oil.csv.
    """
    oil_df = oil_df.copy()
    oil_df['date'] = pd.to_datetime(oil_df['date'])
    oil_df = oil_df.drop_duplicates(subset=['date']).sort_values('date')

    # Continuous date index
    full_dates = pd.date_range(start=start_date, end=end_date, freq='D', name='date')
    oil_full = pd.DataFrame(index=full_dates).reset_index()

    merged_oil = pd.merge(oil_full, oil_df, on='date', how='left')
    # Linear interpolation + bfill for initial day if NaN
    merged_oil['dcoilwtico'] = (
        merged_oil['dcoilwtico']
        .interpolate(method='linear')
        .bfill()
        .ffill()
        .astype('float32')
    )
    return merged_oil

def preprocess_holidays(holidays_df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean and deduplicate holiday events.
    - Filter transferred holidays (transferred=True means it was moved to another date).
    - Avoid duplicates when multiple events happen on the same day.
    """
    df = holidays_df.copy()
    df['date'] = pd.to_datetime(df['date'])

    # Exclude transferred holidays (they were celebrated elsewhere)
    effective_holidays = df[~df['transferred']].copy()
    # Exclude Work Day
    effective_holidays = effective_holidays[effective_holidays['type'].astype(str) != 'Work Day']

    # Priority mapping for multiple holidays on the same date
    locale_priority = {'National': 3, 'Regional': 2, 'Local': 1}
    type_priority = {'Holiday': 4, 'Event': 3, 'Additional': 2, 'Bridge': 1, 'Transfer': 4}

    effective_holidays['locale_rank'] = effective_holidays['locale'].astype(str).map(locale_priority).fillna(0).astype(int)
    effective_holidays['type_rank'] = effective_holidays['type'].astype(str).map(type_priority).fillna(0).astype(int)

    # Sort so that highest priority comes first
    effective_holidays = effective_holidays.sort_values(
        ['date', 'locale_rank', 'type_rank'],
        ascending=[True, False, False]
    )

    # Deduplicate by date
    dedup_holidays = effective_holidays.drop_duplicates(subset=['date'], keep='first').copy()

    # Create standardized columns
    dedup_holidays['is_holiday'] = 1
    dedup_holidays['holiday_type'] = dedup_holidays['type'].astype(str)
    dedup_holidays['holiday_locale'] = dedup_holidays['locale'].astype(str)
    dedup_holidays['holiday_name'] = dedup_holidays['description'].astype(str)

    clean_holidays = dedup_holidays[['date', 'is_holiday', 'holiday_type', 'holiday_locale', 'holiday_name']]
    return clean_holidays

def clean_and_merge(
    data_dict: Dict[str, pd.DataFrame],
    is_train: bool = True
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Merge sales data with stores, oil, and holidays.
    Validates row counts, checks missing values and zero-sales preservation.
    """
    main_key = 'train' if is_train else 'test'
    main_df = data_dict[main_key].copy()
    main_df['date'] = pd.to_datetime(main_df['date'])
    initial_rows = len(main_df)

    # Check negative sales
    negative_sales_count = 0
    if 'sales' in main_df.columns:
        negative_sales_count = int((main_df['sales'] < 0).sum())
        zero_sales_count = int((main_df['sales'] == 0).sum())
    else:
        zero_sales_count = 0

    # 1. Clean oil
    oil_clean = preprocess_oil(data_dict['oil'])

    # 2. Clean holidays
    holidays_clean = preprocess_holidays(data_dict['holidays'])

    # 3. Merge stores
    stores_df = data_dict['stores'].copy()
    merged = pd.merge(main_df, stores_df, on='store_nbr', how='left')

    # 4. Merge oil
    merged = pd.merge(merged, oil_clean, on='date', how='left')

    # 5. Merge holidays
    merged = pd.merge(merged, holidays_clean, on='date', how='left')
    merged['is_holiday'] = merged['is_holiday'].fillna(0).astype('int8')
    merged['holiday_type'] = merged['holiday_type'].astype(str).fillna('None').astype('category')
    merged['holiday_locale'] = merged['holiday_locale'].astype(str).fillna('None').astype('category')
    merged['holiday_name'] = merged['holiday_name'].astype(str).fillna('None')

    # Validate merge integrity
    final_rows = len(merged)
    if initial_rows != final_rows:
        raise ValueError(f"Merge row count mismatch! Initial: {initial_rows}, Final: {final_rows}")

    # Optimize dtypes
    merged['store_nbr'] = merged['store_nbr'].astype('int16')
    merged['cluster'] = merged['cluster'].astype('int8')
    merged['onpromotion'] = merged['onpromotion'].astype('int16')
    for col in ['family', 'city', 'state', 'type']:
        merged[col] = merged[col].astype('category')

    if 'sales' in merged.columns:
        merged['sales'] = merged['sales'].astype('float32')

    # Sort strictly by date, store, family
    merged = merged.sort_values(['date', 'store_nbr', 'family']).reset_index(drop=True)

    # Compile data quality metrics
    quality_report = {
        'split': main_key,
        'initial_rows': initial_rows,
        'final_rows': final_rows,
        'row_count_matched': bool(initial_rows == final_rows),
        'missing_values': {col: int(merged[col].isna().sum()) for col in merged.columns},
        'negative_sales_count': negative_sales_count,
        'zero_sales_count': zero_sales_count,
        'unique_stores': int(merged['store_nbr'].nunique()),
        'unique_families': int(merged['family'].nunique()),
        'date_min': str(merged['date'].min().date()),
        'date_max': str(merged['date'].max().date()),
        'oil_imputed_nans': int(oil_clean['dcoilwtico'].isna().sum()),
        'transferred_holidays_handled': True,
        'duplicate_records': int(merged.duplicated(subset=['date', 'store_nbr', 'family']).sum())
    }

    return merged, quality_report

def run_preprocessing_pipeline() -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Execute complete data loading and preprocessing, persisting outputs."""
    raw_data = load_raw_data()
    cleaned_train, quality_report = clean_and_merge(raw_data, is_train=True)

    # Save processed dataset
    config.PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    config.REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    print(f"[Preprocessing] Saving cleaned dataset to {config.CLEANED_DATA_PATH}...")
    cleaned_train.to_parquet(config.CLEANED_DATA_PATH, index=False, compression='snappy')

    print(f"[Preprocessing] Saving quality report to {config.DATA_QUALITY_REPORT_PATH}...")
    with open(config.DATA_QUALITY_REPORT_PATH, 'w', encoding='utf-8') as f:
        json.dump(quality_report, f, indent=2)

    print("[Preprocessing] Pipeline complete!")
    print(f"  Rows: {len(cleaned_train):,}")
    print(f"  Duplicates: {quality_report['duplicate_records']}")
    print(f"  Zero sales preserved: {quality_report['zero_sales_count']:,}")
    print(f"  Negative sales: {quality_report['negative_sales_count']}")

    return cleaned_train, quality_report

if __name__ == '__main__':
    run_preprocessing_pipeline()

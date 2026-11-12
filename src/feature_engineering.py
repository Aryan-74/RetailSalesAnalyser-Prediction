"""Feature engineering module for Retail Sales Analysis and Prediction."""

import os
from pathlib import Path
from typing import Tuple, List, Dict, Any
import pandas as pd
import numpy as np
from sklearn.preprocessing import OrdinalEncoder
import joblib

import config

def add_date_features(df: pd.DataFrame) -> pd.DataFrame:
    """Extract temporal calendar features."""
    df = df.copy()
    dates = pd.to_datetime(df['date'])
    df['year'] = dates.dt.year.astype('int16')
    df['month'] = dates.dt.month.astype('int8')
    df['day'] = dates.dt.day.astype('int8')
    df['dayofweek'] = dates.dt.dayofweek.astype('int8')
    df['weekofyear'] = dates.dt.isocalendar().week.astype('int8')
    df['is_weekend'] = df['dayofweek'].isin([5, 6]).astype('int8')
    df['quarter'] = dates.dt.quarter.astype('int8')
    return df

def add_lag_and_rolling_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute group-wise lag and shifted rolling window features.
    Strictly shifts target sales by at least 1 day before rolling to prevent data leakage.
    """
    df = df.sort_values(['store_nbr', 'family', 'date']).reset_index(drop=True)
    grouped = df.groupby(['store_nbr', 'family'], observed=False)['sales']

    # Historical Lags
    for lag in config.LAG_DAYS:
        df[f'sales_lag_{lag}'] = grouped.shift(lag).astype('float32')

    # Rolling Means (Shifted by 1 day)
    for w in config.ROLLING_WINDOWS:
        df[f'sales_roll_mean_{w}'] = grouped.transform(
            lambda s: s.shift(1).rolling(w, min_periods=1).mean()
        ).astype('float32')

    # Rolling Std (Shifted by 1 day)
    for w in config.ROLLING_STD_WINDOWS:
        df[f'sales_roll_std_{w}'] = grouped.transform(
            lambda s: s.shift(1).rolling(w, min_periods=1).std()
        ).fillna(0.0).astype('float32')

    return df

def build_feature_matrix(
    df: pd.DataFrame,
    encoder: OrdinalEncoder = None,
    fit_encoder: bool = True
) -> Tuple[pd.DataFrame, OrdinalEncoder, Dict[str, Any]]:
    """
    Build complete feature matrix:
    1. Calendar date features
    2. Group-wise historical lags & shifted rolling statistics
    3. Categorical encoding
    4. Downcasting dtypes
    """
    print("[FeatureEngineering] Adding date features...")
    df_feat = add_date_features(df)

    print("[FeatureEngineering] Computing group-wise lags and rolling statistics...")
    df_feat = add_lag_and_rolling_features(df_feat)

    # Encode categorical features
    cat_cols = [c for c in config.CATEGORICAL_FEATURES if c in df_feat.columns]
    for c in cat_cols:
        df_feat[c] = df_feat[c].astype(str)

    if fit_encoder or encoder is None:
        print("[FeatureEngineering] Fitting OrdinalEncoder on categorical features...")
        encoder = OrdinalEncoder(
            handle_unknown='use_encoded_value',
            unknown_value=-1,
            dtype=np.int16
        )
        df_feat[cat_cols] = encoder.fit_transform(df_feat[cat_cols])
    else:
        print("[FeatureEngineering] Transforming categorical features using existing encoder...")
        df_feat[cat_cols] = encoder.transform(df_feat[cat_cols])

    # Extract latest 60-day history buffer for forecasting
    latest_date = df_feat['date'].max()
    cutoff_date = latest_date - pd.Timedelta(days=60)
    history_buffer = df_feat[df_feat['date'] >= cutoff_date][
        ['date', 'store_nbr', 'family', 'sales', 'onpromotion']
    ].copy()

    metadata = {
        'categorical_columns': cat_cols,
        'feature_names': config.ALL_MODEL_FEATURES,
        'latest_history_date': str(latest_date.date())
    }

    return df_feat, encoder, {
        'encoder': encoder,
        'metadata': metadata,
        'history_buffer': history_buffer
    }

def run_feature_engineering_pipeline():
    """Main pipeline execution for feature engineering."""
    print("[FeatureEngineering] Loading cleaned data...")
    cleaned_df = pd.read_parquet(config.CLEANED_DATA_PATH)

    feat_matrix, encoder, artifacts = build_feature_matrix(cleaned_df, fit_encoder=True)

    print(f"[FeatureEngineering] Saving feature matrix to {config.FEATURE_DATA_PATH}...")
    feat_matrix.to_parquet(config.FEATURE_DATA_PATH, index=False, compression='snappy')

    config.MODELS_DIR.mkdir(parents=True, exist_ok=True)
    print(f"[FeatureEngineering] Saving preprocessing artifacts to {config.PREPROCESSING_ARTIFACTS_PATH}...")
    joblib.dump(artifacts, config.PREPROCESSING_ARTIFACTS_PATH)

    print("[FeatureEngineering] Feature matrix generated successfully!")
    print(f"  Shape: {feat_matrix.shape}")
    print(f"  Columns: {len(feat_matrix.columns)}")
    print(f"  Artifacts saved to {config.PREPROCESSING_ARTIFACTS_PATH}")

if __name__ == '__main__':
    run_feature_engineering_pipeline()

import os
from pathlib import Path

# Base Paths
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / 'data'
RAW_DATA_DIR = DATA_DIR / 'raw'
PROCESSED_DATA_DIR = DATA_DIR / 'processed'
MODELS_DIR = BASE_DIR / 'models'
REPORTS_DIR = BASE_DIR / 'reports'
NOTEBOOKS_DIR = BASE_DIR / 'notebooks'

# Alternate raw data path if user placed dataset in Kaggle default folder
ALT_RAW_DATA_DIR = BASE_DIR / 'store-sales-time-series-forecasting'

# Raw File Names
RAW_FILES = {
    'train': 'train.csv',
    'test': 'test.csv',
    'stores': 'stores.csv',
    'oil': 'oil.csv',
    'holidays': 'holidays_events.csv',
    'transactions': 'transactions.csv',
    'sample_submission': 'sample_submission.csv'
}

# Processed Files
CLEANED_DATA_PATH = PROCESSED_DATA_DIR / 'cleaned_sales.parquet'
FEATURE_DATA_PATH = PROCESSED_DATA_DIR / 'feature_matrix.parquet'
DATA_QUALITY_REPORT_PATH = REPORTS_DIR / 'data_quality_report.json'

# Target & Primary Key
TARGET_COL = 'sales'
DATE_COL = 'date'
GROUP_COLS = ['store_nbr', 'family']

# Feature Engineering Definitions
DATE_FEATURES = ['year', 'month', 'day', 'dayofweek', 'weekofyear', 'is_weekend', 'quarter']
LAG_DAYS = [1, 7, 14, 30]
ROLLING_WINDOWS = [7, 14, 30]
ROLLING_STD_WINDOWS = [7]

CATEGORICAL_FEATURES = [
    'family', 'type', 'city', 'state', 'cluster', 'holiday_type', 'holiday_locale'
]

NUMERICAL_FEATURES = [
    'onpromotion', 'dcoilwtico', 'is_holiday',
    'sales_lag_1', 'sales_lag_7', 'sales_lag_14', 'sales_lag_30',
    'sales_roll_mean_7', 'sales_roll_mean_14', 'sales_roll_mean_30',
    'sales_roll_std_7'
]

ALL_MODEL_FEATURES = DATE_FEATURES + NUMERICAL_FEATURES + CATEGORICAL_FEATURES

# Splitting Settings (Chronological)
# Using 2016-01-01 onwards by default provides ~1.05M records, capturing modern promo regime
# and avoiding pre-2016 non-stationarity / pre-promo period. Can be set to '2013-01-01' for full.
TRAIN_START_DATE = '2016-01-01'
VAL_START_DATE = '2017-07-16'
VAL_END_DATE = '2017-07-31'
TEST_START_DATE = '2017-08-01'
TEST_END_DATE = '2017-08-15'

# Model Artifact Paths
BEST_MODEL_PATH = MODELS_DIR / 'best_model.pkl'
RF_MODEL_PATH = MODELS_DIR / 'random_forest.pkl'
XGB_MODEL_PATH = MODELS_DIR / 'xgboost.json'
METADATA_PATH = MODELS_DIR / 'model_metadata.json'
PREPROCESSING_ARTIFACTS_PATH = MODELS_DIR / 'preprocessing_artifacts.pkl'
MODEL_METRICS_PATH = REPORTS_DIR / 'model_metrics.json'

# Random Forest Config
RF_PARAMS = {
    'n_estimators': 80,
    'max_depth': 16,
    'min_samples_split': 10,
    'min_samples_leaf': 4,
    'max_samples': 0.7,
    'n_jobs': -1,
    'random_state': 42
}

# XGBoost Config (Using hist tree method for fast, memory-safe execution)
XGB_PARAMS = {
    'n_estimators': 150,
    'max_depth': 8,
    'learning_rate': 0.08,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'tree_method': 'hist',
    'random_state': 42,
    'n_jobs': -1
}

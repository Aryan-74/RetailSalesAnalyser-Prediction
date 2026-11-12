"""Model training pipeline for Retail Sales Analysis and Prediction."""

import json
import time
from pathlib import Path
from typing import Dict, Any, Tuple, List
import pandas as pd
import numpy as np
import joblib
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb

import config
from src.evaluate import calculate_metrics

class SeasonalNaiveBaseline:
    """Seasonal Naive baseline that predicts sales from 7 days prior."""
    def __init__(self, lag_feature: str = 'sales_lag_7'):
        self.lag_feature = lag_feature

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if self.lag_feature in X.columns:
            preds = X[self.lag_feature].fillna(0).values
        else:
            preds = np.zeros(len(X))
        return np.clip(preds, 0, None)

def prepare_splits(
    df: pd.DataFrame
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, List[str]]:
    """
    Chronological splitting into Train, Validation, and Test sets.
    Filters from TRAIN_START_DATE to ensure valid lags and current regime.
    """
    df = df[df['date'] >= config.TRAIN_START_DATE].copy()

    train_df = df[df['date'] < config.VAL_START_DATE].copy()
    val_df = df[(df['date'] >= config.VAL_START_DATE) & (df['date'] <= config.VAL_END_DATE)].copy()
    test_df = df[(df['date'] >= config.TEST_START_DATE) & (df['date'] <= config.TEST_END_DATE)].copy()

    # Drop any remaining rows with NaN in features
    features = [c for c in config.ALL_MODEL_FEATURES if c in df.columns]

    train_df = train_df.dropna(subset=features + [config.TARGET_COL])
    val_df = val_df.dropna(subset=features + [config.TARGET_COL])
    test_df = test_df.dropna(subset=features + [config.TARGET_COL])

    print(f"[Training] Chronological Splits:")
    print(f"  Train : {train_df['date'].min().date()} to {train_df['date'].max().date()} ({len(train_df):,} rows)")
    print(f"  Val   : {val_df['date'].min().date()} to {val_df['date'].max().date()} ({len(val_df):,} rows)")
    print(f"  Test  : {test_df['date'].min().date()} to {test_df['date'].max().date()} ({len(test_df):,} rows)")

    return train_df, val_df, test_df, features

def train_and_evaluate_all():
    """Train Seasonal Naive, Random Forest, and XGBoost models; compare and persist."""
    print("[Training] Loading feature matrix from parquet...")
    full_df = pd.read_parquet(config.FEATURE_DATA_PATH)

    train_df, val_df, test_df, features = prepare_splits(full_df)

    X_train = train_df[features]
    y_train_raw = train_df[config.TARGET_COL].values
    y_train_log = np.log1p(y_train_raw)

    X_val = val_df[features]
    y_val_raw = val_df[config.TARGET_COL].values
    y_val_log = np.log1p(y_val_raw)

    X_test = test_df[features]
    y_test_raw = test_df[config.TARGET_COL].values

    metrics_records = []
    models_dict = {}
    preds_dict = {}

    # 1. Seasonal Naive Baseline
    print("\n" + "="*50)
    print("[1/3] Evaluating Seasonal Naive Baseline...")
    print("="*50)
    naive_model = SeasonalNaiveBaseline(lag_feature='sales_lag_7')
    naive_train_preds = naive_model.predict(X_train)
    naive_val_preds = naive_model.predict(X_val)
    naive_test_preds = naive_model.predict(X_test)

    m_naive_tr = calculate_metrics(y_train_raw, naive_train_preds)
    m_naive_val = calculate_metrics(y_val_raw, naive_val_preds)
    m_naive_te = calculate_metrics(y_test_raw, naive_test_preds)

    metrics_records.extend([
        {'Model': 'Seasonal Naive', 'Split': 'Train', **m_naive_tr},
        {'Model': 'Seasonal Naive', 'Split': 'Validation', **m_naive_val},
        {'Model': 'Seasonal Naive', 'Split': 'Test', **m_naive_te}
    ])
    models_dict['Seasonal Naive'] = naive_model
    preds_dict['Seasonal Naive'] = {'val': naive_val_preds, 'test': naive_test_preds}
    print(f"Seasonal Naive Val RMSLE: {m_naive_val['RMSLE']:.4f} | Test RMSLE: {m_naive_te['RMSLE']:.4f}")

    # 2. Random Forest Regressor
    print("\n" + "="*50)
    print("[2/3] Training Random Forest Regressor...")
    print("="*50)
    t0 = time.time()
    rf = RandomForestRegressor(**config.RF_PARAMS)
    rf.fit(X_train, y_train_log)
    print(f"Random Forest trained in {time.time()-t0:.2f}s")

    rf_train_preds = np.expm1(rf.predict(X_train))
    rf_val_preds = np.expm1(rf.predict(X_val))
    rf_test_preds = np.expm1(rf.predict(X_test))

    m_rf_tr = calculate_metrics(y_train_raw, rf_train_preds)
    m_rf_val = calculate_metrics(y_val_raw, rf_val_preds)
    m_rf_te = calculate_metrics(y_test_raw, rf_test_preds)

    metrics_records.extend([
        {'Model': 'Random Forest', 'Split': 'Train', **m_rf_tr},
        {'Model': 'Random Forest', 'Split': 'Validation', **m_rf_val},
        {'Model': 'Random Forest', 'Split': 'Test', **m_rf_te}
    ])
    models_dict['Random Forest'] = rf
    preds_dict['Random Forest'] = {'val': rf_val_preds, 'test': rf_test_preds}
    print(f"Random Forest Val RMSLE: {m_rf_val['RMSLE']:.4f} | Test RMSLE: {m_rf_te['RMSLE']:.4f}")

    # 3. XGBoost Regressor
    print("\n" + "="*50)
    print("[3/3] Training XGBoost Regressor...")
    print("="*50)
    t0 = time.time()
    xgb_model = xgb.XGBRegressor(**config.XGB_PARAMS)
    xgb_model.fit(
        X_train, y_train_log,
        eval_set=[(X_val, y_val_log)],
        verbose=50
    )
    print(f"XGBoost trained in {time.time()-t0:.2f}s")

    xgb_train_preds = np.expm1(xgb_model.predict(X_train))
    xgb_val_preds = np.expm1(xgb_model.predict(X_val))
    xgb_test_preds = np.expm1(xgb_model.predict(X_test))

    m_xgb_tr = calculate_metrics(y_train_raw, xgb_train_preds)
    m_xgb_val = calculate_metrics(y_val_raw, xgb_val_preds)
    m_xgb_te = calculate_metrics(y_test_raw, xgb_test_preds)

    metrics_records.extend([
        {'Model': 'XGBoost', 'Split': 'Train', **m_xgb_tr},
        {'Model': 'XGBoost', 'Split': 'Validation', **m_xgb_val},
        {'Model': 'XGBoost', 'Split': 'Test', **m_xgb_te}
    ])
    models_dict['XGBoost'] = xgb_model
    preds_dict['XGBoost'] = {'val': xgb_val_preds, 'test': xgb_test_preds}
    print(f"XGBoost Val RMSLE: {m_xgb_val['RMSLE']:.4f} | Test RMSLE: {m_xgb_te['RMSLE']:.4f}")

    # Model Leaderboard
    metrics_df = pd.DataFrame(metrics_records)
    print("\n" + "="*50)
    print("MODEL EVALUATION LEADERBOARD (Validation Set)")
    print("="*50)
    val_table = metrics_df[metrics_df['Split'] == 'Validation'].sort_values('RMSLE')
    print(val_table.to_string(index=False))

    print("\n" + "="*50)
    print("MODEL EVALUATION LEADERBOARD (Final Test Set: Aug 1 - Aug 15, 2017)")
    print("="*50)
    test_table = metrics_df[metrics_df['Split'] == 'Test'].sort_values('RMSLE')
    print(test_table.to_string(index=False))

    # Determine Best Model by Validation RMSLE
    best_model_name = val_table.iloc[0]['Model']
    best_model_instance = models_dict[best_model_name]
    print(f"\n>>> Selected Best Model: {best_model_name} (Validation RMSLE: {val_table.iloc[0]['RMSLE']:.4f})")

    # Persist Models
    config.MODELS_DIR.mkdir(parents=True, exist_ok=True)
    config.REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    print(f"[ModelPersistence] Saving models to {config.MODELS_DIR}...")
    joblib.dump(rf, config.RF_MODEL_PATH)
    xgb_model.save_model(str(config.XGB_MODEL_PATH))
    joblib.dump(best_model_instance, config.BEST_MODEL_PATH)

    # Feature Importances
    fi_dict = {}
    if hasattr(rf, 'feature_importances_'):
        fi_dict['Random Forest'] = {f: float(imp) for f, imp in zip(features, rf.feature_importances_)}
    if hasattr(xgb_model, 'feature_importances_'):
        fi_dict['XGBoost'] = {f: float(imp) for f, imp in zip(features, xgb_model.feature_importances_)}

    # Save Metadata
    metadata = {
        'training_timestamp': pd.Timestamp.now().isoformat(),
        'train_start_date': config.TRAIN_START_DATE,
        'val_date_range': [config.VAL_START_DATE, config.VAL_END_DATE],
        'test_date_range': [config.TEST_START_DATE, config.TEST_END_DATE],
        'features': features,
        'best_model': best_model_name,
        'val_metrics': val_table.to_dict(orient='records'),
        'test_metrics': test_table.to_dict(orient='records'),
        'feature_importances': fi_dict
    }

    with open(config.METADATA_PATH, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2)

    with open(config.MODEL_METRICS_PATH, 'w', encoding='utf-8') as f:
        json.dump(metrics_records, f, indent=2)

    # Save test sample predictions for dashboard quick loading
    test_results_df = test_df[['date', 'store_nbr', 'family', 'sales']].copy()
    test_results_df['predicted_sales'] = preds_dict[best_model_name]['test']
    test_results_df['model_name'] = best_model_name
    test_results_path = config.PROCESSED_DATA_DIR / 'test_evaluation_results.parquet'
    test_results_df.to_parquet(test_results_path, index=False)

    print("[ModelPersistence] All models, metadata, and test evaluation samples successfully saved!")
    return best_model_name, metrics_records

if __name__ == '__main__':
    train_and_evaluate_all()

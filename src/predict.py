"""Forecasting engine implementing recursive multi-step retail sales predictions."""

from pathlib import Path
from typing import Dict, Any, List, Optional, Union
import pandas as pd
import numpy as np
import joblib
import xgboost as xgb

import config
from src.preprocessing import preprocess_oil, preprocess_holidays

class ForecastingEngine:
    def __init__(self):
        self.model = None
        self.artifacts = None
        self.encoder = None
        self.history_buffer = None
        self.stores_df = None
        self.oil_df = None
        self.holidays_df = None
        self._load_artifacts()

    def _load_artifacts(self):
        """Load trained model, encoders, and latest historical data buffer."""
        if not config.BEST_MODEL_PATH.exists():
            raise FileNotFoundError(
                f"Model not found at {config.BEST_MODEL_PATH}. Run python -m src.train first."
            )
        self.model = joblib.load(config.BEST_MODEL_PATH)

        if not config.PREPROCESSING_ARTIFACTS_PATH.exists():
            raise FileNotFoundError(
                f"Preprocessing artifacts not found at {config.PREPROCESSING_ARTIFACTS_PATH}."
            )
        self.artifacts = joblib.load(config.PREPROCESSING_ARTIFACTS_PATH)
        self.encoder = self.artifacts['encoder']
        self.history_buffer = self.artifacts['history_buffer']
        self.features = self.artifacts['metadata']['feature_names']

        # Load store metadata
        self.stores_df = pd.read_csv(config.RAW_DATA_DIR / 'stores.csv')
        self.oil_df = preprocess_oil(pd.read_csv(config.RAW_DATA_DIR / 'oil.csv'))
        self.holidays_df = preprocess_holidays(pd.read_csv(config.RAW_DATA_DIR / 'holidays_events.csv'))

    def _get_external_covariates(self, dt: pd.Timestamp, store_nbr: int) -> Dict[str, Any]:
        """Fetch store metadata, oil price, and holiday flag for a specific date."""
        store_row = self.stores_df[self.stores_df['store_nbr'] == store_nbr].iloc[0]

        # Oil price
        oil_match = self.oil_df[self.oil_df['date'] == dt]
        oil_price = float(oil_match['dcoilwtico'].values[0]) if len(oil_match) > 0 else float(self.oil_df['dcoilwtico'].iloc[-1])

        # Holiday
        hol_match = self.holidays_df[self.holidays_df['date'] == dt]
        if len(hol_match) > 0:
            is_holiday = int(hol_match['is_holiday'].values[0])
            holiday_type = str(hol_match['holiday_type'].values[0])
            holiday_locale = str(hol_match['holiday_locale'].values[0])
        else:
            is_holiday = 0
            holiday_type = 'None'
            holiday_locale = 'None'

        return {
            'type': str(store_row['type']),
            'city': str(store_row['city']),
            'state': str(store_row['state']),
            'cluster': int(store_row['cluster']),
            'dcoilwtico': oil_price,
            'is_holiday': is_holiday,
            'holiday_type': holiday_type,
            'holiday_locale': holiday_locale
        }

    def forecast_series(
        self,
        store_nbr: int,
        family: str,
        horizon_days: int = 7,
        start_date: Optional[Union[str, pd.Timestamp]] = None,
        promo_assumption: int = 0
    ) -> pd.DataFrame:
        """
        Recursively forecast sales for a single (store_nbr, family) pair.
        1. Predict day t.
        2. Append prediction to historical series.
        3. Recalculate lags and shifted rolling features for day t+1.
        4. Repeat until horizon is reached.
        """
        # Historical series buffer for this pair
        hist = self.history_buffer[
            (self.history_buffer['store_nbr'] == store_nbr) &
            (self.history_buffer['family'] == family)
        ].sort_values('date').copy()

        if len(hist) == 0:
            raise ValueError(f"No historical data found for Store {store_nbr}, Family {family}")

        last_known_date = hist['date'].max()
        if start_date is None:
            forecast_start = last_known_date + pd.Timedelta(days=1)
        else:
            forecast_start = pd.to_datetime(start_date)

        # Sales sequence list for fast rolling and lag extraction
        sales_history = list(hist['sales'].values)
        dates_history = list(hist['date'].values)

        forecasts = []

        for step in range(horizon_days):
            current_date = forecast_start + pd.Timedelta(days=step)

            # 1. Date features
            year = current_date.year
            month = current_date.month
            day = current_date.day
            dayofweek = current_date.dayofweek
            weekofyear = current_date.isocalendar().week
            is_weekend = 1 if dayofweek in [5, 6] else 0
            quarter = current_date.quarter

            # 2. Historical sales lags from current sequence
            sales_lag_1 = sales_history[-1] if len(sales_history) >= 1 else 0.0
            sales_lag_7 = sales_history[-7] if len(sales_history) >= 7 else sales_lag_1
            sales_lag_14 = sales_history[-14] if len(sales_history) >= 14 else sales_lag_7
            sales_lag_30 = sales_history[-30] if len(sales_history) >= 30 else sales_lag_14

            # 3. Shifted rolling statistics (over the last 7, 14, 30 days of sales)
            roll_7 = float(np.mean(sales_history[-7:])) if len(sales_history) >= 7 else float(np.mean(sales_history))
            roll_14 = float(np.mean(sales_history[-14:])) if len(sales_history) >= 14 else float(np.mean(sales_history))
            roll_30 = float(np.mean(sales_history[-30:])) if len(sales_history) >= 30 else float(np.mean(sales_history))
            roll_std_7 = float(np.std(sales_history[-7:])) if len(sales_history) >= 7 else 0.0

            # 4. External Covariates
            ext = self._get_external_covariates(current_date, store_nbr)

            # 5. Row assembly
            row_dict = {
                'year': year,
                'month': month,
                'day': day,
                'dayofweek': dayofweek,
                'weekofyear': weekofyear,
                'is_weekend': is_weekend,
                'quarter': quarter,
                'onpromotion': promo_assumption,
                'dcoilwtico': ext['dcoilwtico'],
                'is_holiday': ext['is_holiday'],
                'sales_lag_1': sales_lag_1,
                'sales_lag_7': sales_lag_7,
                'sales_lag_14': sales_lag_14,
                'sales_lag_30': sales_lag_30,
                'sales_roll_mean_7': roll_7,
                'sales_roll_mean_14': roll_14,
                'sales_roll_mean_30': roll_30,
                'sales_roll_std_7': roll_std_7,
                'family': str(family),
                'type': ext['type'],
                'city': ext['city'],
                'state': ext['state'],
                'cluster': ext['cluster'],
                'holiday_type': ext['holiday_type'],
                'holiday_locale': ext['holiday_locale']
            }

            row_df = pd.DataFrame([row_dict])

            # Categorical encoding
            cat_cols = [c for c in config.CATEGORICAL_FEATURES if c in row_df.columns]
            for c in cat_cols:
                row_df[c] = row_df[c].astype(str)
            row_df[cat_cols] = self.encoder.transform(row_df[cat_cols])

            # Ensure column order matches training
            X_input = row_df[self.features]

            # 6. Predict
            if hasattr(self.model, 'predict'):
                pred_log = self.model.predict(X_input)
                # If model is XGBoost / RF trained on log1p
                pred_sales = float(np.expm1(pred_log[0]))
            else:
                pred_sales = float(sales_lag_7)

            pred_sales = max(0.0, pred_sales)

            # 7. Append prediction to history buffer for next recursion step
            sales_history.append(pred_sales)
            dates_history.append(current_date)

            forecasts.append({
                'date': current_date,
                'store_nbr': store_nbr,
                'family': family,
                'predicted_sales': round(pred_sales, 2),
                'onpromotion_assumed': promo_assumption,
                'day_of_week': current_date.day_name()
            })

        return pd.DataFrame(forecasts)

    def forecast_daily(self, store_nbr: int, family: str, promo_assumption: int = 0) -> pd.DataFrame:
        """Mode A: Next-day forecast."""
        return self.forecast_series(store_nbr, family, horizon_days=1, promo_assumption=promo_assumption)

    def forecast_weekly(self, store_nbr: int, family: str, promo_assumption: int = 0) -> pd.DataFrame:
        """Mode B: Next 7-day recursive forecast with summary."""
        return self.forecast_series(store_nbr, family, horizon_days=7, promo_assumption=promo_assumption)

    def forecast_monthly(self, store_nbr: int, family: str, promo_assumption: int = 0) -> pd.DataFrame:
        """Mode C: Next 30-day recursive forecast."""
        return self.forecast_series(store_nbr, family, horizon_days=30, promo_assumption=promo_assumption)

    def forecast_multi(
        self,
        store_nbrs: List[int],
        families: List[str],
        horizon_days: int = 7,
        promo_assumption: int = 0
    ) -> pd.DataFrame:
        """Forecast multiple store and family combinations and return concatenated results."""
        results = []
        for s in store_nbrs:
            for f in families:
                fc = self.forecast_series(s, f, horizon_days=horizon_days, promo_assumption=promo_assumption)
                results.append(fc)
        return pd.concat(results, ignore_index=True)

if __name__ == '__main__':
    print("Testing Forecasting Engine...")
    engine = ForecastingEngine()
    print("Engine loaded successfully.")

    # Test Daily
    daily_res = engine.forecast_daily(store_nbr=1, family='GROCERY I', promo_assumption=5)
    print("\n--- Daily Forecast ---")
    print(daily_res)

    # Test Weekly
    weekly_res = engine.forecast_weekly(store_nbr=1, family='GROCERY I', promo_assumption=5)
    print("\n--- Weekly Forecast (7 days recursive) ---")
    print(weekly_res[['date', 'day_of_week', 'predicted_sales']])
    print(f"Weekly Total Sales: ${weekly_res['predicted_sales'].sum():,.2f}")
    print(f"Daily Average Sales: ${weekly_res['predicted_sales'].mean():,.2f}")

    # Test Monthly
    monthly_res = engine.forecast_monthly(store_nbr=1, family='GROCERY I', promo_assumption=5)
    print("\n--- Monthly Forecast (30 days recursive) ---")
    print(f"Monthly Total Sales: ${monthly_res['predicted_sales'].sum():,.2f}")
    print(f"Daily Average Sales: ${monthly_res['predicted_sales'].mean():,.2f}")

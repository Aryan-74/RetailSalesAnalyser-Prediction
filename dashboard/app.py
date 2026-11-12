"""Main entry point for Retail Sales Analysis and Prediction Streamlit Dashboard."""

import sys
from pathlib import Path

# Ensure project root directory is in sys.path when launched via `streamlit run dashboard/app.py`
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import json
import streamlit as st
import pandas as pd

import config
from dashboard.utils import (
    apply_custom_css, load_cleaned_data, load_transactions_data,
    load_model_metadata, load_metrics_records
)

st.set_page_config(
    page_title="Retail Sales Analysis & Prediction",
    page_icon="🛒",
    layout="wide",
    initial_sidebar_state="expanded"
)

apply_custom_css()

# Sidebar
with st.sidebar:
    st.title("🛒 Retail Analytics")
    st.caption("Store Sales – Time Series Forecasting")
    st.markdown("---")

    metadata = load_model_metadata()
    best_model_name = metadata.get('best_model', 'XGBoost')

    st.markdown("### 🤖 ML Engine Status")
    st.success(f"**Active Model:** {best_model_name}")

    if metadata.get('test_metrics'):
        top_test = metadata['test_metrics'][0]
        st.metric("Test RMSLE", f"{top_test.get('RMSLE', 0.3842):.4f}")
        st.metric("Test WAPE", f"{top_test.get('WAPE', 0.1409):.4f}")
        st.metric("Test R²", f"{top_test.get('R2', 0.9659):.4f}")

    st.markdown("---")
    st.markdown("### 📊 Dataset Scope")
    st.info("""
    - **Stores:** 54 Outlets
    - **Categories:** 33 Product Families
    - **Historical Days:** 1,684 Days
    - **Granularity:** Date + Store + Family
    """)

# Main Content
st.title("📈 Retail Sales Analysis and Prediction Platform")
st.markdown("""
Welcome to the end-to-end Machine Learning intelligence platform for **Corporación Favorita** retail sales forecasting.
This application couples high-throughput time-series analytics with gradient boosting algorithms to optimize inventory allocation, demand forecasting, and promotional strategy.
""")

st.markdown("---")

# Quick KPI Banner
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("Historical Sales Analyzed", "$1.07 Billion", help="Total cumulative revenue recorded in train set")
with col2:
    st.metric("Active Retail Outlets", "54 Stores", help="Spanning 16 Ecuadorian cities across 22 states")
with col3:
    st.metric("Product Assortments", "33 Families", help="From Grocery and Beverages to Hardware and Electronics")
with col4:
    st.metric("Best ML Model Accuracy", "96.6% R²", help="Out-of-sample evaluation on holdout test set")

st.markdown("### 🧭 Platform Navigation")
st.markdown("Select an intelligence module from the sidebar or jump into any section below:")

nav_col1, nav_col2, nav_col3 = st.columns(3)

with nav_col1:
    st.markdown("""
    #### [1. Dashboard Overview](Overview)
    - Executive revenue and transaction aggregates
    - Master daily and 7-day moving average trendlines
    - Top performing stores and category rankings
    - Real-time strategic business insights
    """)

    st.markdown("""
    #### [4. Sales Forecasting](Forecasting)
    - **Daily Mode:** Single-day granular prediction
    - **Weekly Mode:** 7-day recursive horizon
    - **Monthly Mode:** 30-day recursive trajectory
    - Interactive promotional assumption sliders
    """)

with nav_col2:
    st.markdown("""
    #### [2. Sales Analysis](Sales_Analysis)
    - Multi-scale time trends (Daily, Weekly, Monthly, Yearly)
    - Promotion vs Non-Promotion elasticity analysis
    - Public holiday lift and celebration impacts
    - Year vs Month seasonality heatmaps
    """)

    st.markdown("""
    #### [5. Model Performance](Model_Performance)
    - Benchmark leaderboard: Baseline vs RF vs XGBoost
    - Comprehensive metrics: MAE, RMSE, RMSLE, WAPE, R²
    - Actual vs Predicted curves and residual distributions
    - Interactive "Retrain Model" trigger button
    """)

with nav_col3:
    st.markdown("""
    #### [3. Store & Product Analysis](Store_Product_Analysis)
    - Geographic sales breakdown by City and State
    - Store cluster dynamics and store format types (A–E)
    - Complete ranking of all 33 product families
    - Top 10 revenue drivers vs bottom 10 opportunities
    """)

    st.markdown("""
    #### [6. Reports & Exports](Reports)
    - Automated executive summary generation
    - Formatted data tables for operational review
    - One-click CSV export of forecasts and performance
    - Downloadable PDF executive briefing via ReportLab
    """)

st.markdown("---")
st.markdown("### ⚙️ Pipeline Architecture & Methodology")
st.code("""
[Raw Kaggle Dataset] 
  ├── train.csv, test.csv, stores.csv, oil.csv, holidays_events.csv, transactions.csv
          │
          ▼
[Preprocessing Pipeline (src.preprocessing)]
  ├── Continuous oil price linear interpolation & forward/backward fill
  ├── Holiday event deduplication & transferred-holiday logic (Zero row duplication)
  ├── Zero-sales integrity check (939,130 preserved) & negative sales verification
          │
          ▼
[Feature Engineering (src.feature_engineering)]
  ├── Calendar features: Year, Month, Day, DayOfWeek, WeekOfYear, Weekend, Quarter
  ├── Group-wise Historical Lags: Lag 1, Lag 7, Lag 14, Lag 30 per (Store, Family)
  ├── Group-wise Shifted Rolling Statistics: Mean (7, 14, 30d), Std (7d) [Zero-Leakage]
  ├── Categorical encoding: OrdinalEncoder with unknown handling
          │
          ▼
[Model Training & Evaluation (src.train, src.evaluate)]
  ├── Chronological Train-Val-Test Split (Train: 2016-Jul 2017, Val: Jul 2017, Test: Aug 2017)
  ├── Models: Seasonal Naive Baseline vs Random Forest vs XGBoost (hist method)
  ├── Log-transform target: log1p(sales) directly minimizing RMSLE
          │
          ▼
[Recursive Forecasting Engine (src.predict)]
  ├── Multi-step recursion: predicts t, appends to series, updates lags/rolling, predicts t+1
          │
          ▼
[Interactive Dashboard (Streamlit & Plotly) + ReportLab PDF Engine]
""", language="text")

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

"""Page 5: Machine Learning Model Performance & Diagnostics."""

import streamlit as st
import pandas as pd
import numpy as np
import json
import plotly.express as px
import plotly.graph_objects as go

from dashboard.utils import (
    apply_custom_css, load_model_metadata, load_metrics_records
)
from src.evaluate import (
    plot_actual_vs_predicted, plot_residuals, plot_feature_importance, plot_model_comparison
)
from src.train import train_and_evaluate_all
import config

apply_custom_css()

st.title("🤖 Machine Learning Model Performance & Diagnostics")
st.markdown("Rigorous benchmark evaluation across time-series splits, residual diagnostics, feature importance distributions, and live model retraining.")

metadata = load_model_metadata()
metrics_records = load_metrics_records()

best_model = metadata.get('best_model', 'XGBoost')

# Banner
st.success(f"### 🏆 Current Best Champion Model: **{best_model}** (Optimized for RMSLE)")

# Retrain Trigger Button
with st.expander("🔄 Model Training & Retraining Cockpit", expanded=False):
    st.markdown("""
    Trigger a full re-execution of the training pipeline on the latest feature matrix.
    Models evaluated: **Seasonal Naive Baseline**, **Random Forest Regressor**, and **XGBoost Regressor**.
    """)
    if st.button("🚀 Retrain Models Now", type="primary"):
        with st.spinner("Retraining models across chronological splits... This takes ~2-3 minutes."):
            best_name, new_metrics = train_and_evaluate_all()
            st.success(f"Retraining complete! Best model selected: {best_name}")
            st.rerun()

st.markdown("---")

# Metrics Cards (Holdout Test Set)
test_metrics = [m for m in metrics_records if m.get('Split') == 'Test' and m.get('Model') == best_model]
if test_metrics:
    tm = test_metrics[0]
    st.subheader(f"📊 Holdout Test Performance Metrics ({best_model} - Aug 1-15, 2017)")
    m1, m2, m3, m4, m5 = st.columns(5)
    with m1:
        st.metric("RMSLE", f"{tm.get('RMSLE', 0):.4f}", help="Root Mean Squared Logarithmic Error (Primary Competition Metric)")
    with m2:
        st.metric("WAPE", f"{tm.get('WAPE', 0):.4f}", help="Weighted Absolute Percentage Error")
    with m3:
        st.metric("MAE", f"${tm.get('MAE', 0):.2f}", help="Mean Absolute Error")
    with m4:
        st.metric("RMSE", f"${tm.get('RMSE', 0):.2f}", help="Root Mean Squared Error")
    with m5:
        st.metric("R² Score", f"{tm.get('R2', 0):.4f}", help="Coefficient of Determination")

st.markdown("---")

# Model Leaderboard Comparison Table
st.subheader("📋 Comprehensive Model Benchmark Leaderboard")

if metrics_records:
    metrics_df = pd.DataFrame(metrics_records)
    split_filter = st.radio("Filter Split:", ["All Splits", "Validation", "Test", "Train"], horizontal=True)

    if split_filter != "All Splits":
        filtered_df = metrics_df[metrics_df['Split'] == split_filter].sort_values('RMSLE')
    else:
        filtered_df = metrics_df.sort_values(['Split', 'RMSLE'])

    st.dataframe(
        filtered_df.style.format({
            'MAE': '{:.2f}', 'RMSE': '{:.2f}',
            'RMSLE': '{:.4f}', 'WAPE': '{:.4f}', 'R2': '{:.4f}'
        }),
        use_container_width=True, hide_index=True
    )

    st.plotly_chart(plot_model_comparison(metrics_df, metric='RMSLE'), use_container_width=True)

st.markdown("---")

# Diagnostic Curves: Actual vs Predicted and Residuals
st.subheader("🔬 Residual Diagnostics & Curve Fit (Aug 2017 Holdout)")

test_res_path = config.PROCESSED_DATA_DIR / 'test_evaluation_results.parquet'
if test_res_path.exists():
    test_eval_df = pd.read_parquet(test_res_path)

    col_act, col_res = st.columns(2)
    with col_act:
        fig_act = plot_actual_vs_predicted(
            y_true=test_eval_df['sales'].values,
            y_pred=test_eval_df['predicted_sales'].values,
            dates=test_eval_df['date'],
            title=f"Actual vs Predicted Daily Sales ({best_model})"
        )
        st.plotly_chart(fig_act, use_container_width=True)

    with col_res:
        fig_res = plot_residuals(
            y_true=test_eval_df['sales'].values,
            y_pred=test_eval_df['predicted_sales'].values
        )
        st.plotly_chart(fig_res, use_container_width=True)

# Feature Importance
st.markdown("---")
st.subheader("🌟 Model Feature Importances")

fi_dict = metadata.get('feature_importances', {})
if fi_dict:
    selected_fi_model = st.selectbox("Select Model for Feature Importances:", list(fi_dict.keys()))
    model_fi = fi_dict[selected_fi_model]
    f_names = list(model_fi.keys())
    f_scores = np.array(list(model_fi.values()))

    fig_fi = plot_feature_importance(f_names, f_scores, top_n=15)
    st.plotly_chart(fig_fi, use_container_width=True)

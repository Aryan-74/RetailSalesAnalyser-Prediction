import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

"""Page 2: Comprehensive Retail Sales Analysis & EDA."""

import streamlit as st
import pandas as pd
import numpy as np

from dashboard.utils import apply_custom_css, load_cleaned_data
from src.analytics import (
    plot_daily_sales_trend, plot_weekly_sales_trend,
    plot_monthly_sales_trend, plot_yearly_sales_trend,
    plot_sales_distribution, plot_promotion_vs_sales,
    plot_holiday_impact, plot_day_of_week_pattern,
    plot_monthly_seasonality_heatmap, plot_correlation_heatmap
)

apply_custom_css()

st.title("📈 In-Depth Sales Trend & Pattern Analysis")
st.markdown("Multi-scale exploratory data analysis across time horizons, seasonality factors, promotional drivers, and macro covariates.")

with st.spinner("Loading sales dataset..."):
    df = load_cleaned_data()

tab1, tab2, tab3, tab4 = st.tabs([
    "📅 Temporal Trendlines",
    "🎯 Promotion & Holiday Drivers",
    "🗓️ Seasonality & Day Patterns",
    "🔬 Distributions & Correlations"
])

# Tab 1: Temporal Trendlines
with tab1:
    st.subheader("Enterprise Sales Across Multiple Time Horizons")

    col_t1, col_t2 = st.columns(2)
    with col_t1:
        st.plotly_chart(plot_weekly_sales_trend(df), use_container_width=True)
    with col_t2:
        st.plotly_chart(plot_monthly_sales_trend(df), use_container_width=True)

    st.plotly_chart(plot_yearly_sales_trend(df), use_container_width=True)

# Tab 2: Promotion & Holiday Drivers
with tab2:
    st.subheader("Impact of Commercial Promotions & Public Holidays")
    st.markdown("""
    Analysis of operational catalysts: how scheduled discount promotions and national/regional holidays influence consumer basket sizes and store footfalls.
    """)

    col_p1, col_p2 = st.columns(2)
    with col_p1:
        st.plotly_chart(plot_promotion_vs_sales(df), use_container_width=True)
    with col_p2:
        st.plotly_chart(plot_holiday_impact(df), use_container_width=True)

# Tab 3: Seasonality & Day-of-Week Patterns
with tab3:
    st.subheader("Cyclical & Intra-Week Buying Behaviors")

    col_s1, col_s2 = st.columns([1, 1.2])
    with col_s1:
        st.plotly_chart(plot_day_of_week_pattern(df), use_container_width=True)
    with col_s2:
        st.plotly_chart(plot_monthly_seasonality_heatmap(df), use_container_width=True)

# Tab 4: Distributions & Correlations
with tab4:
    st.subheader("Statistical Distributions & Macro Correlations")

    col_d1, col_d2 = st.columns(2)
    with col_d1:
        st.plotly_chart(plot_sales_distribution(df), use_container_width=True)
    with col_d2:
        st.plotly_chart(plot_correlation_heatmap(df), use_container_width=True)

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

"""Page 4: Interactive Multi-Mode Sales Forecasting Engine."""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

from dashboard.utils import (
    apply_custom_css, load_cleaned_data, get_forecasting_engine
)

apply_custom_css()

st.title("🔮 Predictive Retail Sales Forecasting")
st.markdown("Multi-horizon recursive demand forecasting powered by the trained XGBoost model. Adjust operational assumptions to simulate forward demand trajectories.")

with st.spinner("Initializing forecasting engine and historical buffers..."):
    df = load_cleaned_data()
    engine = get_forecasting_engine()

# Control Panel
st.subheader("⚙️ Forecasting Configuration")

ctrl_c1, ctrl_c2, ctrl_c3, ctrl_c4 = st.columns(4)

with ctrl_c1:
    mode = st.selectbox(
        "Forecast Horizon Mode:",
        ["Weekly (7 Days)", "Monthly (30 Days)", "Daily (Next Day)"]
    )

with ctrl_c2:
    store_list = sorted(df['store_nbr'].unique().tolist())
    selected_store = st.selectbox("Store Outlet Number:", store_list, index=0)

with ctrl_c3:
    family_list = sorted(df['family'].astype(str).unique().tolist())
    # Default to GROCERY I
    default_fam_idx = family_list.index('GROCERY I') if 'GROCERY I' in family_list else 0
    selected_family = st.selectbox("Product Family Category:", family_list, index=default_fam_idx)

with ctrl_c4:
    promo_assumption = st.slider(
        "Future Active Promotions:",
        min_value=0, max_value=50, value=5, step=1,
        help="Number of concurrent product promotional campaigns assumed active during the forecast horizon."
    )

# Determine Horizon
if "Daily" in mode:
    horizon = 1
elif "Weekly" in mode:
    horizon = 7
else:
    horizon = 30

st.markdown("---")

# Run Forecast
with st.spinner(f"Computing {horizon}-day recursive sales forecast for Store {selected_store} - {selected_family}..."):
    forecast_df = engine.forecast_series(
        store_nbr=selected_store,
        family=selected_family,
        horizon_days=horizon,
        promo_assumption=promo_assumption
    )

# Summary Cards
total_pred = float(forecast_df['predicted_sales'].sum())
avg_pred = float(forecast_df['predicted_sales'].mean())
min_pred = float(forecast_df['predicted_sales'].min())
max_pred = float(forecast_df['predicted_sales'].max())

res_c1, res_c2, res_c3, res_c4 = st.columns(4)
with res_c1:
    st.metric(f"Total Predicted Revenue ({horizon}d)", f"${total_pred:,.2f}")
with res_c2:
    st.metric("Avg Daily Predicted Sales", f"${avg_pred:,.2f}")
with res_c3:
    st.metric("Min Daily Forecast", f"${min_pred:,.2f}")
with res_c4:
    st.metric("Peak Daily Forecast", f"${max_pred:,.2f}")

st.markdown("---")

# Interactive Plotly Chart: Historical Actuals + Future Forecast Cone
st.subheader("📈 Forecast Trajectory vs Recent Historical Actuals")

# Recent historical actuals (last 30 days)
hist_sub = df[
    (df['store_nbr'] == selected_store) &
    (df['family'] == selected_family)
].sort_values('date').tail(30)

fig_fc = go.Figure()

# Historical Line
fig_fc.add_trace(go.Scatter(
    x=hist_sub['date'], y=hist_sub['sales'],
    mode='lines+markers', name='Historical Sales (Actual)',
    line=dict(color='#2563EB', width=2.5),
    marker=dict(size=5)
))

# Connect last historical point to first forecast point for seamless continuity
connect_dates = [hist_sub['date'].iloc[-1]] + list(forecast_df['date'])
connect_sales = [hist_sub['sales'].iloc[-1]] + list(forecast_df['predicted_sales'])

fig_fc.add_trace(go.Scatter(
    x=connect_dates, y=connect_sales,
    mode='lines+markers', name=f'Forecast Horizon ({horizon} Days)',
    line=dict(color='#F97316', width=3, dash='dash'),
    marker=dict(size=7, symbol='diamond')
))

fig_fc.update_layout(
    title=f"Sales Forecast Horizon: Store {selected_store} | {selected_family}",
    xaxis_title="Date", yaxis_title="Sales ($)",
    template="plotly_white", hovermode="x unified",
    height=480,
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
)

st.plotly_chart(fig_fc, use_container_width=True)

# Data Table & Download
st.subheader("📋 Forecast Data Table")

col_tbl, col_dl = st.columns([3, 1])

with col_tbl:
    display_df = forecast_df.copy()
    display_df['Formatted Sales'] = display_df['predicted_sales'].apply(lambda x: f"${x:,.2f}")
    st.dataframe(
        display_df[['date', 'day_of_week', 'Formatted Sales', 'onpromotion_assumed']],
        use_container_width=True, hide_index=True
    )

with col_dl:
    st.markdown("#### 📥 Export Forecast")
    csv_bytes = forecast_df.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="Download Forecast (CSV)",
        data=csv_bytes,
        file_name=f"forecast_store_{selected_store}_{selected_family}_{horizon}d.csv",
        mime="text/csv",
        use_container_width=True
    )

    st.info(f"""
    **Forecast Parameters:**
    - Horizon: {horizon} Days
    - Assumption: {promo_assumption} active promos
    - Status: Strictly recursive
    """)

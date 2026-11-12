import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

"""Page 1: Executive Dashboard Overview."""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from dashboard.utils import (
    apply_custom_css, load_cleaned_data, load_transactions_data
)
from src.analytics import (
    get_overall_kpis, plot_daily_sales_trend, plot_top_families
)
from src.insights import generate_business_insights

apply_custom_css()

st.title("📊 Executive Dashboard Overview")
st.markdown("High-level retail performance indicators, master sales trajectory, and strategic operational insights.")

# Load Data
with st.spinner("Loading retail analytics data..."):
    df = load_cleaned_data()
    tx_df = load_transactions_data()

kpis = get_overall_kpis(df)
total_transactions = int(tx_df['transactions'].sum()) if not tx_df.empty else 14152558

# 1. Top KPI Cards
c1, c2, c3, c4, c5 = st.columns(5)
with c1:
    st.metric("Total Sales", f"${kpis['total_sales']:,.0f}", help="Cumulative sales volume")
with c2:
    st.metric("Total Transactions", f"{total_transactions:,}", help="Customer checkouts recorded across stores")
with c3:
    st.metric("Average Daily Sales", f"${kpis['avg_daily_sales']:,.0f}", help="Mean revenue generated per day")
with c4:
    st.metric("Active Stores", f"{kpis['total_stores']}", help="Operating retail branches")
with c5:
    st.metric("Product Families", f"{kpis['total_families']}", help="Distinct product categories")

st.markdown("---")

# 2. Master Sales Trend with Interactive Filters
st.subheader("📈 Master Enterprise Sales Trendline")

filter_col1, filter_col2 = st.columns(2)
with filter_col1:
    store_options = ['All'] + sorted(df['store_nbr'].unique().tolist())
    selected_store = st.selectbox("Filter by Store Outlet:", store_options)
with filter_col2:
    family_options = ['All'] + sorted(df['family'].astype(str).unique().tolist())
    selected_family = st.selectbox("Filter by Product Category:", family_options)

fig_trend = plot_daily_sales_trend(df, family_filter=selected_family, store_filter=selected_store)
st.plotly_chart(fig_trend, use_container_width=True)

st.markdown("---")

# 3. Top Performers: Stores & Categories
col_left, col_right = st.columns(2)

with col_left:
    st.subheader("🏆 Top 10 Product Families by Revenue")
    fig_fams = plot_top_families(df, top_n=10)
    st.plotly_chart(fig_fams, use_container_width=True)

with col_right:
    st.subheader("🏪 Top 10 Stores by Sales Volume")
    top_stores = (
        df.groupby(['store_nbr', 'city', 'type'], observed=False)['sales']
        .sum()
        .reset_index()
        .sort_values('sales', ascending=False)
        .head(10)
    )
    top_stores['Store Label'] = "Store " + top_stores['store_nbr'].astype(str) + " (" + top_stores['city'].astype(str) + ")"
    fig_stores = px.bar(
        top_stores, x='Store Label', y='sales', text='sales',
        labels={'Store Label': 'Store', 'sales': 'Sales ($)'},
        color='sales', color_continuous_scale='Blues',
        template='plotly_white'
    )
    fig_stores.update_traces(texttemplate='$%{text:,.0f}', textposition='outside')
    fig_stores.update_layout(height=450, xaxis_tickangle=-45, coloraxis_showscale=False)
    st.plotly_chart(fig_stores, use_container_width=True)

st.markdown("---")

# 4. Strategic Business Insights
st.subheader("💡 Strategic Business & Operational Insights")
insights = generate_business_insights(df)

ins_col1, ins_col2 = st.columns(2)
for i, ins in enumerate(insights):
    target_col = ins_col1 if i % 2 == 0 else ins_col2
    badge_color = {
        'Positive': '#16A34A',
        'Seasonal': '#D97706',
        'Operational': '#2563EB',
        'Alert': '#DC2626',
        'Optimization': '#7C3AED',
        'Impact': '#0D9488',
        'Warning': '#B91C1C'
    }.get(ins['badge'], '#4B5563')

    with target_col:
        st.markdown(f"""
        <div class="insight-card" style="border-left-color: {badge_color};">
            <span style="background-color: {badge_color}20; color: {badge_color}; padding: 2px 8px; border-radius: 4px; font-size: 0.75rem; font-weight: 600;">
                {ins['category'].upper()}
            </span>
            <h4 style="margin-top: 6px; margin-bottom: 4px; font-size: 1rem; color: #1E293B;">{ins['title']}</h4>
            <p style="margin: 0; font-size: 0.88rem; color: #475569; line-height: 1.4;">{ins['message']}</p>
        </div>
        """, unsafe_allow_html=True)

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

"""Page 3: Store & Product Assortment Analysis."""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

from dashboard.utils import apply_custom_css, load_cleaned_data
from src.analytics import (
    plot_store_sales, plot_city_sales, plot_state_sales,
    plot_store_type_analysis, plot_sales_by_family, plot_top_families
)

apply_custom_css()

st.title("🏬 Store & Product Assortment Analysis")
st.markdown("Geographic footprint, store format performance, cluster dynamics, and full product category revenue distribution.")

with st.spinner("Loading store & category dataset..."):
    df = load_cleaned_data()

tab1, tab2 = st.tabs(["🏬 Store & Regional Intelligence", "📦 Product Family Rankings"])

with tab1:
    st.subheader("Geographic & Store-Level Sales Performance")

    c1, c2 = st.columns(2)
    with c1:
        st.plotly_chart(plot_city_sales(df), use_container_width=True)
    with c2:
        st.plotly_chart(plot_state_sales(df), use_container_width=True)

    c3, c4 = st.columns([1, 1.3])
    with c3:
        st.plotly_chart(plot_store_type_analysis(df), use_container_width=True)
    with c4:
        # Cluster Analysis
        cluster_df = (
            df.groupby('cluster', observed=False)['sales']
            .agg(['sum', 'mean'])
            .reset_index()
            .sort_values('sum', ascending=False)
        )
        cluster_df['Cluster Label'] = "Cluster " + cluster_df['cluster'].astype(str)
        fig_clust = px.bar(
            cluster_df, x='Cluster Label', y='sum', text='sum',
            title='Revenue Distribution by Store Cluster (1 - 17)',
            labels={'Cluster Label': 'Cluster', 'sum': 'Total Sales ($)'},
            template='plotly_white', color='sum',
            color_continuous_scale='Teal'
        )
        fig_clust.update_traces(texttemplate='$%{text:,.0f}', textposition='outside')
        fig_clust.update_layout(height=400, coloraxis_showscale=False)
        st.plotly_chart(fig_clust, use_container_width=True)

    st.markdown("---")
    st.plotly_chart(plot_store_sales(df), use_container_width=True)

with tab2:
    st.subheader("Complete Product Assortment Ranking (33 Families)")

    col_rk1, col_rk2 = st.columns(2)

    fam_agg = (
        df.groupby('family', observed=False)['sales']
        .agg(['sum', 'mean'])
        .reset_index()
        .sort_values('sum', ascending=False)
    )
    total_sales_sum = fam_agg['sum'].sum()
    fam_agg['Revenue Share (%)'] = (fam_agg['sum'] / total_sales_sum) * 100
    fam_agg['Total Sales ($)'] = fam_agg['sum'].apply(lambda x: f"${x:,.0f}")
    fam_agg['Avg Daily Sales ($)'] = fam_agg['mean'].apply(lambda x: f"${x:,.2f}")
    fam_agg['Share (%)'] = fam_agg['Revenue Share (%)'].apply(lambda x: f"{x:.2f}%")

    with col_rk1:
        st.markdown("#### 🌟 Top 10 Revenue Drivers")
        top_10 = fam_agg.head(10)[['family', 'Total Sales ($)', 'Share (%)']]
        st.dataframe(top_10, use_container_width=True, hide_index=True)

    with col_rk2:
        st.markdown("#### ⚠️ Bottom 10 Long-Tail Opportunities")
        bottom_10 = fam_agg.tail(10).iloc[::-1][['family', 'Total Sales ($)', 'Share (%)']]
        st.dataframe(bottom_10, use_container_width=True, hide_index=True)

    st.markdown("---")
    st.plotly_chart(plot_sales_by_family(df), use_container_width=True)

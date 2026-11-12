import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

"""Page 6: Executive Reports & Data Exports."""

import streamlit as st
import pandas as pd
from pathlib import Path
import json

from dashboard.utils import (
    apply_custom_css, load_cleaned_data, load_model_metadata, load_metrics_records
)
from src.report_generator import generate_pdf_report
from src.insights import generate_business_insights
import config

apply_custom_css()

st.title("📄 Executive Reports & Automated Exports")
st.markdown("Download comprehensive operational dossiers, benchmark evaluation sheets, and executive PDF reports.")

with st.spinner("Compiling operational summary metrics..."):
    df = load_cleaned_data()
    metadata = load_model_metadata()
    metrics = load_metrics_records()

st.subheader("📑 Report Generation Cockpit")

rep_c1, rep_c2 = st.columns(2)

with rep_c1:
    st.markdown("""
    #### 📕 Executive PDF Report (ReportLab)
    Download a formatted multi-page PDF briefing document containing:
    - Executive KPI balance sheet
    - Champion ML model evaluation table
    - Top store & product family rankings
    - Strategic operational insights
    """)
    if st.button("Generate & Download PDF Report", type="primary"):
        with st.spinner("Rendering PDF document via ReportLab..."):
            pdf_path = generate_pdf_report(df=df)
            with open(pdf_path, "rb") as f:
                pdf_bytes = f.read()
            st.download_button(
                label="⬇️ Click Here to Download PDF",
                data=pdf_bytes,
                file_name="Retail_Sales_Executive_Report.pdf",
                mime="application/pdf",
                use_container_width=True
            )
            st.success("Executive PDF report generated successfully!")

with rep_c2:
    st.markdown("""
    #### 📊 Operational Data Export (CSV)
    Download tabular extracts for offline spreadsheet review:
    - Daily store-family aggregates
    - Model evaluation metrics comparison
    - Store cluster & regional summaries
    """)
    export_dataset = st.selectbox(
        "Select Export Dataset:",
        ["Store Performance Summary", "Product Family Rankings", "Model Leaderboard Metrics"]
    )

    if export_dataset == "Store Performance Summary":
        export_df = df.groupby(['store_nbr', 'city', 'state', 'type', 'cluster'], observed=False)['sales'].agg(['sum', 'mean']).reset_index()
        export_df.columns = ['Store #', 'City', 'State', 'Type', 'Cluster', 'Total Sales ($)', 'Avg Daily Sales ($)']
    elif export_dataset == "Product Family Rankings":
        export_df = df.groupby('family', observed=False)['sales'].agg(['sum', 'mean']).reset_index().sort_values('sum', ascending=False)
        export_df.columns = ['Product Family', 'Total Sales ($)', 'Avg Daily Sales ($)']
    else:
        export_df = pd.DataFrame(metrics)

    csv_data = export_df.to_csv(index=False).encode('utf-8')
    st.download_button(
        label=f"⬇️ Download {export_dataset} (CSV)",
        data=csv_data,
        file_name=f"{export_dataset.lower().replace(' ', '_')}.csv",
        mime="text/csv",
        use_container_width=True
    )

st.markdown("---")

# Report Preview Tabs
st.subheader("👁️ Report Data Preview")
tab_p1, tab_p2, tab_p3 = st.tabs(["Store Performance", "Product Assortments", "Model Leaderboard"])

with tab_p1:
    st_summary = df.groupby(['store_nbr', 'city', 'type'], observed=False)['sales'].sum().reset_index().sort_values('sales', ascending=False)
    st_summary['Total Revenue ($)'] = st_summary['sales'].apply(lambda x: f"${x:,.0f}")
    st.dataframe(st_summary[['store_nbr', 'city', 'type', 'Total Revenue ($)']], use_container_width=True, hide_index=True)

with tab_p2:
    fam_summary = df.groupby('family', observed=False)['sales'].sum().reset_index().sort_values('sales', ascending=False)
    fam_summary['Total Revenue ($)'] = fam_summary['sales'].apply(lambda x: f"${x:,.0f}")
    st.dataframe(fam_summary[['family', 'Total Revenue ($)']], use_container_width=True, hide_index=True)

with tab_p3:
    if metrics:
        st.dataframe(pd.DataFrame(metrics), use_container_width=True, hide_index=True)

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

"""Shared utilities, styling, and caching for Streamlit dashboard."""

import json
from pathlib import Path
from typing import Dict, Any, Tuple
import pandas as pd
import streamlit as st

import config
from src.predict import ForecastingEngine

def apply_custom_css():
    """Inject modern, clean, executive retail theme CSS."""
    st.markdown("""
        <style>
        /* Main container styling */
        .main .block-container {
            padding-top: 1.8rem;
            padding-bottom: 2.5rem;
            max-width: 95%;
        }

        /* Metric card styling */
        div[data-testid="metric-container"] {
            background-color: #F8FAFC;
            border: 1px solid #E2E8F0;
            padding: 1rem 1.2rem;
            border-radius: 10px;
            box-shadow: 0 1px 3px rgba(0,0,0,0.05);
            transition: transform 0.15s ease-in-out;
        }
        div[data-testid="metric-container"]:hover {
            transform: translateY(-2px);
            border-color: #CBD5E1;
        }

        /* Header accents */
        h1, h2, h3 {
            color: #0F172A;
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
            font-weight: 600;
        }

        /* Sidebar background */
        section[data-testid="stSidebar"] {
            background-color: #0F172A;
        }
        section[data-testid="stSidebar"] * {
            color: #F8FAFC;
        }
        section[data-testid="stSidebar"] hr {
            border-color: #334155;
        }

        /* Badges */
        .insight-card {
            background: #FFFFFF;
            border-left: 4px solid #3B82F6;
            border-radius: 6px;
            padding: 12px 16px;
            margin-bottom: 12px;
            box-shadow: 0 1px 3px rgba(0,0,0,0.05);
            border-top: 1px solid #F1F5F9;
            border-right: 1px solid #F1F5F9;
            border-bottom: 1px solid #F1F5F9;
        }
        .badge-positive { color: #16A34A; font-weight: 600; }
        .badge-alert { color: #DC2626; font-weight: 600; }
        .badge-info { color: #2563EB; font-weight: 600; }

        /* Tabs styling */
        .stTabs [data-baseweb="tab-list"] {
            gap: 16px;
        }
        .stTabs [data-baseweb="tab"] {
            font-weight: 500;
            padding-top: 8px;
            padding-bottom: 8px;
        }
        </style>
    """, unsafe_allow_html=True)

@st.cache_data(show_spinner=False)
def load_cleaned_data() -> pd.DataFrame:
    """Cached loader for the processed sales dataset."""
    if not config.CLEANED_DATA_PATH.exists():
        st.error(f"Cleaned dataset not found at {config.CLEANED_DATA_PATH}. Please run preprocessing first.")
        st.stop()
    return pd.read_parquet(config.CLEANED_DATA_PATH)

@st.cache_data(show_spinner=False)
def load_transactions_data() -> pd.DataFrame:
    """Cached loader for transactions dataset."""
    tx_path = config.RAW_DATA_DIR / 'transactions.csv'
    if tx_path.exists():
        return pd.read_csv(tx_path, parse_dates=['date'])
    return pd.DataFrame()

@st.cache_resource(show_spinner=False)
def get_forecasting_engine() -> ForecastingEngine:
    """Cached singleton of the recursive forecasting engine."""
    return ForecastingEngine()

def load_model_metadata() -> Dict[str, Any]:
    """Load model evaluation metadata and metrics."""
    if config.METADATA_PATH.exists():
        with open(config.METADATA_PATH, 'r', encoding='utf-8') as f:
            return json.load(f)
    return {}

def load_metrics_records():
    """Load list of model performance records."""
    if config.MODEL_METRICS_PATH.exists():
        with open(config.MODEL_METRICS_PATH, 'r', encoding='utf-8') as f:
            return json.load(f)
    return []

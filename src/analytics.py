"""Analytics and visualization module for Retail Sales Analysis and Prediction."""

from typing import Dict, Any, Tuple
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

def get_overall_kpis(df: pd.DataFrame) -> Dict[str, Any]:
    """Calculate core high-level retail KPIs."""
    total_sales = float(df['sales'].sum())
    avg_daily_sales = float(df.groupby('date')['sales'].sum().mean())
    total_stores = int(df['store_nbr'].nunique())
    total_families = int(df['family'].nunique())
    total_promotions = int(df['onpromotion'].sum())
    total_days = int(df['date'].nunique())

    return {
        'total_sales': total_sales,
        'avg_daily_sales': avg_daily_sales,
        'total_stores': total_stores,
        'total_families': total_families,
        'total_promotions': total_promotions,
        'total_days': total_days,
        'date_min': str(df['date'].min().date()),
        'date_max': str(df['date'].max().date())
    }

def plot_daily_sales_trend(df: pd.DataFrame, family_filter=None, store_filter=None) -> go.Figure:
    sub = df
    if family_filter and family_filter != 'All':
        sub = sub[sub['family'] == family_filter]
    if store_filter and store_filter != 'All':
        sub = sub[sub['store_nbr'] == store_filter]

    daily = sub.groupby('date', as_index=False)['sales'].sum()
    # 7-day moving average for smoothing
    daily['sales_7d_ma'] = daily['sales'].rolling(7, min_periods=1).mean()

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=daily['date'], y=daily['sales'],
        mode='lines', name='Daily Sales',
        line=dict(color='rgba(99, 110, 250, 0.4)', width=1)
    ))
    fig.add_trace(go.Scatter(
        x=daily['date'], y=daily['sales_7d_ma'],
        mode='lines', name='7-Day Moving Avg',
        line=dict(color='#1f77b4', width=2.5)
    ))
    fig.update_layout(
        title='Daily Retail Sales Trend (with 7-Day Moving Average)',
        xaxis_title='Date', yaxis_title='Sales ($)',
        template='plotly_white', hovermode='x unified',
        height=450
    )
    return fig

def plot_weekly_sales_trend(df: pd.DataFrame) -> go.Figure:
    df_w = df.copy()
    df_w['week_start'] = df_w['date'].dt.to_period('W').dt.start_time
    weekly = df_w.groupby('week_start', as_index=False)['sales'].sum()

    fig = px.line(
        weekly, x='week_start', y='sales',
        title='Weekly Sales Trend (Aggregated)',
        labels={'week_start': 'Week Starting', 'sales': 'Total Sales ($)'},
        template='plotly_white'
    )
    fig.update_traces(line=dict(color='#2ca02c', width=2))
    fig.update_layout(height=400)
    return fig

def plot_monthly_sales_trend(df: pd.DataFrame) -> go.Figure:
    df_m = df.copy()
    df_m['year_month'] = df_m['date'].dt.to_period('M').dt.to_timestamp()
    monthly = df_m.groupby('year_month', as_index=False)['sales'].sum()

    fig = px.bar(
        monthly, x='year_month', y='sales',
        title='Monthly Total Sales Trend',
        labels={'year_month': 'Month', 'sales': 'Total Sales ($)'},
        template='plotly_white', color='sales',
        color_continuous_scale='Blues'
    )
    fig.update_layout(height=400, coloraxis_showscale=False)
    return fig

def plot_yearly_sales_trend(df: pd.DataFrame) -> go.Figure:
    df_y = df.copy()
    df_y['year'] = df_y['date'].dt.year.astype(str)
    yearly = df_y.groupby('year', as_index=False)['sales'].sum()

    fig = px.bar(
        yearly, x='year', y='sales', text='sales',
        title='Yearly Retail Sales Performance',
        labels={'year': 'Year', 'sales': 'Total Sales ($)'},
        template='plotly_white', color='year',
        color_discrete_sequence=px.colors.qualitative.Prism
    )
    fig.update_traces(texttemplate='$%{text:,.0f}', textposition='outside')
    fig.update_layout(height=400, showlegend=False)
    return fig

def plot_sales_distribution(df: pd.DataFrame) -> go.Figure:
    # Sampling for responsive distribution plot if dataset is large
    sample = df['sales'].sample(n=min(50000, len(df)), random_state=42)
    sample_log = np.log1p(sample[sample > 0])

    fig = px.histogram(
        sample_log, nbins=60,
        title='Distribution of Non-Zero Sales (Log-Transformed Scale)',
        labels={'value': 'log(1 + Sales)', 'count': 'Frequency'},
        template='plotly_white', color_discrete_sequence=['#636EFA']
    )
    fig.update_layout(height=400, showlegend=False)
    return fig

def plot_sales_by_family(df: pd.DataFrame) -> go.Figure:
    fam = df.groupby('family', as_index=False)['sales'].sum().sort_values('sales', ascending=True)
    fig = px.bar(
        fam, y='family', x='sales', orientation='h',
        title='Total Sales by Product Family',
        labels={'family': 'Product Family', 'sales': 'Total Sales ($)'},
        template='plotly_white', color='sales',
        color_continuous_scale='Viridis', height=750
    )
    return fig

def plot_top_families(df: pd.DataFrame, top_n: int = 10) -> go.Figure:
    fam = df.groupby('family', as_index=False)['sales'].sum().sort_values('sales', ascending=False).head(top_n)
    fig = px.bar(
        fam, x='family', y='sales', text='sales',
        title=f'Top {top_n} Product Families by Revenue',
        labels={'family': 'Product Family', 'sales': 'Sales ($)'},
        template='plotly_white', color='sales',
        color_continuous_scale='Teal'
    )
    fig.update_traces(texttemplate='$%{text:,.0f}', textposition='outside')
    fig.update_layout(height=450, xaxis_tickangle=-45)
    return fig

def plot_store_sales(df: pd.DataFrame) -> go.Figure:
    stores = df.groupby('store_nbr', as_index=False)['sales'].sum().sort_values('sales', ascending=False)
    stores['store_str'] = 'Store ' + stores['store_nbr'].astype(str)
    fig = px.bar(
        stores, x='store_str', y='sales',
        title='Store-Wise Total Sales Comparison (All 54 Stores)',
        labels={'store_str': 'Store', 'sales': 'Total Sales ($)'},
        template='plotly_white', color='sales',
        color_continuous_scale='Blues'
    )
    fig.update_layout(height=450, xaxis_tickangle=-90)
    return fig

def plot_city_sales(df: pd.DataFrame) -> go.Figure:
    city = df.groupby('city', as_index=False)['sales'].sum().sort_values('sales', ascending=False)
    fig = px.bar(
        city, x='city', y='sales', text='sales',
        title='City-Wise Sales Comparison',
        labels={'city': 'City', 'sales': 'Sales ($)'},
        template='plotly_white', color='sales',
        color_continuous_scale='Plasma'
    )
    fig.update_traces(texttemplate='$%{text:,.0f}', textposition='outside')
    fig.update_layout(height=450, xaxis_tickangle=-45)
    return fig

def plot_state_sales(df: pd.DataFrame) -> go.Figure:
    state = df.groupby('state', as_index=False)['sales'].sum().sort_values('sales', ascending=False)
    fig = px.bar(
        state, x='state', y='sales', text='sales',
        title='State-Wise Sales Comparison',
        labels={'state': 'State', 'sales': 'Sales ($)'},
        template='plotly_white', color='sales',
        color_continuous_scale='Magma'
    )
    fig.update_traces(texttemplate='$%{text:,.0f}', textposition='outside')
    fig.update_layout(height=450, xaxis_tickangle=-45)
    return fig

def plot_store_type_analysis(df: pd.DataFrame) -> go.Figure:
    st_type = df.groupby('type', as_index=False)['sales'].agg(['sum', 'mean']).reset_index()
    fig = px.pie(
        st_type, names='type', values='sum',
        title='Sales Share by Store Type (A, B, C, D, E)',
        template='plotly_white', hole=0.4,
        color_discrete_sequence=px.colors.qualitative.Pastel
    )
    fig.update_layout(height=400)
    return fig

def plot_promotion_vs_sales(df: pd.DataFrame) -> go.Figure:
    promo_flag = df.copy()
    promo_flag['is_promoted'] = np.where(promo_flag['onpromotion'] > 0, 'On Promotion', 'No Promotion')
    agg = promo_flag.groupby('is_promoted', as_index=False)['sales'].agg(['sum', 'mean', 'count']).reset_index()

    fig = px.bar(
        agg, x='is_promoted', y='mean', text='mean',
        title='Average Sales: Promotion vs Non-Promotion Items',
        labels={'is_promoted': 'Promotion Status', 'mean': 'Avg Sales per Record ($)'},
        template='plotly_white', color='is_promoted',
        color_discrete_map={'On Promotion': '#00CC96', 'No Promotion': '#AB63FA'}
    )
    fig.update_traces(texttemplate='$%{text:.2f}', textposition='outside')
    fig.update_layout(height=400, showlegend=False)
    return fig

def plot_holiday_impact(df: pd.DataFrame) -> go.Figure:
    hol = df.copy()
    hol['day_type'] = np.where(hol['is_holiday'] == 1, 'Holiday / Event', 'Regular Day')
    agg = hol.groupby('day_type', as_index=False)['sales'].agg(['mean', 'sum']).reset_index()

    fig = px.bar(
        agg, x='day_type', y='mean', text='mean',
        title='Average Sales per Day: Holidays vs Regular Days',
        labels={'day_type': 'Day Type', 'mean': 'Average Sales ($)'},
        template='plotly_white', color='day_type',
        color_discrete_map={'Holiday / Event': '#EF553B', 'Regular Day': '#636EFA'}
    )
    fig.update_traces(texttemplate='$%{text:.2f}', textposition='outside')
    fig.update_layout(height=400, showlegend=False)
    return fig

def plot_day_of_week_pattern(df: pd.DataFrame) -> go.Figure:
    dow_df = df.copy()
    days = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    dow_df['day_name'] = dow_df['date'].dt.day_name()
    dow_agg = dow_df.groupby('day_name', as_index=False)['sales'].mean()
    dow_agg['day_order'] = dow_agg['day_name'].apply(lambda d: days.index(d) if d in days else 0)
    dow_agg = dow_agg.sort_values('day_order')

    fig = px.line(
        dow_agg, x='day_name', y='sales', markers=True,
        title='Average Sales by Day of Week',
        labels={'day_name': 'Day of Week', 'sales': 'Average Daily Sales ($)'},
        template='plotly_white'
    )
    fig.update_traces(line=dict(color='#FFA15A', width=3), marker=dict(size=9))
    fig.update_layout(height=400)
    return fig

def plot_monthly_seasonality_heatmap(df: pd.DataFrame) -> go.Figure:
    h_df = df.copy()
    h_df['year'] = h_df['date'].dt.year
    h_df['month'] = h_df['date'].dt.month
    pivot = h_df.groupby(['year', 'month'])['sales'].sum().unstack(level='month')
    months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']

    fig = px.imshow(
        pivot,
        labels=dict(x="Month", y="Year", color="Sales ($)"),
        x=months[:pivot.shape[1]],
        y=pivot.index.astype(str),
        title="Monthly Seasonality Sales Heatmap (Year vs Month)",
        color_continuous_scale="YlOrRd",
        aspect="auto"
    )
    fig.update_layout(height=420)
    return fig

def plot_correlation_heatmap(df: pd.DataFrame) -> go.Figure:
    numeric_df = df[['sales', 'onpromotion', 'dcoilwtico', 'is_holiday', 'cluster']].copy()
    corr = numeric_df.corr()

    fig = px.imshow(
        corr,
        text_auto=".2f",
        title="Correlation Heatmap (Sales vs Macro & Operational Drivers)",
        color_continuous_scale="RdBu_r",
        zmin=-1, zmax=1,
        aspect="auto"
    )
    fig.update_layout(height=450)
    return fig

if __name__ == '__main__':
    print('Testing analytics module...')
    df = pd.read_parquet('data/processed/cleaned_sales.parquet')
    kpis = get_overall_kpis(df)
    print('Overall KPIs:', kpis)
    fig = plot_top_families(df)
    print('plot_top_families figure successfully generated.')

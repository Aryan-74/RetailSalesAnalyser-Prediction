"""Model evaluation and metrics calculation module."""

from typing import Dict, Any, List, Tuple
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import plotly.express as px
import plotly.graph_objects as go

def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """
    Calculate MAE, RMSE, RMSLE, WAPE, and R2.
    Ensures non-negative predictions for valid RMSLE.
    """
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    y_pred = np.clip(y_pred, 0, None)  # Sales cannot be negative

    mae = float(mean_absolute_error(y_true, y_pred))
    mse = float(mean_squared_error(y_true, y_pred))
    rmse = float(np.sqrt(mse))

    # RMSLE: sqrt(mean((log1p(y_true) - log1p(y_pred))^2))
    log_true = np.log1p(y_true)
    log_pred = np.log1p(y_pred)
    rmsle = float(np.sqrt(mean_squared_error(log_true, log_pred)))

    # WAPE: sum(|y_true - y_pred|) / sum(y_true)
    sum_true = float(np.sum(y_true))
    wape = float(np.sum(np.abs(y_true - y_pred)) / sum_true) if sum_true > 0 else 0.0

    r2 = float(r2_score(y_true, y_pred))

    return {
        'MAE': round(mae, 4),
        'RMSE': round(rmse, 4),
        'RMSLE': round(rmsle, 4),
        'WAPE': round(wape, 4),
        'R2': round(r2, 4)
    }

def plot_actual_vs_predicted(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    dates: pd.Series = None,
    title: str = "Actual vs Predicted Sales"
) -> go.Figure:
    """Line and scatter comparison of actual versus predicted sales."""
    y_pred = np.clip(y_pred, 0, None)

    if dates is not None:
        plot_df = pd.DataFrame({'date': dates, 'actual': y_true, 'predicted': y_pred})
        daily = plot_df.groupby('date', as_index=False).sum()

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=daily['date'], y=daily['actual'],
            mode='lines+markers', name='Actual Sales',
            line=dict(color='#1f77b4', width=2.5)
        ))
        fig.add_trace(go.Scatter(
            x=daily['date'], y=daily['predicted'],
            mode='lines+markers', name='Predicted Sales',
            line=dict(color='#ff7f0e', width=2, dash='dot')
        ))
        fig.update_layout(
            title=title,
            xaxis_title='Date', yaxis_title='Total Daily Sales ($)',
            template='plotly_white', hovermode='x unified',
            height=450
        )
    else:
        sample_size = min(5000, len(y_true))
        idx = np.random.RandomState(42).choice(len(y_true), sample_size, replace=False)
        fig = px.scatter(
            x=y_true[idx], y=y_pred[idx],
            labels={'x': 'Actual Sales', 'y': 'Predicted Sales'},
            title=title, template='plotly_white', opacity=0.5
        )
        max_val = max(float(np.max(y_true[idx])), float(np.max(y_pred[idx])))
        fig.add_trace(go.Scatter(
            x=[0, max_val], y=[0, max_val],
            mode='lines', name='Ideal Fit (y=x)',
            line=dict(color='red', dash='dash')
        ))
        fig.update_layout(height=450)
    return fig

def plot_residuals(y_true: np.ndarray, y_pred: np.ndarray) -> go.Figure:
    """Plot residual distribution histogram and density."""
    y_pred = np.clip(y_pred, 0, None)
    residuals = y_true - y_pred

    sample_size = min(20000, len(residuals))
    idx = np.random.RandomState(42).choice(len(residuals), sample_size, replace=False)
    sample_res = residuals[idx]

    fig = px.histogram(
        sample_res, nbins=80,
        title="Residual Error Distribution (Actual - Predicted)",
        labels={'value': 'Residual Error ($)', 'count': 'Frequency'},
        template='plotly_white', color_discrete_sequence=['#EF553B']
    )
    fig.add_vline(x=0, line_dash="dash", line_color="black")
    fig.update_layout(height=400, showlegend=False)
    return fig

def plot_feature_importance(feature_names: List[str], importances: np.ndarray, top_n: int = 20) -> go.Figure:
    """Bar chart of model feature importances."""
    fi_df = pd.DataFrame({
        'feature': feature_names,
        'importance': importances
    }).sort_values('importance', ascending=False).head(top_n)

    fig = px.bar(
        fi_df.sort_values('importance', ascending=True),
        x='importance', y='feature', orientation='h',
        title=f"Top {top_n} Feature Importances",
        labels={'importance': 'Importance Score', 'feature': 'Feature'},
        template='plotly_white', color='importance',
        color_continuous_scale='Blues'
    )
    fig.update_layout(height=500, coloraxis_showscale=False)
    return fig

def plot_model_comparison(comparison_df: pd.DataFrame, metric: str = 'RMSLE') -> go.Figure:
    """Bar chart comparing all models across splits."""
    fig = px.bar(
        comparison_df, x='Model', y=metric, color='Split',
        barmode='group',
        title=f"Model Comparison - {metric}",
        template='plotly_white',
        color_discrete_sequence=px.colors.qualitative.Set2
    )
    fig.update_layout(height=400)
    return fig

if __name__ == '__main__':
    y_t = np.array([100, 200, 300, 400])
    y_p = np.array([110, 190, 310, 380])
    m = calculate_metrics(y_t, y_p)
    print("Metrics test:", m)

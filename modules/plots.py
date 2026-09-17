import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from arch import arch_model


def plot_price(df: pd.DataFrame, ticker: str) -> go.Figure:
    """
    Line chart of adjusted closing price over time.
    """
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=df.index,
        y=df["Price"],
        mode="lines",
        line=dict(color="#2196F3", width=1.5),
        name="Price",
        hovertemplate="%{x|%Y-%m-%d}<br>Price: %{y:.2f}<extra></extra>",
    ))

    fig.update_layout(
        title=dict(text=f"{ticker} — Adjusted Closing Price", font=dict(size=16)),
        xaxis_title="Date",
        yaxis_title="Price (USD)",
        template="plotly_dark",
        height=350,
        margin=dict(l=40, r=20, t=50, b=40),
        hovermode="x unified",
    )

    return fig


def plot_returns(df: pd.DataFrame, ticker: str) -> go.Figure:
    """
    Line chart of daily log returns over time.
    """
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=df.index,
        y=df["Return"],
        mode="lines",
        line=dict(color="#FF9800", width=1),
        name="Log Return",
        hovertemplate="%{x|%Y-%m-%d}<br>Return: %{y:.4f}<extra></extra>",
    ))

    fig.add_hline(y=0, line_dash="dash", line_color="white", line_width=0.8, opacity=0.4)

    fig.update_layout(
        title=dict(text=f"{ticker} — Daily Log Returns", font=dict(size=16)),
        xaxis_title="Date",
        yaxis_title="Log Return",
        template="plotly_dark",
        height=350,
        margin=dict(l=40, r=20, t=50, b=40),
        hovermode="x unified",
    )

    return fig


def plot_distribution(returns: pd.Series, var_es_df: pd.DataFrame, ticker: str) -> go.Figure:
    """
    Return distribution histogram with VaR and ES lines for all three methods
    at both 95% and 99% confidence levels.
    """
    fig = go.Figure()

    # Histogram
    fig.add_trace(go.Histogram(
        x=returns,
        nbinsx=80,
        name="Returns",
        marker_color="#455A64",
        opacity=0.75,
        hovertemplate="Return: %{x:.4f}<br>Count: %{y}<extra></extra>",
    ))

    # Colour scheme: method × confidence
    colours = {
        ("Historical Simulation", "VaR 95%"): "#64B5F6",
        ("Historical Simulation", "VaR 99%"): "#1565C0",
        ("Parametric Normal",     "VaR 95%"): "#81C784",
        ("Parametric Normal",     "VaR 99%"): "#2E7D32",
        ("GARCH-FHS",             "VaR 95%"): "#FFB74D",
        ("GARCH-FHS",             "VaR 99%"): "#E65100",
    }

    dash_style = {"VaR 95%": "dash", "VaR 99%": "dot"}

    for method in var_es_df.index:
        for col in ["VaR 95%", "VaR 99%"]:
            val   = -var_es_df.loc[method, col]   # negative → left tail
            color = colours.get((method, col), "#FFFFFF")
            label = f"{method} {col}"

            fig.add_vline(
                x=val,
                line_dash=dash_style[col],
                line_color=color,
                line_width=1.8,
                annotation_text=label,
                annotation_position="top right",
                annotation_font_size=9,
                annotation_font_color=color,
            )

    fig.update_layout(
        title=dict(text=f"{ticker} — Return Distribution with VaR Thresholds", font=dict(size=16)),
        xaxis_title="Log Return",
        yaxis_title="Frequency",
        template="plotly_dark",
        height=450,
        margin=dict(l=40, r=20, t=50, b=40),
        showlegend=False,
        bargap=0.05,
    )

    return fig


def plot_garch_volatility(returns: pd.Series, ticker: str) -> go.Figure:
    """
    GARCH(1,1) conditional volatility over time (annualised).
    """
    r_scaled = returns * 100
    model    = arch_model(r_scaled, vol="Garch", p=1, q=1, dist="normal", rescale=False)
    result   = model.fit(disp="off")

    # Conditional volatility in decimal, annualised
    cond_vol = result.conditional_volatility / 100
    ann_vol  = cond_vol * np.sqrt(252)

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=returns.index,
        y=ann_vol,
        mode="lines",
        line=dict(color="#CE93D8", width=1.5),
        name="Annualised Vol",
        hovertemplate="%{x|%Y-%m-%d}<br>Ann. Vol: %{y:.2%}<extra></extra>",
    ))

    fig.update_layout(
        title=dict(text=f"{ticker} — GARCH(1,1) Conditional Volatility (Annualised)", font=dict(size=16)),
        xaxis_title="Date",
        yaxis_title="Annualised Volatility",
        yaxis_tickformat=".0%",
        template="plotly_dark",
        height=350,
        margin=dict(l=40, r=20, t=50, b=40),
        hovermode="x unified",
    )

    return fig
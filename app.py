import os
os.chdir(r"D:\Documents\APPLIED ECONOMETRICS WORK\VaR-ES-Dashboard")

import sys
sys.path.insert(0, r"D:\Documents\APPLIED ECONOMETRICS WORK\VaR-ES-Dashboard")

import streamlit as st
import pandas as pd
import io

from modules.data import load_data, get_summary_stats
from modules.var_es import compute_all, get_garch_params
from modules.backtesting import run_all_backtests
from modules.plots import plot_price, plot_returns, plot_distribution, plot_garch_volatility
from modules.interpretation import generate_interpretation

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="VaR & ES Risk Dashboard",
    page_icon="📊",
    layout="wide",
)

# ── Header ────────────────────────────────────────────────────────────────────
st.title("📊 Value at Risk & Expected Shortfall Dashboard")
st.markdown(
    "Interactive risk analysis using **Historical Simulation**, "
    "**Parametric Normal**, and **GARCH-FHS** methods. "
    "Enter any valid [yfinance](https://finance.yahoo.com/) ticker below."
)
st.divider()

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ Parameters")

    ticker = st.text_input(
        label="Ticker Symbol",
        value="GC=F",
        help="Examples: GC=F (Gold), AAPL, ^GSPC, MSFT, ^KSE",
    ).strip().upper()

    col1, col2 = st.columns(2)
    with col1:
        start_date = st.date_input("Start Date", value=pd.Timestamp("2018-01-01"))
    with col2:
        end_date = st.date_input("End Date", value=pd.Timestamp("2024-01-01"))

    run = st.button("▶ Run Analysis", use_container_width=True, type="primary")

    st.divider()
    st.markdown(
        "**Author:** Ahmer  \n"
        "**GitHub:** [ahmer-econ](https://github.com/ahmer-econ)  \n"
        f"**Date:** {pd.Timestamp.today().strftime('%B %Y')}"
    )

# ── Main panel ────────────────────────────────────────────────────────────────
if not run:
    st.info("Configure parameters in the sidebar and click **▶ Run Analysis** to begin.")
    st.stop()

# Validate dates
if start_date >= end_date:
    st.error("Start date must be before end date.")
    st.stop()

start_str = start_date.strftime("%Y-%m-%d")
end_str   = end_date.strftime("%Y-%m-%d")

# ── Data loading ──────────────────────────────────────────────────────────────
with st.spinner(f"Downloading data for **{ticker}**..."):
    try:
        df = load_data(ticker, start_str, end_str)
    except ValueError as e:
        st.error(str(e))
        st.stop()

if len(df) < 100:
    st.error("Fewer than 100 observations returned. Please widen the date range.")
    st.stop()

# ── Section 1: Data Overview ──────────────────────────────────────────────────
st.header("1 · Data Overview")

col_l, col_r = st.columns([3, 1])

with col_l:
    tab1, tab2 = st.tabs(["📈 Price", "📉 Returns"])
    with tab1:
        st.plotly_chart(plot_price(df, ticker), use_container_width=True)
    with tab2:
        st.plotly_chart(plot_returns(df, ticker), use_container_width=True)

with col_r:
    st.subheader("Summary Statistics")
    stats = get_summary_stats(df)
    st.dataframe(stats, use_container_width=True)

st.divider()

# ── Computation ───────────────────────────────────────────────────────────────
with st.spinner("Running VaR / ES estimation..."):
    var_es_df = compute_all(df["Return"])

with st.spinner("Running backtests..."):
    backtest_df = run_all_backtests(df["Return"], var_es_df)

with st.spinner("Fitting GARCH model..."):
    garch_params = get_garch_params(df["Return"])

# ── Section 2: VaR / ES Results ───────────────────────────────────────────────
st.header("2 · VaR / ES Results")

st.markdown(
    "All values expressed as **proportion of portfolio value** (e.g. 0.017 = 1.7% loss)."
)

# Colour-code pass/fail in backtest later; show plain results table here
st.dataframe(
    var_es_df.style.format("{:.4f}"),
    use_container_width=True,
)

st.divider()

# ── Section 3: Risk Interpretation ───────────────────────────────────────────
st.header("3 · Risk Interpretation")

interpretation = generate_interpretation(
    ticker, var_es_df, backtest_df, garch_params, start_str, end_str
)
st.markdown(interpretation)

st.divider()

# ── Section 4: Visualisations ─────────────────────────────────────────────────
st.header("4 · Visualisations")

col_a, col_b = st.columns(2)

with col_a:
    st.plotly_chart(
        plot_distribution(df["Return"], var_es_df, ticker),
        use_container_width=True,
    )

with col_b:
    st.plotly_chart(
        plot_garch_volatility(df["Return"], ticker),
        use_container_width=True,
    )

st.divider()

# ── Section 5: Backtesting ────────────────────────────────────────────────────
st.header("5 · Backtesting Results")

def colour_result(val):
    if val == "Pass":
        return "color: #81C784; font-weight: bold"
    elif val == "Fail":
        return "color: #E57373; font-weight: bold"
    return ""

styled_bt = backtest_df.style.map(
    colour_result,
    subset=["Kupiec Result", "Christoffersen Result"]
)

st.dataframe(styled_bt, use_container_width=True, hide_index=True)

st.divider()

# ── Download ──────────────────────────────────────────────────────────────────
st.header("6 · Download Results")

@st.cache_data
def build_csv(var_es_df, backtest_df):
    buf = io.StringIO()
    buf.write("VaR and ES Estimates\n")
    var_es_df.to_csv(buf)
    buf.write("\nBacktesting Results\n")
    backtest_df.to_csv(buf, index=False)
    return buf.getvalue()

csv_data = build_csv(var_es_df, backtest_df)

st.download_button(
    label="⬇️ Download Results as CSV",
    data=csv_data,
    file_name=f"VaR_ES_{ticker}_{end_str}.csv",
    mime="text/csv",
    use_container_width=True,
)

st.caption(
    f"Analysis run on {pd.Timestamp.today().strftime('%Y-%m-%d')} · "
    "Data sourced from Yahoo Finance via yfinance · "
    "Author: Ahmer | GitHub: ahmer-econ"
)
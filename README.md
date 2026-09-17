# VaR & ES Risk Dashboard

An interactive financial risk dashboard built with Python and Streamlit.

**Author:** Ahmer | [GitHub: ahmer-econ](https://github.com/ahmer-econ) | September 2026

---

## What It Does

This dashboard takes any valid Yahoo Finance ticker as input and computes
Value at Risk (VaR) and Expected Shortfall (ES) using three methods:

- **Historical Simulation** — non-parametric, uses the empirical return distribution
- **Parametric Normal** — assumes normally distributed returns
- **GARCH-FHS** — Filtered Historical Simulation using a GARCH(1,1) volatility model

Results are validated using two backtests:

- **Kupiec POF Test** — checks whether the violation rate matches the stated confidence level
- **Christoffersen Independence Test** — checks whether violations are clustered in time

---

## Features

- Live data download via yfinance — any ticker, any date range
- VaR and ES at both 95% and 99% confidence simultaneously
- Interactive Plotly charts — price, returns, distribution, GARCH volatility
- Plain English risk interpretation auto-generated from results
- Colour-coded backtesting results
- Downloadable CSV of all outputs

---

## How to Run Locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

---

## Project Structurevar-es-dashboard/
├── app.py # Main Streamlit application
├── requirements.txt
├── README.md
└── modules/
├── data.py # Data download and summary statistics
├── var_es.py # HS, Parametric, GARCH-FHS estimation
├── backtesting.py # Kupiec and Christoffersen tests
├── plots.py # Plotly chart functions
└── interpretation.py # Plain English risk summary generator

---

## Data Source

Yahoo Finance via the `yfinance` library. No API key required.

---

## Methodology Notes

- Returns computed as daily log returns
- GARCH(1,1) fitted with Normal innovations using the `arch` library
- One-step-ahead volatility forecast used for GARCH-FHS scaling
- Backtests applied to the full in-sample return series
import os
os.chdir(r"D:\Documents\APPLIED ECONOMETRICS WORK\VaR-ES-Dashboard")

import pandas as pd


def generate_interpretation(
    ticker: str,
    var_es_df: pd.DataFrame,
    backtest_df: pd.DataFrame,
    garch_params: dict,
    start: str,
    end: str,
) -> str:
    """
    Generate a plain-English risk summary from the computed results.

    Returns a markdown-formatted string for display in Streamlit.
    """

    # Extract key numbers
    hs_var95  = var_es_df.loc["Historical Simulation", "VaR 95%"]
    hs_es95   = var_es_df.loc["Historical Simulation", "ES 95%"]
    hs_var99  = var_es_df.loc["Historical Simulation", "VaR 99%"]
    hs_es99   = var_es_df.loc["Historical Simulation", "ES 99%"]

    pn_var95  = var_es_df.loc["Parametric Normal", "VaR 95%"]
    pn_var99  = var_es_df.loc["Parametric Normal", "VaR 99%"]

    gf_var95  = var_es_df.loc["GARCH-FHS", "VaR 95%"]
    gf_var99  = var_es_df.loc["GARCH-FHS", "VaR 99%"]

    persistence = garch_params["persistence"]
    alpha       = garch_params["alpha"]
    beta        = garch_params["beta"]

    # Backtest summary
    def get_result(method, confidence):
        row = backtest_df[
            (backtest_df["Method"] == method) &
            (backtest_df["Confidence"] == confidence)
        ]
        if row.empty:
            return "N/A", "N/A"
        return row["Kupiec Result"].values[0], row["Christoffersen Result"].values[0]

    hs_k95,  hs_c95  = get_result("Historical Simulation", "95%")
    hs_k99,  hs_c99  = get_result("Historical Simulation", "99%")
    pn_k95,  pn_c95  = get_result("Parametric Normal",     "95%")
    pn_k99,  pn_c99  = get_result("Parametric Normal",     "99%")
    gf_k95,  gf_c95  = get_result("GARCH-FHS",             "95%")
    gf_k99,  gf_c99  = get_result("GARCH-FHS",             "99%")

    def pass_fail_icon(result):
        return "✅" if result == "Pass" else "❌"

    # Volatility regime description
    if persistence >= 0.97:
        vol_desc = "extremely persistent — shocks to volatility decay very slowly"
    elif persistence >= 0.93:
        vol_desc = "highly persistent — volatility tends to cluster and takes time to revert"
    elif persistence >= 0.85:
        vol_desc = "moderately persistent — volatility responds to shocks but reverts at a reasonable pace"
    else:
        vol_desc = "relatively low — volatility reverts quickly after shocks"

    text = f"""
### What Do These Results Mean?

**Asset:** `{ticker}` &nbsp;|&nbsp; **Period:** {start} to {end}

---

#### 📉 Downside Risk at a Glance

On a typical trading day, the three methods agree that **{ticker}** carries a
**95% VaR between {min(hs_var95, pn_var95, gf_var95)*100:.2f}% and {max(hs_var95, pn_var95, gf_var95)*100:.2f}%**.
This means that on 19 out of every 20 trading days, losses are expected to stay within that range.

On the worst 5% of days, the average loss (Expected Shortfall) is estimated at
**{hs_es95*100:.2f}%** by Historical Simulation — this is the average loss you
would experience when things go wrong, not just the threshold.

At the stricter **99% confidence level**, VaR estimates range from
**{min(hs_var99, pn_var99, gf_var99)*100:.2f}% to {max(hs_var99, pn_var99, gf_var99)*100:.2f}%**,
with an average tail loss of **{hs_es99*100:.2f}%**.

---

#### ⚙️ Volatility Dynamics (GARCH)

The GARCH(1,1) model estimates a volatility persistence of **{persistence:.4f}**
(α = {alpha:.4f}, β = {beta:.4f}). This means volatility for `{ticker}` is {vol_desc}.
Investors should expect periods of elevated risk to persist rather than dissipate quickly.

---

#### 🧪 Backtesting Summary

| Method | Kupiec 95% | Christoffersen 95% | Kupiec 99% | Christoffersen 99% |
|---|---|---|---|---|
| Historical Simulation | {pass_fail_icon(hs_k95)} {hs_k95} | {pass_fail_icon(hs_c95)} {hs_c95} | {pass_fail_icon(hs_k99)} {hs_k99} | {pass_fail_icon(hs_c99)} {hs_c99} |
| Parametric Normal | {pass_fail_icon(pn_k95)} {pn_k95} | {pass_fail_icon(pn_c95)} {pn_c95} | {pass_fail_icon(pn_k99)} {pn_k99} | {pass_fail_icon(pn_c99)} {pn_c99} |
| GARCH-FHS | {pass_fail_icon(gf_k95)} {gf_k95} | {pass_fail_icon(gf_c95)} {gf_c95} | {pass_fail_icon(gf_k99)} {gf_k99} | {pass_fail_icon(gf_c99)} {gf_c99} |

A **Pass** on Kupiec means the model produced violations at a rate statistically
consistent with its stated confidence level. A **Pass** on Christoffersen means
those violations were not clustered in time — a critical property for a reliable risk model.
"""

    return text.strip()
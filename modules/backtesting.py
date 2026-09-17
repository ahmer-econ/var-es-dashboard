import os
os.chdir(r"D:\Documents\APPLIED ECONOMETRICS WORK\VaR-ES-Dashboard")

import numpy as np
import pandas as pd
from scipy.stats import chi2


def kupiec_pof(returns: pd.Series, var: float, confidence: float) -> dict:
    """
    Kupiec Proportion of Failures (POF) test.

    Tests whether the observed violation rate matches the expected rate.

    H0: violation rate = (1 - confidence)
    Reject H0 if p-value < 0.05 → VaR model is mis-specified.

    Parameters
    ----------
    returns    : pd.Series of out-of-sample log returns
    var        : VaR estimate (positive number representing a loss)
    confidence : float, 0.95 or 0.99

    Returns
    -------
    dict with violations, expected_violations, lr_stat, p_value, result
    """
    alpha = 1 - confidence
    n     = len(returns)

    # A violation occurs when the actual loss exceeds VaR
    violations = int((returns < -var).sum())
    v_rate     = violations / n

    # Avoid log(0)
    if violations == 0 or violations == n:
        return {
            "Violations"         : violations,
            "Expected Violations": round(alpha * n, 2),
            "Violation Rate"     : round(v_rate, 4),
            "LR Statistic"       : np.nan,
            "p-value"            : np.nan,
            "Result"             : "Insufficient violations to test",
        }

    lr = -2 * (
        np.log(alpha ** violations * (1 - alpha) ** (n - violations))
        - np.log(v_rate ** violations * (1 - v_rate) ** (n - violations))
    )

    p_value = 1 - chi2.cdf(lr, df=1)

    return {
        "Violations"         : violations,
        "Expected Violations": round(alpha * n, 2),
        "Violation Rate"     : round(v_rate, 4),
        "LR Statistic"       : round(lr, 4),
        "p-value"            : round(p_value, 4),
        "Result"             : "Pass" if p_value > 0.05 else "Fail",
    }


def christoffersen_ind(returns: pd.Series, var: float) -> dict:
    """
    Christoffersen Independence test.

    Tests whether violations are independent over time (no clustering).

    H0: violations are independent
    Reject H0 if p-value < 0.05 → violations are clustered.

    Returns
    -------
    dict with lr_stat, p_value, result
    """
    hit = (returns < -var).astype(int).values
    n   = len(hit)

    # Transition counts
    n00 = n01 = n10 = n11 = 0
    for i in range(1, n):
        prev, curr = hit[i - 1], hit[i]
        if prev == 0 and curr == 0:
            n00 += 1
        elif prev == 0 and curr == 1:
            n01 += 1
        elif prev == 1 and curr == 0:
            n10 += 1
        elif prev == 1 and curr == 1:
            n11 += 1

    # Avoid division by zero
    if (n00 + n01) == 0 or (n10 + n11) == 0:
        return {
            "LR Statistic": np.nan,
            "p-value"     : np.nan,
            "Result"      : "Insufficient data",
        }

    p01 = n01 / (n00 + n01)
    p11 = n11 / (n10 + n11)
    p   = (n01 + n11) / (n00 + n01 + n10 + n11)

    if p == 0 or p == 1 or p01 == 0 or p11 == 0:
        return {
            "LR Statistic": np.nan,
            "p-value"     : np.nan,
            "Result"      : "Insufficient data",
        }

    try:
        lr = -2 * (
            np.log((1 - p) ** (n00 + n10) * p ** (n01 + n11))
            - np.log(
                (1 - p01) ** n00 * p01 ** n01 *
                (1 - p11) ** n10 * p11 ** n11
            )
        )
    except (ValueError, ZeroDivisionError):
        return {
            "LR Statistic": np.nan,
            "p-value"     : np.nan,
            "Result"      : "Insufficient data",
        }

    p_value = 1 - chi2.cdf(lr, df=1)

    return {
        "LR Statistic": round(lr, 4),
        "p-value"     : round(p_value, 4),
        "Result"      : "Pass" if p_value > 0.05 else "Fail",
    }


def run_all_backtests(returns: pd.Series, var_es_df: pd.DataFrame) -> pd.DataFrame:
    """
    Run Kupiec and Christoffersen tests for all three methods at 95% and 99%.

    Parameters
    ----------
    returns    : pd.Series of log returns (same series used for estimation)
    var_es_df  : DataFrame from compute_all() in var_es.py

    Returns
    -------
    pd.DataFrame with one row per method-confidence combination
    """
    methods = {
        "Historical Simulation": ("VaR 95%", "VaR 99%"),
        "Parametric Normal"    : ("VaR 95%", "VaR 99%"),
        "GARCH-FHS"            : ("VaR 95%", "VaR 99%"),
    }

    rows = []

    for method in var_es_df.index:
        for conf, col in [(0.95, "VaR 95%"), (0.99, "VaR 99%")]:
            var = var_es_df.loc[method, col]

            kup  = kupiec_pof(returns, var, conf)
            chri = christoffersen_ind(returns, var)

            rows.append({
                "Method"             : method,
                "Confidence"         : f"{int(conf*100)}%",
                "Violations"         : kup["Violations"],
                "Expected"           : kup["Expected Violations"],
                "Kupiec p-value"     : kup["p-value"],
                "Kupiec Result"      : kup["Result"],
                "Christoffersen p-value" : chri["p-value"],
                "Christoffersen Result"  : chri["Result"],
            })

    return pd.DataFrame(rows)
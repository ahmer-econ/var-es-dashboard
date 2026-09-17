import numpy as np
import pandas as pd
from scipy.stats import norm
from arch import arch_model


def historical_simulation(returns: pd.Series, confidence: float = 0.95) -> dict:
    """
    Historical Simulation VaR and ES.

    Parameters
    ----------
    returns    : pd.Series of log returns (full in-sample series)
    confidence : float, e.g. 0.95 or 0.99

    Returns
    -------
    dict with keys 'VaR' and 'ES' (both positive numbers representing losses)
    """
    sorted_returns = np.sort(returns)
    index = int((1 - confidence) * len(sorted_returns))
    var = -sorted_returns[index]
    es  = -sorted_returns[:index].mean()

    return {"VaR": round(var, 6), "ES": round(es, 6)}


def parametric_normal(returns: pd.Series, confidence: float = 0.95) -> dict:
    """
    Parametric Normal VaR and ES.
    Assumes returns are normally distributed.

    Returns
    -------
    dict with keys 'VaR' and 'ES'
    """
    mu    = returns.mean()
    sigma = returns.std()

    z   = norm.ppf(1 - confidence)
    var = -(mu + z * sigma)
    es  = -(mu - sigma * norm.pdf(z) / (1 - confidence))

    return {"VaR": round(var, 6), "ES": round(es, 6)}


def garch_fhs(returns: pd.Series, confidence: float = 0.95,
              n_simulations: int = 10000) -> dict:
    """
    GARCH Filtered Historical Simulation VaR and ES.

    Steps:
    1. Fit GARCH(1,1) with Normal innovations to the full return series
    2. Extract standardised residuals
    3. Forecast next-period volatility (one-step-ahead)
    4. Scale standardised residuals by forecast volatility
    5. Compute VaR and ES from the empirical distribution of scaled residuals

    Returns
    -------
    dict with keys 'VaR', 'ES', 'omega', 'alpha', 'beta', 'persistence'
    """
    # Scale returns to percentage for arch library stability
    r_scaled = returns * 100

    model  = arch_model(r_scaled, vol="Garch", p=1, q=1, dist="normal", rescale=False)
    result = model.fit(disp="off")

    omega  = result.params["omega"]
    alpha  = result.params["alpha[1]"]
    beta   = result.params["beta[1]"]
    persistence = alpha + beta

    # Standardised residuals
    std_resid = result.resid / result.conditional_volatility

    # One-step-ahead variance forecast
    last_var    = result.conditional_volatility.iloc[-1] ** 2
    last_resid2 = result.resid.iloc[-1] ** 2
    forecast_var = omega + alpha * last_resid2 + beta * last_var
    forecast_vol = np.sqrt(forecast_var)

    # Scaled residuals → simulated return distribution (still in % scale)
    scaled_resid = std_resid * forecast_vol

    # Convert back to decimal
    scaled_resid_dec = scaled_resid / 100

    sorted_scaled = np.sort(scaled_resid_dec)
    index = int((1 - confidence) * len(sorted_scaled))
    var = -sorted_scaled[index]
    es  = -sorted_scaled[:index].mean()

    return {
        "VaR"        : round(var, 6),
        "ES"         : round(es, 6),
        "omega"      : round(omega, 6),
        "alpha"      : round(alpha, 6),
        "beta"       : round(beta, 6),
        "persistence": round(persistence, 6),
    }


def compute_all(returns: pd.Series) -> pd.DataFrame:
    """
    Run all three methods at both 95% and 99% confidence.

    Returns
    -------
    pd.DataFrame with rows = methods, columns = VaR_95, ES_95, VaR_99, ES_99
    """
    results = {}

    for conf, label in [(0.95, "95%"), (0.99, "99%")]:
        hs   = historical_simulation(returns, conf)
        pn   = parametric_normal(returns, conf)
        gfhs = garch_fhs(returns, conf)

        results[label] = {
            "HS_VaR"  : hs["VaR"],
            "HS_ES"   : hs["ES"],
            "PN_VaR"  : pn["VaR"],
            "PN_ES"   : pn["ES"],
            "GFHS_VaR": gfhs["VaR"],
            "GFHS_ES" : gfhs["ES"],
        }

    # Reshape into a clean display table
    rows = []
    for method, key_var, key_es in [
        ("Historical Simulation", "HS_VaR",   "HS_ES"),
        ("Parametric Normal",     "PN_VaR",   "PN_ES"),
        ("GARCH-FHS",             "GFHS_VaR", "GFHS_ES"),
    ]:
        rows.append({
            "Method" : method,
            "VaR 95%": results["95%"][key_var],
            "ES 95%" : results["95%"][key_es],
            "VaR 99%": results["99%"][key_var],
            "ES 99%" : results["99%"][key_es],
        })

    return pd.DataFrame(rows).set_index("Method")


def get_garch_params(returns: pd.Series) -> dict:
    """
    Return GARCH(1,1) parameters separately for display in the dashboard.
    """
    res_95 = garch_fhs(returns, confidence=0.95)
    return {
        "omega"      : res_95["omega"],
        "alpha"      : res_95["alpha"],
        "beta"       : res_95["beta"],
        "persistence": res_95["persistence"],
    }
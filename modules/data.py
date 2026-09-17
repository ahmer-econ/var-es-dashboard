import yfinance as yf
import pandas as pd
import numpy as np

def load_data(ticker: str, start: str, end: str) -> pd.DataFrame:
    """
    Download adjusted closing prices from yfinance and compute log returns.

    Parameters
    ----------
    ticker : str
        Valid yfinance ticker symbol e.g. 'GC=F', 'AAPL', '^GSPC'
    start : str
        Start date in 'YYYY-MM-DD' format
    end : str
        End date in 'YYYY-MM-DD' format

    Returns
    -------
    pd.DataFrame with columns:
        'Price'  — adjusted closing price
        'Return' — daily log return (first row will be NaN, dropped)
    """

    raw = yf.download(ticker, start=start, end=end, auto_adjust=True, progress=False)

    if raw.empty:
        raise ValueError(f"No data returned for ticker '{ticker}'. "
                         f"Check the symbol and date range.")

    # Keep closing price only
    price = raw[["Close"]].copy()
    price.columns = ["Price"]

    # Drop any rows where price is missing
    price.dropna(inplace=True)

    # Log returns
    price["Return"] = np.log(price["Price"] / price["Price"].shift(1))

    # Drop the first NaN return row
    price.dropna(inplace=True)

    return price


def get_summary_stats(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute summary statistics for the Return series.

    Returns a single-column DataFrame suitable for display in Streamlit.
    """
    from scipy.stats import skew, kurtosis

    r = df["Return"]

    stats = {
        "Observations"   : len(r),
        "Mean (daily)"   : round(r.mean(), 6),
        "Std Dev (daily)" : round(r.std(), 6),
        "Min"            : round(r.min(), 6),
        "Max"            : round(r.max(), 6),
        "Skewness"       : round(skew(r), 4),
        "Excess Kurtosis": round(kurtosis(r), 4),
    }

    summary_df = pd.DataFrame.from_dict(stats, orient="index", columns=["Value"])
    return summary_df

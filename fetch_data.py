"""
Data fetch layer for the H&S Pattern Scanner — uses yfinance, same as MaverickPICKS.
Kept separate from hs_detector.py so the detection logic stays data-source agnostic.
"""

import pandas as pd
import yfinance as yf
import time


def fetch_ohlcv(symbol: str, lookback_days: int = 365, retries: int = 2) -> pd.DataFrame:
    """
    Fetches daily OHLCV data for an NSE symbol via yfinance.
    `symbol` should be the bare NSE symbol (e.g. 'RELIANCE') — the .NS suffix is added here.

    Returns a DataFrame with columns: Date, Open, High, Low, Close, Volume
    sorted ascending by Date. Returns an empty DataFrame on failure.
    """
    ticker = f"{symbol}.NS"
    period_str = f"{lookback_days}d" if lookback_days <= 365 else f"{lookback_days // 365 + 1}y"

    for attempt in range(retries + 1):
        try:
            df = yf.Ticker(ticker).history(period=period_str, interval="1d", auto_adjust=True)
            if df is None or df.empty:
                return pd.DataFrame()

            df = df.reset_index()
            df = df.rename(columns={
                "Date": "Date", "Open": "Open", "High": "High",
                "Low": "Low", "Close": "Close", "Volume": "Volume"
            })
            df = df[["Date", "Open", "High", "Low", "Close", "Volume"]].copy()
            df["Date"] = pd.to_datetime(df["Date"]).dt.tz_localize(None)
            df = df.sort_values("Date").reset_index(drop=True)

            # basic sanity check — reject rows with zero/NaN volume or price (corp action glitches, etc.)
            df = df[(df["Volume"] > 0) & (df["Close"] > 0)].reset_index(drop=True)
            return df

        except Exception as e:
            if attempt < retries:
                time.sleep(1.5)
                continue
            print(f"  [WARN] Failed to fetch {symbol} after {retries + 1} attempts: {e}")
            return pd.DataFrame()

    return pd.DataFrame()


def load_symbol_list(filepath: str) -> list:
    """
    Loads a list of NSE symbols from a CSV file. Expects a column named 'Symbol'
    (case-insensitive). Reuse your existing MaverickPICKS NIFTY500 list file here.
    """
    df = pd.read_csv(filepath)
    col = next((c for c in df.columns if c.strip().lower() == "symbol"), None)
    if col is None:
        raise ValueError(
            f"No 'Symbol' column found in {filepath}. Columns present: {list(df.columns)}"
        )
    symbols = df[col].dropna().astype(str).str.strip().str.upper().tolist()
    return symbols

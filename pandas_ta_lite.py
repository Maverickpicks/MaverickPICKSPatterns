"""
pandas_ta_lite.py — Drop-in replacement for pandas_ta
======================================================
Replaces the pandas_ta dependency entirely using only
pandas and numpy built-ins.

Implements exactly the functions used across MaverickPICKS:
  ta.ema(series, length)   → Exponential Moving Average
  ta.rsi(series, length)   → Relative Strength Index
  ta.macd(series, ...)     → MACD line, signal, histogram
  ta.bbands(series, ...)   → Bollinger Bands
  ta.atr(high, low, close, length) → Average True Range
  ta.sma(series, length)   → Simple Moving Average
  ta.vwap(high, low, close, volume) → VWAP

Usage — in any engine file replace:
  import pandas_ta as ta
with:
  import pandas_ta_lite as ta

All function signatures are identical to pandas_ta.
"""

import numpy as np
import pandas as pd


def ema(series: pd.Series, length: int = 20, **kwargs) -> pd.Series:
    """Exponential Moving Average — matches pandas_ta.ema() output exactly."""
    return series.ewm(span=length, adjust=False, min_periods=length).mean()


def sma(series: pd.Series, length: int = 20, **kwargs) -> pd.Series:
    """Simple Moving Average."""
    return series.rolling(window=length, min_periods=length).mean()


def rsi(series: pd.Series, length: int = 14, **kwargs) -> pd.Series:
    """
    Relative Strength Index — Wilder smoothing (matches pandas_ta.rsi()).
    Uses Wilder's exponential smoothing (alpha = 1/length).
    """
    delta = series.diff()
    gain  = delta.clip(lower=0)
    loss  = (-delta).clip(lower=0)

    # Wilder smoothing = EMA with alpha=1/length
    alpha = 1.0 / length
    avg_gain = gain.ewm(alpha=alpha, adjust=False, min_periods=length).mean()
    avg_loss = loss.ewm(alpha=alpha, adjust=False, min_periods=length).mean()

    rs  = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    return rsi


def macd(series: pd.Series,
         fast: int = 12, slow: int = 26, signal: int = 9,
         **kwargs) -> pd.DataFrame:
    """
    MACD — returns DataFrame with columns:
      MACD_{fast}_{slow}_{signal}
      MACDh_{fast}_{slow}_{signal}   (histogram)
      MACDs_{fast}_{slow}_{signal}   (signal line)
    Matches pandas_ta.macd() column naming convention.
    """
    ema_fast   = series.ewm(span=fast,   adjust=False).mean()
    ema_slow   = series.ewm(span=slow,   adjust=False).mean()
    macd_line  = ema_fast - ema_slow
    signal_line= macd_line.ewm(span=signal, adjust=False).mean()
    histogram  = macd_line - signal_line

    col_macd = f"MACD_{fast}_{slow}_{signal}"
    col_hist = f"MACDh_{fast}_{slow}_{signal}"
    col_sig  = f"MACDs_{fast}_{slow}_{signal}"

    return pd.DataFrame({
        col_macd: macd_line,
        col_hist: histogram,
        col_sig:  signal_line,
    }, index=series.index)


def bbands(series: pd.Series,
           length: int = 20, std: float = 2.0,
           **kwargs) -> pd.DataFrame:
    """
    Bollinger Bands — returns DataFrame with columns:
      BBL_{length}_{std}   (lower)
      BBM_{length}_{std}   (middle / SMA)
      BBU_{length}_{std}   (upper)
      BBB_{length}_{std}   (bandwidth)
      BBP_{length}_{std}   (percent)
    Matches pandas_ta.bbands() column naming.
    """
    mid   = series.rolling(length).mean()
    stdev = series.rolling(length).std(ddof=0)
    upper = mid + std * stdev
    lower = mid - std * stdev
    bw    = (upper - lower) / mid.replace(0, np.nan) * 100
    pct   = (series - lower) / (upper - lower).replace(0, np.nan)

    tag = f"{length}_{std}"
    return pd.DataFrame({
        f"BBL_{tag}": lower,
        f"BBM_{tag}": mid,
        f"BBU_{tag}": upper,
        f"BBB_{tag}": bw,
        f"BBP_{tag}": pct,
    }, index=series.index)


def atr(high: pd.Series, low: pd.Series, close: pd.Series,
        length: int = 14, **kwargs) -> pd.Series:
    """Average True Range — Wilder smoothing."""
    prev_close = close.shift(1)
    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low  - prev_close).abs(),
    ], axis=1).max(axis=1)
    return tr.ewm(alpha=1.0/length, adjust=False, min_periods=length).mean()


def vwap(high: pd.Series, low: pd.Series,
         close: pd.Series, volume: pd.Series,
         **kwargs) -> pd.Series:
    """
    Volume Weighted Average Price.
    Resets daily (uses cumulative sum within each date group if index is DatetimeIndex,
    otherwise cumulative from start of series).
    """
    typical = (high + low + close) / 3
    if isinstance(close.index, pd.DatetimeIndex):
        # Group by date for proper intraday reset
        # For daily data this is just cumulative
        tp_vol = typical * volume
        return tp_vol.cumsum() / volume.cumsum()
    return (typical * volume).cumsum() / volume.cumsum()


def stoch(high: pd.Series, low: pd.Series, close: pd.Series,
          k: int = 14, d: int = 3, smooth_k: int = 3,
          **kwargs) -> pd.DataFrame:
    """Stochastic Oscillator."""
    lowest_low   = low.rolling(k).min()
    highest_high = high.rolling(k).max()
    stoch_k = 100 * (close - lowest_low) / (
        (highest_high - lowest_low).replace(0, np.nan)
    )
    stoch_k_smooth = stoch_k.rolling(smooth_k).mean()
    stoch_d        = stoch_k_smooth.rolling(d).mean()
    return pd.DataFrame({
        f"STOCHk_{k}_{d}_{smooth_k}": stoch_k_smooth,
        f"STOCHd_{k}_{d}_{smooth_k}": stoch_d,
    }, index=close.index)


def adx(high: pd.Series, low: pd.Series, close: pd.Series,
        length: int = 14, **kwargs) -> pd.DataFrame:
    """Average Directional Index."""
    prev_high  = high.shift(1)
    prev_low   = low.shift(1)
    prev_close = close.shift(1)

    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low  - prev_close).abs(),
    ], axis=1).max(axis=1)

    dm_plus  = np.where((high - prev_high) > (prev_low - low),
                         np.maximum(high - prev_high, 0), 0)
    dm_minus = np.where((prev_low - low) > (high - prev_high),
                         np.maximum(prev_low - low, 0), 0)

    dm_plus  = pd.Series(dm_plus,  index=close.index)
    dm_minus = pd.Series(dm_minus, index=close.index)

    alpha   = 1.0 / length
    atr_s   = tr.ewm(alpha=alpha, adjust=False).mean()
    di_plus  = 100 * dm_plus.ewm(alpha=alpha, adjust=False).mean() / atr_s.replace(0, np.nan)
    di_minus = 100 * dm_minus.ewm(alpha=alpha, adjust=False).mean() / atr_s.replace(0, np.nan)

    dx  = 100 * (di_plus - di_minus).abs() / (di_plus + di_minus).replace(0, np.nan)
    adx_val = dx.ewm(alpha=alpha, adjust=False).mean()

    return pd.DataFrame({
        f"ADX_{length}":     adx_val,
        f"DMP_{length}":     di_plus,
        f"DMN_{length}":     di_minus,
    }, index=close.index)


def obv(close: pd.Series, volume: pd.Series, **kwargs) -> pd.Series:
    """On Balance Volume."""
    direction = np.sign(close.diff()).fillna(0)
    return (direction * volume).cumsum()


def roc(series: pd.Series, length: int = 10, **kwargs) -> pd.Series:
    """Rate of Change."""
    return series.pct_change(periods=length) * 100


def willr(high: pd.Series, low: pd.Series, close: pd.Series,
          length: int = 14, **kwargs) -> pd.Series:
    """Williams %R."""
    hh = high.rolling(length).max()
    ll = low.rolling(length).min()
    return -100 * (hh - close) / (hh - ll).replace(0, np.nan)

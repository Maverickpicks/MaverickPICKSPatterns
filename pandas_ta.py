"""
pandas_ta.py — drop-in shim for the pandas_ta package.
This file lives in the repo root so Python finds it before
looking in site-packages. No pip install needed.
All engine files that do `import pandas_ta as ta` will use
this automatically — zero changes to existing code required.
"""
from pandas_ta_lite import (
    ema, sma, rsi, macd, bbands, atr, vwap,
    stoch, adx, obv, roc, willr,
)

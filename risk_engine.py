import pandas as pd
import pandas_ta as ta
from utils import last_value, safe_divide


def default_risk():
    return {
        "Entry":        None,
        "ATR":          None,
        "Swing_Low":    None,
        "Stop_Loss":    None,
        "Risk_Percent": None,
        "Target_1":     None,
        "Target_2":     None,
        "Reward_Risk":  None,
        "Trade_Quality":"AVOID"
    }


def risk_analysis(df):

    try:

        if df is None or df.empty or len(df) < 50:
            return default_risk()


        # =====================================
        # ATR (14-period)
        # =====================================

        df["ATR"] = ta.atr(
            high=df["High"], low=df["Low"], close=df["Close"], length=14
        )

        atr   = last_value(df["ATR"])
        entry = last_value(df["Close"])

        if atr is None or entry is None:
            return default_risk()


        # =====================================
        # STOP LOSS
        # Use the TIGHTER of swing low or ATR stop
        # (max gives the higher price = closer to entry = tighter stop)
        # This prevents stops from being absurdly wide on trending stocks
        # =====================================

        swing_low = float(df["Low"].tail(15).min())   # 15-day swing low
        atr_stop  = entry - (1.5 * atr)               # 1.5x ATR below entry

        # Take the HIGHER value = tighter stop = better R:R
        stop_loss = max(swing_low, atr_stop)

        # Safety: stop cannot be above entry
        if stop_loss >= entry:
            stop_loss = entry - atr


        # =====================================
        # RISK %
        # =====================================

        risk_dollar  = entry - stop_loss
        risk_percent = (risk_dollar / entry) * 100


        # =====================================
        # TARGETS (based on ATR multiples)
        # =====================================

        target_1 = entry + (2.0 * atr)
        target_2 = entry + (3.5 * atr)


        # =====================================
        # REWARD:RISK
        # =====================================

        reward_risk = safe_divide(target_1 - entry, risk_dollar)


        # =====================================
        # TRADE QUALITY
        # =====================================

        quality = "AVOID"

        if (
            risk_percent < 3
            and reward_risk is not None
            and reward_risk >= 2.0
        ):
            quality = "EXCELLENT"

        elif (
            risk_percent < 5
            and reward_risk is not None
            and reward_risk >= 1.5
        ):
            quality = "GOOD"

        elif (
            risk_percent < 8
            and reward_risk is not None
            and reward_risk >= 1.0
        ):
            quality = "FAIR"


        return {
            "Entry":        round(float(entry),       2),
            "ATR":          round(float(atr),         2),
            "Swing_Low":    round(float(swing_low),   2),
            "Stop_Loss":    round(float(stop_loss),   2),
            "Risk_Percent": round(float(risk_percent),2),
            "Target_1":     round(float(target_1),    2),
            "Target_2":     round(float(target_2),    2),
            "Reward_Risk":  round(float(reward_risk), 2) if reward_risk is not None else None,
            "Trade_Quality": quality
        }

    except Exception as e:
        print("Risk Engine Error:", e)
        return default_risk()

import pandas as pd
import numpy as np
from utils import last_value, safe_divide


# ============================================================
# VOLUME ENGINE — HOW A SWING TRADER READS VOLUME
#
# Question 1: Is today's volume meaningful?
#   Today vs 20-day avg. >1.5x on bull candle = meaningful.
#
# Question 2: Are institutions accumulating over weeks?
#   Up-day volume vs down-day volume over 20 sessions.
#   If more volume flows in on up days → smart money buying.
#
# Question 3: Is participation GROWING over months?
#   Current month volume vs 12-month average.
#   Rising monthly volume = growing institutional interest.
#   This is the long-term institutional participation signal.
#
# Question 4: Is the pullback healthy?
#   Price falling on BELOW average volume = no panic.
#   Sellers are not aggressive. Institutions holding.
#   This is a POSITIVE signal, not negative.
#
# Question 5: Is there distribution?
#   Heavy volume on DOWN days consistently = smart money exiting.
#   Hard avoid signal.
# ============================================================


def default_volume():
    return {
        "Volume_Ratio":         None,
        "Weekly_Vol_Ratio":     None,
        "Monthly_Vol_Ratio":    None,
        "Monthly_12M_Ratio":    None,
        "Bull_Candle":          False,
        "Bear_Candle":          False,
        "Strong_Bull":          False,
        "Strong_Bear":          False,
        "Close_Near_High":      False,
        "Breakout_Volume":      False,
        "Weekly_Accumulation":  False,
        "Monthly_Accumulation": False,
        "Monthly_Trend":        "FLAT",
        "Accum_Distribution":   None,
        "Accum_Day_Count":      0,
        "Dry_Pullback":         False,
        "Volume_State":         "DRY",
        "Volume_Score":         0,
        "Volume_Warning":       ""
    }


def volume_analysis(daily_df, weekly_df, monthly_df):

    try:

        if daily_df is None or daily_df.empty or len(daily_df) < 60:
            return default_volume()

        if weekly_df is None or weekly_df.empty:
            return default_volume()

        if monthly_df is None or monthly_df.empty:
            return default_volume()


        # ============================================================
        # TODAY'S CANDLE
        # ============================================================

        open_  = float(last_value(daily_df["Open"]))
        high   = float(last_value(daily_df["High"]))
        low    = float(last_value(daily_df["Low"]))
        close  = float(last_value(daily_df["Close"]))
        vol    = float(last_value(daily_df["Volume"]))

        bull = close > open_
        bear = close < open_

        candle_range = high - low
        position     = (close - low) / candle_range if candle_range > 0 else 0.5

        close_near_high = position > 0.75
        strong_bull     = bull and position > 0.75
        strong_bear     = bear and position < 0.25


        # ============================================================
        # DAILY VOLUME RATIO (today vs 20-day avg)
        # ============================================================

        avg_vol_20   = float(daily_df["Volume"].tail(20).mean())
        volume_ratio = safe_divide(vol, avg_vol_20)

        breakout_volume = (
            volume_ratio is not None
            and float(volume_ratio) >= 1.5
            and bull
        )


        # ============================================================
        # ACCUMULATION / DISTRIBUTION SCORE
        # Compare avg volume on up days vs down days (last 20 sessions)
        # ============================================================

        recent = daily_df.tail(20).copy()

        up_days   = recent[recent["Close"] > recent["Open"]]
        down_days = recent[recent["Close"] < recent["Open"]]

        avg_up_vol   = float(up_days["Volume"].mean())   if len(up_days)   > 0 else 0.0
        avg_down_vol = float(down_days["Volume"].mean()) if len(down_days) > 0 else 0.0

        accum_distribution = safe_divide(avg_up_vol, avg_down_vol)

        # Count strong up days (bull candle + above avg volume)
        strong_up_days = up_days[up_days["Volume"] > avg_vol_20]
        accum_day_count = len(strong_up_days)


        # ============================================================
        # WEEKLY VOLUME TREND
        # Recent 4 weeks vs prior 8 weeks
        # ============================================================

        avg_4w      = float(weekly_df["Volume"].tail(4).mean())
        avg_8w      = float(weekly_df["Volume"].tail(12).mean())
        weekly_ratio = safe_divide(avg_4w, avg_8w)

        weekly_accum = (
            weekly_ratio is not None
            and float(weekly_ratio) > 1.05
        )


        # ============================================================
        # 12-MONTH VOLUME TREND (what you asked for specifically)
        # Compare current month's volume to 12-month average
        # This is the cleanest institutional participation signal
        # Rising = growing interest. Falling = fading interest.
        # ============================================================

        monthly_12m_ratio = None
        monthly_trend     = "FLAT"
        monthly_accum     = False

        if len(monthly_df) >= 13:
            # Current month (most recent)
            current_month_vol = float(monthly_df["Volume"].iloc[-1])

            # 12-month average (excluding current month)
            avg_12m_vol = float(monthly_df["Volume"].iloc[-13:-1].mean())

            monthly_12m_ratio = safe_divide(current_month_vol, avg_12m_vol)

            if monthly_12m_ratio is not None:
                r = float(monthly_12m_ratio)
                if r >= 1.3:
                    monthly_trend = "SURGING"    # Big institutional move
                elif r >= 1.1:
                    monthly_trend = "RISING"     # Growing participation
                elif r >= 0.9:
                    monthly_trend = "FLAT"
                elif r >= 0.7:
                    monthly_trend = "FADING"     # Losing interest
                else:
                    monthly_trend = "DEAD"       # Abandoned

                monthly_accum = r >= 1.1


        # ============================================================
        # DRY PULLBACK DETECTION
        # Price declining on below-average volume = healthy correction
        # Institutions are holding, not selling
        # ============================================================

        recent_5d_vol = float(daily_df["Volume"].tail(5).mean())
        price_5d_ago  = float(daily_df["Close"].iloc[-5])
        price_falling = close <= price_5d_ago

        dry_pullback = (
            price_falling
            and recent_5d_vol < avg_vol_20 * 0.82
        )


        # ============================================================
        # MONTHLY RATIO (traditional 3M vs 6M)
        # ============================================================

        avg_3m       = float(monthly_df["Volume"].tail(3).mean())
        avg_6m       = float(monthly_df["Volume"].tail(6).mean())
        monthly_ratio = safe_divide(avg_3m, avg_6m)


        # ============================================================
        # VOLUME STATE — priority order
        # ============================================================

        state   = "DRY"
        score   = 0
        warning = ""

        # DISTRIBUTION — hard avoid
        if (
            volume_ratio is not None
            and float(volume_ratio) >= 1.5
            and strong_bear
            and accum_distribution is not None
            and float(accum_distribution) < 0.75
        ):
            state   = "DISTRIBUTION"
            score   = 0
            warning = "Institutional Selling — Avoid"

        # BREAKOUT — high volume bull day + weekly accumulation + monthly rising
        elif (
            volume_ratio is not None
            and float(volume_ratio) >= 1.5
            and strong_bull
            and close_near_high
            and (weekly_accum or monthly_accum)
        ):
            state   = "BREAKOUT"
            score   = 15
            warning = ""

        # ACCUMULATION — institutions buying quietly over 20 sessions
        elif (
            accum_distribution is not None
            and float(accum_distribution) >= 1.2
            and accum_day_count >= 3
            and (weekly_accum or monthly_trend in ["RISING", "SURGING"])
        ):
            state   = "ACCUMULATION"
            score   = 10
            warning = ""

        # DRY PULLBACK — healthy, price falling on low volume
        elif dry_pullback:
            state   = "DRY_PULLBACK"
            score   = 6
            warning = ""

        # NORMAL — average participation
        elif (
            volume_ratio is not None
            and float(volume_ratio) >= 0.8
        ):
            state   = "NORMAL"
            score   = 3
            warning = ""

        # DRY — below average, no signal
        else:
            state   = "DRY"
            score   = 0
            warning = "Low Participation"

        # Bonus if monthly volume is surging (institutional interest growing)
        if monthly_trend == "SURGING" and state in ["ACCUMULATION", "BREAKOUT"]:
            score = min(score + 3, 15)


        return {
            "Volume_Ratio":         round(float(volume_ratio), 2)         if volume_ratio is not None      else None,
            "Weekly_Vol_Ratio":     round(float(weekly_ratio), 2)         if weekly_ratio is not None      else None,
            "Monthly_Vol_Ratio":    round(float(monthly_ratio), 2)        if monthly_ratio is not None     else None,
            "Monthly_12M_Ratio":    round(float(monthly_12m_ratio), 2)    if monthly_12m_ratio is not None else None,
            "Monthly_Trend":        monthly_trend,
            "Accum_Distribution":   round(float(accum_distribution), 2)   if accum_distribution is not None else None,
            "Accum_Day_Count":      accum_day_count,
            "Bull_Candle":          bull,
            "Bear_Candle":          bear,
            "Strong_Bull":          strong_bull,
            "Strong_Bear":          strong_bear,
            "Close_Near_High":      close_near_high,
            "Breakout_Volume":      breakout_volume,
            "Weekly_Accumulation":  weekly_accum,
            "Monthly_Accumulation": monthly_accum,
            "Dry_Pullback":         dry_pullback,
            "Volume_State":         state,
            "Volume_Score":         score,
            "Volume_Warning":       warning,
        }

    except Exception as e:
        print("Volume Engine Error:", e)
        return default_volume()

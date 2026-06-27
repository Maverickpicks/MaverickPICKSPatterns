import pandas as pd
import numpy as np
from utils import last_value

# ============================================================
# PATTERN ENGINE
# Thinks like a swing trader reading a chart
#
# RULES FOR EVERY PATTERN:
# 1. Must occur in the right CONTEXT (trend, location)
# 2. Must have structural validity (not just math)
# 3. A Hammer at resistance is meaningless
#    A Hammer at support after a pullback = powerful
#
# SUPPORT / RESISTANCE:
# - Use 52-week price history to find key levels
# - Support = price zone where stock bounced multiple times
# - Resistance = price zone where stock struggled multiple times
# - Proximity to these levels gives context to every candle pattern
# ============================================================


def default_pattern():
    return {
        "Bullish_Engulfing":    False,
        "Hammer":               False,
        "Morning_Star":         False,
        "Inside_Bar":           False,
        "Double_Bottom":        False,
        "HHHL":                 False,
        "At_Support":           False,
        "At_Resistance":        False,
        "Support_Level":        None,
        "Resistance_Level":     None,
        "Buyers_At_Support":    False,
        "Pattern_Score":        0,
        "Pattern_State":        "NEUTRAL",
        "Primary_Pattern":      "None",
        "Chart_Context":        "NEUTRAL",
    }


# ============================================================
# SUPPORT & RESISTANCE DETECTION
# Finds key price levels using pivot points over 52 weeks
# A support level is one where price bounced at least twice
# A resistance level is one where price reversed at least twice
# ============================================================

def find_support_resistance(df, lookback=252, tolerance=0.025):
    """
    Scan last `lookback` bars for swing highs and swing lows.
    Group nearby levels (within `tolerance` %) as the same zone.
    Return the most significant support and resistance levels
    relative to current price.
    """
    try:
        data = df.tail(lookback).copy().reset_index(drop=True)
        close = last_value(df["Close"])

        swing_highs = []
        swing_lows  = []

        # Find swing highs and lows (local extremes with 5-bar lookback)
        for i in range(5, len(data) - 5):
            hi = data["High"].iloc[i]
            lo = data["Low"].iloc[i]

            # Swing high: higher than 5 bars on each side
            if hi == data["High"].iloc[i-5:i+6].max():
                swing_highs.append(hi)

            # Swing low: lower than 5 bars on each side
            if lo == data["Low"].iloc[i-5:i+6].min():
                swing_lows.append(lo)

        # Cluster nearby levels
        def cluster_levels(levels, tol):
            if not levels:
                return []
            levels = sorted(levels)
            clusters = []
            group = [levels[0]]
            for lv in levels[1:]:
                if (lv - group[0]) / group[0] <= tol:
                    group.append(lv)
                else:
                    clusters.append((np.mean(group), len(group)))
                    group = [lv]
            clusters.append((np.mean(group), len(group)))
            # Only keep levels tested 2+ times
            return [(lvl, cnt) for lvl, cnt in clusters if cnt >= 2]

        support_levels    = cluster_levels(swing_lows,  tolerance)
        resistance_levels = cluster_levels(swing_highs, tolerance)

        # Find nearest support BELOW current price
        supports_below = [(lvl, cnt) for lvl, cnt in support_levels if lvl < close * 0.99]
        nearest_support = max(supports_below, key=lambda x: x[0])[0] if supports_below else None

        # Find nearest resistance ABOVE current price
        resist_above = [(lvl, cnt) for lvl, cnt in resistance_levels if lvl > close * 1.01]
        nearest_resist = min(resist_above, key=lambda x: x[0])[0] if resist_above else None

        # Is price AT support (within 3%)?
        at_support = (
            nearest_support is not None
            and close <= nearest_support * 1.03
        )

        # Is price AT resistance (within 2%)?
        at_resistance = (
            nearest_resist is not None
            and close >= nearest_resist * 0.98
        )

        return {
            "support":       round(nearest_support, 2) if nearest_support else None,
            "resistance":    round(nearest_resist, 2)  if nearest_resist  else None,
            "at_support":    at_support,
            "at_resistance": at_resistance,
        }

    except:
        return {
            "support":       None,
            "resistance":    None,
            "at_support":    False,
            "at_resistance": False,
        }


# ============================================================
# WEEKLY CHART CONTEXT
# Read the weekly candle and weekly EMA for broader trend
# ============================================================

def weekly_context(weekly_df):
    """
    Returns whether the weekly chart is bullish, neutral, or bearish.
    A swing trader checks weekly FIRST before acting on daily signals.
    """
    try:
        if weekly_df is None or weekly_df.empty or len(weekly_df) < 52:
            return "UNKNOWN"

        # Weekly EMAs
        wc = weekly_df["Close"]
        ema20w = wc.ewm(span=20).mean().iloc[-1]
        ema10w = wc.ewm(span=10).mean().iloc[-1]
        close_w = last_value(wc)

        # This week's candle
        w = weekly_df.iloc[-1]
        week_bull = w["Close"] > w["Open"]
        week_close_near_high = (
            (w["Close"] - w["Low"]) / (w["High"] - w["Low"]) > 0.6
            if w["High"] > w["Low"] else False
        )

        # Previous week
        w_prev = weekly_df.iloc[-2]
        prev_bull = w_prev["Close"] > w_prev["Open"]

        # Weekly trend
        above_ema20w = close_w > ema20w
        above_ema10w = close_w > ema10w
        ema10w_rising = ema10w > weekly_df["Close"].ewm(span=10).mean().iloc[-4]

        if (
            above_ema20w
            and above_ema10w
            and ema10w_rising
            and week_bull
        ):
            return "BULLISH"

        elif (
            above_ema20w
            and (week_bull or week_close_near_high)
        ):
            return "POSITIVE"

        elif not above_ema20w and not week_bull:
            return "BEARISH"

        else:
            return "NEUTRAL"

    except:
        return "UNKNOWN"


# ============================================================
# PATTERN ANALYSIS — MAIN
# ============================================================

def pattern_analysis(df, weekly_df=None):

    try:

        if df is None or df.empty or len(df) < 50:
            return default_pattern()

        score   = 0
        primary = "None"

        c1 = df.iloc[-1]   # Today
        c2 = df.iloc[-2]   # Yesterday
        c3 = df.iloc[-3]   # 2 days ago

        close = float(c1["Close"])


        # ============================================================
        # SUPPORT / RESISTANCE LEVELS
        # ============================================================

        sr = find_support_resistance(df)
        at_support    = sr["at_support"]
        at_resistance = sr["at_resistance"]
        support_level = sr["support"]
        resist_level  = sr["resistance"]


        # ============================================================
        # WEEKLY CONTEXT
        # ============================================================

        w_context = weekly_context(weekly_df) if weekly_df is not None else "UNKNOWN"


        # ============================================================
        # BULLISH ENGULFING
        # Yesterday bearish, today bullish and body engulfs yesterday
        # Context bonus: only meaningful at support or after pullback
        # ============================================================

        bullish_engulfing = False

        if (
            c2["Close"] < c2["Open"]           # Yesterday bearish
            and c1["Close"] > c1["Open"]       # Today bullish
            and c1["Open"]  < c2["Close"]      # Today opens below yesterday close
            and c1["Close"] > c2["Open"]       # Today closes above yesterday open
        ):
            bullish_engulfing = True
            base_score = 5

            # Context bonus: at support or after pullback = stronger signal
            if at_support:
                base_score += 4
            if w_context in ["BULLISH", "POSITIVE"]:
                base_score += 2

            score   += base_score
            primary  = "Bullish Engulfing"


        # ============================================================
        # HAMMER
        # Long lower shadow (2x body), small upper shadow
        # Must be after a decline to be meaningful
        # ============================================================

        hammer = False

        body         = abs(float(c1["Close"]) - float(c1["Open"]))
        lower_shadow = float(min(c1["Open"], c1["Close"])) - float(c1["Low"])
        upper_shadow = float(c1["High"]) - float(max(c1["Open"], c1["Close"]))

        # Minimum body size — avoid doji-hammers (not meaningful)
        candle_range = float(c1["High"]) - float(c1["Low"])
        body_pct     = body / candle_range if candle_range > 0 else 0

        if (
            lower_shadow > body * 2
            and upper_shadow < body * 0.5
            and body_pct > 0.10        # Body must be at least 10% of range
        ):
            hammer = True
            base_score = 4

            if at_support:
                base_score += 4       # Hammer at support = high conviction
            if w_context in ["BULLISH", "POSITIVE"]:
                base_score += 2

            score += base_score
            if primary == "None":
                primary = "Hammer at Support" if at_support else "Hammer"


        # ============================================================
        # MORNING STAR
        # 3-candle reversal: big bear → small body (indecision) → big bull
        # Closes above midpoint of first candle
        # ============================================================

        morning_star = False

        body1 = abs(float(c3["Close"]) - float(c3["Open"]))
        body2 = abs(float(c2["Close"]) - float(c2["Open"]))
        body3 = abs(float(c1["Close"]) - float(c1["Open"]))

        if (
            c3["Close"] < c3["Open"]               # Day 1: big bearish
            and body2 < body1 * 0.5                # Day 2: small body (indecision)
            and c1["Close"] > c1["Open"]           # Day 3: bullish
            and c1["Close"] > (float(c3["Open"]) + float(c3["Close"])) / 2  # Closes above D1 midpoint
            and body3 > body1 * 0.5                # Day 3 body meaningful
        ):
            morning_star = True
            base_score   = 7

            if at_support:
                base_score += 3
            if w_context in ["BULLISH", "POSITIVE"]:
                base_score += 2

            score += base_score
            if primary == "None":
                primary = "Morning Star"


        # ============================================================
        # INSIDE BAR
        # Today's range completely inside yesterday's range
        # Means compression — breakout imminent
        # Only meaningful if preceded by a trending move
        # ============================================================

        inside_bar = False

        if (
            float(c1["High"]) < float(c2["High"])
            and float(c1["Low"])  > float(c2["Low"])
        ):
            # Only valid if yesterday was a reasonably sized candle
            prev_range = float(c2["High"]) - float(c2["Low"])
            if prev_range > 0:
                inside_bar  = True
                base_score  = 3

                if at_support:
                    base_score += 2
                if w_context == "BULLISH":
                    base_score += 2

                score += base_score
                if primary == "None":
                    primary = "Inside Bar"


        # ============================================================
        # REAL DOUBLE BOTTOM
        # Two distinct lows separated by a rally of at least 5%
        # Second low within 2% of first low
        # Price has now moved above the rally high between the two lows
        # This is a proper W pattern, not just two similar lows
        # ============================================================

        double_bottom = False

        if len(df) >= 60:
            data_60 = df.tail(60).reset_index(drop=True)
            lows_60  = data_60["Low"].values
            highs_60 = data_60["High"].values
            n        = len(lows_60)

            found_db = False

            for i in range(5, n - 10):
                l1 = lows_60[i]

                # Find the rally high after first low
                rally_section = highs_60[i:i+20]
                if len(rally_section) == 0:
                    continue
                rally_high_idx = i + int(np.argmax(rally_section))
                rally_high     = highs_60[rally_high_idx]

                # Rally must be at least 5% above first low
                if rally_high < l1 * 1.05:
                    continue

                # Find second low after the rally
                second_section = lows_60[rally_high_idx:min(rally_high_idx+20, n)]
                if len(second_section) < 3:
                    continue
                l2_idx = rally_high_idx + int(np.argmin(second_section))
                l2     = lows_60[l2_idx]

                # Second low must be within 2.5% of first low
                if abs(l1 - l2) / l1 > 0.025:
                    continue

                # There must be at least 5 bars between the two lows
                if l2_idx - i < 5:
                    continue

                # Current price should be breaking above the rally high
                # (confirmation of the W pattern)
                if close >= rally_high * 0.97:
                    double_bottom = True
                    found_db      = True
                    break

            if found_db:
                base_score = 8
                if at_support:
                    base_score += 3
                if w_context in ["BULLISH", "POSITIVE"]:
                    base_score += 2
                score += base_score
                if primary == "None":
                    primary = "Double Bottom"


        # ============================================================
        # HIGHER HIGHS HIGHER LOWS (HHHL)
        # 5 consecutive bars making HH and HL
        # Means trend is actively progressing
        # ============================================================

        hhhl = False

        highs5 = df["High"].tail(5).tolist()
        lows5  = df["Low"].tail(5).tolist()

        hh = all(highs5[i] > highs5[i-1] for i in range(1, 5))
        hl = all(lows5[i]  > lows5[i-1]  for i in range(1, 5))

        if hh and hl:
            hhhl       = True
            base_score = 5

            if w_context == "BULLISH":
                base_score += 3

            score += base_score
            if primary == "None":
                primary = "HHHL"


        # ============================================================
        # BUYERS AT SUPPORT
        # The key signal: price AT support AND bullish candle today
        # AND volume above average (buyers defending the level)
        # ============================================================

        today_vol   = float(c1["Volume"])
        avg_vol_20  = float(df["Volume"].tail(20).mean())
        vol_above_avg = today_vol > avg_vol_20

        buyers_at_support = (
            at_support
            and c1["Close"] > c1["Open"]    # Bullish candle
            and vol_above_avg               # Volume confirming
        )

        if buyers_at_support:
            score   += 6
            if primary == "None":
                primary = "Support Hold + Buyers"


        # ============================================================
        # CHART CONTEXT
        # Overall reading of what the chart is saying
        # ============================================================

        if at_resistance and not buyers_at_support:
            chart_context = "AT_RESISTANCE"
        elif buyers_at_support and w_context in ["BULLISH", "POSITIVE"]:
            chart_context = "STRONG_BULLISH"
        elif (bullish_engulfing or morning_star or double_bottom) and at_support:
            chart_context = "REVERSAL_SETUP"
        elif hhhl and w_context == "BULLISH":
            chart_context = "TRENDING_UP"
        elif at_support and not buyers_at_support:
            chart_context = "AT_SUPPORT_WATCH"
        elif w_context == "BEARISH":
            chart_context = "BEARISH"
        else:
            chart_context = "NEUTRAL"


        # ============================================================
        # CONTEXT PENALTY: at resistance = reduce score
        # ============================================================

        if at_resistance:
            score = max(0, score - 6)


        # ============================================================
        # PATTERN STATE
        # ============================================================

        state = "NEUTRAL"

        if score >= 14:
            state = "BULLISH"
        elif score >= 7:
            state = "POSITIVE"


        return {
            "Bullish_Engulfing":    bullish_engulfing,
            "Hammer":               hammer,
            "Morning_Star":         morning_star,
            "Inside_Bar":           inside_bar,
            "Double_Bottom":        double_bottom,
            "HHHL":                 hhhl,
            "At_Support":           at_support,
            "At_Resistance":        at_resistance,
            "Support_Level":        support_level,
            "Resistance_Level":     resist_level,
            "Buyers_At_Support":    buyers_at_support,
            "Weekly_Context":       w_context,
            "Pattern_Score":        score,
            "Pattern_State":        state,
            "Primary_Pattern":      primary,
            "Chart_Context":        chart_context,
        }

    except Exception as e:
        print("Pattern Engine Error:", e)
        return default_pattern()

import pandas as pd
import numpy as np
from utils import last_value


# ============================================================
# NARRATIVE ENGINE
#
# Generates a dynamic, specific story per stock — not a templated
# string. Counts actual support touches, references the real
# historical hit rate, and produces concrete entry/exit/scaling
# rules tied to both a PRICE LEVEL and a SIGNAL.
# ============================================================


def count_support_touches(df, support_level, tolerance=0.02, lookback=90):
    """
    Count how many times in the last `lookback` DAILY sessions price has
    come within `tolerance`% of the support level and bounced.
    This function reads the DAILY chart only.
    Returns (touch_count, days_since_last_touch, bounce_outcomes),
    where each outcome includes the actual calendar date of the touch.
    """
    if support_level is None:
        return 0, None, []

    recent = df.tail(lookback).copy()
    recent = recent.reset_index()
    date_col = recent.columns[0]   # original index (dates) becomes first column

    touches = []

    i = 0
    while i < len(recent):
        low = recent["Low"].iloc[i]
        if abs(low - support_level) / support_level <= tolerance:
            touch_idx = i
            touch_date = recent[date_col].iloc[i]
            forward = recent["Close"].iloc[touch_idx: touch_idx+6]
            if len(forward) >= 2:
                entry = forward.iloc[0]
                peak  = forward.max()
                bounce_pct = ((peak - entry) / entry) * 100
                touches.append({
                    "day_index": touch_idx,
                    "date": touch_date.strftime("%d %b %Y") if hasattr(touch_date, "strftime") else str(touch_date),
                    "bounce_pct": round(float(bounce_pct), 1)
                })
            i += 5
        else:
            i += 1

    touch_count = len(touches)
    days_since_last = (len(recent) - touches[-1]["day_index"]) if touches else None

    return touch_count, days_since_last, touches


def build_scenario_text(symbol, trend, momentum, rs, volume, pattern, risk, setup_result, daily_df=None):
    """
    Constructs the dynamic narrative for one stock based on its
    specific combination of signals. Returns a multi-part dict:
    headline, evidence, entry_exit, scaling_rule.
    """

    at_support      = pattern.get("At_Support", False)
    support_level   = pattern.get("Support_Level")
    buyers_support  = pattern.get("Buyers_At_Support", False)
    weekly_ctx      = pattern.get("Weekly_Context", "UNKNOWN")
    primary_pattern = pattern.get("Primary_Pattern", "None")

    rsi          = momentum.get("RSI")
    rsi_bucket   = setup_result.get("RSI_Bucket", "UNKNOWN")
    vol_bucket   = setup_result.get("Volume_Bucket", "UNKNOWN")

    confidence   = setup_result.get("Confidence_Pct")
    sample_size  = setup_result.get("Sample_Size", 0)
    avg_gain     = setup_result.get("Avg_Max_Gain")
    median_days  = setup_result.get("Median_Days_To_Target")
    used_fallback = setup_result.get("Used_Sector_Fallback", False)
    ath_bucket   = setup_result.get("ATH_Bucket", "UNKNOWN")
    data_as_of   = setup_result.get("Data_As_Of", "unknown date")

    entry  = risk.get("Entry")
    sl     = risk.get("Stop_Loss")
    t1     = risk.get("Target_1")
    atr    = risk.get("ATR")

    trend_state = trend.get("Trend_State")


    # ------------------------------------------------------------
    # HEADLINE — the one-line "why this stock" summary
    # ------------------------------------------------------------

    headline = ""

    if at_support and buyers_support:
        headline = (
            f"{symbol} is showing buyers actively defending support near "
            f"₹{support_level} with above-average volume today."
        )
    elif at_support:
        headline = (
            f"{symbol} is sitting at a tested support level (₹{support_level}) "
            f"and showing early signs of stabilizing."
        )
    elif rsi_bucket == "COOL" and trend_state in ["STRONG", "LEADER"]:
        headline = (
            f"{symbol} is cooling off within an intact uptrend — RSI has "
            f"pulled back to {round(rsi,1) if rsi else 'a healthy zone'} without breaking trend structure."
        )
    elif primary_pattern not in ("None", None):
        headline = f"{symbol} has formed a {primary_pattern} pattern, suggesting a potential turn."
    else:
        headline = f"{symbol} matches a historically favourable setup based on RSI, volume, and price structure."


    # ------------------------------------------------------------
    # EVIDENCE — the specific historical/structural backing
    # ------------------------------------------------------------

    evidence_lines = []

    if sample_size > 0 and confidence is not None:
        source = "similar setups across its sector peers" if used_fallback else "similar setups in its own 2-year history"
        evidence_lines.append(
            f"This exact RSI+Volume+Price fingerprint ({rsi_bucket} RSI, {vol_bucket} volume) "
            f"has occurred {sample_size} times in {source}. "
            f"In {confidence}% of those cases, the stock moved 4-10% within "
            f"{int(median_days) if median_days else '~7'} trading days on average "
            f"(average peak gain: {avg_gain}%)."
        )

    if at_support and support_level and daily_df is not None:
        support_text = build_support_narrative(daily_df, support_level, symbol)
        if support_text:
            evidence_lines.append(support_text)

    if weekly_ctx in ["BULLISH", "POSITIVE"]:
        evidence_lines.append(f"Weekly chart structure supports this: trend reading is {weekly_ctx}.")
    elif weekly_ctx == "NEUTRAL":
        evidence_lines.append("Weekly chart is neutral — no strong tailwind, but no resistance either.")

    if volume.get("Monthly_Trend") in ["RISING", "SURGING"]:
        evidence_lines.append(
            f"Monthly volume trend is {volume.get('Monthly_Trend')} "
            f"({volume.get('Monthly_12M_Ratio')}x the 12-month average) — growing institutional interest."
        )

    # ATH caveat — if the historical match sample is itself drawn from
    # AT_ATH or NEAR_ATH days, say so explicitly. A 9/10 hit rate while
    # "at a new high" is a different, weaker claim than the same hit
    # rate from a genuine pullback/support setup, and you should see
    # that distinction rather than read Sample_Size as same-price repeats.
    if ath_bucket in ["AT_ATH", "NEAR_ATH"] and sample_size > 0:
        evidence_lines.append(
            f"Caveat: this stock is currently {('at' if ath_bucket=='AT_ATH' else 'near')} its 52-week high. "
            f"The {sample_size} historical matches above are ALSO drawn only from days when the stock was "
            f"{('at' if ath_bucket=='AT_ATH' else 'near')} a 52-week high (not from random pullback days at a lower price) — "
            f"so this reads as 'continuation of an existing uptrend,' not a fresh support/pullback signal."
        )


    # ------------------------------------------------------------
    # ENTRY / EXIT
    # ------------------------------------------------------------

    entry_exit = (
        f"Entry near ₹{entry}. Stop loss at ₹{sl} "
        f"({risk.get('Risk_Percent')}% risk). "
        f"First target ₹{t1} ({risk.get('Reward_Risk')}x reward:risk)."
    )


    # ------------------------------------------------------------
    # SCALING RULE — price level AND signal, both directions
    # ------------------------------------------------------------

    double_up_price = round(entry * 1.02, 2) if entry else None  # 2% above entry
    double_down_price = round(sl, 2) if sl else None

    scaling_rule = ""

    if double_up_price and atr:
        scaling_rule += (
            f"Add to position if price closes above ₹{double_up_price} "
            f"AND today's volume is above its 20-day average (confirms buyers stepping in). "
        )

    if double_down_price:
        scaling_rule += (
            f"Cut the position if price closes below ₹{double_down_price} "
            f"(stop level) OR if RSI rolls back below 40 with rising volume "
            f"(signals sellers regaining control) — whichever comes first."
        )

    headline = f"[Data as of {data_as_of}] " + headline

    return {
        "Headline":      headline,
        "Evidence":      " ".join(evidence_lines) if evidence_lines else "Limited additional confirming evidence available.",
        "Entry_Exit":    entry_exit,
        "Scaling_Rule":  scaling_rule.strip(),
        "Data_As_Of":    data_as_of,
    }


def build_support_narrative(df, support_level, symbol):
    """
    Builds the specific 'Stock is at its support for X days, now
    could go up' scenario text, using real touch-count history with
    actual calendar dates. Explicitly states this reads the DAILY chart,
    since support/resistance levels are computed from daily price data
    elsewhere in the system (pattern_engine.py).
    """

    if support_level is None:
        return None

    touch_count, days_since_last, touches = count_support_touches(df, support_level)

    if touch_count == 0:
        return None

    successful_bounces = [t for t in touches if t["bounce_pct"] >= 3]
    success_rate = (len(successful_bounces) / touch_count) * 100 if touch_count else 0
    avg_bounce = np.mean([t["bounce_pct"] for t in touches]) if touches else 0

    touch_dates = [t["date"] for t in touches]
    dates_str = ", ".join(touch_dates)

    text = (
        f"On the DAILY chart, {symbol} has tested the ₹{support_level} support level "
        f"{touch_count} time(s) in the last 90 sessions, on: {dates_str}. "
    )

    if touch_count >= 2:
        text += (
            f"On {len(successful_bounces)} of those {touch_count} occasions "
            f"({round(success_rate)}%), it bounced at least 3% within 5 trading days "
            f"(average bounce: {round(avg_bounce,1)}%). "
        )

    if days_since_last is not None and days_since_last <= 3:
        text += f"It is currently on a fresh test of this level (last touched on {touch_dates[-1]}, {days_since_last} session(s) ago)."
    elif days_since_last is not None:
        text += f"It last touched this level on {touch_dates[-1]} ({days_since_last} sessions ago)."

    return text

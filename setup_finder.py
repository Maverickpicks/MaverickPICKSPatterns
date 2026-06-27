import pandas as pd
import numpy as np
import pandas_ta as ta
from utils import last_value, safe_divide


# ============================================================
# SETUP FINDER — THE CORE "DOES THIS ACTUALLY MOVE" ENGINE
#
# Philosophy:
#   Don't trust a generic score. Test the THEORY against the
#   stock's own price history.
#
#   "Today this stock has RSI 45, price at EMA20, volume drying up."
#   "How many times in the last 2 years did this EXACT fingerprint
#    occur? What happened in the next 5-10 trading days?"
#
#   If it moved 4-10% within 10 days, 7 times out of 9 historical
#   matches -> that's a 78% confidence, evidence-based forecast.
#
#   If a stock has too little history (recent listing), borrow the
#   fingerprint outcome statistics from its sector peers instead.
# ============================================================


# ------------------------------------------------------------
# FINGERPRINT DEFINITION
# A "fingerprint" is a discretized snapshot of conditions:
#   RSI bucket      : OVERSOLD / COOL / NEUTRAL / WARM / OVERBOUGHT
#   Volume bucket   : DRY / NORMAL / ELEVATED / SURGE
#   Price zone      : ABOVE_EMA9 / EMA9_EMA20 / EMA20_EMA50 /
#                      EMA50_EMA200 / BELOW_EMA200
# Two days match if all three buckets are identical.
# ------------------------------------------------------------

def rsi_bucket(rsi):
    if rsi is None or pd.isna(rsi):
        return "UNKNOWN"
    if rsi < 35:
        return "OVERSOLD"
    elif rsi < 48:
        return "COOL"
    elif rsi < 62:
        return "NEUTRAL"
    elif rsi < 70:
        return "WARM"
    else:
        return "OVERBOUGHT"


def volume_bucket(vol_ratio):
    if vol_ratio is None or pd.isna(vol_ratio):
        return "UNKNOWN"
    if vol_ratio < 0.8:
        return "DRY"
    elif vol_ratio < 1.3:
        return "NORMAL"
    elif vol_ratio < 2.0:
        return "ELEVATED"
    else:
        return "SURGE"


def price_zone_bucket(close, ema9, ema20, ema50, ema200):
    if any(pd.isna(x) for x in [close, ema9, ema20, ema50, ema200]):
        return "UNKNOWN"
    if close >= ema9:
        return "ABOVE_EMA9"
    elif close >= ema20:
        return "EMA9_EMA20"
    elif close >= ema50:
        return "EMA20_EMA50"
    elif close >= ema200:
        return "EMA50_EMA200"
    else:
        return "BELOW_EMA200"


def ath_distance_bucket(close, rolling_52w_high):
    """
    Buckets how far price is below its trailing 52-week high.
    This is the missing piece that prevented MCX-style false matches:
    without it, a stock in a 2-year uptrend matches itself at every
    price level it has ever passed through, since RSI/Volume/EMA-zone
    repeat constantly even as the absolute price triples.
    Adding this means a match at ATH only counts other ATH-region days,
    not random pullback days from a year ago at a third of the price.
    """
    if pd.isna(close) or pd.isna(rolling_52w_high) or rolling_52w_high == 0:
        return "UNKNOWN"

    dist_pct = ((rolling_52w_high - close) / rolling_52w_high) * 100

    if dist_pct <= 3:
        return "AT_ATH"
    elif dist_pct <= 10:
        return "NEAR_ATH"
    elif dist_pct <= 20:
        return "MODERATE_PULLBACK"
    elif dist_pct <= 35:
        return "DEEP_PULLBACK"
    else:
        return "FAR_FROM_ATH"


# ------------------------------------------------------------
# BUILD FULL FEATURE SERIES FOR A STOCK
# Computes RSI, volume ratio, EMAs, and resulting fingerprint
# for every single day in the dataframe history
# ------------------------------------------------------------

def build_feature_series(df):

    data = df.copy()

    data["RSI"]   = ta.rsi(data["Close"], length=14)
    data["EMA9"]  = ta.ema(data["Close"], length=9)
    data["EMA20"] = ta.ema(data["Close"], length=20)
    data["EMA50"] = ta.ema(data["Close"], length=50)
    data["EMA200"]= ta.ema(data["Close"], length=200)

    data["Vol_Avg20"]  = data["Volume"].rolling(20).mean()
    data["Vol_Ratio"]  = data["Volume"] / data["Vol_Avg20"]

    # Trailing 52-week (252 trading day) high, known at each point in time
    # (no lookahead — uses only data up to and including that day)
    data["Rolling_52W_High"] = data["High"].rolling(252, min_periods=20).max()

    data["RSI_B"]   = data["RSI"].apply(rsi_bucket)
    data["VOL_B"]   = data["Vol_Ratio"].apply(volume_bucket)

    data["ZONE_B"] = data.apply(
        lambda r: price_zone_bucket(
            r["Close"], r["EMA9"], r["EMA20"], r["EMA50"], r["EMA200"]
        ),
        axis=1
    )

    data["ATH_B"] = data.apply(
        lambda r: ath_distance_bucket(r["Close"], r["Rolling_52W_High"]),
        axis=1
    )

    # Fingerprint now includes ATH distance — this is the fix.
    # A stock at AT_ATH only matches other AT_ATH days (same regime),
    # never pullback days from a year ago at a fraction of the price.
    data["Fingerprint"] = data["RSI_B"] + "|" + data["VOL_B"] + "|" + data["ZONE_B"] + "|" + data["ATH_B"]

    return data


# ------------------------------------------------------------
# FIND HISTORICAL MATCHES AND THEIR FORWARD OUTCOMES
#
# For every past day with the same fingerprint as today,
# look forward 10 trading days and record:
#   - max % gain reached within that window
#   - number of days taken to reach a 4%+ gain (if any)
#   - whether a 4-10% move happened, and whether it overshot 10%
#
# Excludes the last 12 days (no room to look forward) and
# excludes today itself.
# ------------------------------------------------------------

def find_historical_matches(feature_df, today_fingerprint, forward_window=10,
                              target_low=4.0, target_high=10.0):

    matches = []

    usable = feature_df.iloc[:-forward_window-1] if len(feature_df) > forward_window + 1 else feature_df.iloc[0:0]

    matched_rows = usable[usable["Fingerprint"] == today_fingerprint]

    for idx in matched_rows.index:
        loc = feature_df.index.get_loc(idx)

        entry_price = feature_df["Close"].iloc[loc]

        forward_slice = feature_df["Close"].iloc[loc+1: loc+1+forward_window]
        forward_high  = feature_df["High"].iloc[loc+1: loc+1+forward_window]

        if len(forward_slice) == 0 or pd.isna(entry_price) or entry_price == 0:
            continue

        pct_changes = ((forward_high - entry_price) / entry_price) * 100

        max_gain = pct_changes.max()

        # Day index (1-based, positional) on which it first crossed target_low%
        days_to_target = None
        for i, val in enumerate(pct_changes.values, start=1):
            if val >= target_low:
                days_to_target = i
                break

        hit_target_range = (max_gain >= target_low)
        overshot = (max_gain > target_high)

        matches.append({
            "max_gain": round(float(max_gain), 2),
            "hit_target": bool(hit_target_range),
            "overshot": bool(overshot),
            "days_to_target": days_to_target,
        })

    return matches


# ------------------------------------------------------------
# SUMMARIZE MATCHES INTO CONFIDENCE + EXPECTED TIMING
# ------------------------------------------------------------

def summarize_matches(matches, min_samples=5):

    n = len(matches)

    if n < min_samples:
        return {
            "Sample_Size":       n,
            "Hit_Rate_Pct":      None,
            "Confidence_Pct":    None,
            "Avg_Max_Gain":      None,
            "Median_Days_To_Target": None,
            "Overshoot_Rate_Pct": None,
            "Insufficient_Data": True,
        }

    hits = [m for m in matches if m["hit_target"]]
    hit_rate = (len(hits) / n) * 100

    overshoots = [m for m in matches if m["overshot"]]
    overshoot_rate = (len(overshoots) / n) * 100

    avg_max_gain = float(np.mean([m["max_gain"] for m in matches]))

    days_list = [m["days_to_target"] for m in hits if m["days_to_target"] is not None]
    median_days = float(np.median(days_list)) if days_list else None

    # Confidence = hit rate, but penalised if sample size is small (5-9 matches)
    # and penalised if the historical moves wildly overshoot or undershoot target
    confidence = hit_rate

    if n < 9:
        confidence *= 0.85   # small sample penalty

    if overshoot_rate > 50:
        confidence *= 0.92   # frequently overshoots — less precise, slight penalty

    confidence = max(0, min(100, confidence))

    return {
        "Sample_Size":            n,
        "Hit_Rate_Pct":           round(hit_rate, 1),
        "Confidence_Pct":         round(confidence, 1),
        "Avg_Max_Gain":           round(avg_max_gain, 2),
        "Median_Days_To_Target":  median_days,
        "Overshoot_Rate_Pct":     round(overshoot_rate, 1),
        "Insufficient_Data":      False,
    }


# ------------------------------------------------------------
# SECTOR FALLBACK
# If a stock has insufficient history (e.g. < 18 months), pool
# fingerprint matches from sector peers instead.
# Requires a dict: {symbol: feature_df} for peers in same sector,
# already pre-built by the caller.
# ------------------------------------------------------------

def find_sector_fallback_matches(peer_feature_dfs, today_fingerprint,
                                   forward_window=10, target_low=4.0, target_high=10.0,
                                   max_peers=15):

    pooled = []

    for sym, fdf in list(peer_feature_dfs.items())[:max_peers]:
        try:
            m = find_historical_matches(
                fdf, today_fingerprint, forward_window, target_low, target_high
            )
            pooled.extend(m)
        except Exception:
            continue

    return pooled


# ------------------------------------------------------------
# MAIN ENTRY POINT
# Given a stock's daily df (+ optional sector peer dfs), returns
# the full setup-quality verdict for "will this move 4-10% soon"
# ------------------------------------------------------------

def evaluate_setup(daily_df, symbol, sector_peer_feature_dfs=None,
                     target_low=4.0, target_high=10.0, forward_window=10,
                     min_samples=5):

    try:

        if daily_df is None or daily_df.empty or len(daily_df) < 60:
            return _empty_setup_result()

        feature_df = build_feature_series(daily_df)

        data_as_of = feature_df.index[-1].strftime("%Y-%m-%d") if hasattr(feature_df.index[-1], "strftime") else str(feature_df.index[-1])

        last_row = feature_df.iloc[-1]
        today_fingerprint = last_row["Fingerprint"]

        if "UNKNOWN" in today_fingerprint:
            return _empty_setup_result(today_fingerprint, data_as_of)

        # Try self-history first (need decent depth: ~300 trading days = ~14 months)
        used_fallback = False

        if len(feature_df) >= 300:
            matches = find_historical_matches(
                feature_df, today_fingerprint, forward_window, target_low, target_high
            )
        else:
            matches = []

        summary = summarize_matches(matches, min_samples=min_samples)

        # If self-history insufficient, fall back to sector peers
        if summary["Insufficient_Data"] and sector_peer_feature_dfs:
            peer_matches = find_sector_fallback_matches(
                sector_peer_feature_dfs, today_fingerprint,
                forward_window, target_low, target_high
            )
            if len(peer_matches) >= min_samples:
                summary = summarize_matches(peer_matches, min_samples=min_samples)
                used_fallback = True

        summary["Used_Sector_Fallback"] = used_fallback
        summary["Fingerprint"] = today_fingerprint
        summary["RSI_Bucket"]   = last_row["RSI_B"]
        summary["Volume_Bucket"] = last_row["VOL_B"]
        summary["Zone_Bucket"]  = last_row["ZONE_B"]
        summary["ATH_Bucket"]   = last_row["ATH_B"]
        summary["Current_RSI"]  = round(float(last_row["RSI"]), 1) if pd.notna(last_row["RSI"]) else None
        summary["Data_As_Of"]   = feature_df.index[-1].strftime("%Y-%m-%d") if hasattr(feature_df.index[-1], "strftime") else str(feature_df.index[-1])

        return summary

    except Exception as e:
        print(f"Setup Finder Error ({symbol}):", e)
        return _empty_setup_result()


def _empty_setup_result(fingerprint="UNKNOWN", data_as_of=None):
    return {
        "Sample_Size":            0,
        "Hit_Rate_Pct":           None,
        "Confidence_Pct":         None,
        "Avg_Max_Gain":           None,
        "Median_Days_To_Target":  None,
        "Overshoot_Rate_Pct":     None,
        "Insufficient_Data":      True,
        "Used_Sector_Fallback":   False,
        "Fingerprint":            fingerprint,
        "RSI_Bucket":             "UNKNOWN",
        "Volume_Bucket":          "UNKNOWN",
        "Zone_Bucket":            "UNKNOWN",
        "ATH_Bucket":             "UNKNOWN",
        "Current_RSI":            None,
        "Data_As_Of":             data_as_of,
    }

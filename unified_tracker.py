"""
unified_tracker.py — MaverickPICKS Unified Pick Tracker
=========================================================
Tracks picks from all three scanners in one place:
  • main_v3          → MaverickPICKS_Top10_Report.xlsx (Mon/Wed scans)
  • pattern_detector → todays_picks.csv                (daily EOD scans)
  • hs_detector      → hs_watchlist.csv                (daily EOD scans)

Generates two dashboards viewable on GitHub via HTMLPreview:
  • dashboard_maverick.html  → main_v3 picks
  • dashboard_patterns.html  → Murphy patterns + H&S combined

Usage:
  Import after scan:
    python unified_tracker.py --import_maverick MaverickPICKS_Top10_Report.xlsx
    python unified_tracker.py --import_patterns todays_picks.csv
    python unified_tracker.py --import_hs       hs_watchlist.csv

  Daily check (run at 4:15 PM IST after all scanners finish):
    python unified_tracker.py --check

  Combined (recommended):
    python unified_tracker.py \
        --import_patterns todays_picks.csv \
        --import_hs hs_watchlist.csv \
        --check

  Mon/Wed combined:
    python unified_tracker.py \
        --import_maverick MaverickPICKS_Top10_Report.xlsx \
        --import_patterns todays_picks.csv \
        --import_hs hs_watchlist.csv \
        --check
"""

import argparse
import json
import os
import time
import warnings
from datetime import datetime, timedelta, timezone
from typing import Optional

import numpy as np
import pandas as pd
import yfinance as yf
from scipy.stats import linregress

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────────────────────

WATCHLIST_FILE       = "unified_watchlist.json"
DASH_MAVERICK        = "dashboard_maverick.html"
DASH_PATTERNS        = "dashboard_patterns.html"

IST = timezone(timedelta(hours=5, minutes=30))

# Source tags
SRC_MAVERICK = "MaverickPICKS"
SRC_MURPHY   = "Murphy Pattern"
SRC_HS       = "H&S Pattern"

# States
ST_WATCHING  = "WATCHING"
ST_BREAKOUT  = "BREAKOUT"
ST_BREAKDOWN = "BREAKDOWN"
ST_EXPIRED   = "EXPIRED"
ST_FAILED    = "FAILED"        # H&S specific

# Track signal
TRK_BREAKOUT  = "ON TRACK → BREAKOUT"
TRK_BREAKDOWN = "CAUTION → BREAKDOWN"
TRK_NEUTRAL   = "NEUTRAL"
TRK_STALE     = "STALE"


# ─────────────────────────────────────────────────────────────────────────────
# DATE UTILITIES
# ─────────────────────────────────────────────────────────────────────────────

def _last_trading_day() -> datetime:
    now = datetime.now(IST)
    d   = now.date()
    if d.weekday() == 5: d -= timedelta(days=1)
    elif d.weekday() == 6: d -= timedelta(days=2)
    else:
        mc = now.replace(hour=15, minute=30, second=0, microsecond=0)
        if now < mc:
            d -= timedelta(days=1)
            if d.weekday() == 6: d -= timedelta(days=2)
            elif d.weekday() == 5: d -= timedelta(days=1)
    return datetime(d.year, d.month, d.day)


_ANCHOR = _last_trading_day()


def _fmt(ts) -> str:
    try:
        return pd.Timestamp(ts).strftime("%d-%b-%Y")
    except Exception:
        return str(ts) if ts else "—"


# ─────────────────────────────────────────────────────────────────────────────
# WATCHLIST PERSISTENCE
# ─────────────────────────────────────────────────────────────────────────────

def load_wl() -> dict:
    if os.path.exists(WATCHLIST_FILE):
        with open(WATCHLIST_FILE) as f:
            return json.load(f)
    return {"maverick": {}, "patterns": {}, "hs": {}, "last_updated": None}


def save_wl(wl: dict):
    wl["last_updated"] = datetime.now(IST).strftime("%Y-%m-%d %H:%M IST")
    with open(WATCHLIST_FILE, "w") as f:
        json.dump(wl, f, indent=2, default=str)


def _key(symbol: str, tag: str) -> str:
    return f"{symbol}|{tag}"


# ─────────────────────────────────────────────────────────────────────────────
# PRICE FETCH
# ─────────────────────────────────────────────────────────────────────────────

def fetch_ohlcv(symbol: str, days: int = 30) -> Optional[pd.DataFrame]:
    """Fetch recent OHLCV — mirrors data_loader.py exactly."""
    end   = _ANCHOR + timedelta(days=1)
    start = _ANCHOR - timedelta(days=days + 10)
    for attempt in range(3):
        try:
            df = yf.download(
                symbol if symbol.endswith(".NS") else symbol + ".NS",
                start=start, end=end,
                interval="1d", auto_adjust=False,
                progress=False, threads=False,
            )
            if df.empty:
                raise ValueError("empty")
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = [c[0] for c in df.columns]
            df = df[["Open", "High", "Low", "Close", "Volume"]].dropna()
            df.index = pd.to_datetime(df.index)
            return df.tail(days)
        except Exception:
            time.sleep(1)
    return None


def _last_bar(df: pd.DataFrame) -> tuple:
    """Returns (close, volume, date_str) from last row."""
    if df is None or df.empty:
        return None, None, None
    last = df.iloc[-1]
    return float(last["Close"]), float(last["Volume"]), df.index[-1].strftime("%Y-%m-%d")


# ─────────────────────────────────────────────────────────────────────────────
# TRACK SIGNAL ENGINE
# ─────────────────────────────────────────────────────────────────────────────

def _slope(series: pd.Series) -> float:
    if len(series) < 2:
        return 0.0
    x = np.arange(len(series))
    s, *_ = linregress(x, series.values.astype(float))
    return float(s)


def _gap_velocity(history: list) -> Optional[float]:
    """
    Rate of change of gap_to_breakout over last 5 days.
    Negative = gap shrinking = price approaching breakout = bullish.
    Positive = gap widening = price moving away = bearish.
    """
    gaps = [h["gap_pct"] for h in history[-6:] if h.get("gap_pct") is not None]
    if len(gaps) < 2:
        return None
    return float(np.mean(np.diff(gaps)))   # avg daily change in gap %


def _days_to_target(gap_pct: float, velocity: float) -> Optional[int]:
    """Estimate trading days until price reaches breakout at current velocity."""
    if velocity is None or velocity >= 0 or gap_pct <= 0:
        return None
    days = gap_pct / abs(velocity)
    return int(round(days)) if days < 60 else None


def murphy_track_signal(pick: dict, df: pd.DataFrame) -> tuple:
    """
    Assess whether a Murphy pattern pick is on track for breakout or breakdown.
    Returns (signal, detail_str, eta_days).

    Criteria — Breakout track:
      • Volume declining within consolidation (Murphy's core requirement)
      • Gap to breakout shrinking over last 3-5 days
      • Price making higher lows (close slope positive within flag)
      • RSI recovering (close vs recent lows improving)

    Criteria — Breakdown track:
      • Volume NOT declining (sellers active)
      • Gap to breakout widening
      • Price making lower lows beyond 50% retracement
      • Close approaching stop loss
    """
    history  = pick.get("price_history", [])
    brk      = pick["breakout_level"]
    stop     = pick["stop_loss"]
    gap_pct  = history[-1]["gap_pct"] if history else pick.get("gap_to_break_pct", 0)

    signals_bull = 0
    signals_bear = 0
    details = []

    # ── 1. Gap velocity ───────────────────────────────────────────────────────
    vel = _gap_velocity(history)
    if vel is not None:
        if vel < -0.3:
            signals_bull += 2
            details.append(f"gap closing {abs(vel):.2f}%/day ✓")
        elif vel > 0.3:
            signals_bear += 2
            details.append(f"gap widening {vel:.2f}%/day ✗")

    # ── 2. Volume trend in recent bars ────────────────────────────────────────
    if df is not None and len(df) >= 5:
        recent_vol = df["Volume"].tail(5)
        vol_slope  = _slope(recent_vol)
        if vol_slope < 0:
            signals_bull += 1
            details.append("vol contracting ✓")
        else:
            signals_bear += 1
            details.append("vol expanding ✗")

    # ── 3. Price making higher lows in flag ───────────────────────────────────
    if df is not None and len(df) >= 5:
        close_slope = _slope(df["Close"].tail(5))
        if close_slope > 0:
            signals_bull += 1
            details.append("price recovering ✓")
        else:
            signals_bear += 1
            details.append("price still declining ✗")

    # ── 4. Proximity to stop loss ─────────────────────────────────────────────
    if df is not None:
        last_close = float(df["Close"].iloc[-1])
        dist_to_stop = (last_close - stop) / stop * 100
        if dist_to_stop < 2.0:
            signals_bear += 2
            details.append(f"only {dist_to_stop:.1f}% above stop ✗")
        elif gap_pct < 2.0:
            signals_bull += 1
            details.append(f"only {gap_pct:.1f}% from breakout ✓")

    # ── Verdict ────────────────────────────────────────────────────────────────
    eta = _days_to_target(gap_pct, vel) if vel and vel < 0 else None

    if signals_bull >= 3 and signals_bull > signals_bear:
        return TRK_BREAKOUT, " | ".join(details), eta
    elif signals_bear >= 3 and signals_bear > signals_bull:
        return TRK_BREAKDOWN, " | ".join(details), None
    else:
        return TRK_NEUTRAL, " | ".join(details), eta


def hs_track_signal(pick: dict, df: pd.DataFrame) -> tuple:
    """
    Assess whether an H&S pick is on track for breakout/breakdown.
    Returns (signal, detail_str, eta_days).

    H&S Forming → Breakout track (Inverse H&S):
      • Price holding above right shoulder level
      • Volume expanding on up-days, contracting on down-days
      • Price approaching neckline from below

    H&S Forming → Breakdown track (H&S):
      • Price drifting toward neckline
      • Volume expanding (selling pressure)
      • Is_stale = True
    """
    history    = pick.get("price_history", [])
    neckline   = pick.get("neckline_price", 0)
    stop       = pick.get("stop_loss", 0)
    is_inverse = pick.get("pattern_type", "") == "Inverse H&S"
    is_stale   = pick.get("is_stale", False)

    signals_bull = 0
    signals_bear = 0
    details = []

    if is_stale:
        signals_bear += 2
        details.append("pattern overdue ✗")

    if df is not None and len(df) >= 5:
        last_close  = float(df["Close"].iloc[-1])
        close_slope = _slope(df["Close"].tail(5))
        vol_slope   = _slope(df["Volume"].tail(5))

        # Distance to neckline
        dist_pct = abs(last_close - neckline) / neckline * 100 if neckline else None

        if is_inverse:
            # Inverse H&S: bullish reversal — want price rising toward neckline
            if close_slope > 0:
                signals_bull += 2
                details.append("price rising toward neckline ✓")
            else:
                signals_bear += 1
                details.append("price not rising ✗")
            if vol_slope > 0:
                signals_bull += 1
                details.append("volume expanding ✓")
            if dist_pct and dist_pct < 3:
                signals_bull += 2
                details.append(f"only {dist_pct:.1f}% from neckline ✓")
        else:
            # H&S: bearish reversal — want price falling toward neckline
            if close_slope < 0:
                signals_bear += 2
                details.append("price falling toward neckline ✓ (bearish)")
            else:
                signals_bull += 1
                details.append("price holding up ✓")
            if vol_slope > 0:
                signals_bear += 1
                details.append("volume expanding (bearish) ✗")
            if dist_pct and dist_pct < 3:
                signals_bear += 2
                details.append(f"only {dist_pct:.1f}% from neckline break ✓")

    # Gap velocity (approaching neckline)
    vel = _gap_velocity(history)
    eta = None
    if vel is not None and dist_pct:
        if not is_inverse and vel < -0.3:
            eta = _days_to_target(dist_pct, vel)
        elif is_inverse and vel < -0.3:
            eta = _days_to_target(dist_pct, vel)

    if signals_bull >= 3 and signals_bull > signals_bear:
        return TRK_BREAKOUT, " | ".join(details), eta
    elif signals_bear >= 3 and signals_bear > signals_bull:
        return TRK_BREAKDOWN, " | ".join(details), eta
    else:
        return TRK_NEUTRAL, " | ".join(details), eta


def maverick_track_signal(pick: dict, df: pd.DataFrame) -> tuple:
    """
    Assess whether a main_v3 pick is on track for entry.
    Returns (signal, detail_str, eta_days).

    On track signals:
      • Price gap to entry narrowing
      • Volume improving (DRY → NORMAL → ELEVATED)
      • Momentum improving (RSI recovering)
      • Price above key EMAs

    Off track signals:
      • Price moving away from entry
      • Volume drying up
      • Price below stop loss
    """
    history  = pick.get("price_history", [])
    entry    = pick.get("entry", 0)
    stop     = pick.get("stop_loss", 0)
    conf     = pick.get("confidence_pct", 0)

    signals_bull = 0
    signals_bear = 0
    details = []

    # ── Gap velocity ───────────────────────────────────────────────────────────
    vel = _gap_velocity(history)
    if vel is not None:
        if vel < -0.3:
            signals_bull += 2
            details.append(f"closing in on entry {abs(vel):.2f}%/day ✓")
        elif vel > 0.5:
            signals_bear += 2
            details.append(f"moving away from entry {vel:.2f}%/day ✗")

    if df is not None and len(df) >= 10:
        last_close  = float(df["Close"].iloc[-1])
        close_slope = _slope(df["Close"].tail(5))
        vol_slope   = _slope(df["Volume"].tail(5))

        # EMA9 check
        ema9 = df["Close"].ewm(span=9, adjust=False).mean().iloc[-1]
        ema20 = df["Close"].ewm(span=20, adjust=False).mean().iloc[-1]

        if last_close > ema9 and close_slope > 0:
            signals_bull += 2
            details.append("above EMA9, rising ✓")
        elif last_close < ema9:
            signals_bear += 1
            details.append("below EMA9 ✗")

        if last_close > ema20:
            signals_bull += 1
            details.append("above EMA20 ✓")

        # Volume
        avg_vol = df["Volume"].tail(20).mean()
        if vol_slope > 0 and last_close > ema9:
            signals_bull += 1
            details.append("vol expanding with price ✓")
        elif vol_slope < 0 and last_close < ema9:
            signals_bear += 1
            details.append("vol drying up ✗")

        # Stop proximity
        dist_to_stop = (last_close - stop) / stop * 100 if stop > 0 else 100
        if dist_to_stop < 2.0:
            signals_bear += 2
            details.append(f"near stop ({dist_to_stop:.1f}% away) ✗")

    # ETA
    gap_pct = history[-1]["gap_pct"] if history else pick.get("gap_to_entry_pct", 0)
    eta = _days_to_target(gap_pct, vel) if vel and vel < 0 else None

    if signals_bull >= 3 and signals_bull > signals_bear:
        return TRK_BREAKOUT, " | ".join(details), eta
    elif signals_bear >= 3 and signals_bear > signals_bull:
        return TRK_BREAKDOWN, " | ".join(details), None
    else:
        return TRK_NEUTRAL, " | ".join(details), eta


# ─────────────────────────────────────────────────────────────────────────────
# PRIORITY SCORE
# ─────────────────────────────────────────────────────────────────────────────

def priority_score_murphy(pick: dict, track: str, eta: Optional[int]) -> float:
    """
    Priority = Pattern_score × volume_mult × time_mult × proximity_mult
    Higher = show first in dashboard.
    """
    score = pick.get("score", 50)

    # Volume multiplier
    vol_all = pick.get("vol_all_3_ok", False)
    vol_avg = pick.get("vol_consol_avg_ok", False)
    vol_mul = 1.3 if vol_all else (1.1 if vol_avg else 0.9)

    # Time multiplier
    expiry   = pick.get("expiry_date")
    time_mul = 1.0
    if expiry:
        try:
            days_left = (pd.to_datetime(expiry) - pd.Timestamp(_ANCHOR)).days
            if 3 <= days_left <= 5:
                time_mul = 1.15   # good window
            elif days_left <= 2:
                time_mul = 0.8    # too rushed
            elif days_left > 10:
                time_mul = 0.95   # plenty of time, lower urgency
        except Exception:
            pass

    # Proximity multiplier
    gap = pick.get("gap_to_break_pct", 10)
    try:
        gap = float(gap)
        prox_mul = 1.4 if gap < 2 else (1.2 if gap < 4 else 1.0)
    except Exception:
        prox_mul = 1.0

    # Track multiplier
    trk_mul = 1.3 if track == TRK_BREAKOUT else (0.8 if track == TRK_BREAKDOWN else 1.0)

    return round(score * vol_mul * time_mul * prox_mul * trk_mul, 1)


def priority_score_hs(pick: dict, track: str) -> float:
    """Priority for H&S picks — quality_score × staleness × track."""
    score    = pick.get("quality_score", 50)
    stale    = 0.6 if pick.get("is_stale") else 1.0
    trk_mul  = 1.3 if track == TRK_BREAKOUT else (0.8 if track == TRK_BREAKDOWN else 1.0)
    # Recent breakouts (Confirmed status) get a boost
    status   = pick.get("status", "")
    conf_mul = 1.5 if status == "Confirmed" else 1.0
    return round(score * stale * trk_mul * conf_mul, 1)


def priority_score_maverick(pick: dict, track: str, eta: Optional[int]) -> float:
    """Priority for main_v3 picks — Confidence% × progress × proximity."""
    conf = pick.get("confidence_pct", 55)

    # Progress multiplier based on track signal
    trk_mul = 1.3 if track == TRK_BREAKOUT else (0.8 if track == TRK_BREAKDOWN else 1.0)

    # Proximity to entry
    gap = pick.get("gap_to_entry_pct", 10)
    try:
        gap = float(gap)
        prox_mul = 1.4 if gap < 2 else (1.2 if gap < 5 else 1.0)
    except Exception:
        prox_mul = 1.0

    # ETA bonus — if we know it's arriving soon
    eta_mul = 1.2 if eta and eta <= 3 else 1.0

    return round(conf * trk_mul * prox_mul * eta_mul, 1)


# ─────────────────────────────────────────────────────────────────────────────
# IMPORTERS
# ─────────────────────────────────────────────────────────────────────────────

def import_maverick(path: str, wl: dict) -> tuple:
    """Import main_v3 Excel report into watchlist."""
    if not os.path.exists(path):
        print(f"  [SKIP] Not found: {path}")
        return 0, 0

    try:
        df = pd.read_excel(path, sheet_name="TOP 10 PICKS")
    except Exception as e:
        print(f"  [ERROR] Cannot read {path}: {e}")
        return 0, 0

    added = skipped = 0
    for _, row in df.iterrows():
        sym = str(row.get("Symbol", "")).strip()
        if not sym:
            continue
        key = _key(sym, "maverick")
        if key in wl["maverick"] and wl["maverick"][key]["state"] == ST_WATCHING:
            skipped += 1
            continue

        entry = float(row.get("Entry") or 0)
        stop  = float(row.get("Stop_Loss") or 0)
        t1    = float(row.get("Target_1") or 0)
        last_close_approx = entry  # we'll update on first check

        gap_pct = 0.0
        if entry > 0 and last_close_approx > 0:
            gap_pct = round((entry - last_close_approx) / last_close_approx * 100, 2)

        wl["maverick"][key] = {
            "symbol":          sym,
            "source":          SRC_MAVERICK,
            "state":           ST_WATCHING,
            "date_added":      datetime.now(IST).strftime("%Y-%m-%d"),
            "confidence_pct":  float(row.get("Confidence_Pct") or 0),
            "setup_type":      str(row.get("Setup_Type", "")),
            "setup_grade":     str(row.get("Setup_Grade", "")),
            "entry":           entry,
            "stop_loss":       stop,
            "target_1":        t1,
            "target_2":        float(row.get("Target_2") or 0),
            "risk_reward":     float(row.get("Reward_Risk") or 0),
            "expected_gain":   float(row.get("Expected_Gain_Pct") or 0),
            "median_days":     row.get("Median_Days"),
            "trend_state":     str(row.get("Trend_State", "")),
            "momentum_state":  str(row.get("Momentum_State", "")),
            "volume_state":    str(row.get("Volume_State", "")),
            "headline":        str(row.get("Headline", "")),
            "gap_to_entry_pct": gap_pct,
            "date_resolved":   None,
            "resolved_note":   None,
            "price_history":   [],
        }
        added += 1

    return added, skipped


def import_patterns(path: str, wl: dict) -> tuple:
    """Import pattern_detector_v2 CSV into watchlist."""
    if not os.path.exists(path):
        print(f"  [SKIP] Not found: {path}")
        return 0, 0

    df = pd.read_csv(path)
    added = skipped = 0

    for _, row in df.iterrows():
        sym     = str(row.get("Symbol", "")).strip()
        pattern = str(row.get("Pattern", "")).strip()
        if not sym or not pattern:
            continue
        key = _key(sym, pattern)
        if key in wl["patterns"] and wl["patterns"][key]["state"] == ST_WATCHING:
            skipped += 1
            continue

        wl["patterns"][key] = {
            "symbol":           sym,
            "pattern":          pattern,
            "source":           SRC_MURPHY,
            "state":            ST_WATCHING,
            "date_added":       datetime.now(IST).strftime("%Y-%m-%d"),
            "score":            float(row.get("Score") or 0),
            "confidence":       str(row.get("Confidence", "")),
            "breakout_level":   float(row.get("Breakout_Level") or 0),
            "stop_loss":        float(row.get("Stop_Loss") or 0),
            "target_1":         float(row.get("Target_1") or 0),
            "risk_reward":      float(row.get("Risk_Reward") or 0),
            "gap_to_break_pct": float(row.get("Gap_To_B_%") or row.get("Gap_To_Break_%") or 0),
            "pole_return_pct":  float(row.get("Pole_Return_%") or 0),
            "vol_all_3_ok":     bool(row.get("Vol_All_3_OK", False)),
            "vol_consol_avg_ok": bool(row.get("Vol_Consol_Avg_OK", False)),
            "vol_trend_decline": bool(row.get("Vol_Trend_Decline", False)),
            "breakout_vol_watch": float(row.get("Breakout_Vol_Watch") or 0),
            "expiry_date":      str(row.get("Expiry_Date", "") or ""),
            "weekly_confirmed": bool(row.get("Weekly_Confirmed", False)),
            "date_resolved":    None,
            "resolved_note":    None,
            "price_history":    [],
        }
        added += 1

    return added, skipped


def import_hs(path: str, wl: dict) -> tuple:
    """Import hs_watchlist.csv into watchlist."""
    if not os.path.exists(path):
        print(f"  [SKIP] Not found: {path}")
        return 0, 0

    df = pd.read_csv(path)
    added = skipped = 0

    for _, row in df.iterrows():
        sym  = str(row.get("Symbol", "")).strip()
        ptype = str(row.get("Pattern Type", "")).strip()
        if not sym or not ptype:
            continue
        status = str(row.get("Status", "")).strip()
        # Only import actionable statuses
        if status in ("Failed", "Invalidated", "False Start"):
            continue
        key = _key(sym, ptype)
        if key in wl["hs"] and wl["hs"][key]["state"] == ST_WATCHING:
            skipped += 1
            continue

        neckline = float(row.get("Neckline Price") or 0)
        stop     = float(row.get("Stop Loss") or 0)
        target   = float(row.get("Target") or 0)
        rr       = float(row.get("Risk:Reward") or 0)
        brk_date = str(row.get("Breakout Date") or "")
        brk_price= float(row.get("Breakout Price") or 0)

        # Map H&S status to tracker state
        if status == "Confirmed":
            state = ST_BREAKOUT
        elif status == "Forming":
            state = ST_WATCHING
        else:
            state = ST_WATCHING

        wl["hs"][key] = {
            "symbol":            sym,
            "pattern_type":      ptype,
            "source":            SRC_HS,
            "state":             state,
            "status":            status,
            "date_added":        datetime.now(IST).strftime("%Y-%m-%d"),
            "watchlist_category": str(row.get("Watchlist Category", "")),
            "quality_score":     float(row.get("Quality Score") or 0),
            "left_shoulder_date":  str(row.get("Left Shoulder Date", "")),
            "left_shoulder_price": float(row.get("Left Shoulder Price") or 0),
            "head_date":           str(row.get("Head Date", "")),
            "head_price":          float(row.get("Head Price") or 0),
            "right_shoulder_date": str(row.get("Right Shoulder Date", "")),
            "right_shoulder_price":float(row.get("Right Shoulder Price") or 0),
            "rs_confirmation":     str(row.get("RS Confirmation", "")),
            "rs_age_days":         int(row.get("RS Age (days)") or 0),
            "is_stale":            str(row.get("Is Stale", "No")).strip().lower() == "yes",
            "neckline_price":      neckline,
            "breakout_date":       brk_date,
            "breakout_price":      brk_price,
            "volume_confirmed":    str(row.get("Volume Confirmed", "No")).strip() == "Yes",
            "target":              target,
            "stop_loss":           stop,
            "risk_reward":         rr,
            "trigger_condition":   str(row.get("Trigger Condition", "")),
            "notes":               str(row.get("Notes", "")),
            "date_resolved":       None,
            "resolved_note":       None,
            "price_history":       [],
        }
        added += 1

    return added, skipped


# ─────────────────────────────────────────────────────────────────────────────
# DAILY CHECK
# ─────────────────────────────────────────────────────────────────────────────

def _eval_maverick(pick: dict, close: float, volume: float,
                   date_str: str, df: pd.DataFrame) -> dict:
    pick = pick.copy()
    entry = pick["entry"]
    stop  = pick["stop_loss"]
    t1    = pick["target_1"]

    gap_pct = round((entry - close) / close * 100, 2) if close > 0 else 0
    pick["price_history"].append({
        "date": date_str, "close": round(close, 2),
        "volume": int(volume), "gap_pct": gap_pct,
    })

    # State checks
    if close >= t1 and t1 > 0:
        pick["state"] = ST_BREAKOUT
        pick["date_resolved"] = date_str
        pick["resolved_note"] = f"TARGET HIT ✓ — closed ₹{close:.2f} at/above Target ₹{t1:.2f}"
    elif close >= entry and entry > 0:
        pick["state"] = ST_BREAKOUT
        pick["date_resolved"] = date_str
        pick["resolved_note"] = f"ENTRY TRIGGERED ✓ — closed ₹{close:.2f} above entry ₹{entry:.2f}"
    elif close <= stop and stop > 0:
        pick["state"] = ST_BREAKDOWN
        pick["date_resolved"] = date_str
        pick["resolved_note"] = f"STOP HIT ✗ — closed ₹{close:.2f} below stop ₹{stop:.2f}"
    else:
        track, detail, eta = maverick_track_signal(pick, df)
        pick["track_signal"] = track
        pick["track_detail"] = detail
        pick["eta_days"]     = eta
        pick["gap_to_entry_pct"] = gap_pct

    return pick


def _eval_pattern(pick: dict, close: float, volume: float,
                  date_str: str, df: pd.DataFrame) -> dict:
    pick = pick.copy()
    brk  = pick["breakout_level"]
    stop = pick["stop_loss"]
    bvw  = pick.get("breakout_vol_watch", 0)
    exp  = pick.get("expiry_date", "")

    gap_pct = round((brk - close) / close * 100, 2) if close > 0 else 0
    pick["price_history"].append({
        "date": date_str, "close": round(close, 2),
        "volume": int(volume), "gap_pct": gap_pct,
    })

    if close >= brk:
        vol_ok = volume >= bvw if bvw > 0 else True
        pick["state"]         = ST_BREAKOUT
        pick["date_resolved"] = date_str
        pick["resolved_note"] = (
            f"BREAKOUT ✓ — closed ₹{close:.2f} above ₹{brk:.2f}. "
            + (f"Vol {volume:,.0f} ✓" if vol_ok else f"⚠ Vol {volume:,.0f} below {bvw:,.0f}")
        )
    elif close <= stop:
        pick["state"]         = ST_BREAKDOWN
        pick["date_resolved"] = date_str
        pick["resolved_note"] = f"BREAKDOWN ✗ — closed ₹{close:.2f} below stop ₹{stop:.2f}"
    elif exp:
        try:
            exp_dt = pd.to_datetime(exp)
            if pd.Timestamp(_ANCHOR) > exp_dt:
                pick["state"]         = ST_EXPIRED
                pick["date_resolved"] = date_str
                pick["resolved_note"] = f"EXPIRED — past Murphy time limit ({_fmt(exp_dt)})"
        except Exception:
            pass
    
    if pick["state"] == ST_WATCHING:
        track, detail, eta = murphy_track_signal(pick, df)
        pick["track_signal"]    = track
        pick["track_detail"]    = detail
        pick["eta_days"]        = eta
        pick["gap_to_break_pct"]= gap_pct
        pick["priority"]        = priority_score_murphy(pick, track, eta)

    return pick


def _eval_hs(pick: dict, close: float, volume: float,
             date_str: str, df: pd.DataFrame) -> dict:
    pick = pick.copy()
    neckline  = pick.get("neckline_price", 0)
    stop      = pick.get("stop_loss", 0)
    target    = pick.get("target", 0)
    is_inverse= pick.get("pattern_type", "") == "Inverse H&S"
    bvw       = volume * 1.5  # approx 1.5× today's vol

    gap_pct = round(abs(close - neckline) / neckline * 100, 2) if neckline > 0 else 0
    pick["price_history"].append({
        "date": date_str, "close": round(close, 2),
        "volume": int(volume), "gap_pct": gap_pct,
    })

    # Breakout check
    if is_inverse and close >= neckline and neckline > 0:
        pick["state"]         = ST_BREAKOUT
        pick["date_resolved"] = date_str
        pick["resolved_note"] = f"BREAKOUT ✓ — Inverse H&S neckline broken at ₹{close:.2f}"
    elif not is_inverse and close <= neckline and neckline > 0:
        pick["state"]         = ST_BREAKOUT
        pick["date_resolved"] = date_str
        pick["resolved_note"] = f"BREAKOUT ✓ — H&S neckline broken (bearish) at ₹{close:.2f}"
    elif stop > 0 and is_inverse and close < stop:
        pick["state"]         = ST_FAILED
        pick["date_resolved"] = date_str
        pick["resolved_note"] = f"FAILED ✗ — closed ₹{close:.2f} below stop ₹{stop:.2f}"
    elif stop > 0 and not is_inverse and close > stop:
        pick["state"]         = ST_FAILED
        pick["date_resolved"] = date_str
        pick["resolved_note"] = f"FAILED ✗ — closed ₹{close:.2f} above stop ₹{stop:.2f}"
    else:
        track, detail, eta = hs_track_signal(pick, df)
        pick["track_signal"] = track
        pick["track_detail"] = detail
        pick["eta_days"]     = eta
        pick["priority"]     = priority_score_hs(pick, track)

    return pick


def check_all(wl: dict) -> dict:
    """Fetch prices for all WATCHING picks and evaluate state + track signal."""
    today    = datetime.now(IST).strftime("%Y-%m-%d")
    alerts   = []
    watching = []

    all_buckets = [
        ("maverick",  wl["maverick"],  _eval_maverick),
        ("patterns",  wl["patterns"],  _eval_pattern),
        ("hs",        wl["hs"],        _eval_hs),
    ]

    for bucket_name, bucket, eval_fn in all_buckets:
        active_keys = [k for k, p in bucket.items() if p["state"] == ST_WATCHING]
        print(f"\n  [{bucket_name}] Checking {len(active_keys)} pick(s)...")

        for key in active_keys:
            pick   = bucket[key]
            sym    = pick["symbol"]
            print(f"    {sym:<20}", end=" ")

            df_ohlcv = fetch_ohlcv(sym, days=30)
            close, volume, date_str = _last_bar(df_ohlcv)

            if close is None:
                print("→ [DATA ERROR]")
                continue

            updated = eval_fn(pick, close, volume, date_str, df_ohlcv)
            bucket[key] = updated
            state = updated["state"]
            track = updated.get("track_signal", "")
            print(f"→ ₹{close:.2f}  {state}  {track}")

            if state != ST_WATCHING:
                alerts.append(updated)
            else:
                updated["priority"] = updated.get("priority", 0)
                watching.append(updated)

    save_wl(wl)
    return {"date": today, "alerts": alerts, "watching": watching, "wl": wl}


# ─────────────────────────────────────────────────────────────────────────────
# TERMINAL REPORT
# ─────────────────────────────────────────────────────────────────────────────

def print_report(summary: dict):
    SEP = "═" * 72
    print(f"\n{SEP}")
    print(f"  MaverickPICKS UNIFIED TRACKER — {summary['date']}")
    print(f"  Alerts: {len(summary['alerts'])}  |  Watching: {len(summary['watching'])}")
    print(SEP)

    if summary["alerts"]:
        print("\n  🚨  ALERTS — ACT ON THESE")
        print("  " + "─"*70)
        for p in summary["alerts"]:
            sym = p.get("symbol", "")
            src = p.get("source", "")
            note = p.get("resolved_note", "")
            print(f"  {sym:<18} [{src}]")
            print(f"    {note}")

    if summary["watching"]:
        # Sort by priority descending
        by_src = {}
        for p in summary["watching"]:
            by_src.setdefault(p.get("source", "Other"), []).append(p)

        for src, picks in by_src.items():
            picks_sorted = sorted(picks, key=lambda x: x.get("priority", 0), reverse=True)
            print(f"\n  👁  {src.upper()} — WATCHING ({len(picks)})")
            print("  " + "─"*70)
            for p in picks_sorted:
                sym   = p.get("symbol", "")
                track = p.get("track_signal", TRK_NEUTRAL)
                eta   = p.get("eta_days")
                pri   = p.get("priority", 0)
                detail= p.get("track_detail", "")
                eta_str = f"  ETA ~{eta}d" if eta else ""
                print(f"  {sym:<18} Pri:{pri:>6.1f}  {track}{eta_str}")
                if detail:
                    print(f"    {detail}")

    print(f"\n{SEP}\n")


# ─────────────────────────────────────────────────────────────────────────────
# DASHBOARD GENERATORS
# ─────────────────────────────────────────────────────────────────────────────

def _track_badge(track: str) -> str:
    cfg = {
        TRK_BREAKOUT:  ("#166534", "#dcfce7", "🟢"),
        TRK_BREAKDOWN: ("#991b1b", "#fee2e2", "🔴"),
        TRK_NEUTRAL:   ("#374151", "#f3f4f6", "⚪"),
        TRK_STALE:     ("#6b7280", "#f9fafb", "⚫"),
    }.get(track, ("#374151", "#f3f4f6", "—"))
    return (f'<span style="background:{cfg[1]};color:{cfg[0]};padding:2px 9px;'
            f'border-radius:10px;font-size:11px;font-weight:600">{cfg[2]} {track}</span>')


def _state_badge(state: str) -> str:
    cfg = {
        ST_BREAKOUT:  ("#166534", "#dcfce7", "🟢 BREAKOUT"),
        ST_BREAKDOWN: ("#991b1b", "#fee2e2", "🔴 BREAKDOWN"),
        ST_FAILED:    ("#991b1b", "#fee2e2", "🔴 FAILED"),
        ST_EXPIRED:   ("#6b7280", "#f3f4f6", "⚫ EXPIRED"),
        ST_WATCHING:  ("#1e40af", "#dbeafe", "👁 WATCHING"),
    }.get(state, ("#374151", "#f3f4f6", state))
    return (f'<span style="background:{cfg[1]};color:{cfg[0]};padding:2px 9px;'
            f'border-radius:10px;font-size:11px;font-weight:600">{cfg[2]}</span>')


def _gap_bar(gap: float, reverse: bool = False) -> str:
    """Visual bar showing how close price is to trigger level."""
    try:
        gap = float(gap)
        pct_fill = max(0, min(100, int((1 - gap / 20) * 100)))
        color = "#22c55e" if gap < 2 else ("#f59e0b" if gap < 5 else "#94a3b8")
        return (f'<div style="display:flex;align-items:center;gap:6px">'
                f'<div style="width:70px;height:7px;background:#e5e7eb;border-radius:4px">'
                f'<div style="width:{pct_fill}%;height:7px;background:{color};border-radius:4px"></div>'
                f'</div><span style="font-size:11px;color:#6b7280">{gap:.1f}%</span></div>')
    except Exception:
        return "—"


def _sparkline(history: list) -> str:
    if not history or len(history) < 2:
        return "<span style='color:#9ca3af'>—</span>"
    closes = [h["close"] for h in history[-10:]]
    mn, mx = min(closes), max(closes)
    rng    = mx - mn if mx > mn else 1
    W, H   = 70, 22
    pts    = []
    for i, c in enumerate(closes):
        x = int(i / (len(closes)-1) * W)
        y = H - int((c - mn) / rng * H)
        pts.append(f"{x},{y}")
    col = "#22c55e" if closes[-1] >= closes[0] else "#ef4444"
    return (f'<svg width="{W}" height="{H}">'
            f'<polyline points="{" ".join(pts)}" fill="none" stroke="{col}" stroke-width="1.5"/>'
            f'</svg>')


_DASH_CSS = """
<style>
* { box-sizing:border-box; margin:0; padding:0 }
body { font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",Arial,sans-serif;
       background:#f1f5f9; color:#1e293b; }
.hdr { background:linear-gradient(135deg,#0f172a,#1d4ed8);
       color:#fff; padding:20px 28px; }
.hdr h1 { font-size:20px; font-weight:700 }
.hdr p  { font-size:12px; opacity:.75; margin-top:3px }
.stats { display:flex; gap:12px; padding:16px 28px; flex-wrap:wrap }
.stat { background:#fff; border:1px solid #e2e8f0; border-radius:10px;
        padding:12px 18px; min-width:110px;
        box-shadow:0 1px 2px rgba(0,0,0,.05) }
.stat .n { font-size:26px; font-weight:700 }
.stat .l { font-size:11px; color:#64748b; margin-top:2px }
.stat.g { border-left:4px solid #22c55e }
.stat.r { border-left:4px solid #ef4444 }
.stat.b { border-left:4px solid #3b82f6 }
.stat.s { border-left:4px solid #94a3b8 }
.section { padding:0 28px 24px }
.sec-title { font-size:13px; font-weight:600; color:#475569;
             letter-spacing:.06em; text-transform:uppercase;
             margin:16px 0 8px; border-left:3px solid #3b82f6;
             padding-left:10px }
table { width:100%; border-collapse:collapse; background:#fff;
        border-radius:10px; overflow:hidden;
        box-shadow:0 1px 2px rgba(0,0,0,.06); font-size:12px }
thead tr { background:#0f172a; color:#fff }
thead th { padding:9px 10px; font-size:11px; font-weight:600;
           text-transform:uppercase; letter-spacing:.04em;
           white-space:nowrap; text-align:left }
tbody tr:hover { background:#f8fafc!important }
tbody td { padding:9px 10px; border-bottom:1px solid #f1f5f9;
           vertical-align:middle }
.pri { font-weight:700; color:#1d4ed8; font-size:13px }
.sym { font-weight:600; font-size:13px }
.note { font-size:11px; color:#64748b; padding:10px 28px 20px }
.empty { text-align:center; padding:40px; color:#94a3b8; font-size:13px }
</style>
"""


def generate_maverick_dashboard(wl: dict):
    picks = list(wl["maverick"].values())
    now   = datetime.now(IST).strftime("%d-%b-%Y %H:%M IST")

    n_alert   = sum(1 for p in picks if p["state"] != ST_WATCHING)
    n_watch   = sum(1 for p in picks if p["state"] == ST_WATCHING)
    n_breakout= sum(1 for p in picks if p["state"] == ST_BREAKOUT)
    n_breakdown=sum(1 for p in picks if p["state"] == ST_BREAKDOWN)

    # Sort watching by priority desc
    watching = sorted(
        [p for p in picks if p["state"] == ST_WATCHING],
        key=lambda x: x.get("priority", 0), reverse=True
    )
    resolved = [p for p in picks if p["state"] != ST_WATCHING]

    def row_html(p, bg="#ffffff"):
        h         = p.get("price_history", [])
        last_h    = h[-1] if h else {}
        close     = last_h.get("close", "—")
        gap       = last_h.get("gap_pct", p.get("gap_to_entry_pct", 0))
        track     = p.get("track_signal", TRK_NEUTRAL)
        eta       = p.get("eta_days")
        pri       = p.get("priority", 0)
        detail    = p.get("track_detail", "")
        eta_str   = f"~{eta}d" if eta else "—"
        close_str = f"₹{close:,.2f}" if isinstance(close, (int, float)) else "—"
        conf_col  = {"HIGH": "#166534", "MEDIUM": "#92400e"}.get(
                    "HIGH" if p.get("confidence_pct", 0) >= 70 else "MEDIUM", "#6b7280")

        return f"""<tr style="background:{bg}">
          <td><span class="sym">{p['symbol']}</span></td>
          <td>{p.get('setup_type','—')}</td>
          <td>{p.get('setup_grade','—')}</td>
          <td style="color:{conf_col};font-weight:600">{p.get('confidence_pct',0):.0f}%</td>
          <td>{_state_badge(p['state'])}</td>
          <td>{_track_badge(track)}</td>
          <td style="font-size:11px;color:#475569;max-width:180px">{detail[:80]}</td>
          <td style="text-align:center;color:#64748b">{eta_str}</td>
          <td style="text-align:right">{close_str}</td>
          <td>{_gap_bar(gap)}</td>
          <td style="text-align:right;color:#1d4ed8;font-weight:600">₹{p.get('entry',0):,.2f}</td>
          <td style="text-align:right;color:#dc2626">₹{p.get('stop_loss',0):,.2f}</td>
          <td style="text-align:right;color:#16a34a">₹{p.get('target_1',0):,.2f}</td>
          <td style="text-align:right">{p.get('risk_reward',0):.1f}x</td>
          <td style="text-align:center">{len(h)}d</td>
          <td>{_sparkline(h)}</td>
          <td style="font-size:11px;color:#475569;max-width:200px">{p.get('headline','')[:80]}</td>
        </tr>"""

    rows_watching  = "".join(row_html(p) for p in watching) or \
                     '<tr><td colspan="17" class="empty">No active picks</td></tr>'
    rows_resolved  = "".join(
        row_html(p, "#fff7ed" if p["state"] == ST_BREAKOUT else "#fff1f2")
        for p in resolved
    ) or '<tr><td colspan="17" class="empty">No resolved picks yet</td></tr>'

    th = lambda s: f'<th>{s}</th>'
    headers = "".join(th(h) for h in [
        "Symbol","Setup","Grade","Conf","State","Track Signal","Signal Detail",
        "ETA","Last Close","Gap to Entry","Entry","Stop","Target","R:R","Days","Trend","Headline"
    ])

    html = f"""<!DOCTYPE html><html lang="en"><head>
<meta charset="UTF-8"><title>MaverickPICKS Dashboard</title>{_DASH_CSS}</head><body>
<div class="hdr">
  <h1>MaverickPICKS — Main v3 Dashboard</h1>
  <p>Updated: {now} &nbsp;|&nbsp; Evidence-based momentum + fingerprint picks</p>
</div>
<div class="stats">
  <div class="stat g"><div class="n">{n_breakout}</div><div class="l">🟢 Entry Hit</div></div>
  <div class="stat r"><div class="n">{n_breakdown}</div><div class="l">🔴 Stop Hit</div></div>
  <div class="stat b"><div class="n">{n_watch}</div><div class="l">👁 Watching</div></div>
</div>
<div class="section">
  <div class="sec-title">👁 Watching — sorted by priority (highest first)</div>
  <table><thead><tr>{headers}</tr></thead><tbody>{rows_watching}</tbody></table>
  <div class="sec-title">✅ Resolved picks</div>
  <table><thead><tr>{headers}</tr></thead><tbody>{rows_resolved}</tbody></table>
</div>
<p class="note">
  Priority = Confidence% × track signal × proximity to entry × ETA bonus.
  Higher priority = act sooner. Green track = all signals pointing to entry.
  Red track = setup weakening, watch carefully.
</p></body></html>"""

    with open(DASH_MAVERICK, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"  Dashboard → {DASH_MAVERICK}")


def generate_patterns_dashboard(wl: dict):
    murphy_picks = list(wl["patterns"].values())
    hs_picks     = list(wl["hs"].values())
    all_picks    = murphy_picks + hs_picks
    now          = datetime.now(IST).strftime("%d-%b-%Y %H:%M IST")

    n_breakout  = sum(1 for p in all_picks if p["state"] in (ST_BREAKOUT, ST_FAILED))
    n_watching  = sum(1 for p in all_picks if p["state"] == ST_WATCHING)
    n_expired   = sum(1 for p in all_picks if p["state"] in (ST_EXPIRED, ST_BREAKDOWN))

    watching_murphy = sorted(
        [p for p in murphy_picks if p["state"] == ST_WATCHING],
        key=lambda x: x.get("priority", 0), reverse=True
    )
    watching_hs = sorted(
        [p for p in hs_picks if p["state"] == ST_WATCHING],
        key=lambda x: x.get("priority", 0), reverse=True
    )
    resolved = [p for p in all_picks if p["state"] != ST_WATCHING]

    def murphy_row(p, bg="#ffffff"):
        h      = p.get("price_history", [])
        last_h = h[-1] if h else {}
        close  = last_h.get("close", "—")
        gap    = last_h.get("gap_pct", p.get("gap_to_break_pct", 0))
        track  = p.get("track_signal", TRK_NEUTRAL)
        eta    = p.get("eta_days")
        pri    = p.get("priority", 0)
        detail = p.get("track_detail", "")
        close_s= f"₹{close:,.2f}" if isinstance(close, (int, float)) else "—"
        vol_ok = "✓" if p.get("vol_all_3_ok") else "✗"
        wkly   = "✓" if p.get("weekly_confirmed") else "✗"
        exp    = p.get("expiry_date", "—")
        exp_col = "#dc2626" if exp != "—" and (
            pd.to_datetime(exp) - pd.Timestamp(_ANCHOR)).days <= 2 else "#475569"

        return f"""<tr style="background:{bg}">
          <td><span class="sym">{p['symbol'].replace('.NS','')}</span></td>
          <td style="font-size:11px;color:#475569">{p.get('pattern','')}</td>
          <td><span class="pri">{pri:.0f}</span></td>
          <td>{_state_badge(p['state'])}</td>
          <td>{_track_badge(track)}</td>
          <td style="font-size:11px;color:#475569;max-width:160px">{detail[:70]}</td>
          <td style="text-align:center;color:#64748b">{'~'+str(eta)+'d' if eta else '—'}</td>
          <td style="text-align:right">{close_s}</td>
          <td>{_gap_bar(gap)}</td>
          <td style="text-align:right;color:#1d4ed8;font-weight:600">₹{p.get('breakout_level',0):,.2f}</td>
          <td style="text-align:right;color:#dc2626">₹{p.get('stop_loss',0):,.2f}</td>
          <td style="text-align:right;color:#16a34a">₹{p.get('target_1',0):,.2f}</td>
          <td style="text-align:right">{p.get('risk_reward',0):.1f}x</td>
          <td style="text-align:center">{p.get('score',0):.0f}</td>
          <td style="text-align:center">{vol_ok}</td>
          <td style="text-align:center">{wkly}</td>
          <td style="text-align:center;color:{exp_col};font-size:11px">{exp}</td>
          <td style="text-align:center">{len(h)}d</td>
          <td>{_sparkline(h)}</td>
        </tr>"""

    def hs_row(p, bg="#ffffff"):
        h      = p.get("price_history", [])
        last_h = h[-1] if h else {}
        close  = last_h.get("close", "—")
        gap    = last_h.get("gap_pct", 0)
        track  = p.get("track_signal", TRK_NEUTRAL)
        eta    = p.get("eta_days")
        pri    = p.get("priority", 0)
        detail = p.get("track_detail", "")
        stale  = "⚠ STALE" if p.get("is_stale") else ""
        close_s= f"₹{close:,.2f}" if isinstance(close, (int, float)) else "—"

        return f"""<tr style="background:{bg}">
          <td><span class="sym">{p['symbol']}</span></td>
          <td style="font-size:11px;color:#475569">{p.get('pattern_type','')}</td>
          <td><span class="pri">{pri:.0f}</span></td>
          <td>{_state_badge(p['state'])}</td>
          <td>{_track_badge(track)}</td>
          <td style="font-size:11px;color:#475569;max-width:160px">{detail[:70]}</td>
          <td style="text-align:center;color:#64748b">{'~'+str(eta)+'d' if eta else '—'}</td>
          <td style="text-align:right">{close_s}</td>
          <td>{_gap_bar(gap)}</td>
          <td style="text-align:right;color:#1d4ed8;font-weight:600">₹{p.get('neckline_price',0):,.2f}</td>
          <td style="text-align:right;color:#dc2626">₹{p.get('stop_loss',0):,.2f}</td>
          <td style="text-align:right;color:#16a34a">₹{p.get('target',0):,.2f}</td>
          <td style="text-align:right">{p.get('risk_reward',0):.1f}x</td>
          <td style="text-align:center">{p.get('quality_score',0):.0f}</td>
          <td style="text-align:center;font-size:11px">{p.get('rs_confirmation','')}</td>
          <td style="text-align:center;font-size:11px;color:#dc2626">{stale}</td>
          <td style="font-size:10px;color:#6b7280">{p.get('watchlist_category','')[:30]}</td>
          <td style="text-align:center">{len(h)}d</td>
          <td>{_sparkline(h)}</td>
        </tr>"""

    m_th = lambda s: f'<th>{s}</th>'
    m_headers = "".join(m_th(h) for h in [
        "Symbol","Pattern","Priority","State","Track","Signal Detail",
        "ETA","Last Close","Gap to Break","Breakout","Stop","Target",
        "R:R","Score","Vol✓","Wkly✓","Expiry","Days","Trend"
    ])
    hs_headers = "".join(m_th(h) for h in [
        "Symbol","Type","Priority","State","Track","Signal Detail",
        "ETA","Last Close","Gap","Neckline","Stop","Target",
        "R:R","Quality","RS Conf","Stale","Category","Days","Trend"
    ])

    rows_murphy = "".join(murphy_row(p) for p in watching_murphy) or \
                  '<tr><td colspan="19" class="empty">No Murphy pattern picks</td></tr>'
    rows_hs     = "".join(hs_row(p) for p in watching_hs) or \
                  '<tr><td colspan="19" class="empty">No H&S picks</td></tr>'
    rows_res    = "".join(
        (murphy_row(p, "#f0fdf4") if p.get("source") == SRC_MURPHY
         else hs_row(p, "#f0fdf4"))
        for p in resolved
    ) or '<tr><td colspan="19" class="empty">No resolved picks yet</td></tr>'

    html = f"""<!DOCTYPE html><html lang="en"><head>
<meta charset="UTF-8"><title>MaverickPICKS Patterns</title>{_DASH_CSS}</head><body>
<div class="hdr">
  <h1>MaverickPICKS — Patterns Dashboard</h1>
  <p>Updated: {now} &nbsp;|&nbsp;
     Murphy (Bull Flag / Pennant / Triangles) + Head & Shoulders</p>
</div>
<div class="stats">
  <div class="stat g"><div class="n">{n_breakout}</div><div class="l">🟢 Breakouts</div></div>
  <div class="stat b"><div class="n">{n_watching}</div><div class="l">👁 Watching</div></div>
  <div class="stat s"><div class="n">{n_expired}</div><div class="l">⚫ Expired/Failed</div></div>
</div>
<div class="section">
  <div class="sec-title">📐 Murphy Patterns — sorted by priority</div>
  <table><thead><tr>{m_headers}</tr></thead><tbody>{rows_murphy}</tbody></table>
  <div class="sec-title">🔺 H&S Patterns — sorted by priority</div>
  <table><thead><tr>{hs_headers}</tr></thead><tbody>{rows_hs}</tbody></table>
  <div class="sec-title">✅ Resolved picks</div>
  <table><thead><tr>{m_headers}</tr></thead><tbody>{rows_res}</tbody></table>
</div>
<p class="note">
  Priority = Pattern score × volume confirmation × time window × proximity to trigger.
  Green track signal = volume contracting, gap closing, price recovering → breakout likely.
  Red track signal = volume not contracting or price approaching stop → watch carefully.
  ETA = estimated trading days to reach trigger at current velocity.
</p></body></html>"""

    with open(DASH_PATTERNS, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"  Dashboard → {DASH_PATTERNS}")


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(
        description="MaverickPICKS Unified Tracker"
    )
    ap.add_argument("--import_maverick", type=str, default="",
                    help="main_v3 Excel report to import")
    ap.add_argument("--import_patterns", type=str, default="",
                    help="pattern_detector CSV to import")
    ap.add_argument("--import_hs", type=str, default="",
                    help="hs_watchlist.csv to import")
    ap.add_argument("--check", action="store_true",
                    help="Fetch prices and evaluate all watching picks")
    ap.add_argument("--status", action="store_true",
                    help="Print watchlist status and regenerate dashboards")
    args = ap.parse_args()

    print(f"\n{'═'*60}")
    print("  MaverickPICKS Unified Tracker")
    print(f"  {datetime.now(IST).strftime('%d-%b-%Y  %H:%M IST')}")
    print(f"  Anchor date: {_fmt(_ANCHOR)}")
    print(f"{'═'*60}")

    wl = load_wl()

    if args.import_maverick:
        print(f"\n  Importing main_v3 picks from: {args.import_maverick}")
        a, s = import_maverick(args.import_maverick, wl)
        print(f"  Added: {a}  |  Skipped: {s}")
        save_wl(wl)

    if args.import_patterns:
        print(f"\n  Importing Murphy patterns from: {args.import_patterns}")
        a, s = import_patterns(args.import_patterns, wl)
        print(f"  Added: {a}  |  Skipped: {s}")
        save_wl(wl)

    if args.import_hs:
        print(f"\n  Importing H&S picks from: {args.import_hs}")
        a, s = import_hs(args.import_hs, wl)
        print(f"  Added: {a}  |  Skipped: {s}")
        save_wl(wl)

    if args.check:
        summary = check_all(wl)
        print_report(summary)
        wl = load_wl()
        generate_maverick_dashboard(wl)
        generate_patterns_dashboard(wl)

    if args.status:
        wl = load_wl()
        total = sum(
            len(wl[b]) for b in ("maverick", "patterns", "hs")
        )
        watching = sum(
            1 for b in ("maverick", "patterns", "hs")
            for p in wl[b].values() if p["state"] == ST_WATCHING
        )
        print(f"\n  Total picks: {total}  |  Watching: {watching}")
        generate_maverick_dashboard(wl)
        generate_patterns_dashboard(wl)
        print("  Dashboards regenerated.")

    if not any([args.import_maverick, args.import_patterns,
                args.import_hs, args.check, args.status]):
        ap.print_help()


if __name__ == "__main__":
    main()

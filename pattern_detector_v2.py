"""
pattern_detector.py — MaverickPICKS Pattern Detection Engine v2.0
==================================================================
Detects: Bull Flag | Pennant | Symmetrical Triangle | Ascending Triangle

What changed from v1.0 → v2.0:
  1. IN-FORMATION ONLY  — rejects patterns where price has already broken out
  2. POLE DURATION CAP  — pole must form in ≤15 bars (Murphy: near-vertical)
  3. VOLUME (3 checks)  — surge on pole | declining TREND in consolidation
                          (linear regression slope, not just avg) | breakout
                          threshold printed so you know what to watch for
  4. WEEKLY CONFIRMATION— pulls weekly candles and confirms trend + vol on
                          the weekly chart before awarding HIGH confidence
                          (disable with --no-weekly)

Principles: John Murphy — "Technical Analysis of the Financial Markets"

Author : MaverickPICKS
Version: 2.0
Usage  : python pattern_detector.py --symbols RELIANCE.NS TCS.NS --lookback 90
         python pattern_detector.py --symbols RELIANCE.NS --no-weekly --csv out.csv
"""

import argparse
import warnings
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Optional, Tuple

import numpy as np
import pandas as pd
import yfinance as yf
from scipy.stats import linregress

warnings.filterwarnings("ignore")


# ─────────────────────────────────────────────────────────────────────────────
# MURPHY PRINCIPLES — v2.0 implementation notes
# ─────────────────────────────────────────────────────────────────────────────
# POLE
#   • Must be sharp and near-vertical: ≥8% gain in 5–15 bars (MAX 15)
#   • Volume must surge: pole avg ≥ 1.5× pre-pole 20-bar avg
#
# CONSOLIDATION VOLUME (3-stage check)
#   Stage 1 — Average level : consol avg < 65% of pole avg
#   Stage 2 — Declining trend: linear regression slope across consol bars < 0
#              AND last 3 bars avg < first 3 bars avg by ≥20%
#   Stage 3 — Breakout watch : flag expected breakout vol = 1.5× 20d avg
#              (printed as a watch level — cannot measure before breakout)
#
# IN-FORMATION CHECK
#   After all structural checks pass, verify:
#   • Last close is BELOW the pattern's breakout level (not yet broken out)
#   • Consolidation is still "active" (price inside the pattern boundaries)
#
# WEEKLY CONFIRMATION (Murphy: always confirm daily signal on weekly chart)
#   • Weekly EMA20 slope must be positive (uptrend intact)
#   • Last week's volume below 8-week average (weekly consolidation visible)
#   • Only applied to upgrade confidence to HIGH; doesn't disqualify patterns
# ─────────────────────────────────────────────────────────────────────────────


# ─────────────────────────────────────────────────────────────────────────────
# DATA STRUCTURES
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class VolumeProfile:
    """Holds all three Murphy volume checks."""
    pole_surge: bool           # pole avg >= 1.5x pre-pole avg
    pole_vs_prepole_ratio: float
    consol_avg_ok: bool        # consol avg < 65% of pole avg
    consol_avg_ratio: float    # consol avg / pole avg
    consol_trend_declining: bool   # regression slope < 0 AND late < early
    consol_slope_per_day: float    # raw shares/day change
    consol_early_avg: float    # first-3-bar avg in consolidation
    consol_late_avg: float     # last-3-bar avg in consolidation
    breakout_vol_threshold: float  # 1.5 x 20d avg — watch level
    all_confirmed: bool        # all three stages pass

    def summary(self) -> str:
        parts = []
        parts.append(f"Pole surge: {'✓' if self.pole_surge else '✗'} ({self.pole_vs_prepole_ratio:.1f}x pre-pole)")
        parts.append(f"Consol avg: {'✓' if self.consol_avg_ok else '✗'} ({self.consol_avg_ratio*100:.0f}% of pole avg)")
        parts.append(f"Vol trend: {'✓ declining' if self.consol_trend_declining else '✗ not declining'} (slope {self.consol_slope_per_day:+,.0f}/day)")
        parts.append(f"Breakout watch: >{self.breakout_vol_threshold:,.0f} shares")
        return " | ".join(parts)


@dataclass
class WeeklyConfirmation:
    available: bool
    trend_up: bool         # weekly EMA20 slope positive
    vol_contracting: bool  # last week vol < 8-week avg
    ema20w_slope_pct: float
    last_week_vs_8w: float  # ratio

    def confirmed(self) -> bool:
        return self.available and self.trend_up and self.vol_contracting

    def summary(self) -> str:
        if not self.available:
            return "Weekly data unavailable"
        return (f"Weekly trend: {'✓ up' if self.trend_up else '✗ flat/down'} "
                f"(EMA20W slope {self.ema20w_slope_pct:+.2f}%/wk) | "
                f"Weekly vol: {'✓ contracting' if self.vol_contracting else '✗ expanding'} "
                f"({self.last_week_vs_8w:.2f}x 8w avg)")


@dataclass
class PatternResult:
    symbol: str
    pattern: str
    detected: bool
    in_formation: bool         # NEW v2.0 — price hasn't broken out yet
    score: float
    confidence: str
    breakout_level: Optional[float] = None
    stop_loss: Optional[float] = None
    target: Optional[float] = None
    risk_reward: float = 0.0
    gap_to_breakout_pct: float = 0.0
    pole_return_pct: float = 0.0
    pole_bars: int = 0
    consolidation_bars: int = 0
    volume_profile: Optional[VolumeProfile] = None
    weekly: Optional[WeeklyConfirmation] = None
    notes: list = field(default_factory=list)
    narrative: str = ""        # human-readable story with dates and prices

    def to_dict(self) -> dict:
        vp = self.volume_profile
        wk = self.weekly
        return {
            "Symbol":             self.symbol,
            "Pattern":            self.pattern,
            "Detected":           self.detected,
            "In_Formation":       self.in_formation,
            "Score":              round(self.score, 1),
            "Confidence":         self.confidence,
            "Breakout_Level":     round(self.breakout_level, 2) if self.breakout_level else None,
            "Stop_Loss":          round(self.stop_loss, 2) if self.stop_loss else None,
            "Target_1":           round(self.target, 2) if self.target else None,
            "Risk_Reward":        round(self.risk_reward, 2),
            "Gap_To_Break_%":     round(self.gap_to_breakout_pct, 2),
            "Pole_Return_%":      round(self.pole_return_pct, 1),
            "Pole_Bars":          self.pole_bars,
            "Consol_Bars":        self.consolidation_bars,
            "Vol_Pole_Surge":     vp.pole_surge if vp else None,
            "Vol_Consol_Avg_OK":  vp.consol_avg_ok if vp else None,
            "Vol_Trend_Decline":  vp.consol_trend_declining if vp else None,
            "Vol_All_3_OK":       vp.all_confirmed if vp else None,
            "Breakout_Vol_Watch": round(vp.breakout_vol_threshold) if vp else None,
            "Weekly_Confirmed":   wk.confirmed() if wk else None,
            "Notes":              " | ".join(self.notes),
            "Narrative":          self.narrative,
        }


# ─────────────────────────────────────────────────────────────────────────────
# DATA FETCHING
# ─────────────────────────────────────────────────────────────────────────────

def _flatten_cols(df: pd.DataFrame) -> pd.DataFrame:
    df.columns = [c[0] if isinstance(c, tuple) else c for c in df.columns]
    return df


def _last_trading_day() -> datetime:
    """
    Returns the most recent confirmed NSE trading day as of now.
    On weekdays after 3:30pm IST → today.
    On weekdays before 3:30pm IST → yesterday.
    On Saturday → Friday.
    On Sunday → Friday.
    This anchors all downloads to the same fixed date regardless of
    when during the weekend/week the script is run — eliminating the
    yfinance `period=` boundary inconsistency that caused different
    results across runs on the same day.
    """
    from datetime import timezone, timedelta as _td
    IST = timezone(_td(hours=5, minutes=30))
    now = datetime.now(IST)

    # Roll back to Friday if weekend
    d = now.date()
    if d.weekday() == 5:   # Saturday
        d = d - timedelta(days=1)
    elif d.weekday() == 6: # Sunday
        d = d - timedelta(days=2)
    else:
        # Weekday: if before market close use previous day
        market_close = now.replace(hour=15, minute=30, second=0, microsecond=0)
        if now < market_close:
            d = d - timedelta(days=1)
            # Roll back further if that lands on weekend
            if d.weekday() == 6:
                d = d - timedelta(days=2)
            elif d.weekday() == 5:
                d = d - timedelta(days=1)

    return datetime(d.year, d.month, d.day)


# Compute once at import time so every download in a session uses the same anchor
_ANCHOR_DATE = _last_trading_day()


def _download(ticker: str, years: float, interval: str,
              retries: int = 3) -> pd.DataFrame:
    """
    Downloads OHLCV using explicit start/end dates anchored to the last
    confirmed trading day — NOT a relative period= string.

    Why: yfinance's period='2y' resolves relative to "now" at call time.
    On weekends or near market open/close, successive calls can return
    different row counts as yfinance's boundary calculation shifts. This
    produced different scan results across runs on the same day.

    Fix: use start = anchor - N years, end = anchor + 1 day (to include
    the anchor date). The anchor is computed once per session in _ANCHOR_DATE
    so every symbol in a batch scan uses the identical date window.

    Settings that match MaverickPICKS data_loader.py:
      auto_adjust=False  — raw prices = what TradingView shows
      threads=False      — avoids NSE threading issues
    """
    end   = _ANCHOR_DATE + timedelta(days=1)
    start = _ANCHOR_DATE - timedelta(days=int(years * 365.25))

    for attempt in range(retries):
        try:
            df = yf.download(
                ticker,
                start=start,
                end=end,
                interval=interval,
                auto_adjust=False,
                progress=False,
                threads=False,
            )
            df = _flatten_cols(df)
            df = df.dropna(how="all")
            cols = [c for c in ["Open", "High", "Low", "Close", "Volume"]
                    if c in df.columns]
            if len(cols) < 5 or len(df) < 20:
                raise ValueError(f"Insufficient data: {len(df)} rows")
            df = df[cols]
            df.index = pd.to_datetime(df.index)
            return df
        except Exception:
            time.sleep(1)
    return pd.DataFrame()


def fetch_daily(symbol: str, lookback_days: int = 120) -> Optional[pd.DataFrame]:
    """
    Fetches 2 years of daily data anchored to last confirmed trading day.
    Returns last `lookback_days` rows.
    """
    df = _download(symbol, years=2.0, interval="1d")
    if df.empty or len(df) < 40:
        return None
    return df.tail(lookback_days)


def fetch_weekly(symbol: str, weeks: int = 52) -> Optional[pd.DataFrame]:
    """
    Fetches 5 years of weekly data anchored to last confirmed trading day.
    Returns last `weeks` rows.
    """
    df = _download(symbol, years=5.0, interval="1wk")
    if df.empty or len(df) < 10:
        return None
    return df.tail(weeks)


# ─────────────────────────────────────────────────────────────────────────────
# SHARED UTILITIES
# ─────────────────────────────────────────────────────────────────────────────

def _slope_pct_per_bar(series: pd.Series) -> float:
    """Linear regression slope normalised as % of mean per bar."""
    if len(series) < 2:
        return 0.0
    x = np.arange(len(series))
    slope, *_ = linregress(x, series.values.astype(float))
    return (slope / series.mean()) * 100 if series.mean() != 0 else 0.0


def _raw_slope(series: pd.Series) -> float:
    """Absolute linear regression slope (shares/day for volume)."""
    if len(series) < 2:
        return 0.0
    x = np.arange(len(series))
    slope, *_ = linregress(x, series.values.astype(float))
    return slope


def _swing_highs(high: pd.Series, order: int = 3) -> pd.Series:
    result = pd.Series(False, index=high.index)
    for i in range(order, len(high) - order):
        if high.iloc[i] == high.iloc[i - order: i + order + 1].max():
            result.iloc[i] = True
    return result


def _swing_lows(low: pd.Series, order: int = 3) -> pd.Series:
    result = pd.Series(False, index=low.index)
    for i in range(order, len(low) - order):
        if low.iloc[i] == low.iloc[i - order: i + order + 1].min():
            result.iloc[i] = True
    return result



# ─────────────────────────────────────────────────────────────────────────────
# NARRATIVE GENERATOR — date-stamped story for human chart verification
# ─────────────────────────────────────────────────────────────────────────────

def _fmt_date(ts) -> str:
    """Format a pandas Timestamp as DD-Mon-YYYY."""
    return pd.Timestamp(ts).strftime("%d-%b-%Y")



def _pattern_expiry(pattern: str,
                    pole_df,
                    consol_df,
                    actual_consol_bars: int = None,
                    swing_highs_dates: list = None,
                    swing_lows_dates:  list = None) -> tuple:
    """
    Calculate the date by which the pattern must break out, or it is invalid.
    Returns (expiry_date_str, days_remaining, is_expiring_soon, reason_str).

    actual_consol_bars: the true number of flag/pennant bars (the `cl` loop
    variable in each detector). Do NOT use len(consol_df) — that slice
    includes the full lookback window, not just the consolidation portion.

    Murphy's rules:
      Bull Flag / Pennant : breakout must occur within 2/3 of pole duration
                            counted from consolidation start.
                            e.g. pole = 10 bars → max flag = 6-7 bars

      Sym / Asc Triangle  : breakout should happen before 3/4 of the
                            distance from triangle start to apex.
    """
    import math
    from datetime import timedelta

    D = _fmt_date
    today = pd.Timestamp.today().normalize()

    if pattern in ("Bull Flag", "Pennant"):
        if pole_df is None or len(pole_df) == 0 or consol_df is None:
            return None, None, False, ""

        pole_bars = len(pole_df)

        # Use actual_consol_bars if provided — this is the true flag length
        # (the `cl` variable from the detector loop).
        # Fallback to len(consol_df) only if not provided.
        bars_used = actual_consol_bars if actual_consol_bars is not None else len(consol_df)

        # Murphy: flag/pennant should not exceed 2/3 of pole duration
        max_consol = math.ceil(pole_bars * (2 / 3))
        bars_left  = max_consol - bars_used   # can be negative if already exceeded

        # Expiry date = TODAY + bars_left trading days.
        # Anchoring to today (not last_consol_date) is critical:
        # last_consol_date is the last downloaded bar — for a live scan
        # that is yesterday or today, so adding bars_left to it would
        # give an expiry that is already in the past if bars_left is small.
        # For expired patterns bars_left is negative — show days overdue
        expiry_date    = today + pd.tseries.offsets.BDay(bars_left)
        days_to_expiry = bars_left   # negative = overdue, 0 = today, positive = remaining
        is_soon        = 0 <= days_to_expiry <= 1

        overdue_note = (f"{abs(bars_left)} bar(s) overdue — technically expired"
                        if bars_left < 0
                        else f"{bars_left} trading day(s) remaining from today")
        reason = (
            f"Murphy rule: flag/pennant must break out within 2/3 of pole duration "
            f"({pole_bars} bars × 2/3 = {max_consol} max consolidation bars). "
            f"Pattern has used {bars_used} of {max_consol} allowed bars — "
            f"{overdue_note}."
        )
        return D(expiry_date), days_to_expiry, is_soon, reason

    elif pattern in ("Symmetrical Triangle", "Ascending Triangle"):
        if consol_df is None or len(consol_df) < 5:
            return None, None, False, ""

        # Estimate apex: use trendline slopes on highs and lows
        # Apex = where high trendline meets low trendline
        n = len(consol_df)
        x = np.arange(n)

        h_slope, h_intercept, *_ = linregress(x, consol_df["High"].values.astype(float))
        l_slope, l_intercept, *_ = linregress(x, consol_df["Low"].values.astype(float))

        # Solve: h_intercept + h_slope*t = l_intercept + l_slope*t
        # t = (l_intercept - h_intercept) / (h_slope - l_slope)
        denom = h_slope - l_slope
        if abs(denom) < 1e-8:
            # Lines parallel — no apex, use a fixed 60-bar cap
            bars_to_apex = 60
        else:
            bars_to_apex = int((l_intercept - h_intercept) / denom)
            bars_to_apex = max(0, bars_to_apex)

        # Murphy: breakout should happen before 3/4 of apex distance
        max_bars    = math.ceil(bars_to_apex * 0.75)
        bars_used   = n
        bars_left   = max_bars - bars_used   # can be negative if already exceeded

        # Anchor to today; bars_left can be negative for overdue patterns
        expiry_date    = today + pd.tseries.offsets.BDay(bars_left)
        days_to_expiry = bars_left
        is_soon        = 0 <= days_to_expiry <= 1

        overdue_note = (f"{abs(bars_left)} bar(s) overdue — technically expired"
                        if bars_left < 0
                        else f"{bars_left} trading day(s) remaining from today")
        reason = (
            f"Murphy rule: triangle must break out before 3/4 of the distance to apex. "
            f"Estimated apex in ~{bars_to_apex} bars from triangle start — "
            f"3/4 mark = {max_bars} bars. "
            f"Pattern has used {bars_used} bars, {overdue_note}."
        )
        return D(expiry_date), days_to_expiry, is_soon, reason

    return None, None, False, ""

def _build_narrative(symbol: str,
                     pattern: str,
                     pole_df,          # DataFrame slice or None
                     consol_df,        # DataFrame slice
                     vp,               # VolumeProfile
                     weekly,           # WeeklyConfirmation or None
                     breakout_level: float,
                     stop_loss: float,
                     target: float,
                     retrace_pct: float = 0.0,
                     swing_highs_dates: list = None,
                     swing_lows_dates:  list = None,
                     expiry_date: str = None,
                     days_remaining: int = None,
                     is_expiring_soon: bool = False,
                     expiry_reason: str = "") -> str:
    """
    Build a human-readable verification story.
    Tells the trader exactly what to look for on the chart,
    with specific dates, prices and volume context.
    """
    lines = []
    D = _fmt_date

    # ── STEP 1: Prior move / Pole ──────────────────────────────────────────
    if pole_df is not None and len(pole_df) > 0:
        p_start  = D(pole_df.index[0])
        p_end    = D(pole_df.index[-1])
        p_open   = pole_df["Close"].iloc[0]
        p_close  = pole_df["Close"].iloc[-1]
        p_gain   = (p_close - p_open) / p_open * 100
        p_bars   = len(pole_df)
        p_vol    = pole_df["Volume"].mean()

        lines.append(
            f"[1. POLE] From {p_start} to {p_end} ({p_bars} sessions): "
            f"price moved from ₹{p_open:.1f} to ₹{p_close:.1f} "
            f"(+{p_gain:.1f}%). "
            f"Open your chart and verify this was a sharp, near-vertical move. "
        )
        if vp and vp.pole_surge:
            lines.append(
                f"    Volume during this pole averaged {p_vol:,.0f} shares/day — "
                f"{vp.pole_vs_prepole_ratio:.1f}x higher than the 20 sessions before it. "
                f"You should see clearly taller volume bars during {p_start}–{p_end}."
            )
        else:
            lines.append(
                f"    ⚠ Volume on the pole was NOT significantly above average "
                f"({vp.pole_vs_prepole_ratio:.1f}x pre-pole). Murphy requires a surge — "
                f"treat this pole as weaker."
            )
    else:
        # Triangle patterns — no explicit pole
        lines.append(
            f"[1. PRIOR TREND] This is a {pattern}. "
            f"Open your chart and confirm there was a meaningful uptrend "
            f"before the pattern started forming."
        )

    # ── STEP 2: Consolidation / Pattern body ─────────────────────────────
    c_start = D(consol_df.index[0])
    c_end   = D(consol_df.index[-1])
    c_high  = consol_df["High"].max()
    c_low   = consol_df["Low"].min()
    c_bars  = len(consol_df)
    c_vol   = consol_df["Volume"].mean()

    if pattern == "Bull Flag":
        lines.append(
            f"[2. FLAG] From {c_start} to {c_end} ({c_bars} sessions): "
            f"price consolidated between ₹{c_low:.1f} and ₹{c_high:.1f}. "
            f"On the chart you should see a tight downward-drifting channel — "
            f"both the highs and lows making slightly lower values each day. "
            f"The channel should look parallel (upper and lower boundary nearly same angle)."
        )
        if retrace_pct > 0:
            lines.append(
                f"    Retracement of the pole: {retrace_pct:.1f}% "
                f"(Murphy ideal: 25–50%). "
                + ("Good — shallow pullback." if retrace_pct <= 50
                   else "⚠ Deeper than ideal — reduces pattern quality.")
            )

    elif pattern == "Pennant":
        lines.append(
            f"[2. PENNANT] From {c_start} to {c_end} ({c_bars} sessions): "
            f"price compressed between ₹{c_low:.1f} and ₹{c_high:.1f}. "
            f"On the chart look for a small triangle shape — "
            f"the highs should be falling and the lows rising, "
            f"converging toward a point (apex). "
            f"This should look smaller and tighter than the pole."
        )

    elif pattern == "Symmetrical Triangle":
        h_desc = (f"swing highs at {', '.join([D(d) for d in swing_highs_dates])}"
                  if swing_highs_dates else "a series of lower swing highs")
        l_desc = (f"swing lows at {', '.join([D(d) for d in swing_lows_dates])}"
                  if swing_lows_dates else "a series of higher swing lows")
        lines.append(
            f"[2. TRIANGLE] From {c_start} to {c_end} ({c_bars} sessions): "
            f"pattern formed with {h_desc} and {l_desc}. "
            f"On the chart draw a line connecting the swing highs (should slope down) "
            f"and another connecting the swing lows (should slope up). "
            f"They should converge toward each other — forming a symmetric triangle. "
            f"Upper boundary is the resistance, lower is support."
        )

    elif pattern == "Ascending Triangle":
        h_desc = (f"at {', '.join([D(d) for d in swing_highs_dates])}"
                  if swing_highs_dates else "")
        l_desc = (f"swing lows at {', '.join([D(d) for d in swing_lows_dates])}"
                  if swing_lows_dates else "a series of higher swing lows")
        lines.append(
            f"[2. TRIANGLE] From {c_start} to {c_end} ({c_bars} sessions): "
            f"flat resistance around ₹{c_high:.1f} tested multiple times {h_desc}. "
            f"{l_desc.capitalize()} — price bouncing at higher levels each time. "
            f"On the chart draw a horizontal line at ₹{breakout_level:.1f} — "
            f"this is the resistance buyers keep hitting. "
            f"The lows should form a rising line. Buyers are accumulating."
        )

    # ── STEP 3: Volume in consolidation ──────────────────────────────────
    if vp:
        lines.append(
            f"[3. VOLUME CHECK] During the consolidation ({c_start}–{c_end}): "
            f"average volume was {c_vol:,.0f} shares/day — "
            f"{vp.consol_avg_ratio*100:.0f}% of the pole's average "
            f"({'✓ contracted as Murphy requires' if vp.consol_avg_ok else '⚠ not contracted enough'}). "
        )
        if vp.consol_trend_declining:
            lines.append(
                f"    Volume was also trending DOWN within the consolidation "
                f"(slope: {vp.consol_slope_per_day:+,.0f} shares/day). "
                f"On the chart the volume bars should visibly shrink from left to right "
                f"across the {c_start}–{c_end} period. This is exactly what Murphy describes."
            )
        else:
            lines.append(
                f"    ⚠ Volume within consolidation was NOT clearly declining "
                f"(slope: {vp.consol_slope_per_day:+,.0f} shares/day). "
                f"Check the chart manually — if volume bars are erratic or rising, "
                f"the pattern is weaker."
            )

    # ── STEP 4: Current status ─────────────────────────────────────────────
    lines.append(
        f"[4. STATUS — IN FORMATION] As of {c_end}: "
        f"price has NOT yet broken out. "
        f"Breakout level to watch: ₹{breakout_level:.1f}. "
        f"Price needs to close above this level on volume above "
        f"{vp.breakout_vol_threshold:,.0f} shares (1.5× 20-day avg) "
        f"to confirm the breakout."
    )

    # ── STEP 5: Trade plan ─────────────────────────────────────────────────
    lines.append(
        f"[5. TRADE PLAN] "
        f"Entry: buy on a closing breakout above ₹{breakout_level:.1f}. "
        f"Stop-loss: ₹{stop_loss:.1f} (below pattern low). "
        f"Target: ₹{target:.1f} (pole-length projected from breakout). "
        f"Do NOT enter if the breakout bar volume is below "
        f"{vp.breakout_vol_threshold:,.0f} shares — that is a false breakout signal."
        if vp else
        f"[5. TRADE PLAN] Entry: ₹{breakout_level:.1f} | Stop: ₹{stop_loss:.1f} | Target: ₹{target:.1f}."
    )

    # ── STEP 6: Weekly context ─────────────────────────────────────────────
    if weekly and weekly.available:
        if weekly.confirmed():
            lines.append(
                f"[6. WEEKLY CHART ✓] Weekly trend is UP "
                f"(EMA20 slope {weekly.ema20w_slope_pct:+.3f}%/week) and "
                f"weekly volume is contracting ({weekly.last_week_vs_8w:.2f}× 8-week average). "
                f"The daily pattern is aligned with the bigger trend — stronger setup."
            )
        else:
            problems = []
            if not weekly.trend_up:
                problems.append(f"weekly trend is FLAT/DOWN (EMA20 slope {weekly.ema20w_slope_pct:+.3f}%/wk)")
            if not weekly.vol_contracting:
                problems.append(f"weekly volume is NOT contracting ({weekly.last_week_vs_8w:.2f}× 8-week avg)")
            lines.append(
                f"[6. WEEKLY CHART ⚠] "
                + "; ".join(problems)
                + ". Check the weekly chart before trading — "
                  "daily pattern is present but weekly context is not ideal."
            )

    # ── STEP 7: Actionable watch instruction + ETA ─────────────────────────
    vol_note = (f"on volume above {vp.breakout_vol_threshold:,.0f} shares"
                if vp else "on above-average volume")

    if expiry_date and days_remaining is not None:
        if is_expiring_soon:
            eta_tag = f"⚠ EXPIRING SOON — {days_remaining} trading day(s) left (by {expiry_date})"
            urgency = (
                f"Pattern validity is running out fast. "
                f"If price does not close above ₹{breakout_level:.1f} {vol_note} "
                f"by {expiry_date}, remove this stock from your watchlist immediately — "
                f"the setup will have failed and a breakdown is more likely than a breakout."
            )
        else:
            eta_tag = f"Valid until {expiry_date} ({days_remaining} trading day(s) remaining)"
            urgency = (
                f"You have time, but do not get complacent. "
                f"If price has not broken out by {expiry_date}, "
                f"the pattern is invalid — remove it from your watchlist regardless of how good it looks."
            )

        lines.append(
            f"[7. WHAT TO WATCH FOR] "
            f"Wait for a CLOSING candle above ₹{breakout_level:.1f} {vol_note}. "
            f"Do NOT enter intraday — only a day-end close above this level counts. "
            f"Until that happens, this is a watch-only setup.  "
            f"{eta_tag}.  "
            f"{expiry_reason}  "
            f"{urgency}"
        )
    else:
        lines.append(
            f"[7. WHAT TO WATCH FOR] "
            f"Wait for a CLOSING candle above ₹{breakout_level:.1f} {vol_note}. "
            f"Do NOT enter intraday — only a day-end close above this level counts. "
            f"Until that happens, this is a watch-only setup."
        )

    return "  //  ".join(lines)

# ─────────────────────────────────────────────────────────────────────────────
# CORE MURPHY CHECKS
# ─────────────────────────────────────────────────────────────────────────────

def _detect_pole(df: pd.DataFrame,
                 consol_start_idx: int,
                 min_pole_bars: int = 5,
                 max_pole_bars: int = 15,       # v2.0 — Murphy: near-vertical cap
                 min_pole_gain_pct: float = 8.0
                 ) -> Tuple[Optional[pd.DataFrame], float]:
    """
    Find the pole immediately preceding consolidation.
    v2.0: enforces MAX pole duration of 15 bars (Murphy near-vertical rule).
    Returns (pole_df, gain_pct) or (None, 0).
    """
    if consol_start_idx < min_pole_bars:
        return None, 0.0

    # Search window = exactly last max_pole_bars bars before consolidation.
    # Do NOT extend further back — flat pre-pattern bars would drag pole_low
    # too far left, making pole_len exceed max_pole_bars and get rejected.
    search_start = max(0, consol_start_idx - max_pole_bars)
    search = df.iloc[search_start: consol_start_idx]
    if search.empty:
        return None, 0.0

    pole_low_idx  = search["Low"].idxmin()
    pole_low_pos  = df.index.get_loc(pole_low_idx)
    pole_high_pos = consol_start_idx - 1
    pole_len      = pole_high_pos - pole_low_pos

    # Duration gate (both sides)
    if pole_len < min_pole_bars or pole_len > max_pole_bars:
        return None, 0.0

    pole_df   = df.iloc[pole_low_pos: pole_high_pos + 1]
    pole_gain = ((pole_df["Close"].iloc[-1] - pole_df["Close"].iloc[0])
                 / pole_df["Close"].iloc[0] * 100)

    if pole_gain < min_pole_gain_pct:
        return None, 0.0

    return pole_df, pole_gain


def _volume_profile(df: pd.DataFrame,
                    pole_df: pd.DataFrame,
                    consol_df: pd.DataFrame) -> VolumeProfile:
    """
    3-stage Murphy volume check — v2.0.
    Stage 1: pole avg >= 1.5x pre-pole avg  (surge)
    Stage 2: consol avg < 65% of pole avg   (level contraction)
    Stage 3: vol slope negative AND late<early in consol (trend declining)
    Also computes breakout watch threshold = 1.5x 20d avg.
    """
    pole_start_pos = df.index.get_loc(pole_df.index[0])
    pre_pole = df.iloc[max(0, pole_start_pos - 20): pole_start_pos]
    pre_pole_avg = pre_pole["Volume"].mean() if len(pre_pole) > 0 else 0

    pole_avg   = pole_df["Volume"].mean()
    consol_avg = consol_df["Volume"].mean()

    # Stage 1 — surge
    pole_ratio = pole_avg / pre_pole_avg if pre_pole_avg > 0 else 0
    pole_surge = pole_ratio >= 1.5

    # Stage 2 — average contraction
    consol_ratio = consol_avg / pole_avg if pole_avg > 0 else 1.0
    consol_avg_ok = consol_ratio < 0.65

    # Stage 3 — declining trend inside consolidation
    vol_slope = _raw_slope(consol_df["Volume"])
    n = len(consol_df)
    if n >= 6:
        early_avg = consol_df["Volume"].iloc[:3].mean()
        late_avg  = consol_df["Volume"].iloc[-3:].mean()
    else:
        early_avg = consol_df["Volume"].iloc[0] if n > 0 else 0
        late_avg  = consol_df["Volume"].iloc[-1] if n > 0 else 0

    trend_declining = (vol_slope < 0) and (late_avg < early_avg * 0.80)

    # Breakout watch level — 1.5× 20d avg of whole df
    avg_20d = df["Volume"].tail(20).mean()
    breakout_threshold = avg_20d * 1.5

    all_ok = pole_surge and consol_avg_ok and trend_declining

    return VolumeProfile(
        pole_surge=pole_surge,
        pole_vs_prepole_ratio=round(pole_ratio, 2),
        consol_avg_ok=consol_avg_ok,
        consol_avg_ratio=round(consol_ratio, 2),
        consol_trend_declining=trend_declining,
        consol_slope_per_day=round(vol_slope),
        consol_early_avg=round(early_avg),
        consol_late_avg=round(late_avg),
        breakout_vol_threshold=round(breakout_threshold),
        all_confirmed=all_ok,
    )


def _weekly_confirmation(symbol: str) -> WeeklyConfirmation:
    """
    Pull weekly candles, check:
    • EMA20 weekly slope positive (weekly uptrend)
    • Last week volume < 8-week average (weekly vol contracting)
    """
    wdf = fetch_weekly(symbol, weeks=30)
    if wdf is None or len(wdf) < 12:
        return WeeklyConfirmation(available=False, trend_up=False,
                                  vol_contracting=False,
                                  ema20w_slope_pct=0.0, last_week_vs_8w=0.0)

    ema20 = wdf["Close"].ewm(span=20, adjust=False).mean()
    ema_slope = _slope_pct_per_bar(ema20.tail(4))   # last 4 weeks slope
    trend_up = ema_slope > 0

    last_week_vol = wdf["Volume"].iloc[-1]
    avg_8w_vol    = wdf["Volume"].tail(8).mean()
    vol_ratio     = last_week_vol / avg_8w_vol if avg_8w_vol > 0 else 1.0
    vol_contracting = vol_ratio < 1.0

    return WeeklyConfirmation(
        available=True,
        trend_up=trend_up,
        vol_contracting=vol_contracting,
        ema20w_slope_pct=round(ema_slope, 3),
        last_week_vs_8w=round(vol_ratio, 2),
    )


def _in_formation_check(last_close: float,
                        breakout_level: float,
                        pattern_low: float) -> Tuple[bool, float]:
    """
    Returns (in_formation, gap_to_breakout_pct).
    In formation = price is below breakout level AND above pattern low.
    """
    if last_close >= breakout_level:
        return False, 0.0           # Already broke out — too late
    if last_close < pattern_low:
        return False, 0.0           # Broke down — pattern failed
    gap_pct = (breakout_level - last_close) / last_close * 100
    return True, round(gap_pct, 2)


def _risk_reward(entry: float, stop: float, target: float) -> float:
    risk   = entry - stop
    reward = target - entry
    if risk <= 0:
        return 0.0
    return round(reward / risk, 2)


def _confidence(score: float, vol: VolumeProfile,
                weekly: Optional[WeeklyConfirmation]) -> str:
    """
    HIGH requires: score>=70 AND all 3 vol checks AND weekly confirmed.
    MEDIUM: score>=50 OR partial vol.
    LOW: everything else.
    """
    weekly_ok = weekly.confirmed() if weekly else False
    if score >= 70 and vol.all_confirmed and weekly_ok:
        return "HIGH"
    if score >= 70 and vol.all_confirmed:
        return "HIGH"     # weekly unavailable — don't penalise
    if score >= 50:
        return "MEDIUM"
    return "LOW"


# ─────────────────────────────────────────────────────────────────────────────
# PATTERN 1 — BULL FLAG
# ─────────────────────────────────────────────────────────────────────────────

def detect_bull_flag(symbol: str, df: pd.DataFrame,
                     weekly: Optional[WeeklyConfirmation] = None) -> PatternResult:
    empty = PatternResult(symbol=symbol, pattern="Bull Flag",
                          detected=False, in_formation=False,
                          score=0.0, confidence="LOW")
    MIN_C, MAX_C = 5, 20

    best_score, best = 0.0, None

    for ce in range(len(df) - 1, len(df) - 3, -1):
        for cl in range(MIN_C, MAX_C + 1):
            cs = ce - cl
            if cs < 10:
                continue

            pole_df, pole_gain = _detect_pole(df, cs)
            if pole_df is None:
                continue

            consol_df = df.iloc[cs: ce + 1]
            last_close = df["Close"].iloc[-1]
            breakout   = consol_df["High"].max()
            stop       = consol_df["Low"].min()

            # ── In-formation gate (v2.0) ──────────────────────────────────
            in_form, gap = _in_formation_check(last_close, breakout, stop)
            if not in_form:
                continue

            score = 0.0
            notes = []

            # Pole quality
            if pole_gain >= 15:
                score += 25; notes.append(f"Strong pole +{pole_gain:.1f}% ✓")
            elif pole_gain >= 8:
                score += 15; notes.append(f"Pole +{pole_gain:.1f}%")

            # Pole duration (Murphy: near-vertical)
            pole_bars = len(pole_df)
            if pole_bars <= 10:
                score += 5; notes.append(f"Sharp pole ({pole_bars}d) ✓")

            # Flag trendline slopes — both must be negative
            h_slope = _slope_pct_per_bar(consol_df["High"])
            l_slope = _slope_pct_per_bar(consol_df["Low"])
            if h_slope < -0.05 and l_slope < -0.05:
                score += 20; notes.append("Downward-sloping channel ✓")
            else:
                continue   # Not a flag

            # Parallel check
            if abs(h_slope) > 0:
                parallelism = abs((h_slope - l_slope) / h_slope)
                if parallelism < 0.5:
                    score += 15; notes.append("Parallel channel ✓")

            # Retracement — close-based (Murphy measures pole close-to-close)
            pole_range  = pole_df["Close"].iloc[-1] - pole_df["Close"].iloc[0]
            pole_top    = pole_df["Close"].iloc[-1]
            retrace_pct = (pole_top - consol_df["Low"].min()) / pole_range * 100 if pole_range > 0 else 0
            if 25 <= retrace_pct <= 50:
                score += 15; notes.append(f"Retracement {retrace_pct:.1f}% (ideal) ✓")
            elif retrace_pct > 61.8:
                score -= 10; notes.append(f"Deep retracement {retrace_pct:.1f}% ✗")
            else:
                score += 5

            # Duration
            if 7 <= cl <= 15:
                score += 5; notes.append(f"Duration {cl}d (ideal) ✓")

            # Volume profile
            vp = _volume_profile(df, pole_df, consol_df)
            if vp.pole_surge:
                score += 10; notes.append(f"Pole vol surge {vp.pole_vs_prepole_ratio}x ✓")
            if vp.consol_avg_ok:
                score += 5
            if vp.consol_trend_declining:
                score += 5; notes.append("Vol declining in flag ✓")

            if score > best_score:
                best_score = score
                pole_len = pole_df["High"].max() - pole_df["Low"].min()
                target   = breakout + pole_len
                conf     = _confidence(score, vp, weekly)
                exp_date, exp_days, exp_soon, exp_reason = _pattern_expiry(
                    "Bull Flag", pole_df, consol_df, actual_consol_bars=cl)
                # Murphy time-limit gate: expired patterns are invalid — exclude entirely
                # A score of 100 on an expired setup is meaningless and actively misleading
                if exp_days is not None and exp_days < 0:
                    continue
                narr = _build_narrative(
                    symbol=symbol, pattern="Bull Flag",
                    pole_df=pole_df, consol_df=consol_df, vp=vp,
                    weekly=weekly,
                    breakout_level=breakout, stop_loss=stop, target=target,
                    retrace_pct=retrace_pct,
                    expiry_date=exp_date, days_remaining=exp_days,
                    is_expiring_soon=exp_soon, expiry_reason=exp_reason,
                )
                best = PatternResult(
                    symbol=symbol, pattern="Bull Flag",
                    detected=score >= 40, in_formation=True,
                    score=min(score, 100), confidence=conf,
                    breakout_level=round(breakout, 2),
                    stop_loss=round(stop, 2),
                    target=round(target, 2),
                    risk_reward=_risk_reward(breakout, stop, target),
                    gap_to_breakout_pct=gap,
                    pole_return_pct=round(pole_gain, 1),
                    pole_bars=pole_bars,
                    consolidation_bars=cl,
                    volume_profile=vp,
                    weekly=weekly,
                    notes=notes,
                    narrative=narr,
                )

    return best or empty


# ─────────────────────────────────────────────────────────────────────────────
# PATTERN 2 — PENNANT
# ─────────────────────────────────────────────────────────────────────────────

def detect_pennant(symbol: str, df: pd.DataFrame,
                   weekly: Optional[WeeklyConfirmation] = None) -> PatternResult:
    empty = PatternResult(symbol=symbol, pattern="Pennant",
                          detected=False, in_formation=False,
                          score=0.0, confidence="LOW")
    MIN_C, MAX_C = 5, 15

    best_score, best = 0.0, None

    for ce in range(len(df) - 1, len(df) - 3, -1):
        for cl in range(MIN_C, MAX_C + 1):
            cs = ce - cl
            if cs < 10:
                continue

            pole_df, pole_gain = _detect_pole(df, cs)
            if pole_df is None:
                continue

            consol_df  = df.iloc[cs: ce + 1]
            last_close = df["Close"].iloc[-1]
            breakout   = consol_df["High"].max()
            stop       = consol_df["Low"].min()

            # ── In-formation gate ─────────────────────────────────────────
            in_form, gap = _in_formation_check(last_close, breakout, stop)
            if not in_form:
                continue

            h_slope = _slope_pct_per_bar(consol_df["High"])
            l_slope = _slope_pct_per_bar(consol_df["Low"])

            # Pennant: upper descends, lower ascends — converging
            if not (h_slope < -0.03 and l_slope > 0.03):
                continue

            score = 0.0
            notes = []

            if pole_gain >= 12:
                score += 25; notes.append(f"Strong pole +{pole_gain:.1f}% ✓")
            elif pole_gain >= 8:
                score += 15; notes.append(f"Pole +{pole_gain:.1f}%")

            pole_bars = len(pole_df)
            if pole_bars <= 10:
                score += 5; notes.append(f"Sharp pole ({pole_bars}d) ✓")

            score += 25; notes.append("Converging trendlines ✓")

            # Compression ratio — consol range vs pole close-to-close range
            pole_range   = pole_df["Close"].iloc[-1] - pole_df["Close"].iloc[0]
            consol_range = consol_df["High"].max() - consol_df["Low"].min()
            compression  = consol_range / pole_range if pole_range > 0 else 1
            if compression < 0.35:
                score += 15; notes.append(f"Price compressed to {compression*100:.0f}% of pole ✓")
            elif compression < 0.50:
                score += 8

            # Retracement — close-based
            pole_range   = pole_df["Close"].iloc[-1] - pole_df["Close"].iloc[0]
            pole_top    = pole_df["Close"].iloc[-1]
            retrace_pct = (pole_top - consol_df["Low"].min()) / pole_range * 100 if pole_range > 0 else 0
            if 20 <= retrace_pct <= 40:
                score += 15; notes.append(f"Retracement {retrace_pct:.1f}% ✓")
            elif retrace_pct > 50:
                score -= 10

            # Volume
            vp = _volume_profile(df, pole_df, consol_df)
            if vp.pole_surge:
                score += 10; notes.append(f"Pole vol surge {vp.pole_vs_prepole_ratio}x ✓")
            if vp.consol_avg_ok:
                score += 5
            if vp.consol_trend_declining:
                score += 5; notes.append("Vol declining in pennant ✓")

            if score > best_score:
                best_score = score
                pole_len = pole_df["High"].max() - pole_df["Low"].min()
                target   = breakout + pole_len
                conf     = _confidence(score, vp, weekly)
                exp_date, exp_days, exp_soon, exp_reason = _pattern_expiry(
                    "Pennant", pole_df, consol_df, actual_consol_bars=cl)
                # Murphy time-limit gate
                if exp_days is not None and exp_days < 0:
                    continue
                narr = _build_narrative(
                    symbol=symbol, pattern="Pennant",
                    pole_df=pole_df, consol_df=consol_df, vp=vp,
                    weekly=weekly,
                    breakout_level=breakout, stop_loss=stop, target=target,
                    retrace_pct=retrace_pct,
                    expiry_date=exp_date, days_remaining=exp_days,
                    is_expiring_soon=exp_soon, expiry_reason=exp_reason,
                )
                best = PatternResult(
                    symbol=symbol, pattern="Pennant",
                    detected=score >= 40, in_formation=True,
                    score=min(score, 100), confidence=conf,
                    breakout_level=round(breakout, 2),
                    stop_loss=round(stop, 2),
                    target=round(target, 2),
                    risk_reward=_risk_reward(breakout, stop, target),
                    gap_to_breakout_pct=gap,
                    pole_return_pct=round(pole_gain, 1),
                    pole_bars=pole_bars,
                    consolidation_bars=cl,
                    volume_profile=vp,
                    weekly=weekly,
                    notes=notes,
                    narrative=narr,
                )

    return best or empty


# ─────────────────────────────────────────────────────────────────────────────
# PATTERN 3 — SYMMETRICAL TRIANGLE
# ─────────────────────────────────────────────────────────────────────────────

def detect_symmetrical_triangle(symbol: str, df: pd.DataFrame,
                                 weekly: Optional[WeeklyConfirmation] = None) -> PatternResult:
    empty = PatternResult(symbol=symbol, pattern="Symmetrical Triangle",
                          detected=False, in_formation=False,
                          score=0.0, confidence="LOW")

    if len(df) < 30:
        return empty

    window = df.tail(60).copy()
    sh = _swing_highs(window["High"], order=3)
    sl = _swing_lows(window["Low"],   order=3)

    highs_idx = window.index[sh]
    lows_idx  = window.index[sl]

    if len(highs_idx) < 2 or len(lows_idx) < 2:
        return empty

    h1_pos = window.index.get_loc(highs_idx[-2])
    h2_pos = window.index.get_loc(highs_idx[-1])
    l1_pos = window.index.get_loc(lows_idx[-2])
    l2_pos = window.index.get_loc(lows_idx[-1])

    h1, h2 = window["High"].iloc[h1_pos], window["High"].iloc[h2_pos]
    l1, l2 = window["Low"].iloc[l1_pos],  window["Low"].iloc[l2_pos]

    if not (h2 < h1 and h2_pos > h1_pos):
        return empty
    if not (l2 > l1 and l2_pos > l1_pos):
        return empty

    tri_start = min(h1_pos, l1_pos)
    tri_end   = max(h2_pos, l2_pos)
    tri_df    = window.iloc[tri_start: tri_end + 1]
    if len(tri_df) < 10:
        return empty

    breakout   = h2
    stop       = l2
    last_close = df["Close"].iloc[-1]

    in_form, gap = _in_formation_check(last_close, breakout, stop)
    if not in_form:
        return empty

    score = 0.0
    notes = []

    score += 20; notes.append("Descending swing highs ✓")
    score += 20; notes.append("Ascending swing lows ✓")

    h_slope = _slope_pct_per_bar(tri_df["High"])
    l_slope = _slope_pct_per_bar(tri_df["Low"])
    if h_slope < 0 and l_slope > 0:
        score += 20; notes.append("Trendlines converging ✓")
        sym_ratio = abs(abs(h_slope) - abs(l_slope)) / max(abs(h_slope), abs(l_slope)) if max(abs(h_slope), abs(l_slope)) > 0 else 1
        if sym_ratio < 0.4:
            score += 10; notes.append("Near-symmetrical slopes ✓")

    dur = tri_end - tri_start
    if 15 <= dur <= 60:
        score += 15; notes.append(f"Duration {dur}d (Murphy ideal) ✓")
    elif dur < 15:
        score += 5

    # Volume — use triangle portion vs pre-triangle
    pre_start = max(0, len(df) - len(window) + tri_start - 20)
    pre_end   = max(0, len(df) - len(window) + tri_start)
    pre_df    = df.iloc[pre_start: pre_end]

    tri_vol_avg = tri_df["Volume"].mean()
    pre_vol_avg = pre_df["Volume"].mean() if len(pre_df) > 5 else tri_vol_avg
    vol_slope   = _raw_slope(tri_df["Volume"])
    n = len(tri_df)
    early_avg = tri_df["Volume"].iloc[:3].mean() if n >= 6 else tri_df["Volume"].iloc[0]
    late_avg  = tri_df["Volume"].iloc[-3:].mean() if n >= 6 else tri_df["Volume"].iloc[-1]
    trend_declining = (vol_slope < 0) and (late_avg < early_avg * 0.80)
    avg_ok = (tri_vol_avg / pre_vol_avg < 0.80) if pre_vol_avg > 0 else False

    # Build a minimal VolumeProfile for triangles (no explicit pole)
    avg_20d = df["Volume"].tail(20).mean()
    vp = VolumeProfile(
        pole_surge=True,          # no pole concept for triangles
        pole_vs_prepole_ratio=1.0,
        consol_avg_ok=avg_ok,
        consol_avg_ratio=round(tri_vol_avg / pre_vol_avg, 2) if pre_vol_avg > 0 else 1.0,
        consol_trend_declining=trend_declining,
        consol_slope_per_day=round(vol_slope),
        consol_early_avg=round(early_avg),
        consol_late_avg=round(late_avg),
        breakout_vol_threshold=round(avg_20d * 1.5),
        all_confirmed=avg_ok and trend_declining,
    )

    if avg_ok:
        score += 10; notes.append("Volume contracted in triangle ✓")
    if trend_declining:
        score += 5;  notes.append("Vol declining trend ✓")

    approx_height = (h1 + h2) / 2 - (l1 + l2) / 2
    target = breakout + approx_height

    detected = score >= 50
    conf = _confidence(score, vp, weekly)

    exp_date, exp_days, exp_soon, exp_reason = _pattern_expiry(
        "Symmetrical Triangle", None, tri_df,
        swing_highs_dates=list(highs_idx[-2:]),
        swing_lows_dates=list(lows_idx[-2:]),
    )
    # Murphy time-limit gate
    if exp_days is not None and exp_days < 0:
        return empty
    narr = _build_narrative(
        symbol=symbol, pattern="Symmetrical Triangle",
        pole_df=None, consol_df=tri_df, vp=vp,
        weekly=weekly,
        breakout_level=breakout, stop_loss=stop, target=target,
        swing_highs_dates=list(highs_idx[-2:]),
        swing_lows_dates=list(lows_idx[-2:]),
        expiry_date=exp_date, days_remaining=exp_days,
        is_expiring_soon=exp_soon, expiry_reason=exp_reason,
    )
    return PatternResult(
        symbol=symbol, pattern="Symmetrical Triangle",
        detected=detected, in_formation=True,
        score=min(score, 100), confidence=conf,
        breakout_level=round(breakout, 2),
        stop_loss=round(stop, 2),
        target=round(target, 2),
        risk_reward=_risk_reward(breakout, stop, target),
        gap_to_breakout_pct=gap,
        pole_return_pct=0.0,
        pole_bars=0,
        consolidation_bars=dur,
        volume_profile=vp,
        weekly=weekly,
        notes=notes,
        narrative=narr,
    )


# ─────────────────────────────────────────────────────────────────────────────
# PATTERN 4 — ASCENDING TRIANGLE
# ─────────────────────────────────────────────────────────────────────────────

def detect_ascending_triangle(symbol: str, df: pd.DataFrame,
                               weekly: Optional[WeeklyConfirmation] = None) -> PatternResult:
    empty = PatternResult(symbol=symbol, pattern="Ascending Triangle",
                          detected=False, in_formation=False,
                          score=0.0, confidence="LOW")

    if len(df) < 25:
        return empty

    window = df.tail(60).copy()
    sh = _swing_highs(window["High"], order=3)
    sl = _swing_lows(window["Low"],   order=3)

    highs_idx = window.index[sh]
    lows_idx  = window.index[sl]

    if len(highs_idx) < 2 or len(lows_idx) < 2:
        return empty

    recent_highs = window["High"].loc[highs_idx[-3:]] if len(highs_idx) >= 3 else window["High"].loc[highs_idx[-2:]]
    high_range_pct = (recent_highs.max() - recent_highs.min()) / recent_highs.mean() * 100

    if high_range_pct > 2.5:
        return empty   # Not flat enough

    resistance = recent_highs.mean()
    last_close = df["Close"].iloc[-1]
    stop       = window["Low"].iloc[-1]

    in_form, gap = _in_formation_check(last_close, resistance, stop)
    if not in_form:
        return empty

    score = 0.0
    notes = []

    if high_range_pct <= 1.5:
        score += 30; notes.append(f"Flat resistance (±{high_range_pct:.1f}%) ✓")
    else:
        score += 15; notes.append(f"Near-flat resistance (±{high_range_pct:.1f}%)")

    recent_lows = window["Low"].loc[lows_idx[-3:]] if len(lows_idx) >= 3 else window["Low"].loc[lows_idx[-2:]]
    lows_list   = recent_lows.values
    rising = all(lows_list[i] > lows_list[i - 1] for i in range(1, len(lows_list)))
    if rising:
        score += 30; notes.append("Rising swing lows ✓")

    n_touches = len(highs_idx)
    if n_touches >= 3:
        score += 10; notes.append(f"{n_touches} touches of resistance ✓")
    elif n_touches >= 2:
        score += 5

    if gap <= 2.0:
        score += 15; notes.append(f"Price near resistance ({gap:.1f}% away) ✓")
    elif gap <= 4.0:
        score += 8

    # Volume
    tri_start_idx = (window.index.get_loc(lows_idx[-3])
                     if len(lows_idx) >= 3
                     else window.index.get_loc(lows_idx[-2]))
    tri_df  = window.iloc[tri_start_idx:]
    pre_start = max(0, len(df) - len(window) + tri_start_idx - 20)
    pre_df  = df.iloc[pre_start: max(0, len(df) - len(window) + tri_start_idx)]

    tri_vol = tri_df["Volume"].mean()
    pre_vol = pre_df["Volume"].mean() if len(pre_df) > 5 else tri_vol
    vol_slope = _raw_slope(tri_df["Volume"])
    n = len(tri_df)
    early_v = tri_df["Volume"].iloc[:3].mean() if n >= 6 else tri_df["Volume"].iloc[0]
    late_v  = tri_df["Volume"].iloc[-3:].mean() if n >= 6 else tri_df["Volume"].iloc[-1]
    trend_d = (vol_slope < 0) and (late_v < early_v * 0.80)
    avg_ok  = (tri_vol / pre_vol < 0.85) if pre_vol > 0 else False

    avg_20d = df["Volume"].tail(20).mean()
    vp = VolumeProfile(
        pole_surge=True,
        pole_vs_prepole_ratio=1.0,
        consol_avg_ok=avg_ok,
        consol_avg_ratio=round(tri_vol / pre_vol, 2) if pre_vol > 0 else 1.0,
        consol_trend_declining=trend_d,
        consol_slope_per_day=round(vol_slope),
        consol_early_avg=round(early_v),
        consol_late_avg=round(late_v),
        breakout_vol_threshold=round(avg_20d * 1.5),
        all_confirmed=avg_ok and trend_d,
    )

    if avg_ok:
        score += 10; notes.append("Volume contracting ✓")
    if trend_d:
        score += 5;  notes.append("Vol declining trend ✓")

    tri_height = resistance - window["Low"].loc[lows_idx].min()
    target = resistance + tri_height
    dur    = len(tri_df)

    detected = score >= 55
    conf = _confidence(score, vp, weekly)

    exp_date, exp_days, exp_soon, exp_reason = _pattern_expiry(
        "Ascending Triangle", None, tri_df,
        swing_highs_dates=list(highs_idx[-3:] if len(highs_idx) >= 3 else highs_idx[-2:]),
        swing_lows_dates=list(lows_idx[-3:] if len(lows_idx) >= 3 else lows_idx[-2:]),
    )
    # Murphy time-limit gate
    if exp_days is not None and exp_days < 0:
        return empty
    narr = _build_narrative(
        symbol=symbol, pattern="Ascending Triangle",
        pole_df=None, consol_df=tri_df, vp=vp,
        weekly=weekly,
        breakout_level=resistance, stop_loss=stop, target=target,
        swing_highs_dates=list(highs_idx[-3:] if len(highs_idx) >= 3 else highs_idx[-2:]),
        swing_lows_dates=list(lows_idx[-3:] if len(lows_idx) >= 3 else lows_idx[-2:]),
        expiry_date=exp_date, days_remaining=exp_days,
        is_expiring_soon=exp_soon, expiry_reason=exp_reason,
    )
    return PatternResult(
        symbol=symbol, pattern="Ascending Triangle",
        detected=detected, in_formation=True,
        score=min(score, 100), confidence=conf,
        breakout_level=round(resistance, 2),
        stop_loss=round(stop, 2),
        target=round(target, 2),
        risk_reward=_risk_reward(resistance, stop, target),
        gap_to_breakout_pct=gap,
        pole_return_pct=0.0,
        pole_bars=0,
        consolidation_bars=dur,
        volume_profile=vp,
        weekly=weekly,
        notes=notes,
        narrative=narr,
    )


# ─────────────────────────────────────────────────────────────────────────────
# SCANNER + REPORTER
# ─────────────────────────────────────────────────────────────────────────────

def scan_symbol(symbol: str, lookback: int = 90,
                use_weekly: bool = True) -> list:
    df = fetch_daily(symbol, lookback)
    if df is None:
        print(f"  [SKIP] {symbol} — no daily data")
        return []

    weekly = _weekly_confirmation(symbol) if use_weekly else None

    detectors = [
        detect_bull_flag,
        detect_pennant,
        detect_symmetrical_triangle,
        detect_ascending_triangle,
    ]

    results = []
    for fn in detectors:
        try:
            r = fn(symbol, df, weekly)
            results.append(r)
        except Exception as e:
            print(f"  [WARN] {symbol}/{fn.__name__}: {e}")
    return results


def run_scanner(symbols: list, lookback: int = 90,
                min_score: float = 40.0,
                use_weekly: bool = True) -> pd.DataFrame:
    rows = []
    for sym in symbols:
        print(f"  Scanning {sym}...")
        for r in scan_symbol(sym, lookback, use_weekly):
            if r.detected and r.in_formation and r.score >= min_score:
                rows.append(r.to_dict())

    if not rows:
        return pd.DataFrame()

    out = pd.DataFrame(rows).sort_values("Score", ascending=False).reset_index(drop=True)
    return out


def print_report(results_by_symbol: dict, use_weekly: bool = True):
    """Detailed per-symbol, per-pattern report matching the mockup layout."""
    SEP  = "═" * 72
    SEP2 = "─" * 72

    all_detected = []
    for sym, results in results_by_symbol.items():
        for r in results:
            if r.detected and r.in_formation:
                all_detected.append(r)

    print(f"\n{SEP}")
    print("  MaverickPICKS — PATTERN DETECTION REPORT  v2.0")
    print("  John Murphy principles | Daily TF + Weekly confirmation")
    print(f"  Mode: IN-FORMATION ONLY  |  Weekly check: {'ON' if use_weekly else 'OFF'}")
    print(SEP)

    if not all_detected:
        print("\n  No in-formation patterns detected above threshold.\n")
        print(SEP + "\n")
        return

    CONF_TAG = {"HIGH": "🟢 HIGH", "MEDIUM": "🟡 MED ", "LOW": "🔴 LOW "}
    PAT_SYM  = {"Bull Flag": "▲ ", "Pennant": "◆ ",
                "Symmetrical Triangle": "◇ ", "Ascending Triangle": "△ "}

    for r in sorted(all_detected, key=lambda x: x.score, reverse=True):
        vp = r.volume_profile
        wk = r.weekly
        ct = CONF_TAG.get(r.confidence, "   ")
        ps = PAT_SYM.get(r.pattern, "  ")

        print(f"\n  {ps}{r.symbol:<16} {r.pattern:<24} Score: {r.score:>5.1f}/100  {ct}")
        print(f"  {SEP2}")

        if r.pole_bars > 0:
            print(f"  Pole      : +{r.pole_return_pct:.1f}% in {r.pole_bars}d  "
                  f"(Murphy max 15d {'✓' if r.pole_bars <= 15 else '✗'})")

        print(f"  Consol    : {r.consolidation_bars}d  |  "
              f"Gap to breakout: {r.gap_to_breakout_pct:.1f}%")

        # Trade levels
        print(f"  Entry     : {r.breakout_level or '—'}")
        print(f"  Stop      : {r.stop_loss or '—'}")
        print(f"  Target    : {r.target or '—'}   R:R {r.risk_reward:.1f}x")

        # Volume — 3-stage
        if vp:
            print(f"  {SEP2}")
            print(f"  Volume (Murphy 3-stage check):")
            ps_str = "✓" if vp.pole_surge else "✗"
            ca_str = "✓" if vp.consol_avg_ok else "✗"
            td_str = "✓" if vp.consol_trend_declining else "✗"
            print(f"    Stage 1 — Pole surge    : {ps_str}  ({vp.pole_vs_prepole_ratio:.1f}x pre-pole avg)")
            print(f"    Stage 2 — Consol avg    : {ca_str}  ({vp.consol_avg_ratio*100:.0f}% of pole avg)")
            print(f"    Stage 3 — Vol declining : {td_str}  (slope {vp.consol_slope_per_day:+,.0f} shares/day)")
            print(f"              Early consol  : {vp.consol_early_avg:,.0f}  →  Late consol: {vp.consol_late_avg:,.0f}")
            print(f"    Breakout watch          : >{vp.breakout_vol_threshold:,.0f} shares  (1.5× 20d avg)")
            all3 = "ALL 3 CONFIRMED ✓" if vp.all_confirmed else "not all confirmed"
            print(f"    Overall                 : {all3}")

        # Weekly
        if wk and use_weekly:
            print(f"  {SEP2}")
            print(f"  Weekly confirmation:")
            print(f"    Trend (EMA20W slope)    : {'✓ UP' if wk.trend_up else '✗ flat/down'}  "
                  f"({wk.ema20w_slope_pct:+.3f}%/wk)")
            print(f"    Vol contracting (wkly)  : {'✓' if wk.vol_contracting else '✗'}  "
                  f"(last wk = {wk.last_week_vs_8w:.2f}× 8w avg)")
            print(f"    Confirmed               : {'YES ✓' if wk.confirmed() else 'NO'}")

        # Criteria notes
        print(f"  {SEP2}")
        print(f"  Criteria: {' | '.join(r.notes)}")

    # Summary table
    print(f"\n{SEP}")
    print(f"  SUMMARY — {len(all_detected)} in-formation pattern(s) detected")
    print(f"  {'Rank':<5} {'Symbol':<16} {'Pattern':<24} {'Score':>6} {'Conf':<8} {'R:R':>5} {'Vol':>5} {'Wkly':>5}")
    print(f"  {'─'*4} {'─'*15} {'─'*23} {'─'*6} {'─'*7} {'─'*5} {'─'*5} {'─'*5}")
    for i, r in enumerate(sorted(all_detected, key=lambda x: x.score, reverse=True), 1):
        vok  = "✓" if (r.volume_profile and r.volume_profile.all_confirmed) else "✗"
        wok  = "✓" if (r.weekly and r.weekly.confirmed()) else ("—" if not use_weekly else "✗")
        conf = r.confidence[:3]
        print(f"  {i:<5} {r.symbol:<16} {r.pattern:<24} {r.score:>6.1f} {conf:<8} {r.risk_reward:>5.1f} {vok:>5} {wok:>5}")

    print(SEP + "\n")


# ─────────────────────────────────────────────────────────────────────────────
# BATCH SCANNING — parallel workers, progress bar, CSV universe input
# ─────────────────────────────────────────────────────────────────────────────

def _load_symbols_from_csv(path: str) -> list:
    """
    Read symbols from a CSV file.
    Handles NIFTY500_MASTER.csv format (single 'Symbol' column, no .NS suffix)
    and also generic CSVs with columns named 'Symbol', 'Ticker', 'NSE_Symbol'.
    Appends '.NS' if suffix is missing.
    """
    df = pd.read_csv(path)
    # Find the symbol column (case-insensitive)
    col = None
    for candidate in ["Symbol", "symbol", "Ticker", "ticker",
                       "NSE_Symbol", "nse_symbol", "SYMBOL"]:
        if candidate in df.columns:
            col = candidate
            break
    if col is None:
        col = df.columns[0]   # fallback: first column

    symbols = df[col].dropna().astype(str).str.strip().tolist()
    # Append .NS if not already present
    symbols = [s if s.endswith(".NS") or s.endswith(".BO") else s + ".NS"
               for s in symbols if s]
    return symbols


def _scan_one(args_tuple) -> tuple:
    """Worker function for parallel scanning. Returns (sym, results_list)."""
    sym, lookback, use_weekly, min_score = args_tuple
    try:
        results = scan_symbol(sym, lookback, use_weekly)
        hits = [r for r in results
                if r.detected and r.in_formation and r.score >= min_score]
        return sym, results, hits
    except Exception as e:
        return sym, [], []


def run_batch(symbols: list,
              lookback: int = 90,
              min_score: float = 50.0,
              use_weekly: bool = True,
              workers: int = 4) -> tuple:
    """
    Scan all symbols in parallel.
    Returns (results_by_symbol dict, summary_rows list).
    Shows a live progress counter — no external tqdm required.
    """
    from concurrent.futures import ThreadPoolExecutor, as_completed

    total = len(symbols)
    results_by_symbol = {}
    summary_rows = []
    done = 0
    hits_found = 0

    args_list = [(sym, lookback, use_weekly, min_score) for sym in symbols]

    print(f"  Scanning {total} symbols with {workers} parallel workers...\n")

    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(_scan_one, a): a[0] for a in args_list}
        for future in as_completed(futures):
            sym, sym_results, hits = future.result()
            done += 1
            results_by_symbol[sym] = sym_results
            for r in hits:
                summary_rows.append(r.to_dict())
                hits_found += 1

            # Live progress line
            pct   = done / total * 100
            bar   = "█" * int(pct / 5) + "░" * (20 - int(pct / 5))
            found = f"  {hits_found} pattern(s) found" if hits_found else ""
            print(f"\r  [{bar}] {pct:5.1f}%  {done}/{total}  {sym:<20}{found}",
                  end="", flush=True)

    print()   # newline after progress bar
    return results_by_symbol, summary_rows


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(
        description="MaverickPICKS Pattern Detector v2.0 — NIFTY500 batch scanner"
    )

    # Symbol input — either a CSV universe file OR explicit symbols
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--csv_file", type=str,
                     help="CSV with Symbol column (e.g. NIFTY500_MASTER.csv)")
    src.add_argument("--symbols",  nargs="+",
                     help="Explicit NSE symbols e.g. RELIANCE.NS TCS.NS")

    ap.add_argument("--lookback",  type=int,   default=90,
                    help="Days of daily history to fetch (default: 90)")
    ap.add_argument("--min_score", type=float, default=50.0,
                    help="Min quality score 0-100 to include in report (default: 50)")
    ap.add_argument("--no_weekly", action="store_true",
                    help="Skip weekly chart confirmation — faster but less strict")
    ap.add_argument("--workers",   type=int,   default=4,
                    help="Parallel download threads (default: 4, max recommended: 8)")
    ap.add_argument("--out_csv",   type=str,   default="",
                    help="Save results to this CSV path e.g. patterns_today.csv")

    args = ap.parse_args()
    use_weekly = not args.no_weekly

    # ── Market hours check ───────────────────────────────────────────────────
    # NSE trades 9:15am–3:30pm IST (UTC+5:30).
    # Running during market hours means the last bar is a live partial candle
    # with incomplete volume — results will vary between runs on the same day.
    # Best practice: run AFTER 3:30pm IST when the day's candle is confirmed.
    from datetime import timezone, timedelta as _td
    _IST = timezone(_td(hours=5, minutes=30))
    _now_ist = datetime.now(_IST)
    _market_open  = _now_ist.replace(hour=9,  minute=15, second=0, microsecond=0)
    _market_close = _now_ist.replace(hour=15, minute=30, second=0, microsecond=0)
    _is_weekday   = _now_ist.weekday() < 5

    if _is_weekday and _market_open <= _now_ist <= _market_close:
        print()
        print("  ⚠  WARNING: NSE market is currently OPEN")
        print(f"     Current IST time : {_now_ist.strftime('%H:%M:%S')}")
        print("     The last price bar is a live partial candle with incomplete volume.")
        print("     Running now may give different results than running after market close.")
        print("     RECOMMENDED: Run this script after 3:30 PM IST for consistent results.")
        print("     You can continue now, but treat output as preliminary — re-run after close.")
        print()
    else:
        _ts = _now_ist.strftime("%H:%M IST")
        print(f"  ✓  Market closed — scanning on confirmed end-of-day data ({_ts})")
        print()

    # ── Load symbols ─────────────────────────────────────────────────────────
    if args.csv_file:
        symbols = _load_symbols_from_csv(args.csv_file)
        universe_label = f"{args.csv_file}  ({len(symbols)} symbols)"
    else:
        symbols = [s if s.endswith(".NS") else s + ".NS" for s in args.symbols]
        universe_label = f"{len(symbols)} symbol(s) specified"

    if not symbols:
        print("  [ERROR] No symbols found. Check your CSV or --symbols argument.")
        return

    # ── Header ───────────────────────────────────────────────────────────────
    print(f"\n{'═'*64}")
    print("  MaverickPICKS Pattern Detector v2.0")
    print(f"  Universe : {universe_label}")
    print(f"  Lookback : {args.lookback}d  |  Min score : {args.min_score}")
    print(f"  Weekly   : {'ON' if use_weekly else 'OFF (--no_weekly)'}  "
          f"|  Workers  : {args.workers}")
    print(f"  Patterns : Bull Flag | Pennant | Sym Triangle | Asc Triangle")
    print(f"  Data anchor : {_ANCHOR_DATE.strftime('%d-%b-%Y')} "
          f"(all symbols use this as end date — results are run-to-run consistent)")
    print(f"{'═'*64}\n")

    import time
    t0 = time.time()

    # ── Run batch scan ────────────────────────────────────────────────────────
    results_by_symbol, summary_rows = run_batch(
        symbols=symbols,
        lookback=args.lookback,
        min_score=args.min_score,
        use_weekly=use_weekly,
        workers=args.workers,
    )

    elapsed = time.time() - t0
    print(f"\n  Scan complete in {elapsed/60:.1f} min  "
          f"({elapsed/len(symbols):.1f}s per symbol)\n")

    # ── Print detailed report ─────────────────────────────────────────────────
    print_report(results_by_symbol, use_weekly)

    # ── Save CSV ──────────────────────────────────────────────────────────────
    if args.out_csv and summary_rows:
        out_df = pd.DataFrame(summary_rows).sort_values("Score", ascending=False)
        out_df.to_csv(args.out_csv, index=False)
        print(f"  Results saved → {args.out_csv}  ({len(out_df)} rows)\n")
    elif not summary_rows:
        print("  No patterns detected above threshold — try lowering --min_score.\n")


if __name__ == "__main__":
    main()

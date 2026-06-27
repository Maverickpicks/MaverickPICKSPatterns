"""
maverick_scanner.py — MaverickPICKS Unified Pattern Scanner
============================================================
Scans all NIFTY500 stocks for bullish patterns in formation:
  • Bull Flag, Pennant, Symmetrical Triangle, Ascending Triangle (Murphy)
  • Inverse H&S (bullish reversal)

Writes: scan_results.csv — one row per detected pattern

Usage:
  python maverick_scanner.py
  python maverick_scanner.py --min_score 40 --workers 4
"""

import argparse
import math
import time
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta, timezone
from typing import Optional

import numpy as np
import pandas as pd
import yfinance as yf
from scipy.stats import linregress

warnings.filterwarnings("ignore")

IST        = timezone(timedelta(hours=5, minutes=30))
OUTPUT_CSV = "scan_results.csv"


# ─────────────────────────────────────────────────────────────────────────────
# DATA FETCH — identical to data_loader.py
# ─────────────────────────────────────────────────────────────────────────────

def _anchor_date() -> datetime:
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


ANCHOR = _anchor_date()


def fetch_daily(symbol: str, years: float = 2.0) -> Optional[pd.DataFrame]:
    ticker = symbol if symbol.endswith(".NS") else symbol + ".NS"
    end    = ANCHOR + timedelta(days=1)
    start  = ANCHOR - timedelta(days=int(years * 365.25))
    for attempt in range(3):
        try:
            df = yf.download(ticker, start=start, end=end,
                             interval="1d", auto_adjust=False,
                             progress=False, threads=False)
            if df.empty: raise ValueError("empty")
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = [c[0] for c in df.columns]
            df = df[["Open","High","Low","Close","Volume"]].dropna()
            df.index = pd.to_datetime(df.index)
            return df
        except Exception:
            time.sleep(1)
    return None


def fetch_weekly(symbol: str) -> Optional[pd.DataFrame]:
    ticker = symbol if symbol.endswith(".NS") else symbol + ".NS"
    end    = ANCHOR + timedelta(days=1)
    start  = ANCHOR - timedelta(days=int(5 * 365.25))
    for attempt in range(3):
        try:
            df = yf.download(ticker, start=start, end=end,
                             interval="1wk", auto_adjust=False,
                             progress=False, threads=False)
            if df.empty: raise ValueError("empty")
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = [c[0] for c in df.columns]
            df = df[["Open","High","Low","Close","Volume"]].dropna()
            df.index = pd.to_datetime(df.index)
            return df.tail(52)
        except Exception:
            time.sleep(1)
    return None


# ─────────────────────────────────────────────────────────────────────────────
# SHARED UTILITIES
# ─────────────────────────────────────────────────────────────────────────────

def _slope(series: pd.Series) -> float:
    if len(series) < 2: return 0.0
    x = np.arange(len(series))
    s, *_ = linregress(x, series.values.astype(float))
    return float(s)


def _slope_pct(series: pd.Series) -> float:
    s = _slope(series)
    m = series.mean()
    return (s / m * 100) if m != 0 else 0.0


def _swing_highs(high: pd.Series, order: int = 3) -> pd.Series:
    r = pd.Series(False, index=high.index)
    for i in range(order, len(high) - order):
        if high.iloc[i] == high.iloc[i-order:i+order+1].max():
            r.iloc[i] = True
    return r


def _swing_lows(low: pd.Series, order: int = 3) -> pd.Series:
    r = pd.Series(False, index=low.index)
    for i in range(order, len(low) - order):
        if low.iloc[i] == low.iloc[i-order:i+order+1].min():
            r.iloc[i] = True
    return r


def _fmt(ts) -> str:
    try: return pd.Timestamp(ts).strftime("%d-%b-%Y")
    except: return ""


def _weekly_trend(symbol: str) -> str:
    wdf = fetch_weekly(symbol)
    if wdf is None or len(wdf) < 12: return "UNKNOWN"
    ema20 = wdf["Close"].ewm(span=20, adjust=False).mean()
    slope = _slope_pct(ema20.tail(4))
    last_vol = wdf["Volume"].iloc[-1]
    avg_vol  = wdf["Volume"].tail(8).mean()
    trend_up = slope > 0
    vol_ok   = last_vol < avg_vol
    if trend_up and vol_ok: return "UP + VOL_CONTRACTING"
    elif trend_up:          return "UP"
    else:                   return "FLAT_OR_DOWN"


# ─────────────────────────────────────────────────────────────────────────────
# MURPHY PATTERN DETECTORS
# ─────────────────────────────────────────────────────────────────────────────

def _detect_pole(df, cs, min_bars=5, max_bars=15, min_gain=8.0):
    if cs < min_bars: return None, 0.0
    search = df.iloc[max(0, cs-max_bars): cs]
    if search.empty: return None, 0.0
    low_idx  = search["Low"].idxmin()
    low_pos  = df.index.get_loc(low_idx)
    high_pos = cs - 1
    pole_len = high_pos - low_pos
    if pole_len < min_bars or pole_len > max_bars: return None, 0.0
    pole_df  = df.iloc[low_pos: high_pos+1]
    gain     = (pole_df["Close"].iloc[-1] - pole_df["Close"].iloc[0]) / pole_df["Close"].iloc[0] * 100
    if gain < min_gain: return None, 0.0
    pre = df.iloc[max(0, low_pos-20): low_pos]
    if len(pre) > 5 and pole_df["Volume"].mean() < pre["Volume"].mean() * 0.9:
        return None, 0.0
    return pole_df, gain


def _vol_profile(df, pole_df, consol_df):
    pole_start = df.index.get_loc(pole_df.index[0])
    pre  = df.iloc[max(0, pole_start-20): pole_start]
    pre_avg  = pre["Volume"].mean() if len(pre) > 0 else 1
    pole_avg = pole_df["Volume"].mean()
    c_avg    = consol_df["Volume"].mean()
    surge    = pole_avg >= pre_avg * 1.5
    avg_ok   = (c_avg / pole_avg < 0.65) if pole_avg > 0 else False
    vs       = _slope(consol_df["Volume"])
    n        = len(consol_df)
    early    = consol_df["Volume"].iloc[:3].mean() if n >= 6 else consol_df["Volume"].iloc[0]
    late     = consol_df["Volume"].iloc[-3:].mean() if n >= 6 else consol_df["Volume"].iloc[-1]
    trend_dn = vs < 0 and late < early * 0.80
    avg20    = df["Volume"].tail(20).mean()
    return {
        "surge": surge, "avg_ok": avg_ok, "trend_dn": trend_dn,
        "all3": surge and avg_ok and trend_dn,
        "ratio": round(pole_avg/pre_avg, 2) if pre_avg > 0 else 0,
        "consol_ratio": round(c_avg/pole_avg, 2) if pole_avg > 0 else 1,
        "slope_per_day": round(vs),
        "breakout_vol_watch": round(avg20 * 1.5),
    }


def _expiry(pole_df, consol_bars):
    if pole_df is None: return None, None
    pole_bars  = len(pole_df)
    max_consol = math.ceil(pole_bars * 2/3)
    bars_left  = max_consol - consol_bars
    today      = pd.Timestamp(ANCHOR)
    expiry     = today + pd.tseries.offsets.BDay(bars_left)
    days_left  = bars_left
    return _fmt(expiry), days_left


def _narrative_flag(sym, pole_df, consol_df, vp, brk, stop, tgt, retrace, exp_date, days_left, weekly):
    parts = []
    parts.append(
        f"[POLE] {_fmt(pole_df.index[0])} to {_fmt(pole_df.index[-1])} "
        f"({len(pole_df)} sessions): ₹{pole_df['Close'].iloc[0]:.1f} → "
        f"₹{pole_df['Close'].iloc[-1]:.1f} (+{(pole_df['Close'].iloc[-1]/pole_df['Close'].iloc[0]-1)*100:.1f}%). "
        f"Volume averaged {pole_df['Volume'].mean():,.0f}/day — {vp['ratio']:.1f}x above prior 20-day average."
    )
    parts.append(
        f"[FLAG] {_fmt(consol_df.index[0])} to {_fmt(consol_df.index[-1])} "
        f"({len(consol_df)} sessions): price drifted down between "
        f"₹{consol_df['Low'].min():.1f} and ₹{consol_df['High'].max():.1f}. "
        f"Retracement {retrace:.1f}% of pole (Murphy ideal: 25-50%). "
        f"Volume contracted to {vp['consol_ratio']*100:.0f}% of pole average — "
        f"{'drying up as Murphy requires ✓' if vp['avg_ok'] else 'not yet contracted enough'}."
    )
    parts.append(
        f"[ENTRY] Wait for a closing candle above ₹{brk:.1f} on volume above "
        f"{vp['breakout_vol_watch']:,.0f} shares. Stop: ₹{stop:.1f}. Target: ₹{tgt:.1f}."
    )
    if exp_date:
        urgency = f"⚠ Only {days_left} trading day(s) left." if days_left and days_left <= 2 else f"Valid until {exp_date}."
        parts.append(f"[EXPIRY] {urgency} Pattern invalidates if no breakout by {exp_date}.")
    if weekly != "UNKNOWN":
        parts.append(f"[WEEKLY] {weekly}.")
    return "  //  ".join(parts)


def _narrative_triangle(sym, pattern, consol_df, brk, stop, tgt, h_dates, l_dates, vp_note, weekly):
    parts = []
    h_str = ", ".join(_fmt(d) for d in h_dates) if h_dates else "multiple highs"
    l_str = ", ".join(_fmt(d) for d in l_dates) if l_dates else "multiple lows"
    parts.append(
        f"[TRIANGLE] {pattern} forming from {_fmt(consol_df.index[0])} to "
        f"{_fmt(consol_df.index[-1])} ({len(consol_df)} sessions). "
        f"Descending swing highs at {h_str}. Ascending swing lows at {l_str}. "
        f"Trendlines converging toward apex."
    )
    parts.append(
        f"[VOLUME] {vp_note}"
    )
    parts.append(
        f"[ENTRY] Breakout above ₹{brk:.1f}. Stop: ₹{stop:.1f}. Target: ₹{tgt:.1f}."
    )
    if weekly != "UNKNOWN":
        parts.append(f"[WEEKLY] {weekly}.")
    return "  //  ".join(parts)


def scan_murphy(symbol: str, df: pd.DataFrame, weekly: str,
                min_score: float = 40) -> list:
    results = []
    last_close = float(df["Close"].iloc[-1])

    # ── Bull Flag & Pennant ───────────────────────────────────────────────────
    for pattern in ("Bull Flag", "Pennant"):
        MIN_C = 5; MAX_C = 20 if pattern == "Bull Flag" else 15
        best_score = 0; best = None

        for ce in range(len(df)-1, len(df)-3, -1):
            for cl in range(MIN_C, MAX_C+1):
                cs = ce - cl
                if cs < 10: continue
                pole_df, pole_gain = _detect_pole(df, cs)
                if pole_df is None: continue
                consol_df = df.iloc[cs:ce+1]
                brk  = consol_df["High"].max()
                stop = consol_df["Low"].min()

                # In-formation check
                if last_close >= brk or last_close <= stop: continue

                h_slope = _slope_pct(consol_df["High"])
                l_slope = _slope_pct(consol_df["Low"])

                if pattern == "Bull Flag":
                    if not (h_slope < -0.05 and l_slope < -0.05): continue
                else:
                    if not (h_slope < -0.03 and l_slope > 0.03): continue

                score = 0
                if pole_gain >= 15: score += 25
                elif pole_gain >= 8: score += 15
                if len(pole_df) <= 10: score += 5

                if pattern == "Bull Flag":
                    score += 20
                    if abs(h_slope) > 0 and abs((h_slope-l_slope)/h_slope) < 0.5:
                        score += 15
                else:
                    score += 25
                    pole_range = pole_df["Close"].iloc[-1] - pole_df["Close"].iloc[0]
                    c_range = consol_df["High"].max() - consol_df["Low"].min()
                    if pole_range > 0 and c_range/pole_range < 0.35: score += 15

                pole_range = pole_df["Close"].iloc[-1] - pole_df["Close"].iloc[0]
                retrace = (pole_df["Close"].iloc[-1] - consol_df["Low"].min()) / pole_range * 100 if pole_range > 0 else 0
                ideal_r = (25 <= retrace <= 50) if pattern == "Bull Flag" else (20 <= retrace <= 40)
                if ideal_r: score += 15
                elif retrace > 61.8: score -= 10

                if 7 <= cl <= 15: score += 5

                vp = _vol_profile(df, pole_df, consol_df)
                if vp["surge"]: score += 10
                if vp["avg_ok"]: score += 5
                if vp["trend_dn"]: score += 5

                if score > best_score:
                    best_score = score
                    exp_date, days_left = _expiry(pole_df, cl)
                    if days_left is not None and days_left < 0: continue

                    pole_len = pole_df["High"].max() - pole_df["Low"].min()
                    tgt  = brk + pole_len
                    rr   = round((tgt-brk)/(brk-stop), 2) if brk > stop else 0
                    gap  = round((brk-last_close)/last_close*100, 2)
                    conf = "HIGH" if (score >= 70 and vp["all3"]) else ("MEDIUM" if score >= 50 else "LOW")
                    narr = _narrative_flag(symbol, pole_df, consol_df, vp, brk, stop, tgt, retrace, exp_date, days_left, weekly)

                    best = {
                        "Symbol": symbol, "Pattern": pattern,
                        "Score": round(min(score,100),1), "Confidence": conf,
                        "Entry_Breakout": round(brk,2), "Stop_Loss": round(stop,2),
                        "Target": round(tgt,2), "Risk_Reward": rr,
                        "Gap_To_Entry_%": gap,
                        "Pattern_Start": _fmt(consol_df.index[0]),
                        "Pattern_Expiry": exp_date or "",
                        "Days_To_Expiry": days_left,
                        "Pole_Start": _fmt(pole_df.index[0]),
                        "Pole_End": _fmt(pole_df.index[-1]),
                        "Pole_Return_%": round(pole_gain,1),
                        "Pole_Bars": len(pole_df),
                        "Consol_Bars": cl,
                        "Vol_All3_OK": vp["all3"],
                        "Vol_Surge": vp["surge"],
                        "Breakout_Vol_Watch": vp["breakout_vol_watch"],
                        "Weekly_Trend": weekly,
                        "Narrative": narr,
                    }

        if best and best["Score"] >= min_score:
            results.append(best)

    # ── Symmetrical Triangle ──────────────────────────────────────────────────
    window = df.tail(60)
    sh = _swing_highs(window["High"], order=3)
    sl = _swing_lows(window["Low"], order=3)
    hi = window.index[sh]; lo = window.index[sl]

    if len(hi) >= 2 and len(lo) >= 2:
        h1p = window.index.get_loc(hi[-2]); h2p = window.index.get_loc(hi[-1])
        l1p = window.index.get_loc(lo[-2]); l2p = window.index.get_loc(lo[-1])
        h1,h2 = window["High"].iloc[h1p], window["High"].iloc[h2p]
        l1,l2 = window["Low"].iloc[l1p],  window["Low"].iloc[l2p]

        if h2 < h1 and h2p > h1p and l2 > l1 and l2p > l1p:
            tri_start = min(h1p, l1p); tri_end = max(h2p, l2p)
            tri_df = window.iloc[tri_start:tri_end+1]
            brk = h2; stop = l2

            if last_close < brk and last_close > stop and len(tri_df) >= 10:
                h_sl = _slope_pct(tri_df["High"]); l_sl = _slope_pct(tri_df["Low"])
                if h_sl < 0 and l_sl > 0:
                    score = 55
                    dur   = tri_end - tri_start
                    if 15 <= dur <= 60: score += 15
                    gap = round((brk-last_close)/last_close*100, 2)
                    height = (h1+h2)/2 - (l1+l2)/2
                    tgt  = brk + height
                    rr   = round((tgt-brk)/(brk-stop),2) if brk > stop else 0
                    pre_v = df.iloc[max(0,len(df)-60+tri_start-20): max(0,len(df)-60+tri_start)]["Volume"].mean()
                    tri_v = tri_df["Volume"].mean()
                    vol_note = f"Volume contracted to {tri_v/pre_v*100:.0f}% of prior average ✓" if pre_v > 0 and tri_v < pre_v*0.80 else "Volume contraction not clear"
                    narr = _narrative_triangle(symbol, "Symmetrical Triangle", tri_df, brk, stop, tgt, list(hi[-2:]), list(lo[-2:]), vol_note, weekly)
                    if score >= min_score:
                        results.append({
                            "Symbol": symbol, "Pattern": "Symmetrical Triangle",
                            "Score": round(score,1), "Confidence": "HIGH" if score>=75 else "MEDIUM",
                            "Entry_Breakout": round(brk,2), "Stop_Loss": round(stop,2),
                            "Target": round(tgt,2), "Risk_Reward": rr, "Gap_To_Entry_%": gap,
                            "Pattern_Start": _fmt(tri_df.index[0]),
                            "Pattern_Expiry": "", "Days_To_Expiry": None,
                            "Pole_Start": "", "Pole_End": "", "Pole_Return_%": 0,
                            "Pole_Bars": 0, "Consol_Bars": dur,
                            "Vol_All3_OK": tri_v < pre_v*0.80 if pre_v > 0 else False,
                            "Vol_Surge": False,
                            "Breakout_Vol_Watch": round(df["Volume"].tail(20).mean()*1.5),
                            "Weekly_Trend": weekly, "Narrative": narr,
                        })

    # ── Ascending Triangle ────────────────────────────────────────────────────
    window2 = df.tail(60)
    sh2 = _swing_highs(window2["High"], order=3)
    sl2 = _swing_lows(window2["Low"], order=3)
    hi2 = window2.index[sh2]; lo2 = window2.index[sl2]

    if len(hi2) >= 2 and len(lo2) >= 2:
        recent_highs = window2["High"].loc[hi2[-3:]] if len(hi2)>=3 else window2["High"].loc[hi2[-2:]]
        hr = (recent_highs.max()-recent_highs.min())/recent_highs.mean()*100
        if hr <= 2.5:
            resistance = recent_highs.mean()
            stop = window2["Low"].iloc[-1]
            if last_close < resistance and last_close > stop:
                recent_lows = window2["Low"].loc[lo2[-3:]] if len(lo2)>=3 else window2["Low"].loc[lo2[-2:]]
                rising = all(recent_lows.values[i]>recent_lows.values[i-1] for i in range(1,len(recent_lows)))
                score = 30 if hr<=1.5 else 15
                if rising: score += 30
                gap_r = (resistance-last_close)/resistance*100
                if gap_r<=2: score+=15
                elif gap_r<=4: score+=8
                n_touch = len(hi2)
                if n_touch>=3: score+=10
                elif n_touch>=2: score+=5

                tri_h = resistance - window2["Low"].loc[lo2].min()
                tgt  = resistance + tri_h
                rr   = round((tgt-resistance)/(resistance-stop),2) if resistance>stop else 0
                gap  = round((resistance-last_close)/last_close*100,2)

                tri_start2 = window2.index.get_loc(lo2[-3]) if len(lo2)>=3 else window2.index.get_loc(lo2[-2])
                tri_df2 = window2.iloc[tri_start2:]
                pre_v2 = df.iloc[max(0,len(df)-60+tri_start2-20):max(0,len(df)-60+tri_start2)]["Volume"].mean()
                tri_v2 = tri_df2["Volume"].mean()
                vol_ok2 = tri_v2 < pre_v2*0.85 if pre_v2>0 else False
                if vol_ok2: score+=10
                h_dates2 = [d for d in hi2[-3:]] if len(hi2)>=3 else [d for d in hi2[-2:]]
                l_dates2 = [d for d in lo2[-3:]] if len(lo2)>=3 else [d for d in lo2[-2:]]
                vol_note2= f"Volume contracted to {tri_v2/pre_v2*100:.0f}% of prior average ✓" if vol_ok2 else "Volume contraction unclear"
                narr2 = _narrative_triangle(symbol, "Ascending Triangle", tri_df2, resistance, stop, tgt, h_dates2, l_dates2, vol_note2, weekly)
                if score >= min_score:
                    results.append({
                        "Symbol": symbol, "Pattern": "Ascending Triangle",
                        "Score": round(score,1), "Confidence": "HIGH" if score>=75 else ("MEDIUM" if score>=55 else "LOW"),
                        "Entry_Breakout": round(resistance,2), "Stop_Loss": round(stop,2),
                        "Target": round(tgt,2), "Risk_Reward": rr, "Gap_To_Entry_%": gap,
                        "Pattern_Start": _fmt(tri_df2.index[0]),
                        "Pattern_Expiry": "", "Days_To_Expiry": None,
                        "Pole_Start": "", "Pole_End": "", "Pole_Return_%": 0,
                        "Pole_Bars": 0, "Consol_Bars": len(tri_df2),
                        "Vol_All3_OK": vol_ok2, "Vol_Surge": False,
                        "Breakout_Vol_Watch": round(df["Volume"].tail(20).mean()*1.5),
                        "Weekly_Trend": weekly, "Narrative": narr2,
                    })

    return results


# ─────────────────────────────────────────────────────────────────────────────
# H&S DETECTOR (Inverse H&S only — bullish)
# ─────────────────────────────────────────────────────────────────────────────

PIVOT_ORDER  = 5
DEPTH_TOL    = 0.18
TIME_RATIO   = 2.2
MIN_PROMIN   = 0.03
STALE_MULT   = 1.5
VOL_BRK_MULT = 1.5


def scan_inverse_hs(symbol: str, df: pd.DataFrame, weekly: str,
                    min_quality: float = 40) -> list:
    """Detect Inverse H&S patterns (bullish reversal) only."""
    results = []
    if len(df) < 60: return results
    last_close = float(df["Close"].iloc[-1])

    # Find swing lows (for Inverse H&S: LS, Head, RS are lows)
    pivots_low = []
    for i in range(PIVOT_ORDER, len(df)-PIVOT_ORDER):
        if df["Low"].iloc[i] == df["Low"].iloc[i-PIVOT_ORDER:i+PIVOT_ORDER+1].min():
            pivots_low.append((i, df.index[i], float(df["Low"].iloc[i])))

    # Find swing highs (for neckline)
    pivots_high = []
    for i in range(PIVOT_ORDER, len(df)-PIVOT_ORDER):
        if df["High"].iloc[i] == df["High"].iloc[i-PIVOT_ORDER:i+PIVOT_ORDER+1].max():
            pivots_high.append((i, df.index[i], float(df["High"].iloc[i])))

    if len(pivots_low) < 3: return results

    # Try all combinations of 3 consecutive lows as LS, Head, RS
    for i in range(len(pivots_low)-2):
        ls_i, ls_dt, ls_p = pivots_low[i]
        hd_i, hd_dt, hd_p = pivots_low[i+1]
        rs_i, rs_dt, rs_p = pivots_low[i+2]

        # Head must be lowest
        if not (hd_p < ls_p and hd_p < rs_p): continue

        # Head prominence
        avg_sh = (ls_p + rs_p) / 2
        prominence = (avg_sh - hd_p) / avg_sh
        if prominence < MIN_PROMIN: continue

        # Shoulder depth symmetry
        if avg_sh > 0 and abs(ls_p - rs_p) / avg_sh > DEPTH_TOL: continue

        # Time symmetry
        leg1 = hd_i - ls_i
        leg2 = rs_i - hd_i
        if leg1 == 0 or leg2 == 0: continue
        ratio = max(leg1,leg2) / min(leg1,leg2)
        if ratio > TIME_RATIO: continue

        # Neckline: find highs between LS-Head and Head-RS
        nh = [h for h in pivots_high if ls_i < h[0] < hd_i]
        rh = [h for h in pivots_high if hd_i < h[0] < rs_i]
        if not nh or not rh: continue
        nl_left  = max(nh, key=lambda x: x[2])
        nl_right = max(rh, key=lambda x: x[2])
        neckline = (nl_left[2] + nl_right[2]) / 2

        # RS must be confirmed (at least PIVOT_ORDER bars old)
        last_idx = len(df) - 1
        if last_idx - rs_i < PIVOT_ORDER: continue

        # Staleness check
        avg_leg = (leg1 + leg2) / 2
        rs_age  = last_idx - rs_i
        is_stale = rs_age > avg_leg * STALE_MULT

        # Pattern must not have already broken out
        if last_close >= neckline: continue

        # Quality score
        depth_score = max(0, 1 - abs(ls_p-rs_p)/avg_sh/DEPTH_TOL) * 25
        time_score  = max(0, 1 - (ratio-1)/(TIME_RATIO-1)) * 20
        promin_score= min(prominence/0.20, 1.0) * 15
        nl_slope    = abs((nl_right[2]-nl_left[2])/neckline/max(nl_right[0]-nl_left[0],1)*100)
        slope_score = max(0, 1 - nl_slope/0.15) * 15
        vol_score   = 25  # default, can check volume profile
        quality     = round(depth_score+time_score+promin_score+slope_score+vol_score, 1)
        if quality < min_quality: continue

        # Trade levels
        pattern_height = neckline - hd_p
        target  = neckline + pattern_height
        stop    = rs_p * 0.98
        rr      = round((target-neckline)/(neckline-stop),2) if neckline > stop else 0
        gap     = round((neckline-last_close)/last_close*100,2)
        avg20   = df["Volume"].tail(20).mean()

        cat = f"{'Stale - ' if is_stale else ''}Watching - Confirmed RS"
        brk_vol = round(avg20 * VOL_BRK_MULT)

        narrative = (
            f"[INVERSE H&S] Bullish reversal pattern. "
            f"Left shoulder low: ₹{ls_p:.1f} on {_fmt(ls_dt)}. "
            f"Head (lowest low): ₹{hd_p:.1f} on {_fmt(hd_dt)}. "
            f"Right shoulder low: ₹{rs_p:.1f} on {_fmt(rs_dt)}. "
            f"Neckline at ₹{neckline:.1f}. "
            f"Head is {prominence*100:.1f}% below shoulder average — "
            f"{'strong' if prominence>0.08 else 'adequate'} prominence. "
            f"Pattern has used {rs_age} trading days since right shoulder formed "
            f"({'overdue — act with caution' if is_stale else 'within normal timeframe'}). "
            f"[ENTRY] Buy on a closing candle above ₹{neckline:.1f} on volume above "
            f"{brk_vol:,.0f} shares. Stop: ₹{stop:.1f}. Target: ₹{target:.1f} "
            f"(neckline + pattern height). "
            f"[WEEKLY] {weekly}."
        )

        results.append({
            "Symbol": symbol, "Pattern": "Inverse H&S",
            "Score": quality, "Confidence": "HIGH" if quality>=75 else ("MEDIUM" if quality>=55 else "LOW"),
            "Entry_Breakout": round(neckline,2), "Stop_Loss": round(stop,2),
            "Target": round(target,2), "Risk_Reward": rr, "Gap_To_Entry_%": gap,
            "Pattern_Start": _fmt(ls_dt),
            "Pattern_Expiry": "",
            "Days_To_Expiry": None,
            "Pole_Start": _fmt(ls_dt), "Pole_End": _fmt(rs_dt),
            "Pole_Return_%": 0,
            "Pole_Bars": 0, "Consol_Bars": rs_age,
            "Vol_All3_OK": False, "Vol_Surge": False,
            "Breakout_Vol_Watch": brk_vol,
            "Weekly_Trend": weekly,
            "Narrative": narrative,
            "_hs_category": cat,
            "_quality": quality,
        })

    return results


# ─────────────────────────────────────────────────────────────────────────────
# SCAN ONE SYMBOL
# ─────────────────────────────────────────────────────────────────────────────

def scan_symbol(symbol: str, min_score: float) -> list:
    try:
        df = fetch_daily(symbol)
        if df is None or len(df) < 60: return []

        weekly = _weekly_trend(symbol)
        results = []
        results.extend(scan_murphy(symbol, df, weekly, min_score))
        results.extend(scan_inverse_hs(symbol, df, weekly, min_score))
        return results
    except Exception as e:
        return []


# ─────────────────────────────────────────────────────────────────────────────
# BATCH SCANNER
# ─────────────────────────────────────────────────────────────────────────────

def load_symbols() -> list:
    try:
        df = pd.read_csv("NIFTY500_MASTER.csv")
        col = next((c for c in df.columns if c.lower() in ("symbol","ticker")), df.columns[0])
        syms = df[col].dropna().astype(str).str.strip().tolist()
        return syms
    except Exception as e:
        print(f"  [ERROR] Cannot load symbols: {e}")
        return []


def run_scan(symbols: list, min_score: float, workers: int) -> pd.DataFrame:
    total = len(symbols)
    all_rows = []
    done = 0

    print(f"\n  Scanning {total} symbols ({workers} workers)...\n")

    with ThreadPoolExecutor(max_workers=workers) as ex:
        futures = {ex.submit(scan_symbol, sym, min_score): sym for sym in symbols}
        for fut in as_completed(futures):
            sym  = futures[fut]
            done += 1
            rows = fut.result()
            all_rows.extend(rows)
            pct = done/total*100
            bar = "█"*int(pct/5) + "░"*(20-int(pct/5))
            n   = len(all_rows)
            print(f"\r  [{bar}] {pct:5.1f}%  {done}/{total}  {sym:<20}  {n} found",
                  end="", flush=True)

    print()
    return pd.DataFrame(all_rows) if all_rows else pd.DataFrame()


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description="MaverickPICKS Unified Pattern Scanner")
    ap.add_argument("--min_score", type=float, default=40.0)
    ap.add_argument("--workers",   type=int,   default=4)
    ap.add_argument("--out_csv",   type=str,   default=OUTPUT_CSV)
    args = ap.parse_args()

    print(f"\n{'='*60}")
    print("  MaverickPICKS Pattern Scanner")
    print(f"  {datetime.now(IST).strftime('%d-%b-%Y  %H:%M IST')}")
    print(f"  Anchor date : {ANCHOR.strftime('%d-%b-%Y')}")
    print(f"  Min score   : {args.min_score}")
    print(f"  Workers     : {args.workers}")
    print(f"  Patterns    : Bull Flag | Pennant | Sym Triangle | Asc Triangle | Inverse H&S")
    print(f"{'='*60}")

    symbols = load_symbols()
    if not symbols:
        print("  No symbols found. Exiting.")
        return

    t0 = time.time()
    df = run_scan(symbols, args.min_score, args.workers)
    elapsed = time.time() - t0

    print(f"\n  Scan complete in {elapsed/60:.1f} min")

    # Always write CSV — even if empty
    if not df.empty:
        df = df.sort_values("Score", ascending=False).reset_index(drop=True)
        cols = ["Symbol","Pattern","Score","Confidence","Entry_Breakout","Stop_Loss",
                "Target","Risk_Reward","Gap_To_Entry_%","Pattern_Start","Pattern_Expiry",
                "Days_To_Expiry","Pole_Start","Pole_End","Pole_Return_%","Pole_Bars",
                "Consol_Bars","Vol_All3_OK","Vol_Surge","Breakout_Vol_Watch",
                "Weekly_Trend","Narrative"]
        cols = [c for c in cols if c in df.columns]
        df[cols].to_csv(args.out_csv, index=False)
        print(f"  {len(df)} patterns saved → {args.out_csv}")

        # Summary
        print(f"\n{'='*60}")
        for _, r in df.head(10).iterrows():
            print(f"  {r['Symbol']:<18} {r['Pattern']:<22} Score:{r['Score']:>5.1f}  {r['Confidence']}")
    else:
        # Write empty CSV with headers
        pd.DataFrame(columns=["Symbol","Pattern","Score","Confidence","Entry_Breakout",
                               "Stop_Loss","Target","Risk_Reward","Gap_To_Entry_%",
                               "Pattern_Start","Pattern_Expiry","Days_To_Expiry",
                               "Pole_Start","Pole_End","Pole_Return_%","Pole_Bars",
                               "Consol_Bars","Vol_All3_OK","Vol_Surge","Breakout_Vol_Watch",
                               "Weekly_Trend","Narrative"]).to_csv(args.out_csv, index=False)
        print(f"  No patterns found above threshold. Empty {args.out_csv} written.")

    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()

"""
maverick_tracker.py — MaverickPICKS Unified Tracker
=====================================================
Reads scan_results.csv → imports new picks into watchlist.json
Fetches latest price/volume for all watched picks
Determines: on track / off track / breakout / breakdown
Generates: dashboard_picks.html, dashboard_tracker.html
Accumulates: performance_history.json

Usage:
  python maverick_tracker.py                    # import + check + dashboards
  python maverick_tracker.py --check_only       # skip import, just check prices
  python maverick_tracker.py --import_only      # just import, no price check
"""

import argparse
import json
import os
import time
import warnings
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd
import yfinance as yf

warnings.filterwarnings("ignore")

IST              = timezone(timedelta(hours=5, minutes=30))
WATCHLIST_FILE   = "watchlist.json"
HISTORY_FILE     = "performance_history.json"
SCAN_CSV         = "scan_results.csv"
DASH_PICKS       = "dashboard_picks.html"
DASH_TRACKER     = "dashboard_tracker.html"

# States
WATCHING  = "WATCHING"
BREAKOUT  = "BREAKOUT"
BREAKDOWN = "BREAKDOWN"
EXPIRED   = "EXPIRED"


# ─────────────────────────────────────────────────────────────────────────────
# DATE UTILITIES
# ─────────────────────────────────────────────────────────────────────────────

def _anchor() -> datetime:
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


ANCHOR = _anchor()


def _fmt(ts) -> str:
    try: return pd.Timestamp(ts).strftime("%d-%b-%Y")
    except: return ""


# ─────────────────────────────────────────────────────────────────────────────
# PERSISTENCE
# ─────────────────────────────────────────────────────────────────────────────

def load_wl() -> dict:
    if os.path.exists(WATCHLIST_FILE):
        with open(WATCHLIST_FILE) as f:
            return json.load(f)
    return {"picks": {}, "last_updated": None}


def save_wl(wl: dict):
    wl["last_updated"] = datetime.now(IST).strftime("%Y-%m-%d %H:%M IST")
    with open(WATCHLIST_FILE, "w") as f:
        json.dump(wl, f, indent=2, default=str)


def load_history() -> dict:
    if os.path.exists(HISTORY_FILE):
        with open(HISTORY_FILE) as f:
            return json.load(f)
    return {"outcomes": [], "last_updated": None}


def save_history(h: dict):
    h["last_updated"] = datetime.now(IST).strftime("%Y-%m-%d %H:%M IST")
    with open(HISTORY_FILE, "w") as f:
        json.dump(h, f, indent=2, default=str)


# ─────────────────────────────────────────────────────────────────────────────
# PRICE FETCH
# ─────────────────────────────────────────────────────────────────────────────

def fetch_price(symbol: str) -> tuple:
    """Returns (close, volume, date_str) or (None, None, None)."""
    ticker = symbol if symbol.endswith(".NS") else symbol + ".NS"
    end    = ANCHOR + timedelta(days=1)
    start  = ANCHOR - timedelta(days=10)
    for attempt in range(3):
        try:
            df = yf.download(ticker, start=start, end=end,
                             interval="1d", auto_adjust=False,
                             progress=False, threads=False)
            if df.empty: raise ValueError("empty")
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = [c[0] for c in df.columns]
            df = df[["Close","Volume"]].dropna()
            last = df.iloc[-1]
            return float(last["Close"]), float(last["Volume"]), df.index[-1].strftime("%Y-%m-%d")
        except Exception:
            time.sleep(1)
    return None, None, None


# ─────────────────────────────────────────────────────────────────────────────
# IMPORT FROM SCAN CSV
# ─────────────────────────────────────────────────────────────────────────────

def import_picks(wl: dict) -> tuple:
    if not os.path.exists(SCAN_CSV):
        print(f"  [SKIP] {SCAN_CSV} not found")
        return 0, 0

    df = pd.read_csv(SCAN_CSV)
    if df.empty:
        print(f"  [SKIP] {SCAN_CSV} is empty")
        return 0, 0

    added = skipped = 0
    scan_date = datetime.now(IST).strftime("%Y-%m-%d")

    for _, row in df.iterrows():
        sym     = str(row.get("Symbol","")).strip()
        pattern = str(row.get("Pattern","")).strip()
        if not sym or not pattern: continue

        key = f"{sym}|{pattern}"

        # Skip if already watching this exact pick
        if key in wl["picks"] and wl["picks"][key]["state"] == WATCHING:
            skipped += 1
            continue

        # Support both old column names (patterns_today.csv) and new (scan_results.csv)
        def _f(row, *keys, default=0.0):
            for k in keys:
                v = row.get(k)
                if v is not None and str(v).strip() not in ("","nan","None"):
                    try: return float(v)
                    except: pass
            return default

        def _s(row, *keys, default=""):
            for k in keys:
                v = row.get(k)
                if v is not None and str(v).strip() not in ("","nan","None"):
                    return str(v)
            return default

        def _b(row, *keys):
            for k in keys:
                v = row.get(k)
                if v is not None:
                    return str(v).lower() in ("true","1","yes")
            return False

        wl["picks"][key] = {
            "symbol":        sym,
            "pattern":       pattern,
            "state":         WATCHING,
            "date_added":    scan_date,
            "score":         _f(row, "Score"),
            "confidence":    _s(row, "Confidence"),
            "entry":         _f(row, "Entry_Breakout", "Breakout_Level"),
            "stop_loss":     _f(row, "Stop_Loss"),
            "target":        _f(row, "Target", "Target_1"),
            "risk_reward":   _f(row, "Risk_Reward"),
            "gap_pct":       _f(row, "Gap_To_Entry_%", "Gap_To_Break_%"),
            "pattern_start": _s(row, "Pattern_Start"),
            "pattern_expiry":_s(row, "Pattern_Expiry"),
            "days_to_expiry":row.get("Days_To_Expiry"),
            "pole_return":   _f(row, "Pole_Return_%"),
            "vol_all3":      _b(row, "Vol_All3_OK", "Vol_All_3_OK"),
            "breakout_vol":  _f(row, "Breakout_Vol_Watch"),
            "weekly_trend":  _s(row, "Weekly_Trend", "Weekly_Confirmed"),
            "narrative":     _s(row, "Narrative"),
            "price_history":[],
            "track_signal": "NEUTRAL",
            "track_detail": "",
            "eta_days":     None,
            "date_resolved":None,
            "resolved_note":None,
        }
        added += 1

    print(f"  Import: {added} added, {skipped} already watching")
    return added, skipped


# ─────────────────────────────────────────────────────────────────────────────
# TRACK SIGNAL
# ─────────────────────────────────────────────────────────────────────────────

def _track_signal(pick: dict, close: float, volume: float) -> tuple:
    """
    Returns (signal, detail, eta_days).
    ON TRACK    = price approaching entry + volume contracting
    OFF TRACK   = price moving away or volume expanding
    NEUTRAL     = mixed signals
    """
    h       = pick.get("price_history", [])
    entry   = pick["entry"]
    stop    = pick["stop_loss"]
    signals_bull = 0
    signals_bear = 0
    details = []

    # Gap trend over last 5 days
    gaps = [x["gap_pct"] for x in h[-5:] if x.get("gap_pct") is not None]
    if len(gaps) >= 2:
        vel = float(np.mean(np.diff(gaps)))
        if vel < -0.3:
            signals_bull += 2
            details.append(f"gap closing {abs(vel):.2f}%/day ✓")
        elif vel > 0.3:
            signals_bear += 2
            details.append(f"gap widening {vel:.2f}%/day ✗")
    else:
        vel = None

    # Volume trend over last 3 days
    vols = [x["volume"] for x in h[-4:] if x.get("volume")]
    if len(vols) >= 3:
        if vols[-1] < vols[0] * 0.85:
            signals_bull += 1
            details.append("vol contracting ✓")
        elif vols[-1] > vols[0] * 1.15:
            signals_bear += 1
            details.append("vol expanding ✗")

    # Distance to stop
    if stop > 0:
        dist_stop = (close - stop) / stop * 100
        if dist_stop < 2:
            signals_bear += 2
            details.append(f"near stop ({dist_stop:.1f}%) ✗")

    # Current gap
    gap = (entry - close) / close * 100 if close > 0 else 0
    if gap < 2:
        signals_bull += 1
        details.append(f"only {gap:.1f}% from entry ✓")

    eta = None
    if vel and vel < 0 and gap > 0:
        days = gap / abs(vel)
        if days < 30:
            eta = int(round(days))

    if signals_bull >= 3 and signals_bull > signals_bear:
        return "ON TRACK → BREAKOUT", " | ".join(details), eta
    elif signals_bear >= 3 and signals_bear > signals_bull:
        return "CAUTION → OFF TRACK", " | ".join(details), None
    else:
        return "NEUTRAL", " | ".join(details), eta


# ─────────────────────────────────────────────────────────────────────────────
# DAILY CHECK
# ─────────────────────────────────────────────────────────────────────────────

def check_all(wl: dict, history: dict) -> dict:
    picks    = wl["picks"]
    watching = [k for k,p in picks.items() if p["state"] == WATCHING]
    alerts   = []
    still_watching = []
    today    = ANCHOR.strftime("%Y-%m-%d")

    print(f"\n  Checking {len(watching)} watching pick(s)...")

    for key in watching:
        pick = picks[key]
        sym  = pick["symbol"]
        print(f"    {sym:<20} {pick['pattern']:<22}", end=" ")

        close, volume, date_str = fetch_price(sym)

        if close is None:
            print("→ DATA ERROR")
            still_watching.append(pick)
            continue

        entry  = pick["entry"]
        stop   = pick["stop_loss"]
        tgt    = pick["target"]
        bvw    = pick.get("breakout_vol", 0)
        expiry = pick.get("pattern_expiry","")
        gap    = round((entry - close)/close*100, 2) if close > 0 else 0

        # Append to price history
        pick["price_history"].append({
            "date": date_str, "close": round(close,2),
            "volume": int(volume), "gap_pct": gap,
        })

        # State evaluation
        if close >= entry:
            vol_ok = volume >= bvw if bvw > 0 else True
            pick["state"]          = BREAKOUT
            pick["date_resolved"]  = date_str
            pick["resolved_note"]  = (
                f"BREAKOUT ✓ — closed ₹{close:.2f} above entry ₹{entry:.2f}. "
                + (f"Volume {volume:,.0f} ✓ confirmed." if vol_ok
                   else f"⚠ Volume {volume:,.0f} below watch level {bvw:,.0f} — verify on chart.")
            )
            print(f"→ ₹{close:.2f}  🟢 BREAKOUT")
            alerts.append(pick)
            # Log to performance history
            _log_outcome(pick, history)

        elif close <= stop:
            pick["state"]         = BREAKDOWN
            pick["date_resolved"] = date_str
            pick["resolved_note"] = f"BREAKDOWN ✗ — closed ₹{close:.2f} below stop ₹{stop:.2f}."
            print(f"→ ₹{close:.2f}  🔴 BREAKDOWN")
            alerts.append(pick)
            _log_outcome(pick, history)

        elif expiry:
            try:
                exp_dt = pd.to_datetime(expiry, dayfirst=True)
                if pd.Timestamp(ANCHOR) > exp_dt:
                    pick["state"]         = EXPIRED
                    pick["date_resolved"] = date_str
                    pick["resolved_note"] = f"EXPIRED — pattern passed deadline {_fmt(exp_dt)}."
                    print(f"→ ₹{close:.2f}  ⚫ EXPIRED")
                    alerts.append(pick)
                    _log_outcome(pick, history)
                    picks[key] = pick
                    continue
            except Exception:
                pass
            # Still watching
            sig, detail, eta = _track_signal(pick, close, volume)
            pick["track_signal"] = sig
            pick["track_detail"] = detail
            pick["eta_days"]     = eta
            pick["gap_pct"]      = gap
            print(f"→ ₹{close:.2f}  {sig}")
            still_watching.append(pick)

        else:
            sig, detail, eta = _track_signal(pick, close, volume)
            pick["track_signal"] = sig
            pick["track_detail"] = detail
            pick["eta_days"]     = eta
            pick["gap_pct"]      = gap
            print(f"→ ₹{close:.2f}  {sig}")
            still_watching.append(pick)

        picks[key] = pick

    return {"alerts": alerts, "watching": still_watching,
            "all_picks": picks, "date": today}


def _log_outcome(pick: dict, history: dict):
    """Log a resolved pick to performance history."""
    h       = pick.get("price_history",[])
    entry_p = pick["entry"]
    exit_p  = h[-1]["close"] if h else None
    pnl     = round((exit_p - entry_p)/entry_p*100, 2) if exit_p and entry_p else None
    days    = len(h)

    existing = {o["_key"] for o in history["outcomes"]}
    key      = f"{pick['symbol']}|{pick['pattern']}|{pick.get('date_added','')}"
    if key in existing: return

    history["outcomes"].append({
        "_key":        key,
        "symbol":      pick["symbol"],
        "pattern":     pick["pattern"],
        "source":      "scanner",
        "pick_date":   pick.get("date_added",""),
        "exit_date":   pick.get("date_resolved",""),
        "days":        days,
        "state":       pick["state"],
        "is_win":      pick["state"] == BREAKOUT,
        "is_loss":     pick["state"] == BREAKDOWN,
        "score":       pick.get("score",0),
        "confidence":  pick.get("confidence",""),
        "vol_all3":    pick.get("vol_all3", False),
        "weekly":      pick.get("weekly_trend",""),
        "entry_price": entry_p,
        "exit_price":  exit_p,
        "pnl_pct":     pnl,
    })


# ─────────────────────────────────────────────────────────────────────────────
# DASHBOARD 1: TODAY'S PICKS (from scan_results.csv)
# ─────────────────────────────────────────────────────────────────────────────

def _conf_color(conf):
    return {"HIGH":"#166534","MEDIUM":"#92400e","LOW":"#6b7280"}.get(conf,"#374151")


def _score_bar(score):
    w   = int(score)
    col = "#166534" if score>=70 else ("#f59e0b" if score>=50 else "#ef4444")
    return (f'<div style="display:flex;align-items:center;gap:6px">'
            f'<div style="width:80px;height:6px;background:#e5e7eb;border-radius:3px">'
            f'<div style="width:{w}%;height:6px;background:{col};border-radius:3px"></div></div>'
            f'<span style="font-size:12px;color:#374151">{score:.0f}</span></div>')


def _rv(row, *keys, default=0.0):
    """Read value from CSV row trying multiple column names."""
    for k in keys:
        try:
            v = row.get(k)
            if v is not None and str(v).strip() not in ("","nan","None","0","0.0"):
                return float(v)
        except Exception:
            pass
    return default


def generate_picks_dashboard():
    """
    Generate dashboard_picks.html from patterns_today.csv.

    BUG FIXES (2026-06):
      1. Now reads patterns_today.csv (pattern_detector_v2 output) instead of
         scan_results.csv (main_v3 output). scan_results.csv has Entry_Breakout
         and Target columns that are 0 for pattern picks — the correct field
         names in patterns_today.csv are Breakout_Level and Target_1.
      2. Expiry date is now parsed from the Narrative text (no standalone
         Expiry_Date column exists in patterns_today.csv). Also supports the
         Expiry_Date column if pattern_detector_v2 is updated to emit it.
      3. _rv() now correctly reads Breakout_Level / Target_1 from the CSV.
    """
    # Read scan_results.csv — this is what maverick_scanner.py writes during the workflow.
    # patterns_today.csv is only written by pattern_detector_v2.py which is not in the workflow.
    PATTERNS_CSV = SCAN_CSV   # scan_results.csv
    now = datetime.now(IST).strftime("%d-%b-%Y %H:%M IST")

    if not os.path.exists(PATTERNS_CSV):
        html = f"""<!DOCTYPE html><html><body style="font-family:sans-serif;padding:40px">
        <h2>MaverickPICKS — Today's Picks</h2>
        <p style="color:#6b7280">No scan results found. Run maverick_scanner.py first.</p>
        </body></html>"""
        with open(DASH_PICKS,"w",encoding="utf-8") as f: f.write(html)
        return

    df = pd.read_csv(PATTERNS_CSV)
    n  = len(df)

    # Stats — use correct column names from pattern_detector_v2 output
    n_high = sum(df["Confidence"] == "HIGH") if "Confidence" in df.columns else 0
    n_vol  = sum(df["Vol_All_3_OK"] == True) if "Vol_All_3_OK" in df.columns else \
             sum(df["Vol_All3_OK"]  == True) if "Vol_All3_OK"  in df.columns else 0
    patterns = df["Pattern"].value_counts().to_dict() if "Pattern" in df.columns else {}

    def _parse_expiry(row):
        """
        Extract expiry date from either:
          a) Expiry_Date column (if pattern_detector_v2 emits it — future proof)
          b) Narrative text — parses "Valid until DD-Mon-YYYY" or
             "EXPIRING SOON ... by DD-Mon-YYYY"
        Returns (expiry_str, days_remaining_int_or_None)
        """
        # (a) Standalone column — preferred, future proof
        for col in ("Expiry_Date", "Pattern_Expiry"):
            val = row.get(col)
            if val and str(val).strip() not in ("", "nan", "None"):
                exp_str = str(val).strip()
                try:
                    exp_dt  = pd.to_datetime(exp_str, dayfirst=True)
                    days_e  = (exp_dt - pd.Timestamp.today().normalize()).days
                    return exp_str, days_e
                except Exception:
                    return exp_str, None

        # (b) Parse from Narrative text
        narr = str(row.get("Narrative", ""))
        try:
            if "Valid until" in narr:
                part = narr.split("Valid until")[1].split("(")[0].strip()
                exp_dt = pd.to_datetime(part, dayfirst=True)
                days_e = (exp_dt - pd.Timestamp.today().normalize()).days
                return exp_dt.strftime("%d-%b-%Y"), days_e
            if "EXPIRING SOON" in narr and "by" in narr:
                part = narr.split("EXPIRING SOON")[1]
                part = part.split("by")[1].split(")")[0].strip().rstrip(".")
                exp_dt = pd.to_datetime(part, dayfirst=True)
                days_e = (exp_dt - pd.Timestamp.today().normalize()).days
                return exp_dt.strftime("%d-%b-%Y"), days_e
        except Exception:
            pass

        return "", None

    def _parse_formed(row):
        """
        Extract pattern formation start date from Narrative.
        Looks for the consolidation/triangle start date in Step 2.
        """
        for col in ("Pattern_Start", "Formed"):
            val = row.get(col)
            if val and str(val).strip() not in ("", "nan", "None"):
                return str(val).strip()

        narr = str(row.get("Narrative", ""))
        # Bull Flag / Pennant: "[2. FLAG] From DD-Mon-YYYY to ..."
        # Triangle:            "[2. TRIANGLE] From DD-Mon-YYYY to ..."
        import re
        m = re.search(r'\[2\..*?\] From (\d{2}-[A-Za-z]{3}-\d{4})', narr)
        if m:
            return m.group(1)
        return ""

    # Pre-compute empty-state fallback outside f-string (backslashes not allowed
    # inside f-string expressions in Python < 3.12)
    _empty_row = ('<tr><td colspan="13" style="text-align:center;'
                  'padding:40px;color:#9ca3af">No patterns detected today</td></tr>')

    # ── Fetch CMP: one ticker at a time using existing fetch_price() ─────────
    # Batch yfinance downloads have proven unreliable in this environment due to
    # MultiIndex column structure varying by yfinance version. Using individual
    # fetch_price() calls (which already work for the watchlist check) is safer.
    # fetch_price() already handles retries, ANCHOR date, and .NS suffix.
    print("  Fetching CMP for pattern symbols...")
    _symbols = df["Symbol"].dropna().tolist()
    _cmp_map = {}
    for _sym in _symbols:
        # Strip .NS before passing to fetch_price() — it adds .NS itself.
        # Without this, CONCORDBIO.NS becomes CONCORDBIO.NS.NS and returns None.
        _sym_clean = _sym.replace(".NS", "").replace(".BO", "")
        _sym_key   = _sym_clean   # map key = bare symbol e.g. CONCORDBIO
        try:
            _close, _vol, _date = fetch_price(_sym_clean)
            if _close is not None:
                _cmp_map[_sym_key] = _close
                print(f"    CMP {_sym_key}: Rs{_close:.2f} ({_date})")
            else:
                print(f"    CMP {_sym_key}: fetch returned None")
        except Exception as _e:
            print(f"    CMP {_sym_key}: exception — {_e}")
    print(f"  CMP fetched for {len(_cmp_map)}/{len(_symbols)} symbols")

    rows = ""
    for _, r in df.iterrows():
        # ── CMP: look up from pre-fetched batch map ───────────────────────────
        sym_raw  = str(r.get("Symbol", "")).replace(".NS", "")
        cmp_val  = _cmp_map.get(sym_raw)
        if cmp_val:
            # Colour: green = at/above breakout, red = within 2% of stop, neutral otherwise
            _entry_chk = _rv(r, "Breakout_Level", "Entry_Breakout", "Entry")
            _stop_chk  = _rv(r, "Stop_Loss")
            if cmp_val >= _entry_chk and _entry_chk > 0:
                cmp_color = "#16a34a"   # green — at/above breakout level
            elif _stop_chk > 0 and cmp_val <= _stop_chk * 1.02:
                cmp_color = "#dc2626"   # red — dangerously close to stop
            else:
                cmp_color = "#1e293b"   # neutral
            cmp_html = f'<span style="font-weight:600;color:{cmp_color}">₹{cmp_val:,.2f}</span>'
        else:
            cmp_html = '<span style="color:#9ca3af">—</span>'

        # ── ENTRY: Breakout_Level is the correct column in patterns_today.csv ──
        entry_val  = _rv(r, "Breakout_Level", "Entry_Breakout", "Entry")

        # ── STOP: Stop_Loss — same name in both CSVs ─────────────────────────
        stop_val   = _rv(r, "Stop_Loss")

        # ── TARGET: Target_1 is the correct column in patterns_today.csv ──────
        target_val = _rv(r, "Target_1", "Target")

        # ── EXPIRY: parse from Narrative (no standalone column in CSV yet) ────
        exp_str, days_e = _parse_expiry(r)

        # ── FORMED: pattern start date from Narrative ─────────────────────────
        formed_str = _parse_formed(r)

        # ── EXPIRY colour: red if ≤2 trading days left ────────────────────────
        exp_col = "#dc2626" if (
            days_e is not None and days_e <= 2
        ) else "#f59e0b" if (
            days_e is not None and days_e <= 5
        ) else "#475569"

        # ── Vol check: Vol_All_3_OK is the v2 column name ─────────────────────
        vol_ok = bool(r.get("Vol_All_3_OK", r.get("Vol_All3_OK", False)))

        # ── Narrative: split on separator and render as sections ──────────────
        narr      = str(r.get("Narrative", ""))
        narr_html = ""
        for chunk in narr.split("  //  "):
            if chunk.strip():
                narr_html += (
                    f'<div style="margin-bottom:6px;padding:6px 10px;'
                    f'background:#f8fafc;border-left:3px solid #e2e8f0;'
                    f'border-radius:0 4px 4px 0;font-size:11px;color:#475569;'
                    f'line-height:1.6">{chunk.strip()}</div>'
                )

        # ── Expiry display: show days remaining badge ─────────────────────────
        if exp_str:
            if days_e is not None and days_e <= 0:
                exp_display = f'<span style="color:#dc2626;font-weight:600">{exp_str} (EXPIRED)</span>'
            elif days_e is not None and days_e <= 2:
                exp_display = f'<span style="color:#dc2626;font-weight:600">{exp_str} ⚠ {days_e}d left</span>'
            elif days_e is not None and days_e <= 5:
                exp_display = f'<span style="color:#f59e0b;font-weight:600">{exp_str} ({days_e}d left)</span>'
            else:
                exp_display = f'<span style="color:#475569">{exp_str}</span>'
                if days_e is not None:
                    exp_display = f'<span style="color:#475569">{exp_str} ({days_e}d)</span>'
        else:
            exp_display = "—"

        rows += f"""
        <tr>
          <td style="padding:12px 10px;font-weight:700;font-size:13px;vertical-align:top">{str(r.get("Symbol","")).replace(".NS","")}</td>
          <td style="padding:12px 10px;font-size:12px;color:#475569;vertical-align:top">{r.get("Pattern","")}</td>
          <td style="padding:12px 10px;vertical-align:top">{_score_bar(float(r.get("Score",0)))}</td>
          <td style="padding:12px 10px;vertical-align:top;font-weight:600;color:{_conf_color(str(r.get('Confidence','')))}">{r.get("Confidence","")}</td>
          <td style="padding:12px 10px;text-align:right;vertical-align:top">{cmp_html}</td>
          <td style="padding:12px 10px;text-align:right;font-weight:600;color:#1d4ed8;vertical-align:top">₹{entry_val:,.2f}</td>
          <td style="padding:12px 10px;text-align:right;color:#dc2626;vertical-align:top">₹{stop_val:,.2f}</td>
          <td style="padding:12px 10px;text-align:right;color:#16a34a;vertical-align:top">₹{target_val:,.2f}</td>
          <td style="padding:12px 10px;text-align:center;vertical-align:top">{float(r.get("Risk_Reward",0)):.1f}x</td>
          <td style="padding:12px 10px;text-align:center;vertical-align:top">{"✓" if vol_ok else "✗"}</td>
          <td style="padding:12px 10px;font-size:11px;vertical-align:top">{exp_display}</td>
          <td style="padding:12px 10px;font-size:11px;color:#475569;vertical-align:top">{formed_str}</td>
          <td style="padding:12px 10px;vertical-align:top;max-width:400px">
            <details>
              <summary style="cursor:pointer;font-size:11px;color:#1d4ed8;font-weight:600">
                View narrative ▼</summary>
              <div style="margin-top:6px">{narr_html}</div>
            </details>
          </td>
        </tr>"""

    pat_summary = "  ".join(f"{k}: {v}" for k,v in patterns.items())

    html = f"""<!DOCTYPE html><html lang="en"><head>
<meta charset="UTF-8">
<title>MaverickPICKS — Today's Picks</title>
<style>
* {{ box-sizing:border-box; margin:0; padding:0 }}
body {{ font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",Arial,sans-serif;
       background:#f1f5f9; color:#1e293b }}
.hdr {{ background:linear-gradient(135deg,#0f172a,#1d4ed8);color:#fff;padding:20px 28px }}
.hdr h1 {{ font-size:20px;font-weight:700 }}
.hdr p  {{ font-size:12px;opacity:.75;margin-top:3px }}
.stats {{ display:flex;gap:12px;padding:16px 28px;flex-wrap:wrap }}
.stat {{ background:#fff;border:1px solid #e2e8f0;border-radius:10px;
         padding:12px 18px;min-width:110px }}
.stat .n {{ font-size:24px;font-weight:700 }}
.stat .l {{ font-size:11px;color:#64748b;margin-top:2px }}
.wrap {{ padding:0 28px 28px;overflow-x:auto }}
table {{ width:100%;border-collapse:collapse;background:#fff;
         border-radius:10px;overflow:hidden;
         box-shadow:0 1px 2px rgba(0,0,0,.06) }}
thead tr {{ background:#0f172a;color:#fff }}
thead th {{ padding:10px 10px;font-size:11px;font-weight:600;
            text-transform:uppercase;letter-spacing:.04em;
            white-space:nowrap;text-align:left }}
tbody tr {{ border-bottom:1px solid #f1f5f9 }}
tbody tr:hover {{ background:#f8fafc }}
.note {{ font-size:11px;color:#64748b;padding:8px 28px 20px }}
</style></head><body>
<div class="hdr">
  <h1>MaverickPICKS — Today's Pattern Picks</h1>
  <p>Scan date: {now} &nbsp;|&nbsp; {n} pattern(s) detected &nbsp;|&nbsp; {pat_summary}</p>
</div>
<div class="stats">
  <div class="stat"><div class="n">{n}</div><div class="l">Total Picks</div></div>
  <div class="stat"><div class="n" style="color:#166534">{n_high}</div><div class="l">HIGH Confidence</div></div>
  <div class="stat"><div class="n" style="color:#1d4ed8">{n_vol}</div><div class="l">Vol All 3 ✓</div></div>
</div>
<div class="wrap">
<table>
  <thead><tr>
    <th>Symbol</th><th>Pattern</th><th>Score</th><th>Conf</th>
    <th>CMP</th><th>Entry</th><th>Stop</th><th>Target</th><th>R:R</th>
    <th>Vol✓</th><th>Expiry</th><th>Formed</th><th>Narrative & Reason</th>
  </tr></thead>
  <tbody>{_empty_row if not rows else rows}</tbody>
</table>
</div>
<p class="note">
  Score = Murphy pattern quality 0-100. Entry = close above this price on required volume to confirm breakout.
  Vol✓ = all 3 Murphy volume checks passed (pole surge + consol contraction + declining trend).
  Expiry = last valid date per Murphy time rules — remove from watchlist if not broken out by then.
</p>
</body></html>"""

    with open(DASH_PICKS,"w",encoding="utf-8") as f:
        f.write(html)
    print(f"  Dashboard → {DASH_PICKS}  ({n} picks, reading from {PATTERNS_CSV})")


# ─────────────────────────────────────────────────────────────────────────────
# DASHBOARD 2: TRACKER (active watchlist + performance)
# ─────────────────────────────────────────────────────────────────────────────

def generate_tracker_dashboard(wl: dict, history: dict):
    now     = datetime.now(IST).strftime("%d-%b-%Y %H:%M IST")
    picks   = list(wl.get("picks",{}).values())
    outcomes= history.get("outcomes",[])

    watching  = sorted([p for p in picks if p["state"]==WATCHING],
                       key=lambda x: x.get("gap_pct",99))
    resolved  = [p for p in picks if p["state"]!=WATCHING]
    breakouts = [p for p in resolved if p["state"]==BREAKOUT]
    breakdowns= [p for p in resolved if p["state"]==BREAKDOWN]

    # Performance stats
    total_res = len(outcomes)
    wins      = sum(1 for o in outcomes if o.get("is_win"))
    losses    = sum(1 for o in outcomes if o.get("is_loss"))
    win_rate  = round(wins/total_res*100,1) if total_res>0 else 0
    avg_pnl   = round(float(np.mean([o["pnl_pct"] for o in outcomes if o.get("pnl_pct") is not None])),2) if outcomes else 0

    def track_badge(sig):
        cfg = {
            "ON TRACK → BREAKOUT":  ("#166534","#dcfce7","🟢"),
            "CAUTION → OFF TRACK":  ("#991b1b","#fee2e2","🔴"),
            "NEUTRAL":              ("#374151","#f3f4f6","⚪"),
        }.get(sig, ("#374151","#f3f4f6","—"))
        return (f'<span style="background:{cfg[1]};color:{cfg[0]};padding:2px 8px;'
                f'border-radius:10px;font-size:11px;font-weight:600">{cfg[2]} {sig}</span>')

    def gap_bar(gap):
        try:
            g   = float(gap)
            pct = max(0, min(100, int((1-g/20)*100)))
            col = "#22c55e" if g<2 else ("#f59e0b" if g<5 else "#94a3b8")
            return (f'<div style="display:flex;align-items:center;gap:5px">'
                    f'<div style="width:60px;height:6px;background:#e5e7eb;border-radius:3px">'
                    f'<div style="width:{pct}%;height:6px;background:{col};border-radius:3px"></div></div>'
                    f'<span style="font-size:11px;color:#6b7280">{g:.1f}%</span></div>')
        except: return "—"

    def sparkline(history_list):
        if not history_list or len(history_list)<2:
            return '<span style="color:#9ca3af">—</span>'
        closes = [h["close"] for h in history_list[-10:]]
        mn,mx  = min(closes),max(closes)
        rng    = mx-mn if mx>mn else 1
        W,H    = 60,18
        pts    = []
        for i,c in enumerate(closes):
            x = int(i/(len(closes)-1)*W)
            y = H - int((c-mn)/rng*H)
            pts.append(f"{x},{y}")
        col = "#22c55e" if closes[-1]>=closes[0] else "#ef4444"
        return (f'<svg width="{W}" height="{H}">'
                f'<polyline points="{" ".join(pts)}" fill="none" stroke="{col}" stroke-width="1.5"/>'
                f'</svg>')

    # Watching rows
    watch_rows = ""
    for p in watching:
        h     = p.get("price_history",[])
        last_h= h[-1] if h else {}
        close = last_h.get("close","—")
        vol   = last_h.get("volume","—")
        gap   = last_h.get("gap_pct", p.get("gap_pct",0))
        sig   = p.get("track_signal","NEUTRAL")
        eta   = p.get("eta_days")
        det   = p.get("track_detail","")
        days  = len(h)
        exp   = p.get("pattern_expiry","—")

        watch_rows += f"""<tr>
          <td style="padding:10px;font-weight:700;font-size:13px">{p["symbol"].replace(".NS","")}</td>
          <td style="padding:10px;font-size:11px;color:#475569">{p["pattern"]}</td>
          <td style="padding:10px">{track_badge(sig)}</td>
          <td style="padding:10px;font-size:11px;color:#475569;max-width:200px">{det[:80]}</td>
          <td style="padding:10px;text-align:center;color:#64748b">{'~'+str(eta)+'d' if eta else '—'}</td>
          <td style="padding:10px;text-align:right;font-weight:600">
            {'₹'+f'{close:,.2f}' if isinstance(close,(int,float)) else '—'}</td>
          <td style="padding:10px">{gap_bar(gap)}</td>
          <td style="padding:10px;text-align:right;color:#1d4ed8;font-weight:600">₹{p["entry"]:,.2f}</td>
          <td style="padding:10px;text-align:right;color:#dc2626">₹{p["stop_loss"]:,.2f}</td>
          <td style="padding:10px;text-align:right;color:#16a34a">₹{p["target"]:,.2f}</td>
          <td style="padding:10px;text-align:center">{p.get("risk_reward",0):.1f}x</td>
          <td style="padding:10px;text-align:center;font-size:11px;color:#475569">{days}d</td>
          <td style="padding:10px;font-size:11px;color:#6b7280">{exp}</td>
          <td style="padding:10px">{sparkline(h)}</td>
        </tr>"""

    # Resolved rows
    res_rows = ""
    for p in sorted(resolved, key=lambda x: x.get("date_resolved",""), reverse=True):
        h     = p.get("price_history",[])
        ep    = h[0]["close"] if h else p["entry"]
        xp    = h[-1]["close"] if h else None
        pnl   = round((xp-ep)/ep*100,2) if xp and ep else None
        pnl_c = "#166534" if pnl and pnl>0 else "#dc2626"
        state_c = {"BREAKOUT":"#166534","BREAKDOWN":"#991b1b","EXPIRED":"#6b7280"}.get(p["state"],"#374151")
        res_rows += f"""<tr>
          <td style="padding:8px 10px;font-weight:700">{p["symbol"].replace(".NS","")}</td>
          <td style="padding:8px 10px;font-size:11px">{p["pattern"]}</td>
          <td style="padding:8px 10px;font-weight:600;color:{state_c}">{p["state"]}</td>
          <td style="padding:8px 10px;font-size:11px">{p.get("date_added","")}</td>
          <td style="padding:8px 10px;font-size:11px">{p.get("date_resolved","")}</td>
          <td style="padding:8px 10px;text-align:center">{len(h)}d</td>
          <td style="padding:8px 10px;text-align:right">₹{p["entry"]:,.2f}</td>
          <td style="padding:8px 10px;text-align:right">{'₹'+f'{xp:,.2f}' if xp else '—'}</td>
          <td style="padding:8px 10px;text-align:right;font-weight:600;color:{pnl_c}">
            {f'{pnl:+.1f}%' if pnl is not None else '—'}</td>
          <td style="padding:8px 10px;font-size:11px;color:#475569;max-width:250px">
            {str(p.get("resolved_note",""))[:120]}</td>
        </tr>"""

    # Performance section
    need_more = max(0, 30-total_res)
    perf_html = ""
    if total_res > 0:
        # By pattern
        by_pat = {}
        for o in outcomes:
            pat = o.get("pattern","?")
            by_pat.setdefault(pat, {"wins":0,"losses":0,"pnls":[]})
            if o.get("is_win"): by_pat[pat]["wins"]+=1
            if o.get("is_loss"): by_pat[pat]["losses"]+=1
            if o.get("pnl_pct") is not None: by_pat[pat]["pnls"].append(o["pnl_pct"])

        pat_rows = ""
        for pat, d in sorted(by_pat.items()):
            n = d["wins"]+d["losses"]
            hr= round(d["wins"]/n*100,1) if n>0 else 0
            ap= round(float(np.mean(d["pnls"])),1) if d["pnls"] else 0
            pat_rows += f"""<tr>
              <td style="padding:6px 10px">{pat}</td>
              <td style="padding:6px 10px;text-align:center">{n}</td>
              <td style="padding:6px 10px;text-align:center;color:{'#166534' if hr>=60 else '#991b1b'};font-weight:600">{hr}%</td>
              <td style="padding:6px 10px;text-align:center;color:{'#166534' if ap>0 else '#991b1b'};font-weight:600">{ap:+.1f}%</td>
            </tr>"""

        perf_html = f"""
        <div style="margin-bottom:20px">
          <div style="font-size:13px;font-weight:600;color:#475569;text-transform:uppercase;
                      letter-spacing:.05em;margin-bottom:8px;border-left:3px solid #a855f7;
                      padding-left:10px">📊 Performance by Pattern</div>
          {'<div style="background:#fef3c7;border:1px solid #fbbf24;border-radius:8px;padding:10px;font-size:12px;color:#92400e;margin-bottom:10px">⏳ Collecting data — ' + str(total_res) + ' of 30 picks needed. Need ' + str(need_more) + ' more.</div>' if total_res < 30 else ''}
          <table style="width:100%;border-collapse:collapse;background:#fff;border-radius:8px;font-size:12px;overflow:hidden;box-shadow:0 1px 2px rgba(0,0,0,.06)">
            <thead style="background:#0f172a;color:#fff">
              <tr><th style="padding:8px 10px;text-align:left">Pattern</th>
                  <th style="padding:8px 10px">Picks</th>
                  <th style="padding:8px 10px">Hit Rate</th>
                  <th style="padding:8px 10px">Avg P&L</th></tr>
            </thead>
            <tbody>{pat_rows}</tbody>
          </table>
        </div>"""

    CSS = """<style>
* { box-sizing:border-box; margin:0; padding:0 }
body { font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",Arial,sans-serif;
       background:#f1f5f9; color:#1e293b }
.hdr { background:linear-gradient(135deg,#0f172a,#7c3aed);color:#fff;padding:20px 28px }
.hdr h1 { font-size:20px;font-weight:700 }
.hdr p  { font-size:12px;opacity:.75;margin-top:3px }
.stats { display:flex;gap:12px;padding:16px 28px;flex-wrap:wrap }
.stat { background:#fff;border:1px solid #e2e8f0;border-radius:10px;
        padding:12px 18px;min-width:110px }
.stat .n { font-size:24px;font-weight:700 }
.stat .l { font-size:11px;color:#64748b;margin-top:2px }
.section { padding:0 28px 24px }
.sec-title { font-size:13px;font-weight:600;color:#475569;text-transform:uppercase;
             letter-spacing:.05em;margin:16px 0 8px;border-left:3px solid #3b82f6;
             padding-left:10px }
table { width:100%;border-collapse:collapse;background:#fff;border-radius:10px;
        overflow:hidden;box-shadow:0 1px 2px rgba(0,0,0,.06);font-size:12px }
thead tr { background:#0f172a;color:#fff }
thead th { padding:9px 10px;font-size:11px;font-weight:600;text-transform:uppercase;
           letter-spacing:.04em;white-space:nowrap;text-align:left }
tbody tr { border-bottom:1px solid #f1f5f9 }
tbody tr:hover { background:#f8fafc }
.note { font-size:11px;color:#64748b;padding:8px 28px 20px }
</style>"""

    html = f"""<!DOCTYPE html><html lang="en"><head>
<meta charset="UTF-8"><title>MaverickPICKS Tracker</title>{CSS}</head><body>
<div class="hdr">
  <h1>MaverickPICKS — Active Pick Tracker</h1>
  <p>Updated: {now} &nbsp;|&nbsp; {len(watching)} watching &nbsp;|&nbsp;
     {len(breakouts)} breakouts &nbsp;|&nbsp; {len(breakdowns)} breakdowns</p>
</div>
<div class="stats">
  <div class="stat"><div class="n">{len(watching)}</div><div class="l">👁 Watching</div></div>
  <div class="stat"><div class="n" style="color:#166534">{len(breakouts)}</div><div class="l">🟢 Breakouts</div></div>
  <div class="stat"><div class="n" style="color:#991b1b">{len(breakdowns)}</div><div class="l">🔴 Breakdowns</div></div>
  <div class="stat"><div class="n" style="color:#1d4ed8">{win_rate}%</div><div class="l">Hit Rate</div></div>
  <div class="stat"><div class="n" style="color:{'#166534' if avg_pnl>=0 else '#991b1b'}">{avg_pnl:+.1f}%</div><div class="l">Avg P&L</div></div>
  <div class="stat"><div class="n">{total_res}</div><div class="l">Resolved</div></div>
</div>
<div class="section">
  <div class="sec-title">👁 Watching — sorted by proximity to entry</div>
  <div style="overflow-x:auto">
  <table>
    <thead><tr>
      <th>Symbol</th><th>Pattern</th><th>Track Signal</th><th>Signal Detail</th>
      <th>ETA</th><th>Last Close</th><th>Gap to Entry</th>
      <th>Entry</th><th>Stop</th><th>Target</th><th>R:R</th>
      <th>Days</th><th>Expiry</th><th>Trend</th>
    </tr></thead>
    <tbody>{'<tr><td colspan="14" style="text-align:center;padding:30px;color:#9ca3af">No active picks</td></tr>' if not watch_rows else watch_rows}</tbody>
  </table>
  </div>

  <div class="sec-title">✅ Resolved Picks</div>
  <div style="overflow-x:auto">
  <table>
    <thead><tr>
      <th>Symbol</th><th>Pattern</th><th>State</th>
      <th>Added</th><th>Resolved</th><th>Days</th>
      <th>Entry</th><th>Exit</th><th>P&L</th><th>Note</th>
    </tr></thead>
    <tbody>{'<tr><td colspan="10" style="text-align:center;padding:30px;color:#9ca3af">No resolved picks yet</td></tr>' if not res_rows else res_rows}</tbody>
  </table>
  </div>

  {perf_html}

</div>
<p class="note">
  Track Signal: ON TRACK = gap closing + volume contracting. CAUTION = moving away from entry or volume expanding.
  ETA = estimated trading days to reach entry at current velocity. Gap = how far last close is from entry level.
  Performance stats need 30+ resolved picks to be reliable.
</p>
</body></html>"""

    with open(DASH_TRACKER,"w",encoding="utf-8") as f:
        f.write(html)
    print(f"  Dashboard → {DASH_TRACKER}")


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description="MaverickPICKS Unified Tracker")
    ap.add_argument("--check_only",  action="store_true", help="Skip import, just check prices")
    ap.add_argument("--import_only", action="store_true", help="Just import, no price check")
    args = ap.parse_args()

    print(f"\n{'='*60}")
    print("  MaverickPICKS Tracker")
    print(f"  {datetime.now(IST).strftime('%d-%b-%Y  %H:%M IST')}")
    print(f"  Anchor: {ANCHOR.strftime('%d-%b-%Y')}")
    print(f"{'='*60}")

    wl      = load_wl()
    history = load_history()

    # Step 1: Import new picks
    if not args.check_only:
        print(f"\n  Importing from {SCAN_CSV}...")
        import_picks(wl)
        save_wl(wl)

    total_watching = sum(1 for p in wl["picks"].values() if p["state"]==WATCHING)
    print(f"\n  Watchlist: {len(wl['picks'])} total | {total_watching} watching")

    # Step 2: Check prices
    if not args.import_only and total_watching > 0:
        print()
        summary = check_all(wl, history)
        save_wl(wl)
        save_history(history)

        # Print alerts
        if summary["alerts"]:
            print(f"\n  {'='*50}")
            for p in summary["alerts"]:
                print(f"  *** {p['state']}: {p['symbol']} — {p.get('resolved_note','')[:80]}")
            print(f"  {'='*50}")

    # Step 3: Generate dashboards
    print("\n  Generating dashboards...")
    generate_picks_dashboard()
    generate_tracker_dashboard(wl, history)

    print(f"\n{'='*60}\n")


if __name__ == "__main__":
    main()

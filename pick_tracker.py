"""
MaverickPICKS Pick Tracker
==========================

Usage:
  python pick_tracker.py                          # Daily run — fetch, analyze, report
  python pick_tracker.py --import report.xlsx     # Import picks from Top 10 Excel
  python pick_tracker.py --status                 # Quick status check (no fetch)
  python pick_tracker.py --remove BEL HAL         # Remove symbols from tracking

Workflow:
  1. Run main_v3.py → generates MaverickPICKS_Top10_Report.xlsx
  2. python pick_tracker.py --import MaverickPICKS_Top10_Report.xlsx
  3. Every day after market close: python pick_tracker.py
  4. Opens tracker_report.html in your browser automatically
"""

import os
import sys
import json
import time
import webbrowser
import pandas as pd
from datetime import datetime, date, timedelta

from data_loader import load_stock, load_nifty
from trend_engine_v2 import trend_analysis
from momentum_engine_v2 import momentum_analysis
from relative_strength_v2 import relative_strength_analysis
from volume_engine import volume_analysis
from pattern_engine import pattern_analysis
from risk_engine import risk_analysis
from ranking_engine_v2 import ranking_engine
from reason_engine_v2 import generate_reason


# ============================================================
# CONFIG
# ============================================================

PICKS_FILE    = "active_picks.json"
REPORT_FILE   = "tracker_report.html"
MIN_ROWS      = 100

# Validity window: how many trading sessions to wait for entry before expiring
VALIDITY_DAYS = {
    "TREND_PULLBACK":  5,
    "SUPPORT_BOUNCE":  5,
    "RECOVERY":        7,
    "BREAKOUT":        2,
    "LEADER":          5,
}
DEFAULT_VALIDITY = 5


# ============================================================
# JSON STATE MANAGEMENT
# ============================================================

def load_picks():
    if not os.path.exists(PICKS_FILE):
        return {"picks": [], "last_updated": None}
    try:
        with open(PICKS_FILE, "r") as f:
            return json.load(f)
    except Exception as e:
        print(f"Error loading {PICKS_FILE}: {e}")
        return {"picks": [], "last_updated": None}


def save_picks(data):
    data["last_updated"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(PICKS_FILE, "w") as f:
        json.dump(data, f, indent=2, default=str)
    print(f"Tracking file saved: {PICKS_FILE}")


# ============================================================
# IMPORT PICKS FROM EXCEL
# ============================================================

def import_picks(excel_path):

    print(f"\nImporting picks from: {excel_path}")

    try:
        # Try "TOP 10 PICKS" sheet first, fall back to first sheet
        try:
            df = pd.read_excel(excel_path, sheet_name="TOP 10 PICKS")
        except Exception:
            df = pd.read_excel(excel_path, sheet_name=0)

        required = ["Symbol", "Entry", "Stop_Loss", "Target_1", "Target_2"]
        for col in required:
            if col not in df.columns:
                print(f"ERROR: Missing required column '{col}' in Excel.")
                return

        data = load_picks()
        # Dedup on (symbol, pick_date) — same Data_As_Of batch = same pick.
        # Monday's scan with Friday's data should NOT create duplicates of Friday's picks.
        # We use a dict so we can migrate scan_date on existing picks that don't have it yet.
        existing_by_key = {
            (p["symbol"], str(p.get("pick_date", ""))): p
            for p in data["picks"]
            if p.get("status") in ["ACTIVE", "PENDING"]
        }

        imported = 0
        skipped  = 0

        for _, row in df.iterrows():
            symbol = str(row["Symbol"]).strip()
            pick_date_str = str(row.get("Data_As_Of", date.today()))
            scan_date_str = str(date.today())   # when main_v3 was actually run

            pick_key = (symbol, pick_date_str)   # dedup on data batch, not run date
            if pick_key in existing_by_key:
                existing = existing_by_key[pick_key]
                if not existing.get("scan_date"):
                    # Auto-migrate: stamp scan_date on old picks that predate this field
                    existing["scan_date"] = scan_date_str
                    print(f"  {symbol} ({pick_date_str}): already tracked — migrated scan_date → {scan_date_str}")
                else:
                    print(f"  {symbol} ({pick_date_str}): already tracked, skipping")
                skipped += 1
                continue

            setup_type = str(row.get("Setup_Type", ""))
            validity   = VALIDITY_DAYS.get(setup_type, DEFAULT_VALIDITY)

            pick = {
                "symbol":         symbol,
                "scan_date":      scan_date_str,  # when main_v3 ran (today)
                "pick_date":      pick_date_str,  # last market data date (Data_As_Of)
                "entry":          float(row["Entry"]),
                "stop_loss":      float(row["Stop_Loss"]),
                "target_1":       float(row["Target_1"]),
                "target_2":       float(row["Target_2"]),
                "confidence_pct": float(row.get("Confidence_Pct", 0)),
                "setup_type":     setup_type,
                "setup_grade":    str(row.get("Setup_Grade", "")),
                "headline":       str(row.get("Headline", "")),

                "status":              "PENDING",
                "validity_days":       validity,
                "entry_triggered_date": None,
                "t1_hit_date":         None,
                "t2_hit_date":         None,
                "sl_hit_date":         None,
                "exit_date":           None,
                "exit_price":          None,
                "exit_reason":         None,

                "daily_log":      [],
            }

            data["picks"].append(pick)
            existing_by_key[pick_key] = pick
            imported += 1
            print(f"  {symbol}: imported as PENDING (Entry ₹{pick['entry']}, SL ₹{pick['stop_loss']}, T1 ₹{pick['target_1']}, valid {validity} days)")

        save_picks(data)
        print(f"\nImported: {imported}, Skipped (already tracked): {skipped}")
        print(f"Total pending: {sum(1 for p in data['picks'] if p['status']=='PENDING')}")
        print(f"Total active:  {sum(1 for p in data['picks'] if p['status']=='ACTIVE')}")

    except Exception as e:
        print(f"Import error: {e}")



# ============================================================
# DEDUP + MIGRATION
# Runs once at the start of every daily run.
# Fixes two problems:
#   1. Duplicate picks created when the dedup key was accidentally changed
#      in a prior code version (same stock imported twice with different scan_dates).
#   2. Legacy picks that have no scan_date field — backfills it from pick_date.
# Safe to run repeatedly — idempotent.
# ============================================================

def _dedup_and_migrate_picks(data):
    from collections import defaultdict

    trackable    = [p for p in data["picks"] if p.get("status") in ["ACTIVE", "PENDING"]]
    non_trackable = [p for p in data["picks"] if p.get("status") not in ["ACTIVE", "PENDING"]]

    # Group by (symbol, pick_date) — this is the natural identity of a pick
    groups = defaultdict(list)
    for p in trackable:
        key = (p["symbol"], str(p.get("pick_date", "")))
        groups[key].append(p)

    merged    = []
    changed   = False
    dup_count = 0

    for key, picks in groups.items():
        sym, pd = key

        if len(picks) == 1:
            p = picks[0]
            # Migrate: backfill scan_date from pick_date if missing
            if not p.get("scan_date"):
                p["scan_date"] = str(p.get("pick_date", date.today()))
                changed = True
            merged.append(p)
            continue

        # --- Duplicate group: merge into one pick ---
        changed   = True
        dup_count += len(picks) - 1

        # Prefer: ACTIVE > PENDING; then most daily_log entries; then has scan_date
        primary = max(
            picks,
            key=lambda p: (
                p["status"] == "ACTIVE",
                len(p.get("daily_log", [])),
                bool(p.get("scan_date")),
            )
        )

        # Merge all daily_logs (union keyed by date — latest write wins)
        all_logs = {}
        for p in picks:
            for log in p.get("daily_log", []):
                all_logs[log["date"]] = log
        primary["daily_log"] = sorted(all_logs.values(), key=lambda l: l["date"])

        # Use the most recent scan_date across all duplicates; fallback to pick_date
        scan_dates = [p["scan_date"] for p in picks if p.get("scan_date")]
        primary["scan_date"] = max(scan_dates) if scan_dates else str(primary.get("pick_date", date.today()))

        # Carry forward milestone dates from any of the duplicates
        for field in ["entry_triggered_date", "t1_hit_date", "t2_hit_date", "sl_hit_date"]:
            vals = [p.get(field) for p in picks if p.get(field)]
            if vals:
                primary[field] = min(vals)   # earliest milestone wins

        print(f"  [dedup] {sym} ({pd}): merged {len(picks)} entries → scan_date={primary['scan_date']}")
        merged.append(primary)

    if changed:
        data["picks"] = merged + non_trackable
        save_picks(data)
        if dup_count:
            print(f"Dedup complete — removed {dup_count} duplicate pick(s).")
        else:
            print("Migration complete — scan_date backfilled on legacy picks.")

    return data


# ============================================================
# DAILY RUN — FETCH + ANALYZE + UPDATE
# ============================================================

def run_daily():

    data = load_picks()

    # Step 0: clean up any duplicates + backfill scan_date on legacy picks
    data = _dedup_and_migrate_picks(data)

    # Always check Excel for new picks (handles Monday scan + existing tracking)
    report_names = [
        "MaverickPICKS_Top10_Report.xlsx",
        "MaverickPICKS_Report.xlsx",
    ]
    for name in report_names:
        if os.path.exists(name):
            print(f"\nChecking {name} for new picks...")
            import_picks(name)
            data = load_picks()
            break

    trackable = [p for p in data["picks"] if p["status"] in ["ACTIVE", "PENDING"]]

    if not trackable:
        print("\nNo picks to track. Run main_v3.py first to generate picks.")
        return

    n_pending = sum(1 for p in trackable if p["status"] == "PENDING")
    n_active  = sum(1 for p in trackable if p["status"] == "ACTIVE")
    print(f"\nTracking {len(trackable)} picks ({n_pending} pending entry, {n_active} active)...")

    # Load NIFTY benchmark
    print("Downloading NIFTY benchmark...")
    nifty_df = load_nifty()
    if nifty_df is None or nifty_df.empty:
        print("WARNING: Could not load NIFTY. RS analysis will be skipped.")
        nifty_df = None

    for pick in trackable:
        symbol = pick["symbol"]
        days_since_pick = _trading_day_number(pick.get("scan_date", pick["pick_date"]))

        print(f"\n{'='*50}")
        print(f"  {symbol}  |  Entry: ₹{pick['entry']}  |  Status: {pick['status']}  |  Day {days_since_pick}")
        print(f"{'='*50}")

        # Fetch OHLCV
        stock_data = load_stock(symbol)
        daily   = stock_data.get("daily",   pd.DataFrame())
        weekly  = stock_data.get("weekly",  pd.DataFrame())
        monthly = stock_data.get("monthly", pd.DataFrame())

        if daily is None or daily.empty or len(daily) < MIN_ROWS:
            print(f"  SKIP: insufficient data for {symbol}")
            continue

        close = round(float(daily["Close"].dropna().iloc[-1]), 2)
        high  = round(float(daily["High"].dropna().iloc[-1]), 2)
        low   = round(float(daily["Low"].dropna().iloc[-1]), 2)
        vol   = float(daily["Volume"].dropna().iloc[-1])
        last_date = daily["Close"].dropna().index[-1]
        last_date = last_date.strftime("%Y-%m-%d") if hasattr(last_date, "strftime") else str(last_date)

        # --- Run all engines ---
        trend    = trend_analysis(daily.copy())
        momentum = momentum_analysis(daily.copy())

        if nifty_df is not None and not nifty_df.empty:
            rs = relative_strength_analysis(daily.copy(), nifty_df.copy())
        else:
            rs = {"RS_State": "UNKNOWN", "RS_Score": 0}

        volume   = volume_analysis(daily.copy(), weekly.copy(), monthly.copy())
        pattern  = pattern_analysis(daily.copy(), weekly.copy())
        risk     = risk_analysis(daily.copy())
        ranking  = ranking_engine(trend, momentum, rs, volume, pattern, risk)
        reason   = generate_reason(trend, momentum, rs, volume, pattern, risk, ranking)


        # ===========================================================
        # PENDING PICKS — waiting for entry to trigger
        # ===========================================================

        if pick["status"] == "PENDING":

            entry = pick["entry"]
            sl    = pick["stop_loss"]
            validity = pick.get("validity_days", DEFAULT_VALIDITY)

            # Count trading sessions since pick (approximate via daily_log length + 1)
            sessions_waited = len(pick["daily_log"]) + 1

            # --- Check 1: Has price reached entry level? ---
            # Entry triggered when stock closes AT or ABOVE entry level.
            # This confirms the stock is moving in the right direction.
            # If stock is below entry, it hasn't confirmed yet — wait.
            entry_triggered = (close >= entry)

            # --- Check 2: Has setup been invalidated? ---
            # SL breached before entry = thesis broken, don't enter
            invalidated = (close <= sl)

            # --- Check 3: Has stock been below entry too long? ---
            dist_to_entry = round(((close - entry) / entry) * 100, 2)
            expired = (sessions_waited > validity and not entry_triggered)

            if invalidated:
                pick["status"]      = "INVALIDATED"
                pick["exit_date"]   = last_date
                pick["exit_reason"] = (
                    f"Setup invalidated — price fell to ₹{close} below SL (₹{sl}) "
                    f"before entry was confirmed. No position taken."
                )
                action = "INVALIDATED"
                action_detail = pick["exit_reason"]
                pnl_pct = 0

                print(f"  ✗ INVALIDATED: SL breached before entry confirmed")

            elif entry_triggered:
                pick["status"]              = "ACTIVE"
                pick["entry_triggered_date"] = last_date
                pnl_pct = round(((close - entry) / entry) * 100, 2)
                action = "ENTRY_TRIGGERED"
                action_detail = (
                    f"Entry confirmed — stock at ₹{close} (entry level ₹{entry}, "
                    f"P&L: {pnl_pct:+.1f}%). Position now active. "
                    f"Trend: {trend.get('Trend_State')}, "
                    f"Momentum: {momentum.get('Momentum_State')}, "
                    f"Volume: {volume.get('Volume_State')}. "
                    f"SL: ₹{sl}, T1: ₹{pick['target_1']}, T2: ₹{pick['target_2']}."
                )

                print(f"  ✓ ENTRY CONFIRMED at ₹{close} (entry was ₹{entry}, P&L: {pnl_pct:+.1f}%). Now ACTIVE.")

            elif expired:
                pick["status"]      = "EXPIRED"
                pick["exit_date"]   = last_date
                pick["exit_reason"] = (
                    f"Stock didn't reach entry level ₹{entry} within {validity} trading days. "
                    f"Current price: ₹{close} ({dist_to_entry:+.1f}% from entry). "
                    f"Setup may have changed — re-scan required."
                )
                action = "EXPIRED"
                action_detail = pick["exit_reason"]
                pnl_pct = 0

                print(f"  ⏰ EXPIRED: Didn't reach entry ₹{entry} in {validity} trading days")

            else:
                # Below entry, waiting for stock to come up
                action = "WAITING"
                pnl_pct = 0

                action_detail = _build_pending_commentary(
                    pick, close, dist_to_entry, sessions_waited,
                    trend, momentum, volume, pattern, ranking
                )

                print(f"  ⏳ PENDING: Close ₹{close} is {dist_to_entry:.1f}% below entry ₹{entry} (day {sessions_waited}/{validity})")

            # Log
            log_entry = {
                "date":           last_date,
                "close":          close,
                "high":           high,
                "low":            low,
                "volume":         vol,
                "pnl_pct":        pnl_pct,
                "action":         action,
                "action_detail":  action_detail,
                "trend_state":    trend.get("Trend_State"),
                "momentum_state": momentum.get("Momentum_State"),
                "volume_state":   volume.get("Volume_State"),
                "rsi":            momentum.get("RSI"),
                "pattern":        pattern.get("Primary_Pattern"),
                "at_support":     pattern.get("At_Support"),
                "verdict":        ranking.get("Verdict"),
            }

            pick["daily_log"] = [l for l in pick["daily_log"] if l["date"] != last_date]
            pick["daily_log"].append(log_entry)
            time.sleep(0.3)
            continue


        # ===========================================================
        # ACTIVE PICKS — position is live, tracking P&L and exits
        # ===========================================================

        # P&L is tracked from the original entry price (the plan level)
        pnl_pct = round(((close - pick["entry"]) / pick["entry"]) * 100, 2)

        # --- Check SL / T1 / T2 (closing price basis) ---
        sl_hit_today = (close <= pick["stop_loss"]) and not pick["sl_hit_date"]
        t1_hit_today = (close >= pick["target_1"]) and not pick["t1_hit_date"]
        t2_hit_today = (close >= pick["target_2"]) and not pick["t2_hit_date"]

        if sl_hit_today:
            pick["sl_hit_date"] = last_date
        if t1_hit_today:
            pick["t1_hit_date"] = last_date
        if t2_hit_today:
            pick["t2_hit_date"] = last_date

        # --- Build recommendation ---
        action, action_detail = _build_recommendation(
            pick, close, pnl_pct, trend, momentum, volume, pattern, ranking, reason, daily_df=daily
        )

        # --- Auto-complete if SL or T2 hit ---
        if pick["sl_hit_date"] and action in ["EXIT_LOSS", "EXIT_TRAIL"]:
            pick["status"]      = "COMPLETED_SL"
            pick["exit_date"]   = last_date
            pick["exit_price"]  = close
            pick["exit_reason"] = action_detail

        if pick["t2_hit_date"] and action == "BOOK_FULL":
            pick["status"]      = "COMPLETED_T2"
            pick["exit_date"]   = last_date
            pick["exit_price"]  = close
            pick["exit_reason"] = action_detail

        # --- Log today ---
        log_entry = {
            "date":           last_date,
            "close":          close,
            "high":           high,
            "low":            low,
            "volume":         vol,
            "pnl_pct":        pnl_pct,
            "action":         action,
            "action_detail":  action_detail,
            "trend_state":    trend.get("Trend_State"),
            "momentum_state": momentum.get("Momentum_State"),
            "volume_state":   volume.get("Volume_State"),
            "rsi":            momentum.get("RSI"),
            "pattern":        pattern.get("Primary_Pattern"),
            "at_support":     pattern.get("At_Support"),
            "verdict":        ranking.get("Verdict"),
        }

        pick["daily_log"] = [l for l in pick["daily_log"] if l["date"] != last_date]
        pick["daily_log"].append(log_entry)

        # Print summary
        pnl_sym = "+" if pnl_pct >= 0 else ""
        print(f"  Close: ₹{close}  |  P&L: {pnl_sym}{pnl_pct}%  |  Data: {last_date}")
        print(f"  Trend: {trend.get('Trend_State')}  |  Momentum: {momentum.get('Momentum_State')}  |  Volume: {volume.get('Volume_State')}")
        print(f"  RSI: {momentum.get('RSI')}  |  Pattern: {pattern.get('Primary_Pattern')}")
        print(f"  Verdict: {ranking.get('Verdict')}  |  Action: {action}")
        print(f"  → {action_detail}")

        if pick["t1_hit_date"]:
            print(f"  ✓ T1 hit on {pick['t1_hit_date']}")
        if pick["t2_hit_date"]:
            print(f"  ✓ T2 hit on {pick['t2_hit_date']}")
        if pick["sl_hit_date"]:
            print(f"  ✗ SL hit on {pick['sl_hit_date']}")

        time.sleep(0.3)

    save_picks(data)

    # Generate report
    print(f"\nGenerating report...")
    generate_report(data)


# ============================================================
# PENDING PICK COMMENTARY
# ============================================================

def _build_pending_commentary(pick, close, dist_to_entry, sessions_waited,
                                trend, momentum, volume, pattern, ranking):

    entry      = round(float(pick["entry"]), 2)
    sl         = round(float(pick["stop_loss"]), 2)
    validity   = pick.get("validity_days", DEFAULT_VALIDITY)
    remaining  = max(0, validity - sessions_waited)
    setup_type = pick.get("setup_type", "")

    trend_state = trend.get("Trend_State", "WEAK")
    mom_state   = momentum.get("Momentum_State", "WEAK")
    vol_state   = volume.get("Volume_State", "DRY")
    rsi         = momentum.get("RSI", 50) or 50
    at_support  = pattern.get("At_Support", False)
    verdict     = ranking.get("Verdict", "AVOID")

    parts = []

    # Status line — stock is below entry, waiting for it to rise
    abs_dist = abs(dist_to_entry)

    if abs_dist <= 1.0:
        parts.append(
            f"Stock at ₹{close}, just {abs_dist:.1f}% below entry ₹{entry}. "
            f"Very close — a small move up would confirm entry."
        )
    elif abs_dist <= 3.0:
        parts.append(
            f"Stock at ₹{close}, {abs_dist:.1f}% below entry ₹{entry}. "
            f"Needs to rise to confirm the setup."
        )
    else:
        parts.append(
            f"Stock at ₹{close}, {abs_dist:.1f}% below entry ₹{entry}. "
            f"Significant gap to entry level — setup weakening."
        )

    # Setup validity check
    if verdict in ["AVOID"]:
        parts.append(
            f"WARNING: Current re-scan verdict is {verdict}. Original {setup_type} "
            f"setup may be deteriorating — trend: {trend_state}, momentum: {mom_state}."
        )
    elif verdict in ["WATCHLIST", "NEUTRAL"]:
        parts.append(
            f"Setup still partially valid — verdict: {verdict}. Trend: {trend_state}, "
            f"RSI: {rsi:.0f}, Volume: {vol_state}."
        )
    else:
        parts.append(
            f"Setup still valid — verdict: {verdict}. Trend: {trend_state}, "
            f"momentum: {mom_state}, RSI: {rsi:.0f}."
        )

    if at_support:
        parts.append(f"Price near support — may trigger entry soon.")

    parts.append(f"{remaining} trading day(s) remaining before this pick expires.")

    return " ".join(parts)


# ============================================================
# RECOMMENDATION LOGIC (for ACTIVE picks)
# Thinks like a seasoned stock analyst:
# 1. Current trajectory — estimated days to T1
# 2. If price slips — what to do at key levels
# 3. At T1 — hold or book based on conditions
# ============================================================

def _build_recommendation(pick, close, pnl_pct, trend, momentum, volume, pattern, ranking, reason, daily_df=None):

    close   = round(float(close), 2)
    pnl_pct = round(float(pnl_pct), 2)

    entry  = round(float(pick["entry"]), 2)
    sl     = round(float(pick["stop_loss"]), 2)
    t1     = round(float(pick["target_1"]), 2)
    t2     = round(float(pick["target_2"]), 2)

    trend_state  = trend.get("Trend_State", "WEAK")
    mom_state    = momentum.get("Momentum_State", "WEAK")
    vol_state    = volume.get("Volume_State", "DRY")
    rsi          = float(momentum.get("RSI", 50) or 50)
    verdict      = ranking.get("Verdict", "AVOID")
    at_support   = pattern.get("At_Support", False)
    buyers       = pattern.get("Buyers_At_Support", False)
    ema20        = trend.get("EMA20")
    ema50        = trend.get("EMA50")
    support_lvl  = pattern.get("Support_Level")
    hist_pos     = momentum.get("Histogram_Positive", False)
    hist_rising  = momentum.get("Histogram_Rising", False)

    trend_intact  = trend_state in ["STRONG", "LEADER"]
    momentum_ok   = mom_state in ["IMPROVING", "STRONG", "LEADER"]
    volume_ok     = vol_state in ["ACCUMULATION", "BREAKOUT", "DRY_PULLBACK", "NORMAL"]
    thesis_alive  = trend_intact and (momentum_ok or volume_ok)
    thesis_weak   = not trend_intact and not momentum_ok

    # --- Estimate days to T1 ---
    days_to_t1 = _estimate_days_to_target(daily_df, close, t1) if daily_df is not None else None
    days_to_t2 = _estimate_days_to_target(daily_df, close, t2) if daily_df is not None else None

    # Key levels for slip-back scenario
    key_support = support_lvl or (round(float(ema20), 2) if ema20 else entry)


    # === SL HIT ===
    if close <= sl:
        return "EXIT_LOSS", (
            f"⛔ Stop loss breached — closed at ₹{close} below SL ₹{sl}. "
            f"Trend: {trend_state}, Momentum: {mom_state}. Exit to preserve capital. "
            f"Do not average down."
        )

    # === T2 HIT ===
    if close >= t2:
        return "BOOK_FULL", (
            f"🎯 Target 2 reached at ₹{close} (T2 was ₹{t2}). P&L: {pnl_pct:+.1f}%. "
            f"Full target achieved — book profits and move on."
        )

    # === T1 HIT — re-assess ===
    if close >= t1 or pick.get("t1_hit_date"):

        trail_sl = round(max(t1 * 0.97, float(ema20 or t1 * 0.97)), 2)

        if vol_state == "DISTRIBUTION":
            return "BOOK_FULL", (
                f"⚠️ T1 achieved but volume shows DISTRIBUTION (institutional selling). "
                f"RSI: {rsi:.0f}. Book full profits at ₹{close} before reversal."
            )

        if thesis_alive and rsi < 70:
            t2_est = f" (~{int(days_to_t2)} trading days at current pace)" if days_to_t2 else ""
            return "HOLD", (
                f"✅ T1 hit at ₹{close}. Thesis intact — Trend: {trend_state}, "
                f"Momentum: {mom_state}, Volume: {vol_state}, RSI: {rsi:.0f}. "
                f"HOLD for T2 (₹{t2}){t2_est}. Trail stop to ₹{trail_sl}. "
                f"📉 IF price slips below ₹{trail_sl}: book remaining and exit. "
                f"📈 IF volume surges with RSI still healthy: add on strength toward T2."
            )

        if rsi > 70:
            return "BOOK_PARTIAL", (
                f"⚠️ T1 hit but RSI overbought at {rsi:.0f} — momentum may fade. "
                f"Book 50% at ₹{close}. Trail remaining with SL at ₹{trail_sl}. "
                f"📉 IF RSI drops below 60 with falling volume: book remaining too. "
                f"📈 IF RSI cools to 55-65 and holds above ₹{trail_sl}: continue holding for T2."
            )

        return "BOOK_PARTIAL", (
            f"T1 reached at ₹{close}. Mixed signals — Trend: {trend_state}, "
            f"Momentum: {mom_state}, RSI: {rsi:.0f}. "
            f"Book 50% now, trail rest with SL at ₹{trail_sl}. "
            f"📉 IF drops below ₹{trail_sl}: exit remaining. "
            f"📈 IF trend strengthens with volume: hold for T2 (₹{t2})."
        )

    # === BETWEEN ENTRY AND T1 (in profit) ===
    if pnl_pct > 0:

        t1_est = f" Estimated ~{int(days_to_t1)} trading days to T1 at current pace." if days_to_t1 else ""
        dist_to_t1_pct = round(((t1 - close) / close) * 100, 1)

        if thesis_alive:
            return "HOLD", (
                f"📈 In profit at ₹{close} ({pnl_pct:+.1f}%), {dist_to_t1_pct:.1f}% away from T1 (₹{t1}).{t1_est} "
                f"Trend: {trend_state}, Momentum: {mom_state}, Volume: {vol_state}, RSI: {rsi:.0f}. "
                f"Thesis intact — continue holding. "
                f"📉 IF price slips back to ₹{entry}: hold as long as it stays above ₹{sl}. "
                f"IF price breaks ₹{key_support} with rising volume: tighten SL to ₹{key_support}. "
                f"🎯 AT T1 (₹{t1}): if RSI < 70 and volume healthy, hold for T2. If RSI > 70, book 50%."
            )

        if thesis_weak and pnl_pct > 3:
            return "BOOK_PARTIAL", (
                f"⚠️ In profit ({pnl_pct:+.1f}%) but thesis weakening — Trend: {trend_state}, "
                f"Momentum: {mom_state}, Volume: {vol_state}. "
                f"Book 50% at ₹{close}, trail rest with SL at ₹{entry}. "
                f"📉 IF drops back to ₹{entry}: exit remaining — don't let a winner become a loser. "
                f"📈 IF momentum turns STRONG with volume: resume holding for T1 (₹{t1})."
            )

        return "HOLD", (
            f"In modest profit at ₹{close} ({pnl_pct:+.1f}%), {dist_to_t1_pct:.1f}% from T1 (₹{t1}).{t1_est} "
            f"Trend: {trend_state}, Momentum: {mom_state}. "
            f"📉 IF price slips to ₹{entry}: still within plan, hold with SL at ₹{sl}. "
            f"🎯 AT T1 (₹{t1}): reassess based on RSI and volume at that point."
        )

    # === BETWEEN SL AND ENTRY (in drawdown but not stopped out) ===
    # This happens when close < entry but close > SL (stock triggered then pulled back)

    if vol_state == "DISTRIBUTION":
        return "EXIT_TRAIL", (
            f"⛔ Price at ₹{close} ({pnl_pct:+.1f}%) with DISTRIBUTION volume — "
            f"institutional selling detected. Exit before SL (₹{sl}) to limit damage. "
            f"Do not average down into distribution."
        )

    if at_support and buyers:
        return "ADD_MORE", (
            f"💪 Price pulled back to ₹{close} ({pnl_pct:+.1f}%) but buyers defending support. "
            f"Trend: {trend_state}, Volume: {vol_state}. "
            f"Consider adding at ₹{close} — strict SL remains at ₹{sl}. "
            f"📈 IF bounces from here with volume: original thesis reconfirmed. "
            f"📉 IF breaks below ₹{sl}: exit entire position immediately."
        )

    if at_support and trend_intact:
        return "HOLD", (
            f"Price at ₹{close} ({pnl_pct:+.1f}%), testing support. "
            f"Trend still {trend_state}, EMA structure intact. "
            f"Hold with SL at ₹{sl} — support holding is a positive sign. "
            f"📈 IF bounces with volume confirmation: setup reactivates toward T1 (₹{t1}). "
            f"📉 IF closes below ₹{sl}: exit, thesis is broken."
        )

    if thesis_weak and pnl_pct < -5:
        return "EXIT_TRAIL", (
            f"⚠️ Significant drawdown ({pnl_pct:+.1f}%) with weakening thesis — "
            f"Trend: {trend_state}, Momentum: {mom_state}, Volume: {vol_state}. "
            f"Consider exiting at ₹{close} rather than waiting for SL (₹{sl}). "
            f"The setup that got you in is no longer valid."
        )

    if momentum_ok or hist_rising:
        t1_est = f" T1 (₹{t1}) could take ~{int(days_to_t1)} trading days." if days_to_t1 else ""
        return "HOLD", (
            f"Price at ₹{close} ({pnl_pct:+.1f}%), below entry but momentum is {mom_state}"
            f"{' with histogram rising' if hist_rising else ''}. Recovery in progress.{t1_est} "
            f"📉 IF drops to ₹{sl}: exit, do not hold below SL. "
            f"📈 IF closes back above ₹{entry}: thesis reconfirmed, hold for T1."
        )

    return "HOLD", (
        f"Price at ₹{close} ({pnl_pct:+.1f}%). Trend: {trend_state}, "
        f"Momentum: {mom_state}, RSI: {rsi:.0f}. No clear exit signal yet. "
        f"📉 IF drops to ₹{sl}: exit immediately. "
        f"📈 IF closes above ₹{entry} with volume: thesis reactivates toward T1 (₹{t1})."
    )


def _estimate_days_to_target(daily_df, current_price, target_price):
    """
    Estimate trading days to reach target based on recent avg daily move.
    Returns None if not estimable.
    """
    try:
        if daily_df is None or len(daily_df) < 15:
            return None
        if target_price <= current_price:
            return 0

        # Average absolute daily close-to-close change over last 10 days
        recent_closes = daily_df["Close"].dropna().tail(11)
        if len(recent_closes) < 5:
            return None

        daily_moves = recent_closes.diff().dropna().abs()
        avg_daily_move = float(daily_moves.mean())

        if avg_daily_move <= 0:
            return None

        distance = target_price - current_price
        est_days = round(distance / avg_daily_move, 0)

        # Cap at reasonable range
        return min(max(est_days, 1), 60)

    except:
        return None


# ============================================================
# HELPERS
# ============================================================

def _trading_day_number(scan_date_str):
    """
    Return the trading-day number for a pick, starting at Day 1 on the scan date.
    Counts Mon–Fri only; skips weekends. Minimum = 1.

    Day 1 = scan date itself (or any weekend/holiday before the first trading session).
    e.g. scan on Fri 19 Jun → Mon 22 Jun = Day 1 (first actual market day after scan).
         scan on Tue 16 Jun → Mon 22 Jun = Day 4 (17,18,19,22 = 4 trading days elapsed,
         excluding the scan day itself, +0 offset).
    """
    try:
        start = datetime.strptime(str(scan_date_str)[:10], "%Y-%m-%d").date()
        today = date.today()
        if today <= start:
            return 1
        count = 0
        cur = start + timedelta(days=1)          # start counting the day AFTER the scan
        while cur <= today:
            if cur.weekday() < 5:                # Mon=0 … Fri=4
                count += 1
            cur += timedelta(days=1)
        return max(1, count)
    except:
        return 1


# keep old name as alias so any external callers don't break
def _days_since(pick_date_str):
    return _trading_day_number(pick_date_str)


# ============================================================
# HTML REPORT GENERATOR
# ============================================================

def generate_report(data):

    pending   = [p for p in data["picks"] if p["status"] == "PENDING"]
    active    = [p for p in data["picks"] if p["status"] == "ACTIVE"]
    completed = [p for p in data["picks"] if p["status"].startswith("COMPLETED")]
    expired   = [p for p in data["picks"] if p["status"] in ["EXPIRED", "INVALIDATED"]]
    all_picks = pending + active + completed + expired

    if not all_picks:
        print("No picks to report.")
        return

    # Summary stats
    total_active = len(active)
    total_pending = len(pending)
    in_profit  = sum(1 for p in active if p["daily_log"] and p["daily_log"][-1]["pnl_pct"] > 0)
    in_loss    = sum(1 for p in active if p["daily_log"] and p["daily_log"][-1]["pnl_pct"] < 0)
    t1_hits    = sum(1 for p in active if p["t1_hit_date"])
    sl_hits    = sum(1 for p in active if p["sl_hit_date"])
    avg_pnl    = 0
    if active:
        pnls = [p["daily_log"][-1]["pnl_pct"] for p in active if p["daily_log"]]
        avg_pnl = sum(pnls) / len(pnls) if pnls else 0

    now_str   = datetime.now().strftime("%d %b %Y, %H:%M")
    last_upd  = data.get("last_updated", "unknown")

    # Collect unique scan dates for the filter (scan_date = when main_v3 ran;
    # fall back to pick_date for picks imported before this field was added)
    pick_dates = sorted(set(
        p.get("scan_date", p["pick_date"]) for p in (pending + active)
        if p.get("scan_date") or p.get("pick_date")
    ), reverse=True)

    # Build JSON data for active picks (for JS sidebar calculation)
    INVEST_PER_STOCK = 10000
    active_js_data = []
    for p in active:
        if p["daily_log"]:
            latest = p["daily_log"][-1]
            active_js_data.append({
                "symbol":    p["symbol"],
                "entry":     p["entry"],
                "close":     latest["close"],
                "pnl_pct":   latest["pnl_pct"],
                "pick_date": p.get("scan_date", p["pick_date"]),  # scan_date drives filter
            })

    import json as _json
    active_js_json = _json.dumps(active_js_data)

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>MaverickPICKS Tracker</title>
<style>
* {{ box-sizing: border-box; margin: 0; padding: 0; }}
body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; background: #0f1117; color: #e0e0e0; padding: 20px; }}
.container {{ max-width: 1440px; margin: 0 auto; }}
h1 {{ font-size: 22px; font-weight: 600; margin-bottom: 4px; color: #fff; }}
.subtitle {{ font-size: 13px; color: #888; margin-bottom: 16px; }}
.page-layout {{ display: flex; gap: 20px; align-items: flex-start; }}
.sidebar {{ width: 280px; flex-shrink: 0; position: sticky; top: 20px; }}
@media (max-width: 1000px) {{ .page-layout {{ flex-direction: column; }} .sidebar {{ width: 100%; position: static; }} }}
.main-content {{ flex: 1; min-width: 0; }}
.sb-card {{ background: #1a1d27; border-radius: 10px; padding: 16px; margin-bottom: 14px; }}
.sb-title {{ font-size: 12px; color: #888; text-transform: uppercase; letter-spacing: 0.5px; margin-bottom: 12px; }}
.sb-big {{ font-size: 28px; font-weight: 600; text-align: center; margin: 8px 0; }}
.sb-row {{ display: flex; justify-content: space-between; padding: 6px 0; border-bottom: 1px solid #1f2130; font-size: 13px; }}
.sb-row:last-child {{ border-bottom: none; }}
.sb-row .lbl {{ color: #888; }}
.sb-row .val {{ font-family: 'SF Mono', 'Fira Code', monospace; font-weight: 500; }}
.lb-row {{ display: flex; justify-content: space-between; align-items: center; padding: 7px 10px; margin-bottom: 4px; background: #12141c; border-radius: 6px; font-size: 12px; }}
.lb-rank {{ color: #666; width: 20px; }}
.lb-sym {{ font-weight: 500; flex: 1; color: #fff; margin-left: 6px; }}
.lb-pnl {{ font-family: 'SF Mono', 'Fira Code', monospace; font-weight: 600; }}
.summary {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(120px, 1fr)); gap: 10px; margin-bottom: 16px; }}
.sum-card {{ background: #1a1d27; border-radius: 8px; padding: 12px; text-align: center; }}
.sum-num {{ font-size: 22px; font-weight: 600; }}
.sum-label {{ font-size: 10px; color: #888; margin-top: 3px; }}
.green {{ color: #34d399; }}
.red {{ color: #f87171; }}
.amber {{ color: #fbbf24; }}
.blue {{ color: #60a5fa; }}
.muted {{ color: #888; }}
.filter-bar {{ display: flex; gap: 8px; flex-wrap: wrap; margin-bottom: 16px; align-items: center; }}
.filter-label {{ font-size: 12px; color: #888; margin-right: 4px; }}
.filter-btn {{ font-size: 12px; padding: 5px 12px; border-radius: 6px; border: 1px solid #2a2d3a; background: #1a1d27; color: #aaa; cursor: pointer; }}
.filter-btn:hover {{ border-color: #60a5fa; color: #fff; }}
.filter-btn.active {{ background: #1e3a5f; border-color: #60a5fa; color: #60a5fa; }}
.picks-grid {{ display: grid; grid-template-columns: 1fr 1fr; gap: 14px; }}
@media (max-width: 1000px) {{ .picks-grid {{ grid-template-columns: 1fr; }} }}
.pick-card {{ background: #1a1d27; border-radius: 10px; padding: 16px; border-left: 4px solid #333; }}
.pick-card.action-hold {{ border-left-color: #60a5fa; }}
.pick-card.action-add {{ border-left-color: #34d399; }}
.pick-card.action-book {{ border-left-color: #fbbf24; }}
.pick-card.action-exit {{ border-left-color: #f87171; }}
.pick-card.action-waiting {{ border-left-color: #a78bfa; }}
.pick-card.action-expired {{ border-left-color: #666; opacity: 0.6; }}
.pick-card.hidden {{ display: none; }}
.pick-header {{ display: flex; justify-content: space-between; align-items: center; margin-bottom: 10px; }}
.pick-sym {{ font-size: 16px; font-weight: 600; color: #fff; }}
.pick-pnl {{ font-size: 16px; font-weight: 600; font-family: 'SF Mono', 'Fira Code', monospace; }}
.tag {{ font-size: 10px; padding: 2px 6px; border-radius: 5px; display: inline-block; margin-left: 4px; }}
.tag-green {{ background: #064e3b; color: #34d399; }}
.tag-red {{ background: #450a0a; color: #f87171; }}
.tag-amber {{ background: #451a03; color: #fbbf24; }}
.tag-blue {{ background: #1e3a5f; color: #60a5fa; }}
.metrics {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(75px, 1fr)); gap: 6px; margin-bottom: 10px; }}
.metric {{ background: #12141c; border-radius: 5px; padding: 6px 8px; }}
.m-label {{ font-size: 9px; color: #666; text-transform: uppercase; letter-spacing: 0.5px; }}
.m-value {{ font-size: 13px; font-weight: 500; font-family: 'SF Mono', 'Fira Code', monospace; margin-top: 1px; }}
.engine-bar {{ display: flex; gap: 6px; flex-wrap: wrap; margin-bottom: 8px; }}
.engine-tag {{ font-size: 10px; padding: 2px 6px; border-radius: 5px; background: #12141c; color: #aaa; }}
.recommendation {{ background: #12141c; border-radius: 6px; padding: 10px 12px; font-size: 12px; line-height: 1.5; }}
.rec-title {{ font-weight: 600; margin-bottom: 3px; font-size: 13px; }}
.history {{ margin-top: 10px; }}
.history-title {{ font-size: 11px; color: #666; margin-bottom: 4px; cursor: pointer; }}
.history-row {{ font-size: 10px; color: #777; padding: 2px 0; font-family: 'SF Mono', 'Fira Code', monospace; border-bottom: 1px solid #1f2130; }}
.completed-section {{ margin-top: 30px; padding-top: 20px; border-top: 1px solid #2a2d3a; }}
.completed-section h2 {{ font-size: 16px; color: #888; margin-bottom: 12px; }}
.completed-grid {{ display: grid; grid-template-columns: 1fr 1fr; gap: 14px; }}
@media (max-width: 1000px) {{ .completed-grid {{ grid-template-columns: 1fr; }} }}
</style>
</head>
<body>
<div class="container">
<h1>MaverickPICKS tracker</h1>
<p class="subtitle">Report generated: {now_str} &nbsp;|&nbsp; Last data update: {last_upd}</p>

<div class="page-layout">

<!-- SIDEBAR -->
<div class="sidebar">
  <div class="sb-card">
    <div class="sb-title">Portfolio (₹{INVEST_PER_STOCK:,} per stock)</div>
    <div class="sb-big" id="sb-pnl-pct">—</div>
    <div style="text-align:center;font-size:11px;color:#888;margin-bottom:10px">avg P&L</div>
    <div class="sb-row"><span class="lbl">Active picks</span><span class="val" id="sb-count">0</span></div>
    <div class="sb-row"><span class="lbl">Invested</span><span class="val" id="sb-invested">₹0</span></div>
    <div class="sb-row"><span class="lbl">Current value</span><span class="val" id="sb-current">₹0</span></div>
    <div class="sb-row"><span class="lbl">P&L (₹)</span><span class="val" id="sb-pnl-abs">₹0</span></div>
    <div class="sb-row"><span class="lbl">Best</span><span class="val green" id="sb-best">—</span></div>
    <div class="sb-row"><span class="lbl">Worst</span><span class="val red" id="sb-worst">—</span></div>
  </div>
  <div class="sb-card">
    <div class="sb-title">Leaderboard</div>
    <div id="sb-leaderboard"></div>
  </div>
</div>

<!-- MAIN CONTENT -->
<div class="main-content">

<div class="summary">
  <div class="sum-card"><div class="sum-num amber" id="stat-pending">{total_pending}</div><div class="sum-label">Pending entry</div></div>
  <div class="sum-card"><div class="sum-num blue" id="stat-active">{total_active}</div><div class="sum-label">Active</div></div>
  <div class="sum-card"><div class="sum-num green" id="stat-profit">{in_profit}</div><div class="sum-label">In profit</div></div>
  <div class="sum-card"><div class="sum-num red" id="stat-loss">{in_loss}</div><div class="sum-label">In loss</div></div>
  <div class="sum-card"><div class="sum-num {'green' if avg_pnl >= 0 else 'red'}" id="stat-avgpnl">{avg_pnl:+.1f}%</div><div class="sum-label">Avg P&L</div></div>
  <div class="sum-card"><div class="sum-num green" id="stat-t1">{t1_hits}</div><div class="sum-label">T1 hit</div></div>
</div>

<div class="filter-bar">
  <span class="filter-label">Suggested on:</span>
  <button class="filter-btn active" onclick="filterPicks('all')">All</button>
"""

    for pd_str in pick_dates:
        try:
            label = datetime.strptime(str(pd_str)[:10], "%Y-%m-%d").strftime("%a %d %b")
        except:
            label = str(pd_str)
        html += f'  <button class="filter-btn" onclick="filterPicks(\'{pd_str}\')">{label}</button>\n'

    html += """</div>
<div class="picks-grid">
"""

    # PENDING picks — render first with distinct style
    for p in pending:
        if not p["daily_log"]:
            continue

        latest = p["daily_log"][-1]
        close  = latest["close"]
        action = latest["action"]
        detail = latest["action_detail"]
        days   = _trading_day_number(p.get("scan_date", p["pick_date"]))
        validity = p.get("validity_days", DEFAULT_VALIDITY)
        sessions = len(p["daily_log"])
        remaining = max(0, validity - sessions)
        dist = round(((close - p["entry"]) / p["entry"]) * 100, 1)

        setup_tag = f'<span class="tag tag-blue">{p["setup_type"]}</span>' if p.get("setup_type") else ""

        engine_html = ""
        for key, label in [("trend_state", "Trend"), ("momentum_state", "Mom"), ("volume_state", "Vol"), ("rsi", "RSI")]:
            val = latest.get(key, "-")
            if val and val != "None":
                display = f"{label}: {val}" if key != "rsi" else f"RSI: {round(float(val), 0):.0f}"
                engine_html += f'<span class="engine-tag">{display}</span>'
        verdict_tag = latest.get("verdict", "")
        if verdict_tag:
            engine_html += f'<span class="engine-tag" style="color:#a78bfa">{verdict_tag}</span>'

        try:
            pd_label = datetime.strptime(str(p.get("scan_date", p["pick_date"]))[:10], "%Y-%m-%d").strftime("%d %b")
        except:
            pd_label = str(p.get("scan_date", p["pick_date"]))[:10]

        html += f"""
<div class="pick-card action-waiting" data-pick-date="{p.get('scan_date', p.get('pick_date', ''))}" data-status="PENDING" data-pnl="0" data-t1-hit="false">
  <div class="pick-header">
    <div>
      <span class="pick-sym">{p["symbol"]}</span>
      <span class="tag" style="background:#1a1d27;color:#666;border:1px solid #333">{pd_label}</span>
      {setup_tag}
      <span class="tag" style="background:#2d2150;color:#a78bfa">PENDING</span>
      <span class="tag tag-amber">{remaining} trading days left</span>
    </div>
    <span class="pick-pnl amber">{dist:+.1f}%</span>
  </div>
  <div class="metrics">
    <div class="metric"><div class="m-label">Entry level</div><div class="m-value">₹{p["entry"]:.2f}</div></div>
    <div class="metric"><div class="m-label">Current</div><div class="m-value">₹{close:.2f}</div></div>
    <div class="metric"><div class="m-label">SL</div><div class="m-value">₹{p["stop_loss"]:.2f}</div></div>
    <div class="metric"><div class="m-label">T1</div><div class="m-value">₹{p["target_1"]:.2f}</div></div>
    <div class="metric"><div class="m-label">T2</div><div class="m-value">₹{p["target_2"]:.2f}</div></div>
    <div class="metric"><div class="m-label">Day</div><div class="m-value">{days}</div></div>
  </div>
  <div class="engine-bar">{engine_html}</div>
  <div class="recommendation" style="border-left: 3px solid #a78bfa; border-radius: 0;">
    <div class="rec-title" style="color:#a78bfa">WAITING FOR ENTRY</div>
    {detail}
  </div>
</div>
"""

    # ACTIVE picks
    for p in active:
        if not p["daily_log"]:
            continue

        latest   = p["daily_log"][-1]
        pnl      = latest["pnl_pct"]
        close    = latest["close"]
        action   = latest["action"]
        detail   = latest["action_detail"]
        days     = _trading_day_number(p.get("scan_date", p["pick_date"]))

        pnl_cls = "green" if pnl > 0 else "red" if pnl < 0 else "muted"

        action_cls = "hold"
        action_color = "blue"
        if action in ["ADD_MORE"]:
            action_cls = "add"; action_color = "green"
        elif action in ["BOOK_PARTIAL", "BOOK_FULL"]:
            action_cls = "book"; action_color = "amber"
        elif action in ["EXIT_LOSS", "EXIT_TRAIL"]:
            action_cls = "exit"; action_color = "red"

        tags_html = ""
        if p["t1_hit_date"]:
            tags_html += f'<span class="tag tag-green">T1 hit {p["t1_hit_date"]}</span>'
        if p["t2_hit_date"]:
            tags_html += f'<span class="tag tag-green">T2 hit {p["t2_hit_date"]}</span>'
        if p["sl_hit_date"]:
            tags_html += f'<span class="tag tag-red">SL hit {p["sl_hit_date"]}</span>'

        setup_tag = f'<span class="tag tag-blue">{p["setup_type"]}</span>' if p.get("setup_type") else ""

        # Engine states from latest log
        engine_html = ""
        for key, label in [("trend_state", "Trend"), ("momentum_state", "Mom"), ("volume_state", "Vol"), ("rsi", "RSI"), ("pattern", "Pattern")]:
            val = latest.get(key, "-")
            if val and val != "None":
                display = f"{label}: {val}" if key != "rsi" else f"RSI: {round(float(val), 0):.0f}"
                engine_html += f'<span class="engine-tag">{display}</span>'

        verdict_tag = latest.get("verdict", "")
        if verdict_tag:
            vcolor = "green" if verdict_tag in ["STRONG BUY", "BUY"] else "amber" if verdict_tag == "WATCHLIST" else "red" if verdict_tag == "AVOID" else "blue"
            engine_html += f'<span class="engine-tag" style="color:{"#34d399" if vcolor=="green" else "#fbbf24" if vcolor=="amber" else "#f87171" if vcolor=="red" else "#60a5fa"}">{verdict_tag}</span>'

        # P&L history (last 10 entries)
        history_html = ""
        for log in p["daily_log"][-10:]:
            lp = log["pnl_pct"]
            lc = "#34d399" if lp > 0 else "#f87171" if lp < 0 else "#888"
            history_html += f'<div class="history-row">{log["date"]}  ₹{log["close"]:>10.2f}  <span style="color:{lc}">{lp:>+6.1f}%</span>  {log["action"]}</div>'

        try:
            pd_label_a = datetime.strptime(str(p.get("scan_date", p["pick_date"]))[:10], "%Y-%m-%d").strftime("%d %b")
        except:
            pd_label_a = str(p.get("scan_date", p["pick_date"]))[:10]

        html += f"""
<div class="pick-card action-{action_cls}" data-pick-date="{p.get('scan_date', p.get('pick_date', ''))}" data-status="ACTIVE" data-pnl="{pnl}" data-t1-hit="{'true' if p.get('t1_hit_date') else 'false'}">
  <div class="pick-header">
    <div>
      <span class="pick-sym">{p["symbol"]}</span>
      <span class="tag" style="background:#1a1d27;color:#666;border:1px solid #333">{pd_label_a}</span>
      {setup_tag}{tags_html}
    </div>
    <span class="pick-pnl {pnl_cls}">{pnl:+.1f}%</span>
  </div>
  <div class="metrics">
    <div class="metric"><div class="m-label">Entry</div><div class="m-value">₹{p["entry"]:.2f}</div></div>
    <div class="metric"><div class="m-label">Current</div><div class="m-value">₹{close:.2f}</div></div>
    <div class="metric"><div class="m-label">SL</div><div class="m-value">₹{p["stop_loss"]:.2f}</div></div>
    <div class="metric"><div class="m-label">T1</div><div class="m-value">₹{p["target_1"]:.2f}</div></div>
    <div class="metric"><div class="m-label">T2</div><div class="m-value">₹{p["target_2"]:.2f}</div></div>
    <div class="metric"><div class="m-label">Day</div><div class="m-value">{days}</div></div>
    <div class="metric"><div class="m-label">Confidence</div><div class="m-value">{p['confidence_pct']:.0f}%</div></div>
  </div>
  <div class="engine-bar">{engine_html}</div>
  <div class="recommendation">
    <div class="rec-title" style="color:{"#34d399" if action_color=="green" else "#fbbf24" if action_color=="amber" else "#f87171" if action_color=="red" else "#60a5fa"}">{action.replace("_", " ")}</div>
    {detail}
  </div>
  <div class="history">
    <div class="history-title">Daily log (last {min(len(p["daily_log"]), 10)} entries)</div>
    {history_html}
  </div>
</div>
"""

    html += '</div>'  # close picks-grid

    # Completed picks
    if completed:
        html += '<div class="completed-section"><h2>Completed picks</h2><div class="completed-grid">'
        for p in completed:
            result_cls = "green" if "T2" in p["status"] else "red" if "SL" in p["status"] else "muted"
            exit_pnl = round(((p.get("exit_price", p["entry"]) - p["entry"]) / p["entry"]) * 100, 1) if p.get("exit_price") else 0
            html += f"""
<div class="pick-card" style="border-left-color: {'#34d399' if 'T2' in p['status'] else '#f87171'}; opacity: 0.7;">
  <div class="pick-header">
    <span class="pick-sym">{p["symbol"]} <span class="tag {'tag-green' if 'T2' in p['status'] else 'tag-red'}">{p["status"]}</span></span>
    <span class="pick-pnl {result_cls}">{exit_pnl:+.1f}%</span>
  </div>
  <div style="font-size:12px;color:#777;">
    Picked: {p["pick_date"]} | Entry: ₹{p["entry"]:.2f} | Exit: ₹{round(p["exit_price"], 2) if p.get("exit_price") else "-"} on {p.get("exit_date", "-")}
    <br>{p.get("exit_reason", "")}
  </div>
</div>"""
        html += '</div></div>'

    # Expired / Invalidated picks
    if expired:
        html += '<div class="completed-section"><h2>Expired / Invalidated</h2><div class="completed-grid">'
        for p in expired:
            html += f"""
<div class="pick-card action-expired">
  <div class="pick-header">
    <span class="pick-sym">{p["symbol"]} <span class="tag" style="background:#333;color:#888">{p["status"]}</span></span>
  </div>
  <div style="font-size:12px;color:#777;">
    Picked: {p["pick_date"]} | Entry level: ₹{p["entry"]:.2f} (never triggered)
    <br>{p.get("exit_reason", "")}
  </div>
</div>"""
        html += '</div></div>'

    html += '</div>'   # close main-content
    html += '</div>'   # close page-layout

    html += f"""
<script>
var ACTIVE_PICKS = {active_js_json};
var INVEST = {INVEST_PER_STOCK};

function updateSidebar(dateFilter) {{
  var picks = ACTIVE_PICKS;
  if (dateFilter && dateFilter !== 'all') {{
    picks = picks.filter(function(p) {{ return p.pick_date === dateFilter; }});
  }}

  var count = picks.length;
  var invested = count * INVEST;
  var currentVal = 0;
  var best = null;
  var worst = null;

  picks.forEach(function(p) {{
    var units = INVEST / p.entry;
    currentVal += units * p.close;
    if (!best || p.pnl_pct > best.pnl_pct) best = p;
    if (!worst || p.pnl_pct < worst.pnl_pct) worst = p;
  }});

  var pnlAbs = currentVal - invested;
  var pnlPct = invested > 0 ? (pnlAbs / invested * 100) : 0;

  var pnlColor = pnlPct >= 0 ? '#34d399' : '#f87171';

  document.getElementById('sb-count').textContent = count;
  document.getElementById('sb-invested').textContent = '₹' + invested.toLocaleString('en-IN');
  document.getElementById('sb-current').textContent = '₹' + Math.round(currentVal).toLocaleString('en-IN');

  var pnlEl = document.getElementById('sb-pnl-abs');
  pnlEl.textContent = (pnlAbs >= 0 ? '+₹' : '-₹') + Math.abs(Math.round(pnlAbs)).toLocaleString('en-IN');
  pnlEl.style.color = pnlColor;

  var pctEl = document.getElementById('sb-pnl-pct');
  pctEl.textContent = (pnlPct >= 0 ? '+' : '') + pnlPct.toFixed(1) + '%';
  pctEl.style.color = pnlColor;

  document.getElementById('sb-best').textContent = best ? best.symbol + ' ' + best.pnl_pct.toFixed(1) + '%' : '—';
  document.getElementById('sb-worst').textContent = worst ? worst.symbol + ' ' + worst.pnl_pct.toFixed(1) + '%' : '—';

  // Leaderboard — show date tag only for duplicate symbols
  var sorted = picks.slice().sort(function(a,b) {{ return b.pnl_pct - a.pnl_pct; }});
  // Count symbol occurrences to decide whether to show date
  var symCounts = {{}};
  sorted.forEach(function(p) {{ symCounts[p.symbol] = (symCounts[p.symbol] || 0) + 1; }});
  var lbHtml = '';
  sorted.forEach(function(p, i) {{
    var c = p.pnl_pct >= 0 ? '#34d399' : '#f87171';
    var dateTag = symCounts[p.symbol] > 1 ? ' <span style="color:#666;font-size:10px">(' + p.pick_date.slice(5) + ')</span>' : '';
    lbHtml += '<div class="lb-row">' +
      '<span class="lb-rank">' + (i+1) + '</span>' +
      '<span class="lb-sym">' + p.symbol + dateTag + '</span>' +
      '<span class="lb-pnl" style="color:' + c + '">' + (p.pnl_pct >= 0 ? '+' : '') + p.pnl_pct.toFixed(1) + '%</span>' +
      '</div>';
  }});
  document.getElementById('sb-leaderboard').innerHTML = lbHtml || '<div style="color:#666;font-size:12px">No active picks for this filter</div>';
}}

function updateSummaryStats() {{
  var cards = document.querySelectorAll('.pick-card[data-status]');
  var pending = 0, active = 0, inProfit = 0, inLoss = 0, t1Hits = 0;
  var pnls = [];

  cards.forEach(function(card) {{
    if (card.classList.contains('hidden')) return;
    var status = card.getAttribute('data-status');
    var pnl    = parseFloat(card.getAttribute('data-pnl') || '0');
    var t1     = card.getAttribute('data-t1-hit') === 'true';

    if (status === 'PENDING') {{
      pending++;
    }} else if (status === 'ACTIVE') {{
      active++;
      if (pnl > 0) inProfit++;
      if (pnl < 0) inLoss++;
      if (t1) t1Hits++;
      pnls.push(pnl);
    }}
  }});

  var avgPnl = pnls.length > 0 ? pnls.reduce(function(a, b) {{ return a + b; }}, 0) / pnls.length : 0;

  document.getElementById('stat-pending').textContent = pending;
  document.getElementById('stat-active').textContent  = active;
  document.getElementById('stat-profit').textContent  = inProfit;
  document.getElementById('stat-loss').textContent    = inLoss;
  document.getElementById('stat-t1').textContent      = t1Hits;
  var avgEl = document.getElementById('stat-avgpnl');
  avgEl.textContent  = (avgPnl >= 0 ? '+' : '') + avgPnl.toFixed(1) + '%';
  avgEl.style.color  = avgPnl >= 0 ? '#34d399' : '#f87171';
}}

function filterPicks(dateVal) {{
  var cards = document.querySelectorAll('.pick-card[data-pick-date]');
  var btns = document.querySelectorAll('.filter-btn');
  btns.forEach(function(b) {{ b.classList.remove('active'); }});
  event.target.classList.add('active');
  cards.forEach(function(card) {{
    if (dateVal === 'all' || card.getAttribute('data-pick-date') === dateVal) {{
      card.classList.remove('hidden');
    }} else {{
      card.classList.add('hidden');
    }}
  }});
  updateSidebar(dateVal);
  updateSummaryStats();
}}

// Initialize on page load
updateSidebar('all');
updateSummaryStats();
</script>
</div>
</body>
</html>"""

    with open(REPORT_FILE, "w", encoding="utf-8") as f:
        f.write(html)

    print(f"Report saved: {REPORT_FILE}")
    try:
        webbrowser.open(f"file://{os.path.abspath(REPORT_FILE)}")
    except:
        pass


# ============================================================
# STATUS (quick check without fetching)
# ============================================================

def show_status():
    data = load_picks()
    trackable = [p for p in data["picks"] if p["status"] in ["ACTIVE", "PENDING"]]

    if not trackable:
        print("No active or pending picks.")
        return

    print(f"\n{'Symbol':<12} {'Status':<10} {'Entry':>8} {'Latest':>8} {'P&L':>8} {'Days':>5} {'Action':<15}")
    print("-" * 85)

    for p in trackable:
        if p["daily_log"]:
            latest = p["daily_log"][-1]
            pnl = latest['pnl_pct'] if p['status'] == 'ACTIVE' else 0
            pnl_str = f"{pnl:>+7.1f}%" if p['status'] == 'ACTIVE' else "    n/a"
            print(f"{p['symbol']:<12} {p['status']:<10} ₹{p['entry']:>7.1f} ₹{latest['close']:>7.1f} {pnl_str} {_trading_day_number(p.get('scan_date', p['pick_date'])):>4}d  {latest['action']:<15}")
        else:
            print(f"{p['symbol']:<12} {p['status']:<10} ₹{p['entry']:>7.1f} {'?':>8} {'?':>8} {_trading_day_number(p.get('scan_date', p['pick_date'])):>4}d  {'NOT YET RUN':<15}")


# ============================================================
# REMOVE PICKS
# ============================================================

def remove_picks(symbols_to_remove):
    data = load_picks()
    symbols_upper = {s.upper() for s in symbols_to_remove}
    before = len(data["picks"])
    data["picks"] = [p for p in data["picks"] if p["symbol"].upper() not in symbols_upper]
    after = len(data["picks"])
    save_picks(data)
    print(f"Removed {before - after} pick(s). {after} remaining.")


# ============================================================
# MAIN CLI
# ============================================================

if __name__ == "__main__":

    args = sys.argv[1:]

    print("=" * 55)
    print("  MaverickPICKS Pick Tracker")
    print(f"  {datetime.now().strftime('%d %b %Y  %H:%M:%S')}")
    print("=" * 55)

    if not args:
        run_daily()

    elif args[0] == "--import" and len(args) >= 2:
        import_picks(args[1])

    elif args[0] == "--status":
        show_status()

    elif args[0] == "--remove" and len(args) >= 2:
        remove_picks(args[1:])

    elif args[0] == "--help":
        print(__doc__)

    else:
        print("Unknown command. Use --help for usage.")

"""
performance_logger.py — MaverickPICKS Self-Improvement Engine
==============================================================
Reads resolved picks from active_picks.json (main_v3 tracker)
and unified_watchlist.json (pattern + H&S tracker), accumulates
outcomes into performance_history.json, and generates a
performance analysis dashboard.

The goal: understand WHAT works, WHY it works, and WHERE it fails
so future picks are better quality.

Run daily after tracker_check:
  python performance_logger.py

Or manually:
  python performance_logger.py --analyse   (generate analysis only)
  python performance_logger.py --reset     (clear history — careful!)
"""

import argparse
import json
import os
from collections import defaultdict
from datetime import datetime, timezone, timedelta

import numpy as np
import pandas as pd

IST              = timezone(timedelta(hours=5, minutes=30))
HISTORY_FILE     = "performance_history.json"
ACTIVE_PICKS     = "active_picks.json"
UNIFIED_WL       = "unified_watchlist.json"
PERF_DASHBOARD   = "dashboard_performance.html"

# Minimum resolved picks before we trust any stat
MIN_SAMPLE       = 10
MIN_RELIABLE     = 30


# ─────────────────────────────────────────────────────────────────────────────
# DATA STRUCTURES
# ─────────────────────────────────────────────────────────────────────────────

def load_history() -> dict:
    if os.path.exists(HISTORY_FILE):
        with open(HISTORY_FILE) as f:
            return json.load(f)
    return {
        "outcomes":      [],    # list of resolved pick outcome dicts
        "last_updated":  None,
        "total_logged":  0,
    }


def save_history(h: dict):
    h["last_updated"] = datetime.now(IST).strftime("%Y-%m-%d %H:%M IST")
    with open(HISTORY_FILE, "w") as f:
        json.dump(h, f, indent=2, default=str)


def _outcome_key(symbol: str, pick_date: str, source: str) -> str:
    return f"{symbol}|{pick_date}|{source}"


# ─────────────────────────────────────────────────────────────────────────────
# INGEST FROM active_picks.json (main_v3 tracker)
# ─────────────────────────────────────────────────────────────────────────────

RESOLVED_STATUSES = {"COMPLETED_T1", "COMPLETED_T2", "COMPLETED_SL",
                     "INVALIDATED", "EXITED", "EXPIRED"}


def ingest_mainv3(history: dict) -> int:
    """
    Read active_picks.json and log any newly resolved picks.
    Extracts full context: setup type, grade, confidence, daily signals,
    outcome, P&L, how many days it took.
    """
    if not os.path.exists(ACTIVE_PICKS):
        print(f"  [SKIP] {ACTIVE_PICKS} not found")
        return 0

    with open(ACTIVE_PICKS) as f:
        data = json.load(f)

    existing_keys = {o["_key"] for o in history["outcomes"]}
    added = 0

    for pick in data.get("picks", []):
        status = pick.get("status", "")
        if status not in RESOLVED_STATUSES:
            continue

        key = _outcome_key(
            pick["symbol"], pick["pick_date"], "main_v3"
        )
        if key in existing_keys:
            continue

        # Calculate outcome
        entry      = pick.get("entry", 0)
        exit_price = pick.get("exit_price")
        t1_hit     = pick.get("t1_hit_date")
        sl_hit     = pick.get("sl_hit_date")

        if exit_price and entry:
            pnl_pct = round((exit_price - entry) / entry * 100, 2)
        else:
            pnl_pct = None

        # Was it a win?
        is_win = status in ("COMPLETED_T1", "COMPLETED_T2")
        is_loss = status == "COMPLETED_SL"
        is_invalid = status == "INVALIDATED"

        # How many days from pick to resolution
        pick_dt = pd.to_datetime(pick.get("pick_date"))
        exit_dt = pd.to_datetime(pick.get("exit_date") or
                                  pick.get("t1_hit_date") or
                                  pick.get("sl_hit_date"))
        days_to_resolve = (exit_dt - pick_dt).days if (
            exit_dt and pick_dt and not pd.isna(exit_dt)
        ) else None

        # Extract daily log signals at entry (first day)
        daily_log  = pick.get("daily_log", [])
        entry_day  = daily_log[0] if daily_log else {}
        peak_pnl   = max((d.get("pnl_pct", 0) for d in daily_log), default=0)
        worst_pnl  = min((d.get("pnl_pct", 0) for d in daily_log), default=0)

        outcome = {
            "_key":              key,
            "symbol":            pick["symbol"],
            "source":            "main_v3",
            "pick_date":         str(pick.get("pick_date", "")),
            "exit_date":         str(exit_dt.date()) if exit_dt and not pd.isna(exit_dt) else None,
            "days_to_resolve":   days_to_resolve,

            # Setup context
            "setup_type":        pick.get("setup_type", ""),
            "setup_grade":       pick.get("setup_grade", ""),
            "confidence_pct":    pick.get("confidence_pct", 0),
            "headline":          pick.get("headline", "")[:100],

            # Entry conditions (what signals looked like when picked)
            "entry_trend":       entry_day.get("trend_state", ""),
            "entry_momentum":    entry_day.get("momentum_state", ""),
            "entry_volume":      entry_day.get("volume_state", ""),
            "entry_rsi":         entry_day.get("rsi"),
            "entry_verdict":     entry_day.get("verdict", ""),

            # Outcome
            "status":            status,
            "is_win":            is_win,
            "is_loss":           is_loss,
            "is_invalid":        is_invalid,
            "entry_price":       entry,
            "exit_price":        exit_price,
            "pnl_pct":           pnl_pct,
            "t1_hit":            t1_hit is not None,
            "sl_hit":            sl_hit is not None,
            "peak_pnl_pct":      round(peak_pnl, 2),
            "worst_pnl_pct":     round(worst_pnl, 2),

            # Pattern context (from main_v3)
            "pattern":           None,
            "pattern_score":     None,
            "vol_confirmed":     None,
        }

        history["outcomes"].append(outcome)
        existing_keys.add(key)
        added += 1
        print(f"  Logged: {pick['symbol']:<15} {status:<15} P&L: {pnl_pct}%")

    return added


# ─────────────────────────────────────────────────────────────────────────────
# INGEST FROM unified_watchlist.json (pattern + H&S tracker)
# ─────────────────────────────────────────────────────────────────────────────

def ingest_patterns(history: dict) -> int:
    """
    Read unified_watchlist.json and log any newly resolved pattern picks.
    """
    if not os.path.exists(UNIFIED_WL):
        print(f"  [SKIP] {UNIFIED_WL} not found")
        return 0

    with open(UNIFIED_WL) as f:
        wl = json.load(f)

    existing_keys = {o["_key"] for o in history["outcomes"]}
    added = 0

    resolved_states = {"BREAKOUT", "BREAKDOWN", "EXPIRED", "FAILED"}

    for bucket_name in ("patterns", "hs"):
        bucket = wl.get(bucket_name, {})
        for key_str, pick in bucket.items():
            state = pick.get("state", "")
            if state not in resolved_states:
                continue

            source   = pick.get("source", bucket_name)
            date_add = pick.get("date_added", "")
            key      = _outcome_key(pick["symbol"], date_add, source)

            if key in existing_keys:
                continue

            # P&L calculation
            h         = pick.get("price_history", [])
            entry_bar = h[0] if h else {}
            exit_bar  = h[-1] if h else {}

            if bucket_name == "patterns":
                entry_price = pick.get("breakout_level", 0)
            else:
                entry_price = pick.get("neckline_price", 0)

            exit_price  = exit_bar.get("close")
            pnl_pct     = round((exit_price - entry_price) / entry_price * 100, 2) \
                          if (exit_price and entry_price) else None

            is_win   = state == "BREAKOUT"
            is_loss  = state in ("BREAKDOWN", "FAILED")
            is_exp   = state == "EXPIRED"

            pick_dt  = pd.to_datetime(date_add) if date_add else None
            exit_dt  = pd.to_datetime(pick.get("date_resolved")) \
                       if pick.get("date_resolved") else None
            days_res = (exit_dt - pick_dt).days \
                       if (exit_dt and pick_dt) else len(h)

            peak_pnl  = max((d.get("gap_pct", 0) for d in h), default=0)

            outcome = {
                "_key":             key,
                "symbol":           pick["symbol"],
                "source":           source,
                "pick_date":        date_add,
                "exit_date":        str(exit_dt.date()) if exit_dt else None,
                "days_to_resolve":  days_res,

                # Setup context
                "setup_type":       pick.get("pattern", pick.get("pattern_type", "")),
                "setup_grade":      pick.get("confidence", ""),
                "confidence_pct":   pick.get("score", pick.get("quality_score", 0)),
                "headline":         pick.get("trigger_condition", "")[:100],

                # Entry conditions
                "entry_trend":      "",
                "entry_momentum":   "",
                "entry_volume":     "vol_ok" if pick.get("vol_all_3_ok") else "vol_fail",
                "entry_rsi":        None,
                "entry_verdict":    pick.get("watchlist_category", ""),

                # Outcome
                "status":           state,
                "is_win":           is_win,
                "is_loss":          is_loss,
                "is_invalid":       is_exp,
                "entry_price":      entry_price,
                "exit_price":       exit_price,
                "pnl_pct":          pnl_pct,
                "t1_hit":           state == "BREAKOUT",
                "sl_hit":           is_loss,
                "peak_pnl_pct":     round(peak_pnl, 2),
                "worst_pnl_pct":    0.0,

                # Pattern specific
                "pattern":          pick.get("pattern", pick.get("pattern_type")),
                "pattern_score":    pick.get("score", pick.get("quality_score")),
                "vol_confirmed":    pick.get("vol_all_3_ok",
                                             pick.get("volume_confirmed")),
                "weekly_confirmed": pick.get("weekly_confirmed"),
                "expiry_used":      pick.get("expiry_date"),
            }

            history["outcomes"].append(outcome)
            existing_keys.add(key)
            added += 1
            print(f"  Logged [{source}]: {pick['symbol']:<15} {state:<12}")

    return added


# ─────────────────────────────────────────────────────────────────────────────
# ANALYSIS ENGINE
# ─────────────────────────────────────────────────────────────────────────────

def analyse(outcomes: list) -> dict:
    """
    Build analysis from all resolved outcomes.
    Returns structured dict with hit rates, P&L, and insights.
    """
    if not outcomes:
        return {"total": 0, "sufficient": False}

    total     = len(outcomes)
    wins      = [o for o in outcomes if o.get("is_win")]
    losses    = [o for o in outcomes if o.get("is_loss")]
    invalids  = [o for o in outcomes if o.get("is_invalid")]

    win_rate  = len(wins) / total * 100 if total > 0 else 0
    avg_win   = np.mean([o["pnl_pct"] for o in wins if o.get("pnl_pct")]) \
                if wins else 0
    avg_loss  = np.mean([o["pnl_pct"] for o in losses if o.get("pnl_pct")]) \
                if losses else 0
    avg_days  = np.mean([o["days_to_resolve"] for o in outcomes
                         if o.get("days_to_resolve")]) if outcomes else 0

    # ── By setup type ─────────────────────────────────────────────────────────
    by_setup = defaultdict(lambda: {"wins": 0, "losses": 0, "pnls": [], "days": []})
    for o in outcomes:
        st = o.get("setup_type", "UNKNOWN")
        if o.get("is_win"):   by_setup[st]["wins"] += 1
        if o.get("is_loss"):  by_setup[st]["losses"] += 1
        if o.get("pnl_pct") is not None:
            by_setup[st]["pnls"].append(o["pnl_pct"])
        if o.get("days_to_resolve"):
            by_setup[st]["days"].append(o["days_to_resolve"])

    setup_stats = {}
    for st, d in by_setup.items():
        n = d["wins"] + d["losses"]
        setup_stats[st] = {
            "total":    n,
            "wins":     d["wins"],
            "losses":   d["losses"],
            "hit_rate": round(d["wins"] / n * 100, 1) if n > 0 else 0,
            "avg_pnl":  round(np.mean(d["pnls"]), 2) if d["pnls"] else 0,
            "avg_days": round(np.mean(d["days"]), 1) if d["days"] else 0,
            "reliable": n >= MIN_SAMPLE,
        }

    # ── By confidence band ────────────────────────────────────────────────────
    bands = {"55-65%": (55,65), "65-75%": (65,75),
             "75-85%": (75,85), "85%+":   (85,100)}
    conf_stats = {}
    for band, (lo, hi) in bands.items():
        band_picks = [o for o in outcomes
                      if lo <= (o.get("confidence_pct") or 0) < hi]
        n_b = len(band_picks)
        wins_b = sum(1 for o in band_picks if o.get("is_win"))
        pnls_b = [o["pnl_pct"] for o in band_picks if o.get("pnl_pct") is not None]
        conf_stats[band] = {
            "total":    n_b,
            "hit_rate": round(wins_b / n_b * 100, 1) if n_b > 0 else 0,
            "avg_pnl":  round(np.mean(pnls_b), 2) if pnls_b else 0,
            "reliable": n_b >= MIN_SAMPLE,
        }

    # ── By grade ──────────────────────────────────────────────────────────────
    by_grade = defaultdict(lambda: {"wins": 0, "losses": 0, "pnls": []})
    for o in outcomes:
        g = o.get("setup_grade", "?")
        if o.get("is_win"):  by_grade[g]["wins"] += 1
        if o.get("is_loss"): by_grade[g]["losses"] += 1
        if o.get("pnl_pct") is not None:
            by_grade[g]["pnls"].append(o["pnl_pct"])

    grade_stats = {}
    for g, d in sorted(by_grade.items()):
        n = d["wins"] + d["losses"]
        grade_stats[g] = {
            "total":    n,
            "hit_rate": round(d["wins"] / n * 100, 1) if n > 0 else 0,
            "avg_pnl":  round(np.mean(d["pnls"]), 2) if d["pnls"] else 0,
            "reliable": n >= MIN_SAMPLE,
        }

    # ── By source ─────────────────────────────────────────────────────────────
    by_source = defaultdict(lambda: {"wins": 0, "losses": 0, "pnls": []})
    for o in outcomes:
        s = o.get("source", "unknown")
        if o.get("is_win"):  by_source[s]["wins"] += 1
        if o.get("is_loss"): by_source[s]["losses"] += 1
        if o.get("pnl_pct") is not None:
            by_source[s]["pnls"].append(o["pnl_pct"])

    source_stats = {}
    for s, d in by_source.items():
        n = d["wins"] + d["losses"]
        source_stats[s] = {
            "total":    n,
            "hit_rate": round(d["wins"] / n * 100, 1) if n > 0 else 0,
            "avg_pnl":  round(np.mean(d["pnls"]), 2) if d["pnls"] else 0,
        }

    # ── Entry condition analysis ──────────────────────────────────────────────
    # What trend/momentum/volume state at entry correlates with wins?
    def signal_hit_rate(field: str, value: str) -> dict:
        matched = [o for o in outcomes if o.get(field) == value]
        n = len(matched)
        w = sum(1 for o in matched if o.get("is_win"))
        return {"n": n, "hit_rate": round(w/n*100, 1) if n > 0 else 0}

    entry_signals = {}
    for field in ("entry_trend", "entry_momentum", "entry_volume"):
        vals = set(o.get(field, "") for o in outcomes if o.get(field))
        entry_signals[field] = {
            v: signal_hit_rate(field, v) for v in vals
        }

    # ── KEY INSIGHTS ──────────────────────────────────────────────────────────
    insights = []
    data_note = (f"Based on {total} resolved picks. "
                 + (f"Need {MIN_RELIABLE - total} more for reliable conclusions."
                    if total < MIN_RELIABLE else "Sample size is reliable."))
    insights.append(data_note)

    # Best setup type
    reliable_setups = {k: v for k, v in setup_stats.items() if v["reliable"]}
    if reliable_setups:
        best = max(reliable_setups, key=lambda k: reliable_setups[k]["hit_rate"])
        worst = min(reliable_setups, key=lambda k: reliable_setups[k]["hit_rate"])
        insights.append(
            f"Best setup: {best} ({reliable_setups[best]['hit_rate']}% hit rate, "
            f"avg P&L {reliable_setups[best]['avg_pnl']:+.1f}%)"
        )
        if reliable_setups[worst]["hit_rate"] < 40:
            insights.append(
                f"⚠ Underperformer: {worst} ({reliable_setups[worst]['hit_rate']}% hit rate) "
                f"— consider raising min confidence or grade threshold for this setup type"
            )

    # Confidence vs outcome
    reliable_bands = {k: v for k, v in conf_stats.items() if v["reliable"]}
    if reliable_bands:
        best_band = max(reliable_bands, key=lambda k: reliable_bands[k]["hit_rate"])
        insights.append(
            f"Best confidence band: {best_band} "
            f"({reliable_bands[best_band]['hit_rate']}% hit rate)"
        )

    # Volume confirmation
    vol_wins = [o for o in outcomes if o.get("vol_confirmed") and o.get("is_win")]
    vol_losses = [o for o in outcomes if o.get("vol_confirmed") and o.get("is_loss")]
    if vol_wins or vol_losses:
        n_vol = len(vol_wins) + len(vol_losses)
        if n_vol >= 5:
            insights.append(
                f"Volume confirmed picks: {len(vol_wins)}/{n_vol} wins "
                f"({len(vol_wins)/n_vol*100:.0f}% hit rate)"
            )

    return {
        "total":         total,
        "wins":          len(wins),
        "losses":        len(losses),
        "invalids":      len(invalids),
        "win_rate":      round(win_rate, 1),
        "avg_win_pct":   round(avg_win, 2),
        "avg_loss_pct":  round(avg_loss, 2),
        "avg_days":      round(avg_days, 1),
        "sufficient":    total >= MIN_SAMPLE,
        "reliable":      total >= MIN_RELIABLE,
        "setup_stats":   setup_stats,
        "conf_stats":    conf_stats,
        "grade_stats":   grade_stats,
        "source_stats":  source_stats,
        "entry_signals": entry_signals,
        "insights":      insights,
        "need_more":     max(0, MIN_RELIABLE - total),
    }


# ─────────────────────────────────────────────────────────────────────────────
# PERFORMANCE DASHBOARD GENERATOR
# ─────────────────────────────────────────────────────────────────────────────

def generate_performance_dashboard(history: dict, analysis: dict):
    now      = datetime.now(IST).strftime("%d-%b-%Y %H:%M IST")
    outcomes = history.get("outcomes", [])
    total    = analysis.get("total", 0)
    reliable = analysis.get("reliable", False)
    need     = analysis.get("need_more", 0)

    def pct_bar(pct: float, max_pct: float = 100,
                color: str = "#22c55e") -> str:
        w = min(100, int(pct / max_pct * 100))
        return (f'<div style="width:120px;height:8px;background:#e5e7eb;'
                f'border-radius:4px;display:inline-block;vertical-align:middle">'
                f'<div style="width:{w}%;height:8px;background:{color};'
                f'border-radius:4px"></div></div> '
                f'<span style="font-size:12px;color:#374151">{pct:.1f}%</span>')

    def stat_card(n, label, color="#1d4ed8"):
        return (f'<div style="background:#fff;border:1px solid #e2e8f0;'
                f'border-radius:10px;padding:14px 18px;min-width:110px;'
                f'box-shadow:0 1px 2px rgba(0,0,0,.05)">'
                f'<div style="font-size:26px;font-weight:700;color:{color}">{n}</div>'
                f'<div style="font-size:11px;color:#64748b;margin-top:2px">{label}</div>'
                f'</div>')

    def section_table(title: str, stats: dict,
                      cols: list, emoji: str = "") -> str:
        if not stats:
            return ""
        rows = ""
        for key, d in sorted(stats.items(),
                              key=lambda x: x[1].get("hit_rate", 0),
                              reverse=True):
            reliable_tag = ("" if d.get("reliable", True)
                            else ' <span style="color:#94a3b8;font-size:10px">'
                                 '(low sample)</span>')
            row = f'<tr><td style="padding:8px 10px;font-weight:600">{key}{reliable_tag}</td>'
            for col in cols:
                val = d.get(col, "—")
                if col == "hit_rate":
                    color = ("#166534" if isinstance(val, (int,float)) and val >= 60
                             else "#991b1b" if isinstance(val, (int,float)) and val < 40
                             else "#92400e")
                    row += (f'<td style="padding:8px 10px;text-align:center">'
                            f'{pct_bar(val if isinstance(val,(int,float)) else 0, color=color)}'
                            f'</td>')
                elif col == "avg_pnl":
                    color = "#166534" if isinstance(val,(int,float)) and val > 0 else "#991b1b"
                    row += (f'<td style="padding:8px 10px;text-align:center;'
                            f'color:{color};font-weight:600">'
                            f'{val:+.1f}%</td>' if isinstance(val,(int,float))
                            else f'<td style="padding:8px 10px;text-align:center">—</td>')
                else:
                    row += f'<td style="padding:8px 10px;text-align:center">{val}</td>'
            rows += row + "</tr>"

        col_headers = "".join(f'<th style="padding:8px 10px;text-align:center;'
                               f'white-space:nowrap">{c.replace("_"," ").title()}</th>'
                               for c in cols)
        return f"""
        <div style="margin-bottom:24px">
          <div style="font-size:13px;font-weight:600;color:#475569;
                      text-transform:uppercase;letter-spacing:.06em;
                      margin-bottom:8px;border-left:3px solid #3b82f6;
                      padding-left:10px">{emoji} {title}</div>
          <table style="width:100%;border-collapse:collapse;background:#fff;
                        border-radius:10px;overflow:hidden;
                        box-shadow:0 1px 2px rgba(0,0,0,.06);font-size:12px">
            <thead style="background:#0f172a;color:#fff">
              <tr>
                <th style="padding:8px 10px;text-align:left">Category</th>
                {col_headers}
              </tr>
            </thead>
            <tbody>{rows}</tbody>
          </table>
        </div>"""

    # Recent outcomes table
    recent_rows = ""
    for o in sorted(outcomes, key=lambda x: x.get("pick_date",""), reverse=True)[:20]:
        win_col = ("#166534" if o.get("is_win") else
                   "#991b1b" if o.get("is_loss") else "#6b7280")
        result  = ("WIN ✓" if o.get("is_win") else
                   "LOSS ✗" if o.get("is_loss") else
                   "INVALID" if o.get("is_invalid") else o.get("status","—"))
        pnl     = o.get("pnl_pct")
        pnl_str = f"{pnl:+.1f}%" if pnl is not None else "—"
        pnl_col = "#166534" if pnl and pnl > 0 else "#991b1b"

        recent_rows += f"""<tr style="border-bottom:1px solid #f1f5f9">
          <td style="padding:7px 10px;font-weight:600">{o.get('symbol','')}</td>
          <td style="padding:7px 10px;font-size:11px;color:#475569">{o.get('source','')}</td>
          <td style="padding:7px 10px;font-size:11px">{o.get('setup_type','')}</td>
          <td style="padding:7px 10px;font-size:11px">{o.get('setup_grade','')}</td>
          <td style="padding:7px 10px;font-size:11px">{o.get('confidence_pct',0):.0f}%</td>
          <td style="padding:7px 10px;font-size:11px">{o.get('pick_date','')}</td>
          <td style="padding:7px 10px;font-size:11px">{o.get('exit_date','—')}</td>
          <td style="padding:7px 10px;font-size:11px;text-align:center">
            {o.get('days_to_resolve','—')}</td>
          <td style="padding:7px 10px;font-weight:600;color:{win_col}">{result}</td>
          <td style="padding:7px 10px;font-weight:600;color:{pnl_col}">{pnl_str}</td>
          <td style="padding:7px 10px;font-size:11px;color:#64748b">
            {o.get('entry_trend','')}/{o.get('entry_momentum','')}</td>
        </tr>"""

    # Insights
    insights_html = "".join(
        f'<div style="padding:6px 0;font-size:12px;color:#374151;'
        f'border-bottom:1px solid #f1f5f9">'
        f'{"⚠" if "⚠" in i else "→"} {i}</div>'
        for i in analysis.get("insights", [])
    )

    data_status = (
        f'<div style="background:#fef3c7;border:1px solid #fbbf24;'
        f'border-radius:8px;padding:10px 14px;font-size:12px;color:#92400e;'
        f'margin-bottom:16px">'
        f'⏳ <b>Collecting data</b> — {total} of {MIN_RELIABLE} picks needed '
        f'for reliable conclusions. Need {need} more resolved picks. '
        f'Stats shown are directional only.</div>'
    ) if not reliable else (
        f'<div style="background:#d1fae5;border:1px solid #6ee7b7;'
        f'border-radius:8px;padding:10px 14px;font-size:12px;color:#065f46;'
        f'margin-bottom:16px">'
        f'✅ <b>Reliable sample</b> — {total} resolved picks. '
        f'Stats are statistically meaningful.</div>'
    )

    html = f"""<!DOCTYPE html><html lang="en"><head>
<meta charset="UTF-8">
<title>MaverickPICKS Performance</title>
<style>
* {{ box-sizing:border-box; margin:0; padding:0 }}
body {{ font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",Arial,sans-serif;
       background:#f1f5f9; color:#1e293b; }}
.hdr {{ background:linear-gradient(135deg,#0f172a,#7c3aed);
       color:#fff; padding:20px 28px; }}
.hdr h1 {{ font-size:20px; font-weight:700 }}
.hdr p  {{ font-size:12px; opacity:.75; margin-top:3px }}
.stats {{ display:flex; gap:12px; padding:16px 28px; flex-wrap:wrap }}
.section {{ padding:0 28px 24px }}
table {{ font-size:12px }}
tbody tr:hover {{ background:#f8fafc!important }}
</style>
</head><body>

<div class="hdr">
  <h1>MaverickPICKS — Performance & Self-Improvement Engine</h1>
  <p>Updated: {now} &nbsp;|&nbsp;
     Tracks what works, what doesn't, and why — to improve future picks</p>
</div>

<div class="stats">
  {stat_card(total, "Total Resolved")}
  {stat_card(analysis.get('wins',0), "Wins ✓", "#166534")}
  {stat_card(analysis.get('losses',0), "Losses ✗", "#991b1b")}
  {stat_card(f"{analysis.get('win_rate',0):.0f}%", "Hit Rate", "#1d4ed8")}
  {stat_card(f"{analysis.get('avg_win_pct',0):+.1f}%", "Avg Win", "#166534")}
  {stat_card(f"{analysis.get('avg_loss_pct',0):+.1f}%", "Avg Loss", "#991b1b")}
  {stat_card(f"{analysis.get('avg_days',0):.0f}d", "Avg Days to Resolve")}
</div>

<div class="section">
  {data_status}

  <div style="background:#fff;border:1px solid #e2e8f0;border-radius:10px;
              padding:14px 16px;margin-bottom:20px">
    <div style="font-size:13px;font-weight:600;margin-bottom:8px">
      💡 Key Insights</div>
    {insights_html or '<div style="color:#94a3b8;font-size:12px">Not enough data yet.</div>'}
  </div>

  {section_table("Performance by Setup Type", analysis.get("setup_stats",{}),
                 ["total","wins","losses","hit_rate","avg_pnl","avg_days"], "📊")}

  {section_table("Performance by Confidence Band",
                 analysis.get("conf_stats",{}),
                 ["total","hit_rate","avg_pnl"], "🎯")}

  {section_table("Performance by Setup Grade",
                 analysis.get("grade_stats",{}),
                 ["total","hit_rate","avg_pnl"], "🏅")}

  {section_table("Performance by Scanner Source",
                 analysis.get("source_stats",{}),
                 ["total","hit_rate","avg_pnl"], "🔍")}

  <div style="margin-bottom:24px">
    <div style="font-size:13px;font-weight:600;color:#475569;
                text-transform:uppercase;letter-spacing:.06em;
                margin-bottom:8px;border-left:3px solid #3b82f6;
                padding-left:10px">📋 Recent Resolved Picks (last 20)</div>
    <div style="overflow-x:auto">
    <table style="width:100%;border-collapse:collapse;background:#fff;
                  border-radius:10px;overflow:hidden;
                  box-shadow:0 1px 2px rgba(0,0,0,.06)">
      <thead style="background:#0f172a;color:#fff">
        <tr>
          <th style="padding:8px 10px;text-align:left">Symbol</th>
          <th style="padding:8px 10px">Source</th>
          <th style="padding:8px 10px">Setup</th>
          <th style="padding:8px 10px">Grade</th>
          <th style="padding:8px 10px">Conf</th>
          <th style="padding:8px 10px">Pick Date</th>
          <th style="padding:8px 10px">Exit Date</th>
          <th style="padding:8px 10px">Days</th>
          <th style="padding:8px 10px">Result</th>
          <th style="padding:8px 10px">P&L</th>
          <th style="padding:8px 10px">Entry Signals</th>
        </tr>
      </thead>
      <tbody>
        {recent_rows or '<tr><td colspan="11" style="text-align:center;padding:30px;color:#94a3b8">No resolved picks yet — keep tracking.</td></tr>'}
      </tbody>
    </table>
    </div>
  </div>

  <div style="font-size:11px;color:#64748b;padding-bottom:20px">
    ⓘ Hit rate = wins ÷ (wins + losses), excluding invalidated picks.
    Stats marked "low sample" have fewer than {MIN_SAMPLE} data points and should not be acted on.
    Reliable conclusions require {MIN_RELIABLE}+ resolved picks.
    This engine updates automatically after each daily tracker run.
  </div>
</div>
</body></html>"""

    with open(PERF_DASHBOARD, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"  Performance dashboard → {PERF_DASHBOARD}")


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(
        description="MaverickPICKS Performance Logger & Self-Improvement Engine"
    )
    ap.add_argument("--analyse", action="store_true",
                    help="Re-run analysis and regenerate dashboard only")
    ap.add_argument("--reset", action="store_true",
                    help="Clear performance history (careful!)")
    args = ap.parse_args()

    print(f"\n{'═'*60}")
    print("  MaverickPICKS Performance Logger")
    print(f"  {datetime.now(IST).strftime('%d-%b-%Y  %H:%M IST')}")
    print(f"{'═'*60}")

    if args.reset:
        confirm = input("  Clear ALL performance history? (type YES): ")
        if confirm.strip() == "YES":
            save_history({"outcomes": [], "last_updated": None, "total_logged": 0})
            print("  History cleared.")
        return

    history = load_history()

    if not args.analyse:
        # Ingest new resolved picks from both trackers
        print("\n  Ingesting resolved picks...")
        a1 = ingest_mainv3(history)
        a2 = ingest_patterns(history)
        total_added = a1 + a2
        history["total_logged"] = len(history["outcomes"])
        save_history(history)
        print(f"\n  Added: {total_added} new outcome(s)")
        print(f"  Total in history: {len(history['outcomes'])}")

    # Run analysis
    print("\n  Running analysis...")
    analysis = analyse(history["outcomes"])

    print(f"  Total resolved : {analysis['total']}")
    print(f"  Win rate       : {analysis['win_rate']:.1f}%")
    print(f"  Avg win        : {analysis['avg_win_pct']:+.1f}%")
    print(f"  Avg loss       : {analysis['avg_loss_pct']:+.1f}%")
    if not analysis["reliable"]:
        print(f"  ⚠ Need {analysis['need_more']} more resolved picks for reliability")

    for insight in analysis.get("insights", []):
        print(f"  → {insight}")

    generate_performance_dashboard(history, analysis)
    print(f"\n{'═'*60}\n")


if __name__ == "__main__":
    main()

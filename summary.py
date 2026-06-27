"""
summary.py — MaverickPICKS Daily Run Summary
Prints a clean summary to the Actions log after each tracker run.
Called by the tracker_check job in daily_run.yml.
"""
import json
import os
from datetime import datetime, timezone, timedelta

IST = timezone(timedelta(hours=5, minutes=30))

SEP = "=" * 55

print(SEP)
print("  MaverickPICKS DAILY RUN COMPLETE")
print(f"  Time : {datetime.now(IST).strftime('%d-%b-%Y %H:%M IST')}")
print(SEP)

# ── Watchlist summary ─────────────────────────────────────────
wl_file = "unified_watchlist.json"
if os.path.exists(wl_file):
    with open(wl_file) as f:
        wl = json.load(f)

    watching   = 0
    breakouts  = []
    breakdowns = []

    for bucket in ("maverick", "patterns", "hs"):
        for p in wl.get(bucket, {}).values():
            state = p.get("state", "")
            sym   = p.get("symbol", "")
            if state == "WATCHING":
                watching += 1
            elif state == "BREAKOUT":
                breakouts.append(sym)
            elif state in ("BREAKDOWN", "FAILED"):
                breakdowns.append(sym)

    print(f"\n  WATCHLIST")
    print(f"  Watching   : {watching} pick(s)")
    print(f"  Breakouts  : {len(breakouts)}")
    print(f"  Breakdowns : {len(breakdowns)}")

    if breakouts:
        print()
        for sym in breakouts:
            print(f"  *** BREAKOUT ALERT : {sym} ***")

    if breakdowns:
        print()
        for sym in breakdowns:
            print(f"  *** STOP HIT ALERT : {sym} ***")
else:
    print("\n  Watchlist: not found")

# ── Performance summary ───────────────────────────────────────
ph_file = "performance_history.json"
if os.path.exists(ph_file):
    with open(ph_file) as f:
        ph = json.load(f)

    outcomes  = ph.get("outcomes", [])
    total     = len(outcomes)
    wins      = sum(1 for o in outcomes if o.get("is_win"))
    losses    = sum(1 for o in outcomes if o.get("is_loss"))
    win_rate  = round(wins / total * 100, 1) if total > 0 else 0

    print(f"\n  PERFORMANCE HISTORY")
    print(f"  Resolved   : {total} pick(s)")
    print(f"  Wins       : {wins}")
    print(f"  Losses     : {losses}")
    print(f"  Hit rate   : {win_rate}%")

    if total < 30:
        print(f"  Note       : Need {30 - total} more picks for reliable stats")

# ── Dashboard links ───────────────────────────────────────────
print(f"\n  DASHBOARDS")
repo = os.environ.get("GITHUB_REPOSITORY", "Maverickpicks/MaverickPICKSPatterns")
base = f"https://htmlpreview.github.io/?https://raw.githubusercontent.com/{repo}/main"
print(f"  Main v3  : {base}/dashboard_maverick.html")
print(f"  Patterns : {base}/dashboard_patterns.html")
print(f"  Perf     : {base}/dashboard_performance.html")

print(f"\n{SEP}\n")

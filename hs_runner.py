"""
hs_runner.py — MaverickPICKS H&S Pattern Runner
=================================================
Thin wrapper that:
  1. Loads all NIFTY500 symbols from NIFTY500_MASTER.csv
  2. Fetches daily OHLCV for each using the same data_loader.py
     that main_v3.py uses (auto_adjust=False, period=2y)
  3. Runs hs_detector.scan_symbol() on each
  4. Filters to actionable watchlist categories only
  5. Writes hs_watchlist.csv — same format the unified_tracker reads

Run:
  python hs_runner.py
  python hs_runner.py --min_quality 50 --workers 4
"""

import argparse
import time
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone, timedelta

import pandas as pd

from data_loader import load_stock
from hs_detector import scan_symbol, PatternResult

warnings.filterwarnings("ignore")

IST = timezone(timedelta(hours=5, minutes=30))

OUTPUT_FILE = "hs_watchlist.csv"

# Watchlist categories we actually want to track
# Excludes: Failed, Invalidated, False Start, stale patterns (optional)
ACTIONABLE_CATEGORIES = {
    "Watching - Confirmed RS",
    "Watching - Provisional RS",
    "Recent Breakout",
}

# Include stale patterns as a warning — they show in dashboard as STALE
INCLUDE_STALE = True


def load_symbols() -> list:
    try:
        df = pd.read_csv("NIFTY500_MASTER.csv")
        symbols = df["Symbol"].dropna().astype(str).str.strip().tolist()
        print(f"  Loaded {len(symbols)} symbols from NIFTY500_MASTER.csv")
        return symbols
    except Exception as e:
        print(f"  [ERROR] Could not load symbols: {e}")
        return []


def scan_one(symbol: str, min_quality: float) -> list:
    """
    Fetch data for one symbol and run H&S detection.
    Returns list of dicts (one per actionable pattern found).
    """
    try:
        data  = load_stock(symbol)
        daily = data.get("daily", pd.DataFrame())

        if daily is None or daily.empty or len(daily) < 60:
            return []

        # hs_detector expects Date column, not index
        df = daily.copy().reset_index()
        if "Date" not in df.columns:
            df = df.rename(columns={"index": "Date"})
        # Ensure correct column names
        df.columns = [c.strip().title() if c.strip().lower() in
                      ("open","high","low","close","volume","date")
                      else c for c in df.columns]
        df["Date"] = pd.to_datetime(df["Date"])
        df = df.sort_values("Date").reset_index(drop=True)

        results = scan_symbol(df, symbol)

        rows = []
        for r in results:
            # Filter by watchlist category
            cat = r.watchlist_category or ""
            is_stale = "Stale" in cat

            if not cat:
                continue  # Failed / Invalidated / False Start — skip
            if is_stale and not INCLUDE_STALE:
                continue
            if r.quality_score is not None and r.quality_score < min_quality:
                continue

            rows.append(_to_row(symbol, r))

        return rows

    except Exception as e:
        print(f"  [WARN] {symbol}: {e}")
        return []


def _to_row(symbol: str, r: PatternResult) -> dict:
    """Convert PatternResult to a flat dict matching hs_watchlist.csv format."""
    return {
        "Symbol":               symbol,
        "Pattern Type":         r.pattern_type,
        "Status":               r.status,
        "Watchlist Category":   r.watchlist_category or "",
        "Left Shoulder Date":   _fmt_date(r.left_shoulder_date),
        "Left Shoulder Price":  _fmt_price(r.left_shoulder_price),
        "Head Date":            _fmt_date(r.head_date),
        "Head Price":           _fmt_price(r.head_price),
        "Right Shoulder Date":  _fmt_date(r.right_shoulder_date),
        "Right Shoulder Price": _fmt_price(r.right_shoulder_price),
        "RS Confirmation":      r.rs_confirmation,
        "RS Age (days)":        r.rs_age_days or "",
        "Is Stale":             "Yes" if r.is_stale else "No",
        "Neckline Price":       _fmt_price(r.neckline_price),
        "Breakout Date":        _fmt_date(r.breakout_date),
        "Breakout Price":       _fmt_price(r.breakout_price),
        "Volume Confirmed":     "Yes" if r.volume_confirmed else "No",
        "Breakout Volume Ratio":round(r.breakout_volume_ratio, 2)
                                if r.breakout_volume_ratio else "",
        "Target":               _fmt_price(r.target),
        "Stop Loss":            _fmt_price(r.stop_loss),
        "Risk:Reward":          round(r.risk_reward, 2) if r.risk_reward else "",
        "Target Hit":           "Yes" if r.target_hit else "No",
        "Quality Score":        round(r.quality_score, 1) if r.quality_score else "",
        "Trigger Condition":    r.trigger_condition or "",
        "Notes":                r.notes or "",
    }


def _fmt_date(val) -> str:
    if val is None:
        return ""
    try:
        return pd.Timestamp(val).strftime("%m/%d/%Y")
    except Exception:
        return str(val)


def _fmt_price(val) -> str:
    if val is None:
        return ""
    try:
        return str(round(float(val), 2))
    except Exception:
        return ""


def run_scanner(symbols: list, min_quality: float,
                workers: int = 3) -> pd.DataFrame:
    """Parallel scan across all symbols."""
    total = len(symbols)
    all_rows = []
    done = 0

    print(f"\n  Scanning {total} symbols for H&S patterns "
          f"({workers} workers)...\n")

    with ThreadPoolExecutor(max_workers=workers) as ex:
        futures = {ex.submit(scan_one, sym, min_quality): sym
                   for sym in symbols}
        for fut in as_completed(futures):
            sym  = futures[fut]
            done += 1
            rows = fut.result()
            all_rows.extend(rows)

            pct = done / total * 100
            bar = "█" * int(pct / 5) + "░" * (20 - int(pct / 5))
            found = f"  {len(all_rows)} pattern(s) found" if all_rows else ""
            print(f"\r  [{bar}] {pct:5.1f}%  {done}/{total}  "
                  f"{sym:<20}{found}", end="", flush=True)

    print()
    return pd.DataFrame(all_rows) if all_rows else pd.DataFrame()


def main():
    ap = argparse.ArgumentParser(
        description="MaverickPICKS H&S Pattern Runner"
    )
    ap.add_argument("--min_quality", type=float, default=40.0,
                    help="Minimum quality score 0-100 (default: 40)")
    ap.add_argument("--workers", type=int, default=3,
                    help="Parallel download threads (default: 3)")
    ap.add_argument("--include_stale", action="store_true",
                    help="Include stale patterns in output")
    args = ap.parse_args()

    global INCLUDE_STALE
    INCLUDE_STALE = args.include_stale

    print(f"\n{'═'*60}")
    print("  MaverickPICKS — H&S Pattern Runner")
    print(f"  {datetime.now(IST).strftime('%d-%b-%Y  %H:%M IST')}")
    print(f"  Min quality : {args.min_quality}")
    print(f"  Workers     : {args.workers}")
    print(f"  Stale       : {'included' if INCLUDE_STALE else 'excluded'}")
    print(f"{'═'*60}")

    symbols = load_symbols()
    if not symbols:
        print("  No symbols found. Exiting.")
        return

    t0  = time.time()
    df  = run_scanner(symbols, args.min_quality, args.workers)
    elapsed = time.time() - t0

    print(f"\n  Scan complete in {elapsed/60:.1f} min")

    if df.empty:
        print("  No actionable H&S patterns found.")
        # Write empty CSV with headers so tracker doesn't error
        pd.DataFrame(columns=[
            "Symbol", "Pattern Type", "Status", "Watchlist Category",
            "Left Shoulder Date", "Left Shoulder Price", "Head Date",
            "Head Price", "Right Shoulder Date", "Right Shoulder Price",
            "RS Confirmation", "RS Age (days)", "Is Stale", "Neckline Price",
            "Breakout Date", "Breakout Price", "Volume Confirmed",
            "Breakout Volume Ratio", "Target", "Stop Loss", "Risk:Reward",
            "Target Hit", "Quality Score", "Trigger Condition", "Notes",
        ]).to_csv(OUTPUT_FILE, index=False)
        print(f"  Empty {OUTPUT_FILE} written.")
        return

    # Sort by quality score descending
    df = df.sort_values("Quality Score", ascending=False).reset_index(drop=True)

    df.to_csv(OUTPUT_FILE, index=False)
    print(f"\n  {len(df)} pattern(s) saved → {OUTPUT_FILE}")

    # Summary
    print(f"\n{'═'*60}")
    print(f"  H&S SCAN SUMMARY")
    print(f"{'═'*60}")
    for _, row in df.iterrows():
        stale_tag = " ⚠ STALE" if row["Is Stale"] == "Yes" else ""
        print(f"  {row['Symbol']:<18} {row['Pattern Type']:<14} "
              f"Q:{row['Quality Score']:>5}  "
              f"{row['Watchlist Category'][:35]}{stale_tag}")
    print(f"{'═'*60}\n")


if __name__ == "__main__":
    main()

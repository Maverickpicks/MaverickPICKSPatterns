"""
H&S / Inverse H&S Pattern Scanner — Batch Runner
==================================================
Standalone script, separate from MaverickPICKS main_v3.

Usage:
    python batch_scan.py --symbols nifty500.csv --lookback 365 --out hs_scan_results.csv

Requires a CSV file with a 'Symbol' column (reuse your existing MaverickPICKS
NIFTY500 list). Each symbol should be the bare NSE symbol e.g. RELIANCE, TCS
(no .NS suffix — that's added automatically).
"""

import argparse
import csv
import time
import sys
from datetime import datetime

from fetch_data import fetch_ohlcv, load_symbol_list
from hs_detector import scan_symbol


CSV_COLUMNS = [
    "Symbol", "Pattern Type", "Status", "Watchlist Category",
    "Left Shoulder Date", "Left Shoulder Price",
    "Head Date", "Head Price",
    "Right Shoulder Date", "Right Shoulder Price", "RS Confirmation",
    "RS Age (days)", "Is Stale",
    "Neckline Price",
    "Breakout Date", "Breakout Price",
    "Volume Confirmed", "Breakout Volume Ratio",
    "Target", "Stop Loss", "Risk:Reward", "Target Hit",
    "Quality Score",
    "Trigger Condition",
    "Notes",
]


def result_to_row(r):
    return {
        "Symbol": r.symbol,
        "Pattern Type": r.pattern_type,
        "Status": r.status,
        "Watchlist Category": r.watchlist_category or "",
        "Left Shoulder Date": r.left_shoulder_date.date().isoformat(),
        "Left Shoulder Price": round(r.left_shoulder_price, 2),
        "Head Date": r.head_date.date().isoformat(),
        "Head Price": round(r.head_price, 2),
        "Right Shoulder Date": r.right_shoulder_date.date().isoformat(),
        "Right Shoulder Price": round(r.right_shoulder_price, 2),
        "RS Confirmation": r.rs_confirmation,
        "RS Age (days)": r.rs_age_days if r.rs_age_days is not None else "",
        "Is Stale": "Yes" if r.is_stale else "No",
        "Neckline Price": round(r.neckline_price, 2),
        "Breakout Date": r.breakout_date.date().isoformat() if r.breakout_date else "",
        "Breakout Price": round(r.breakout_price, 2) if r.breakout_price else "",
        "Volume Confirmed": "Yes" if r.volume_confirmed else "No",
        "Breakout Volume Ratio": r.breakout_volume_ratio if r.breakout_volume_ratio else "",
        "Target": round(r.target, 2) if r.target else "",
        "Stop Loss": round(r.stop_loss, 2) if r.stop_loss else "",
        "Risk:Reward": r.risk_reward if r.risk_reward else "",
        "Target Hit": "Yes" if r.target_hit else "No",
        "Quality Score": r.quality_score if r.quality_score is not None else "",
        "Trigger Condition": r.trigger_condition or "",
        "Notes": r.notes,
    }


def run_scan(symbols_file: str, lookback_days: int, out_file: str,
             only_confirmed: bool = False, watchlist_only: bool = False,
             sleep_between: float = 0.3):
    symbols = load_symbol_list(symbols_file)
    print(f"Loaded {len(symbols)} symbols from {symbols_file}")
    print(f"Lookback: {lookback_days} days | Output: {out_file}")
    if watchlist_only:
        print("Mode: WATCHLIST ONLY (pre-breakout + recent breakouts, excludes stale/target-hit/failed)")
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

    all_rows = []
    failed_fetches = []

    for i, symbol in enumerate(symbols, 1):
        print(f"[{i}/{len(symbols)}] {symbol} ...", end=" ")
        df = fetch_ohlcv(symbol, lookback_days=lookback_days)

        if df.empty:
            print("no data, skipped")
            failed_fetches.append(symbol)
            time.sleep(sleep_between)
            continue

        try:
            results = scan_symbol(df, symbol)
        except Exception as e:
            print(f"detector error: {e}")
            failed_fetches.append(symbol)
            time.sleep(sleep_between)
            continue

        if watchlist_only:
            results = [r for r in results if r.watchlist_category is not None]
        elif only_confirmed:
            results = [r for r in results if r.status == "Confirmed"]

        if results:
            print(f"{len(results)} pattern(s) found")
            for r in results:
                all_rows.append((r.quality_score or 0, result_to_row(r)))
        else:
            print("none")

        time.sleep(sleep_between)  # be polite to the data source

    # sort by quality score descending so the cleanest setups are at the top
    all_rows.sort(key=lambda x: x[0], reverse=True)

    with open(out_file, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        for _, row in all_rows:
            writer.writerow(row)

    print(f"\nDone. {len(all_rows)} pattern(s) written to {out_file}")
    if failed_fetches:
        print(f"Failed/empty fetches ({len(failed_fetches)}): {', '.join(failed_fetches[:20])}"
              + (" ..." if len(failed_fetches) > 20 else ""))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="H&S / Inverse H&S pattern scanner")
    parser.add_argument("--symbols", required=True, help="CSV file with a 'Symbol' column")
    parser.add_argument("--lookback", type=int, default=365, help="Days of price history to fetch")
    parser.add_argument("--out", default="hs_scan_results.csv", help="Output CSV path")
    parser.add_argument("--only-confirmed", action="store_true",
                         help="Only write Confirmed patterns (skip Forming/Failed/False Start)")
    parser.add_argument("--watchlist", action="store_true",
                         help="Only write actionable setups: pre-breakout (Watching) and recent "
                              "breakouts not yet hit target. Excludes stale/Failed/False Start. "
                              "Output sorted by Quality Score descending.")
    args = parser.parse_args()

    run_scan(args.symbols, args.lookback, args.out,
              only_confirmed=args.only_confirmed, watchlist_only=args.watchlist)

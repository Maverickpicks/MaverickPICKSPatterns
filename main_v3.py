import time
import pandas as pd
from datetime import datetime

from data_loader import load_stock, load_nifty
from sector_map import load_sector_map, build_sector_map, get_sector_peers

from trend_engine_v2 import trend_analysis
from momentum_engine_v2 import momentum_analysis
from relative_strength_v2 import relative_strength_analysis
from volume_engine import volume_analysis
from pattern_engine import pattern_analysis
from risk_engine import risk_analysis
from ranking_engine_v2 import ranking_engine
from setup_finder import evaluate_setup, build_feature_series
from narrative_engine import build_scenario_text


# ============================================================
# CONFIG
# ============================================================

REPORT_FILE       = "MaverickPICKS_Top10_Report.xlsx"
MIN_DAILY_ROWS    = 260
TOP_N             = 10
TARGET_LOW        = 4.0
TARGET_HIGH       = 10.0
FORWARD_WINDOW    = 10        # trading days (roughly 2 weeks)
MIN_CONFIDENCE    = 55.0      # hard floor — below this, never shown in Top 10
MIN_SAMPLE_SIZE   = 5
MAX_SECTOR_PEERS  = 12


# ============================================================
# LOAD SYMBOLS
# ============================================================

def load_symbols():
    try:
        df = pd.read_csv("NIFTY500_MASTER.csv")
        symbols = df["Symbol"].dropna().astype(str).str.strip().tolist()
        print(f"Loaded {len(symbols)} symbols from NIFTY500_MASTER.csv")
        return symbols
    except Exception as e:
        print(f"Error loading symbols: {e}")
        return []


# ============================================================
# PASS 1: Load all stock data + run core engines
# Cache feature_df (for setup_finder) per symbol for reuse as
# sector-fallback source for OTHER stocks that need it.
# ============================================================

def run_core_engines(symbol, nifty_df):

    try:
        data    = load_stock(symbol)
        daily   = data.get("daily",   pd.DataFrame())
        weekly  = data.get("weekly",  pd.DataFrame())
        monthly = data.get("monthly", pd.DataFrame())

        if daily is None or daily.empty or len(daily) < MIN_DAILY_ROWS:
            return None

        trend    = trend_analysis(daily.copy())
        momentum = momentum_analysis(daily.copy())
        rs       = relative_strength_analysis(daily.copy(), nifty_df.copy())
        volume   = volume_analysis(daily.copy(), weekly.copy(), monthly.copy())
        pattern  = pattern_analysis(daily.copy(), weekly.copy())
        risk     = risk_analysis(daily.copy())

        ranking  = ranking_engine(trend, momentum, rs, volume, pattern, risk)

        feature_df = build_feature_series(daily.copy())

        return {
            "symbol":     symbol,
            "daily":      daily,
            "feature_df": feature_df,
            "trend":      trend,
            "momentum":   momentum,
            "rs":         rs,
            "volume":     volume,
            "pattern":    pattern,
            "risk":       risk,
            "ranking":    ranking,
        }

    except Exception as e:
        print(f"  ERROR (core engines) {symbol}: {e}")
        return None


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 60)
    print("  MaverickPICKS v3 — Evidence-Based Top 10")
    print(f"  {datetime.now().strftime('%d %b %Y  %H:%M:%S')}")
    print("=" * 60)

    print("\nDownloading NIFTY benchmark...")
    nifty_df = load_nifty()
    if nifty_df is None or nifty_df.empty:
        print("FATAL: Could not load NIFTY data. Exiting.")
        return
    print(f"NIFTY loaded: {len(nifty_df)} sessions")
    nifty_last_date = nifty_df.index[-1].strftime("%Y-%m-%d") if hasattr(nifty_df.index[-1], "strftime") else str(nifty_df.index[-1])
    print(f"Latest data date (NIFTY benchmark): {nifty_last_date}")
    print("NOTE: Each stock's own latest data date is shown in its 'Data_As_Of' column,")
    print("      since individual stocks can have slightly different last-trade dates.")

    symbols = load_symbols()
    if not symbols:
        print("FATAL: No symbols found. Exiting.")
        return

    # --- Sector map (cached) ---
    print("\nLoading sector map...")
    sector_map = load_sector_map()
    missing_sectors = [s for s in symbols if s not in sector_map]
    if missing_sectors:
        print(f"{len(missing_sectors)} symbols missing sector info — fetching now (one-time cost)...")
        sector_map = build_sector_map(symbols)
    else:
        print(f"Sector map ready ({len(sector_map)} symbols cached).")


    # ============================================================
    # PASS 1: Core engines for every stock
    # ============================================================

    total = len(symbols)
    core_results = {}

    print(f"\n[Pass 1/2] Running core engines on {total} stocks...\n")

    for i, symbol in enumerate(symbols, 1):
        print(f"[{i:>3}/{total}] {symbol:<20}", end=" ")
        result = run_core_engines(symbol, nifty_df)
        if result:
            core_results[symbol] = result
            print(f"-> OK  Setup: {result['ranking'].get('Setup_Type')}")
        else:
            print("-> SKIPPED")
        time.sleep(0.3)

    print(f"\nPass 1 complete: {len(core_results)}/{total} stocks processed.\n")


    # ============================================================
    # PASS 2: Setup Finder (historical pattern matching)
    # Only run on stocks that already passed a basic quality bar
    # in Pass 1 — no point backtesting a stock the ranking engine
    # already rejected outright.
    # ============================================================

    candidate_symbols = [
        s for s, r in core_results.items()
        if r["ranking"].get("Verdict") not in ["AVOID"]
        and r["ranking"].get("Setup_Type") != "NONE"
    ]

    print(f"[Pass 2/2] Running historical setup-matching on {len(candidate_symbols)} candidates...\n")

    final_rows = []

    for i, symbol in enumerate(candidate_symbols, 1):
        r = core_results[symbol]
        daily = r["daily"]

        # Build sector peer feature dfs lazily, only if needed
        peer_feature_dfs = None

        setup_result = evaluate_setup(
            daily, symbol,
            sector_peer_feature_dfs=None,   # try self-history first
            target_low=TARGET_LOW, target_high=TARGET_HIGH,
            forward_window=FORWARD_WINDOW, min_samples=MIN_SAMPLE_SIZE
        )

        if setup_result.get("Insufficient_Data"):
            peers = get_sector_peers(symbol, sector_map, list(core_results.keys()), max_peers=MAX_SECTOR_PEERS)
            peer_feature_dfs = {
                p: core_results[p]["feature_df"]
                for p in peers
                if p in core_results
            }
            if peer_feature_dfs:
                setup_result = evaluate_setup(
                    daily, symbol,
                    sector_peer_feature_dfs=peer_feature_dfs,
                    target_low=TARGET_LOW, target_high=TARGET_HIGH,
                    forward_window=FORWARD_WINDOW, min_samples=MIN_SAMPLE_SIZE
                )

        print(f"[{i:>3}/{len(candidate_symbols)}] {symbol:<20} Confidence: {setup_result.get('Confidence_Pct')}")

        # Hard floor: must have a real confidence number and meet the bar
        confidence = setup_result.get("Confidence_Pct")
        if confidence is None or confidence < MIN_CONFIDENCE:
            continue

        # Flag (don't silently hide) ATH-only evidence — this is a
        # weaker, "trend continuation" claim, not a fresh setup signal,
        # per the MCX-style false-confidence issue this was built to catch.
        ath_bucket = setup_result.get("ATH_Bucket")
        if ath_bucket == "AT_ATH":
            print(f"    NOTE: {symbol} confidence is based on AT_ATH historical matches (trend continuation, not a fresh setup)")

        scenario = build_scenario_text(
            symbol, r["trend"], r["momentum"], r["rs"], r["volume"],
            r["pattern"], r["risk"], setup_result, daily_df=daily
        )

        row = {
            "Symbol":            symbol,
            "Data_As_Of":        setup_result.get("Data_As_Of"),
            "Verdict":           r["ranking"].get("Verdict"),
            "Setup_Type":        r["ranking"].get("Setup_Type"),
            "Setup_Grade":       r["ranking"].get("Setup_Grade"),
            "Confidence_Pct":    confidence,
            "Sample_Size":       setup_result.get("Sample_Size"),
            "ATH_Bucket":        setup_result.get("ATH_Bucket"),
            "Used_Sector_Fallback": setup_result.get("Used_Sector_Fallback"),
            "Expected_Gain_Pct": setup_result.get("Avg_Max_Gain"),
            "Median_Days":       setup_result.get("Median_Days_To_Target"),
            "Overshoot_Rate":    setup_result.get("Overshoot_Rate_Pct"),

            "Entry":             r["risk"].get("Entry"),
            "Stop_Loss":         r["risk"].get("Stop_Loss"),
            "Target_1":          r["risk"].get("Target_1"),
            "Target_2":          r["risk"].get("Target_2"),
            "Risk_Percent":      r["risk"].get("Risk_Percent"),
            "Reward_Risk":       r["risk"].get("Reward_Risk"),
            "Trade_Quality":     r["risk"].get("Trade_Quality"),

            "Trend_State":       r["trend"].get("Trend_State"),
            "Momentum_State":    r["momentum"].get("Momentum_State"),
            "RS_State":          r["rs"].get("RS_State"),
            "Volume_State":      r["volume"].get("Volume_State"),
            "At_Support":        r["pattern"].get("At_Support"),
            "Support_Level":     r["pattern"].get("Support_Level"),
            "Weekly_Context":    r["pattern"].get("Weekly_Context"),

            "Headline":          scenario["Headline"],
            "Evidence":          scenario["Evidence"],
            "Entry_Exit":        scenario["Entry_Exit"],
            "Scaling_Rule":      scenario["Scaling_Rule"],
        }

        final_rows.append(row)
        time.sleep(0.1)

    print(f"\nPass 2 complete: {len(final_rows)} stocks passed the {MIN_CONFIDENCE}% confidence floor.\n")


    # ============================================================
    # RANK & TRIM TO TOP 10
    # ============================================================

    if not final_rows:
        print("No stocks met the confidence threshold this scan. Try lowering MIN_CONFIDENCE or re-running on a different day.")
        return

    df_final = pd.DataFrame(final_rows)
    df_final = df_final.sort_values("Confidence_Pct", ascending=False).reset_index(drop=True)
    df_top10 = df_final.head(TOP_N)

    print("=" * 60)
    print(f"  TOP {len(df_top10)} PICKS")
    print("=" * 60)
    for _, row in df_top10.iterrows():
        print(f"\n{row['Symbol']} — Confidence: {row['Confidence_Pct']}% | {row['Setup_Type']} (Grade {row['Setup_Grade']}) | Data as of: {row.get('Data_As_Of')}")
        print(f"  {row['Headline']}")


    # ============================================================
    # WRITE EXCEL
    # ============================================================

    write_excel(df_top10, df_final)


def write_excel(df_top10, df_all_candidates):

    try:
        with pd.ExcelWriter(REPORT_FILE, engine="openpyxl") as writer:

            display_cols = [
                "Symbol", "Data_As_Of", "Verdict", "Confidence_Pct", "Setup_Type", "Setup_Grade",
                "ATH_Bucket", "Expected_Gain_Pct", "Median_Days", "Sample_Size", "Used_Sector_Fallback",
                "Entry", "Stop_Loss", "Target_1", "Target_2", "Risk_Percent", "Reward_Risk",
                "Trade_Quality", "Trend_State", "Momentum_State", "RS_State", "Volume_State",
                "At_Support", "Support_Level", "Weekly_Context",
                "Headline", "Evidence", "Entry_Exit", "Scaling_Rule",
            ]

            available = [c for c in display_cols if c in df_top10.columns]

            df_top10[available].to_excel(writer, sheet_name="TOP 10 PICKS", index=False)
            df_all_candidates[available].to_excel(writer, sheet_name="ALL CANDIDATES", index=False)

        print(f"\nReport saved: {REPORT_FILE}")

    except Exception as e:
        print(f"Excel write error: {e}")


if __name__ == "__main__":
    main()

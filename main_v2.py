import os
import time
import pandas as pd
from datetime import datetime

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

REPORT_FILE   = "MaverickPICKS_Report.xlsx"
MIN_DAILY_ROWS = 260


# ============================================================
# BTST FILTER
# ============================================================

def is_btst_candidate(trend, momentum, volume, risk):

    try:

        return (
            trend.get("Trend_State") in ["STRONG", "LEADER"]
            and momentum.get("Momentum_State") in ["STRONG", "LEADER"]
            and volume.get("Volume_Ratio", 0) >= 1.5
            and volume.get("Close_Near_High", False)
            and volume.get("Bull_Candle", False)
            and risk.get("Trade_Quality") in ["GOOD", "EXCELLENT"]
            and not trend.get("ATH_Risk", False)   # No BTST on ATH chasers
        )

    except:
        return False


# ============================================================
# LOAD SYMBOLS
# ============================================================

def load_symbols():

    try:

        df = pd.read_csv("NIFTY500_MASTER.csv")

        symbols = (
            df["Symbol"]
            .dropna()
            .astype(str)
            .str.strip()
            .tolist()
        )

        print(f"Loaded {len(symbols)} symbols from NIFTY500_MASTER.csv")
        return symbols

    except Exception as e:

        print(f"Error loading symbols: {e}")
        return []


# ============================================================
# PROCESS ONE STOCK
# ============================================================

def process_stock(symbol, nifty_df):

    try:

        data    = load_stock(symbol)
        daily   = data.get("daily",   pd.DataFrame())
        weekly  = data.get("weekly",  pd.DataFrame())
        monthly = data.get("monthly", pd.DataFrame())

        if daily is None or daily.empty or len(daily) < MIN_DAILY_ROWS:
            return None

        # --- Run all engines ---

        trend    = trend_analysis(daily.copy())
        momentum = momentum_analysis(daily.copy())
        rs       = relative_strength_analysis(daily.copy(), nifty_df.copy())
        volume   = volume_analysis(daily.copy(), weekly.copy(), monthly.copy())
        pattern  = pattern_analysis(daily.copy(), weekly.copy())
        risk     = risk_analysis(daily.copy())

        ranking  = ranking_engine(trend, momentum, rs, volume, pattern, risk)
        reason   = generate_reason(trend, momentum, rs, volume, pattern, risk, ranking)

        btst = is_btst_candidate(trend, momentum, volume, risk)

        # --- Build row ---

        row = {

            # Identity & Verdict
            "Symbol":           symbol,
            "Swing_Score":      ranking.get("Swing_Score"),
            "Verdict":          ranking.get("Verdict"),
            "Setup_Type":       ranking.get("Setup_Type"),
            "Action":           reason.get("Action"),

            # Trend
            "Trend_State":          trend.get("Trend_State"),
            "Trend_Score":          trend.get("Trend_Score"),
            "Price_Zone":           trend.get("Price_Zone"),
            "EMA9":                 trend.get("EMA9"),
            "EMA20":                trend.get("EMA20"),
            "EMA50":                trend.get("EMA50"),
            "EMA200":               trend.get("EMA200"),
            "Bullish_Alignment":    trend.get("Bullish_Alignment"),
            "EMA20_Rising":         trend.get("EMA20_Rising"),
            "EMA50_Rising":         trend.get("EMA50_Rising"),
            "EMA200_Rising":        trend.get("EMA200_Rising"),
            "Golden_Cross_9_20":    trend.get("Golden_Cross_9_20"),
            "Golden_Cross_20_50":   trend.get("Golden_Cross_20_50"),
            "Golden_Cross_50_200":  trend.get("Golden_Cross_50_200"),

            # Pullback
            "Pullback_Setup":           trend.get("Pullback_Setup"),
            "Pullback_To_EMA20":        trend.get("Pullback_To_EMA20"),
            "Pullback_To_EMA50":        trend.get("Pullback_To_EMA50"),
            "RSI_Cooling":              trend.get("RSI_Cooling"),
            "Volume_Drying_Pullback":   trend.get("Volume_Drying_Pullback"),

            # ATH
            "ATH_Risk":                 trend.get("ATH_Risk"),
            "Dist_From_52W_High_Pct":   trend.get("Dist_From_52W_High_Pct"),

            # Momentum
            "Momentum_State":       momentum.get("Momentum_State"),
            "Momentum_Score":       momentum.get("Momentum_Score"),
            "RSI":                  momentum.get("RSI"),
            "MACD":                 momentum.get("MACD"),
            "Histogram":            momentum.get("Histogram"),
            "Histogram_Positive":   momentum.get("Histogram_Positive"),
            "Histogram_Rising":     momentum.get("Histogram_Rising"),
            "Mom_1W":               momentum.get("Mom_1W"),
            "Mom_1M":               momentum.get("Mom_1M"),
            "Mom_3M":               momentum.get("Mom_3M"),
            "Mom_6M":               momentum.get("Mom_6M"),
            "Mom_1Y":               momentum.get("Mom_1Y"),
            "Momentum_Warning":     momentum.get("Momentum_Warning"),

            # RS
            "RS_State":     rs.get("RS_State"),
            "RS_Score":     rs.get("RS_Score"),
            "RS_1M":        rs.get("RS_1M"),
            "RS_3M":        rs.get("RS_3M"),
            "RS_6M":        rs.get("RS_6M"),
            "RS_1Y":        rs.get("RS_1Y"),
            "RS_Warning":   rs.get("RS_Warning"),

            # Volume
            "Volume_State":         volume.get("Volume_State"),
            "Volume_Score":         volume.get("Volume_Score"),
            "Volume_Ratio":         volume.get("Volume_Ratio"),
            "Weekly_Vol_Ratio":     volume.get("Weekly_Vol_Ratio"),
            "Monthly_Vol_Ratio":    volume.get("Monthly_Vol_Ratio"),
            "Weekly_Accumulation":  volume.get("Weekly_Accumulation"),
            "Monthly_Accumulation": volume.get("Monthly_Accumulation"),
            "Breakout_Volume":          volume.get("Breakout_Volume"),
            "Bull_Candle":              volume.get("Bull_Candle"),
            "Close_Near_High":          volume.get("Close_Near_High"),
            "Volume_Warning":           volume.get("Volume_Warning"),
            "Monthly_Trend":            volume.get("Monthly_Trend"),
            "Monthly_12M_Ratio":        volume.get("Monthly_12M_Ratio"),
            "Accum_Distribution":       volume.get("Accum_Distribution"),
            "Accum_Day_Count":          volume.get("Accum_Day_Count"),
            "Dry_Pullback":             volume.get("Dry_Pullback"),

            # Pattern
            "Pattern_State":        pattern.get("Pattern_State"),
            "Pattern_Score":        pattern.get("Pattern_Score"),
            "Primary_Pattern":      pattern.get("Primary_Pattern"),
            "Chart_Context":        pattern.get("Chart_Context"),
            "Weekly_Context":       pattern.get("Weekly_Context"),
            "At_Support":           pattern.get("At_Support"),
            "At_Resistance":        pattern.get("At_Resistance"),
            "Support_Level":        pattern.get("Support_Level"),
            "Resistance_Level":     pattern.get("Resistance_Level"),
            "Buyers_At_Support":    pattern.get("Buyers_At_Support"),
            "Bullish_Engulfing":    pattern.get("Bullish_Engulfing"),
            "Hammer":               pattern.get("Hammer"),
            "Morning_Star":         pattern.get("Morning_Star"),
            "Double_Bottom":        pattern.get("Double_Bottom"),
            "HHHL":                 pattern.get("HHHL"),

            # Risk
            "Trade_Quality":    risk.get("Trade_Quality"),
            "Entry":            risk.get("Entry"),
            "ATR":              risk.get("ATR"),
            "Swing_Low":        risk.get("Swing_Low"),
            "Stop_Loss":        risk.get("Stop_Loss"),
            "Risk_Percent":     risk.get("Risk_Percent"),
            "Target_1":         risk.get("Target_1"),
            "Target_2":         risk.get("Target_2"),
            "Reward_Risk":      risk.get("Reward_Risk"),

            # Score breakdown
            "Recovery_Bonus":   ranking.get("Recovery_Bonus"),
            "Leadership_Bonus": ranking.get("Leadership_Bonus"),
            "Pullback_Bonus":   ranking.get("Pullback_Bonus"),
            "ATH_Penalty":      ranking.get("ATH_Penalty"),
            "Setup_Grade":      ranking.get("Setup_Grade"),
            "Gate_Fail":        ranking.get("Gate_Fail"),

            # BTST
            "BTST": btst,

            # Reason
            "Reason": reason.get("Reason"),

        }

        return row

    except Exception as e:

        print(f"  ERROR processing {symbol}: {e}")
        return None


# ============================================================
# EXCEL WRITER
# ============================================================

def write_excel(all_results):

    try:

        df_all = pd.DataFrame(all_results)

        if df_all.empty:
            print("No results to write.")
            return

        verdict_order = ["STRONG BUY", "BUY", "WATCHLIST", "NEUTRAL", "AVOID"]

        df_all["_vrank"] = df_all["Verdict"].apply(
            lambda v: verdict_order.index(v) if v in verdict_order else 99
        )

        df_all = df_all.sort_values(
            by=["_vrank", "Swing_Score"],
            ascending=[True, False]
        ).drop(columns=["_vrank"]).reset_index(drop=True)

        # --- Sheet filters ---

        df_top = df_all[
            df_all["Verdict"].isin(["STRONG BUY", "BUY"])
        ].reset_index(drop=True)

        df_pullback = df_all[
            df_all["Pullback_Setup"] == True
        ].sort_values("Swing_Score", ascending=False).reset_index(drop=True)

        df_recovery = df_all[
            df_all["Recovery_Bonus"] > 0
        ].sort_values("Swing_Score", ascending=False).reset_index(drop=True)

        df_leaders = df_all[
            (df_all["Setup_Type"] == "LEADER")
            & (df_all["ATH_Risk"] == False)
        ].sort_values("Swing_Score", ascending=False).reset_index(drop=True)

        df_breakout = df_all[
            df_all["Setup_Type"] == "BREAKOUT"
        ].sort_values("Swing_Score", ascending=False).reset_index(drop=True)

        df_support = df_all[
            (df_all["Setup_Type"] == "SUPPORT_BOUNCE") |
            (df_all["Buyers_At_Support"] == True)
        ].sort_values("Swing_Score", ascending=False).reset_index(drop=True)

        df_btst = df_all[
            df_all["BTST"] == True
        ].sort_values("Swing_Score", ascending=False).reset_index(drop=True)

        df_avoid = df_all[
            df_all["Verdict"] == "AVOID"
        ].sort_values("Swing_Score", ascending=False).reset_index(drop=True)

        # --- Display columns for focused sheets ---

        display_cols = [
            "Symbol",
            "Verdict",
            "Swing_Score",
            "Setup_Type",
            "Action",
            "Price_Zone",
            "Trend_State",
            "Momentum_State",
            "RS_State",
            "Volume_State",
            "Primary_Pattern",
            "Chart_Context",
            "Weekly_Context",
            "At_Support",
            "At_Resistance",
            "Support_Level",
            "Resistance_Level",
            "Buyers_At_Support",
            "Monthly_Trend",
            "Monthly_12M_Ratio",
            "Trade_Quality",
            "Entry",
            "Stop_Loss",
            "Target_1",
            "Target_2",
            "Risk_Percent",
            "Reward_Risk",
            "RSI",
            "RS_3M",
            "Mom_3M",
            "Volume_Ratio",
            "Dist_From_52W_High_Pct",
            "ATH_Risk",
            "Pullback_Setup",
            "Accum_Distribution",
            "Accum_Day_Count",
            "Dry_Pullback",
            "Setup_Grade",
            "Gate_Fail",
            "Pullback_Bonus",
            "Recovery_Bonus",
            "ATH_Penalty",
            "Reason",
        ]

        def filter_cols(df, cols):
            available = [c for c in cols if c in df.columns]
            return df[available]

        # --- Write ---

        with pd.ExcelWriter(REPORT_FILE, engine="openpyxl") as writer:

            df_all.drop(columns=["BTST"], errors="ignore").to_excel(
                writer, sheet_name="ALL STOCKS", index=False
            )

            filter_cols(df_top, display_cols).to_excel(
                writer, sheet_name="TOP PICKS", index=False
            )

            filter_cols(df_pullback, display_cols).to_excel(
                writer, sheet_name="PULLBACKS", index=False
            )

            filter_cols(df_recovery, display_cols).to_excel(
                writer, sheet_name="RECOVERY", index=False
            )

            filter_cols(df_leaders, display_cols).to_excel(
                writer, sheet_name="LEADERS", index=False
            )

            filter_cols(df_breakout, display_cols).to_excel(
                writer, sheet_name="BREAKOUTS", index=False
            )

            filter_cols(df_support, display_cols).to_excel(
                writer, sheet_name="SUPPORT BOUNCE", index=False
            )

            filter_cols(df_btst, display_cols).to_excel(
                writer, sheet_name="BTST", index=False
            )

            filter_cols(df_avoid, display_cols).to_excel(
                writer, sheet_name="AVOID", index=False
            )

        print(f"\nReport saved: {REPORT_FILE}")
        print(f"  ALL STOCKS : {len(df_all)}")
        print(f"  TOP PICKS  : {len(df_top)}")
        print(f"  PULLBACKS  : {len(df_pullback)}")
        print(f"  RECOVERY   : {len(df_recovery)}")
        print(f"  LEADERS    : {len(df_leaders)}")
        print(f"  BREAKOUTS  : {len(df_breakout)}")
        print(f"  SUPPORT    : {len(df_support)}")
        print(f"  BTST       : {len(df_btst)}")
        print(f"  AVOID      : {len(df_avoid)}")

    except Exception as e:

        print(f"Excel write error: {e}")


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 55)
    print("  MaverickPICKS v2")
    print(f"  {datetime.now().strftime('%d %b %Y  %H:%M:%S')}")
    print("=" * 55)

    print("\nDownloading NIFTY benchmark...")
    nifty_df = load_nifty()

    if nifty_df is None or nifty_df.empty:
        print("FATAL: Could not load NIFTY data. Exiting.")
        return

    print(f"NIFTY loaded: {len(nifty_df)} sessions\n")

    symbols = load_symbols()

    if not symbols:
        print("FATAL: No symbols found. Exiting.")
        return

    total   = len(symbols)
    results = []
    failed  = []

    print(f"\nScanning {total} stocks...\n")

    for i, symbol in enumerate(symbols, 1):

        print(f"[{i:>3}/{total}] {symbol:<20}", end=" ")

        row = process_stock(symbol, nifty_df)

        if row:
            results.append(row)
            verdict    = row.get("Verdict",    "?")
            score      = row.get("Swing_Score", 0)
            setup      = row.get("Setup_Type",  "")
            ath_flag   = " [ATH]" if row.get("ATH_Risk") else ""
            pullback_f = " [PB]"  if row.get("Pullback_Setup") else ""
            print(f"-> {verdict:<12} Score: {score:<6} {setup}{ath_flag}{pullback_f}")
        else:
            failed.append(symbol)
            print("-> SKIPPED")

        time.sleep(0.3)

    print("\n" + "=" * 55)
    print(f"  Scan complete")
    print(f"  Processed : {len(results)}")
    print(f"  Skipped   : {len(failed)}")
    print("=" * 55)

    if failed:
        print(f"\nSkipped: {', '.join(failed[:20])}")
        if len(failed) > 20:
            print(f"  ... and {len(failed)-20} more")

    if results:
        print("\nGenerating MaverickPICKS_Report.xlsx ...")
        write_excel(results)
    else:
        print("\nNo results to report.")


# ============================================================

if __name__ == "__main__":
    main()

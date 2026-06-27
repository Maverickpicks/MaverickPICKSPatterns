# ============================================================
# RANKING ENGINE V2
#
# HARD RULES — non-negotiable:
#
# 1. Momentum WEAK = max WATCHLIST (stock hasn't turned yet)
# 2. Trade Quality AVOID = max WATCHLIST (risk unacceptable)
# 3. Volume DISTRIBUTION = AVOID immediately
# 4. AT RESISTANCE without buyers = WATCHLIST, not BUY
# 5. ATH risk = max WATCHLIST
# 6. No setup identified = max NEUTRAL
#
# SETUP HIERARCHY (what we actually want to find):
#
# A. SUPPORT BOUNCE  — at tested support, buyers arriving, weekly ok
# B. TREND PULLBACK  — strong trend, price resting at EMA, vol drying
# C. RECOVERY        — prior leader correcting, momentum just turning
# D. BREAKOUT        — breaking out of base on high volume
# E. LEADER          — momentum leader, not at ATH, vol healthy
# ============================================================


def default_rank():
    return {
        "Swing_Score":       0,
        "Verdict":           "AVOID",
        "Setup_Type":        "NONE",
        "Setup_Grade":       "F",
        "Recovery_Bonus":    0,
        "Leadership_Bonus":  0,
        "Pullback_Bonus":    0,
        "ATH_Penalty":       0,
        "Gate_Fail":         "",
    }


def ranking_engine(trend, momentum, rs, volume, pattern, risk):

    try:

        # ============================================================
        # EXTRACT ALL SIGNALS
        # ============================================================

        trend_state    = trend.get("Trend_State",       "WEAK")
        momentum_state = momentum.get("Momentum_State", "WEAK")
        rs_state       = rs.get("RS_State",             "WEAK")
        volume_state   = volume.get("Volume_State",     "DRY")
        pattern_state  = pattern.get("Pattern_State",   "NEUTRAL")
        risk_quality   = risk.get("Trade_Quality",      "AVOID")
        chart_context  = pattern.get("Chart_Context",   "NEUTRAL")

        rsi             = float(momentum.get("RSI",              50) or 50)
        hist_rising     = momentum.get("Histogram_Rising",       False)
        hist_positive   = momentum.get("Histogram_Positive",     False)

        pullback_setup  = trend.get("Pullback_Setup",            False)
        pullback_ema20  = trend.get("Pullback_To_EMA20",         False)
        pullback_ema50  = trend.get("Pullback_To_EMA50",         False)
        vol_drying      = trend.get("Volume_Drying_Pullback",    False)
        rsi_cooling     = trend.get("RSI_Cooling",               False)
        ath_risk        = trend.get("ATH_Risk",                  False)
        price_zone      = trend.get("Price_Zone",                "UNKNOWN")
        dist_ath        = float(trend.get("Dist_From_52W_High_Pct", 100) or 100)

        dry_pullback    = volume.get("Dry_Pullback",             False)
        breakout_vol    = volume.get("Breakout_Volume",          False)
        weekly_accum    = volume.get("Weekly_Accumulation",      False)
        monthly_trend   = volume.get("Monthly_Trend",            "FLAT")
        accum_dist      = volume.get("Accum_Distribution",       None)
        accum_day_cnt   = int(volume.get("Accum_Day_Count",      0))
        strong_bull     = volume.get("Strong_Bull",              False)
        close_near_high = volume.get("Close_Near_High",          False)

        at_support      = pattern.get("At_Support",              False)
        at_resistance   = pattern.get("At_Resistance",           False)
        buyers_support  = pattern.get("Buyers_At_Support",       False)
        weekly_ctx      = pattern.get("Weekly_Context",          "UNKNOWN")
        primary_pattern = pattern.get("Primary_Pattern",         "None")

        rs_1m = float(rs.get("RS_1M", 0) or 0)
        rs_3m = float(rs.get("RS_3M", 0) or 0)
        rs_6m = float(rs.get("RS_6M", 0) or 0)
        rs_1y = float(rs.get("RS_1Y", 0) or 0)

        risk_pct = float(risk.get("Risk_Percent", 99) or 99)
        rr       = float(risk.get("Reward_Risk",   0) or 0)


        # ============================================================
        # HARD BLOCKS — immediate AVOID
        # ============================================================

        if volume_state == "DISTRIBUTION":
            return {**default_rank(),
                    "Verdict":   "AVOID",
                    "Gate_Fail": "Distribution: institutional selling detected"}

        if (trend_state == "WEAK"
                and momentum_state == "WEAK"
                and rs_state == "WEAK"):
            return {**default_rank(),
                    "Verdict":   "AVOID",
                    "Gate_Fail": "All three weak: trend, momentum, RS"}

        if chart_context == "BEARISH":
            return {**default_rank(),
                    "Verdict":   "AVOID",
                    "Gate_Fail": "Weekly chart bearish — no trade"}


        # ============================================================
        # BASE SCORES
        # ============================================================

        trend_score    = int(trend.get("Trend_Score",       5))
        momentum_score = int(momentum.get("Momentum_Score", 5))
        rs_score       = int(rs.get("RS_Score",             5))
        volume_score   = int(volume.get("Volume_Score",     0))
        pattern_score  = int(pattern.get("Pattern_Score",   0))

        risk_score = {"EXCELLENT": 10, "GOOD": 8, "FAIR": 5, "AVOID": 0}.get(risk_quality, 0)

        base_score = (
            trend_score + momentum_score + rs_score
            + volume_score + pattern_score + risk_score
        )


        # ============================================================
        # SETUP DETECTION WITH HARD GATES
        # ============================================================

        setup_type       = "NONE"
        setup_grade      = "F"
        pullback_bonus   = 0
        recovery_bonus   = 0
        leadership_bonus = 0
        ath_penalty      = 0


        # ------------------------------------------------------------
        # SETUP A: SUPPORT BOUNCE
        # This is your "sitting on support with buyers coming in"
        # Hard gates: AT support + bullish pattern/candle + weekly ok
        # Momentum doesn't have to be LEADER — it just needs to not be fully bearish
        # This is where stocks TURN, so we catch early
        # ------------------------------------------------------------

        if (
            (at_support or buyers_support)
            and chart_context in ["STRONG_BULLISH", "REVERSAL_SETUP", "AT_SUPPORT_WATCH"]
            and volume_state != "DISTRIBUTION"
            and trend_state != "WEAK"
            and weekly_ctx not in ["BEARISH"]
        ):
            setup_type = "SUPPORT_BOUNCE"

            # Grade A: buyers confirmed + weekly bullish + good risk
            if (
                buyers_support
                and weekly_ctx in ["BULLISH", "POSITIVE"]
                and risk_quality in ["EXCELLENT", "GOOD"]
                and momentum_state != "WEAK"
            ):
                recovery_bonus = 22
                setup_grade    = "A"

            # Grade B: buyers at support + weekly ok
            elif (
                buyers_support
                and weekly_ctx not in ["BEARISH", "UNKNOWN"]
                and risk_quality in ["EXCELLENT", "GOOD", "FAIR"]
            ):
                recovery_bonus = 16
                setup_grade    = "B"

            # Grade C: at support with bullish pattern, volume drying
            elif (
                at_support
                and (dry_pullback or volume_state == "DRY_PULLBACK")
                and pattern_state in ["BULLISH", "POSITIVE"]
            ):
                recovery_bonus = 11
                setup_grade    = "C"

            # Grade D: at support, watching
            elif at_support and pattern_state in ["BULLISH", "POSITIVE"]:
                recovery_bonus = 6
                setup_grade    = "D"


        # ------------------------------------------------------------
        # SETUP B: TREND PULLBACK
        # Strong trend, price resting at EMA20/50, vol drying
        # Momentum cooling is OK here — that's the point of a pullback
        # But momentum must not be WEAK overall (downtrending)
        # ------------------------------------------------------------

        elif (
            trend_state in ["STRONG", "LEADER"]
            and pullback_setup
            and rs_state not in ["WEAK"]
            and volume_state != "DISTRIBUTION"
            and momentum_state != "WEAK"        # ← KEY FIX: WEAK momentum = not ready
            and weekly_ctx not in ["BEARISH"]
        ):
            setup_type = "TREND_PULLBACK"

            if (
                pullback_ema20
                and (dry_pullback or volume_state == "DRY_PULLBACK")
                and rs_state in ["STRONG", "LEADER"]
                and risk_quality in ["EXCELLENT", "GOOD"]
            ):
                pullback_bonus = 22
                setup_grade    = "A"

            elif (
                pullback_ema20
                and (dry_pullback or volume_state == "DRY_PULLBACK")
            ):
                pullback_bonus = 16
                setup_grade    = "B"

            elif pullback_ema20 and rs_state in ["STRONG", "LEADER"]:
                pullback_bonus = 13
                setup_grade    = "B"

            elif (
                pullback_ema50
                and (dry_pullback or volume_state == "DRY_PULLBACK")
            ):
                pullback_bonus = 12
                setup_grade    = "C"

            elif pullback_ema20:
                pullback_bonus = 9
                setup_grade    = "C"

            elif pullback_ema50:
                pullback_bonus = 6
                setup_grade    = "D"


        # ------------------------------------------------------------
        # SETUP C: RECOVERY
        # Was strong, corrected, momentum just starting to turn
        # Hard gate: RS_6M positive (was a leader) + momentum IMPROVING
        # ---------------------------------------------------------------

        elif (
            rs_6m > 0
            and rs_1m < 0
            and momentum_state == "IMPROVING"
            and volume_state != "DISTRIBUTION"
            and weekly_ctx not in ["BEARISH"]
        ):
            setup_type = "RECOVERY"

            if (
                rs_1y > 5
                and hist_positive
                and volume_state in ["ACCUMULATION", "DRY_PULLBACK", "NORMAL"]
                and risk_quality in ["EXCELLENT", "GOOD", "FAIR"]
            ):
                recovery_bonus = 20
                setup_grade    = "A"

            elif rs_6m > 5 and hist_rising and risk_quality != "AVOID":
                recovery_bonus = 14
                setup_grade    = "B"

            elif rs_6m > 0 and hist_rising:
                recovery_bonus = 9
                setup_grade    = "C"

            else:
                recovery_bonus = 5
                setup_grade    = "D"


        # ------------------------------------------------------------
        # SETUP D: BREAKOUT
        # Breaking out of base on institutional volume
        # Weekly must be supportive — no breakouts in bearish weeks
        # ------------------------------------------------------------

        elif (
            breakout_vol
            and close_near_high
            and (weekly_accum or monthly_trend in ["RISING", "SURGING"])
            and rs_state not in ["WEAK"]
            and weekly_ctx not in ["BEARISH"]
        ):
            setup_type = "BREAKOUT"

            if (
                strong_bull
                and volume_state == "BREAKOUT"
                and trend_state in ["STRONG", "LEADER"]
                and not ath_risk
                and monthly_trend in ["RISING", "SURGING"]
            ):
                pullback_bonus = 20
                setup_grade    = "A"

            elif (
                strong_bull
                and volume_state in ["BREAKOUT", "ACCUMULATION"]
                and not ath_risk
            ):
                pullback_bonus = 14
                setup_grade    = "B"

            else:
                pullback_bonus = 8
                setup_grade    = "C"


        # ------------------------------------------------------------
        # SETUP E: LEADER
        # Ongoing market leader, NOT at ATH, volume healthy
        # Momentum must be STRONG or LEADER — no weak momentum leaders
        # ------------------------------------------------------------

        elif (
            trend_state == "LEADER"
            and rs_state in ["LEADER", "STRONG"]
            and not ath_risk
            and dist_ath >= 5
            and momentum_state in ["LEADER", "STRONG"]     # ← hard gate
            and volume_state in ["ACCUMULATION", "BREAKOUT", "DRY_PULLBACK", "NORMAL"]
            and weekly_ctx not in ["BEARISH"]
        ):
            setup_type = "LEADER"

            if (
                volume_state in ["ACCUMULATION", "BREAKOUT"]
                and monthly_trend in ["RISING", "SURGING"]
            ):
                leadership_bonus = 14
                setup_grade      = "A"

            elif volume_state in ["ACCUMULATION", "BREAKOUT"]:
                leadership_bonus = 10
                setup_grade      = "B"

            else:
                leadership_bonus = 6
                setup_grade      = "C"


        # ============================================================
        # ATH PENALTY — applied regardless of setup
        # ============================================================

        if ath_risk:
            ath_penalty = 18 if rsi > 72 else 12

        elif dist_ath <= 5 and rsi > 68:
            ath_penalty = 8


        # ============================================================
        # FINAL SCORE
        # ============================================================

        final_score = (
            base_score
            + pullback_bonus
            + recovery_bonus
            + leadership_bonus
            - ath_penalty
        )

        final_score = max(0, min(100, final_score))


        # ============================================================
        # VERDICT
        # ============================================================

        if final_score >= 82:
            verdict = "STRONG BUY"
        elif final_score >= 68:
            verdict = "BUY"
        elif final_score >= 52:
            verdict = "WATCHLIST"
        elif final_score >= 38:
            verdict = "NEUTRAL"
        else:
            verdict = "AVOID"


        # ============================================================
        # VERDICT CAPS — hard rules
        # ============================================================

        # Momentum WEAK = not BUY regardless of score
        if momentum_state == "WEAK" and verdict in ["BUY", "STRONG BUY"]:
            verdict = "WATCHLIST"

        # Trade Quality AVOID = not BUY unless pullback setup at support
        if (
            risk_quality == "AVOID"
            and not (pullback_setup or at_support)
            and verdict in ["BUY", "STRONG BUY"]
        ):
            verdict = "WATCHLIST"

        # ATH chaser = max WATCHLIST
        if ath_risk and verdict in ["BUY", "STRONG BUY"]:
            verdict = "WATCHLIST"

        # No setup = max NEUTRAL
        if setup_type == "NONE" and verdict in ["BUY", "STRONG BUY", "WATCHLIST"]:
            verdict = "NEUTRAL"

        # At resistance without buyers = don't buy
        if at_resistance and not buyers_support and verdict in ["BUY", "STRONG BUY"]:
            verdict = "WATCHLIST"


        return {
            "Swing_Score":       round(final_score, 2),
            "Verdict":           verdict,
            "Setup_Type":        setup_type,
            "Setup_Grade":       setup_grade,
            "Recovery_Bonus":    recovery_bonus,
            "Leadership_Bonus":  leadership_bonus,
            "Pullback_Bonus":    pullback_bonus,
            "ATH_Penalty":       ath_penalty,
            "Gate_Fail":         "",
        }

    except Exception as e:
        print("Ranking Engine Error:", e)
        return default_rank()

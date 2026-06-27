# ============================================================
# REASON ENGINE V2
# Generates human-readable, trader-style reasoning
# ============================================================


def get_action(verdict, setup_type):

    if verdict == "STRONG BUY":
        if setup_type == "TREND_PULLBACK":
            return "Buy the Pullback"
        elif setup_type == "RECOVERY":
            return "Accumulate on Dips"
        elif setup_type == "BREAKOUT":
            return "Buy Breakout"
        return "Buy Aggressively"

    elif verdict == "BUY":
        if setup_type == "RECOVERY":
            return "Start Accumulating"
        elif setup_type == "TREND_PULLBACK":
            return "Buy on Dips"
        elif setup_type == "BREAKOUT":
            return "Buy Breakout — Confirm Volume"
        elif setup_type == "SUPPORT_BOUNCE":
            return "Buy at Support"
        return "Buy on Dips"

    elif verdict == "WATCHLIST":
        if setup_type == "RECOVERY":
            return "Watch for Momentum Confirmation"
        elif setup_type == "TREND_PULLBACK":
            return "Wait for Bounce Candle"
        elif setup_type == "BREAKOUT":
            return "Watch for Volume Confirmation"
        return "Wait for Confirmation"

    elif verdict == "NEUTRAL":
        return "Observe"

    return "Ignore"


def generate_reason(trend, momentum, rs, volume, pattern, risk, ranking):

    reasons = []

    verdict    = ranking.get("Verdict",    "AVOID")
    setup_type = ranking.get("Setup_Type", "NONE")
    setup_grade = ranking.get("Setup_Grade", "F")
    gate_fail  = ranking.get("Gate_Fail",  "")
    ath_risk   = trend.get("ATH_Risk",     False)
    dist_ath   = trend.get("Dist_From_52W_High_Pct", None)
    price_zone = trend.get("Price_Zone",   "UNKNOWN")

    pullback_setup = trend.get("Pullback_Setup", False)
    pullback_ema20 = trend.get("Pullback_To_EMA20", False)
    pullback_ema50 = trend.get("Pullback_To_EMA50", False)
    dry_pullback   = volume.get("Dry_Pullback", False)
    rsi_cooling    = trend.get("RSI_Cooling", False)

    trend_state    = trend.get("Trend_State",    "WEAK")
    momentum_state = momentum.get("Momentum_State", "WEAK")
    rs_state       = rs.get("RS_State",          "WEAK")
    volume_state   = volume.get("Volume_State",  "DRY")
    rsi            = momentum.get("RSI",         None)
    rs_1m          = rs.get("RS_1M", None)
    rs_3m          = rs.get("RS_3M", None)
    rs_6m          = rs.get("RS_6M", None)
    accum_dist     = volume.get("Accum_Distribution", None)
    risk_quality   = risk.get("Trade_Quality",   "AVOID")
    rr             = risk.get("Reward_Risk",     None)
    risk_pct       = risk.get("Risk_Percent",    None)


    # ============================================================
    # GATE FAIL — short circuit
    # ============================================================

    if gate_fail:
        action = get_action(verdict, setup_type)
        return {
            "Reason": f"BLOCKED: {gate_fail}",
            "Action": "Ignore"
        }


    # ============================================================
    # SETUP HEADLINE — lead with the trade thesis
    # ============================================================

    if setup_type == "TREND_PULLBACK":
        ema_level = "EMA20" if pullback_ema20 else "EMA50"
        reasons.append(f"[Grade {setup_grade}] Pullback to {ema_level} in strong uptrend — potential re-entry")

    elif setup_type == "RECOVERY":
        reasons.append(f"[Grade {setup_grade}] Recovery setup — prior leader correcting, now turning up")

    elif setup_type == "BREAKOUT":
        reasons.append(f"[Grade {setup_grade}] Breaking out with institutional volume confirmation")

    elif setup_type == "SUPPORT_BOUNCE":
        reasons.append(f"[Grade {setup_grade}] Bouncing from key support level")

    elif setup_type == "LEADER":
        reasons.append(f"[Grade {setup_grade}] Market leader — trend intact, not overextended")

    elif setup_type == "ATH_CHASER":
        reasons.append("WARNING: Stock at all-time high — chasing risk, avoid fresh entry")

    else:
        reasons.append("No clear swing setup identified")


    # ============================================================
    # TREND
    # ============================================================

    ema20 = trend.get("EMA20")
    ema50 = trend.get("EMA50")

    if trend_state == "LEADER":
        reasons.append(f"Trend: Full EMA stack aligned bullishly")
    elif trend_state == "STRONG":
        reasons.append(f"Trend: Strong (EMA9>20>50)")
    elif trend_state == "NEUTRAL":
        if pullback_setup:
            reasons.append(f"Trend: Intact at higher timeframe, price resting near EMA{' 20 (' + str(ema20) + ')' if pullback_ema20 else ' 50 (' + str(ema50) + ')'}")
        else:
            reasons.append("Trend: Neutral — stabilizing")
    else:
        reasons.append("Trend: Weak")


    # ============================================================
    # PULLBACK QUALITY
    # ============================================================

    if pullback_setup:
        cues = []
        if dry_pullback:
            cues.append("volume drying — sellers absent")
        if rsi_cooling:
            cues.append(f"RSI cooling ({rsi}) — not oversold")
        if cues:
            reasons.append("Pullback quality: " + ", ".join(cues))


    # ============================================================
    # MOMENTUM
    # ============================================================

    hist_positive = momentum.get("Histogram_Positive", False)
    hist_rising   = momentum.get("Histogram_Rising",   False)
    macd_val      = momentum.get("MACD",   None)
    signal_val    = momentum.get("Signal", None)

    if momentum_state == "LEADER":
        reasons.append(f"Momentum: Strong — RSI {rsi}, MACD above signal")
    elif momentum_state == "STRONG":
        reasons.append(f"Momentum: Positive — RSI {rsi}, MACD {macd_val}")
    elif momentum_state == "IMPROVING":
        if hist_positive:
            reasons.append(f"Momentum: Turning up — histogram positive, RSI {rsi}")
        elif hist_rising:
            reasons.append(f"Momentum: Early recovery — histogram rising, RSI {rsi}")
        else:
            reasons.append(f"Momentum: Improving — RSI {rsi}")
    else:
        reasons.append(f"Momentum: Weak — RSI {rsi}")


    # ============================================================
    # RELATIVE STRENGTH
    # ============================================================

    if rs_state == "LEADER":
        reasons.append(f"RS: Market leader — outperforming NIFTY across all periods")
    elif rs_state == "STRONG":
        rs_str = f"1M: {rs_1m:+.1f}%, 3M: {rs_3m:+.1f}%" if rs_1m and rs_3m else ""
        reasons.append(f"RS: Outperforming NIFTY {rs_str}")
    elif rs_state == "RECOVERING":
        reasons.append(f"RS: Recovering — short term weak but 6M: {rs_6m:+.1f}% vs NIFTY still positive")
    else:
        reasons.append(f"RS: Underperforming NIFTY (3M: {rs_3m:+.1f}%)" if rs_3m else "RS: Underperforming NIFTY")


    # ============================================================
    # VOLUME / INSTITUTIONAL ACTIVITY
    # ============================================================

    vol_ratio = volume.get("Volume_Ratio", None)
    accum_day = volume.get("Accum_Day_Count", 0)

    if volume_state == "BREAKOUT":
        reasons.append(f"Volume: Breakout ({vol_ratio}x avg) — institutional commitment visible")
    elif volume_state == "ACCUMULATION":
        reasons.append(f"Volume: Accumulation — up-day vol {accum_dist:.1f}x down-day vol, {accum_day} strong up days in 20 sessions" if accum_dist else "Volume: Accumulation pattern detected")
    elif volume_state == "DRY_PULLBACK":
        reasons.append("Volume: Drying on pullback — institutions holding, not selling")
    elif volume_state == "DISTRIBUTION":
        reasons.append("Volume: DISTRIBUTION — institutional selling, avoid")
    elif volume_state == "NORMAL":
        reasons.append(f"Volume: Normal ({vol_ratio}x avg) — no institutional signal yet")
    else:
        reasons.append(f"Volume: Low participation ({vol_ratio}x avg)" if vol_ratio else "Volume: Low participation")


    # ============================================================
    # PATTERN
    # ============================================================

    primary_pattern = pattern.get("Primary_Pattern", "None")
    pattern_state_v = pattern.get("Pattern_State",   "NEUTRAL")

    if primary_pattern and primary_pattern != "None":
        reasons.append(f"Pattern: {primary_pattern} ({pattern_state_v})")


    # ============================================================
    # ATH WARNING
    # ============================================================

    if ath_risk:
        reasons.append(f"CAUTION: {dist_ath}% from 52W high, RSI {rsi} — getting stuck at top risk")
    elif dist_ath is not None and dist_ath <= 8:
        reasons.append(f"Note: Only {dist_ath}% below 52W high — limited upside headroom")


    # ============================================================
    # RISK
    # ============================================================

    sl  = risk.get("Stop_Loss")
    t1  = risk.get("Target_1")
    t2  = risk.get("Target_2")

    if risk_quality == "EXCELLENT":
        reasons.append(f"Risk: Excellent R:R {rr}x | Risk {risk_pct}% | SL {sl} | T1 {t1} | T2 {t2}")
    elif risk_quality == "GOOD":
        reasons.append(f"Risk: Good R:R {rr}x | Risk {risk_pct}% | SL {sl} | T1 {t1}")
    elif risk_quality == "FAIR":
        reasons.append(f"Risk: Fair R:R {rr}x | Risk {risk_pct}% | SL {sl}")
    else:
        reasons.append(f"Risk: Unattractive R:R {rr}x | Risk {risk_pct}% — size small or wait")


    # ============================================================
    # SCORE TRANSPARENCY
    # ============================================================

    pb_bonus  = ranking.get("Pullback_Bonus",  0)
    rec_bonus = ranking.get("Recovery_Bonus",  0)
    ld_bonus  = ranking.get("Leadership_Bonus", 0)
    ath_pen   = ranking.get("ATH_Penalty",     0)

    tags = []
    if pb_bonus  > 0: tags.append(f"+{pb_bonus} pullback")
    if rec_bonus > 0: tags.append(f"+{rec_bonus} recovery")
    if ld_bonus  > 0: tags.append(f"+{ld_bonus} leader")
    if ath_pen   > 0: tags.append(f"-{ath_pen} ATH")
    if tags:
        reasons.append("Score adj: " + ", ".join(tags))


    action = get_action(verdict, setup_type)

    return {
        "Reason": " | ".join(reasons),
        "Action": action
    }

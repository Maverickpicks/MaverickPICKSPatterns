# ======================================

# ACTION GENERATOR

# ======================================

def get_action(verdict):

    if verdict=="STRONG BUY":

        return "Buy Aggressively"


    elif verdict=="BUY":

        return "Buy on Dips"


    elif verdict=="WATCHLIST":

        return "Wait for Confirmation"


    elif verdict=="NEUTRAL":

        return "Observe"


    else:

        return "Ignore"




# ======================================

# REASON GENERATOR

# ======================================

def generate_reason(

        trend,

        momentum,

        rs,

        volume,

        pattern,

        risk,

        ranking

):



    reasons=[]



    # ===============================

    # Trend

    # ===============================



    trend_state=trend.get(

        "Trend_State",

        "WEAK"

    )



    if trend_state=="LEADER":


        reasons.append(

            "Exceptional trend strength"

        )



    elif trend_state=="STRONG":


        reasons.append(

            "Strong EMA alignment"

        )



    elif trend_state=="NEUTRAL":


        reasons.append(

            "Trend stabilizing after correction"

        )



    else:


        reasons.append(

            "Trend remains weak"

        )



    # ===============================

    # Momentum

    # ===============================



    momentum_state=momentum.get(

        "Momentum_State",

        "WEAK"

    )



    if momentum_state=="LEADER":


        reasons.append(

            "Momentum leader"

        )



    elif momentum_state=="STRONG":


        reasons.append(

            "Positive momentum"

        )



    elif momentum_state=="IMPROVING":


        reasons.append(

            "Momentum improving"

        )



    else:


        reasons.append(

            "Momentum weak"

        )



    # ===============================

    # Relative Strength

    # ===============================



    rs_state=rs.get(

        "RS_State",

        "WEAK"

    )



    if rs_state=="LEADER":


        reasons.append(

            "Market leader"

        )



    elif rs_state=="STRONG":


        reasons.append(

            "Outperforming NIFTY"

        )



    elif rs_state=="RECOVERING":


        reasons.append(

            "Healthy correction with strong long term RS"

        )



    else:


        reasons.append(

            "Underperforming market"

        )



    # ===============================

    # Volume

    # ===============================



    volume_state=volume.get(

        "Volume_State",

        "DRY"

    )



    if volume_state=="BREAKOUT":


        reasons.append(

            "Breakout with strong volume"

        )



    elif volume_state=="ACCUMULATION":


        reasons.append(

            "Institutional accumulation"

        )



    elif volume_state=="DRY":


        reasons.append(

            "Volume participation weak"

        )



    # ===============================

    # Pattern

    # ===============================



    pattern_name=pattern.get(

        "Primary_Pattern",

        ""

    )



    if pattern_name:



        reasons.append(

            f"Pattern: {pattern_name}"

        )



    # ===============================

    # Risk

    # ===============================



    trade_quality=risk.get(

        "Trade_Quality",

        "AVOID"

    )



    if trade_quality=="EXCELLENT":


        reasons.append(

            "Excellent risk reward"

        )



    elif trade_quality=="GOOD":


        reasons.append(

            "Risk reward favourable"

        )



    elif trade_quality=="FAIR":


        reasons.append(

            "Moderate risk reward"

        )



    else:


        reasons.append(

            "Risk reward unattractive"

        )



    # ===============================

    # Verdict

    # ===============================



    verdict=ranking.get(

        "Verdict",

        "AVOID"

    )



    action=get_action(

        verdict

    )



    reason_text=" | ".join(

        reasons

    )



    return {



        "Reason":

        reason_text,



        "Action":

        action

    }
# ============================================
# Ranking Engine
# ============================================

def ranking_engine(

        trend,

        momentum,

        rs,

        volume,

        pattern,

        risk

):


    try:


        final_score=0


        # =================================

        # TREND

        # =================================

        trend_score=min(

            trend.get(

                "trend_score",

                0

            ),

            25

        )



        # =================================

        # MOMENTUM

        # =================================

        momentum_raw=min(

            momentum.get(

                "momentum_score",

                0

            ),

            20

        )



        # =================================

        # RS

        # =================================

        rs_raw=min(

            rs.get(

                "RS_Score",

                0

            ),

            10

        )



        rs_score=rs_raw*2



        # =================================

        # VOLUME

        # =================================

        volume_raw=volume.get(

            "Volume_Score",

            0

        )



        if volume_raw>15:

            volume_raw=15



        # =================================

        # PATTERN

        # =================================

        pattern_raw=pattern.get(

            "Pattern_Score",

            0

        )



        if pattern_raw>10:

            pattern_raw=10



        # =================================

        # RISK

        # =================================

        trade_quality=risk.get(

            "Trade_Quality",

            "AVOID"

        )



        risk_score=0



        if trade_quality=="EXCELLENT":

            risk_score=10



        elif trade_quality=="GOOD":

            risk_score=8



        elif trade_quality=="FAIR":

            risk_score=5



        else:

            risk_score=0



        # =================================

        # TOTAL

        # =================================



        final_score=(

            trend_score

            +

            momentum_raw

            +

            rs_score

            +

            volume_raw

            +

            pattern_raw

            +

            risk_score

        )



        if final_score>100:

            final_score=100



        # =================================

        # Verdict

        # =================================



        verdict="AVOID"



        if final_score>=90:

            verdict="STRONG BUY"



        elif final_score>=75:

            verdict="BUY"



        elif final_score>=60:

            verdict="WATCHLIST"



        elif final_score>=40:

            verdict="NEUTRAL"



        else:

            verdict="AVOID"



        # =================================

        # Risk Override

        # =================================



        if (

            trade_quality=="AVOID"

            and

            verdict in [

                "BUY",

                "STRONG BUY"

            ]

        ):



            verdict="WATCHLIST"



        # =================================



        return {



            "Swing_Score":

            round(

                final_score,

                2

            ),



            "Verdict":

            verdict

        }



    except Exception as e:



        print(

            "Ranking Engine Error:",

            e

        )



        return {



            "Swing_Score":0,

            "Verdict":"AVOID"

        }
import pandas as pd
import pandas_ta as ta

from utils import last_value
from utils import nth_last


# ============================================
# Default Return
# ============================================

def default_trend():

    return {

        "trend_score":0,

        "ema9":None,

        "ema20":None,

        "ema50":None,

        "ema100":None,

        "ema200":None,

        "bullish_alignment":False,

        "ema20_rising":False,

        "ema50_rising":False,

        "ema200_rising":False,

        "ema_gap_expanding":False,

        "golden_cross_9_20":False,

        "golden_cross_20_50":False,

        "golden_cross_50_200":False

    }



# ============================================
# Trend Analysis
# ============================================

def trend_analysis(df):

    try:

        if df is None:

            return default_trend()


        if df.empty:

            return default_trend()


        if len(df)<220:

            return default_trend()



        # EMA


        df["EMA9"]=ta.ema(

            df["Close"],

            length=9

        )



        df["EMA20"]=ta.ema(

            df["Close"],

            length=20

        )



        df["EMA50"]=ta.ema(

            df["Close"],

            length=50

        )



        df["EMA100"]=ta.ema(

            df["Close"],

            length=100

        )



        df["EMA200"]=ta.ema(

            df["Close"],

            length=200

        )



        ema9=last_value(

            df["EMA9"]

        )


        ema20=last_value(

            df["EMA20"]

        )


        ema50=last_value(

            df["EMA50"]

        )


        ema100=last_value(

            df["EMA100"]

        )


        ema200=last_value(

            df["EMA200"]

        )



        if pd.isna(ema9):

            return default_trend()



        # --------------------------------

        bullish_alignment=all([

            ema9>ema20,

            ema20>ema50,

            ema50>ema100,

            ema100>ema200

        ])



        # --------------------------------

        ema20_prev=nth_last(

            df["EMA20"],

            6

        )


        ema50_prev=nth_last(

            df["EMA50"],

            6

        )


        ema200_prev=nth_last(

            df["EMA200"],

            6

        )



        ema20_rising=False

        ema50_rising=False

        ema200_rising=False



        if not pd.isna(ema20_prev):

            ema20_rising=ema20>ema20_prev



        if not pd.isna(ema50_prev):

            ema50_rising=ema50>ema50_prev



        if not pd.isna(ema200_prev):

            ema200_rising=ema200>ema200_prev



        # --------------------------------

        gap_today=ema9-ema20



        gap_prev=(

            nth_last(

                df["EMA9"],

                6

            )

            -

            nth_last(

                df["EMA20"],

                6

            )

        )



        ema_gap_expanding=False



        if not pd.isna(gap_prev):


            ema_gap_expanding=(

                gap_today

                >

                gap_prev

            )



        # --------------------------------

        golden_cross_9_20=False

        golden_cross_20_50=False

        golden_cross_50_200=False



        if (

            df["EMA9"].iloc[-2]

            <=

            df["EMA20"].iloc[-2]

        ):


            if (

                ema9

                >

                ema20

            ):


                golden_cross_9_20=True




        if (

            df["EMA20"].iloc[-2]

            <=

            df["EMA50"].iloc[-2]

        ):


            if (

                ema20

                >

                ema50

            ):


                golden_cross_20_50=True




        if (

            df["EMA50"].iloc[-2]

            <=

            df["EMA200"].iloc[-2]

        ):


            if (

                ema50

                >

                ema200

            ):


                golden_cross_50_200=True



        # --------------------------------

        score=0



        if bullish_alignment:

            score+=20



        if ema_gap_expanding:

            score+=5



        if ema20_rising:

            score+=2



        if ema50_rising:

            score+=2



        if ema200_rising:

            score+=1



        # --------------------------------

        return {


            "trend_score":score,


            "ema9":round(

                ema9,

                2

            ),


            "ema20":round(

                ema20,

                2

            ),


            "ema50":round(

                ema50,

                2

            ),


            "ema100":round(

                ema100,

                2

            ),


            "ema200":round(

                ema200,

                2

            ),


            "bullish_alignment":

            bullish_alignment,


            "ema20_rising":

            ema20_rising,


            "ema50_rising":

            ema50_rising,


            "ema200_rising":

            ema200_rising,


            "ema_gap_expanding":

            ema_gap_expanding,


            "golden_cross_9_20":

            golden_cross_9_20,


            "golden_cross_20_50":

            golden_cross_20_50,


            "golden_cross_50_200":

            golden_cross_50_200

        }



    except Exception as e:


        print(


            "Trend Engine Error:",


            e


        )


        return default_trend()
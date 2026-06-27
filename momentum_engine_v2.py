import pandas as pd

import pandas_ta as ta

from utils import last_value


# ======================================

# DEFAULT

# ======================================

def default_momentum():

    return {

        "Momentum_State":"WEAK",

        "Momentum_Score":5,

        "RSI":None,

        "RSI_Rising":False,

        "RSI_Cross_50":False,

        "RSI_Cross_60":False,

        "RSI_Cross_70":False,

        "MACD":None,

        "Signal":None,

        "Histogram":None,

        "Histogram_Rising":False,

        "Histogram_Positive":False,

        "Mom_1W":None,

        "Mom_1M":None,

        "Mom_3M":None,

        "Mom_6M":None,

        "Mom_1Y":None,

        "Momentum_Warning":"Momentum Missing"

    }




# ======================================

# MOMENTUM ANALYSIS

# ======================================

def momentum_analysis(df):



    try:



        if df is None:

            return default_momentum()



        if df.empty:

            return default_momentum()



        if len(df)<260:

            return default_momentum()



        # ================================

        # RSI

        # ================================



        df["RSI"]=ta.rsi(

            df["Close"],

            length=14

        )



        rsi=last_value(

            df["RSI"]

        )



        rsi_prev=df["RSI"].iloc[-5]



        rsi_rising=(

            rsi

            >

            rsi_prev

        )



        rsi_cross_50=(

            rsi>50

        )



        rsi_cross_60=(

            rsi>60

        )



        rsi_cross_70=(

            rsi>70

        )



        # ================================

        # MACD

        # ================================



        macd=ta.macd(

            df["Close"]

        )



        df["MACD"]=macd["MACD_12_26_9"]

        df["Signal"]=macd["MACDs_12_26_9"]

        df["Histogram"]=macd["MACDh_12_26_9"]



        macd_val=last_value(

            df["MACD"]

        )



        signal=last_value(

            df["Signal"]

        )



        hist=last_value(

            df["Histogram"]

        )



        hist_prev=df["Histogram"].iloc[-5]



        histogram_rising=(

            hist

            >

            hist_prev

        )



        histogram_positive=(

            hist

            >

            0

        )



        # ================================

        # MOMENTUM RETURNS

        # ================================



        close=last_value(

            df["Close"]

        )



        mom_1w=(

            (

                close

                -

                df["Close"].iloc[-5]

            )

            /

            df["Close"].iloc[-5]

        )*100



        mom_1m=(

            (

                close

                -

                df["Close"].iloc[-21]

            )

            /

            df["Close"].iloc[-21]

        )*100



        mom_3m=(

            (

                close

                -

                df["Close"].iloc[-63]

            )

            /

            df["Close"].iloc[-63]

        )*100



        mom_6m=(

            (

                close

                -

                df["Close"].iloc[-126]

            )

            /

            df["Close"].iloc[-126]

        )*100



        mom_1y=(

            (

                close

                -

                df["Close"].iloc[-252]

            )

            /

            df["Close"].iloc[-252]

        )*100



        # ================================

        # STATE

        # ================================



        state="WEAK"



        score=5



        warning="Momentum Missing"



        # --------------------------------

        # LEADER

        # --------------------------------



        if (

            rsi>65

            and

            histogram_positive

            and

            mom_1m>0

            and

            mom_3m>0

        ):



            state="LEADER"



            score=20



            warning=""



        # --------------------------------

        # STRONG

        # --------------------------------



        elif (

            rsi>55

            and

            macd_val>signal

            and

            mom_1m>0

        ):



            state="STRONG"



            score=15



            warning=""



        # --------------------------------

        # IMPROVING

        # --------------------------------



        elif (

            rsi_rising

            and

            histogram_rising

        ):



            state="IMPROVING"



            score=10



            warning="Recovery Momentum"



        # =================================



        return {



            "Momentum_State":

            state,



            "Momentum_Score":

            score,



            "RSI":

            round(

                rsi,

                2

            ),



            "RSI_Rising":

            bool(

                rsi_rising

            ),



            "RSI_Cross_50":

            bool(

                rsi_cross_50

            ),



            "RSI_Cross_60":

            bool(

                rsi_cross_60

            ),



            "RSI_Cross_70":

            bool(

                rsi_cross_70

            ),



            "MACD":

            round(

                macd_val,

                2

            ),



            "Signal":

            round(

                signal,

                2

            ),



            "Histogram":

            round(

                hist,

                2

            ),



            "Histogram_Rising":

            bool(

                histogram_rising

            ),



            "Histogram_Positive":

            bool(

                histogram_positive

            ),



            "Mom_1W":

            round(

                float(


                mom_1w
                ),

                2

            ),



            "Mom_1M":

            round(

                float(

                mom_1m

                ),

                2

            ),



            "Mom_3M":

            round(

                float(

                mom_3m

                ),

                2

            ),



            "Mom_6M":

            round(

                float(

                mom_6m

                ),

                2

            ),



            "Mom_1Y":

            round(

                float(mom_1y),

                2

            ),



            "Momentum_Warning":

            warning

        }



    except Exception as e:



        print(

            "Momentum Engine Error:",

            e

        )



        return default_momentum()
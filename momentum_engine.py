import pandas as pd
import pandas_ta as ta

from utils import last_value
from utils import nth_last
from utils import pct_change


# ============================================
# Default Return
# ============================================

def default_momentum():

    return {

        "momentum_score":0,

        "RSI":None,

        "RSI_Rising":False,

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

        "Momentum_State":"WEAK",

        "Momentum_Warning":""

    }



# ============================================
# Momentum Analysis
# ============================================

def momentum_analysis(df):


    try:


        if df is None:

            return default_momentum()


        if df.empty:

            return default_momentum()


        if len(df)<260:

            return default_momentum()



        # RSI

        df["RSI"]=ta.rsi(

            df["Close"],

            length=14

        )



        # MACD

        macd=ta.macd(

            df["Close"]

        )



        df["MACD"]=macd.iloc[:,0]

        df["Signal"]=macd.iloc[:,1]

        df["Histogram"]=macd.iloc[:,2]



        # -----------------------------

        rsi=last_value(

            df["RSI"]

        )


        rsi_prev=nth_last(

            df["RSI"],

            6

        )


        rsi_yesterday=nth_last(

            df["RSI"],

            2

        )



        rsi_rising=False


        if (

            not pd.isna(rsi)

            and

            not pd.isna(rsi_prev)

        ):



            rsi_rising=(

                rsi

                >

                rsi_prev

            )



        rsi_cross_60=False



        if (

            not pd.isna(rsi)

            and

            not pd.isna(rsi_yesterday)

        ):



            rsi_cross_60=(

                rsi_yesterday<=60

                and

                rsi>60

            )



        rsi_cross_70=False



        if (

            not pd.isna(rsi)

            and

            not pd.isna(rsi_yesterday)

        ):



            rsi_cross_70=(

                rsi_yesterday<=70

                and

                rsi>70

            )



        # -----------------------------

        macd_val=last_value(

            df["MACD"]

        )


        signal=last_value(

            df["Signal"]

        )


        hist=last_value(

            df["Histogram"]

        )


        hist_prev=nth_last(

            df["Histogram"],

            6

        )



        hist_rising=False



        if (

            not pd.isna(hist)

            and

            not pd.isna(hist_prev)

        ):



            hist_rising=(

                hist

                >

                hist_prev

            )



        hist_positive=False



        if (

            not pd.isna(hist)

        ):



            hist_positive=(

                hist

                >

                0

            )



        # -----------------------------

        close=last_value(

            df["Close"]

        )



        mom_1w=pct_change(

            close,

            nth_last(

                df["Close"],

                6

            )

        )



        mom_1m=pct_change(

            close,

            nth_last(

                df["Close"],

                21

            )

        )



        mom_3m=pct_change(

            close,

            nth_last(

                df["Close"],

                63

            )

        )



        mom_6m=pct_change(

            close,

            nth_last(

                df["Close"],

                126

            )

        )



        mom_1y=pct_change(

            close,

            nth_last(

                df["Close"],

                252

            )

        )



        # =================================

        # SCORING

        # =================================



        score=0



        # RSI



        if rsi is not None:



            if rsi>70:

                score+=6



            elif rsi>60:

                score+=5



            elif rsi>55:

                score+=3



        if rsi_rising:

            score+=2



        # MACD



        if (

            macd_val is not None

            and

            signal is not None

        ):



            if macd_val>signal:

                score+=3



        if hist_positive:

            score+=2



        if hist_rising:

            score+=2



        # MOMENTUM



        if (

            mom_1m is not None

            and

            mom_1m>0

        ):



            score+=2



        if (

            mom_3m is not None

            and

            mom_3m>0

        ):



            score+=3



        if (

            mom_6m is not None

            and

            mom_6m>0

        ):



            score+=2



        if (

            mom_1y is not None

            and

            mom_1y>0

        ):



            score+=1



        # =================================

        # MOMENTUM STATE

        # =================================



        momentum_state="WEAK"



        if (

            rsi is not None

            and

            rsi>60

            and

            macd_val is not None

            and

            signal is not None

            and

            macd_val>signal

            and

            hist_rising

            and

            mom_3m is not None

            and

            mom_3m>0

        ):



            momentum_state="STRONG"



        if (

            rsi is not None

            and

            rsi>70

            and

            macd_val>signal

            and

            hist_rising

            and

            mom_1m is not None

            and

            mom_1m>10

            and

            mom_3m is not None

            and

            mom_3m>20

        ):



            momentum_state="EXTREME"



        if (

            momentum_state=="WEAK"

            and

            rsi is not None

            and

            rsi>=50

        ):



            momentum_state="NEUTRAL"



        # =================================

        # WARNING

        # =================================



        warning=""


        if (

            momentum_state=="EXTREME"

        ):



            warning="Overheated"



        elif (

            momentum_state=="WEAK"

        ):



            warning="Momentum Missing"



        # =================================



        return {



            "momentum_score":score,


            "RSI":

            round(rsi,2)

            if rsi is not None

            else None,


            "RSI_Rising":

            rsi_rising,


            "RSI_Cross_60":

            rsi_cross_60,


            "RSI_Cross_70":

            rsi_cross_70,


            "MACD":

            round(macd_val,2)

            if macd_val is not None

            else None,


            "Signal":

            round(signal,2)

            if signal is not None

            else None,


            "Histogram":

            round(hist,2)

            if hist is not None

            else None,


            "Histogram_Rising":

            hist_rising,


            "Histogram_Positive":

            hist_positive,


            "Mom_1W":

            mom_1w,


            "Mom_1M":

            mom_1m,


            "Mom_3M":

            mom_3m,


            "Mom_6M":

            mom_6m,


            "Mom_1Y":

            mom_1y,


            "Momentum_State":

            momentum_state,


            "Momentum_Warning":

            warning

        }



    except Exception as e:



        print(

            "Momentum Engine Error:",

            e

        )



        return default_momentum()
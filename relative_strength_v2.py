import pandas as pd

from utils import last_value


# =====================================

# DEFAULT

# =====================================

def default_rs():

    return {

        "RS_1M":None,

        "RS_3M":None,

        "RS_6M":None,

        "RS_1Y":None,

        "RS_State":"WEAK",

        "RS_Score":5,

        "RS_Warning":"Underperforming"

    }



# =====================================

# RETURN %

# =====================================

def pct_change(

        latest,

        old

):


    return (

        (

            latest

            -

            old

        )

        /

        old

    )*100



# =====================================

# RS ANALYSIS

# =====================================

def relative_strength_analysis(

        stock_df,

        nifty_df

):



    try:



        if stock_df is None:

            return default_rs()



        if nifty_df is None:

            return default_rs()



        if len(stock_df)<260:

            return default_rs()



        if len(nifty_df)<260:

            return default_rs()



        # ==============================

        # STOCK RETURNS

        # ==============================



        stock_close=last_value(

            stock_df["Close"]

        )



        nifty_close=last_value(

            nifty_df["Close"]

        )



        stock_1m=pct_change(

            stock_close,

            stock_df["Close"].iloc[-21]

        )



        nifty_1m=pct_change(

            nifty_close,

            nifty_df["Close"].iloc[-21]

        )



        stock_3m=pct_change(

            stock_close,

            stock_df["Close"].iloc[-63]

        )



        nifty_3m=pct_change(

            nifty_close,

            nifty_df["Close"].iloc[-63]

        )



        stock_6m=pct_change(

            stock_close,

            stock_df["Close"].iloc[-126]

        )



        nifty_6m=pct_change(

            nifty_close,

            nifty_df["Close"].iloc[-126]

        )



        stock_1y=pct_change(

            stock_close,

            stock_df["Close"].iloc[-252]

        )



        nifty_1y=pct_change(

            nifty_close,

            nifty_df["Close"].iloc[-252]

        )



        # ==============================

        # RS

        # ==============================



        rs_1m=stock_1m-nifty_1m

        rs_3m=stock_3m-nifty_3m

        rs_6m=stock_6m-nifty_6m

        rs_1y=stock_1y-nifty_1y



        # ==============================

        # STATE

        # ==============================



        state="WEAK"

        score=5

        warning="Underperforming"



        # ------------------------------

        # LEADER

        # ------------------------------



        if (

            rs_1m>10

            and

            rs_3m>15

            and

            rs_6m>20

        ):



            state="LEADER"



            score=20



            warning="Market Leader"



        # ------------------------------

        # STRONG

        # ------------------------------



        elif (

            rs_1m>0

            and

            rs_3m>0

            and

            rs_6m>0

        ):



            state="STRONG"



            score=15



            warning="Outperforming Market"



        # ------------------------------

        # RECOVERING

        # ------------------------------



        elif (

            rs_1m<0

            and

            rs_3m<0

            and

            rs_6m>0

            and

            rs_1y>0

        ):



            state="RECOVERING"



            score=10



            warning="Healthy Correction"



        # =================================



        return {



            "RS_1M":

            round(

                float(

                    rs_1m

                ),

                2

            ),



            "RS_3M":

            round(

                float(

                    rs_3m

                ),

                2

            ),



            "RS_6M":

            round(

                float(

                    rs_6m

                ),

                2

            ),



            "RS_1Y":

            round(

                float(

                    rs_1y

                ),

                2

            ),



            "RS_State":

            state,



            "RS_Score":

            score,



            "RS_Warning":

            warning

        }



    except Exception as e:



        print(

            "RS Engine Error:",

            e

        )



        return default_rs()
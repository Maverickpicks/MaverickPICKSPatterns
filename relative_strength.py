import pandas as pd

from utils import last_value
from utils import nth_last
from utils import pct_change


# ============================================
# Default Return
# ============================================

def default_rs():

    return {

        "RS_1M":None,

        "RS_3M":None,

        "RS_6M":None,

        "RS_1Y":None,

        "RS_State":"WEAK",

        "RS_Score":0

    }



# ============================================
# Relative Strength Analysis
# ============================================

def relative_strength_analysis(

        stock_df,

        nifty_df

):


    try:


        if stock_df is None:

            return default_rs()


        if nifty_df is None:

            return default_rs()


        if stock_df.empty:

            return default_rs()


        if nifty_df.empty:

            return default_rs()



        # -------------------------

        stock_close=last_value(

            stock_df["Close"]

        )


        nifty_close=last_value(

            nifty_df["Close"]

        )



        # -------------------------

        stock_1m=pct_change(

            stock_close,

            nth_last(

                stock_df["Close"],

                21

            )

        )



        nifty_1m=pct_change(

            nifty_close,

            nth_last(

                nifty_df["Close"],

                21

            )

        )



        rs_1m=None


        if (

            stock_1m is not None

            and

            nifty_1m is not None

        ):



            rs_1m=round(

                stock_1m

                -

                nifty_1m,

                2

            )



        # -------------------------

        stock_3m=pct_change(

            stock_close,

            nth_last(

                stock_df["Close"],

                63

            )

        )



        nifty_3m=pct_change(

            nifty_close,

            nth_last(

                nifty_df["Close"],

                63

            )

        )



        rs_3m=None



        if (

            stock_3m is not None

            and

            nifty_3m is not None

        ):



            rs_3m=round(

                stock_3m

                -

                nifty_3m,

                2

            )



        # -------------------------

        stock_6m=pct_change(

            stock_close,

            nth_last(

                stock_df["Close"],

                126

            )

        )



        nifty_6m=pct_change(

            nifty_close,

            nth_last(

                nifty_df["Close"],

                126

            )

        )



        rs_6m=None



        if (

            stock_6m is not None

            and

            nifty_6m is not None

        ):



            rs_6m=round(

                stock_6m

                -

                nifty_6m,

                2

            )



        # -------------------------

        stock_1y=pct_change(

            stock_close,

            nth_last(

                stock_df["Close"],

                252

            )

        )



        nifty_1y=pct_change(

            nifty_close,

            nth_last(

                nifty_df["Close"],

                252

            )

        )



        rs_1y=None



        if (

            stock_1y is not None

            and

            nifty_1y is not None

        ):



            rs_1y=round(

                stock_1y

                -

                nifty_1y,

                2

            )



        # ===================================

        # RS STATE

        # ===================================


        rs_state="WEAK"



        if (

            rs_1m is not None

            and

            rs_1m>0

            and

            rs_3m is not None

            and

            rs_3m<0

        ):



            rs_state="IMPROVING"



        if (

            rs_1m is not None

            and

            rs_1m>0

            and

            rs_3m is not None

            and

            rs_3m>0

            and

            rs_6m is not None

            and

            rs_6m>0

        ):



            rs_state="STRONG"



        if (

            rs_1m is not None

            and

            rs_1m>10

            and

            rs_3m is not None

            and

            rs_3m>15

            and

            rs_6m is not None

            and

            rs_6m>20

        ):



            rs_state="LEADER"



        # ===================================

        # SCORE

        # ===================================


        score=0



        if rs_state=="IMPROVING":

            score=4



        elif rs_state=="STRONG":

            score=8



        elif rs_state=="LEADER":

            score=10



        # ===================================



        return {


            "RS_1M":

            rs_1m,


            "RS_3M":

            rs_3m,


            "RS_6M":

            rs_6m,


            "RS_1Y":

            rs_1y,


            "RS_State":

            rs_state,


            "RS_Score":

            score

        }



    except Exception as e:



        print(

            "Relative Strength Error:",

            e

        )



        return default_rs()
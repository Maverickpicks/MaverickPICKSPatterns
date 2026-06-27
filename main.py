import pandas as pd

import yfinance as yf



from data_loader import load_stock

from trend_engine import trend_analysis

from momentum_engine import momentum_analysis

from relative_strength import relative_strength

from volume_engine import volume_analysis

from risk_engine import risk_analysis

from ranking_engine import final_score

from ranking_engine import build_reason





symbols_df=pd.read_csv(

    "symbols.csv"

)



symbols=(

    symbols_df["Symbol"]

    .dropna()

    .astype(str)

    .tolist()

)



print(

    "Downloading NIFTY..."

)



nifty=yf.download(

    "^NSEI",

    period="2y",

    interval="1d",

    progress=False

)



results=[]





for symbol in symbols:



    try:



        print(

            f"Processing {symbol}"

        )



        data=load_stock(

            symbol

        )



        daily=data["daily"]



        if len(daily)<250:

            continue



        trend=trend_analysis(

            daily.copy()

        )



        momentum=momentum_analysis(

            daily.copy()

        )



        rs=relative_strength(

            daily,

            nifty

        )



        volume=volume_analysis(

            daily.copy()

        )



        risk=risk_analysis(

            daily.copy()

        )



        rank=final_score(

            trend,

            momentum,

            rs,

            volume

        )



        reason=build_reason(

            trend,

            momentum,

            rs,

            volume

        )



        row={



            "Symbol":symbol,



            "Swing":

            rank["Swing"],



            "Confidence":

            rank["Confidence"],



            "RSI":

            momentum["RSI"],



            "RS_3M":

            rs["RS_3M"],



            "VolRatio":

            volume["vol_ratio"],



            "Entry":

            risk["Entry"],



            "SL":

            risk["SL"],



            "T1":

            risk["T1"],



            "T2":

            risk["T2"],



            "Reason":

            reason



        }



        results.append(

            row

        )



    except Exception as e:



        print(

            symbol,

            e

        )





rank_df=pd.DataFrame(

    results

)





rank_df=(

    rank_df

    .sort_values(

        by="Swing",

        ascending=False

    )

)





print(

    rank_df

)





rank_df.to_excel(

    "outputs/Swing_BTST_Ranking.xlsx",

    index=False

)





rank_df.to_csv(

    "outputs/Swing_BTST_Ranking.csv",

    index=False

)





print(

    "\nDone."

)
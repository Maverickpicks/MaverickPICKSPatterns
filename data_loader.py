import yfinance as yf
import pandas as pd
import time

from utils import normalize_columns
from utils import has_ohlcv


# ============================================
# Validate dataframe
# ============================================

def validate_dataframe(df, min_rows=100):

    try:

        if df is None:

            return False

        if df.empty:

            return False

        df = normalize_columns(df)

        if not has_ohlcv(df):

            return False

        if len(df) < min_rows:

            return False

        return True

    except:

        return False


# ============================================
# Download helper with retries
# ============================================

def download_data(

        ticker,

        period,

        interval,

        retries=3

):

    for attempt in range(retries):

        try:

            df = yf.download(

                ticker,

                period=period,

                interval=interval,

                auto_adjust=False,

                progress=False,

                threads=False

            )

            df = normalize_columns(df)

            df = df.dropna(how="all")

            if validate_dataframe(df):

                return df

        except Exception as e:

            print(

                f"{ticker} "

                f"{interval} "

                f"attempt {attempt+1} "

                f"failed"

            )

        time.sleep(1)

    return pd.DataFrame()


# ============================================
# Load stock
# ============================================

def load_stock(symbol):

    ticker = f"{symbol}.NS"


    daily = download_data(

        ticker=ticker,

        period="2y",

        interval="1d"

    )


    weekly = download_data(

        ticker=ticker,

        period="5y",

        interval="1wk"

    )


    monthly = download_data(

        ticker=ticker,

        period="10y",

        interval="1mo"

    )


    return {

        "daily": daily,

        "weekly": weekly,

        "monthly": monthly

    }


# ============================================
# Load NIFTY
# ============================================

def load_nifty():

    nifty = download_data(

        ticker="^NSEI",

        period="2y",

        interval="1d"

    )


    return nifty


# ============================================
# Quick test
# ============================================

if __name__ == "__main__":

    data = load_stock("BEL")

    print()

    print("DAILY")

    print(

        data["daily"]

        .tail()

    )



    print()

    print("WEEKLY")

    print(

        data["weekly"]

        .tail()

    )



    print()

    print("MONTHLY")

    print(

        data["monthly"]

        .tail()

    )



    nifty=load_nifty()

    print()

    print("NIFTY")

    print(

        nifty.tail()

    )
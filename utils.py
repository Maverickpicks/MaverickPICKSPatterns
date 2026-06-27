import pandas as pd
import numpy as np


# ---------------------------------------
# Flatten MultiIndex Columns
# ---------------------------------------

def normalize_columns(df):

    if df is None:

        return pd.DataFrame()


    if df.empty:

        return df


    if isinstance(df.columns, pd.MultiIndex):

        df.columns = [

            col[0]

            for col in df.columns

        ]


    df.columns = [

        str(c).strip()

        for c in df.columns

    ]


    return df


# ---------------------------------------
# Safe float extraction
# ---------------------------------------

def safe_float(value):

    try:

        if pd.isna(value):

            return np.nan


        return float(value)

    except:

        return np.nan


# ---------------------------------------
# Last valid value from series
# ---------------------------------------

def last_value(series):

    try:

        s=series.dropna()

        if len(s)==0:

            return np.nan


        return safe_float(

            s.iloc[-1]

        )

    except:

        return np.nan



# ---------------------------------------
# nth last valid value
# ---------------------------------------

def nth_last(series,n):

    try:

        s=series.dropna()


        if len(s)<=n:

            return np.nan


        return safe_float(

            s.iloc[-n]

        )

    except:

        return np.nan



# ---------------------------------------
# Safe Percentage Change
# ---------------------------------------

def pct_change(now,past):

    try:


        now=safe_float(now)

        past=safe_float(past)


        if pd.isna(now):

            return np.nan


        if pd.isna(past):

            return np.nan


        if past==0:

            return np.nan



        return round(

            ((now-past)/past)*100,

            2

        )


    except:

        return np.nan




# ---------------------------------------
# Safe divide
# ---------------------------------------

def safe_divide(a,b):

    try:


        a=safe_float(a)

        b=safe_float(b)


        if pd.isna(a):

            return np.nan


        if pd.isna(b):

            return np.nan


        if b==0:

            return np.nan


        return round(

            a/b,

            2

        )


    except:

        return np.nan




# ---------------------------------------
# Boolean conversion
# ---------------------------------------

def bool_safe(x):

    try:

        return bool(x)

    except:

        return False




# ---------------------------------------
# Empty result dictionary
# ---------------------------------------

def empty_dict():

    return {}




# ---------------------------------------
# Check required columns
# ---------------------------------------

def has_ohlcv(df):


    required=[

        "Open",

        "High",

        "Low",

        "Close",

        "Volume"

    ]


    for col in required:

        if col not in df.columns:

            return False


    return True
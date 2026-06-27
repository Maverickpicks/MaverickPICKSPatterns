import yfinance as yf

from utils import normalize_columns


df = yf.download(

    "BEL.NS",

    period="2y",

    progress=False

)


df = normalize_columns(df)


print(df.columns)

print()

print(df.tail())
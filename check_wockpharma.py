from fetch_data import fetch_ohlcv
df = fetch_ohlcv("WOCKPHARMA", lookback_days=365)
print(df[(df['Date'] >= '2026-04-20') & (df['Date'] <= '2026-05-10')][['Date','Open','High','Low','Close','Volume']].to_string())
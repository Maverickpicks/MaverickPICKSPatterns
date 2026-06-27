import json
import os
import time
import yfinance as yf

# ============================================================
# SECTOR MAP
# Fetches sector/industry for each NIFTY500 symbol via yfinance.
# Caches to disk so this only needs to run once (or periodically),
# since .info calls are slow and rate-limit prone at 500-stock scale.
# ============================================================

CACHE_FILE = "sector_map_cache.json"
CACHE_MAX_AGE_DAYS = 30


def _cache_is_fresh():
    if not os.path.exists(CACHE_FILE):
        return False
    age_days = (time.time() - os.path.getmtime(CACHE_FILE)) / 86400
    return age_days < CACHE_MAX_AGE_DAYS


def load_sector_map():
    """Load cached sector map if fresh, else return empty dict."""
    if _cache_is_fresh():
        try:
            with open(CACHE_FILE, "r") as f:
                return json.load(f)
        except Exception:
            return {}
    return {}


def save_sector_map(mapping):
    try:
        with open(CACHE_FILE, "w") as f:
            json.dump(mapping, f, indent=2)
    except Exception as e:
        print("Could not save sector cache:", e)


def fetch_sector(symbol, retries=2):
    """Fetch sector/industry for one symbol. Returns 'Unknown' on failure."""
    ticker = f"{symbol}.NS"
    for attempt in range(retries):
        try:
            info = yf.Ticker(ticker).info
            sector = info.get("sector") or "Unknown"
            industry = info.get("industry") or "Unknown"
            return sector, industry
        except Exception:
            time.sleep(1)
    return "Unknown", "Unknown"


def build_sector_map(symbols, force_refresh=False, save_every=25):
    """
    Build (or update) the sector map for a list of symbols.
    Uses cache for symbols already known unless force_refresh=True.
    Saves incrementally so a long run can be interrupted safely.
    """
    mapping = {} if force_refresh else load_sector_map()

    missing = [s for s in symbols if s not in mapping]

    if not missing:
        print(f"Sector map cache hit for all {len(symbols)} symbols.")
        return mapping

    print(f"Fetching sector info for {len(missing)} symbols (this can take a while)...")

    for i, symbol in enumerate(missing, 1):
        sector, industry = fetch_sector(symbol)
        mapping[symbol] = {"sector": sector, "industry": industry}

        if i % 10 == 0 or i == len(missing):
            print(f"  [{i}/{len(missing)}] {symbol}: {sector} / {industry}")

        if i % save_every == 0:
            save_sector_map(mapping)

        time.sleep(0.3)

    save_sector_map(mapping)
    return mapping


def get_sector_peers(symbol, mapping, all_symbols, max_peers=15):
    """
    Return a list of symbols sharing the same sector as `symbol`,
    excluding the symbol itself, capped at max_peers.
    """
    info = mapping.get(symbol)
    if not info or info.get("sector") in (None, "Unknown"):
        return []

    sector = info["sector"]

    peers = [
        s for s in all_symbols
        if s != symbol
        and mapping.get(s, {}).get("sector") == sector
    ]

    return peers[:max_peers]


if __name__ == "__main__":
    import pandas as pd
    df = pd.read_csv("NIFTY500_MASTER.csv")
    symbols = df["Symbol"].dropna().astype(str).str.strip().tolist()
    mapping = build_sector_map(symbols)
    print(f"\nDone. {len(mapping)} symbols mapped.")

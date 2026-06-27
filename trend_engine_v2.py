import pandas as pd
import pandas_ta as ta
from utils import last_value


# ======================================
# DEFAULT
# ======================================

def default_trend():

    return {
        "Trend_State": "WEAK",
        "Trend_Score": 5,
        "EMA9": None,
        "EMA20": None,
        "EMA50": None,
        "EMA100": None,
        "EMA200": None,
        "Bullish_Alignment": False,
        "EMA20_Rising": False,
        "EMA50_Rising": False,
        "EMA200_Rising": False,
        "EMA_Gap_Expanding": False,
        "Golden_Cross_9_20": False,
        "Golden_Cross_20_50": False,
        "Golden_Cross_50_200": False,
        # New fields
        "Pullback_To_EMA20": False,
        "Pullback_To_EMA50": False,
        "RSI_Cooling": False,
        "Volume_Drying_Pullback": False,
        "Pullback_Setup": False,
        "ATH_Proximity_Pct": None,
        "ATH_Risk": False,
        "Dist_From_52W_High_Pct": None,
        "Price_Zone": "UNKNOWN"
    }


# ======================================
# TREND ANALYSIS
# ======================================

def trend_analysis(df):

    try:

        if df is None:
            return default_trend()

        if df.empty:
            return default_trend()

        if len(df) < 250:
            return default_trend()


        # ================================
        # EMA
        # ================================

        df["EMA9"]   = ta.ema(df["Close"], length=9)
        df["EMA20"]  = ta.ema(df["Close"], length=20)
        df["EMA50"]  = ta.ema(df["Close"], length=50)
        df["EMA100"] = ta.ema(df["Close"], length=100)
        df["EMA200"] = ta.ema(df["Close"], length=200)

        ema9   = last_value(df["EMA9"])
        ema20  = last_value(df["EMA20"])
        ema50  = last_value(df["EMA50"])
        ema100 = last_value(df["EMA100"])
        ema200 = last_value(df["EMA200"])

        close = last_value(df["Close"])


        # ================================
        # RSI (needed for pullback check)
        # ================================

        df["RSI"] = ta.rsi(df["Close"], length=14)
        rsi = last_value(df["RSI"])


        # ================================
        # Alignment
        # ================================

        bullish_alignment = (ema9 > ema20 > ema50 > ema200)


        # ================================
        # EMA Rising
        # ================================

        ema20_rising  = (df["EMA20"].iloc[-1]  > df["EMA20"].iloc[-5])
        ema50_rising  = (df["EMA50"].iloc[-1]  > df["EMA50"].iloc[-5])
        ema200_rising = (df["EMA200"].iloc[-1] > df["EMA200"].iloc[-10])


        # ================================
        # EMA Gap
        # ================================

        current_gap  = ema9 - ema20
        previous_gap = df["EMA9"].iloc[-5] - df["EMA20"].iloc[-5]
        ema_gap_expanding = (current_gap > previous_gap)


        # ================================
        # Golden Cross
        # ================================

        golden_9_20 = (
            df["EMA9"].iloc[-2] <= df["EMA20"].iloc[-2]
            and ema9 > ema20
        )

        golden_20_50 = (
            df["EMA20"].iloc[-2] <= df["EMA50"].iloc[-2]
            and ema20 > ema50
        )

        golden_50_200 = (
            df["EMA50"].iloc[-2] <= df["EMA200"].iloc[-2]
            and ema50 > ema200
        )


        # ================================
        # ATH PROXIMITY
        # How close is price to 52-week high?
        # Danger zone = within 3% of ATH
        # ================================

        high_52w = df["High"].tail(252).max()

        dist_from_52w_high_pct = round(
            ((high_52w - close) / high_52w) * 100, 2
        )

        # ATH Risk: price within 3% of 52-week high AND RSI overbought
        ath_risk = (
            dist_from_52w_high_pct <= 3
            and rsi is not None
            and rsi > 68
        )

        # Proximity % to ATH (0 = AT ATH, 20 = 20% below)
        ath_proximity_pct = dist_from_52w_high_pct


        # ================================
        # PRICE ZONE
        # Where is price relative to EMAs?
        # This tells us the setup quality
        # ================================

        price_zone = "UNKNOWN"

        if close is not None and ema20 is not None and ema50 is not None:

            if close >= ema9:
                # Price above EMA9 - extended or strong
                if ath_risk:
                    price_zone = "EXTENDED_ATH"
                else:
                    price_zone = "EXTENDED"

            elif close >= ema20:
                # Between EMA9 and EMA20 - mild pullback, still healthy
                price_zone = "PULLBACK_EMA20"

            elif close >= ema50:
                # Between EMA20 and EMA50 - deeper pullback, sweet spot
                price_zone = "PULLBACK_EMA50"

            elif close >= ema200:
                # Between EMA50 and EMA200 - weak but above long term
                price_zone = "WEAK_ABOVE_200"

            else:
                # Below EMA200 - bearish territory
                price_zone = "BELOW_200"


        # ================================
        # PULLBACK SETUP DETECTION
        # The sweet spot for swing entries:
        # - Long term trend still intact (EMA50 and EMA200 rising)
        # - Price pulled back to EMA20 or EMA50
        # - RSI cooling (not overbought)
        # - Volume drying up on the pullback (sellers exhausted)
        # ================================

        pullback_to_ema20 = (
            price_zone == "PULLBACK_EMA20"
            and ema50_rising
            and ema200_rising
        )

        pullback_to_ema50 = (
            price_zone == "PULLBACK_EMA50"
            and ema200_rising
            and ema50 > ema200   # Long term structure intact
        )

        # RSI cooling = between 38 and 58, not overbought
        rsi_cooling = (
            rsi is not None
            and 38 <= rsi <= 58
        )

        # Volume drying on pullback = recent volume below 20-day avg
        # Means sellers are not aggressive, just natural correction
        recent_vol   = df["Volume"].tail(5).mean()
        avg_vol_20   = df["Volume"].tail(20).mean()
        volume_drying_pullback = (
            recent_vol < avg_vol_20 * 0.85
        )

        # Full pullback setup = all conditions met
        pullback_setup = (
            (pullback_to_ema20 or pullback_to_ema50)
            and rsi_cooling
        )


        # ================================
        # TREND STATE
        # ================================

        trend_state = "WEAK"
        trend_score = 5

        if (
            bullish_alignment
            and ema20_rising
            and ema50_rising
            and ema200_rising
        ):
            trend_state = "LEADER"
            trend_score = 25

        elif (
            ema9 > ema20 > ema50
            and ema20_rising
        ):
            trend_state = "STRONG"
            trend_score = 20

        elif (
            ema9 > ema20
            or ema_gap_expanding
            or golden_9_20
            or pullback_setup   # Pullback within healthy trend = NEUTRAL not WEAK
        ):
            trend_state = "NEUTRAL"
            trend_score = 12


        # ================================
        # PULLBACK BONUS SCORE
        # Reward the setup we actually want
        # ================================

        pullback_bonus = 0

        if pullback_setup and volume_drying_pullback:
            pullback_bonus = 8   # Strong pullback with volume confirmation
        elif pullback_setup:
            pullback_bonus = 4   # Good pullback setup

        trend_score = min(trend_score + pullback_bonus, 25)


        # ================================

        return {

            "Trend_State":          trend_state,
            "Trend_Score":          trend_score,

            "EMA9":                 round(ema9,   2),
            "EMA20":                round(ema20,  2),
            "EMA50":                round(ema50,  2),
            "EMA100":               round(ema100, 2),
            "EMA200":               round(ema200, 2),

            "Bullish_Alignment":    bool(bullish_alignment),
            "EMA20_Rising":         bool(ema20_rising),
            "EMA50_Rising":         bool(ema50_rising),
            "EMA200_Rising":        bool(ema200_rising),
            "EMA_Gap_Expanding":    bool(ema_gap_expanding),

            "Golden_Cross_9_20":    golden_9_20,
            "Golden_Cross_20_50":   golden_20_50,
            "Golden_Cross_50_200":  golden_50_200,

            # Pullback
            "Pullback_To_EMA20":        bool(pullback_to_ema20),
            "Pullback_To_EMA50":        bool(pullback_to_ema50),
            "RSI_Cooling":              bool(rsi_cooling),
            "Volume_Drying_Pullback":   bool(volume_drying_pullback),
            "Pullback_Setup":           bool(pullback_setup),

            # ATH
            "ATH_Proximity_Pct":        ath_proximity_pct,
            "Dist_From_52W_High_Pct":   dist_from_52w_high_pct,
            "ATH_Risk":                 bool(ath_risk),

            # Zone
            "Price_Zone":               price_zone,

        }

    except Exception as e:

        print("Trend Engine Error:", e)

        return default_trend()

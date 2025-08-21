import os
import time
import datetime as dt
import pandas as pd
import numpy as np
import streamlit as st
import plotly.graph_objects as go

from dotenv import load_dotenv
from kiteconnect import KiteConnect
from ta.momentum import RSIIndicator
from ta.trend import SMAIndicator
from ta.volume import VolumeWeightedAveragePrice

# ========== Setup ==========
st.set_page_config(page_title="NIFTY Intraday Trade Dashboard", layout="wide")
load_dotenv()

API_KEY = os.getenv("API_KEY")
ACCESS_TOKEN = os.getenv("ACCESS_TOKEN")

if not API_KEY or not ACCESS_TOKEN:
    st.error("API_KEY or ACCESS_TOKEN missing in .env. Please add and restart.")
    st.stop()

kite = KiteConnect(api_key=API_KEY)
kite.set_access_token(ACCESS_TOKEN)

# ========== Helpers ==========

def get_nifty_token():
    try:
        q = kite.ltp(["NSE:NIFTY 50"])
        info = q.get("NSE:NIFTY 50") or list(q.values())[0]
        return info.get("instrument_token") or 256265
    except Exception:
        return 256265

# ---------- Data Fetchers ----------

def fetch_intraday_data():
    token = get_nifty_token()
    today = dt.date.today()
    from_dt = dt.datetime.combine(today, dt.time(9, 15))
    to_dt = dt.datetime.now()

    data = kite.historical_data(
        instrument_token=token,
        from_date=from_dt,
        to_date=to_dt,
        interval="5minute"
    )
    df = pd.DataFrame(data)
    if df.empty:
        return df

    close = df['close']
    df['RSI'] = RSIIndicator(close, window=14).rsi()
    df['SMA20'] = SMAIndicator(close, window=20).sma_indicator()

    cumulative_vol = df['volume'].cumsum()
    cum_vol_price = (df['volume'] * ((df['high']+df['low']+df['close'])/3)).cumsum()
    df['VWAP'] = cum_vol_price / cumulative_vol
    return df

def fetch_pivots():
    token = get_nifty_token()
    today = dt.date.today()
    from_dt = today - dt.timedelta(days=2)
    to_dt = today - dt.timedelta(days=1)
    data = kite.historical_data(token, from_dt, to_dt, interval="day")
    if not data:
        return {}
    prev = data[-1]
    ph, pl, pc = prev["high"], prev["low"], prev["close"]
    p = (ph + pl + pc)/3
    return {
        'P': round(p, 2),
        'R1': round(2*p - pl, 2),
        'S1': round(2*p - ph, 2),
        'R2': round(p + (ph - pl), 2),
        'S2': round(p - (ph - pl), 2),
        'R3': round(ph + 2*(p - pl), 2),
        'S3': round(pl - 2*(ph - p), 2)
    }

# ---------- Option Chain Helpers ----------

def get_nearest_expiry():
    instruments = kite.instruments("NFO")
    nifty_opts = [i for i in instruments if i["name"] == "NIFTY" and i["segment"] == "NFO-OPT"]
    expiries = sorted({i["expiry"] for i in nifty_opts})
    today = dt.date.today()
    for e in expiries:
        if e >= today:
            return e
    return expiries[0]

def fetch_option_chain(spot_price):
    expiry = get_nearest_expiry()
    atm_strike = int(round(spot_price/50)*50)
    strikes = [atm_strike + i*50 for i in range(-3, 4)]
    option_data = []
    for strike in strikes:
        ce = f"NFO:NIFTY{expiry.strftime('%y%b').upper()}{strike}CE"
        pe = f"NFO:NIFTY{expiry.strftime('%y%b').upper()}{strike}PE"
        try:
            ce_q = kite.quote(ce)[ce]
            pe_q = kite.quote(pe)[pe]
            option_data.append({
                "Strike": strike,
                "CE_Price": ce_q["last_price"],
                "PE_Price": pe_q["last_price"],
                "CE_OI": ce_q["oi"],
                "PE_OI": pe_q["oi"]
            })
        except:
            continue
    return pd.DataFrame(option_data)

def calculate_option_metrics(option_df):
    if option_df.empty:
        return 0, 0, 0
    pcr = option_df["PE_OI"].sum() / max(option_df["CE_OI"].sum(), 1)
    pain = option_df.apply(lambda r: (abs(option_df['Strike']-r['Strike'])* 
                                      (r['CE_OI']+r['PE_OI'])).sum(), axis=1)
    maxpain_strike = option_df.iloc[pain.idxmin()]["Strike"]
    atm = option_df.iloc[(option_df['Strike']-option_df['Strike'].median()).abs().idxmin()]["Strike"]
    return round(pcr,2), maxpain_strike, atm

# ---------- Price Action Helpers ----------

def is_bullish_engulfing(df):
    if len(df) < 2:
        return False
    prev, curr = df.iloc[-2], df.iloc[-1]
    return (prev['close'] < prev['open'] and curr['close'] > curr['open'] and
            curr['open'] < prev['close'] and curr['close'] > prev['open'])

def is_bearish_engulfing(df):
    if len(df) < 2:
        return False
    prev, curr = df.iloc[-2], df.iloc[-1]
    return (prev['close'] > prev['open'] and curr['close'] < curr['open'] and
            curr['open'] > prev['close'] and curr['close'] < prev['open'])

# ---------- Signal Generation ----------

def generate_trade_signal(df, pivots, option_df):
    if df.empty:
        return "NEUTRAL", 50, ["No data"]
    latest = df.iloc[-1]
    score = 50
    reasoning = []
    
    # Trend vs SMA
    if latest['close'] > latest['SMA20']:
        score += 10; reasoning.append("Above SMA20")
    else:
        score -= 10; reasoning.append("Below SMA20")

    # VWAP
    if latest['close'] > latest['VWAP']:
        score += 8; reasoning.append("Above VWAP")
    else:
        score -= 8; reasoning.append("Below VWAP")

    # Pivots
    if latest['close'] > pivots.get('R1', 1e9):
        score += 15; reasoning.append("Breakout above R1")
    elif latest['close'] < pivots.get('S1', -1e9):
        score -= 15; reasoning.append("Breakdown below S1")

    # RSI
    if latest['RSI'] > 70:
        score -= 12; reasoning.append("Overbought RSI")
    elif latest['RSI'] < 30:
        score += 12; reasoning.append("Oversold RSI")

    # Engulfing
    if is_bullish_engulfing(df):
        score += 15; reasoning.append("Bullish Engulfing")
    elif is_bearish_engulfing(df):
        score -= 15; reasoning.append("Bearish Engulfing")

    # Options metrics
    if not option_df.empty:
        pcr, maxpain, atm = calculate_option_metrics(option_df)
        if pcr > 1.2:
            score += 10; reasoning.append(f"High PCR {pcr}")
        elif pcr < 0.8:
            score -= 10; reasoning.append(f"Low PCR {pcr}")

    score = max(0, min(100, score))
    if score >= 75: sig = "STRONG_BUY"
    elif score >= 60: sig = "BUY"
    elif score <= 25: sig = "STRONG_SELL"
    elif score <= 40: sig = "SELL"
    else: sig = "NEUTRAL"
    return sig, score, reasoning

def get_nifty_token():
    """Get NIFTY 50 instrument token via LTP (preferred) or fallback 256265."""
    try:
        q = kite.ltp(["NSE:NIFTY 50"])
        info = q.get("NSE:NIFTY 50") or list(q.values())[0]
        return info.get("instrument_token") or 256265
    except Exception:
        return 256265

def get_spot():
    try:
        return float(kite.ltp(["NSE:NIFTY 50"])["NSE:NIFTY 50"]["last_price"])
    except Exception:
        return None

def get_vix():
    try:
        return float(kite.ltp(["NSE:INDIA VIX"])["NSE:INDIA VIX"]["last_price"])
    except Exception:
        return None

def fetch_intraday_5min(token, days=1):
    """Fetch 5-minute candles for recent period."""
    end = dt.datetime.now()
    start = end - dt.timedelta(days=days)
    data = kite.historical_data(token, start, end, "5minute")
    df = pd.DataFrame(data)
    if df.empty:
        return df
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)
    return df

def last_trading_day_ref():
    """Return yesterday date (backtrack over weekends/holidays using available daily candles)."""
    # We'll try the last 7 days until we get a daily candle
    token = get_nifty_token()
    for d in range(1, 8):
        day = dt.date.today() - dt.timedelta(days=d)
        try:
            bars = kite.historical_data(token, day, day, "day")
            if bars:
                return bars[0]  # dict with open/high/low/close/date
        except Exception:
            continue
    return None

def calc_pivots_from_ohlc(ohlc):
    """Standard floor pivots."""
    h, l, c = float(ohlc["high"]), float(ohlc["low"]), float(ohlc["close"])
    p = (h + l + c) / 3.0
    r1 = 2*p - l
    s1 = 2*p - h
    r2 = p + (h - l)
    s2 = p - (h - l)
    r3 = h + 2*(p - l)
    s3 = l - 2*(h - p)
    return {"P": p, "R1": r1, "S1": s1, "R2": r2, "S2": s2, "R3": r3, "S3": s3}

def classify_spot_range(spot, prev_high, prev_low, df5):
    """
    Classify NIFTY spot relative to:
    1) Previous day high/low
    2) First 1-hour candle (9:15 - 10:15)
    Returns dict with textual classification.
    """
    result = {}

    # --- Previous day range ---
    if spot > prev_high:
        result['PrevDay'] = "Above PDH 📈"
    elif spot < prev_low:
        result['PrevDay'] = "Below PDL 📉"
    else:
        result['PrevDay'] = "Inside previous day range ⚖️"

    # --- First 1-hour candle ---
    first_1h = df5[df5["date"].dt.time.between(dt.time(9,15), dt.time(10,15))]
    if not first_1h.empty:
        high_1h = first_1h["high"].max()
        low_1h = first_1h["low"].min()
        if spot > high_1h:
            result['1H'] = "Above 1H candle 📈"
        elif spot < low_1h:
            result['1H'] = "Below 1H candle 📉"
        else:
            result['1H'] = "Inside 1H candle ⚖️"
    else:
        result['1H'] = "1H candle data unavailable"

    return result

def add_intraday_indicators(df):
    if df.empty:
        return df
    close = df["close"].astype(float)
    high = df["high"].astype(float)
    low  = df["low"].astype(float)
    vol  = df.get("volume", pd.Series(np.zeros(len(df)))).astype(float)

    df["RSI"] = RSIIndicator(close, window=14).rsi()
    df["SMA20"] = SMAIndicator(close, window=20).sma_indicator()
    vwap = VolumeWeightedAveragePrice(high=high, low=low, close=close, volume=vol)
    df["VWAP"] = vwap.volume_weighted_average_price()
    return df

def first_5min_breakout_levels(df, buffer_pts=5.0):
    """Using first 5-min candle of the day."""
    if df.empty:
        return None, None
    d0 = df[df["date"].dt.time >= dt.time(9,15)]
    if d0.empty:
        return None, None
    first_bar = d0.iloc[0]
    above = float(first_bar["high"]) + float(buffer_pts)
    below = float(first_bar["low"])  - float(buffer_pts)
    return above, below

def fetch_ohlc(token, interval, days=5):
    end = dt.datetime.now()
    start = end - dt.timedelta(days=days)
    data = kite.historical_data(token, start, end, interval, continuous=False)
    return pd.DataFrame(data)

# -----------------------------------------
# Compute key breakout levels
# -----------------------------------------
def compute_levels(df):
    today = df[df["date"].dt.date == dt.date.today()]
    prev = df[df["date"].dt.date == dt.date.today() - dt.timedelta(days=1)]

    prev_high = prev["high"].max() if not prev.empty else None
    prev_low = prev["low"].min() if not prev.empty else None
    day_open = today.iloc[0]["open"] if not today.empty else None

    # First 5m candle (9:15 - 9:20)
    first5 = today[today["date"].dt.time.between(dt.time(9, 15), dt.time(9, 20))]
    first_high = first5["high"].max() if not first5.empty else None
    first_low = first5["low"].min() if not first5.empty else None

    return prev_high, prev_low, day_open, first_high, first_low

def is_engulfing(prev_candle, curr_candle, direction):
    """Classic engulfing pattern on OHLC dict/Series."""
    po, ph, pl, pc = float(prev_candle["open"]), float(prev_candle["high"]), float(prev_candle["low"]), float(prev_candle["close"])
    co, ch, cl, cc = float(curr_candle["open"]), float(curr_candle["high"]), float(curr_candle["low"]), float(curr_candle["close"])
    if direction == "bullish":
        return (pc < po) and (cc > co) and (co < pc) and (cc > po)
    if direction == "bearish":
        return (pc > po) and (cc < co) and (co > pc) and (cc < po)
    return False

# ---- Option chain helpers (ATM ± N strikes) ----

def nearest_weekly_expiry(instruments_df):
    """Pick the nearest expiry >= today."""
    today = dt.date.today()
    exp = (
        instruments_df.loc[instruments_df["expiry"] >= pd.Timestamp(today)]
        .sort_values("expiry")
        .head(1)
    )
    if exp.empty:
        # fallback to minimum expiry present
        exp = instruments_df.sort_values("expiry").head(1)
    return None if exp.empty else exp.iloc[0]["expiry"].date()

def get_nifty_nfo_instruments():
    """Zerodha instruments for NFO:NIFTY (options)."""
    inst = kite.instruments("NFO")
    df = pd.DataFrame(inst)
    if df.empty:
        return df
    df = df[(df["segment"] == "NFO-OPT") & (df["name"] == "NIFTY")].copy()
    # ensure types
    df["expiry"] = pd.to_datetime(df["expiry"])
    df["strike"] = pd.to_numeric(df["strike"], errors="coerce")
    return df

def build_atm_symbols(atm, nfo_df, strikes_each_side=3):
    """Return CE/PE tradingsymbols for ATM ± N strikes for nearest weekly expiry."""
    if nfo_df.empty:
        return []

    exp_date = nearest_weekly_expiry(nfo_df)
    if not exp_date:
        return []

    strikes = [atm + i*50 for i in range(-strikes_each_side, strikes_each_side+1)]
    rows = nfo_df[(nfo_df["expiry"].dt.date == exp_date) & (nfo_df["strike"].isin(strikes))]
    rows = rows[rows["tradingsymbol"].str.contains("NIFTY", na=False)]

    # return pairs as dict: strike -> {"CE": sym, "PE": sym}
    symmap = {}
    for s in strikes:
        ce = rows[(rows["strike"] == s) & (rows["instrument_type"] == "CE")]
        pe = rows[(rows["strike"] == s) & (rows["instrument_type"] == "PE")]
        if not ce.empty and not pe.empty:
            symmap[s] = {
                "CE": f"NFO:{ce.iloc[0]['tradingsymbol']}",
                "PE": f"NFO:{pe.iloc[0]['tradingsymbol']}",
                "CE_token": int(ce.iloc[0]["instrument_token"]),
                "PE_token": int(pe.iloc[0]["instrument_token"]),
            }
    return symmap

def quote_oi_safe(identifier, fallback_token=None):
    """
    Try to get OI for an option. If not present in quote(), fallback to daily historical (oi=True).
    identifier: string like 'NFO:XYZ123CE'
    fallback_token: instrument_token int
    """
    oi_val = None
    try:
        q = kite.quote([identifier])
        item = q.get(identifier)
        if item:
            oi_val = item.get("oi") or item.get("open_interest")  # if provided
    except Exception:
        pass

    if (oi_val is None) and fallback_token is not None:
        try:
            today = dt.date.today()
            # daily bar for today to read 'oi' if exchange publishes
            bars = kite.historical_data(fallback_token,
                                        dt.datetime.combine(today, dt.time(9,15)),
                                        dt.datetime.combine(today, dt.time(15,30)),
                                        "day", oi=True)
            if bars:
                oi_val = bars[-1].get("oi") or bars[-1].get("open_interest")
        except Exception:
            oi_val = None
    return int(oi_val) if oi_val not in (None, "") else None

def fetch_atm_option_snapshot_and_pcr(atm, strikes_each_side=3):
    """
    Build a small option snapshot around ATM and compute PCR (sum PE_OI / sum CE_OI) when possible.
    Returns: (snapshot_df, pcr_value)
    """
    nfo = get_nifty_nfo_instruments()
    if nfo.empty or atm is None:
        return pd.DataFrame(), None

    symmap = build_atm_symbols(atm, nfo, strikes_each_side=strikes_each_side)
    if not symmap:
        return pd.DataFrame(), None

    # fetch LTPs
    idents = []
    for s, m in symmap.items():
        idents.extend([m["CE"], m["PE"]])
    quotes = {}
    try:
        quotes = kite.quote(idents)
    except Exception:
        quotes = {}

    rows = []
    ce_oi_sum, pe_oi_sum = 0, 0
    have_oi = False

    for strike, m in symmap.items():
        ce_id, pe_id = m["CE"], m["PE"]
        ce_q = quotes.get(ce_id, {})
        pe_q = quotes.get(pe_id, {})
        ce_ltp = ce_q.get("last_price")
        pe_ltp = pe_q.get("last_price")

        # Try OI
        ce_oi = quote_oi_safe(ce_id, m.get("CE_token"))
        pe_oi = quote_oi_safe(pe_id, m.get("PE_token"))
        if ce_oi is not None and pe_oi is not None:
            have_oi = True
            ce_oi_sum += ce_oi
            pe_oi_sum += pe_oi

        rows.append({
            "Strike": strike,
            "CE": ce_id.split(":",1)[1],
            "PE": pe_id.split(":",1)[1],
            "CE_LTP": ce_ltp,
            "PE_LTP": pe_ltp,
            "CE_OI": ce_oi,
            "PE_OI": pe_oi
        })

    df = pd.DataFrame(rows).sort_values("Strike").reset_index(drop=True)
    pcr = round(pe_oi_sum/ce_oi_sum, 3) if have_oi and ce_oi_sum else None
    return df, pcr

# ========== Signal Logic ==========

def trade_signal_logic(df5, pivots, vix, option_df, pcr,
                       breakout_above, breakout_below,
                       prev_high=None, prev_low=None,
                       day_open=None, first_high=None, first_low=None):
    """
    Returns (signal_label, reasoning_str, score_int)
    signal_label in: 'ENGULFING_BUY', 'ENGULFING_SELL', 'BUY', 'SELL', 'NEUTRAL'

    Reasoning & logs are returned as a formatted string split into:
      - Positive factors (with +delta each)
      - Negative factors (with -delta each)
      - Neutral / Observations
    """
    # helper to add reason and update score
    def _add_pos(text, delta=0):
        if delta:
            pos.append(f"+{delta}: {text}")
        else:
            pos.append(f"0: {text}")
        nonlocal score
        score += delta

    def _add_neg(text, delta=0):
        if delta:
            neg.append(f"{delta}: {text}")  # delta expected negative
        else:
            neg.append(f"0: {text}")
        nonlocal score
        score += delta

    def _add_neu(text):
        neu.append(text)

    pos, neg, neu = [], [], []
    score = 50  # baseline

    # quick guard
    if df5.empty:
        reasoning = "No intraday data available."
        return "NEUTRAL", reasoning, 40

    latest = df5.iloc[-1]
    prev   = df5.iloc[-2] if len(df5) >= 2 else latest

    close = float(latest["close"])
    rsi   = float(latest.get("RSI") or 0.0)

    # ---------------------------
    # 1) Confirmed / Partial Breakouts (HIGH priority)
    # ---------------------------
    # Confirmed Bullish (both prev high and 1st5m high)
    if (prev_high is not None) and (first_high is not None) and (close > prev_high) and (close > first_high):
        _add_pos(
            f"Confirmed Bullish Breakout — Close {close:.2f} > Prev High {prev_high:.2f} AND > 1st-5m High {first_high:.2f}. "
            "This is strong confluence: previous resistance breached and opening consolidation cleared, increasing continuation probability.",
            delta=18
        )
        confirmed_breakout = "BULL"
    # Confirmed Bearish
    elif (prev_low is not None) and (first_low is not None) and (close < prev_low) and (close < first_low):
        _add_neg(
            f"Confirmed Bearish Breakdown — Close {close:.2f} < Prev Low {prev_low:.2f} AND < 1st-5m Low {first_low:.2f}. "
            "Strong confluence: prior support broken and opening consolidation failed, increasing continuation to downside.",
            delta=-18
        )
        confirmed_breakout = "BEAR"
    else:
        confirmed_breakout = None
        # Partial / Weak signals (only one level crossed)
        # Bullish partials
        if (prev_high is not None) and (close > prev_high):
            _add_pos(
                f"Weak Bullish — Close {close:.2f} > Prev High {prev_high:.2f} but 1st-5m High not cleared. "
                "Single-level breach; treat as weaker confirmation without 1st-5m confluence.",
                delta=6
            )
        elif (first_high is not None) and (close > first_high):
            _add_pos(
                f"Weak Bullish — Close {close:.2f} > 1st-5m High {first_high:.2f} but Prev High not cleared. "
                "Momentum present but lacks previous-day resistance breach for strong confirmation.",
                delta=6
            )

        # Bearish partials
        if (prev_low is not None) and (close < prev_low):
            _add_neg(
                f"Weak Bearish — Close {close:.2f} < Prev Low {prev_low:.2f} but 1st-5m Low not broken. "
                "Single-level breach; caution as this may be a short-lived move.",
                delta=-6
            )
        elif (first_low is not None) and (close < first_low):
            _add_neg(
                f"Weak Bearish — Close {close:.2f} < 1st-5m Low {first_low:.2f} but Prev Low not broken. "
                "Downside momentum exists but lacks prior support breach.",
                delta=-6
            )

        # if no partials added, note neutral
        if not pos and not neg:
            _add_neu("No breakout confirmation (no partial or confirmed conditions met).")

    # ---------------------------
    # 3) Pivot S/R (structural)
    # ---------------------------
    S1, S2, S3 = pivots["S1"], pivots["S2"], pivots["S3"]
    R1, R2, R3 = pivots["R1"], pivots["R2"], pivots["R3"]

    if close > R1:
        _add_pos(f"Price above R1 ({R1:.2f}) — bullish structural tilt.", delta=10)
    elif close < S1:
        _add_neg(f"Price below S1 ({S1:.2f}) — bearish structural tilt.", delta=-10)
    else:
        _add_neu(f"Price between S1 ({S1:.2f}) and R1 ({R1:.2f}) — range-bound around pivot.")

    # ---------------------------
    # 4) RSI momentum
    # ---------------------------
    if rsi >= 65:
        _add_pos(f"RSI {rsi:.1f} indicates strong upward momentum.", delta=10)
    elif rsi <= 35:
        _add_neg(f"RSI {rsi:.1f} indicates strong downward momentum.", delta=-10)
    else:
        _add_neu(f"RSI {rsi:.1f} is neutral (no extreme momentum).")

    # ---------------------------
    # 5) Candlestick (Engulfing)
    # ---------------------------
    engulf_buy  = is_engulfing(prev, latest, "bullish")
    engulf_sell = is_engulfing(prev, latest, "bearish")
    if engulf_buy:
        _add_pos("Bullish engulfing pattern on last two candles → strong short-term reversal/continuation signal.", delta=15)
    if engulf_sell:
        _add_neg("Bearish engulfing pattern on last two candles → strong short-term reversal/continuation to downside.", delta=-15)

    # ---------------------------
    # 6) Breakout vs first-bar buffer (supporting) — skip if already confirmed
    # ---------------------------
    if confirmed_breakout is None:
        if breakout_above and close > breakout_above:
            _add_pos(
                f"Breakout above first-bar high+buffer ({breakout_above:.2f}) — supports bullish case (buffered breakout).",
                delta=12
            )
        elif breakout_below and close < breakout_below:
            _add_neg(
                f"Breakdown below first-bar low-buffer ({breakout_below:.2f}) — supports bearish case (buffered breakdown).",
                delta=-12
            )
        else:
            _add_neu("No buffered first-bar breakout/breakdown detected.")

    else:
        # If confirmed, note that buffer check is redundant
        _add_neu("Buffered first-bar check skipped — breakout already confirmed by PrevDay + 1st-5m confluence.")

    # ---------------------------
    # 7) Options bias & PCR
    # ---------------------------
    if option_df is not None and not option_df.empty:
        try:
            atm_row = option_df.iloc[(option_df["Strike"] - close).abs().argmin()]
            ce_ltp = atm_row.get("CE_LTP")
            pe_ltp = atm_row.get("PE_LTP")
            if (ce_ltp is not None) and (pe_ltp is not None):
                if ce_ltp > pe_ltp:
                    _add_pos(f"ATM options: CE_LTP ({ce_ltp}) > PE_LTP ({pe_ltp}) → mild bullish skew in option prices.", delta=5)
                elif pe_ltp > ce_ltp:
                    _add_neg(f"ATM options: PE_LTP ({pe_ltp}) > CE_LTP ({ce_ltp}) → mild bearish skew in option prices.", delta=-5)
        except Exception:
            _add_neu("Could not compute ATM CE/PE skew from option snapshot.")
        if pcr is not None:
            if pcr > 1.1:
                _add_pos(f"PCR {pcr:.3f} (put-heavy) — contrarian bullish tilt (puts expensive or heavy).", delta=5)
            elif pcr < 0.9:
                _add_neg(f"PCR {pcr:.3f} (call-heavy) — contrarian bearish tilt (calls expensive/heavy).", delta=-5)
            else:
                _add_neu(f"PCR {pcr:.3f} — balanced.")
    else:
        _add_neu("Option snapshot unavailable — skipping options bias.")

    # ---------------------------
    # 8) VIX (volatility background)
    # ---------------------------
    if vix is not None:
        if vix <= 13:
            _add_pos(f"VIX {vix:.2f} (low) — trend signals likely cleaner.", delta=4)
        elif vix >= 18:
            _add_neg(f"VIX {vix:.2f} (high) — market choppiness / risk of false breakouts.", delta=-4)
        else:
            _add_neu(f"VIX {vix:.2f} — moderate volatility.")
    else:
        _add_neu("VIX not available.")

    # ---------------------------
    # Finalize score and label
    # ---------------------------
    score = int(max(0, min(100, score)))

    # Decide label (preserve your previous rules)
    if engulf_buy and score >= 60:
        label = "ENGULFING_BUY"
    elif engulf_sell and score >= 60:
        label = "ENGULFING_SELL"
    elif score >= 65:
        label = "BUY"
    elif score <= 35:
        label = "SELL"
    else:
        label = "NEUTRAL"

    # ---------------------------
    # Build formatted reasoning string (Positive / Negative / Neutral) with details
    # ---------------------------
    parts = []
    if pos:
        parts.append("### ✅ Positive factors\n" + "\n".join(f"- {p}" for p in pos))
    if neg:
        parts.append("### ❌ Negative factors\n" + "\n".join(f"- {n}" for n in neg))
    if neu:
        parts.append("### ⚪ Neutral / Observations\n" + "\n".join(f"- {m}" for m in neu))

    parts.append(f"\n**Net Confidence Score:** {score}/100")
    parts.append(f"**Signal label:** {label}")

    reasoning = {
    "Positive": pos,
    "Negative": neg,
    "Neutral": neu
    }
    return label, reasoning, score

def plot_confidence_gauge(score):
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=score,
        title={'text': "Trade Confidence"},
        gauge={
            'axis': {'range': [0, 100]},
            'bar': {'color': "darkblue"},
            'steps': [
                {'range': [0, 30], 'color': "red"},
                {'range': [30, 60], 'color': "yellow"},
                {'range': [60, 100], 'color': "green"},
            ],
            'threshold': {'line': {'color': "black", 'width': 4}, 'thickness': 0.75, 'value': score}
        }
    ))
    st.plotly_chart(fig, use_container_width=True)

def fetch_option_oi_delta(atm, strikes_each_side=3):
    nfo_df = get_nifty_nfo_instruments()
    if nfo_df.empty or atm is None:
        return pd.DataFrame()

    symbols_map = build_atm_symbols(atm, nfo_df, strikes_each_side)
    rows = []

    for strike, m in symbols_map.items():
        ce_id, pe_id = m["CE"], m["PE"]
        ce_token, pe_token = m["CE_token"], m["PE_token"]

        # Current OI
        ce_oi = quote_oi_safe(ce_id, ce_token)
        pe_oi = quote_oi_safe(pe_id, pe_token)

        # Previous day's OI (intraday ΔOI)
        today = dt.date.today()
        try:
            ce_hist = kite.historical_data(ce_token,
                                           dt.datetime.combine(today - dt.timedelta(days=1), dt.time(9,15)),
                                           dt.datetime.combine(today - dt.timedelta(days=1), dt.time(15,30)),
                                           "day", oi=True)
            pe_hist = kite.historical_data(pe_token,
                                           dt.datetime.combine(today - dt.timedelta(days=1), dt.time(9,15)),
                                           dt.datetime.combine(today - dt.timedelta(days=1), dt.time(15,30)),
                                           "day", oi=True)
            ce_oi_prev = ce_hist[-1]["oi"] if ce_hist else None
            pe_oi_prev = pe_hist[-1]["oi"] if pe_hist else None
        except Exception:
            ce_oi_prev = pe_oi_prev = None

        ce_delta = ce_oi - ce_oi_prev if ce_oi and ce_oi_prev else None
        pe_delta = pe_oi - pe_oi_prev if pe_oi and pe_oi_prev else None

        rows.append({
            "Strike": strike,
            "CE_OI": ce_oi,
            "PE_OI": pe_oi,
            "ΔCE_OI": ce_delta,
            "ΔPE_OI": pe_delta
        })

    return pd.DataFrame(rows).sort_values("Strike").reset_index(drop=True)
def calc_max_pain(nfo_df):
    """
    Max Pain = strike where total loss of option writers is minimized.
    Uses CE and PE OI data.
    """
    if nfo_df.empty:
        return None

    strikes = sorted(nfo_df["strike"].unique())
    min_loss = float("inf")
    max_pain_strike = None

    for s in strikes:
        total_loss = 0
        for _, row in nfo_df.iterrows():
            if row["instrument_type"] == "CE":
                total_loss += max(row["strike"] - s, 0) * row.get("oi", 0)
            elif row["instrument_type"] == "PE":
                total_loss += max(s - row["strike"], 0) * row.get("oi", 0)
        if total_loss < min_loss:
            min_loss = total_loss
            max_pain_strike = s
    return max_pain_strike

def calc_fibonacci_levels(low, high):
    """
    Returns Fibonacci retracement and extension levels between low and high.
    Standard retracements: 23.6%, 38.2%, 50%, 61.8%, 78.6%
    Extensions: 161.8%, 261.8% (above high) and -61.8%, -161.8% (below low)
    """
    levels = {}
    diff = high - low
    # Retracements
    levels['Fib23.6'] = high - 0.236*diff
    levels['Fib38.2'] = high - 0.382*diff
    levels['Fib50.0'] = high - 0.5*diff
    levels['Fib61.8'] = high - 0.618*diff
    levels['Fib78.6'] = high - 0.786*diff
    # Extensions
    levels['Ext161.8'] = high + 1.618*diff
    levels['Ext261.8'] = high + 2.618*diff
    levels['Neg61.8'] = low - 0.618*diff
    levels['Neg161.8'] = low - 1.618*diff
    return levels

def get_price_action_levels(df, window=5):
    """
    Returns recent swing highs and lows using rolling window.
    """
    if df.empty:
        return [], []
    
    high = df['high']
    low  = df['low']
    
    swing_highs = df['high'][(high.shift(1) < high) & (high.shift(-1) < high)]
    swing_lows  = df['low'][(low.shift(1) > low) & (low.shift(-1) > low)]
    
    # Optional: take last N swings
    swing_highs = swing_highs.tail(window)
    swing_lows  = swing_lows.tail(window)
    
    return swing_highs.tolist(), swing_lows.tolist()

def add_fib_and_price_action(fig, fib_levels, swing_highs, swing_lows):
    # Fibonacci
    for name, level in fib_levels.items():
        fig.add_trace(go.Scatter(
            x=["Fib"],
            y=[level],
            mode="lines+text",
            line=dict(color="purple", dash="dot", width=1),
            text=[name],
            textposition="bottom right",
            name=f"Fib {name}"
        ))
    # Swing Highs
    for sh in swing_highs:
        fig.add_trace(go.Scatter(
            x=["Swing High"],
            y=[sh],
            mode="lines",
            line=dict(color="blue", dash="dash", width=1),
            name="Swing High"
        ))
    # Swing Lows
    for sl in swing_lows:
        fig.add_trace(go.Scatter(
            x=["Swing Low"],
            y=[sl],
            mode="lines",
            line=dict(color="red", dash="dash", width=1),
            name="Swing Low"
        ))
    return fig
import plotly.graph_objects as go

# ===== Helper Charts =====

def create_gauge_chart(score: int):
    """Gauge chart to show signal confidence (0-100)."""
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=score,
        number={'suffix': "%"},
        gauge={
            'axis': {'range': [0, 100]},
            'bar': {'color': "green" if score > 60 else "orange" if score > 40 else "red"},
            'steps': [
                {'range': [0, 40], 'color': "#ffcccc"},
                {'range': [40, 60], 'color': "#fff5cc"},
                {'range': [60, 100], 'color': "#ccffcc"}
            ]
        }
    ))
    fig.update_layout(height=220, margin=dict(l=10, r=10, t=40, b=10))
    return fig


def create_price_chart(df, pivots: dict):
    """Candlestick + SMA + VWAP + Pivot levels + Swing Highs/Lows + Fibonacci retracements."""
    fig = go.Figure()

    # Candlesticks
    fig.add_trace(go.Candlestick(
        x=df['date'],
        open=df['open'], high=df['high'],
        low=df['low'], close=df['close'],
        name="Price"
    ))

    # SMA
    if "SMA20" in df:
        fig.add_trace(go.Scatter(
            x=df['date'], y=df['SMA20'],
            line=dict(color="blue", width=1),
            name="SMA20"
        ))

    # VWAP
    if "VWAP" in df:
        fig.add_trace(go.Scatter(
            x=df['date'], y=df['VWAP'],
            line=dict(color="orange", width=1, dash="dot"),
            name="VWAP"
        ))

    # Pivot levels
    colors = {"P":"grey","R1":"green","S1":"red","R2":"green","S2":"red","R3":"green","S3":"red"}
    for level, price in pivots.items():
        fig.add_hline(y=price, line_dash="dot",
                      line_color=colors.get(level, "black"),
                      annotation_text=level, annotation_position="top right")

    # Swing Highs & Lows
    window = 5
    df["swing_high"] = df['high'][(df['high'].shift(window) < df['high']) &
                                  (df['high'].shift(-window) < df['high'])]
    df["swing_low"] = df['low'][(df['low'].shift(window) > df['low']) &
                                (df['low'].shift(-window) > df['low'])]

    fig.add_trace(go.Scatter(
        x=df['date'], y=df['swing_high'],
        mode="markers", marker=dict(color="red", size=8, symbol="triangle-down"),
        name="Swing Highs"
    ))
    fig.add_trace(go.Scatter(
        x=df['date'], y=df['swing_low'],
        mode="markers", marker=dict(color="green", size=8, symbol="triangle-up"),
        name="Swing Lows"
    ))

    # Fibonacci retracement (last visible swing)
    if not df.empty:
        high_price = df['high'].max()
        low_price = df['low'].min()
        diff = high_price - low_price
        fib_levels = {
            "0%": high_price,
            "23.6%": high_price - 0.236*diff,
            "38.2%": high_price - 0.382*diff,
            "50%": high_price - 0.5*diff,
            "61.8%": high_price - 0.618*diff,
            "100%": low_price
        }
        for lvl, val in fib_levels.items():
            fig.add_hline(y=val, line_dash="dash", line_color="purple",
                          annotation_text=f"Fib {lvl}", annotation_position="left")

    fig.update_layout(
        xaxis_rangeslider_visible=False,
        height=600,
        margin=dict(l=10, r=10, t=40, b=10)
    )
    return fig


def create_rsi_chart(df):
    """RSI chart (14-period)."""
    if "RSI" not in df:
        return None

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df['date'], y=df['RSI'],
        line=dict(color="purple", width=1.5),
        name="RSI(14)"
    ))

    # Overbought/Oversold lines
    fig.add_hline(y=70, line_color="red", line_dash="dot")
    fig.add_hline(y=30, line_color="green", line_dash="dot")

    fig.update_layout(
        title="RSI(14)",
        height=200,
        margin=dict(l=10, r=10, t=40, b=10),
        yaxis=dict(range=[0, 100])
    )
    return fig
# ========== Main Dashboard ==========

def main():
    st.title("📊 NIFTY Intraday Trade Dashboard (Zerodha API)")
    report_time = dt.datetime.now().strftime("%A, %d %B %Y %I:%M %p")
    st.markdown(f"**📅 Report generated on:** {report_time}")

    # -----------------------------
    # 1. Market Context & Index Data
    # -----------------------------
    spot = get_spot()
    vix  = get_vix()
    ohlc_y = last_trading_day_ref()
    if not ohlc_y:
        st.error("Could not fetch last trading day OHLC; cannot compute pivots.")
        st.stop()

    prev_close = float(ohlc_y["close"])
    prev_high  = float(ohlc_y["high"])
    prev_low   = float(ohlc_y["low"])

    token = get_nifty_token()
    df5 = fetch_intraday_5min(token, days=1)
    df5 = add_intraday_indicators(df5)

    day_open = df5.iloc[0]["open"] if not df5.empty else None
    first_bar = df5[df5["date"].dt.time >= dt.time(9,15)]
    first_high = float(first_bar.iloc[0]["high"]) if not first_bar.empty else None
    first_low  = float(first_bar.iloc[0]["low"]) if not first_bar.empty else None
    current_high = df5['high'].max() if not df5.empty else None
    current_low  = df5['low'].min() if not df5.empty else None
    change_pct   = ((spot - prev_close)/prev_close)*100 if spot and prev_close else None

    st.subheader("📌 1. Market Context & Index Data")
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Spot Price", f"{spot:.2f}" if spot else "—", 
                  delta=f"{change_pct:.2f}%" if change_pct is not None else None)
        st.metric("Previous Close", f"{prev_close:.2f}")
        st.metric("Opening Price", f"{day_open:.2f}" if day_open else "—")
    with col2:
        st.metric("Previous High/Low", f"{prev_high:.2f} / {prev_low:.2f}" if prev_high and prev_low else "—")
        st.metric("First 5-min High/Low", f"{first_high:.2f} / {first_low:.2f}" if first_high and first_low else "—")
        st.metric("Current Day High/Low", f"{current_high:.2f} / {current_low:.2f}" if current_high and current_low else "—")

    st.divider()

    # -----------------------------
    # 2. Derivatives Data (Options)
    # -----------------------------
    atm = int(round(spot / 50.0) * 50) if spot else None
    option_df, pcr = fetch_atm_option_snapshot_and_pcr(atm, strikes_each_side=3)
    oi_df = fetch_option_oi_delta(atm, strikes_each_side=3)
    max_pain = calc_max_pain(get_nifty_nfo_instruments())

    st.subheader("📈 2. Options & Derivatives Data")
    col1, col2 = st.columns(2)
    with col1:
        st.metric("ATM Strike", f"{atm}" if atm else "—")
        st.metric("PCR (ATM±3)", f"{pcr:.2f}" if pcr is not None else "—")
    with col2:
        st.metric("Max Pain Strike", f"{max_pain}" if max_pain else "—")
        st.metric("India VIX", f"{vix:.2f}" if vix else "—")

    if oi_df is not None and not oi_df.empty:
        st.dataframe(oi_df)

    if option_df is not None and not option_df.empty:
        st.subheader("ATM ±3 Option Snapshot (LTP & OI)")
        st.dataframe(option_df)

    st.divider()

    # -----------------------------
    # 3. Technical Indicators
    # -----------------------------
    st.subheader("📊 3. Technical Indicators")
    col1, col2 = st.columns(2)
    ema20 = df5['EMA20'].iloc[-1] if 'EMA20' in df5.columns else None
    ema50 = df5['EMA50'].iloc[-1] if 'EMA50' in df5.columns else None
    vwap = df5['VWAP'].iloc[-1] if 'VWAP' in df5.columns else None
    rsi  = df5['RSI'].iloc[-1]  if 'RSI' in df5.columns else None
    macd = df5['MACD'].iloc[-1] if 'MACD' in df5.columns else None

    with col1:
        st.metric("EMA20", f"{ema20:.2f}" if ema20 else "—")
        st.metric("EMA50", f"{ema50:.2f}" if ema50 else "—")
        st.metric("VWAP", f"{vwap:.2f}" if vwap else "—")
    with col2:
        st.metric("RSI(14)", f"{rsi:.2f}" if rsi else "—")
        st.metric("MACD", f"{macd:.2f}" if macd else "—")
        if 'BB_High' in df5.columns and 'BB_Low' in df5.columns:
            st.metric("Bollinger Bands", f"{df5['BB_Low'].iloc[-1]:.2f} - {df5['BB_High'].iloc[-1]:.2f}")
        else:
            st.metric("Bollinger Bands", "—")

    st.divider()

    # -----------------------------
    # 4. Support & Resistance
    # -----------------------------
    pivots = calc_pivots_from_ohlc(ohlc_y)
    first1h = df5[df5["date"].dt.time.between(dt.time(9,15), dt.time(10,15))]
    first1h_high = first1h['high'].max() if not first1h.empty else None
    first1h_low  = first1h['low'].min()  if not first1h.empty else None
    fib_levels = calc_fibonacci_levels(first1h_low, first1h_high) if first1h_low and first1h_high else {}
    swing_highs, swing_lows = get_price_action_levels(df5)

    st.subheader("🛡️ 4. Support & Resistance Levels")
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Pivot P", f"{pivots['P']:.2f}")
        st.metric("S1", f"{pivots['S1']:.2f}")
        st.metric("First 1H High/Low", f"{first1h_high:.2f} / {first1h_low:.2f}" if first1h_high and first1h_low else "—")
    with col2:
        st.metric("R1", f"{pivots['R1']:.2f}")
        st.metric("Recent Swings", f"H:{swing_highs} | L:{swing_lows}")

    st.divider()

    # -----------------------------
    # 5. Fundamental / Macro Sentiment
    # -----------------------------
    st.subheader("🌐 5. Fundamental & Macro Sentiment")
    col1, col2 = st.columns(2)
    with col1:
        st.metric("India VIX", f"{vix:.2f}" if vix else "—")
        st.metric("USD/INR", "—")   # placeholder
    with col2:
        st.metric("FII/DII Flows", "—")
        st.metric("Global Markets", "—")

    st.divider()
    # -----------------------------
    # Output / Signal Generation
    # -----------------------------
    signal, reasoning, score = trade_signal_logic(
        df5=df5, pivots=pivots, vix=vix, option_df=option_df, pcr=pcr,
        breakout_above=first_5min_breakout_levels(df5)[0],
        breakout_below=first_5min_breakout_levels(df5)[1],
        prev_high=prev_high, prev_low=prev_low,
        day_open=day_open, first_high=first_high, first_low=first_low
    )

    st.subheader("📌 Trade Signal & Confidence")
    detection_time = dt.datetime.now().strftime("%I:%M %p")
    breakout_price = df5["close"].iloc[-1] if not df5.empty else None

    # 🎨 Define colors for badge + signal box
    style_map = {
        "STRONG_BUY":  ("Bullish Signal", "#16a34a", "#dcfce7"),   # green
        "BUY":         ("Mild Bullish",   "#0284c7", "#e0f2fe"),   # blue
        "NEUTRAL":     ("Neutral Zone",   "#64748b", "#f1f5f9"),   # gray
        "SELL":        ("Mild Bearish",   "#f97316", "#fff7ed"),   # orange
        "STRONG_SELL": ("Bearish Signal", "#dc2626", "#fee2e2"),   # red
    }
    label, badge_color, bg_color = style_map.get(signal, ("Neutral Zone", "#64748b", "#f1f5f9"))

    # 🚩 Header Badge
    st.markdown(
        f"""
        <div style="display:inline-block; background-color:{badge_color}; 
                    color:white; padding:6px 14px; border-radius:20px; 
                    font-weight:bold; margin-bottom:8px;">
            {label}
        </div>
        """,
        unsafe_allow_html=True
    )

    # 🟩 Signal Box
    st.markdown(
        f"""
        <div style="background-color:{bg_color}; padding:20px; 
                    border-radius:12px; border:1px solid #ddd;">
        """,
        unsafe_allow_html=True
    )

    col1, col2 = st.columns([1,2])
    with col1:
        sig_map = {"STRONG_BUY":"🟢","BUY":"🔵","NEUTRAL":"⚪","SELL":"🟠","STRONG_SELL":"🔴"}
        st.markdown(f"### {sig_map.get(signal,'⚪')} **{signal.replace('_',' ')}**")
        st.metric("Score", score)
        st.metric("Breakout Price", f"{breakout_price:.2f}" if breakout_price else "—")
        st.metric("Detected", detection_time)

    with col2:
        st.markdown("### 📋 Analysis Summary")
        if reasoning.get("Positive"):
            st.markdown("**✅ Bullish Factors:**")
            for f in reasoning["Positive"]:
                st.markdown(f"• {f}")
        if reasoning.get("Negative"):
            st.markdown("**❌ Bearish Factors:**")
            for f in reasoning["Negative"]:
                st.markdown(f"• {f}")
        if reasoning.get("Neutral"):
            st.markdown("**ℹ️ Neutral Factors:**")
            for f in reasoning["Neutral"]:
                st.markdown(f"• {f}")

    st.markdown("</div>", unsafe_allow_html=True)

    # Confidence gauge
    gauge_fig = create_gauge_chart(score)
    st.plotly_chart(gauge_fig, use_container_width=True)

if __name__ == "__main__":
    main()

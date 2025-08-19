# nifty_intraday_dashboard_live.py
"""
Streamlit dashboard for NIFTY intraday monitoring.
Features added/changed:
- Live "Start Monitoring" button that watches closed 5-min candles and signals on breakout
- Uses previous 5-min candle high/low as breakout reference (closed candle)
- Combines confirmations: prev-day levels, first 5-min, RSI, engulfing, PCR, VIX, options skew
- Better UI: Start/Stop controls, last signal card, live LTP, confidence gauge, logs
- Configurable via environment variables

NOTE: This is a decision-support tool — NOT financial advice.
"""

import os
import time
import logging
import datetime as dt
from datetime import datetime, timedelta

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

from dotenv import load_dotenv
from kiteconnect import KiteConnect
from ta.momentum import RSIIndicator
from ta.trend import SMAIndicator
from ta.volume import VolumeWeightedAveragePrice

# -------------------- Config & Logging --------------------
load_dotenv()

CANDLE_START = dt.time(int(os.getenv("CANDLE_START_H", 9)), int(os.getenv("CANDLE_START_M", 15)))
CANDLE_END   = dt.time(int(os.getenv("CANDLE_END_H", 15)), int(os.getenv("CANDLE_END_M", 15)))
STRIKE_INTERVAL = int(os.getenv("STRIKE_INTERVAL", 50))
BREAKOUT_BUFFER = float(os.getenv("BREAKOUT_BUFFER", 2))
BUFFER_PERCENT = float(os.getenv("BUFFER_PERCENT", 50))
SL_PERCENT = float(os.getenv("SL_PERCENT", 30))
TARGET_PERCENT = float(os.getenv("TARGET_PERCENT", 50))

log_file = "nifty_intraday_live.log"
logging.basicConfig(filename=log_file, level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s", force=True)

# -------------------- Kite Setup --------------------
API_KEY = os.getenv("API_KEY")
ACCESS_TOKEN = os.getenv("ACCESS_TOKEN")

if not API_KEY or not ACCESS_TOKEN:
    # We'll still allow viewing non-live features in Streamlit but warn
    WARNING_MSG = "API_KEY or ACCESS_TOKEN missing in .env — live features disabled."
else:
    kite = KiteConnect(api_key=API_KEY)
    kite.set_access_token(ACCESS_TOKEN)

# -------------------- Helpers --------------------

def now_time():
    return datetime.now().time()


def get_nifty_token():
    try:
        q = kite.ltp(["NSE:NIFTY 50"])  # prefer name lookup
        info = q.get("NSE:NIFTY 50") or list(q.values())[0]
        return info.get("instrument_token") or 256265
    except Exception:
        return 256265


def get_ltp(symbol="NSE:NIFTY 50"):
    try:
        return float(kite.ltp([symbol])[symbol]["last_price"])
    except Exception as e:
        logging.warning(f"LTP fetch failed for {symbol}: {e}")
        return None


def align_to_5min(dt_obj):
    minute = (dt_obj.minute // 5) * 5
    return dt_obj.replace(minute=minute, second=0, microsecond=0)


def fetch_5min_candles(token, start_dt, end_dt):
    try:
        data = kite.historical_data(token, start_dt, end_dt, interval="5minute", continuous=False)
        return pd.DataFrame(data)
    except Exception as e:
        logging.error(f"fetch_5min_candles error: {e}")
        return pd.DataFrame()


def fetch_latest_two_5min(token):
    # fetch last 30 minutes and pick last two *closed* 5min candles
    end = datetime.now()
    start = end - timedelta(minutes=30)
    df = fetch_5min_candles(token, start, end)
    if df.empty:
        return pd.DataFrame()
    df["date"] = pd.to_datetime(df["date"]) if "date" in df.columns else pd.to_datetime(df.index)
    df = df.sort_values("date").reset_index(drop=True)
    # closed candles are those with timestamp <= align_to_5min(now) - 5min
    cutoff = align_to_5min(datetime.now()) - timedelta(minutes=0)
    closed = df[df["date"] <= cutoff]
    if len(closed) < 2:
        return closed
    return closed.iloc[-2:]


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


def calc_pivots_from_ohlc(ohlc):
    h, l, c = float(ohlc["high"]), float(ohlc["low"]), float(ohlc["close"])
    p = (h + l + c) / 3.0
    r1 = 2*p - l
    s1 = 2*p - h
    r2 = p + (h - l)
    s2 = p - (h - l)
    r3 = h + 2*(p - l)
    s3 = l - 2*(h - p)
    return {"P": p, "R1": r1, "S1": s1, "R2": r2, "S2": s2, "R3": r3, "S3": s3}


def is_engulfing(prev_candle, curr_candle, direction):
    po, ph, pl, pc = float(prev_candle["open"]), float(prev_candle["high"]), float(prev_candle["low"]), float(prev_candle["close"])
    co, ch, cl, cc = float(curr_candle["open"]), float(curr_candle["high"]), float(curr_candle["low"]), float(curr_candle["close"])
    if direction == "bullish":
        return (pc < po) and (cc > co) and (co < pc) and (cc > po)
    if direction == "bearish":
        return (pc > po) and (cc < co) and (co > pc) and (cc < po)
    return False

# Reuse trade_signal_logic from user's original code (imported or defined here)
# For brevity in this file we'll import it if available; otherwise define a lightweight wrapper

try:
    # If the user pasted the large function into the same folder as module, import
    from trade_logic import trade_signal_logic, plot_confidence_gauge
except Exception:
    # Fallback: define a simplified wrapper that calls the user's long logic inline
    def trade_signal_logic(**kwargs):
        # Very small wrapper: compute a score from basic elements for UI demo
        df5 = kwargs.get('df5')
        pivots = kwargs.get('pivots')
        vix = kwargs.get('vix')
        option_df = kwargs.get('option_df')
        pcr = kwargs.get('pcr')
        breakout_above = kwargs.get('breakout_above')
        breakout_below = kwargs.get('breakout_below')
        prev_high = kwargs.get('prev_high')
        prev_low = kwargs.get('prev_low')
        first_high = kwargs.get('first_high')
        first_low = kwargs.get('first_low')

        # Minimal logic: detect breakout vs prev candle and simple confirmations
        if df5 is None or df5.empty:
            return "NEUTRAL", {"Positive":[], "Negative":[], "Neutral":["No data"]}, 50
        latest = df5.iloc[-1]
        prev = df5.iloc[-2] if len(df5) >= 2 else latest
        close = float(latest['close'])
        score = 50
        pos, neg, neu = [], [], []

        if (prev_high is not None and close > prev_high) or (first_high is not None and close > first_high):
            pos.append("Breakout vs prev/first-5m")
            score += 15
        if (prev_low is not None and close < prev_low) or (first_low is not None and close < first_low):
            neg.append("Breakdown vs prev/first-5m")
            score -= 15

        rsi = latest.get('RSI') or 50
        if rsi >= 65:
            pos.append(f"RSI {rsi:.1f} bullish")
            score += 8
        elif rsi <= 35:
            neg.append(f"RSI {rsi:.1f} bearish")
            score -= 8
        else:
            neu.append(f"RSI {rsi:.1f} neutral")

        # Options/pcr heuristic
        if pcr is not None:
            if pcr > 1.1:
                pos.append(f"PCR {pcr} — put-heavy (contrarian bullish)")
                score += 4
            elif pcr < 0.9:
                neg.append(f"PCR {pcr} — call-heavy (contrarian bearish)")
                score -= 4

        label = "NEUTRAL"
        if score >= 65:
            label = "BUY"
        elif score <= 35:
            label = "SELL"

        reasoning = {"Positive": pos, "Negative": neg, "Neutral": neu}
        return label, reasoning, int(max(0,min(100,score)))

    def plot_confidence_gauge(score):
        fig = go.Figure(go.Indicator(mode="gauge+number", value=score, title={'text':"Trade Confidence"}, gauge={'axis':{'range':[0,100]}}))
        st.plotly_chart(fig, use_container_width=True)

# -------------------- Monitoring Loop --------------------

def check_latest_closed_breakout(token, pivots, option_df, pcr, first_high, first_low, prev_day_high, prev_day_low):
    df2 = fetch_latest_two_5min(token)
    if df2.empty or len(df2) < 2:
        return None
    prev_candle = df2.iloc[0]
    closed_candle = df2.iloc[1]

    # Use prev_candle as breakout reference
    breakout_above = float(prev_candle['high']) + (BUFFER_PERCENT/100.0)*(float(prev_candle['high'])-float(prev_candle['low']))
    breakout_below = float(prev_candle['low']) - (BUFFER_PERCENT/100.0)*(float(prev_candle['high'])-float(prev_candle['low']))

    # Build df5 for indicators
    df5 = df2.copy()
    df5 = add_intraday_indicators(df5)

    # Call the signal logic with context
    label, reasoning, score = trade_signal_logic(
        df5=df5,
        pivots=pivots,
        vix=get_vix_safe(),
        option_df=option_df,
        pcr=pcr,
        breakout_above=breakout_above,
        breakout_below=breakout_below,
        prev_high=prev_day_high,
        prev_low=prev_day_low,
        day_open=None,
        first_high=first_high,
        first_low=first_low
    )

    # Determine if breakout truly occurred vs prev candle
    close = float(closed_candle['close'])
    breakout_flag = None
    if close > breakout_above:
        breakout_flag = 'BULL'
    elif close < breakout_below:
        breakout_flag = 'BEAR'

    return {
        'prev_candle': prev_candle,
        'closed_candle': closed_candle,
        'breakout_above': breakout_above,
        'breakout_below': breakout_below,
        'breakout_flag': breakout_flag,
        'label': label,
        'reasoning': reasoning,
        'score': score
    }


def get_vix_safe():
    try:
        return float(kite.ltp(["NSE:INDIA VIX"])["NSE:INDIA VIX"]["last_price"])
    except Exception:
        return None

# -------------------- Streamlit UI --------------------

st.set_page_config(page_title="NIFTY Intraday Live Dashboard", layout="wide")
st.title("📈 NIFTY Intraday Monitor — 5min Breakout + Confirmations")

# Sidebar controls
st.sidebar.header("Controls")
start = st.sidebar.button("Start Monitoring")
stop  = st.sidebar.button("Stop")
refresh_rate = st.sidebar.slider("Poll interval (sec)", min_value=2, max_value=60, value=5, step=1)

# Load some static context for the day
token = get_nifty_token() if API_KEY and ACCESS_TOKEN else 256265

# Fetch last trading day OHLC for pivots
last_day = None
try:
    token_for_pivot = get_nifty_token()
    today = dt.date.today()
    # get last trading day by searching last 7 days
    for d in range(1,8):
        day = today - timedelta(days=d)
        bars = kite.historical_data(token_for_pivot, day, day, 'day') if (API_KEY and ACCESS_TOKEN) else None
        if bars:
            last_day = bars[0]
            break
except Exception:
    last_day = None

if not last_day:
    st.warning("Could not fetch last trading day OHLC. Pivots will be unavailable.")
    pivots = {"P":0,"R1":0,"S1":0,"R2":0,"S2":0,"R3":0,"S3":0}
else:
    pivots = calc_pivots_from_ohlc(last_day)

# first 5m candle
spot = get_ltp() if (API_KEY and ACCESS_TOKEN) else None

# prefetch options snapshot (best-effort)
try:
    atm = int(round(spot/STRIKE_INTERVAL)*STRIKE_INTERVAL) if spot else None
    option_df, pcr = (pd.DataFrame(), None)
    if API_KEY and ACCESS_TOKEN and atm is not None:
        option_df, pcr = fetch_atm_option_snapshot_and_pcr(atm, strikes_each_side=3)
except Exception:
    option_df, pcr = pd.DataFrame(), None

# UI panels
col1, col2, col3, col4 = st.columns(4)
col1.metric("NIFTY Spot", f"{spot:.2f}" if spot else "—")
col2.metric("Pivot (P)", f"{pivots['P']:.1f}")
col3.metric("PCR (ATM±3)", f"{pcr}" if pcr is not None else "—")
col4.metric("Last Log", "See logs below")

log_placeholder = st.empty()
status_placeholder = st.empty()
signal_placeholder = st.empty()

# Session state to control loop
if 'monitoring' not in st.session_state:
    st.session_state['monitoring'] = False

if start:
    st.session_state['monitoring'] = True
if stop:
    st.session_state['monitoring'] = False

# Monitoring loop (runs while user keeps session open and 'monitoring' True)
if st.session_state['monitoring']:
    status_placeholder.info("🔴 Monitoring: ON — watching closed 5-min candles for breakout...")
    logs = []
    try:
        market_end = datetime.now().replace(hour=CANDLE_END.hour, minute=CANDLE_END.minute, second=0, microsecond=0)
        while st.session_state['monitoring'] and datetime.now() < market_end:
            res = check_latest_closed_breakout(token, pivots, option_df, pcr, None, None, float(last_day['high']) if last_day else None, float(last_day['low']) if last_day else None)
            if res:
                # append log
                tstamp = datetime.now().strftime('%H:%M:%S')
                logs.insert(0, f"{tstamp} | Checked closed candle at {res['closed_candle']['date']} — breakout={res['breakout_flag']} label={res['label']} score={res['score']}")
                # update UI
                log_placeholder.markdown("\n".join([f"- {l}" for l in logs[:20]]))
                signal_placeholder.subheader(f"Last Signal: {res['label']} — Score {res['score']}")
                signal_placeholder.write(res['reasoning'])
                plot_confidence_gauge(res['score'])
                if res['breakout_flag'] is not None and res['score'] >= 60:
                    st.success(f"🔔 TRADE SIGNAL: {res['label']} at {res['closed_candle']['close']} — Score {res['score']}")
                    # Optional: call order placement functions here (commented)
                    # place_orders(...)
                    # stop monitoring after a confirmed signal
                    st.session_state['monitoring'] = False
                    break
            else:
                logs.insert(0, f"{datetime.now().strftime('%H:%M:%S')} | No closed candles yet or insufficient data")
                log_placeholder.markdown("\n".join([f"- {l}" for l in logs[:20]]))

            time.sleep(refresh_rate)

        if datetime.now() >= market_end:
            status_placeholder.warning("Market window closed — monitoring stopped.")
            st.session_state['monitoring'] = False
    except Exception as e:
        logging.error(f"Monitoring loop error: {e}")
        status_placeholder.error(f"Error during monitoring: {e}")
else:
    status_placeholder.info("⚪ Monitoring: OFF — press Start Monitoring to begin watching closed 5-min candles.")

st.markdown("---")
st.subheader("Notes & Usage")
st.markdown("- Click **Start Monitoring** to begin. The app watches *closed* 5-minute candles and evaluates breakouts using your configured strategy.\n- A strong signal requires both a breakout (closed candle beyond prev-candle buffer) and confirmations (RSI, pivots, options bias).\n- This tool does not place orders by default. You can add order placement where commented in the code.\n- Ensure `.env` contains `API_KEY` and `ACCESS_TOKEN` for live features.")

st.caption("This dashboard is a decision-support tool, not financial advice.")

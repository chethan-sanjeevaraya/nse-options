# nifty_intraday_dashboard.py
import os
import datetime as dt
import time
import math

import streamlit as st
import pandas as pd
import numpy as np
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

# ========== App ==========

def main():
    st.title("📊 NIFTY Intraday Trade Dashboard (Zerodha API)")

    # 1) Spot & VIX
    spot = get_spot()
    vix  = get_vix()

    # 2) Intraday 5-min and indicators
    token = get_nifty_token()
    df5 = fetch_intraday_5min(token, days=1)
    df5 = add_intraday_indicators(df5)

    # 3) Pivots from last trading daily OHLC
    ohlc_y = last_trading_day_ref()
    if not ohlc_y:
        st.error("Could not fetch last trading day OHLC; cannot compute pivots.")
        st.stop()
    pivots = calc_pivots_from_ohlc(ohlc_y)

    # --- Previous Day Levels
    prev_high = float(ohlc_y["high"])
    prev_low  = float(ohlc_y["low"])
    prev_close = float(ohlc_y["close"])

    # --- Day Open Price
    day_open = df5.iloc[0]["open"] if not df5.empty else None

    # --- First 5-min Candle High & Low
    first_bar = df5[df5["date"].dt.time >= dt.time(9,15)]
    if not first_bar.empty:
        first_high = float(first_bar.iloc[0]["high"])
        first_low  = float(first_bar.iloc[0]["low"])
    else:
        first_high, first_low = None, None
    # --- Key Levels Panel
    st.subheader("Key Levels")
    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("Prev High", f"{prev_high:.2f}" if prev_high else "-")
    col2.metric("Prev Low", f"{prev_low:.2f}" if prev_low else "-")
    col3.metric("Day Open", f"{day_open:.2f}" if day_open else "-")
    col4.metric("1st 5m High", f"{first_high:.2f}" if first_high else "-")
    col5.metric("1st 5m Low", f"{first_low:.2f}" if first_low else "-")
    if first_high and first_low:
        col4.metric("1st 5m High/Low", f"{first_high:.1f} / {first_low:.1f}")

    # 4) First 5-min breakout levels
    breakout_above, breakout_below = first_5min_breakout_levels(df5, buffer_pts=5.0)

    # 5) Option snapshot (ATM ± 3) and PCR
    atm = int(round(spot / 50.0) * 50) if spot else None
    option_df, pcr = fetch_atm_option_snapshot_and_pcr(atm, strikes_each_side=3)

    # --- Snapshot panel
    st.subheader("Market Snapshot")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("NIFTY Spot", f"{spot:.2f}" if spot else "—")
    col2.metric("India VIX", f"{vix:.2f}" if vix is not None else "—")
    col3.metric("Pivot (P)", f"{pivots['P']:.1f}")
    col4.metric("PCR (ATM±3)", f"{pcr}" if pcr is not None else "—")

    st.write(
        f"**S/R** — S1: {pivots['S1']:.1f}, S2: {pivots['S2']:.1f}, S3: {pivots['S3']:.1f} | "
        f"R1: {pivots['R1']:.1f}, R2: {pivots['R2']:.1f}, R3: {pivots['R3']:.1f}"
    )
    if breakout_above and breakout_below:
        st.caption(f"First-bar breakout levels → Above: {breakout_above:.1f} | Below: {breakout_below:.1f}")

    # Option snapshot table (compact)
    if option_df is not None and not option_df.empty:
        st.subheader("ATM ± 3 Option Snapshot (LTP & OI)")
        show_cols = ["Strike", "CE_LTP", "PE_LTP", "CE_OI", "PE_OI"]
        st.dataframe(option_df[show_cols])

    # 6) Trade signal + reasoning + confidence
    signal, reasoning, score = trade_signal_logic(
        df5=df5,
        pivots=pivots,
        vix=vix,
        option_df=option_df,
        pcr=pcr,
        breakout_above=breakout_above,
        breakout_below=breakout_below,
        prev_high=prev_high,
        prev_low=prev_low,
        day_open=day_open,
        first_high=first_high,
        first_low=first_low
    )

    # --- Show Reasoning & Logs ---
    st.subheader("📋 Reasoning & Logs")
    for category, logs in reasoning.items():
        if logs:  # Only show non-empty categories
            if category == "Positive":
                color = "🟢"
            elif category == "Negative":
                color = "🔴"
            else:
                color = "⚪"
            st.markdown(f"**{color} {category} Signals:**")
            for i, log in enumerate(logs, 1):
                st.markdown(f"- {i}. {log}")

    # --- Final Summary Card ---
    st.markdown("---")
    st.subheader("📌 Final Summary")

    detection_time = dt.datetime.now().strftime("%I:%M %p")  # readable timestamp
    breakout_price = df5["close"].iloc[-1]  # using the latest market price as breakout level

    if score > 1:
        st.success(f"✅ **Strong Bullish Breakout Detected**\n\n"
                f"**Signal:** {signal}\n"
                f"**Breakout Price:** {breakout_price}\n"
                f"**Score:** {score}\n"
                f"**Detected At:** {detection_time}")
    elif score == 1:
        st.info(f"📈 **Weak Bullish Trend**\n\n"
                f"**Signal:** {signal}\n"
                f"**Breakout Price:** {breakout_price}\n"
                f"**Score:** {score}\n"
                f"**Detected At:** {detection_time}")
    elif score < -1:
        st.error(f"❌ **Strong Bearish Breakout Detected**\n\n"
                f"**Signal:** {signal}\n"
                f"**Breakout Price:** {breakout_price}\n"
                f"**Score:** {score}\n"
                f"**Detected At:** {detection_time}")
    elif score == -1:
        st.warning(f"📉 **Weak Bearish Trend**\n\n"
                f"**Signal:** {signal}\n"
                f"**Breakout Price:** {breakout_price}\n"
                f"**Score:** {score}\n"
                f"**Detected At:** {detection_time}")
    else:
        st.info(f"⚖️ **Neutral / No Breakout**\n\n"
                f"**Signal:** {signal}\n"
                f"**Breakout Price:** {breakout_price}\n"
                f"**Score:** {score}\n"
                f"**Detected At:** {detection_time}")

        st.markdown(f"**Confidence Score:** {score}/100")
        plot_confidence_gauge(score)

    st.caption("This dashboard is a decision-support tool, not financial advice.")

if __name__ == "__main__":
    main()

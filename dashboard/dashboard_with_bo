# nifty_intraday_dashboard.py

import os
import datetime as dt
import pandas as pd
import numpy as np
import streamlit as st
import plotly.graph_objects as go

from kiteconnect import KiteConnect
from dotenv import load_dotenv
from ta.momentum import RSIIndicator
from ta.trend import SMAIndicator
from ta.volume import VolumeWeightedAveragePrice
from bs4 import BeautifulSoup
import requests
import time

# -----------------------------
# Load credentials
# -----------------------------
load_dotenv()

API_KEY = os.getenv("API_KEY")
ACCESS_TOKEN = os.getenv("ACCESS_TOKEN")

if not API_KEY or not ACCESS_TOKEN:
    st.error("API_KEY or ACCESS_TOKEN missing in .env. Add them and restart the app.")
    st.stop()

kite = KiteConnect(api_key=API_KEY)
kite.set_access_token(ACCESS_TOKEN)

st.set_page_config(page_title="NIFTY Intraday Dashboard", layout="wide")

# -----------------------------
# Helper Functions
# -----------------------------
def fetch_option_chain():
    try:
        insts = kite.instruments("NFO")
        df = pd.DataFrame(insts)
        df = df[df["name"] == "NIFTY"]

        expiry = df["expiry"].min()
        spot = kite.ltp("NSE:NIFTY 50")["NSE:NIFTY 50"]["last_price"]
        atm_strike = int(round(spot / 50) * 50)

        calls = df[(df["strike"] == atm_strike) & (df["instrument_type"] == "CE") & (df["expiry"] == expiry)]
        puts = df[(df["strike"] == atm_strike) & (df["instrument_type"] == "PE") & (df["expiry"] == expiry)]

        if calls.empty or puts.empty:
            return None

        ce_symbol = calls.iloc[0]["tradingsymbol"]
        pe_symbol = puts.iloc[0]["tradingsymbol"]

        quotes = kite.quote([f"NFO:{ce_symbol}", f"NFO:{pe_symbol}"])

        ce_ltp = quotes[f"NFO:{ce_symbol}"]["last_price"]
        pe_ltp = quotes[f"NFO:{pe_symbol}"]["last_price"]

        option_df = pd.DataFrame([{
            "Strike": atm_strike,
            "CE_Symbol": ce_symbol,
            "PE_Symbol": pe_symbol,
            "CE_LTP": ce_ltp,
            "PE_LTP": pe_ltp,
            "PCR": round(pe_ltp / ce_ltp, 2) if ce_ltp > 0 else None
        }])
        return option_df

    except Exception as e:
        st.error(f"Error fetching option chain: {e}")
        return None


def fetch_vix():
    try:
        return kite.ltp("NSE:INDIA VIX")["NSE:INDIA VIX"]["last_price"]
    except:
        return None


def fetch_fii_dii():
    try:
        url = "https://www.moneycontrol.com/stocks/marketstats/fii_dii_activity/index.php"
        dfs = pd.read_html(url)
        df = dfs[0]
        fii_net = float(df.iloc[0]["FII"].replace(",", ""))
        dii_net = float(df.iloc[0]["DII"].replace(",", ""))
        return fii_net, dii_net
    except:
        return 0, 0


def fetch_nifty_history(days=5, interval="5minute"):
    try:
        instrument_token = 256265  # NIFTY 50
        to_date = dt.datetime.today()
        from_date = to_date - dt.timedelta(days=days)
        data = kite.historical_data(instrument_token, from_date, to_date, interval)
        return pd.DataFrame(data)
    except:
        return pd.DataFrame()


def calc_support_resistance(df, lookback=20):
    if df.empty:
        return None, None
    recent = df.tail(lookback)
    return recent["low"].min(), recent["high"].max()


def breakout_strategy(kite, token, breakout_buffer=5):
    start = dt.datetime.now().replace(hour=9, minute=15, second=0, microsecond=0)
    end = start + dt.timedelta(minutes=5)
    bars = kite.historical_data(token, start, end, interval="5minute")

    if not bars:
        return None, "No 9:15 candle data available."

    first_candle = bars[0]
    breakout_above = first_candle["high"] + breakout_buffer
    breakout_below = first_candle["low"] - breakout_buffer

    ltp = kite.ltp([token])[str(token)]["last_price"]

    if ltp > breakout_above:
        return "BUY", f"Price {ltp} broke above {breakout_above}."
    elif ltp < breakout_below:
        return "SELL", f"Price {ltp} broke below {breakout_below}."
    else:
        return "WAIT", f"No breakout yet. Price {ltp} consolidating between {breakout_below} – {breakout_above}."


def trade_signal_logic(df, option_data, vix, fii_net, kite, nifty_token):
    if option_data is None or vix is None:
        return "NO DATA", "Insufficient market data."

    breakout_signal, breakout_reason = breakout_strategy(kite, nifty_token)

    if breakout_signal == "WAIT":
        return "NO TRADE", breakout_reason

    # Confirmation
    pcr = option_data.iloc[0]["PCR"]
    analysis = [breakout_reason, f"PCR={pcr}, VIX={vix}, FII={fii_net}"]

    if breakout_signal == "BUY":
        if pcr < 1 and vix < 15 and fii_net > 0:
            return "BUY CONFIRMED", " | ".join(analysis)
        else:
            return "BUY WEAK", "Breakout detected but weak conditions: " + " | ".join(analysis)

    elif breakout_signal == "SELL":
        if pcr > 1 and vix > 15 and fii_net < 0:
            return "SELL CONFIRMED", " | ".join(analysis)
        else:
            return "SELL WEAK", "Breakout detected but weak conditions: " + " | ".join(analysis)

    return "NO TRADE", breakout_reason


# ----------------------
# Streamlit Dashboard
# ----------------------
st.title("📊 NIFTY Intraday Trade Dashboard")

st.sidebar.header("Settings")
use_manual = st.sidebar.checkbox("Manual Input Mode", value=False)

if use_manual:
    ce_ltp = st.sidebar.number_input("CE LTP", value=120.0)
    pe_ltp = st.sidebar.number_input("PE LTP", value=110.0)
    vix = st.sidebar.number_input("India VIX", value=12.0)
    fii_net = st.sidebar.number_input("FII Net Flow", value=-5000.0)

    option_data = pd.DataFrame({
        "Strike": [26500],
        "CE_LTP": [ce_ltp],
        "PE_LTP": [pe_ltp],
        "PCR": [round(pe_ltp / ce_ltp, 2)]
    })
else:
    option_data = fetch_option_chain()
    vix = fetch_vix()
    fii_net, dii_net = fetch_fii_dii()

df = fetch_nifty_history(days=5)
support, resistance = calc_support_resistance(df)

# Signal & Explanation
signal, reason = trade_signal_logic(df, option_data, vix, fii_net, kite, 256265)

st.subheader("Trade Signal")
st.metric("Signal", signal)
st.write("📝 Analysis:", reason)

if option_data is not None:
    st.subheader("Option Chain Snapshot")
    st.dataframe(option_data)

st.write("📌 VIX:", vix)
st.write("📌 FII Net:", fii_net)
st.write("📌 Support:", support)
st.write("📌 Resistance:", resistance)

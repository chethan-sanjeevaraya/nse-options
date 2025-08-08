# nifty_intraday_dashboard.py
import os
import datetime as dt
from functools import lru_cache

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from kiteconnect import KiteConnect
from dotenv import load_dotenv
import requests
from bs4 import BeautifulSoup

# technical lib
from ta.momentum import RSIIndicator
from ta.trend import SMAIndicator
from ta.volume import VolumeWeightedAveragePrice

# ----------------------------
# CONFIG / ENV
# ----------------------------
st.set_page_config(page_title="NIFTY Intraday Trade Dashboard", layout="wide")
load_dotenv(".env")

API_KEY = os.getenv("API_KEY")
ACCESS_TOKEN = os.getenv("ACCESS_TOKEN")  # set by your login flow
# Optionally set API_SECRET if you need to re-generate sessions (not required here)

if not API_KEY or not ACCESS_TOKEN:
    st.error("API_KEY or ACCESS_TOKEN is missing in .env. Add them and restart the app.")
    st.stop()

kite = KiteConnect(api_key=API_KEY)
kite.set_access_token(ACCESS_TOKEN)

# ----------------------------
# Helper functions
# ----------------------------
def next_weekly_expiry(reference_date=None):
    """Return next Thursday date (weekly expiry). If today is Thursday before expiry cut-off, return today."""
    if reference_date is None:
        reference_date = dt.date.today()
    # find next Thursday (weekday() : Mon=0 .. Sun=6; Thu = 3)
    days_ahead = (3 - reference_date.weekday()) % 7
    expiry = reference_date + dt.timedelta(days=days_ahead)
    return expiry

@st.cache_data(ttl=600)
def fetch_5min_candles_for_today():
    """Fetch 5-minute candles for NIFTY using historical_data.
       Uses index instrument token for NIFTY 50 (common mapping).
       If token lookup fails, fallback to using 'NSE:NIFTY 50' with available methods.
    """
    today = dt.date.today()
    start = dt.datetime.combine(today, dt.time(9, 15))
    end = dt.datetime.combine(today, dt.time(15, 30))
    # Common NIFTY instrument token (may vary across accounts/market data permissions)
    # We'll try to obtain token by calling kite.ltp for "NSE:NIFTY 50" and reading instrument_token
    try:
        ltp_quote = kite.ltp(["NSE:NIFTY 50"])
        nifty_info = ltp_quote.get("NSE:NIFTY 50") or list(ltp_quote.values())[0]
        token = nifty_info.get("instrument_token")
    except Exception:
        token = None

    if not token:
        # common fallback token used earlier: 256265 — but not guaranteed
        token = 256265

    candles = kite.historical_data(instrument_token=token, from_date=start, to_date=end, interval="5minute")
    df = pd.DataFrame(candles)
    if df.empty:
        raise ValueError("No 5-min candle data returned")
    # normalize df columns to lower-case keys used later
    df.columns = [c.lower() for c in df.columns]
    df["date"] = pd.to_datetime(df["date"])
    df.set_index("date", inplace=True)
    return df

def safe_series(series):
    """Ensure series is 1-D and numeric"""
    s = pd.Series(series).astype(float).squeeze()
    return s

def compute_technical_indicators(df):
    """Compute RSI, SMA20, SMA50, VWAP on provided df (expects columns open, high, low, close, volume)"""
    out = {}
    close = safe_series(df["close"])
    high = safe_series(df["high"])
    low = safe_series(df["low"])
    vol = safe_series(df.get("volume", pd.Series(np.zeros(len(df)))))

    # RSI
    rsi = RSIIndicator(close=close, window=14).rsi()
    out["RSI"] = float(rsi.iloc[-1])

    # SMA 20/50
    sma20 = SMAIndicator(close=close, window=20).sma_indicator()
    sma50 = SMAIndicator(close=close, window=50).sma_indicator()
    out["SMA20"] = float(sma20.iloc[-1])
    out["SMA50"] = float(sma50.iloc[-1])

    # VWAP (use typical period same as df length)
    vwap = VolumeWeightedAveragePrice(high=high, low=low, close=close, volume=vol, window=len(df)).volume_weighted_average_price()
    out["VWAP"] = float(vwap.iloc[-1]) if not vwap.isna().all() else None

    # current close
    out["Close"] = float(close.iloc[-1])
    out["High"] = float(high.max())
    out["Low"] = float(low.min())
    out["Open"] = float(df["open"].iloc[0])
    return out

def get_pivots_from_prices(high, low, close):
    pivot = (high + low + close) / 3.0
    s1 = 2 * pivot - high
    r1 = 2 * pivot - low
    s2 = pivot - (high - low)
    r2 = pivot + (high - low)
    return {"pivot": pivot, "s1": s1, "s2": s2, "r1": r1, "r2": r2}

@st.cache_data(ttl=300)
def instruments_nfo():
    """Return instruments list for NFO (options/futures). Cached to avoid repeated download."""
    instruments = kite.instruments("NFO")
    df = pd.DataFrame(instruments)
    return df

def find_weekly_option_symbols(atm_strike=None, strikes_range=5):
    """Find option tradingsymbols (CE/PE) for weekly expiry around atm_strike.
       Returns lists of CE symbols and PE symbols (fully-qualified: 'NFO:XXX').
    """
    instr = instruments_nfo()
    # get next weekly expiry date string in ISO that appears in tradingsymbol or expiry column
    expiry = next_weekly_expiry()
    # Normalize expiry formatting in instruments - try 'expiry' column exists
    if "expiry" in instr.columns:
        instr["expiry_date"] = pd.to_datetime(instr["expiry"]).dt.date
        weekly_instr = instr[instr["expiry_date"] == expiry]
    else:
        # fallback: filter by tradingsymbol containing expiry string (may not be reliable)
        weekly_instr = instr

    # Filter to options for NIFTY (tradingsymbol contains 'NIFTY' and endswith CE/PE)
    weekly_options = weekly_instr[weekly_instr["tradingsymbol"].str.contains("NIFTY", na=False)]
    # select strike list
    if atm_strike is None:
        # if atm not given, try to use current LTP
        try:
            nifty_quote = kite.ltp(["NSE:NIFTY 50"])
            nifty_ltp = int(round(nifty_quote["NSE:NIFTY 50"]["last_price"]))
            # round to nearest 50
            atm_strike = int(round(nifty_ltp / 50) * 50)
        except Exception:
            atm_strike = None

    ce_symbols = []
    pe_symbols = []
    if atm_strike is not None:
        # Look for strikes +/- strikes_range * 50
        strikes = [atm_strike + i * 50 for i in range(-strikes_range, strikes_range + 1)]
        for s in strikes:
            ce_mask = weekly_options["tradingsymbol"].str.endswith(f"{s}CE")
            pe_mask = weekly_options["tradingsymbol"].str.endswith(f"{s}PE")
            ce_rows = weekly_options[ce_mask]
            pe_rows = weekly_options[pe_mask]
            # choose the first match if multiple
            if not ce_rows.empty:
                sym = f"NFO:{ce_rows.iloc[0]['tradingsymbol']}"
                ce_symbols.append(sym)
            if not pe_rows.empty:
                sym = f"NFO:{pe_rows.iloc[0]['tradingsymbol']}"
                pe_symbols.append(sym)
    return ce_symbols, pe_symbols

def get_option_chain_oi(ce_syms, pe_syms):
    """Return total OI for CE and PE by calling kite.ltp or fallback to historical_data(oi=True)"""
    total_ce_oi = 0
    total_pe_oi = 0
    details = []

    # combined list for ltp call
    all_syms = ce_syms + pe_syms
    if not all_syms:
        return 0, 0, details

    try:
        quotes = kite.ltp(all_syms)
    except Exception:
        quotes = {}

    for sym in ce_syms:
        q = quotes.get(sym)
        oi = None
        if q:
            # common keys may differ; try possible locations
            oi = q.get("oi") or q.get("open_interest") or q.get("oi_day") or None
        if oi is None:
            # fallback: fetch last historical minute/day candle with oi=True (slow)
            try:
                # need instrument token (search via instruments_nfo)
                instr_df = instruments_nfo()
                row = instr_df[instr_df["tradingsymbol"] == sym.split(":", 1)[1]]
                if not row.empty:
                    token = int(row.iloc[0]["instrument_token"])
                    today = dt.date.today()
                    candles = kite.historical_data(instrument_token=token,
                                                   from_date=dt.datetime.combine(today, dt.time(9, 15)),
                                                   to_date=dt.datetime.combine(today, dt.time(15, 30)),
                                                   interval="day", oi=True)
                    if candles:
                        oi = candles[-1].get("oi") or candles[-1].get("open_interest")
            except Exception:
                oi = 0
        oi = int(oi) if oi else 0
        total_ce_oi += oi
        details.append((sym, oi, "CE"))

    for sym in pe_syms:
        q = quotes.get(sym)
        oi = None
        if q:
            oi = q.get("oi") or q.get("open_interest") or None
        if oi is None:
            try:
                instr_df = instruments_nfo()
                row = instr_df[instr_df["tradingsymbol"] == sym.split(":", 1)[1]]
                if not row.empty:
                    token = int(row.iloc[0]["instrument_token"])
                    today = dt.date.today()
                    candles = kite.historical_data(instrument_token=token,
                                                   from_date=dt.datetime.combine(today, dt.time(9, 15)),
                                                   to_date=dt.datetime.combine(today, dt.time(15, 30)),
                                                   interval="day", oi=True)
                    if candles:
                        oi = candles[-1].get("oi") or candles[-1].get("open_interest")
            except Exception:
                oi = 0
        oi = int(oi) if oi else 0
        total_pe_oi += oi
        details.append((sym, oi, "PE"))

    return total_ce_oi, total_pe_oi, details

@st.cache_data(ttl=1800)
def fetch_fii_dii_from_upstox():
    """Scrape FII/DII net flows from Upstox (best-effort). Returns (fii_cr, dii_cr)"""
    try:
        url = "https://upstox.com/fii-dii-data/"
        resp = requests.get(url, headers={"User-Agent": "Mozilla/5.0"}, timeout=10)
        soup = BeautifulSoup(resp.text, "html.parser")
        # Best-effort selector - if site changes you must update this
        net_flow_boxes = soup.find_all("div", class_="net-flow-card-value")
        if len(net_flow_boxes) >= 2:
            fii = net_flow_boxes[0].text.strip().replace("₹", "").replace(",", "")
            dii = net_flow_boxes[1].text.strip().replace("₹", "").replace(",", "")
            return float(fii), float(dii)
    except Exception:
        pass
    return None, None

# ----------------------------
# UI: Inputs & toggles
# ----------------------------
st.title("📊 NIFTY Intraday Trade Dashboard (Zerodha + manual fallback)")

col1, col2 = st.columns([2, 1])

with col2:
    use_realtime = st.checkbox("Use real-time data from Zerodha", value=True)
    strikes_range = st.slider("Number of strikes each side for PCR", min_value=1, max_value=10, value=4)
    historical_days = st.slider("Historical days to simulate signals",  min_value=3, max_value=30, value=7)

with col1:
    st.markdown("### Manual fields (only required if not available via API or to override)")
    # We only ask for fields that are not reliably present via Kite API
    manual_fii = st.number_input("Manual FII Net Flow (₹ Cr) — leave 0 to auto-fetch", value=0)
    manual_dii = st.number_input("Manual DII Net Flow (₹ Cr) — leave 0 to auto-fetch", value=0)
    manual_max_pain = st.number_input("Manual Max Pain Strike (enter 0 to skip)", value=0)
    manual_support = st.number_input("Manual Support Level (0 to skip)", value=0)
    manual_resistance = st.number_input("Manual Resistance Level (0 to skip)", value=0)
    manual_iv_atm = st.number_input("Manual ATM IV (%) (0 to skip)", value=0.0)

# ----------------------------
# Fetch real-time data block (if enabled)
# ----------------------------
realtime_data = {}
if use_realtime:
    st.info("Fetching data from Zerodha (may take a couple seconds)...")
    try:
        # 5-min candles
        df_5 = fetch_5min_candles_for_today()
        tech = compute_technical_indicators(df_5)
        pivots = get_pivots_from_prices(tech["High"], tech["Low"], tech["Close"])
        realtime_data.update(tech)
        realtime_data.update(pivots)
        # VIX
        try:
            vix_quote = kite.ltp(["NSE:INDIA VIX"])
            vix_val = vix_quote.get("NSE:INDIA VIX", {}).get("last_price") or vix_quote.get(list(vix_quote.keys())[0], {}).get("last_price")
            realtime_data["VIX"] = float(vix_val) if vix_val is not None else None
        except Exception:
            realtime_data["VIX"] = None

        # Option chain symbols (ATM weekly)
        ce_syms, pe_syms = find_weekly_option_symbols(atm_strike=None, strikes_range=strikes_range)
        ce_total_oi, pe_total_oi, oi_details = get_option_chain_oi(ce_syms, pe_syms)
        realtime_data["CE_OI"] = ce_total_oi
        realtime_data["PE_OI"] = pe_total_oi
        realtime_data["PCR"] = round(pe_total_oi / ce_total_oi, 3) if ce_total_oi else None
        # Max pain: optional manual or compute from option chain (not implemented compute here - require prices & payoff calc)
        realtime_data["MaxPain"] = manual_max_pain if manual_max_pain else None

        # FII/DII: try auto-scrape if manual not provided
        if manual_fii == 0 and manual_dii == 0:
            fii_auto, dii_auto = fetch_fii_dii_from_upstox()
            realtime_data["FII"] = fii_auto
            realtime_data["DII"] = dii_auto
        else:
            realtime_data["FII"] = manual_fii
            realtime_data["DII"] = manual_dii

        # ATM IV: if manual provided use that else attempt to derive from option LTPs (not full IV calc)
        realtime_data["ATM_IV"] = None if manual_iv_atm == 0.0 else manual_iv_atm

    except Exception as e:
        st.warning(f"Realtime fetch problem: {e}")
        use_realtime = False

# If realtime not used or failed, build values from manual input defaults / ask user to submit minimal form
if not use_realtime:
    st.warning("Realtime disabled — please provide manual values (below) and click 'Analyze'.")
    with st.form("manual_form_block"):
        close_price = st.number_input("Current NIFTY Level (spot)", value=26585.0)
        open_price = st.number_input("Open Price (previous day)", value=26480.0)
        high_price = st.number_input("Day High (or expected)", value=26680.0)
        low_price = st.number_input("Day Low (or expected)", value=26420.0)
        vix_manual = st.number_input("India VIX", value=12.5)
        atm_iv_manual = st.number_input("ATM IV (%)", value=14.5)
        fii_manual = st.number_input("FII Net Flow (₹ Cr)", value=-10000)
        dii_manual = st.number_input("DII Net Flow (₹ Cr)", value=8000)
        maxpain_manual = st.number_input("Max Pain Strike", value=24650)
        support_manual = st.number_input("Support Level", value=24000)
        resistance_manual = st.number_input("Resistance Level", value=25300)
        submit_manual = st.form_submit_button("Analyze")

    if submit_manual:
        realtime_data["Close"] = float(close_price)
        realtime_data["Open"] = float(open_price)
        realtime_data["High"] = float(high_price)
        realtime_data["Low"] = float(low_price)
        realtime_data["VIX"] = float(vix_manual)
        realtime_data["ATM_IV"] = float(atm_iv_manual)
        realtime_data["FII"] = float(fii_manual)
        realtime_data["DII"] = float(dii_manual)
        realtime_data["MaxPain"] = float(maxpain_manual)
        realtime_data["pivot"], realtime_data["s1"], realtime_data["r1"] = None, None, None
        # compute pivots from provided HLC
        piv = get_pivots_from_prices(realtime_data["High"], realtime_data["Low"], realtime_data["Close"])
        realtime_data.update(piv)
    else:
        st.stop()

# Allow the user to override individual values if desired
st.markdown("### Overrides (optional)")
ov1, ov2, ov3 = st.columns(3)
with ov1:
    if "VIX" in realtime_data:
        realtime_data["VIX"] = st.number_input("Override VIX (leave same to keep)", value=float(realtime_data.get("VIX") or 0.0))
    else:
        realtime_data["VIX"] = st.number_input("VIX", value=12.5)
with ov2:
    realtime_data["ATM_IV"] = st.number_input("Override ATM IV (%) (0=auto)", value=float(realtime_data.get("ATM_IV") or 0.0))
with ov3:
    if realtime_data.get("MaxPain"):
        realtime_data["MaxPain"] = st.number_input("Override Max Pain Strike (0=none)", value=float(realtime_data.get("MaxPain")))
    else:
        realtime_data["MaxPain"] = st.number_input("Max Pain Strike (0=none)", value=0.0)

# ----------------------------
# Decision logic - combine indicators into a score and produce recommendation
# ----------------------------
st.markdown("## 🔎 Market Snapshot")
snapshot_df = pd.DataFrame.from_dict({k: [v] for k, v in realtime_data.items()})
st.table(snapshot_df.T.rename(columns={0: "Value"}).head(20))

# Compute simple rule-based signal
def compute_trade_signal(data):
    """
    Logic used (example):
    - Bearish tilt -> SELL CALLS if:
      * RSI > 70 OR FII heavy selling OR PCR < 0.9 OR price > r1 etc.
    - Bullish tilt -> SELL PUTS if opposite
    - For intraday weekly ATM selling bias, prefer:
      FII net negative, price below pivot/s1, increased PE OI (PCR>1.1) => BUY? (in our convention BUY=long)
    We'll follow the user's earlier preference:
      - If bearish tilt: recommend SELL CALL (short premium)
      - If bullish tilt: recommend SELL PUT
      - Else NO TRADE
    """
    score = 0
    reasons = []
    close = data.get("Close")
    rsi = data.get("RSI")
    pcr = data.get("PCR")
    vix = data.get("VIX")
    fii = data.get("FII")
    dii = data.get("DII")
    s1 = data.get("s1")
    r1 = data.get("r1")
    atm_iv = data.get("ATM_IV")

    # Check FII selling strongly
    if fii is not None and fii < -3000:
        score -= 2
        reasons.append("FII heavy selling")

    if dii is not None and dii > 2000:
        score += 1
        reasons.append("DII supporting")

    # RSI extremes
    if rsi is not None:
        if rsi > 70:
            score -= 2
            reasons.append("RSI overbought")
        elif rsi < 30:
            score += 2
            reasons.append("RSI oversold")

    # PCR
    if pcr is not None:
        if pcr < 0.9:
            score -= 1
            reasons.append("PCR bearish (low)")
        elif pcr > 1.1:
            score += 1
            reasons.append("PCR bullish (high)")

    # VIX influence
    if vix is not None:
        if vix > 20:
            score -= 1
            reasons.append("High VIX (fear)")

    # Price vs pivots
    if s1 and close is not None:
        if close < s1:
            score += 1
            reasons.append("Price below S1 (support broken)")
    if r1 and close is not None:
        if close > r1:
            score -= 1
            reasons.append("Price above R1 (resistance breach)")

    # ATM IV - higher IV slightly favors options selling premium
    if atm_iv and atm_iv > 13.5:
        reasons.append("ATM IV elevated")

    # Final rule
    # positive score => bullish bias => SELL PUTS (we use 'SELL PUTS' to express bullish options sell)
    # negative score => bearish bias => SELL CALLS
    if score >= 2:
        return {"signal": "SELL_PUTS", "reason": "; ".join(reasons), "score": score}
    elif score <= -2:
        return {"signal": "SELL_CALLS", "reason": "; ".join(reasons), "score": score}
    else:
        return {"signal": "NO_TRADE", "reason": "; ".join(reasons) or "Indicators mixed", "score": score}

trade = compute_trade_signal(realtime_data)

# Human readable mapping
mapping = {
    "SELL_CALLS": ("Sell Call Options (bearish-neutral)", "Consider ATM or slightly OTM call sell"),
    "SELL_PUTS": ("Sell Put Options (bullish-neutral)", "Consider ATM or slightly OTM put sell"),
    "NO_TRADE": ("No Trade", "Market signals are mixed")
}

sig_label, sig_note = mapping[trade["signal"]]

st.markdown("## 🎯 Trade Recommendation")
st.write(f"**Signal:** `{trade['signal']}`")
st.write(f"**Label:** {sig_label}")
st.write(f"**Reason:** {trade['reason']}")
st.write(f"**Confidence score (simple):** {trade['score']}")

# Suggest strike(s)
def suggest_strikes(close_price):
    if not close_price:
        return "-"
    atm = int(round(close_price / 50) * 50)
    return f"{atm} (ATM), {atm+50}, {atm-50}"

st.write(f"**Suggested Strikes:** {suggest_strikes(realtime_data.get('Close'))}")
st.write(f"**Max Pain (if provided):** {realtime_data.get('MaxPain')}")

# ----------------------------
# Plot support/resistance with Plotly
# ----------------------------
st.markdown("## 📈 Support / Resistance Visualization")
fig = go.Figure()
# If we have df_5 use it to show recent price series
try:
    if "df_5" in locals():
        price_series = df_5["close"].tail(60)
        fig.add_trace(go.Scatter(x=price_series.index, y=price_series.values, mode="lines", name="NIFTY Close"))
    else:
        # fallback: small line of open/high/low/close
        oh = [realtime_data.get("Open"), realtime_data.get("High"), realtime_data.get("Low"), realtime_data.get("Close")]
        x = ["Open", "High", "Low", "Close"]
        fig.add_trace(go.Scatter(x=x, y=oh, mode="lines+markers", name="Price levels"))
except Exception:
    pass

# add pivot lines if available
for k in ("pivot", "s1", "s2", "r1", "r2"):
    if realtime_data.get(k) is not None:
        fig.add_hline(y=realtime_data[k], line_dash="dot", annotation_text=k.upper(), annotation_position="top left")

fig.update_layout(height=420, margin=dict(l=20, r=20, t=40, b=20))
st.plotly_chart(fig, use_container_width=True)

# ----------------------------
# Historical signal simulation (simple)
# ----------------------------
st.markdown("## 🧾 Historical Signal Simulation (last N days intraday)")
@st.cache_data(ttl=600)
def simulate_historical_signals(days=7):
    """Fetch past N days of 5-min data and run same rule set for each day, summarise daily signal outcome."""
    results = []
    for d in range(1, days + 1):
        date = dt.date.today() - dt.timedelta(days=d)
        start = dt.datetime.combine(date, dt.time(9, 15))
        end = dt.datetime.combine(date, dt.time(15, 30))
        # token attempt as earlier
        try:
            ltp_quote = kite.ltp(["NSE:NIFTY 50"])
            token = ltp_quote.get("NSE:NIFTY 50", {}).get("instrument_token")
        except Exception:
            token = 256265
        try:
            candles = kite.historical_data(instrument_token=token, from_date=start, to_date=end, interval="15minute")
            if not candles:
                continue
            day_df = pd.DataFrame(candles)
            day_df.columns = [c.lower() for c in day_df.columns]
            # compute intraday indicators (use close series)
            day_df["rsi"] = RSIIndicator(safe_series(day_df["close"]), window=14).rsi()
            # compute simple pivot for the day
            h = day_df["high"].max()
            l = day_df["low"].min()
            c = day_df["close"].iloc[-1]
            pivs = get_pivots_from_prices(h, l, c)
            # run same compute_trade_signal with day-level aggregated values
            data_day = {"Close": float(c), "RSI": float(day_df["rsi"].iloc[-1]),
                        "PCR": realtime_data.get("PCR"), "VIX": realtime_data.get("VIX"),
                        "FII": realtime_data.get("FII"), "DII": realtime_data.get("DII"),
                        "s1": pivs["s1"], "r1": pivs["r1"], "ATM_IV": realtime_data.get("ATM_IV")}
            res = compute_trade_signal(data_day)
            results.append({
                "date": date.isoformat(),
                "close": c,
                "rsi": float(day_df["rsi"].iloc[-1]),
                "signal": res["signal"],
                "score": res["score"]
            })
        except Exception:
            continue
    return pd.DataFrame(results)

hist_df = simulate_historical_signals(historical_days)
if not hist_df.empty:
    st.table(hist_df)
else:
    st.info("No historical simulation results available (maybe API limits or no data).")

# ----------------------------
# Export / Save trade setup (optional)
# ----------------------------
st.markdown("## 💾 Save trade setup (optional)")
if st.button("Save current setup to CSV"):
    row = pd.Series({
        "timestamp": dt.datetime.now().isoformat(),
        "signal": trade["signal"],
        "reason": trade["reason"],
        "score": trade["score"],
        "close": realtime_data.get("Close"),
        "rsi": realtime_data.get("RSI"),
        "pcr": realtime_data.get("PCR"),
        "vix": realtime_data.get("VIX"),
        "fii": realtime_data.get("FII"),
        "dii": realtime_data.get("DII")
    })
    fname = f"trade_setup_{dt.datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    row.to_frame().T.to_csv(fname, index=False)
    st.success(f"Saved {fname} in current directory.")

st.caption("This dashboard gives a rule-based trade suggestion. Treat the suggestion as decision-support, not financial advice.")

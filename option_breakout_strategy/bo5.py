import os
import logging
import pandas as pd
import numpy as np
import datetime as dt
from datetime import datetime, time, timedelta
from kiteconnect import KiteConnect
from dotenv import load_dotenv
import time as t

# -------------------- Time Utilities --------------------
def to_time_obj(t_str):
    return datetime.strptime(t_str, "%H:%M").time()

def now_time():
    return datetime.now().time()

# -------------------- CONFIG --------------------
load_dotenv(".env")

api_key = os.getenv("API_KEY")
access_token = os.getenv("ACCESS_TOKEN")
quantity = int(os.getenv("QUANTITY", "75"))
log_file = "5MIN_breakout_strategy.log"

CANDLE_START = to_time_obj(os.getenv("CANDLE_START", "09:15"))
CANDLE_END = to_time_obj(os.getenv("CANDLE_END", "09:20"))
STRIKE_INTERVAL = int(os.getenv("STRIKE_INTERVAL", 50))
BREAKOUT_BUFFER = int(os.getenv("BREAKOUT_BUFFER", 2))

# -------------------- LOGGING --------------------
logging.basicConfig(
    filename=log_file,
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    force=True
)
logging.info("🚀 Starting ATM Breakout Strategy...")

# -------------------- INIT --------------------
kite = KiteConnect(api_key=api_key)
kite.set_access_token(access_token)
instrument_list = kite.instruments("NFO")

# -------------------- Kite Utility Functions --------------------
def wait_until(target_time):
    logging.info(f"⏳ Waiting until {target_time.strftime('%H:%M')} ...")
    while now_time() < target_time:
        t.sleep(1)

def fetch_5min_candle(token):
    """Fetch 5-minute candle from CANDLE_START to CANDLE_END"""
    try:
        from_dt = datetime.now().replace(hour=CANDLE_START.hour, minute=CANDLE_START.minute, second=0, microsecond=0)
        to_dt = datetime.now().replace(hour=CANDLE_END.hour, minute=CANDLE_END.minute, second=0, microsecond=0)
        candles = kite.historical_data(token, from_dt, to_dt, interval="5minute", continuous=False)
        if not candles:
            raise ValueError("Empty candle data returned.")
        logging.info(f"[📊] 5-min candle from {from_dt.time()} to {to_dt.time()}: {candles[0]}")
        return candles[0]
    except Exception as e:
        logging.error(f"[❌] Error fetching candle: {e}")
        raise

def get_ltp(symbol):
    try:
        return kite.ltp(symbol)[symbol]["last_price"]
    except Exception as e:
        logging.warning(f"[⚠️] Error fetching LTP for {symbol}: {e}")
        return None

def round_to_strike(price):
    return round(price / STRIKE_INTERVAL) * STRIKE_INTERVAL

# -------------------- Breakout Logic --------------------
def detect_breakout(high, low, monitor_end_time):
    logging.info(f"🧭 Monitoring breakout from {CANDLE_END.strftime('%H:%M')} to {monitor_end_time.strftime('%H:%M')} (High={high}, Low={low})")
    while now_time() < monitor_end_time:
        ltp = get_ltp("NSE:NIFTY 50")
        if not ltp:
            t.sleep(BREAKOUT_BUFFER)
            continue
        if ltp > high:
            logging.info(f"[🔼] Breakout above high: ₹{ltp}")
            return "PE", ltp
        elif ltp < low:
            logging.info(f"[🔽] Breakdown below low: ₹{ltp}")
            return "CE", ltp
        t.sleep(BREAKOUT_BUFFER)
    return None, None

# -------------------- Option Contracts --------------------
def option_contracts(ticker, option_type="BOTH", exchange="NFO"):
    option_contracts = []
    for instrument in instrument_list:
        if (
            instrument["name"] == ticker and
            instrument["exchange"] == exchange and
            instrument["segment"] == f"{exchange}-OPT"
        ):
            if option_type in ["CE", "PE"] and instrument["instrument_type"] == option_type:
                option_contracts.append(instrument)
            elif option_type == "BOTH" and instrument["instrument_type"] in ["CE", "PE"]:
                option_contracts.append(instrument)
    return pd.DataFrame(option_contracts)

def option_contracts_closest(ticker, duration=0, option_type="BOTH", exchange="NFO"):
    df_opt_contracts = option_contracts(ticker, option_type, exchange)
    if df_opt_contracts.empty:
        return pd.DataFrame()
    df_opt_contracts["time_to_expiry"] = (pd.to_datetime(df_opt_contracts["expiry"]) - dt.datetime.now()).dt.days
    min_day_count = np.sort(df_opt_contracts["time_to_expiry"].unique())[duration]
    return df_opt_contracts[df_opt_contracts["time_to_expiry"] == min_day_count].reset_index(drop=True)

def option_chain(ticker, underlying_price, duration=0, num=5, option_type="BOTH", exchange="NFO"):
    df_opt_contracts = option_contracts_closest(ticker, duration, option_type, exchange)
    if df_opt_contracts.empty:
        return pd.DataFrame()
    df_opt_contracts.sort_values(by=["strike"], inplace=True, ignore_index=True)
    atm_idx = abs(df_opt_contracts["strike"] - underlying_price).argmin()
    up = int(num / 2)
    dn = num - up
    return df_opt_contracts.iloc[max(0, atm_idx - up): atm_idx + dn].reset_index(drop=True)

# -------------------- Order Functions --------------------
def placeMarketOrder(order_params):
    return kite.place_order(
        tradingsymbol=order_params['tradingsymbol'],
        exchange=order_params['exchange'],
        transaction_type=order_params['transaction_type'],
        quantity=order_params['quantity'],
        order_type="MARKET",
        product=order_params['product'],
        variety=order_params['variety']
    )

def place_atm_sell_order(price, leg):
    atm_strike = round_to_strike(price)
    df_chain = option_chain("NIFTY", price, 0, 5, leg)
    atm_opt = df_chain[df_chain["strike"] == atm_strike].iloc[0]
    params = {
        "exchange": "NFO",
        "tradingsymbol": atm_opt["tradingsymbol"],
        "transaction_type": "SELL",
        "variety": "regular",
        "product": "MIS",
        "quantity": atm_opt["lot_size"]
    }
    try:
        order_id = placeMarketOrder(params)
        logging.info(f"[✅] SELL {leg}: {atm_opt['tradingsymbol']} | Qty: {atm_opt['lot_size']} | Order ID: {order_id}")
    except Exception as e:
        logging.error(f"[❌] Order placement failed: {e}")

# -------------------- Main Execution --------------------
def run_breakout_strategy():
    wait_until(CANDLE_END)
    try:
        candle = fetch_5min_candle(256265)
        high = candle["high"]
        low = candle["low"]
        monitor_end = (datetime.combine(datetime.today(), CANDLE_END) + timedelta(minutes=5)).time()
        leg, breakout_price = detect_breakout(high, low, monitor_end)
        if not leg:
            logging.warning("⚠️ No breakout occurred within monitoring window.")
            return
        place_atm_sell_order(breakout_price, leg)
    except Exception as e:
        logging.exception(f"[❌] Strategy failed: {e}")

# ✅ Execute Strategy
run_breakout_strategy()

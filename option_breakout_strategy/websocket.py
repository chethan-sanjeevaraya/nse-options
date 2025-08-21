import os
import logging
import pandas as pd
import numpy as np
import datetime as dt
from datetime import datetime, timedelta
from kiteconnect import KiteConnect
from dotenv import load_dotenv
import time as t
from kiteconnect import KiteTicker, KiteConnect
import sys

# -------------------- CONFIG --------------------
load_dotenv(".env")

api_key = os.getenv("API_KEY")
access_token = os.getenv("ACCESS_TOKEN")
quantity = int(os.getenv("QUANTITY", "75"))
log_file = "5MIN_BO_strategy.log"

STRIKE_INTERVAL = int(os.getenv("STRIKE_INTERVAL", 50))
BREAKOUT_BUFFER = float(os.getenv("BREAKOUT_BUFFER", 2))  # seconds between checks
BUFFER_POINTS = float(os.getenv("BUFFER_POINTS", 50))  # % of candle size to add as buffer# Add to CONFIG
SL_PERCENT = float(os.getenv("SL_PERCENT", 30))   # e.g., 30% loss
TARGET_PERCENT = float(os.getenv("TARGET_PERCENT", 50))  # e.g., 50% profit

# ----------------------------------------
# Logging setup
# ----------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[
        logging.FileHandler("strategy.log", mode="a"),
        logging.StreamHandler(sys.stdout),
    ]
)
logging.info("Starting ATM Breakout Strategy...")

# -------------------- INIT --------------------
kite = KiteConnect(api_key=api_key)
kite.set_access_token(access_token)
instrument_list = kite.instruments("NFO")

def get_ltp(symbol):
    try:
        return kite.ltp(symbol)[symbol]["last_price"]
    except Exception as e:
        logging.warning(f"Error fetching LTP for {symbol}: {e}")
        return None

def round_to_strike(price):
    return round(price / STRIKE_INTERVAL) * STRIKE_INTERVAL

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
    if df_chain.empty:
        logging.error(f" No option contracts found for leg={leg} near strike={atm_strike}")
        return None

    atm_opt = df_chain[df_chain["strike"] == atm_strike]
    if atm_opt.empty:
        logging.error(f" No exact ATM option found for strike={atm_strike}")
        return None

    atm_opt = atm_opt.iloc[0]
    symbol = atm_opt["tradingsymbol"]
    lot_size = atm_opt["lot_size"]

    try:
        # 🔹 Main SELL order
        order_id = placeMarketOrder({
            "exchange": "NFO",
            "tradingsymbol": symbol,
            "transaction_type": "SELL",
            "variety": "regular",
            "product": "MIS",
            "quantity": lot_size
        })
        logging.info(f" SELL {leg}: {symbol} | Qty: {lot_size} | Order ID: {order_id}")

        # 🔹 Fetch entry price for SL & Target
        opt_ltp = get_ltp(f"NFO:{symbol}")
        if not opt_ltp:
            logging.error(" Could not fetch option LTP for SL/Target placement")
            return order_id

        sl_price = round(opt_ltp * (1 + SL_PERCENT / 100), 1)
        target_price = round(opt_ltp * (1 - TARGET_PERCENT / 100), 1)
        logging.info(f" Setting SL={sl_price} | Target={target_price} for {symbol}")

        # 🔹 Stop Loss order
        kite.place_order(
        tradingsymbol=atm_opt['tradingsymbol'],
        exchange="NFO",
        transaction_type="BUY",
        quantity=atm_opt['lot_size'],
        order_type="SL",                  # FIX: use SL, not SL-M
        price=sl_price,                   # FIX: required for SL
        trigger_price=sl_price,           # FIX: required for SL
        product="MIS",
        variety="regular"
        )

        # 🔹 Target order
        kite.place_order(
            tradingsymbol=symbol,
            exchange="NFO",
            transaction_type="BUY",
            quantity=lot_size,
            order_type="LIMIT",
            price=target_price,
            product="MIS",
            variety="regular"
        )

        logging.info(f" SL & Target orders placed for {symbol}")
        return order_id

    except Exception as e:
        logging.error(f" Order placement failed for {symbol}: {e}")
        return None

def log_candle_and_breakout(prev_candle, ltp, breakout_high, breakout_low):
    """
    Log details of the last completed candle and breakout levels
    """
    try:
        ts = datetime.now()
        logging.info(
            f" Previous Candle {prev_candle['start']} - {prev_candle['end']} "
            f"| High={prev_candle['high']:.2f} | Low={prev_candle['low']:.2f}"
        )
        logging.info(
            f" {ts.strftime('%H:%M:%S.%f')[:-3]} | Close LTP={ltp:.2f} "
            f"| Breakout Above={breakout_high:.2f} | Below={breakout_low:.2f}"
        )
    except Exception as e:
        logging.error(f"Failed to log candle/breakout: {e}")

# Example: NIFTY spot
instrument_token = 256265  
# ----------------------------------------------------------------------
# CandleBuilder Class
# ----------------------------------------------------------------------
class CandleBuilder:
    def __init__(self, interval=300):
        self.interval = interval
        self.current_candle = None

    def update(self, tick):
        ts = tick["timestamp"].replace(second=0, microsecond=0)
        bucket = ts.minute // (self.interval // 60) * (self.interval // 60)
        start = ts.replace(minute=bucket, second=0, microsecond=0)

        if not self.current_candle:
            self.current_candle = {"start": start, "open": tick["last_price"],
                                   "high": tick["last_price"], "low": tick["last_price"],
                                   "close": tick["last_price"]}
            return None

        if start > self.current_candle["start"]:
            closed = self.current_candle
            self.current_candle = {"start": start, "open": tick["last_price"],
                                   "high": tick["last_price"], "low": tick["last_price"],
                                   "close": tick["last_price"]}
            return closed
        else:
            self.current_candle["high"] = max(self.current_candle["high"], tick["last_price"])
            self.current_candle["low"] = min(self.current_candle["low"], tick["last_price"])
            self.current_candle["close"] = tick["last_price"]
        return None


def run_breakout_strategy_dynamic(api_key, access_token, instrument_token):
    logging.info("🚀 Starting ATM Breakout Strategy with Auto-Reconnect...")

    candle_builder = CandleBuilder(interval=300)  # 5-min candles
    breakout_levels = {"high": None, "low": None}

    retry_delay = 2  # start with 2s backoff

    while True:  # 🔄 Auto-reconnect loop
        try:
            logging.info("🔄 Connecting WebSocket for breakout strategy...")
            kws = KiteTicker(api_key, access_token)

            # -------------------------
            # WS Event Handlers
            # -------------------------
            def on_ticks(ws, ticks):
                if not ticks:
                    return
                tick = ticks[0]

                closed_candle = candle_builder.update(tick)
                if closed_candle:
                    high = closed_candle["high"]
                    low = closed_candle["low"]
                    breakout_levels["high"] = high
                    breakout_levels["low"] = low
                    logging.info(
                        f"📊 Closed Candle @ {closed_candle['start']} | "
                        f"H:{high} L:{low} C:{closed_candle['close']}"
                    )

                if breakout_levels["high"] and breakout_levels["low"]:
                    price = tick["last_price"]
                    if price > breakout_levels["high"]:
                        logging.info(f"⚡ Breakout UP! Price: {price}, Level: {breakout_levels['high']}")
                    elif price < breakout_levels["low"]:
                        logging.info(f"⚡ Breakout DOWN! Price: {price}, Level: {breakout_levels['low']}")

            def on_connect(ws, response):
                logging.info("✅ Connected, subscribing...")
                ws.subscribe([instrument_token])
                ws.set_mode(ws.MODE_FULL, [instrument_token])

            def on_close(ws, code, reason):
                logging.warning(f"⚠️ Connection closed: {code} - {reason}")
                ws.stop()  # <--- ensure clean shutdown

            def on_error(ws, code, reason):
                logging.error(f"❌ Error: {code} - {reason}")
                ws.stop()  # <--- ensure clean shutdown

            # -------------------------
            # Assign Handlers
            # -------------------------
            kws.on_ticks = on_ticks
            kws.on_connect = on_connect
            kws.on_close = on_close
            kws.on_error = on_error

            # Connect WS
            kws.connect(threaded=True, disable_ssl_verification=False)

            # Keep alive while WS is running
            while True:
                if not kws.is_connected():
                    logging.warning("⚡ Lost WebSocket connection, breaking to retry...")
                    break
                t.sleep(1)

        except Exception as e:
            logging.error(f"💥 WebSocket crashed: {e}")

        # Retry with exponential backoff
        logging.info(f"⏳ Reconnecting in {retry_delay}s...")
        t.sleep(retry_delay)
        retry_delay = min(retry_delay * 2, 60)  # cap at 60s
# ======================
# RUN
# ======================
if __name__ == "__main__":
    run_breakout_strategy_dynamic(api_key, access_token, instrument_token)

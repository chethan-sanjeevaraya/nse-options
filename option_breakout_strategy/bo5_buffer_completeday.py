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

def align_to_5min(dt_obj):
    """Align any datetime to the last completed 5-min mark (e.g. 10:31 → 10:30)."""
    minute = (dt_obj.minute // 5) * 5
    return dt_obj.replace(minute=minute, second=0, microsecond=0)

# -------------------- CONFIG --------------------
load_dotenv(".env")

api_key = os.getenv("API_KEY")
access_token = os.getenv("ACCESS_TOKEN")
quantity = int(os.getenv("QUANTITY", "75"))
log_file = "5MIN_BO_strategy.log"

CANDLE_START = to_time_obj(os.getenv("CANDLE_START", "09:15"))
CANDLE_END = to_time_obj(os.getenv("CANDLE_END", "15:15"))  # updated to 3:15 pm
STRIKE_INTERVAL = int(os.getenv("STRIKE_INTERVAL", 50))
BREAKOUT_BUFFER = float(os.getenv("BREAKOUT_BUFFER", 2))  # seconds between checks
BUFFER_PERCENT = float(os.getenv("BUFFER_PERCENT", 50))  # % of candle size to add as buffer# Add to CONFIG
SL_PERCENT = float(os.getenv("SL_PERCENT", 30))   # e.g., 30% loss
TARGET_PERCENT = float(os.getenv("TARGET_PERCENT", 50))  # e.g., 50% profit

# -------------------- LOGGING --------------------
log_file = "5MIN_BO_strategy.log"

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

def fetch_5min_candle(token, start_dt, end_dt):
    """Fetch 5-minute candle between start_dt and end_dt"""
    try:
        candles = kite.historical_data(token, start_dt, end_dt, interval="5minute", continuous=False)
        if not candles:
            raise ValueError("Empty candle data returned.")
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
        logging.error(f"[❌] No option contracts found for leg={leg} near strike={atm_strike}")
        return
    atm_opt = df_chain[df_chain["strike"] == atm_strike]
    if atm_opt.empty:
        logging.error(f"[❌] No exact ATM option found for strike={atm_strike}")
        return
    atm_opt = atm_opt.iloc[0]

    try:
        # Main SELL order
        order_id = placeMarketOrder({
            "exchange": "NFO",
            "tradingsymbol": atm_opt["tradingsymbol"],
            "transaction_type": "SELL",
            "variety": "regular",
            "product": "MIS",
            "quantity": atm_opt["lot_size"]
        })
        logging.info(f"[✅] SELL {leg}: {atm_opt['tradingsymbol']} | Qty: {atm_opt['lot_size']} | Order ID: {order_id}")

        # Get entry price to base SL & Target
        opt_ltp = get_ltp(f"NFO:{atm_opt['tradingsymbol']}")
        if not opt_ltp:
            logging.error("[❌] Could not fetch option LTP for SL/Target placement")
            return

        sl_price = round(opt_ltp * (1 + SL_PERCENT / 100), 1)
        target_price = round(opt_ltp * (1 - TARGET_PERCENT / 100), 1)
        logging.info(f"🎯 Setting SL={sl_price} | Target={target_price} for {atm_opt['tradingsymbol']}")

        # Stop Loss order
        kite.place_order(
            tradingsymbol=atm_opt['tradingsymbol'],
            exchange="NFO",
            transaction_type="BUY",
            quantity=atm_opt['lot_size'],
            order_type="SL-M",
            trigger_price=sl_price,
            product="MIS",
            variety="regular"
        )

        # Target order
        kite.place_order(
            tradingsymbol=atm_opt['tradingsymbol'],
            exchange="NFO",
            transaction_type="BUY",
            quantity=atm_opt['lot_size'],
            order_type="LIMIT",
            price=target_price,
            product="MIS",
            variety="regular"
        )

        logging.info(f"[✅] SL & Target orders placed for {atm_opt['tradingsymbol']}")

    except Exception as e:
        logging.error(f"[❌] Order placement failed: {e}")

# -------------------- Main Strategy --------------------
def run_breakout_strategy_dynamic():
    market_start = datetime.now().replace(hour=CANDLE_START.hour, minute=CANDLE_START.minute, second=0, microsecond=0)
    market_end = datetime.now().replace(hour=CANDLE_END.hour, minute=CANDLE_END.minute, second=0, microsecond=0)

    logging.info(f"🚀 Starting dynamic breakout monitoring from {CANDLE_START} to {CANDLE_END}")

    # Wait until market start
    if datetime.now() < market_start:
        wait_until(market_start.time())

    prev_candle_high = None
    prev_candle_low = None

    while datetime.now() < market_end:
        aligned_start = align_to_5min(datetime.now())
        candle_start_time = aligned_start
        candle_end_time = aligned_start + timedelta(minutes=5)

        try:
            # Fetch the PREVIOUS candle, not the current forming one
            prev_start = candle_start_time - timedelta(minutes=5)
            prev_end = candle_start_time
            candle = fetch_5min_candle(256265, prev_start, prev_end)

            prev_candle_high = candle["high"]
            prev_candle_low = candle["low"]

            logging.info(
                f"📊 Previous Candle {prev_start.time()} - {prev_end.time()} "
                f"| High={prev_candle_high:.2f} | Low={prev_candle_low:.2f}"
            )

        except Exception as e:
            logging.error(f"❌ Error fetching candle: {e}")
            continue

        # Calculate breakout buffer from previous candle
        price_buffer = (BUFFER_PERCENT / 100) * (prev_candle_high - prev_candle_low)
        breakout_above = prev_candle_high + price_buffer
        breakout_below = prev_candle_low - price_buffer

        while datetime.now() < candle_end_time:
            ltp = get_ltp("NSE:NIFTY 50")
            if not ltp:
                logging.warning("⚠️ LTP fetch failed, retrying...")
                t.sleep(BREAKOUT_BUFFER)
                continue

            # Breakout check based on PREVIOUS candle's high/low + buffer
            if ltp > breakout_above:
                logging.info(f"[🔼] Breakout above previous high at LTP={ltp:.2f}")
                place_atm_sell_order(ltp, "PE")
                return

            elif ltp < breakout_below:
                logging.info(f"[🔽] Breakdown below previous low at LTP={ltp:.2f}")
                place_atm_sell_order(ltp, "CE")
                return

            t.sleep(BREAKOUT_BUFFER)

        # Log once after candle closes
        logging.info(
            f"🕒 {datetime.now().time()} | Close LTP={ltp:.2f} "
            f"| Breakout Above={breakout_above:.2f} | Below={breakout_below:.2f}"
        )
        logging.info(f"➡️ No breakout. Moving to next candle after {candle_end_time.time()}")

    logging.info("🏁 Market closed - no breakout detected.")

# ✅ Run the dynamic breakout strategy
run_breakout_strategy_dynamic()

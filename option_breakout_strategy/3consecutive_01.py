import os
import logging
import pandas as pd
import numpy as np
import datetime as dt
from datetime import datetime, timedelta
from kiteconnect import KiteConnect
from dotenv import load_dotenv
import time

# -------------------- CONFIG --------------------
load_dotenv(".env")

api_key = os.getenv("API_KEY")
access_token = os.getenv("ACCESS_TOKEN")
quantity = int(os.getenv("QUANTITY", "75"))
log_file = "5MIN_BO_strategy.log"


BREAKOUT_BUFFER = float(os.getenv("BREAKOUT_BUFFER", 2))
BUFFER_PERCENT = float(os.getenv("BUFFER_PERCENT", 50))
SL_PERCENT = float(os.getenv("SL_PERCENT", 30))
TARGET_PERCENT = float(os.getenv("TARGET_PERCENT", 50))


STRIKE_INTERVAL   = int(os.getenv("STRIKE_INTERVAL", 50))
LOT_SIZE          = int(os.getenv("LOT_SIZE", 1))
CANDLE_INTERVAL   = int(os.getenv("CANDLE_INTERVAL", 5))  # minutes
CANDLE_START_HOUR = int(os.getenv("CANDLE_START_HOUR", 9))
CANDLE_START_MIN  = int(os.getenv("CANDLE_START_MIN", 15))
CANDLE_END_HOUR   = int(os.getenv("CANDLE_END_HOUR", 15))
CANDLE_END_MIN    = int(os.getenv("CANDLE_END_MIN", 30))
MAX_ENTRIES       = int(os.getenv("MAX_ENTRIES", 2))
MAX_RETRIES       = int(os.getenv("MAX_RETRIES", 3))
RETRY_DELAY       = int(os.getenv("RETRY_DELAY", 1))  # seconds
UNDERLYING        = os.getenv("UNDERLYING", "NIFTY")

# -------------------- LOGGING --------------------
logging.basicConfig(
    filename=log_file,
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    force=True
)
logging.info("Starting ATM Breakout Strategy...")

# -------------------- INIT --------------------
kite = KiteConnect(api_key=api_key)
kite.set_access_token(access_token)
instrument_list = kite.instruments("NFO")

# -----------------------------
# Spot and Futures Fetchers
# -----------------------------    
def fetch_5min_candle(token, start_dt, end_dt):
    """Fetch 5-minute candle between start_dt and end_dt"""
    try:
        candles = kite.historical_data(token, start_dt, end_dt, interval="5minute", continuous=False)
        if not candles:
            raise ValueError("Empty candle data returned.")
        return candles[0]   # contains 'date', 'open', 'high', 'low', 'close', 'volume'
    except Exception as e:
        logging.error(f" Error fetching candle: {e}")
        raise

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

def option_chain(ticker, underlying_price, duration=0, num=5, option_type="BOTH", exchange="NFO"):
    df_opt_contracts = option_contracts_closest(ticker, duration, option_type, exchange)
    if df_opt_contracts.empty:
        return pd.DataFrame()
    df_opt_contracts.sort_values(by=["strike"], inplace=True, ignore_index=True)
    atm_idx = abs(df_opt_contracts["strike"] - underlying_price).argmin()
    up = int(num / 2)
    dn = num - up
    return df_opt_contracts.iloc[max(0, atm_idx - up): atm_idx + dn].reset_index(drop=True)

# =========================
# HELPERS
# =========================

def wait_until(target_time):
    """Pause execution until target_time (dt.time)."""
    while datetime.now().time() < target_time:
        time.sleep(1)

def place_order(symbol, qty, side):
    """
    Placeholder: order placement function.
    Replace with real Kite Connect API call.
    """
    logging.info(f"[ORDER] {side} {qty} of {symbol}")
    return True

# =========================
# STRATEGY UTILS
# =========================
def retry(func, *args, retries=3, delay=2, **kwargs):
    for attempt in range(1, retries+1):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            logging.warning(f"[Retry {attempt}/{retries}] {func.__name__} failed: {e}")
            time.sleep(delay)
    logging.error(f"{func.__name__} failed after {retries} attempts")
    return None

# -------------------- UTILITY FUNCTIONS --------------------
def get_nearest_nifty_fut():
    fut_contracts = [inst for inst in instrument_list if inst["name"] == "NIFTY" and inst["tradingsymbol"].endswith("FUT")]
    if not fut_contracts:
        logging.error("No NIFTY Futures contracts found")
        return None
    fut_contracts = sorted(fut_contracts, key=lambda x: x["expiry"])
    return fut_contracts[0]

def option_contracts_closest(ticker, duration=0, option_type="BOTH"):
    df = option_contracts(ticker, option_type)
    if df.empty: return pd.DataFrame()
    df["time_to_expiry"] = (pd.to_datetime(df["expiry"]) - dt.datetime.now()).dt.days
    min_day_count = np.sort(df["time_to_expiry"].unique())[duration]
    return df[df["time_to_expiry"]==min_day_count].reset_index(drop=True)

def retry_kite_call(func, *args, retries=MAX_RETRIES, delay=RETRY_DELAY, **kwargs):
    for attempt in range(retries):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            logging.warning(f"Attempt {attempt+1}/{retries} failed for {func.__name__}: {e}")
            time.sleep(delay)
    logging.error(f"All {retries} attempts failed for {func.__name__}")
    return None

def calculate_sl_tp_from_candles(entry_price, last_candles, leg_type):
    """
    last_candles: list of dicts [{"open":..,"high":..,"low":..,"close":..}, ...]
    """
    highs = [c["high"] for c in last_candles]
    lows  = [c["low"] for c in last_candles]

    if leg_type == "CE":
        sl = max(highs)
        risk = sl - entry_price
        tp = entry_price - 2 * risk
    elif leg_type == "PE":
        sl = min(lows)
        risk = entry_price - sl
        tp = entry_price + 2 * risk
    else:
        sl, tp = entry_price * 1.01, entry_price * 0.99  # fallback
    return round(sl, 1), round(tp, 1)

def place_atm_sell_order_auto_sl_tp(kite, option_symbol, lot_size, entry_price, last_candles):
    leg_type = "CE" if "CE" in option_symbol else "PE"
    sl, tp = calculate_sl_tp_from_candles(entry_price, last_candles, leg_type)
    logging.info(f"Placing {leg_type} SELL {option_symbol} | Entry={entry_price} | SL={sl} | TP={tp}")

    sell_order_id = retry_kite_call(kite.place_order,
                                    tradingsymbol=option_symbol,
                                    exchange="NFO",
                                    transaction_type="SELL",
                                    quantity=lot_size,
                                    order_type="MARKET",
                                    product="MIS",
                                    variety="regular")

    sl_order_id = retry_kite_call(kite.place_order,
                                  tradingsymbol=option_symbol,
                                  exchange="NFO",
                                  transaction_type="BUY",
                                  quantity=lot_size,
                                  order_type="SL",
                                  trigger_price=sl,
                                  product="MIS",
                                  variety="regular")

    tp_order_id = retry_kite_call(kite.place_order,
                                  tradingsymbol=option_symbol,
                                  exchange="NFO",
                                  transaction_type="BUY",
                                  quantity=lot_size,
                                  order_type="LIMIT",
                                  price=tp,
                                  product="MIS",
                                  variety="regular")

    logging.info(f"OCO placed: SELL={sell_order_id}, SL={sl_order_id}, TP={tp_order_id}")
    return sell_order_id, sl_order_id, tp_order_id

def get_nearest_expiry_option(instrument_list, underlying="NIFTY"):
    today = dt.datetime.now().date()
    opt_contracts = [inst for inst in instrument_list if inst["name"] == underlying and inst["instrument_type"] in ["CE","PE"]]
    expiry_dates = sorted({pd.to_datetime(inst["expiry"]).date() for inst in opt_contracts if pd.to_datetime(inst["expiry"]).date() >= today})
    if not expiry_dates:
        raise ValueError("No future expiry dates found")
    return expiry_dates[0].strftime("%d%b%y").upper()

def get_atm_option_symbols(instrument_list, spot_price, underlying="NIFTY", expiry_str=None):
    atm_strike = round(spot_price / 50) * 50
    df_opts = pd.DataFrame([inst for inst in instrument_list
                            if inst["name"]==underlying and inst["instrument_type"] in ["CE","PE"] and inst["expiry"].date() >= dt.datetime.now().date()])
    if df_opts.empty:
        raise ValueError("No option contracts available")
    if not expiry_str:
        expiry_str = get_nearest_expiry_option(instrument_list, underlying)
    df_opts = df_opts[df_opts["tradingsymbol"].str.contains(expiry_str)]
    strikes = df_opts["strike"].unique()
    closest_strike = min(strikes, key=lambda x: abs(x - atm_strike))
    ce_symbol = df_opts[(df_opts["strike"]==closest_strike)&(df_opts["instrument_type"]=="CE")]["tradingsymbol"].values
    pe_symbol = df_opts[(df_opts["strike"]==closest_strike)&(df_opts["instrument_type"]=="PE")]["tradingsymbol"].values
    if not ce_symbol.any() or not pe_symbol.any():
        raise ValueError(f"No ATM option found for strike {closest_strike} and expiry {expiry_str}")
    return ce_symbol[0], pe_symbol[0]

def cleanup_open_orders(kite, open_trades):
    for trade_id, trade in open_trades.items():
        if trade["status"]=="OPEN":
            if trade.get("sl_id"):
                retry_kite_call(kite.cancel_order, variety=kite.VARIETY_REGULAR, order_id=trade["sl_id"])
            if trade.get("tp_id"):
                retry_kite_call(kite.cancel_order, variety=kite.VARIETY_REGULAR, order_id=trade["tp_id"])
            logging.info(f"Cancelled remaining orders for {trade['symbol']}")

# -------------------- MAIN STRATEGY --------------------
def run_consecutive_candle_strategy_auto_oco(kite, instrument_list, fut_symbol, qty, candle_start, candle_end, max_entries=2):
    logging.info(f"Strategy started from {candle_start} to {candle_end}")
    fut_token = instrument_list.loc[instrument_list["tradingsymbol"]==fut_symbol,"instrument_token"].values[0]

    candle_types, volumes, closes, raw_candles = [], [], [], []
    entries_taken = 0
    last_logged_candle_end = None
    open_trades = {}

    try:
        while datetime.now().time() < candle_end:
            aligned = datetime.now().replace(second=0, microsecond=0)
            aligned = aligned - timedelta(minutes=aligned.minute % CANDLE_INTERVAL)
            c_start = aligned - timedelta(minutes=CANDLE_INTERVAL)
            c_end = aligned

            if last_logged_candle_end==c_end:
                time.sleep(1)
                continue

            candle_data = retry_kite_call(kite.historical_data, fut_token, c_start, c_end, interval="5minute", continuous=False)
            if not candle_data:
                logging.warning(f"No candle data for {c_start}-{c_end}")
                time.sleep(1)
                continue

            c = candle_data[-1]  # last closed candle
            o,h,l,c_,v = c["open"], c["high"], c["low"], c["close"], c["volume"]
            ctype = "GREEN" if c_>o else "RED" if c_<o else "DOJI"

            candle_types.append(ctype)
            volumes.append(v)
            closes.append(c_)
            raw_candles.append({"open":o,"high":h,"low":l,"close":c_,"volume":v})
            if len(candle_types)>4:
                candle_types.pop(0)
                volumes.pop(0)
                closes.pop(0)
                raw_candles.pop(0)

            logging.info(f"Candle {c_start.time()}-{c_end.time()} | O={o} C={c_} V={v} Type={ctype}")
            last_logged_candle_end = c_end

            # Signal detection
            if len(candle_types)==4 and entries_taken<max_entries:
                first_three, fourth = candle_types[:3], candle_types[3]
                vol_ok = volumes[3]<volumes[2]
                leg_type = None
                if all(ct=="RED" for ct in first_three) and fourth=="GREEN" and vol_ok:
                    leg_type = "CE"
                elif all(ct=="GREEN" for ct in first_three) and fourth=="RED" and vol_ok:
                    leg_type = "PE"

                if leg_type:
                    spot_price = closes[3]
                    try:
                        expiry_str = get_nearest_expiry_option(instrument_list)
                        ce_sym, pe_sym = get_atm_option_symbols(instrument_list, spot_price, expiry_str=expiry_str)
                        option_symbol = ce_sym if leg_type=="CE" else pe_sym
                    except Exception as e:
                        logging.error(f"ATM option detection failed: {e}")
                        continue

                    ltp = retry_kite_call(kite.ltp, [f"NFO:{option_symbol}"])
                    if ltp:
                        ltp = ltp[f"NFO:{option_symbol}"]["last_price"]
                        sell_id, sl_id, tp_id = place_atm_sell_order_auto_sl_tp(kite, option_symbol, qty, ltp, raw_candles[-3:])
                        if sell_id:
                            entries_taken +=1
                            open_trades[sell_id] = {"symbol":option_symbol,"leg":leg_type,"qty":qty,"entry":ltp,"sl_id":sl_id,"tp_id":tp_id,"status":"OPEN"}
                            logging.info(f"Trade entered: {option_symbol} | Entry={ltp}")

            # Monitor open trades
            if open_trades:
                orders = retry_kite_call(kite.orders)
                for trade_id, trade in list(open_trades.items()):
                    if trade["status"]!="OPEN":
                        continue
                    sl_status = next((o for o in orders if str(o["order_id"])==str(trade["sl_id"])), None)
                    tp_status = next((o for o in orders if str(o["order_id"])==str(trade["tp_id"])), None)
                    if sl_status and sl_status["status"]=="COMPLETE" or tp_status and tp_status["status"]=="COMPLETE":
                        trade["status"]="CLOSED"
                        cleanup_open_orders(kite, {trade_id:trade})
                        logging.info(f"Trade {trade['symbol']} closed via SL/TP")

            time.sleep(1)
    except KeyboardInterrupt:
        logging.info("Strategy manually stopped")
    finally:
        cleanup_open_orders(kite, open_trades)
        logging.info("Strategy finished. All pending orders cleared.")

# -------------------- MAIN EXECUTION BLOCK --------------------
if __name__ == "__main__":
    try:
        logging.info("Initializing fully automated strategy with OCO...")

        instrument_df = pd.DataFrame(instrument_list)
        nearest_fut = get_nearest_nifty_fut()
        if not nearest_fut:
            logging.error("No NIFTY Futures contract found. Exiting.")
            exit(1)

        fut_symbol = nearest_fut["tradingsymbol"]
        fut_expiry = nearest_fut["expiry"]
        logging.info(f"Nearest FUT contract: {fut_symbol} | Expiry: {fut_expiry.date()}")

        market_start = datetime.now().replace(hour=CANDLE_START_HOUR, minute=CANDLE_START_MIN, second=0, microsecond=0)
        market_end   = datetime.now().replace(hour=CANDLE_END_HOUR, minute=CANDLE_END_MIN, second=0, microsecond=0)

        if datetime.now() < market_start:
            logging.info(f"Waiting for market open at {market_start.time()}")
            wait_until(market_start.time())

        run_consecutive_candle_strategy_auto_oco(
        kite=kite,
        instrument_list=instrument_df,
        fut_symbol=fut_symbol,
        qty=quantity,
        candle_start=market_start.time(),
        candle_end=market_end.time(),
        max_entries=MAX_ENTRIES
        )

        logging.info("Strategy execution completed successfully.")

    except Exception as e:
        logging.error(f"Fatal error in strategy execution: {e}")

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

# ------------------ Utility Functions ------------------
def align_to_5min(dt):
    return dt.replace(minute=(dt.minute // 5) * 5, second=0, microsecond=0)

def wait_until(target_time):
    logging.info(f"⏳ Waiting until {target_time.strftime('%H:%M')} ...")
    while now_time() < target_time:
        t.sleep(1)
# ------------------ Solid Candle Filter ------------------
def is_solid_breakout_candle(df):
    """
    Check if the breakout candle is strong enough to avoid fake breakouts.
    df: DataFrame with columns [open, high, low, close]
    """
    last_candle = df.iloc[-1]
    high, low, close, open_ = last_candle['high'], last_candle['low'], last_candle['close'], last_candle['open']

    # Condition 1: Close near high (top 20% of range)
    if (high - close) > (high - low) * 0.2:
        return False

    # ATR(5)
    df['tr'] = np.maximum(df['high'] - df['low'],
                          np.maximum(abs(df['high'] - df['close'].shift(1)),
                                     abs(df['low'] - df['close'].shift(1))))
    atr = df['tr'].rolling(window=5).mean().iloc[-1]

    # Minimum body size requirement
    body_size = abs(close - open_)
    min_body_size = max(atr * 0.5, close * 0.003)

    if body_size < min_body_size:
        return False

    return True

# ------------------ Main Strategy ------------------
def run_breakout_strategy_dynamic():
    market_start = datetime.now().replace(hour=CANDLE_START.hour, minute=CANDLE_START.minute, second=0, microsecond=0)
    market_end = datetime.now().replace(hour=CANDLE_END.hour, minute=CANDLE_END.minute, second=0, microsecond=0)

    logging.info(f"🚀 Starting dynamic breakout monitoring from {CANDLE_START} to {CANDLE_END}")

    if datetime.now() < market_start:
        wait_until(market_start.time())

    # Keep last few candles for ATR calculation
    recent_candles = []

    while datetime.now() < market_end:
        aligned_start = align_to_5min(datetime.now())
        candle_start_time = aligned_start
        candle_end_time = aligned_start + timedelta(minutes=5)

        try:
            # Previous candle time range
            prev_start = candle_start_time - timedelta(minutes=5)
            prev_end = candle_start_time

            # Fetch previous candle OHLC data (dict format)
            prev_candle = fetch_5min_candle(256265, prev_start, prev_end)  
            prev_high = prev_candle["high"]
            prev_low = prev_candle["low"]
            prev_open = prev_candle["open"]
            prev_close = prev_candle["close"]

            logging.info(
                f"📊 Previous Candle {prev_start.time()} - {prev_end.time()} "
                f"| O={prev_open:.2f} | H={prev_high:.2f} | L={prev_low:.2f} | C={prev_close:.2f}"
            )

            # Append to recent candles for ATR calc
            recent_candles.append(prev_candle)
            if len(recent_candles) > 6:  # keep only last 6
                recent_candles.pop(0)

            # Solid candle check
            if len(recent_candles) >= 5:
                atr = calc_atr(recent_candles[-5:])  # ATR(5)
            else:
                atr = 0

            close_near_high = (prev_high - prev_close) <= (prev_high - prev_low) * 0.2
            body_size = abs(prev_close - prev_open)
            min_body_size = max(atr * 0.5, prev_close * 0.003)
            is_solid = close_near_high and (body_size >= min_body_size)

            if not is_solid:
                logging.info("⚠️ Previous candle not solid — skipping breakout checks this candle.")
                wait_until(candle_end_time.time())
                continue

        except Exception as e:
            logging.error(f"❌ Error fetching candle: {e}")
            continue

        # Breakout levels
        price_buffer = (BUFFER_PERCENT / 100) * (prev_high - prev_low)
        breakout_above = prev_high + price_buffer
        breakout_below = prev_low - price_buffer

        # Monitor during current candle
        while datetime.now() < candle_end_time:
            elapsed_sec = (datetime.now() - candle_start_time).total_seconds()

            # Wait until 10% of candle has passed
            if elapsed_sec < (5 * 60) * 0.1:
                t.sleep(1)
                continue

            ltp = get_ltp("NSE:NIFTY 50")
            if not ltp:
                logging.warning("⚠️ LTP fetch failed, retrying...")
                t.sleep(BREAKOUT_BUFFER)
                continue

            if ltp > breakout_above:
                logging.info(f"[🔼] Breakout above previous high at LTP={ltp:.2f}")
                place_atm_sell_order(ltp, "PE")
                return
            elif ltp < breakout_below:
                logging.info(f"[🔽] Breakdown below previous low at LTP={ltp:.2f}")
                place_atm_sell_order(ltp, "CE")
                return

            t.sleep(BREAKOUT_BUFFER)

        logging.info(
            f"🕒 {datetime.now().time()} | Close LTP={ltp:.2f} "
            f"| Breakout Above={breakout_above:.2f} | Below={breakout_below:.2f}"
        )
        logging.info(f"➡️ No breakout. Moving to next candle after {candle_end_time.time()}")

    logging.info("🏁 Market closed — no breakout detected.")


# Helper function for ATR calculation
def calc_atr(candles):
    trs = []
    for i in range(len(candles)):
        high = candles[i]["high"]
        low = candles[i]["low"]
        prev_close = candles[i - 1]["close"] if i > 0 else candles[i]["close"]
        tr = max(
            high - low,
            abs(high - prev_close),
            abs(low - prev_close)
        )
        trs.append(tr)
    return sum(trs) / len(trs)

# ✅ Run the dynamic breakout strategy
run_breakout_strategy_dynamic()

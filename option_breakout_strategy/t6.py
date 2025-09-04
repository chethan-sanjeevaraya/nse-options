import os
import logging
import pandas as pd
import numpy as np
import datetime as dt
from datetime import datetime, timedelta
from kiteconnect import KiteConnect
from dotenv import load_dotenv
import time
import json

# -------------------- LOGGING --------------------
log_file = "t6_strategy.log"

logging.basicConfig(
    filename=log_file,
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    force=True
)
logging.info("Starting Candle Pattern Breakout Strategy...")

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

QUANTITY = int(os.getenv("QUANTITY", "75"))
CANDLE_START = to_time_obj(os.getenv("CANDLE_START", "09:15"))
CANDLE_END = to_time_obj(os.getenv("CANDLE_END", "15:15"))  # updated to 3:15 pm
STRIKE_INTERVAL = int(os.getenv("STRIKE_INTERVAL", 50))
BREAKOUT_BUFFER = float(os.getenv("BREAKOUT_BUFFER", 2))  # seconds between checks
BUFFER_PERCENT = float(os.getenv("BUFFER_PERCENT", 50))  # % of candle size to add as buffer# Add to CONFIG
SL_PERCENT = float(os.getenv("SL_PERCENT", 30))   # e.g., 30% loss
TARGET_PERCENT = float(os.getenv("TARGET_PERCENT", 50))  # e.g., 50% profit
MAX_RETRIES = int(os.getenv("MAX_RETRIES", "4"))
RETRY_DELAY = float(os.getenv("RETRY_DELAY", "1"))
CANDLE_PATTERNS = json.loads(os.getenv("CANDLE_PATTERNS", "[]"))

# -------------------- INIT --------------------
kite = KiteConnect(api_key=api_key)
kite.set_access_token(access_token)
instrument_list = kite.instruments("NFO")

# 1.Health Check --------------------------------------
def prestart_healthcheck(kite):
    """
    Perform a pre-start health check on the Kite Connect API.

    """
    profile = retry_kite_call(kite.profile)
    if not profile:
        margins = retry_kite_call(kite.margins, "equity")
        if not margins:
            return False
    return True

# 2.Call a function with automatic retries ---------------------------------
def retry_kite_call(func, *args, retries=MAX_RETRIES, delay=RETRY_DELAY, **kwargs):
    """
    Call a function with automatic retries and exponential backoff in case of exceptions.
    """

    backoff = delay
    for attempt in range(1, retries + 1):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            msg = str(e)
            logging.warning("Attempt %d/%d failed for %s: %s", attempt, retries, getattr(func, "__name__", str(func)), msg)
            time.sleep(backoff)
            backoff = min(backoff * 2, 30)
    return None

# -------------------- Kite Utility Functions --------------------
def wait_until(target_time):
    logging.info(f"⏳ Waiting until {target_time.strftime('%H:%M')} ...")
    while now_time() < target_time:
        time.sleep(1)

def fetch_5min_candle(token, start_dt, end_dt):
    """Fetch 5-minute candle between start_dt and end_dt"""
    try:
        candles = kite.historical_data(token, start_dt, end_dt, interval="5minute", continuous=False)
        if not candles:
            raise ValueError("Empty candle data returned.")
        return candles[-1]  # contains 'date', 'open', 'high', 'low', 'close', 'volume'
    except Exception as e:
        logging.error(f" Error fetching candle: {e}")
        raise

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
    qty = order_params.get("quantity") or order_params.get("QUANTITY")
    if not qty:
        raise ValueError("quantity missing in order_params")
    return kite.place_order(
        tradingsymbol=order_params["tradingsymbol"],
        exchange=order_params["exchange"],
        transaction_type=order_params["transaction_type"],
        quantity=int(qty),
        order_type="MARKET",
        product=order_params["product"],
        variety=order_params["variety"]
    )

def place_atm_sell_order(price, leg):
    atm_strike = round_to_strike(price)
    df_chain = option_chain("NIFTY", price, 0, 5, leg)
    if df_chain.empty:
        logging.error(f" No option contracts found for leg={leg} near strike={atm_strike}")
        return None, None

    atm_opt = df_chain[df_chain["strike"] == atm_strike]
    if atm_opt.empty:
        logging.error(f" No exact ATM option found for strike={atm_strike}")
        return None, None

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
        logging.info(f"SELL {leg}: {symbol} | Qty: {lot_size} | Order ID: {order_id}")

        # 🔹 Fetch entry price for SL & Target
        opt_ltp = get_ltp(f"NFO:{symbol}")
        if not opt_ltp:
            logging.error(" Could not fetch option LTP for SL/Target placement")
            return order_id, symbol

        sl_price = round(opt_ltp * (1 + SL_PERCENT / 100), 1)
        target_price = round(opt_ltp * (1 - TARGET_PERCENT / 100), 1)
        logging.info(f"Setting SL={sl_price} | Target={target_price} for {symbol}")

        # 🔹 Stop Loss order (use SL-M for reliability)
        kite.place_order(
            tradingsymbol=symbol,
            exchange="NFO",
            transaction_type="BUY",
            quantity=lot_size,
            order_type="SL",                # safer than SL
            trigger_price=sl_price,           # only trigger needed
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

        logging.info(f"SL & Target orders placed for {symbol}")
        return order_id, symbol  # return symbol so we can use in exit

    except Exception as e:
        logging.error(f" Order placement failed for {symbol}: {e}")
        return None, None

def get_nearest_nifty_fut():
    """Return nearest NIFTY Futures contract dict (token, symbol, expiry)."""
    fut_contracts = [
        inst for inst in instrument_list
        if inst["name"] == "NIFTY" and inst["tradingsymbol"].endswith("FUT")
    ]
    if not fut_contracts:
        logging.error(" No NIFTY Futures contracts found")
        return None
    fut_contracts = sorted(fut_contracts, key=lambda x: x["expiry"])  # nearest expiry first
    return fut_contracts[0]  # return full dict for logging
    
def calculate_sl_tp(candles, entry_price, trade_side, buffer_pct=0.05):
    """
    Calculate Stop Loss (SL) and Target (TP) based on last 3 candles.
    
    Args:
        candles (list of dict): Each candle should have keys: open, high, low, close, volume
        entry_price (float): The price at which breakout entry is triggered
        trade_side (str): "BUY_CE" for bullish, "BUY_PE" for bearish
        buffer_pct (float): % buffer for SL adjustment (default 0.05%)
    
    Returns:
        tuple: (stop_loss, target)
    """
    if len(candles) < 3:
        raise ValueError("Need at least 3 candles to calculate SL/TP")

    # Take last 3 candles before breakout
    last_3 = candles[-3:]

    high_3 = max(c["high"] for c in last_3)
    low_3 = min(c["low"] for c in last_3)

    buffer = entry_price * buffer_pct / 100  # convert % to absolute

    if trade_side == "BUY_CE":
        stop_loss = low_3 - buffer
        target = entry_price + 2 * (entry_price - stop_loss)

    elif trade_side == "BUY_PE":
        stop_loss = high_3 + buffer
        target = entry_price - 2 * (stop_loss - entry_price)

    else:
        raise ValueError("trade_side must be either 'BUY_CE' or 'BUY_PE'")

    return round(stop_loss, 2), round(target, 2)

def place_bracket_orders(kite, tradingsymbol, option_type, entry_price, candle_highs, candle_lows, qty, poll_interval=5):
    """
    Place Stop Loss and Target orders for a short option strategy with auto OCO handling.
    
    Args:
        kite: Zerodha KiteConnect object
        tradingsymbol: str (e.g. "NIFTY25SEP24650PE")
        option_type: str, "CE" or "PE"
        entry_price: float, executed entry price
        candle_highs: list of last 3 candle highs
        candle_lows: list of last 3 candle lows
        qty: int, order quantity
        poll_interval: int, seconds to check order status
    """

    try:
        # ---------------------------
        # 1. Compute SL & Target
        # ---------------------------
        if option_type == "CE":  
            stoploss = max(candle_highs)
            risk = stoploss - entry_price
            target = entry_price - 2 * risk
        else:  # PE
            stoploss = min(candle_lows)
            risk = entry_price - stoploss
            target = entry_price + 2 * risk

        logging.info(f"SL set at {stoploss}, Target at {target}, Risk={risk}")

        # ---------------------------
        # 2. Place SL Order
        # ---------------------------
        if option_type == "CE":
            trigger_price = round(stoploss, 1)
            limit_price = round(stoploss * 1.005, 1)
        else:
            trigger_price = round(stoploss, 1)
            limit_price = round(stoploss * 0.995, 1)

        sl_order_id = kite.place_order(
            variety=kite.VARIETY_REGULAR,
            exchange=kite.EXCHANGE_NFO,
            tradingsymbol=tradingsymbol,
            transaction_type=kite.TRANSACTION_TYPE_BUY,
            quantity=QUANTITY,
            product=kite.PRODUCT_MIS,
            order_type=kite.ORDER_TYPE_SL,
            price=limit_price,
            trigger_price=trigger_price
        )

        logging.info(f"SL Order Placed: ID={sl_order_id} @ Trig={trigger_price}, Limit={limit_price}")

        # ---------------------------
        # 3. Place Target Order
        # ---------------------------
        target_order_id = kite.place_order(
            variety=kite.VARIETY_REGULAR,
            exchange=kite.EXCHANGE_NFO,
            tradingsymbol=tradingsymbol,
            transaction_type=kite.TRANSACTION_TYPE_BUY,
            quantity=QUANTITY,
            product=kite.PRODUCT_MIS,
            order_type=kite.ORDER_TYPE_LIMIT,
            price=round(target, 1)
        )

        logging.info(f"Target Order Placed: ID={target_order_id} @ {target}")

        # ---------------------------
        # 4. Auto-Cancel Logic (OCO)
        # ---------------------------
        while True:
            orders = kite.orders()

            sl_status = next((o for o in orders if str(o["order_id"]) == str(sl_order_id)), None)
            tgt_status = next((o for o in orders if str(o["order_id"]) == str(target_order_id)), None)

            # If SL executed → cancel Target
            if sl_status and sl_status["status"] == "COMPLETE":
                logging.info(f"Stoploss Hit for {tradingsymbol}. Cancelling Target Order {target_order_id}")
                kite.cancel_order(variety=kite.VARIETY_REGULAR, order_id=target_order_id)
                break

            # If Target executed → cancel SL
            if tgt_status and tgt_status["status"] == "COMPLETE":
                logging.info(f"Target Hit for {tradingsymbol}. Cancelling SL Order {sl_order_id}")
                kite.cancel_order(variety=kite.VARIETY_REGULAR, order_id=sl_order_id)
                break

            time.sleep(poll_interval)

        return sl_order_id, target_order_id

    except Exception as e:
        logging.error(f"Failed in OCO handler for {tradingsymbol}: {e}")
        return None, None
    
def _symbol_from_token(token):
    for inst in instrument_list:
        if inst.get("instrument_token") == token:
            return inst.get("tradingsymbol")
    return None

def match_pattern(candle_types, volumes):
    if len(candle_types) < 2:  # Need at least 1 sequence + 1 confirm
        return None

    for rule in CANDLE_PATTERNS:
        seq_len = len(rule["sequence"])
        if len(candle_types) < seq_len + 1:
            continue  # not enough candles yet

        sequence_candles = candle_types[-(seq_len + 1):-1]  # last `seq_len` before final
        confirm_candle   = candle_types[-1]

        if sequence_candles == rule["sequence"] and confirm_candle == rule["final"]:
            # Volume condition: only check if rule explicitly has "volume":"LESS"
            if rule.get("volume") == "LESS":
                if volumes[-1] < volumes[-2]:
                    logging.info(f"📊 Pattern matched: {rule} | seq={sequence_candles} final={confirm_candle} | volumes ok")
                    return rule["action"]
            else:
                logging.info(f"📊 Pattern matched: {rule} | seq={sequence_candles} final={confirm_candle}")
                return rule["action"]

    # No match
    logging.debug(f"❌ No pattern matched | candles={candle_types} | volumes={volumes}")
    return None

def run_consecutive_candle_strategy():
    if not prestart_healthcheck(kite):
        logging.critical("Pre-start health check failed — aborting")
        return
    
    # ---- session times ----
    market_start = datetime.now().replace(hour=CANDLE_START.hour, minute=CANDLE_START.minute, second=0, microsecond=0)
    market_end   = datetime.now().replace(hour=CANDLE_END.hour,   minute=CANDLE_END.minute,   second=0, microsecond=0)

    logging.info(f"Starting Candle Pattern Break out strategy (vol filter, 2nd entry only if SL) from {CANDLE_START} to {CANDLE_END}")

    if datetime.now() < market_start:
        wait_until(market_start.time())

    # ---- instruments ----
    fut = get_nearest_nifty_fut()
    if not fut:
        logging.error(" Cannot proceed without NIFTY Futures contract.")
        return
    fut_token  = fut["instrument_token"]
    fut_symbol = fut["tradingsymbol"]
    fut_expiry = fut["expiry"]

    # ---- state ----
    candle_types   = []   # e.g., ["RED","RED","RED","GREEN"]
    volumes        = []   # candle volumes
    closes         = []   # underlying close values
    raw_candles    = []   # [{'open','high','low','close','volume',...}] last few

    in_trade                 = False
    entry_underlying         = None       # underlying entry reference (close of 4th candle)
    sl_underlying            = None       # underlying stop
    tp_underlying            = None       # underlying target
    option_type              = None       # "CE" or "PE"
    option_symbol            = None       # "NIFTY...CE/PE"
    option_entry_price       = None       # captured option price after sell
    option_lot_size          = None       # fetched from chain at entry

    entries_taken            = 0          # max 2
    second_entry_unlocked    = False      # only if first trade hit SL

    last_logged_candle_end   = None
    realized_pnl             = 0.0        # PnL on options

    # ---- main loop ----
    while datetime.now() < market_end:

        # --- tick-reactive exit management (based on underlying thresholds) ---
        if in_trade and option_symbol:
            # Underlying LTP (you can switch to futures LTP if you prefer)
            u_ltp = get_ltp("NSE:NIFTY 50")
            if u_ltp:
                # For 3 RED→GREEN we short CE: reversal up expected, so we EXIT CE if underlying hits SL (goes down past SL?) / TP (goes up?)
                # NOTE: Our SL/TP are computed as numeric levels on the underlying:
                #   If we short CE (expect underlying to fall), TP is BELOW entry and SL is ABOVE entry.
                #   If we short PE (expect underlying to rise), TP is ABOVE entry and SL is BELOW entry.
                # So:
                if option_type == "CE":
                    # TP is BELOW entry; SL is ABOVE entry
                    if u_ltp <= tp_underlying:
                        # Target hit → square off option short
                        if exit_trade(option_symbol,option_lot_size):
                            # realized PnL on option:
                            opt_ltp = get_ltp(f"NFO:{option_symbol}") or option_entry_price
                            realized_pnl += (option_entry_price - opt_ltp) * option_lot_size
                            logging.info(f"Target hit (underlying={u_ltp:.2f}) → exited {option_symbol}. Realized PnL={realized_pnl:.2f}. Day complete.")
                        return  # stop for the day
                    elif u_ltp >= sl_underlying:
                        # Stop hit → square off option short; allow a second entry
                        if exit_trade(option_symbol,option_lot_size):
                            opt_ltp = get_ltp(f"NFO:{option_symbol}") or option_entry_price
                            realized_pnl += (option_entry_price - opt_ltp) * option_lot_size
                            logging.info(f" SL hit (underlying={u_ltp:.2f}) → exited {option_symbol}. Realized PnL={realized_pnl:.2f}. Re-entry unlocked.")
                        in_trade = False
                        second_entry_unlocked = True

                elif option_type == "PE":
                    # TP is ABOVE entry; SL is BELOW entry
                    if u_ltp >= tp_underlying:
                        if exit_trade(option_symbol,option_lot_size):
                            opt_ltp = get_ltp(f"NFO:{option_symbol}") or option_entry_price
                            realized_pnl += (option_entry_price - opt_ltp) * option_lot_size
                            logging.info(f"Target hit (underlying={u_ltp:.2f}) → exited {option_symbol}. Realized PnL={realized_pnl:.2f}. Day complete.")
                        return
                    elif u_ltp <= sl_underlying:
                        if exit_trade(option_symbol,option_lot_size):
                            opt_ltp = get_ltp(f"NFO:{option_symbol}") or option_entry_price
                            realized_pnl += (option_entry_price - opt_ltp) * option_lot_size
                            logging.info(f" SL hit (underlying={u_ltp:.2f}) → exited {option_symbol}. Realized PnL={realized_pnl:.2f}. Re-entry unlocked.")
                        in_trade = False
                        second_entry_unlocked = True

        # --- once-per-candle-close work ---
        aligned = align_to_5min(datetime.now())
        c_start = aligned - timedelta(minutes=5)
        c_end   = aligned

        if last_logged_candle_end == c_end:
            time.sleep(1)
            continue  # wait for next candle close

        try:
            # Fetch the last closed 5-min underlying candle
            k = fetch_5min_candle(fut_token, c_start, c_end)
            o,h,l,c,v = k["open"], k["high"], k["low"], k["close"], k["volume"]

            # classify candle
            ctype = "GREEN" if c > o else "RED" if c < o else "DOJI"

            candle_types.append(ctype)
            volumes.append(v)
            closes.append(c)
            raw_candles.append({"open":o,"high":h,"low":l,"close":c,"volume":v})

            # keep only last 4
            if len(candle_types) > 4:
                candle_types.pop(0)
                volumes.pop(0)
                closes.pop(0)
                raw_candles.pop(0)

            # --- PnL snapshot (unrealized on option, realized cum) ---
            unrealized_pnl = 0.0
            if in_trade and option_symbol:
                opt_ltp = get_ltp(f"NFO:{option_symbol}")
                if opt_ltp is not None and option_entry_price is not None and option_lot_size:
                    # short option: PnL = (entry - current) * qty
                    unrealized_pnl = (option_entry_price - opt_ltp) * option_lot_size

            # Change your logging formatter to drop milliseconds
            formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s", "%Y-%m-%d %H:%M:%S")
            logger = logging.getLogger()
            for handler in logger.handlers:
                handler.setFormatter(formatter)

            logging.info(
                f"Candle {c_start.time()}-{c_end.time()} | O={o:.2f} C={c:.2f} Vol={v} | "
                f"Type={ctype} | FUT={fut_symbol} Exp={fut_expiry} | "
                f"PnL→ Realized={realized_pnl:.2f}, Unrealized={unrealized_pnl:.2f} "
                f"| Position={'SHORT '+option_type if in_trade else 'None'}"
            )
            last_logged_candle_end = c_end

            # --- Signal detection (only when flat) ---
            if (not in_trade) and (entries_taken < 2):
                if len(candle_types) == 4:
                    action = match_pattern(candle_types, volumes)

                    if action and (entries_taken == 0 or (entries_taken == 1 and second_entry_unlocked)):
                        entry_underlying = closes[3]  # confirm candle close

                        if action == "SHORT_CE":
                            sl_u, tp_u = calculate_sl_tp(raw_candles[-3:], entry_underlying, "BUY_CE", buffer_pct=0.05)
                            sl_underlying, tp_underlying = sl_u, tp_u
                            option_type = "CE"

                        elif action == "SHORT_PE":
                            sl_u, tp_u = calculate_sl_tp(raw_candles[-3:], entry_underlying, "BUY_PE", buffer_pct=0.05)
                            sl_underlying, tp_underlying = sl_u, tp_u
                            option_type = "PE"

                        # place order dynamically
                        _, option_symbol = place_atm_sell_order(entry_underlying, option_type)
                        if option_symbol:
                            option_entry_price = get_ltp(f"NFO:{option_symbol}")
                            atm_strike = round_to_strike(entry_underlying)
                            df_chain = option_chain("NIFTY", entry_underlying, 0, 5, option_type)
                            option_lot_size = int(df_chain[df_chain["strike"] == atm_strike].iloc[0]["lot_size"]) if not df_chain.empty else 75
                            in_trade = True
                            entries_taken += 1
                            second_entry_unlocked = False
                            logging.info(f"Signal {candle_types[:3]}→{candle_types[3]} "
                            f"({action}) | {option_symbol} | Entry(U)={entry_underlying:.2f} | SL(U)={sl_underlying:.2f} | TP(U)={tp_underlying:.2f}")

        except Exception as e:
            logging.error(f"Error in candle fetch/logic: {e}")

        time.sleep(1)

    logging.info("🏁 Market closed - strategy stopped.")


# Run the dynamic breakout strategy
run_consecutive_candle_strategy()

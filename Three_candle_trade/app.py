# t6_refactored.py
"""
Production-safe t6 strategy:
 - DRY_RUN mode (do not send real orders when enabled)
 - Auto SL/TP placement (NRML GTT OCO preferred, fallback MIS SL/TP)
 - Safety square-off at configurable time (default 15:25)
 - Realized + unrealized PnL tracking
 - Second-entry-after-SL logic (max 2 entries/day)
"""

import os
import logging
import json
import time
from datetime import datetime, timedelta
from dotenv import load_dotenv
from kiteconnect import KiteConnect

# utils
from common_utils import (
    to_time_obj, align_to_5min, wait_until,
    retry_kite_call, place_order_retry, get_ltp_retry,
    fetch_candle, round_to_strike, option_chain, get_nearest_nifty_fut, safe_get_lot_size
)

# -------------------- CONFIG & LOGGING --------------------
load_dotenv(".env")
API_KEY = os.getenv("API_KEY")
ACCESS_TOKEN = os.getenv("ACCESS_TOKEN")
QUANTITY = int(os.getenv("QUANTITY", "75"))
CANDLE_START = to_time_obj(os.getenv("CANDLE_START", "09:15"))
CANDLE_END = to_time_obj(os.getenv("CANDLE_END", "15:15"))
STRIKE_INTERVAL = int(os.getenv("STRIKE_INTERVAL", "50"))
RETRY_DELAY = float(os.getenv("RETRY_DELAY", 1.0))
CANDLE_PATTERNS = json.loads(os.getenv("CANDLE_PATTERNS", "[]") or "[]")
MAX_DAILY_LOSS = float(os.getenv("MAX_DAILY_LOSS", "5000"))
SAFETY_EXIT_HOUR = int(os.getenv("SAFETY_EXIT_HOUR", "15"))
SAFETY_EXIT_MIN = int(os.getenv("SAFETY_EXIT_MIN", "25"))
MAX_DAILY_LOSS = float(os.getenv("MAX_DAILY_LOSS", "5000"))
CANDLE_INTERVAL = int(os.getenv("CANDLE_INTERVAL", "5"))
MAX_ENTRIES = int(os.getenv("MAX_ENTRIES", "2"))
MAX_RETRIES = int(os.getenv("MAX_RETRIES", "3"))

SL_PERCENT = float(os.getenv("SL_PERCENT", "30"))    # percent on option premium
TARGET_PERCENT = float(os.getenv("TARGET_PERCENT", "50"))  # percent on option premium
UNDERLYING = os.getenv("UNDERLYING", "NIFTY")

SAFETY_EXIT_HOUR = int(os.getenv("SAFETY_EXIT_HOUR", "15"))
SAFETY_EXIT_MIN = int(os.getenv("SAFETY_EXIT_MIN", "25"))

# DRY_RUN: if True, do not place any real orders; log actions instead
DRY_RUN = os.getenv("DRY_RUN", "true").lower() in ("1", "true", "yes")

log_file = os.getenv("LOG_FILE", "t6_strategy.log")
logging.basicConfig(
    filename=log_file,
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    force=True
)
console = logging.StreamHandler()
console.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(message)s", "%Y-%m-%d %H:%M:%S"))
logging.getLogger().addHandler(console)
logging.info("t6_refactored starting (DRY_RUN=%s)", DRY_RUN)

# -------------------- KITE INIT --------------------
if not API_KEY or not ACCESS_TOKEN:
    logging.error("API_KEY or ACCESS_TOKEN not set in .env")
    raise SystemExit("Missing Kite credentials")

kite = KiteConnect(api_key=API_KEY)
kite.set_access_token(ACCESS_TOKEN)

# fetch instruments once (retry)
instrument_list = None
try:
    instrument_list = retry_kite_call(kite.instruments, "NFO", retries=2, delay=1)
except Exception as e:
    logging.error("Failed to fetch instruments: %s", e)
    instrument_list = []

# -------------------- STATE --------------------
def init_state():
    return {
        "candle_types": [], "volumes": [], "closes": [], "raw_candles": [],
        "in_trade": False,
        "option_type": None, "option_symbol": None,
        "option_entry_price": None, "option_lot_size": None,
        "sl_underlying": None, "tp_underlying": None,
        "entries_taken": 0, "second_entry_unlocked": False,
        "last_logged_candle_end": None,
        "realized_pnl": 0.0, "unrealized_pnl": 0.0,
        "sl_price_option": None, "tp_price_option": None,
    }

# -------------------- CORE HELPERS --------------------
def calculate_sl_tp(candles, entry_price, trade_side, buffer=0.5, rr_ratio=2):
    """
    Calculate underlying SL & TP using last candle (sequence candle).
    Returns (sl_underlying, tp_underlying)
    """

    if not candles or len(candles) < 1:
        raise ValueError("Need at least 1 candle for calculate_sl_tp")
    seq = candles[-1]
    high, low = seq["high"], seq["low"]
    if trade_side == "BUY_CE":
        stop_loss = low - buffer
        risk = entry_price - stop_loss
        target = entry_price + rr_ratio * risk
    else:
        stop_loss = high + buffer
        risk = stop_loss - entry_price
        target = entry_price - rr_ratio * risk
    return round(stop_loss, 2), round(target, 2)

def match_pattern(candle_types, volumes):
    """Return action string (SHORT_CE / SHORT_PE) or None."""
    if not CANDLE_PATTERNS:
        return None
    for rule in CANDLE_PATTERNS:
        seq_len = len(rule.get("sequence", []))
        if len(candle_types) < seq_len + 1:
            continue
        seq = candle_types[-(seq_len + 1):-1]
        confirm = candle_types[-1]
        if seq == rule["sequence"] and confirm == rule["final"]:
            if rule.get("volume") == "LESS" and len(volumes) >= 2:
                if volumes[-1] < volumes[-2]:
                    return rule.get("action")
            else:
                return rule.get("action")
    return None

def place_mis_entry(tradingsymbol, lot):
    """Place MIS market SELL entry (or mock in DRY_RUN). Returns order_resp or None."""
    if DRY_RUN:
        logging.info("[DRY_RUN] Would place MIS SELL: %s qty=%s", tradingsymbol, lot)
        return {"mock_order_id": "DRY_ENTRY"}
    try:
        resp = place_order_retry(
            kite,
            retries=MAX_RETRIES,
            delay=RETRY_DELAY,
            variety="regular",
            exchange="NFO",
            tradingsymbol=tradingsymbol,
            transaction_type="SELL",
            quantity=lot,
            order_type="MARKET",
            product="MIS"
        )
        logging.info("Placed MIS SELL entry: %s qty=%s resp=%s", tradingsymbol, lot, resp)
        return resp
    except Exception as e:
        logging.error("Entry placement failed for %s: %s", tradingsymbol, e)
        return None

def exit_trade_mis(tradingsymbol, lot):
    """Exit MIS position with market BUY (or mock). Returns True/False."""
    if DRY_RUN:
        logging.info("[DRY_RUN] Would exit MIS position: %s qty=%s", tradingsymbol, lot)
        return True
    try:
        resp = place_order_retry(
            kite,
            retries=MAX_RETRIES,
            delay=RETRY_DELAY,
            variety="regular",
            exchange="NFO",
            tradingsymbol=tradingsymbol,
            transaction_type="BUY",
            quantity=lot,
            order_type="MARKET",
            product="MIS"
        )
        logging.info("Exit order placed: %s | resp=%s", tradingsymbol, resp)
        return True
    except Exception as e:
        logging.error("Exit order failed for %s: %s", tradingsymbol, e)
        return False
    
# -------------------- SESSION SETUP --------------------
def setup_session_times():
    market_start = datetime.now().replace(hour=CANDLE_START.hour, minute=CANDLE_START.minute, second=0, microsecond=0)
    market_end = datetime.now().replace(hour=CANDLE_END.hour, minute=CANDLE_END.minute, second=0, microsecond=0)
    safety_exit = datetime.now().replace(hour=SAFETY_EXIT_HOUR, minute=SAFETY_EXIT_MIN, second=0, microsecond=0)
    return market_start, market_end, safety_exit

# -------------------- EXIT DETECTION (MIS in-code) --------------------
def check_sl_tp_and_exit(state):
    """
    Poll option LTP and check option-level SL/TP (derived from option premium).
    If hit, send MIS exit and update state.realized_pnl.
    Returns True if an exit occurred.
    """
    if not state.get("in_trade") or not state.get("option_symbol"):
        return False

    symbol = state["option_symbol"]
    lot = int(state.get("option_lot_size") or QUANTITY)
    entry_premium = state.get("option_entry_price")
    # option SL/TP (stored)
    sl_opt = state.get("sl_price_option")
    tp_opt = state.get("tp_price_option")

    # fetch option LTP
    opt_ltp = get_ltp_retry(kite, f"NFO:{symbol}", retries=2, delay=0.4)
    if opt_ltp is None:
        return False

    # SL hit (for SHORT option): if option LTP >= sl_opt -> loss
    if sl_opt is not None and opt_ltp >= sl_opt:
        ok = exit_trade_mis(symbol, lot)
        if ok:
            pnl = (entry_premium - opt_ltp) * lot
            state["realized_pnl"] += pnl
            state["unrealized_pnl"] = 0.0
            state["in_trade"] = False
            state["second_entry_unlocked"] = True  # unlock second entry on SL
            logging.info("[SL HIT] Exited %s at %.2f | PnL=%.2f", symbol, opt_ltp, pnl)
            return True

    # TP hit (for SHORT option): if option LTP <= tp_opt -> profit
    if tp_opt is not None and opt_ltp <= tp_opt:
        ok = exit_trade_mis(symbol, lot)
        if ok:
            pnl = (entry_premium - opt_ltp) * lot
            state["realized_pnl"] += pnl
            state["unrealized_pnl"] = 0.0
            state["in_trade"] = False
            logging.info("[TP HIT] Exited %s at %.2f | PnL=%.2f", symbol, opt_ltp, pnl)
            return True

    # update unrealized PnL
    if entry_premium is not None:
        state["unrealized_pnl"] = (entry_premium - opt_ltp) * lot

    return False

def place_mis_entry_with_sl_tp(kite, option_symbol, lot_size, sl_price, tp_price, buffer_pct=0.005):
    """
    Place MIS entry + visible SL-Limit + TP-Limit orders.
    Uses buffer_pct to set SL trigger < price (avoids rejection).
    Returns dict of order_ids so we can track them.
    """
    if DRY_RUN:
        logging.info(
            "[DRY_RUN] Would place: ENTRY SELL=%s, SL=%.2f, TP=%.2f qty=%d",
            option_symbol, sl_price, tp_price, lot_size
        )
        return {
            "entry_order_id": "MOCK_ENTRY",
            "sl_order_id": "MOCK_SL",
            "tp_order_id": "MOCK_TP"
        }

    try:
        # ---- Entry (SELL MARKET) ----
        entry_order_id = kite.place_order(
            exchange="NFO",
            tradingsymbol=option_symbol,
            transaction_type="SELL",
            variety="regular",
            product="MIS",
            order_type="MARKET",
            quantity=lot_size
        )

        # ---- Stop-loss (BUY SL-LIMIT) ----
        # Add small buffer to avoid rejection: price > trigger
        sl_trigger = round(sl_price, 1)
        sl_limit = round(sl_trigger * (1 + buffer_pct), 1)

        sl_order_id = kite.place_order(
            exchange="NFO",
            tradingsymbol=option_symbol,
            transaction_type="BUY",
            variety="regular",
            product="MIS",
            order_type="SL",
            trigger_price=sl_trigger,
            price=sl_limit,   # must be ≥ trigger
            quantity=lot_size
        )

        # ---- Take-profit (BUY LIMIT) ----
        tp_order_id = kite.place_order(
            exchange="NFO",
            tradingsymbol=option_symbol,
            transaction_type="BUY",
            variety="regular",
            product="MIS",
            order_type="LIMIT",
            price=round(tp_price, 1),
            quantity=lot_size
        )

        logging.info(
            "[ENTRY+OCO] %s | SL_trigger=%.2f SL_limit=%.2f TP=%.2f "
            "| entry_id=%s, sl_id=%s, tp_id=%s",
            option_symbol, sl_trigger, sl_limit, tp_price,
            entry_order_id, sl_order_id, tp_order_id
        )

        return {
            "entry_order_id": entry_order_id,
            "sl_order_id": sl_order_id,
            "tp_order_id": tp_order_id
        }

    except Exception as e:
        logging.error("❌ Order placement failed for %s: %s", option_symbol, e)
        return None
    
def monitor_and_cancel_leftover(kite, order_ids):
    """
    Monitor SL/TP orders and cancel the leftover one if the other is executed.
    """
    if not order_ids:
        return

    try:
        orders = kite.orders()
        sl_status = next((o for o in orders if o["order_id"] == order_ids["sl_order_id"]), None)
        tp_status = next((o for o in orders if o["order_id"] == order_ids["tp_order_id"]), None)

        # If SL executed, cancel TP
        if sl_status and sl_status["status"] == "COMPLETE":
            if tp_status and tp_status["status"] in ("OPEN", "TRIGGER PENDING"):
                kite.cancel_order("regular", order_ids["tp_order_id"])
                logging.info(f"[OCO] SL hit → cancelled TP ({order_ids['tp_order_id']})")

        # If TP executed, cancel SL
        elif tp_status and tp_status["status"] == "COMPLETE":
            if sl_status and sl_status["status"] in ("OPEN", "TRIGGER PENDING"):
                kite.cancel_order("regular", order_ids["sl_order_id"])
                logging.info(f"[OCO] TP hit → cancelled SL ({order_ids['sl_order_id']})")

    except Exception as e:
        logging.error(f"Error in OCO monitor: {e}")

# -------------------- MAIN LOOP --------------------
def run_consecutive_candle_strategy():
    logging.info("Starting strategy run (MIS-only with visible SL/TP orders). DRY_RUN=%s", DRY_RUN)

    market_start, market_end, safety_exit = setup_session_times()
    if datetime.now() < market_start:
        wait_until(market_start.time())

    fut = get_nearest_nifty_fut(instrument_list)
    if not fut:
        logging.error("No NIFTY future instrument found - aborting")
        return

    fut_token = fut.get("instrument_token")
    fut_symbol = fut.get("tradingsymbol")
    fut_expiry = fut.get("expiry")

    state = init_state()

    # main loop
    while datetime.now() < market_end:
        # ---- Safety exit ----
        if datetime.now() >= safety_exit:
            if state["in_trade"]:
                logging.warning("Safety time reached - performing square-off")
                exit_trade_mis(state["option_symbol"], state["option_lot_size"])
            break

        # ---- Daily loss guard ----
        if state["realized_pnl"] <= -abs(MAX_DAILY_LOSS):
            logging.critical("Max daily loss reached (%.2f). Forcing square-off and stopping.", state["realized_pnl"])
            if state.get("in_trade"):
                exit_trade_mis(state["option_symbol"], state["option_lot_size"])
            break

        # ---- Monitor OCO SL/TP orders ----
        if state.get("order_ids"):
            monitor_and_cancel_leftover(kite, state["order_ids"])

        # ---- External/manual exit detection ----
        try:
            pos = retry_kite_call(kite.positions, retries=2, delay=0.5)
            net_positions = pos.get("net", []) if isinstance(pos, dict) else []
            if state.get("in_trade") and state.get("option_symbol"):
                found = next((p for p in net_positions if p.get("tradingsymbol") == state["option_symbol"]), None)
                if not found or int(found.get("quantity", 0)) == 0:
                    # position gone — treat as external exit
                    exit_ltp = get_ltp_retry(kite, f"NFO:{state['option_symbol']}", retries=1, delay=0.3) or 0.0
                    entry = state.get("option_entry_price") or 0.0
                    lot = int(state.get("option_lot_size") or QUANTITY)
                    pnl = (entry - exit_ltp) * lot
                    state["realized_pnl"] += pnl
                    state["unrealized_pnl"] = 0.0
                    state["in_trade"] = False
                    state["order_ids"] = None
                    if exit_ltp > entry:
                        state["second_entry_unlocked"] = True
                    logging.info("External/manual exit detected for %s | exit=%.2f | pnl=%.2f",
                                 state["option_symbol"], exit_ltp, pnl)
        except Exception as e:
            logging.debug("positions check failed: %s", e)

        # ---- Candle logic ----
        aligned = align_to_5min(datetime.now())
        c_start = aligned - timedelta(minutes=CANDLE_INTERVAL)
        c_end = aligned

        if state.get("last_logged_candle_end") == c_end:
            time.sleep(1)
            continue

        candle = fetch_candle(kite, fut_token, c_start, c_end)
        if not candle:
            time.sleep(1)
            continue

        # Mark processed
        state["last_logged_candle_end"] = c_end

        o, h, l, c, v = candle["open"], candle["high"], candle["low"], candle["close"], int(candle["volume"])
        ctype = "GREEN" if c > o else "RED" if c < o else "DOJI"

        # Update state arrays
        state["candle_types"].append(ctype)
        state["volumes"].append(v)
        state["closes"].append(c)
        state["raw_candles"].append({"open": o, "high": h, "low": l, "close": c, "volume": v})
        if len(state["candle_types"]) > 4:
            state["candle_types"].pop(0); state["volumes"].pop(0)
            state["closes"].pop(0); state["raw_candles"].pop(0)

        # Snapshot PnL
        realized_pnl = state.get("realized_pnl", 0.0)
        unrealized_pnl = state.get("unrealized_pnl", 0.0)

        logging.info(
            "Candle %s-%s | O=%.2f H=%.2f L=%.2f C=%.2f V=%d | Type=%s | FUT=%s Exp=%s "
            "| Position=%s | PnL : Realized=%.2f, Unrealized=%.2f | Entries=%d | ReEntry=%s",
            c_start.strftime("%H:%M:%S"),
            c_end.strftime("%H:%M:%S"),
            o, h, l, c, v, ctype,
            fut_symbol,
            fut_expiry.strftime("%Y-%m-%d") if hasattr(fut_expiry, "strftime") else fut_expiry,
            f"SHORT {state['option_type']}" if state.get("in_trade") else "None",
            realized_pnl,
            unrealized_pnl,
            state.get("entries_taken", 0),
            state.get("second_entry_unlocked", False)
        )

        # ---- Entry detection ----
        action = match_pattern(state["candle_types"], state["volumes"])
        if action and not state["in_trade"] and state["entries_taken"] < MAX_ENTRIES:
            if state["entries_taken"] == 1 and not state["second_entry_unlocked"]:
                logging.info("Second entry locked, skipping this signal.")
            else:
                entry_underlying = state["closes"][-1]
                trade_side = "BUY_CE" if action == "SHORT_CE" else "BUY_PE"

                # Underlying SL/TP (for reporting)
                sl_u, tp_u = calculate_sl_tp([state["raw_candles"][-1]], entry_underlying, trade_side)

                atm_strike = round_to_strike(entry_underlying, STRIKE_INTERVAL)
                leg = action.split("_")[-1]  # CE or PE
                df_chain = option_chain(instrument_list, UNDERLYING, entry_underlying, duration=0, window=5, leg=leg)
                if df_chain.empty:
                    logging.warning("Option chain empty at entry - skipping.")
                    time.sleep(1)
                    continue

                atm_rows = df_chain[df_chain["strike"] == atm_strike]
                atm_opt = atm_rows.iloc[0] if not atm_rows.empty else df_chain.iloc[0]

                lot = safe_get_lot_size(atm_opt, default=QUANTITY)
                tradingsymbol = atm_opt["tradingsymbol"]

                # LTP for option premium
                entry_premium = get_ltp_retry(kite, f"NFO:{tradingsymbol}", retries=3, delay=0.5) or 0.0

                # SL/TP on premium
                sl_price_option = round(entry_premium * (1 + SL_PERCENT / 100), 2)
                tp_price_option = round(entry_premium * (1 - TARGET_PERCENT / 100), 2)

                # Place entry + visible SL/TP orders
                order_ids = place_mis_entry_with_sl_tp(kite, tradingsymbol, lot, sl_price_option, tp_price_option)
                if not order_ids:
                    logging.error("Entry failed for %s - skipping", tradingsymbol)
                    time.sleep(1)
                    continue

                # Update state
                state["in_trade"] = True
                state["option_type"] = leg
                state["option_symbol"] = tradingsymbol
                state["option_entry_price"] = entry_premium
                state["option_lot_size"] = lot
                state["sl_price_option"] = sl_price_option
                state["tp_price_option"] = tp_price_option
                state["sl_underlying"] = sl_u
                state["tp_underlying"] = tp_u
                state["entries_taken"] += 1
                state["second_entry_unlocked"] = False
                state["order_ids"] = order_ids

                logging.info(
                    "[ENTRY] %s | %s | entry_underlying=%.2f entry_premium=%.2f "
                    "SL_opt=%.2f TP_opt=%.2f SL_u=%.2f TP_u=%.2f lot=%s",
                    action, tradingsymbol, entry_underlying, entry_premium,
                    sl_price_option, tp_price_option, sl_u, tp_u, lot
                )

        time.sleep(1)

    # ---- End of day cleanup ----
    if state.get("in_trade"):
        logging.warning("Session ending - final safety square-off")
        exit_trade_mis(state["option_symbol"], state["option_lot_size"])

    logging.info("Strategy finished. realized_pnl=%.2f unrealized_pnl=%.2f entries_taken=%d",
                 state["realized_pnl"], state["unrealized_pnl"], state["entries_taken"])

if __name__ == "__main__":
    try:
        run_consecutive_candle_strategy()
    except Exception as e:
        logging.exception("Strategy crashed: %s", e)

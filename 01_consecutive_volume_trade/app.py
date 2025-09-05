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
    fetch_candle, round_to_strike, option_chain,
    placeMarketOrder, get_nearest_nifty_fut, safe_get_lot_size,
    normalize_gtt_id
)

# -------------------- CONFIG & LOGGING --------------------
load_dotenv(".env")
API_KEY = os.getenv("API_KEY")
ACCESS_TOKEN = os.getenv("ACCESS_TOKEN")
QUANTITY = int(os.getenv("QUANTITY", "75"))
CANDLE_START = to_time_obj(os.getenv("CANDLE_START", "09:15"))
CANDLE_END = to_time_obj(os.getenv("CANDLE_END", "15:15"))
STRIKE_INTERVAL = int(os.getenv("STRIKE_INTERVAL", 50))
MAX_RETRIES = int(os.getenv("MAX_RETRIES", 3))
RETRY_DELAY = float(os.getenv("RETRY_DELAY", 1.0))
CANDLE_PATTERNS = json.loads(os.getenv("CANDLE_PATTERNS", "[]") or "[]")
MAX_DAILY_LOSS = float(os.getenv("MAX_DAILY_LOSS", "5000"))
SAFETY_EXIT_HOUR = int(os.getenv("SAFETY_EXIT_HOUR", "15"))
SAFETY_EXIT_MIN = int(os.getenv("SAFETY_EXIT_MIN", "25"))
MAX_DAILY_LOSS = float(os.getenv("MAX_DAILY_LOSS", "5000"))
CANDLE_INTERVAL = int(os.getenv("CANDLE_INTERVAL", "5"))

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
        "gtt_id": None,
        "entries_taken": 0, "second_entry_unlocked": False,
        "last_logged_candle_end": None,
        "realized_pnl": 0.0, "unrealized_pnl": 0.0
    }

# -------------------- CORE HELPERS --------------------
def calculate_sl_tp(candles, entry_price, trade_side, buffer=0.5, rr_ratio=2):
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

# -------------------- ORDER & GTT FLOW --------------------
def place_entry_and_exits(kite, atm_opt, lot_size, sl_u, tp_u, entry_underlying, state):
    """
    Place entry and an exit plan:
     - Entry: MIS SELL (market)
     - Attempt NRML GTT OCO (preferred)
     - If GTT fails, place fallback MIS LIMIT (TP) + SL-M
    Returns: entry_resp, gtt_id (or None)
    """
    tradingsymbol = atm_opt["tradingsymbol"]
    entry_resp = None
    gtt_id = None

    # Entry (MIS SELL) - respect DRY_RUN
    if DRY_RUN:
        logging.info("[DRY_RUN] Would place MIS SELL entry: %s qty=%s", tradingsymbol, lot_size)
        entry_resp = {"mock_order_id": "DRY_ENTRY_1"}
    else:
        try:
            entry_resp = place_order_retry(
                kite,
                retries=MAX_RETRIES,
                delay=RETRY_DELAY,
                variety="regular",
                exchange="NFO",
                tradingsymbol=tradingsymbol,
                transaction_type="SELL",
                quantity=lot_size,
                order_type="MARKET",
                product="MIS"
            )
            logging.info("Entry placed: %s", entry_resp)
        except Exception as e:
            logging.error("Entry failed for %s: %s", tradingsymbol, e)
            return None, None

    # Fetch entry premium (best effort)
    entry_premium = None
    try:
        if not DRY_RUN:
            entry_premium = get_ltp_retry(kite, f"NFO:{tradingsymbol}", retries=2, delay=0.5)
        else:
            entry_premium = 0.0
    except Exception:
        entry_premium = 0.0
    state["option_entry_price"] = entry_premium

    # Try NRML GTT OCO for exits (preferred for OCO behavior)
    if not DRY_RUN:
        try:
            gtt_resp = kite.place_gtt(
                trigger_type=kite.GTT_OCO,
                tradingsymbol=tradingsymbol,
                exchange="NFO",
                trigger_values=[sl_u, tp_u],
                last_price=entry_premium or entry_underlying,
                orders=[
                    {"transaction_type": "BUY", "quantity": lot_size, "order_type": "SL-M", "product": "NRML"},
                    {"transaction_type": "BUY", "quantity": lot_size, "order_type": "LIMIT", "price": tp_u, "product": "NRML"},
                ]
            )
            gtt_id = normalize_gtt_id(gtt_resp)
            logging.info("Placed GTT OCO id=%s for %s (SL=%s TP=%s)", gtt_id, tradingsymbol, sl_u, tp_u)
        except Exception as e:
            logging.warning("GTT OCO placement failed for %s: %s", tradingsymbol, e)
            gtt_id = None

    else:
        logging.info("[DRY_RUN] Would place NRML GTT OCO for %s SL=%s TP=%s", tradingsymbol, sl_u, tp_u)
        gtt_id = "DRY_GTT_1"

    # If GTT failed (or DRY_RUN False but failed), fallback: place MIS SL and TP (non-OCO)
    if not gtt_id and not DRY_RUN:
        try:
            kite.place_order(tradingsymbol=tradingsymbol, exchange="NFO", transaction_type="BUY",
                             quantity=lot_size, order_type="LIMIT", price=tp_u, product="MIS", variety="regular")
            kite.place_order(tradingsymbol=tradingsymbol, exchange="NFO", transaction_type="BUY",
                             quantity=lot_size, order_type="SL-M", trigger_price=sl_u, product="MIS", variety="regular")
            logging.info("Placed fallback MIS TP (LIMIT) and SL (SL-M) for %s", tradingsymbol)
        except Exception as e:
            logging.error("Fallback MIS SL/TP placement failed for %s: %s", tradingsymbol, e)
    elif not gtt_id and DRY_RUN:
        logging.info("[DRY_RUN] Would place fallback MIS TP and SL for %s", tradingsymbol)

    return entry_resp, gtt_id

def cancel_gtt_safe(kite, gtt_id):
    if not gtt_id or gtt_id.startswith("DRY"):
        return
    try:
        kite.delete_gtt(gtt_id)
        logging.info("Cancelled GTT id=%s", gtt_id)
    except Exception as e:
        logging.warning("Failed to cancel GTT id=%s: %s", gtt_id, e)

# -------------------- EXIT DETECTION & PNL --------------------
def detect_external_exit_and_update_pnl(kite, state):
    """
    If our option position is no longer present in positions, treat it as an exit.
    Compute a realized PnL estimate using latest LTP.
    Note: for exact accounting use order fills/trade reports from Kite.
    """
    try:
        pos_resp = kite.positions()
        net = pos_resp.get("net", []) if isinstance(pos_resp, dict) else []
        sym = state.get("option_symbol")
        if not sym:
            return {"exited": False}

        p = next((x for x in net if x.get("tradingsymbol") == sym), None)
        if not p or int(p.get("quantity", 0)) == 0:
            # closed
            exit_price = get_ltp_retry(kite, f"NFO:{sym}", retries=1, delay=0.3) or 0.0
            entry = state.get("option_entry_price") or 0.0
            lot = int(state.get("option_lot_size") or QUANTITY)
            realized = (entry - exit_price) * lot  # short: profit if entry>exit
            state["realized_pnl"] += realized
            state["unrealized_pnl"] = 0.0
            # determine if SL (loss) roughly
            was_sl = exit_price > entry
            logging.info("Detected external exit for %s entry=%.2f exit=%.2f realized=%.2f", sym, entry, exit_price, realized)
            # cleanup GTT
            if state.get("gtt_id"):
                cancel_gtt_safe(kite, state["gtt_id"])
                state["gtt_id"] = None
            state["in_trade"] = False
            if was_sl:
                state["second_entry_unlocked"] = True
            return {"exited": True, "exit_price": exit_price, "was_sl": was_sl}
        else:
            # still present => update unrealized
            ltp = get_ltp_retry(kite, f"NFO:{sym}", retries=1, delay=0.3)
            if ltp is not None and state.get("option_entry_price") is not None:
                state["unrealized_pnl"] = (state["option_entry_price"] - ltp) * int(state.get("option_lot_size") or QUANTITY)
            return {"exited": False}
    except Exception as e:
        logging.warning("detect_external_exit_and_update_pnl error: %s", e)
        return {"exited": False}

# -------------------- SAFETY SQUARE-OFF --------------------
def safety_squareoff(kite, state):
    logging.warning("Safety square-off: cancelling GTTs and squaring MIS positions")
    try:
        if state.get("gtt_id"):
            cancel_gtt_safe(kite, state["gtt_id"])
            state["gtt_id"] = None

        pos_resp = kite.positions()
        for pos in pos_resp.get("net", []):
            if pos.get("product") == "MIS" and int(pos.get("quantity", 0)) != 0:
                qty = abs(int(pos["quantity"]))
                # close by placing opposite
                transaction = "BUY" if pos["quantity"] < 0 else "SELL"
                if DRY_RUN:
                    logging.info("[DRY_RUN] Would square-off %s qty=%s", pos["tradingsymbol"], qty)
                else:
                    try:
                        kite.place_order(tradingsymbol=pos["tradingsymbol"], exchange=pos["exchange"],
                                         transaction_type=transaction, quantity=qty, order_type="MARKET",
                                         product="MIS", variety="regular")
                        logging.info("Squared off %s qty=%s", pos["tradingsymbol"], qty)
                    except Exception as e:
                        logging.error("Failed to square-off %s : %s", pos["tradingsymbol"], e)
    except Exception as e:
        logging.error("safety_squareoff failed: %s", e)

# -------------------- SESSION SETUP --------------------
def setup_session_times():
    market_start = datetime.now().replace(hour=CANDLE_START.hour, minute=CANDLE_START.minute, second=0, microsecond=0)
    market_end = datetime.now().replace(hour=CANDLE_END.hour, minute=CANDLE_END.minute, second=0, microsecond=0)
    safety_exit = datetime.now().replace(hour=SAFETY_EXIT_HOUR, minute=SAFETY_EXIT_MIN, second=0, microsecond=0)
    return market_start, market_end, safety_exit

# -------------------- MAIN LOOP --------------------
def run_consecutive_candle_strategy():
    logging.info("run_consecutive_candle_strategy starting (DRY_RUN=%s)", DRY_RUN)
    market_start, market_end, safety_exit = setup_session_times()
    if datetime.now() < market_start:
        wait_until(market_start.time())

    fut = get_nearest_nifty_fut(instrument_list)
    fut_token, fut_symbol, fut_expiry = fut["instrument_token"], fut["tradingsymbol"], fut["expiry"]
    if not fut:
        logging.error("No NIFTY future found - aborting")
        return
    fut_token = fut["instrument_token"]

    state = init_state()

    while datetime.now() < market_end:
        # safety check
        if datetime.now() >= safety_exit:
            if state["in_trade"]:
                logging.warning("Safety time reached - performing square-off")
                safety_squareoff(kite, state)
            break

        # detect external exit first
        if state["in_trade"]:
            res = detect_external_exit_and_update_pnl(kite, state)
            if res.get("exited"):
                logging.info("Trade closed externally (realized=%.2f). entries_taken=%d", state["realized_pnl"], state["entries_taken"])
                time.sleep(1)
                continue

        # fetch candle (once per candle close)
        aligned = align_to_5min(datetime.now())
        c_start = aligned - timedelta(minutes=5)
        c_end = aligned

        # ---- skip duplicate logs ----
        if state["last_logged_candle_end"] == c_end:
            time.sleep(1)
            continue

        # mark candle as processed now
        state["last_logged_candle_end"] = c_end

        # fetch candle
        candle = fetch_candle(kite, fut_token, c_start, c_end, interval_str=f"{CANDLE_INTERVAL}minute")
        if not candle:
            time.sleep(1)
            continue

        o, h, l, c, v = candle["open"], candle["high"], candle["low"], candle["close"], candle["volume"]
        ctype = "GREEN" if c > o else "RED" if c < o else "DOJI"

        # update state
        state["candle_types"].append(ctype)
        state["volumes"].append(v)
        state["closes"].append(c)
        state["raw_candles"].append({"open": o, "high": h, "low": l, "close": c, "volume": v})
        if len(state["candle_types"]) > 4:
            state["candle_types"].pop(0); state["volumes"].pop(0)
            state["closes"].pop(0); state["raw_candles"].pop(0)

        # ---- PnL snapshot ----
        realized_pnl = state.get("realized_pnl", 0.0)
        unrealized_pnl = 0.0
        if state.get("in_trade") and state.get("option_symbol") and state.get("option_entry_price"):
            opt_ltp = get_ltp_retry(kite, f"NFO:{state['option_symbol']}")
            if opt_ltp is not None and state.get("option_lot_size"):
                unrealized_pnl = (state["option_entry_price"] - opt_ltp) * state["option_lot_size"]

        # ---- Logging ----
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

        # detect signal
        action = match_pattern(state["candle_types"], state["volumes"])
        if action and not state["in_trade"] and state["entries_taken"] < 2:
            if state["entries_taken"] == 1 and not state["second_entry_unlocked"]:
                logging.info("Second entry locked after first trade SL not hit -> skipping")
            else:
                entry_underlying = state["closes"][-1]
                trade_side = "BUY_CE" if action == "SHORT_CE" else "BUY_PE"
                sl_u, tp_u = calculate_sl_tp([state["raw_candles"][-1]], entry_underlying, trade_side)

                atm_strike = round_to_strike(entry_underlying, STRIKE_INTERVAL)
                leg = action.split("_")[-1]  # "CE" or "PE"
                df_chain = option_chain(instrument_list, "NIFTY", entry_underlying, duration=0, window=5, leg=leg)
                if df_chain.empty:
                    logging.warning("No option contracts found around ATM - skipping entry")
                    continue

                atm_rows = df_chain[df_chain["strike"] == atm_strike]
                if atm_rows.empty:
                    # pick closest strike
                    try:
                        idx = (df_chain["strike"] - entry_underlying).abs().argmin()
                        atm_opt = df_chain.iloc[idx]
                    except Exception:
                        atm_opt = df_chain.iloc[0]
                else:
                    atm_opt = atm_rows.iloc[0]

                lot = safe_get_lot_size(atm_opt, default=QUANTITY)

                # place entry + exit orders
                entry_resp, gtt_id = place_entry_and_exits(kite, atm_opt, lot, sl_u, tp_u, entry_underlying, state)
                if not entry_resp:
                    logging.error("Entry placement failed for %s - skipping", atm_opt.get("tradingsymbol"))
                    continue

                # update state
                state["in_trade"] = True
                state["option_symbol"] = atm_opt["tradingsymbol"]
                state["option_lot_size"] = lot
                state["sl_underlying"] = sl_u
                state["tp_underlying"] = tp_u
                state["gtt_id"] = gtt_id
                state["entries_taken"] += 1
                state["second_entry_unlocked"] = False
                logging.info("Entered trade %s | %s | entry_underlying=%.2f SL=%.2f TP=%.2f lot=%s", action, state["option_symbol"], entry_underlying, sl_u, tp_u, lot)

        # update unrealized pnl
        if state["in_trade"] and state.get("option_symbol"):
            ltp = get_ltp_retry(kite, f"NFO:{state['option_symbol']}", retries=1, delay=0.4)
            if ltp is not None and state.get("option_entry_price") is not None:
                state["unrealized_pnl"] = (state["option_entry_price"] - ltp) * int(state.get("option_lot_size") or QUANTITY)

        time.sleep(0.5)

    # cleanup at end of day
    if state["in_trade"]:
        logging.warning("End of session - performing final safety square-off")
        safety_squareoff(kite, state)

    logging.info("Strategy finished. realized_pnl=%.2f unrealized_pnl=%.2f entries_taken=%d", state["realized_pnl"], state["unrealized_pnl"], state["entries_taken"])

if __name__ == "__main__":
    try:
        run_consecutive_candle_strategy()
    except Exception as e:
        logging.exception("Strategy crashed: %s", e)

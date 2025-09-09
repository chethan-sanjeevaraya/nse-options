# t6_multi_strategy.py
"""
Multi-strategy intraday option buying framework (modular).
Drop-in production-ready file that uses helpers from common_utils.py.

ENABLED_STRATEGIES="ORB,MARUBOZU,ENGULFING,INSIDEBAR,GAPREV"
"""

import os
import logging
import json
import time
from datetime import datetime, timedelta
from dotenv import load_dotenv
from typing import Optional, Dict, Any, List

from kiteconnect import KiteConnect

# --- required helpers in your common_utils.py ---
from common_utils import (to_time_obj, wait_until, align_to_5min,retry_kite_call, 
                          place_order_retry, get_ltp_retry,fetch_candle, round_to_strike, 
                          option_chain,get_nearest_nifty_fut, safe_get_lot_size)

load_dotenv(".env")

# ---------------- CONFIG ----------------
API_KEY = os.getenv("API_KEY")
ACCESS_TOKEN = os.getenv("ACCESS_TOKEN")

DRY_RUN = os.getenv("DRY_RUN", "true").lower() in ("1", "true", "yes")
QUANTITY = int(os.getenv("QUANTITY", "75"))
STRIKE_INTERVAL = int(os.getenv("STRIKE_INTERVAL", "50"))
OTM_STEPS = int(os.getenv("OTM_STEPS", "0"))
CANDLE_INTERVAL = int(os.getenv("CANDLE_INTERVAL", "5"))

CANDLE_START = to_time_obj(os.getenv("CANDLE_START", "09:15"))
CANDLE_END = to_time_obj(os.getenv("CANDLE_END", "15:15"))

ORB_START = to_time_obj(os.getenv("ORB_START", "09:15"))
ORB_END = to_time_obj(os.getenv("ORB_END", "09:30"))

SAFETY_EXIT = to_time_obj(os.getenv("SAFETY_EXIT", "15:15"))

MAX_DAILY_LOSS = float(os.getenv("MAX_DAILY_LOSS", "5000"))
PREMIUM_SL_PCT = float(os.getenv("PREMIUM_SL_PCT", "50"))   # percent drop for premium SL
PREMIUM_TP_PCT = float(os.getenv("PREMIUM_TP_PCT", "100"))  # percent gain for premium TP

MAX_RETRIES = int(os.getenv("MAX_RETRIES", "3"))
RETRY_DELAY = float(os.getenv("RETRY_DELAY", "1.0"))

UNDERLYING = os.getenv("UNDERLYING", "NIFTY")

ENABLED_STRATEGIES = os.getenv("ENABLED_STRATEGIES", "ORB").upper().split(",")
ANCHOR_CANDLE_MODE = os.getenv("ANCHOR_CANDLE_MODE", "5MIN").upper()
M_REVERSAL_TOLERANCE = float(os.getenv("M_REVERSAL_TOLERANCE", 0.002))  # 0.2% default
W_REVERSAL_TOLERANCE = float(os.getenv("W_REVERSAL_TOLERANCE", 0.002))  # 0.2% default

LOG_FILE = os.getenv("LOG_FILE", "orb_multi_strategy.log")

logging.basicConfig(filename=LOG_FILE, level=logging.INFO,
                    format="%(asctime)s | %(levelname)s | %(message)s", force=True)
console = logging.StreamHandler()
console.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(message)s", "%Y-%m-%d %H:%M:%S"))
logging.getLogger().addHandler(console)
logging.info("t6_multi_strategy starting (DRY_RUN=%s) ENABLED=%s", DRY_RUN, ENABLED_STRATEGIES)

# ---------------- KITE INIT ----------------
if not API_KEY or not ACCESS_TOKEN:
    logging.error("API_KEY or ACCESS_TOKEN missing in .env")
    raise SystemExit("Missing Kite credentials")

kite = KiteConnect(api_key=API_KEY)
kite.set_access_token(ACCESS_TOKEN)

# fetch static instruments once
try:
    instrument_list = retry_kite_call(kite.instruments, "NFO", retries=2, delay=1) or []
except Exception as e:
    logging.error("Failed to fetch instruments list: %s", e)
    instrument_list = []

# ---------------- STATE ----------------
def init_state() -> Dict[str, Any]:
    return {
        # ORB / candle store
        "raw_candles": [],  # list of {'open','high','low','close','volume','start','end'}
        "orb_built": False, "orb_high": None, "orb_low": None, "orb_open": None, "orb_close": None,

        # trade state
        "in_trade": False,
        "option_symbol": None, # "NIFTY25SEP24500CE"
        "option_lot": None,
        "option_entry_premium": None,
        "entry_underlying": None,
        "sl_underlying": None,
        "tp_underlying": None,
        "order_meta": None,  # optional info (order ids from Kite API, for debugging/cancellation).

        # attribution
        "strategy_name": None,    # e.g. "ORB", "MARUBOZU", "ENGULFING", etc.
        "signal_trigger": None,   # price level that triggered entry
        "entry_side": None,       # "CE" or "PE"

        # PnL
        "realized_pnl": 0.0, "unrealized_pnl": 0.0,

        # bookkeeping
        "entry_time": None
    }

# ---------------- HELPERS ---------------
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
    
def manage_open_position(state: Dict[str, Any], fut_token):
    """
    Poll underlying & option premium; execute market exit when SL/TP hit (underlying or premium).
    Update realized/unrealized pnl.
    """
    if not state.get("in_trade"):
        return

    symbol = state["option_symbol"]
    lot = int(state["option_lot"] or QUANTITY)
    entry_premium = state.get("option_entry_premium")

    opt_ltp = get_ltp_retry(kite, f"NFO:{symbol}", retries=2, delay=0.4)
    u_ltp = get_ltp_retry(kite, fut_token, retries=2, delay=0.4)

    # premium-based exits
    premium_sl = state.get("premium_sl")
    premium_tp = state.get("premium_tp")
    if opt_ltp is not None:
        if premium_sl is not None and opt_ltp <= premium_sl:
            ok = exit_position_market(symbol, lot)
            if ok:
                pnl = (opt_ltp - entry_premium) * lot
                state["realized_pnl"] += pnl
                state["in_trade"] = False
                logging.info("[PREM SL] Exited %s at %.2f pnl=%.2f", symbol, opt_ltp, pnl)
                return
        if premium_tp is not None and opt_ltp >= premium_tp:
            ok = exit_position_market(symbol, lot)
            if ok:
                pnl = (opt_ltp - entry_premium) * lot
                state["realized_pnl"] += pnl
                state["in_trade"] = False
                logging.info("[PREM TP] Exited %s at %.2f pnl=%.2f", symbol, opt_ltp, pnl)
                return

    # underlying-based exits
    if u_ltp is not None and state.get("sl_underlying") is not None and state.get("tp_underlying") is not None:
        if symbol.endswith("CE"):
            # CE long: SL if underlying <= sl_underlying; TP if >= tp_underlying
            if u_ltp <= state["sl_underlying"]:
                ok = exit_position_market(symbol, lot)
                if ok:
                    pnl = (opt_ltp - entry_premium) * lot if opt_ltp is not None else 0.0
                    state["realized_pnl"] += pnl
                    state["in_trade"] = False
                    logging.info("[SL U] Exited %s at underlying=%.2f pnl=%.2f", symbol, u_ltp, pnl)
                    return
            if u_ltp >= state["tp_underlying"]:
                ok = exit_position_market(symbol, lot)
                if ok:
                    pnl = (opt_ltp - entry_premium) * lot if opt_ltp is not None else 0.0
                    state["realized_pnl"] += pnl
                    state["in_trade"] = False
                    logging.info("[TP U] Exited %s at underlying=%.2f pnl=%.2f", symbol, u_ltp, pnl)
                    return
        elif symbol.endswith("PE"):
            # PE long: SL if underlying >= sl_underlying; TP if <= tp_underlying
            if u_ltp >= state["sl_underlying"]:
                ok = exit_position_market(symbol, lot)
                if ok:
                    pnl = (opt_ltp - entry_premium) * lot if opt_ltp is not None else 0.0
                    state["realized_pnl"] += pnl
                    state["in_trade"] = False
                    logging.info("[SL U] Exited %s at underlying=%.2f pnl=%.2f", symbol, u_ltp, pnl)
                    return
            if u_ltp <= state["tp_underlying"]:
                ok = exit_position_market(symbol, lot)
                if ok:
                    pnl = (opt_ltp - entry_premium) * lot if opt_ltp is not None else 0.0
                    state["realized_pnl"] += pnl
                    state["in_trade"] = False
                    logging.info("[TP U] Exited %s at underlying=%.2f pnl=%.2f", symbol, u_ltp, pnl)
                    return

    # update unrealized PnL
    if opt_ltp is not None and entry_premium is not None:
        state["unrealized_pnl"] = (opt_ltp - entry_premium) * lot
        
# ---------------- PATTERN DETECTORS ----------------
def detect_m_reversal(candles: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """
    M Reversal (Double Top):
    - Price tests the first 1H high (pivot zone).
    - Entry short when bearish reversal condition fires near that level.
    - Conditions: Bearish Engulfing, 3 Black Crows, or Volume breakout.
    """
    if len(candles) < 5:
        return None

    # Step 1: Define the first 1H high (first 12x 5-min candles)
    first_hour = candles[:12] if len(candles) >= 12 else candles
    one_hr_high = max(c["high"] for c in first_hour)

    last = candles[-1]
    prev = candles[-2]
    conditions = []
    entry_price = None

    # --- Condition 1: Bearish Engulfing with volume ---
    if prev["close"] > prev["open"] and last["close"] < last["open"]:
        if last["close"] < prev["low"] and last["open"] > prev["high"]:
            if last.get("volume", 0) > prev.get("volume", 0):
                conditions.append("ENGULFING")
                entry_price = last["close"]

    # --- Condition 2: Three Black Crows + green confirmation ---
    if len(candles) >= 4:
        c1, c2, c3, c4 = candles[-4], candles[-3], candles[-2], candles[-1]
        if (c1["close"] < c1["open"] and c2["close"] < c2["open"] and c3["close"] < c3["open"]):
            if c4["close"] > c4["open"] and c4.get("volume", 0) < c3.get("volume", 0):
                conditions.append("3BC")
                entry_price = c4["close"]

    # --- Condition 3: Volume breakout ---
    if last["close"] < last["open"] and prev["close"] > prev["open"]:
        if last.get("volume", 0) > prev.get("volume", 0):
            conditions.append("VOL_BREAKOUT")
            entry_price = last["close"]

    # --- Final check: last candle near first 1H high ---
    if conditions and entry_price and abs(last["high"] - one_hr_high) <= M_REVERSAL_TOLERANCE * one_hr_high:
        swing_low = min(c["low"] for c in candles[-10:])  # recent swing low
        return {
            "side": "PE",
            "type": f"M_REVERSAL_{'+'.join(conditions)}",
            "conditions": conditions,
            "entry": entry_price,
            "sl": one_hr_high,
            "tp": swing_low
        }

    return None

def detect_w_reversal(candles: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """
    W Reversal (Double Bottom):
    - Price tests the first 1H low (pivot zone).
    - Entry long when bullish reversal condition fires near that level.
    - Conditions: Bullish Engulfing, 3 White Soldiers, or Volume breakout.
    """
    if len(candles) < 5:
        return None

    # Step 1: Define the first 1H low (using first 12x 5-min candles)
    first_hour = candles[:12] if len(candles) >= 12 else candles
    one_hr_low = min(c["low"] for c in first_hour)

    last = candles[-1]
    prev = candles[-2]
    conditions = []
    entry_price = None

    # --- Condition 1: Bullish Engulfing with volume ---
    if prev["close"] < prev["open"] and last["close"] > last["open"]:
        if last["close"] > prev["high"] and last["open"] < prev["low"]:
            if last.get("volume", 0) > prev.get("volume", 0):
                conditions.append("ENGULFING")
                entry_price = last["close"]

    # --- Condition 2: Three green with red confirmation ---
    if len(candles) >= 4:
        c1, c2, c3, c4 = candles[-4], candles[-3], candles[-2], candles[-1]
        if (c1["close"] > c1["open"] and c2["close"] > c2["open"] and c3["close"] > c3["open"]):
            if c4["close"] < c4["open"] and c4.get("volume", 0) < c3.get("volume", 0):
                conditions.append("3WS")
                entry_price = c4["close"]

    # --- Condition 3: Volume breakout ---
    if last["close"] > last["open"] and prev["close"] < prev["open"]:
        if last.get("volume", 0) > prev.get("volume", 0):
            conditions.append("VOL_BREAKOUT")
            entry_price = last["close"]

    # --- Final Check: near first 1H low ---
    if conditions and entry_price and abs(last["low"] - one_hr_low) <= W_REVERSAL_TOLERANCE * one_hr_low:
        swing_high = max(c["high"] for c in candles[-10:])  # recent swing high
        return {
            "side": "CE",
            "type": "W_REVERSAL",
            "conditions": conditions,
            "entry": entry_price,
            "sl": one_hr_low,        # SL below 1H low
            "tp": swing_high         # Target back to swing high or 2R (handled outside)
        }

    return None

import logging
from typing import List, Dict, Any, Optional

def detect_cbo_reversal(
    candles: List[Dict[str, Any]],
    u_ltp: float,
    color: str = "RED",
    anchor_mode: str = "5MIN"
) -> Optional[Dict[str, Any]]:
    """
    Generic Candle Breakout Reversal/Continuation Strategy (RCBO & GCBO).
    
    Args:
        candles: List of 5-min OHLCV dicts.
        u_ltp: Current underlying LTP.
        color: "RED" for RCBO, "GREEN" for GCBO.
        anchor_mode: "5MIN" or "15MIN" anchor for first candle.
    
    Logic:
    - Anchor = first 5-min or 15-min candle (based on color).
    - Case 1a: Reversal with volume confirmation candle.
    - Case 1b: Reversal with 3 soldiers/crows + opposite confirmation.
    - Case 2: Continuation breakout if price breaks anchor + confirm candle.
    """

    if len(candles) < 4:
        logging.debug(f"CBO[{color}]: Not enough candles yet.")
        return None

    # ---------------- Anchor Selection ----------------
    if anchor_mode == "15MIN":
        anchor_candles = candles[:3]
    else:
        anchor_candles = candles[:1]

    if color == "RED":
        first_anchor = next((c for c in anchor_candles if c["close"] < c["open"]), None)
    else:  # GREEN
        first_anchor = next((c for c in anchor_candles if c["close"] > c["open"]), None)

    if not first_anchor:
        logging.debug(f"CBO[{color}]: No anchor candle found.")
        return None

    anchor_high, anchor_low = first_anchor["high"], first_anchor["low"]
    logging.debug(f"CBO[{color}]: Anchor selected | high={anchor_high} low={anchor_low} mode={anchor_mode}")

    # ---------------- Volume Confirmation Candle ----------------
    confirm_candle = None
    for i in range(1, len(candles)):
        prev, curr = candles[i-1], candles[i]
        if color == "RED":
            # prev green → curr red with higher volume
            if prev["close"] > prev["open"] and curr["close"] < curr["open"] and curr["volume"] > prev["volume"]:
                confirm_candle = curr
                break
        else:  # GREEN
            # prev red → curr green with higher volume
            if prev["close"] < prev["open"] and curr["close"] > curr["open"] and curr["volume"] > prev["volume"]:
                confirm_candle = curr
                break

    # ---------------- Case 1a: Reversal (Volume Confirmed) ----------------
    if confirm_candle:
        if color == "RED" and u_ltp > confirm_candle["high"] and u_ltp > anchor_low:
            logging.info(f"CBO[{color}]: REVERSAL VOLCONF triggered | u_ltp={u_ltp} > {confirm_candle['high']}")
            return {
                "side": "CE", "type": f"RCBO_REVERSAL_{anchor_mode}",
                "trigger": confirm_candle["high"], "sl": confirm_candle["low"],
                "tp": confirm_candle["high"] + 2 * (confirm_candle["high"] - confirm_candle["low"]),
                "anchor": first_anchor, "confirm": confirm_candle
            }
        if color == "GREEN" and u_ltp < confirm_candle["low"] and u_ltp < anchor_high:
            logging.info(f"CBO[{color}]: REVERSAL VOLCONF triggered | u_ltp={u_ltp} < {confirm_candle['low']}")
            return {
                "side": "PE", "type": f"GCBO_REVERSAL_{anchor_mode}",
                "trigger": confirm_candle["low"], "sl": confirm_candle["high"],
                "tp": confirm_candle["low"] - 2 * (confirm_candle["high"] - confirm_candle["low"]),
                "anchor": first_anchor, "confirm": confirm_candle
            }

    # ---------------- Case 1b: Reversal (3 Soldiers/Crows + Opp Confirmation) ----------------
    c1, c2, c3, c4 = candles[-4], candles[-3], candles[-2], candles[-1]
    if color == "RED":
        if (c1["close"] < c1["open"] and c2["close"] < c2["open"] and c3["close"] < c3["open"]):
            if c4["close"] > c4["open"] and c4["volume"] < c3["volume"] and u_ltp > c4["close"] and u_ltp > anchor_low:
                logging.info(f"CBO[{color}]: REVERSAL 3BC+GREEN triggered | u_ltp={u_ltp}")
                return {
                    "side": "CE", "type": f"RCBO_REVERSAL_3BC_{anchor_mode}",
                    "trigger": c4["close"], "sl": max(c1["high"], c2["high"], c3["high"]),
                    "tp": c4["close"] + 2 * (c4["close"] - min(c1["low"], c2["low"], c3["low"])),
                    "anchor": first_anchor, "pattern": "3BC+GREEN"
                }
    else:  # GREEN
        if (c1["close"] > c1["open"] and c2["close"] > c2["open"] and c3["close"] > c3["open"]):
            if c4["close"] < c4["open"] and c4["volume"] < c3["volume"] and u_ltp < c4["close"] and u_ltp < anchor_high:
                logging.info(f"CBO[{color}]: REVERSAL 3WS+RED triggered | u_ltp={u_ltp}")
                return {
                    "side": "PE", "type": f"GCBO_REVERSAL_3WS_{anchor_mode}",
                    "trigger": c4["close"], "sl": max(c1["high"], c2["high"], c3["high"]),
                    "tp": c4["close"] - 2 * (max(c1["high"], c2["high"], c3["high"]) - c4["close"]),
                    "anchor": first_anchor, "pattern": "3WS+RED"
                }

    # ---------------- Case 2: Continuation ----------------
    if confirm_candle:
        if color == "RED" and u_ltp < confirm_candle["low"] and u_ltp < anchor_low:
            logging.info(f"CBO[{color}]: CONTINUATION triggered | u_ltp={u_ltp} < {confirm_candle['low']}")
            return {
                "side": "CE", "type": f"RCBO_CONT_{anchor_mode}",
                "trigger": confirm_candle["low"], "sl": confirm_candle["high"],
                "tp": confirm_candle["low"] - 2 * (confirm_candle["high"] - confirm_candle["low"]),
                "anchor": first_anchor, "confirm": confirm_candle
            }
        if color == "GREEN" and u_ltp > confirm_candle["high"] and u_ltp > anchor_high:
            logging.info(f"CBO[{color}]: CONTINUATION triggered | u_ltp={u_ltp} > {confirm_candle['high']}")
            return {
                "side": "PE", "type": f"GCBO_CONT_{anchor_mode}",
                "trigger": confirm_candle["high"], "sl": confirm_candle["low"],
                "tp": confirm_candle["high"] + 2 * (confirm_candle["high"] - confirm_candle["low"]),
                "anchor": first_anchor, "confirm": confirm_candle
            }

    logging.debug(f"CBO[{color}]: No valid entry yet | u_ltp={u_ltp}")
    return None

# ---------------- DETECTOR REGISTRY ----------------
DETECTOR_MAP = {
    "M_REVERSAL": lambda state, _: detect_m_reversal(state["raw_candles"]),
    "W_REVERSAL": lambda state, _: detect_w_reversal(state["raw_candles"]),
    "RCBO": lambda state, u_ltp: detect_cbo_reversal(state["raw_candles"], u_ltp)
}

# ---------------- MODULAR STRATEGY LOOP ----------------
def run_intraday_multi_strategy():
    logging.info("Starting intraday strategy (3 patterns only: CBO, M_REVERSAL, W_REVERSAL) | DRY_RUN=%s", DRY_RUN)

    # session windows
    market_start = datetime.now().replace(hour=CANDLE_START.hour, minute=CANDLE_START.minute, second=0, microsecond=0)
    market_end = datetime.now().replace(hour=CANDLE_END.hour, minute=CANDLE_END.minute, second=0, microsecond=0)
    safety = datetime.now().replace(hour=SAFETY_EXIT.hour, minute=SAFETY_EXIT.minute, second=0, microsecond=0)

    # wait until market start
    if datetime.now() < market_start:
        wait_until(market_start.time())

    # nearest NIFTY FUT
    fut = get_nearest_nifty_fut(instrument_list)
    if not fut:
        logging.error("No NIFTY FUT found - aborting.")
        return
    fut_token = fut["instrument_token"]; fut_symbol = fut["tradingsymbol"]

    # init state
    state = init_state()

    # fetch first 5m candle for context
    c = fetch_candle(kite, fut_token, market_start, market_start + timedelta(minutes=CANDLE_INTERVAL))
    if c:
        state["raw_candles"].append({
            "open": c["open"], "high": c["high"], "low": c["low"], "close": c["close"],
            "volume": int(c.get("volume", 0)), "start": c.get("start"), "end": c.get("end")
        })
        logging.info("First candle loaded O=%.2f H=%.2f L=%.2f C=%.2f", c["open"], c["high"], c["low"], c["close"])

    logging.info("Monitoring for signals until %s", safety.strftime("%H:%M:%S"))

    # MAIN LOOP
    while datetime.now() < safety and not state["in_trade"]:
        # daily loss guard
        if state["realized_pnl"] <= -abs(MAX_DAILY_LOSS):
            logging.critical("Max daily loss reached %.2f - stopping", state["realized_pnl"])
            break

        # update LTP
        u_ltp = get_ltp_retry(kite, fut_token, retries=2, delay=0.4)
        if u_ltp is None:
            time.sleep(1)
            continue

        # update rolling candles
        new_candle = fetch_candle(kite, fut_token, datetime.now() - timedelta(minutes=CANDLE_INTERVAL), datetime.now())
        if new_candle:
            state["raw_candles"].append({
                "open": new_candle["open"], "high": new_candle["high"], "low": new_candle["low"],
                "close": new_candle["close"], "volume": int(new_candle.get("volume", 0)),
                "start": new_candle.get("start"), "end": new_candle.get("end")
            })

        # ---------------- Detect signals ----------------
        signal = None

        # 1. Candle Breakout (RCBO/GCBO unified)
        if "RCBO" in ENABLED_STRATEGIES:
            sig = detect_cbo_reversal(state["raw_candles"], u_ltp, color="RED", anchor_mode="5MIN")
            if sig:
                signal = sig
        if not signal and "GCBO" in ENABLED_STRATEGIES:
            sig = detect_cbo_reversal(state["raw_candles"], u_ltp, color="GREEN", anchor_mode="5MIN")
            if sig:
                signal = sig

        # 2. M Reversal (double top)
        if not signal and "M_REVERSAL" in ENABLED_STRATEGIES:
            sig = detect_m_reversal(state["raw_candles"])
            if sig:
                signal = sig

        # 3. W Reversal (double bottom)
        if not signal and "W_REVERSAL" in ENABLED_STRATEGIES:
            sig = detect_w_reversal(state["raw_candles"])
            if sig:
                signal = sig

        # ---------------- Take entry if signal ----------------
        if signal:
            side = signal["side"]
            state["strategy_name"] = signal.get("type")
            state["signal_trigger"] = signal.get("trigger")
            state["entry_side"] = side

            # define underlying SL/TP
            if side == "CE":
                sl_u = round(u_ltp * 0.995, 2)
                tp_u = round(u_ltp + 2 * (u_ltp - sl_u), 2)
            else:
                sl_u = round(u_ltp * 1.005, 2)
                tp_u = round(u_ltp - 2 * (sl_u - u_ltp), 2)

            state["sl_underlying"] = sl_u
            state["tp_underlying"] = tp_u

            logging.info(f"[ENTRY] {side} | {fut_symbol} | strategy={state['strategy_name']} "
                         f"trigger={state.get('signal_trigger')} | u_ltp={u_ltp:.2f} | "
                         f"sl_u={sl_u:.2f} tp_u={tp_u:.2f}")

            success = take_entry(side, u_ltp, state)
            if success:
                logging.info("Entry SUCCESS: %s %s lot=%s | SL=%.2f TP=%.2f",
                             state["option_symbol"], side, state["option_lot"], sl_u, tp_u)
                break
            else:
                logging.warning("Entry attempt FAILED. Continuing monitoring.")
        time.sleep(1)

    # ---------------- Manage position ----------------
    if not state["in_trade"]:
        logging.info("No trade taken before safety exit.")
        return

    logging.info("Managing open position until safety=%s", safety.strftime("%H:%M:%S"))
    while datetime.now() < safety and state["in_trade"]:
        if state["realized_pnl"] <= -abs(MAX_DAILY_LOSS):
            logging.critical("Max daily loss reached %.2f - forcing exit", state["realized_pnl"])
            exit_position_market(state["option_symbol"], state["option_lot"])
            state["in_trade"] = False
            break

        manage_open_position(state, fut_token)
        time.sleep(0.6)

    # cleanup
    if state["in_trade"]:
        logging.warning("Safety square-off: exiting position")
        exit_position_market(state["option_symbol"], state["option_lot"])
        state["in_trade"] = False

    logging.info("Session finished. realized_pnl=%.2f unrealized_pnl=%.2f",
                 state.get("realized_pnl", 0.0), state.get("unrealized_pnl", 0.0))

if __name__ == "__main__":
    try:
        run_intraday_multi_strategy()
    except Exception as e:
        logging.exception("Strategy crashed: %s", e)

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

# ---------------- HELPERS ----------------
def build_orb_from_first_15(kite, fut_token, start_dt: datetime, end_dt: datetime) -> Optional[Dict[str, float]]:
    """
    Build ORB using the first 3 x 5-min candles between start_dt (inclusive) and end_dt (exclusive).
    Returns dict with keys open, high, low, close.
    """
    candles = []
    t = start_dt
    while t < end_dt:
        s = t
        e = t + timedelta(minutes=CANDLE_INTERVAL)
        c = fetch_candle(kite, fut_token, s, e)
        if not c:
            logging.warning("Missing 5-min candle for %s - %s", s, e)
            return None
        if isinstance(c, list):
            candles.append(c[0])
        else:
            candles.append(c)
        t = e

    opens = [c["open"] for c in candles]
    highs = [c["high"] for c in candles]
    lows = [c["low"] for c in candles]
    closes = [c["close"] for c in candles]

    return {"open": opens[0], "high": max(highs), "low": min(lows), "close": closes[-1], "candles": candles}

def select_option_for_buy(instrument_list, underlying_price: float, leg: str, otm_steps: int = 0, strike_interval: int = STRIKE_INTERVAL):
    atm_strike = round_to_strike(underlying_price, strike_interval)
    target_strike = atm_strike + (otm_steps * strike_interval if leg == "CE" else -otm_steps * strike_interval)
    df_chain = option_chain(instrument_list, UNDERLYING, underlying_price, duration=0, window=5, leg=leg)
    if df_chain is None or df_chain.empty:
        logging.error("option_chain empty for %s at %s", leg, underlying_price)
        return None
    row = df_chain[df_chain["strike"] == target_strike]
    if row.empty:
        idx = (df_chain["strike"] - target_strike).abs().argmin()
        return df_chain.iloc[idx]
    return row.iloc[0]

def place_buy_option(tradingsymbol: str, lot: int):
    if DRY_RUN:
        logging.info("[DRY_RUN] BUY %s x %d", tradingsymbol, lot)
        return {"mock_order_id": "DRY_BUY"}
    try:
        resp = place_order_retry(
            kite, retries=MAX_RETRIES, delay=RETRY_DELAY,
            variety="regular", exchange="NFO",
            tradingsymbol=tradingsymbol, transaction_type="BUY",
            quantity=lot, order_type="MARKET", product="MIS"
        )
        logging.info("Placed BUY %s x %d resp=%s", tradingsymbol, lot, resp)
        return resp
    except Exception as e:
        logging.error("BUY failed for %s: %s", tradingsymbol, e)
        return None

def exit_position_market(tradingsymbol: str, lot: int) -> bool:
    if DRY_RUN:
        logging.info("[DRY_RUN] EXIT %s x %d", tradingsymbol, lot)
        return True
    try:
        resp = place_order_retry(
            kite, retries=MAX_RETRIES, delay=RETRY_DELAY,
            variety="regular", exchange="NFO",
            tradingsymbol=tradingsymbol, transaction_type="SELL",
            quantity=lot, order_type="MARKET", product="MIS"
        )
        logging.info("Exit SELL placed %s x %d resp=%s", tradingsymbol, lot, resp)
        return True
    except Exception as e:
        logging.error("Exit failed for %s: %s", tradingsymbol, e)
        return False

def compute_premium_sl_tp(entry_premium: float) -> tuple[Optional[float], Optional[float]]:
    if entry_premium is None:
        return None, None
    sl = round(entry_premium * (1 - PREMIUM_SL_PCT / 100), 2)
    tp = round(entry_premium * (1 + PREMIUM_TP_PCT / 100), 2)
    return sl, tp

# ---------------- PATTERN DETECTORS ----------------
def detect_orb_breakout(state: Dict[str, Any], u_ltp: float) -> Optional[Dict[str, Any]]:
    """Return action dict {'side':'CE'|'PE','type':'ORB','trigger':..} or None"""
    if not state.get("orb_built"):
        return None
    if u_ltp > state["orb_high"]:
        return {"side": "CE", "type": "ORB", "trigger": state["orb_high"]}
    if u_ltp < state["orb_low"]:
        return {"side": "PE", "type": "ORB", "trigger": state["orb_low"]}
    return None

def detect_marubozu(candle: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Detect strong marubozu in a single candle (open-of-day)."""
    if not candle:
        return None
    body = abs(candle["close"] - candle["open"])
    wick_up = candle["high"] - max(candle["close"], candle["open"])
    wick_down = min(candle["close"], candle["open"]) - candle["low"]
    # require body significantly larger than combined wicks
    if body > 2 * (wick_up + wick_down) and candle["volume"] > 0:
        side = "CE" if candle["close"] > candle["open"] else "PE"
        return {"side": side, "type": "MARUBOZU", "trigger": candle["high"] if side == "CE" else candle["low"]}
    return None

def detect_engulfing(candles: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """Bullish/Bearish engulfing on last two candles."""
    if len(candles) < 2:
        return None
    prev = candles[-2]; curr = candles[-1]
    # bullish engulfing: prev red, curr green and curr body engulfs prev body
    if prev["close"] < prev["open"] and curr["close"] > curr["open"]:
        if curr["close"] > prev["high"] and curr["open"] < prev["low"]:
            return {"side": "CE", "type": "ENGULFING", "trigger": curr["high"]}
    # bearish engulfing
    if prev["close"] > prev["open"] and curr["close"] < curr["open"]:
        if curr["open"] > prev["high"] and curr["close"] < prev["low"]:
            return {"side": "PE", "type": "ENGULFING", "trigger": curr["low"]}
    return None

def detect_inside_bar(candles: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """Detect last candle being inside previous. Will require breakout to confirm."""
    if len(candles) < 2:
        return None
    prev = candles[-2]; curr = candles[-1]
    if curr["high"] < prev["high"] and curr["low"] > prev["low"]:
        # we cannot decide direction immediately; return type and boundaries
        return {"side": None, "type": "INSIDEBAR", "high": curr["high"], "low": curr["low"], "parent_high": prev["high"], "parent_low": prev["low"]}
    return None

def detect_gap_reversal(prev_close: float, open_price: float, first_candle: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Gap up + red reversal => short (PE). Gap down + green reversal => long (CE)."""
    if prev_close is None or open_price is None or first_candle is None:
        return None
    gap_up = open_price > prev_close * 1.002  # >0.2%
    gap_down = open_price < prev_close * 0.998
    if gap_up and first_candle["close"] < first_candle["open"]:
        return {"side": "PE", "type": "GAPREV", "trigger": first_candle["high"]}
    if gap_down and first_candle["close"] > first_candle["open"]:
        return {"side": "CE", "type": "GAPREV", "trigger": first_candle["low"]}
    return None

# ---------------- ENTRY / MANAGEMENT ----------------
def take_entry(side: str, underlying_price: float, state: Dict[str, Any]) -> bool:
    """
    Side: 'CE' or 'PE' (we BUY those options).
    Select option, place market BUY MIS, set state values and compute SL/TP on underlying & premium.
    """
    opt_row = select_option_for_buy(instrument_list, underlying_price, leg=side, otm_steps=OTM_STEPS)
    if opt_row is None:
        logging.error("No option row found for %s at underlying %.2f", side, underlying_price)
        return False

    tradingsymbol = opt_row["tradingsymbol"]
    lot = safe_get_lot_size(opt_row, default=QUANTITY)

    resp = place_buy_option(tradingsymbol, lot)
    if not resp:
        return False

    # capture premium
    entry_premium = get_ltp_retry(kite, f"NFO:{tradingsymbol}", retries=3, delay=0.5) or 0.0

    # underlying SL/TP: for BUY CE, SL = ORB low (or other chosen level), TP = entry + 2 * (entry - SL)
    # caller should set sl_underlying/tp_underlying before calling; fallback to small buffer
    sl_u = state.get("sl_underlying") or round(underlying_price * 0.995, 2)
    tp_u = state.get("tp_underlying") or round(underlying_price + 2 * (underlying_price - sl_u), 2)

    # premium-based SL/TP (sanity)
    premium_sl, premium_tp = compute_premium_sl_tp(entry_premium)

    state.update({
        "in_trade": True,
        "option_symbol": tradingsymbol,
        "option_lot": lot,
        "option_entry_premium": entry_premium,
        "entry_underlying": underlying_price,
        "sl_underlying": sl_u,
        "tp_underlying": tp_u,
        "order_meta": {"entry_resp": resp},
        "entry_time": datetime.now(),
        "premium_sl": premium_sl,
        "premium_tp": premium_tp
    })
    logging.info("[ENTRY] BUY %s | underlying=%.2f premium=%.2f lot=%d | sl_u=%.2f tp_u=%.2f premium_sl=%.2f premium_tp=%.2f",
                 tradingsymbol, underlying_price, entry_premium, lot, sl_u, tp_u, premium_sl or 0.0, premium_tp or 0.0)
    return True

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

# ---------------- MAIN STRATEGY LOOP ----------------
def run_intraday_multi_strategy():
    logging.info("Starting intraday multi-strategy (ENABLED=%s DRY_RUN=%s)", ENABLED_STRATEGIES, DRY_RUN)

    # session windows
    market_start = datetime.now().replace(hour=CANDLE_START.hour, minute=CANDLE_START.minute, second=0, microsecond=0)
    market_end = datetime.now().replace(hour=CANDLE_END.hour, minute=CANDLE_END.minute, second=0, microsecond=0)
    safety = datetime.now().replace(hour=SAFETY_EXIT.hour, minute=SAFETY_EXIT.minute,second=0, microsecond=0)

    # Wait until market start (if required)
    if datetime.now() < market_start:
        wait_until(market_start.time())

    # find nearest NIFTY future
    fut = get_nearest_nifty_fut(instrument_list)
    if not fut:
        logging.error("No NIFTY FUT found - aborting.")
        return
    fut_token = fut["instrument_token"]; fut_symbol = fut["tradingsymbol"]; fut_expiry = fut["expiry"]

    # get previous day close (optional): try via instrument LTP or external helper if available
    prev_day_close = None
    try:
        q = kite.quote([fut_token])  # dict with OHLC info
        prev_day_close = q[str(fut_token)]["ohlc"]["close"]
        logging.info("Prev day close fetched = %.2f", prev_day_close)
    except Exception as e:
        logging.warning("Could not fetch prev day close: %s", e)
        prev_day_close = None

    state = init_state()

    # build ORB (first 15 min)
    orb_start_dt = market_start
    orb_end_dt = market_start + timedelta(minutes=15)
    orb = build_orb_from_first_15(kite, fut_token, orb_start_dt, orb_end_dt)
    if orb:
        state["orb_built"] = True
        state["orb_high"], state["orb_low"], state["orb_open"], state["orb_close"] = orb["high"], orb["low"], orb["open"], orb["close"]
        # store the raw candles for detectors
        raw = []
        for c in orb["candles"]:
            raw.append({"open": c["open"], "high": c["high"], "low": c["low"], "close": c["close"], "volume": int(c.get("volume", 0)), "start": c.get("start"), "end": c.get("end")})
        state["raw_candles"] = raw
        logging.info("ORB built O=%.2f H=%.2f L=%.2f C=%.2f", state["orb_open"], state["orb_high"], state["orb_low"], state["orb_close"])
    else:
        logging.warning("ORB could not be built. Some detectors will still work if candles are fetched.")

    # also keep first candle (for gap reversal)
    # fetch first 5-min if not present
    if not state["raw_candles"]:
        c = fetch_candle(kite, fut_token, market_start, market_start + timedelta(minutes=CANDLE_INTERVAL))
        if c:
            state["raw_candles"].append({"open": c["open"], "high": c["high"], "low": c["low"], "close": c["close"], "volume": int(c.get("volume", 0)), "start": c.get("start"), "end": c.get("end")})

    logging.info("Monitoring for signals until %s", safety.strftime("%H:%M:%S"))

    # MAIN monitoring loop - watch for entry signals first
    while datetime.now() < safety and not state["in_trade"]:
        # daily loss guard
        if state["realized_pnl"] <= -abs(MAX_DAILY_LOSS):
            logging.critical("Max daily loss reached %.2f - stopping", state["realized_pnl"])
            break

        # underlying LTP
        u_ltp = get_ltp_retry(kite, fut_token, retries=2, delay=0.4)
        if u_ltp is None:
            time.sleep(1)
            continue

        # detectors priority order (ORB first, then others)
        signal = None

        if "ORB" in ENABLED_STRATEGIES:
            sig = detect_orb_breakout(state, u_ltp)
            if sig:
                signal = sig

        # Marubozu / Engulfing / Inside-bar / GapRev
        if not signal and "MARUBOZU" in ENABLED_STRATEGIES:
            # use first raw candle if exists
            if state["raw_candles"]:
                sig = detect_marubozu(state["raw_candles"][0])
                if sig:
                    signal = sig

        if not signal and "ENGULFING" in ENABLED_STRATEGIES:
            if len(state["raw_candles"]) >= 2:
                sig = detect_engulfing(state["raw_candles"])
                if sig:
                    signal = sig

        if not signal and "INSIDEBAR" in ENABLED_STRATEGIES:
            if len(state["raw_candles"]) >= 2:
                sig = detect_inside_bar(state["raw_candles"])
                # inside bar needs breakout confirmation: we don't immediate enter; we set target breakout levels
                if sig:
                    # If inside-bar detected, we wait for breakout of parent candle
                    # We use the parent's high/low as trigger: parent was earlier candle (-2)
                    # Here we set signal only if u_ltp breaches the parent's high/low
                    parent_high = sig.get("parent_high")
                    parent_low = sig.get("parent_low")
                    if parent_high and u_ltp > parent_high:
                        signal = {"side": "CE", "type": "INSIDEBAR", "trigger": parent_high}
                    elif parent_low and u_ltp < parent_low:
                        signal = {"side": "PE", "type": "INSIDEBAR", "trigger": parent_low}

        if not signal and "GAPREV" in ENABLED_STRATEGIES:
            first_candle = state["raw_candles"][0] if state["raw_candles"] else None
            # determine prev_close (best-effort)
            prev_close_val = prev_day_close
            open_price = first_candle["open"] if first_candle else None
            sig = detect_gap_reversal(prev_close_val, open_price, first_candle)
            if sig:
                # gap reversal trigger: require LTP breach of an appropriate level
                trigger = sig["trigger"]
                if sig["side"] == "CE" and u_ltp > trigger:
                    signal = sig
                elif sig["side"] == "PE" and u_ltp < trigger:
                    signal = sig

        # if we have a signal, validate, calculate SL/TP and take entry
        if signal:
            side = signal["side"]
            state["strategy_name"] = signal.get("type")   # ORB, MARUBOZU, etc.
            state["signal_trigger"] = signal.get("trigger")
            state["entry_side"] = side  # CE / PE

            logging.info(f"[ENTRY] {side} | {fut["tradingsymbol"]} | strategy={state.get('strategy_name')} "
                         f"trigger={state.get('signal_trigger') or 0.0:.2f} | underlying={fut["underlying_price"]:.2f} "
                         f"premium={fut["entry_premium"]:.2f} lot={fut["lot"]} | sl_u={sl_u:.2f} tp_u={tp_u:.2f} "
                         f"premium_sl={fut["premium_sl"] or 0.0:.2f} premium_tp={fut["premium_tp"] or 0.0:.2f}")
            
            # define underlying SL/TP: if ORB, we set opposite; for others we use ORB levels if built else small buffer
            if side == "CE":
                sl_u = state["orb_low"] if state.get("orb_built") else round(u_ltp * 0.995, 2)
                tp_u = round(u_ltp + 2 * (u_ltp - sl_u), 2)
            else:
                sl_u = state["orb_high"] if state.get("orb_built") else round(u_ltp * 1.005, 2)
                tp_u = round(u_ltp - 2 * (sl_u - u_ltp), 2)

            # set SL/TP on state temporarily for take_entry to read
            state["sl_underlying"] = sl_u
            state["tp_underlying"] = tp_u

            success = take_entry(side, u_ltp, state)
            if success:
                logging.info("Entry taken: %s %s (lot=%s) | sl_u=%.2f tp_u=%.2f",
                             state["option_symbol"], side, state["option_lot"], sl_u, tp_u)
                break
            else:
                logging.warning("Entry attempt failed; continuing monitoring")
        time.sleep(1)

    # If we did not enter a trade
    if not state["in_trade"]:
        logging.info("No trade taken before safety exit / end of window.")
        return

    # Manage open position until safety exit
    logging.info("Managing open position %s until safety=%s", state["option_symbol"], safety.strftime("%H:%M:%S"))
    while datetime.now() < safety and state["in_trade"]:
        # daily loss guard
        if state["realized_pnl"] <= -abs(MAX_DAILY_LOSS):
            logging.critical("Max daily loss reached %.2f - forcing exit", state["realized_pnl"])
            exit_position_market(state["option_symbol"], state["option_lot"])
            state["in_trade"] = False
            break

        # monitor and manage
        manage_open_position(state, fut_token)
        time.sleep(0.6)

    # final cleanup
    if state["in_trade"]:
        logging.warning("Safety square-off: exiting remaining position")
        exit_position_market(state["option_symbol"], state["option_lot"])
        state["in_trade"] = False

    logging.info("Session finished. realized_pnl=%.2f unrealized_pnl=%.2f",
                 state.get("realized_pnl", 0.0), state.get("unrealized_pnl", 0.0))


if __name__ == "__main__":
    try:
        run_intraday_multi_strategy()
    except Exception as e:
        logging.exception("Strategy crashed: %s", e)

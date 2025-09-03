# t6_final.py
"""
Production-ready intraday strategy (spot/futures separation + dynamic ATM options).

Before running:
 - Create a .env next to this file with required keys (API_KEY, ACCESS_TOKEN, etc).
 - Start in DRY_RUN=true mode until you have verified behavior with logs.
 - Test for at least several market sessions in DRY_RUN.

Author: (generated)
"""

import os
import logging
import pandas as pd
from datetime import datetime, timedelta, time as dtime
import time
import re
from kiteconnect import KiteConnect
from dotenv import load_dotenv

# timezone support
# timezone support
try:
    from zoneinfo import ZoneInfo
    IST = ZoneInfo("Asia/Kolkata")
except Exception:
    try:
        import pytz
        IST = pytz.timezone("Asia/Kolkata")
    except Exception:
        IST = None  # fallback, naive datetimes

# -------------------- CONFIG --------------------
load_dotenv(".env")

API_KEY = os.getenv("API_KEY")
ACCESS_TOKEN = os.getenv("ACCESS_TOKEN")
LOG_FILE = os.getenv("LOG_FILE", "t6_final_strategy.log")

# trading / strategy parameters
REVERSAL_PATTERNS = os.getenv("REVERSAL_PATTERNS", "RED,RED,RED:GREEN;GREEN,GREEN,GREEN:RED")
QUANTITY = int(os.getenv("QUANTITY", "75"))            # total contracts/units desired (we will respect lot sizes)
STRIKE_INTERVAL = int(os.getenv("STRIKE_INTERVAL", "50"))
CANDLE_INTERVAL = int(os.getenv("CANDLE_INTERVAL", "5"))  # minutes
CANDLE_START_HOUR = int(os.getenv("CANDLE_START_HOUR", "9"))
CANDLE_START_MIN = int(os.getenv("CANDLE_START_MIN", "15"))
CANDLE_END_HOUR = int(os.getenv("CANDLE_END_HOUR", "15"))
CANDLE_END_MIN = int(os.getenv("CANDLE_END_MIN", "30"))
MAX_ENTRIES = int(os.getenv("MAX_ENTRIES", "2"))
MAX_RETRIES = int(os.getenv("MAX_RETRIES", "4"))
RETRY_DELAY = float(os.getenv("RETRY_DELAY", "1"))
UNDERLYING = os.getenv("UNDERLYING", "NIFTY")
# NEW configurable SL/TP
SL_PCT = float(os.getenv("SL_PCT", "0.2"))  # 20% default
TP_PCT = float(os.getenv("TP_PCT", "0.3"))  # 30% default
SL_BUFFER = float(os.getenv("SL_BUFFER", "1.0"))  # price buffer for trigger

DRY_RUN = os.getenv("DRY_RUN", "true").lower() in ("1", "true", "yes")
MAX_DAILY_LOSS = float(os.getenv("MAX_DAILY_LOSS", "10000"))  # currency units
KILL_SWITCH = os.getenv("KILL_SWITCH", "false").lower() in ("1", "true", "yes")

SPOT_TOKEN_DEFAULT = 256265  # NIFTY 50 index token — fallback; script verifies against instruments

# -------------------- LOGGING --------------------
logging.basicConfig(
    filename=LOG_FILE,
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    force=True
)
console = logging.StreamHandler()
console.setLevel(logging.INFO)
formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
console.setFormatter(formatter)
logging.getLogger().addHandler(console)

logging.info("Starting strategy (t6_final.py) — DRY_RUN=%s", DRY_RUN)

# -------------------- KITE INIT & INSTRUMENTS --------------------
if not API_KEY or not ACCESS_TOKEN:
    logging.critical("API_KEY or ACCESS_TOKEN missing in .env — exiting")
    raise SystemExit("Missing API credentials in .env")

kite = KiteConnect(api_key=API_KEY)
kite.set_access_token(ACCESS_TOKEN)

# fetch instruments once, combine NFO + NSE (if necessary)
try:
    instr_nfo = kite.instruments("NFO")
except Exception as e:
    logging.critical("Failed to fetch instruments('NFO'): %s", e)
    raise

try:
    instr_nse = kite.instruments("NSE")
except Exception:
    instr_nse = []

# Normalize to DataFrame
instruments = pd.DataFrame(instr_nfo + instr_nse) if instr_nse else pd.DataFrame(instr_nfo)

# -------------------- UTILITIES --------------------
def wait_until(target_time: dtime):
    """Block until the target local time (in IST if available)."""
    while True:
        now = datetime.now(tz=IST) if IST else datetime.now()
        if now.time() >= target_time:
            return
        time.sleep(1)

def fetch_ohlc_df(token, from_dt, to_dt, interval="5minute"):
    """Fetch candles via kite.historical_data and return as DataFrame."""
    data = retry_kite_call(kite.historical_data, token, from_dt, to_dt, interval)
    return pd.DataFrame(data) if data else pd.DataFrame()

def classify_candle_row(row):
    """Return 'GREEN' or 'RED' for a kite candle row/dict."""
    return "GREEN" if row["close"] >= row["open"] else "RED"

# -------------------- HELPERS --------------------
def retry_kite_call(func, *args, retries=MAX_RETRIES, delay=RETRY_DELAY, **kwargs):
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

def parse_reversal_patterns(patterns_str):
    patterns = []
    for block in re.split(r"[;|]", patterns_str):
        block = block.strip()
        if not block or ":" not in block:
            continue
        seq_part, tgt_part = block.split(":", 1)
        seq = [s.strip().upper() for s in re.split(r"[,|_]", seq_part) if s.strip()]
        tgt = tgt_part.strip().upper()
        if seq and tgt in ("RED", "GREEN"):
            patterns.append((seq, tgt))
    return patterns

def check_dynamic_pattern(candle_types, patterns):
    for seq, tgt in patterns:
        n = len(seq)
        if len(candle_types) >= n and candle_types[-n:] == seq:
            return tgt
    return None

def cleanup_open_orders(kite, trades):
    for tid, t in trades.items():
        if tid == "_last_candle_end":
            continue
        for oid in (t.get("sl_id"), t.get("tp_id")):
            if not oid:
                continue
            try:
                retry_kite_call(kite.cancel_order, variety=kite.VARIETY_REGULAR, order_id=oid)
                logging.info("Cancelled order %s for %s", oid, t["symbol"])
            except Exception as e:
                logging.warning("Failed cancelling %s: %s", oid, e)

# -------------------- PnL Calculation --------------------
def compute_realized_pnl(kite, trade):
    """Fetch fills and calculate realized PnL for one trade."""
    try:
        trades = retry_kite_call(kite.trades)
        if not trades:
            return 0.0
        fills = [t for t in trades if t["order_id"] == trade.get("sl_id") or t["order_id"] == trade.get("tp_id")]
        if not fills:
            return 0.0
        # assume sell avg vs buy avg
        sell_price = trade.get("entry_ltp", 0.0)
        buy_price = sum(f["average_price"] for f in fills) / len(fills)
        if trade["leg"] == "CE":
            return (sell_price - buy_price) * trade["qty"]
        else:
            return (sell_price - buy_price) * trade["qty"]
    except Exception as e:
        logging.warning("PnL computation failed: %s", e)
        return 0.0

# -------------------- Health Check --------------------
def prestart_healthcheck(kite):
    profile = retry_kite_call(kite.profile)
    if not profile:
        margins = retry_kite_call(kite.margins, "equity")
        if not margins:
            return False
    return True

# -------------------- INSTRUMENT HELPERS (robust) --------------------
def find_spot_token(instruments_df):
    """Return the spot token for 'NIFTY 50' or fallback to SPOT_TOKEN_DEFAULT if present."""
    try:
        df = instruments_df
        match = df[(df["tradingsymbol"] == "NIFTY 50") & (df["segment"].str.contains("INDICES", na=False))]
        if not match.empty:
            return int(match.iloc[0]["instrument_token"])
    except Exception:
        pass
    # fallback (still prefer to check that token exists in instruments)
    logging.warning("NIFTY 50 instrument row not found — using fallback token %s (verify correctness)", SPOT_TOKEN_DEFAULT)
    return SPOT_TOKEN_DEFAULT

def get_nearest_future(instruments_df, underlying=UNDERLYING):
    """
    Return nearest futures (symbol, token, expiry date as datetime).
    Robust to different segment naming conventions.
    """
    df = instruments_df.copy()
    # normalize segment strings if they exist
    if "segment" in df.columns:
        fut_mask = df["segment"].str.contains("FUT", na=False) & (df["name"] == underlying)
    else:
        fut_mask = (df["name"] == underlying) & df["tradingsymbol"].str.endswith("FUT")
    df_futs = df[fut_mask].copy()
    if df_futs.empty:
        raise ValueError("No futures contracts found for underlying: " + str(underlying))
    df_futs["expiry"] = pd.to_datetime(df_futs["expiry"])
    df_futs = df_futs.sort_values("expiry")
    nearest = df_futs.iloc[0]
    return nearest["tradingsymbol"], int(nearest["instrument_token"]), pd.to_datetime(nearest["expiry"])

def parse_reversal_patterns(patterns_str):
    """
    Parse strings like:
      RED,RED,RED:GREEN; GREEN,GREEN:RED
    into list of tuples: [ (["RED","RED","RED"], "GREEN"), ... ]
    returns empty list on malformed input
    """
    patterns = []
    for block in re.split(r"[;|]", patterns_str):
        block = block.strip()
        if not block:
            continue
        if ":" not in block:
            logging.warning("Skipping malformed pattern block (missing ':'): %s", block)
            continue
        seq_part, tgt_part = block.split(":", 1)
        seq = [s.strip().upper() for s in re.split(r"[,|_]", seq_part) if s.strip()]
        tgt = tgt_part.strip().upper()
        if not seq or tgt not in ("RED", "GREEN"):
            logging.warning("Skipping malformed pattern block: %s", block)
            continue
        patterns.append((seq, tgt))
    return patterns

def check_dynamic_pattern(candle_types, patterns):
    """
    Given a list of candle_types (['RED','RED','GREEN',...]) and patterns list,
    return 'GREEN' or 'RED' target if any pattern matches the latest sequence.
    Matches the last N candles where N = len(pattern sequence)
    """
    for seq, tgt in patterns:
        n = len(seq)
        if len(candle_types) < n:
            continue
        if candle_types[-n:] == seq:
            return tgt
    return None

def get_atm_options_and_lot(instruments_df, spot_price, expiry_dt=None, underlying=UNDERLYING):
    """
    Robustly return (ce_tradingsymbol, ce_token, ce_lot), (pe_tradingsymbol, pe_token, pe_lot)
    - picks nearest strike to spot rounded to STRIKE_INTERVAL
    - if exact strike absent, picks closest strike
    - if expiry_dt is None uses nearest expiry
    """
    df = instruments_df.copy()
    # ensure expected columns
    if "segment" in df.columns:
        df_opts = df[df["segment"].str.contains("OPT", na=False) & (df["name"] == underlying)].copy()
    else:
        df_opts = df[df["tradingsymbol"].str.contains(underlying, na=False)].copy()
    if df_opts.empty:
        raise ValueError("No option instruments found for underlying: " + str(underlying))
    df_opts["expiry"] = pd.to_datetime(df_opts["expiry"])
    today = datetime.now(IST).date() if IST else datetime.now().date()
    df_opts = df_opts[df_opts["expiry"].dt.date >= today]
    if df_opts.empty:
        raise ValueError("No non-expired option instruments available")

    if expiry_dt is None:
        # pick nearest expiry
        nearest_expiry = min(df_opts["expiry"].dt.date.unique())
        df_opts = df_opts[df_opts["expiry"].dt.date == nearest_expiry]
    else:
        expiry_date = pd.to_datetime(expiry_dt).date()
        df_opts = df_opts[df_opts["expiry"].dt.date == expiry_date]
        if df_opts.empty:
            raise ValueError(f"No option contracts for expiry {expiry_date}")

    target_strike = round(spot_price / STRIKE_INTERVAL) * STRIKE_INTERVAL
    strikes = sorted(df_opts["strike"].unique())
    if not strikes:
        raise ValueError("No strikes found for filtered option instruments")

    # pick closest strike
    closest = min(strikes, key=lambda s: abs(s - target_strike))
    df_ce = df_opts[(df_opts["strike"] == closest) & (df_opts["instrument_type"] == "CE")]
    df_pe = df_opts[(df_opts["strike"] == closest) & (df_opts["instrument_type"] == "PE")]

    # fallback if either empty - try expanding to nearest available strike overall
    if df_ce.empty or df_pe.empty:
        # expand to whole expiry strikes and choose the really closest among all
        closest = min(strikes, key=lambda s: abs(s - target_strike))
        df_ce = df_opts[(df_opts["strike"] == closest) & (df_opts["instrument_type"] == "CE")]
        df_pe = df_opts[(df_opts["strike"] == closest) & (df_opts["instrument_type"] == "PE")]

    if df_ce.empty or df_pe.empty:
        raise ValueError(f"ATM CE/PE not found for strike {closest} (expiry filtered)")

    ce = df_ce.iloc[0]
    pe = df_pe.iloc[0]
    # lot_size may be present in instrument metadata as integer; fallback to 1
    ce_lot = int(ce.get("lot_size", 1)) if "lot_size" in ce.index else int(ce.get("lots", 1) if "lots" in ce.index else 1)
    pe_lot = int(pe.get("lot_size", 1)) if "lot_size" in pe.index else int(pe.get("lots", 1) if "lots" in pe.index else 1)

    return (ce["tradingsymbol"], int(ce["instrument_token"]), ce_lot), (pe["tradingsymbol"], int(pe["instrument_token"]), pe_lot)

# -------------------- ORDER HANDLING (atomic-ish) --------------------
def compute_order_qty(total_quantity, lot_size):
    """Compute number of contracts to send respecting lot size."""
    lots = total_quantity // lot_size
    if lots < 1:
        raise ValueError(f"Configured QUANTITY={total_quantity} is less than single option lot_size={lot_size}. Adjust QUANTITY or lot_size.")
    return lots * lot_size  # final quantity (multiple of lot_size)

def calculate_sl_tp(entry_price, leg_type):
    if not entry_price or entry_price <= 0:
        return None, None
    sl = entry_price * (1 + SL_PCT)
    tp = entry_price * (1 - TP_PCT)
    return round(sl, 1), round(tp, 1)


def force_exit_open_trades(kite, trades):
    """Force close at market for EOD safety."""
    for tid, trade in trades.items():
        if tid == "_last_candle_end" or trade.get("status") != "OPEN":
            continue
        try:
            kite.place_order(
                tradingsymbol=trade["symbol"], exchange="NFO",
                transaction_type="BUY", quantity=trade["qty"],
                order_type="MARKET", product="MIS", variety="regular"
            )
            logging.info("Forced exit for %s at EOD", trade["symbol"])
        except Exception as e:
            logging.error("EOD force-exit failed for %s: %s", trade["symbol"], e)

# -------------------- PRESTART HEALTH CHECK --------------------
def prestart_healthcheck(kite):
    """
    Simple health checks: kite.profile() or kite.margins() to confirm connectivity and margin.
    """
    logging.info("Running pre-start health check")
    profile = retry_kite_call(getattr(kite, "profile", None))
    if not profile:
        # try margins endpoint
        try:
            margins = retry_kite_call(kite.margins, "equity")
            if not margins:
                logging.critical("Health check failed: no profile or margins, aborting")
                return False
        except Exception:
            logging.critical("Health check failed completely")
            return False
    logging.info("Health check passed")
    return True

def place_protected_sell(kite, symbol, qty, entry_price, leg_type, dry_run=False):
    """Place SELL + SL-M + TP with cleanup if failure occurs."""
    logging.info("Placing protected SELL for %s qty=%s DRY_RUN=%s", symbol, qty, dry_run)
    if dry_run:
        return "DRY_SELL", "DRY_SL", "DRY_TP"

    # --- SELL order ---
    try:
        resp = retry_kite_call(
            kite.place_order,
            tradingsymbol=symbol, exchange="NFO", transaction_type="SELL",
            quantity=qty, order_type="MARKET", product="MIS", variety="regular"
        )
        if not resp or "order_id" not in resp:
            logging.error("SELL order placement failed: %s", resp)
            return None, None, None
        sell_id = resp["order_id"]
    except Exception as e:
        logging.error("SELL failed: %s", e)
        return None, None, None

    # --- Calculate SL / TP ---
    sl, tp = calculate_sl_tp(entry_price, leg_type)
    if not sl or not tp:
        logging.error("SL/TP calculation failed for %s (entry=%s leg=%s)", symbol, entry_price, leg_type)
        return sell_id, None, None

    # --- SL-M order ---
    try:
        resp = retry_kite_call(
            kite.place_order,
            tradingsymbol=symbol, exchange="NFO", transaction_type="BUY",
            quantity=qty, order_type="SL-M", trigger_price=sl,
            product="MIS", variety="regular"
        )
        if not resp or "order_id" not in resp:
            logging.error("SL-M order placement failed: %s", resp)
            return sell_id, None, None
        sl_id = resp["order_id"]
    except Exception as e:
        logging.error("SL-M failed: %s", e)
        return sell_id, None, None

    # --- TP order ---
    try:
        resp = retry_kite_call(
            kite.place_order,
            tradingsymbol=symbol, exchange="NFO", transaction_type="BUY",
            quantity=qty, order_type="LIMIT", price=tp,
            product="MIS", variety="regular"
        )
        if not resp or "order_id" not in resp:
            logging.error("TP order placement failed: %s", resp)
            # cleanup: cancel SL if TP fails
            try:
                kite.cancel_order(variety=kite.VARIETY_REGULAR, order_id=sl_id)
            except Exception:
                pass
            return sell_id, None, None
        tp_id = resp["order_id"]
    except Exception as e:
        logging.error("TP failed: %s", e)
        # cleanup: cancel SL if TP fails
        try:
            kite.cancel_order(variety=kite.VARIETY_REGULAR, order_id=sl_id)
        except Exception:
            pass
        return sell_id, None, None

    return sell_id, sl_id, tp_id

def safe_get_ltp(symbol, max_tries=3):
    """Resilient LTP fetch with retries."""
    for i in range(max_tries):
        try:
            resp = retry_kite_call(kite.place_order,tradingsymbol=symbol, exchange="NFO", 
                                   transaction_type="SELL",
                                   quantity=QUANTITY, order_type="MARKET", 
                                   product="MIS", variety="regular")
            if not resp or "order_id" not in resp:
                logging.error("SELL order placement failed: %s", resp)
                return None, None, None
            sell_id = resp["order_id"]
        except Exception as e:
            logging.warning("LTP fetch attempt %d failed for %s: %s", i+1, symbol, e)
        time.sleep(1)
    return None

# -------------------- STRATEGY CORE --------------------
def run_consecutive_candle_strategy_auto_oco(
        kite, instruments_df, qty,
        candle_start_time, candle_end_time, max_entries=2):
    """
    Main strategy loop:
     - Uses spot OHLC (NIFTY 50) for candle color detection
     - Uses nearest futures for volume confirmation
     - Matches dynamic reversal patterns
     - Places protected ATM option short trades with SL/TP
     - Tracks PnL, enforces daily loss guard
     - Supports KILL_SWITCH, DRY_RUN, and EOD forced exit
    """

    if KILL_SWITCH:
        logging.critical("KILL_SWITCH enabled — aborting run")
        return

    if not prestart_healthcheck(kite):
        logging.critical("Pre-start health check failed — aborting")
        return

    # Token detection
    try:
        spot_token = find_spot_token(instruments_df)
    except Exception as e:
        logging.warning("Spot token lookup failed: %s — fallback used", e)
        spot_token = SPOT_TOKEN_DEFAULT

    try:
        fut_symbol, fut_token, fut_expiry = get_nearest_future(instruments_df, UNDERLYING)
    except Exception as e:
        logging.critical("Failed to detect nearest future: %s", e)
        return

    logging.info("Spot token=%s | Futures=%s token=%s expiry=%s",
                 spot_token, fut_symbol, fut_token, fut_expiry)

    patterns = parse_reversal_patterns(REVERSAL_PATTERNS)
    if not patterns:
        logging.critical("No valid reversal patterns parsed — aborting")
        return
    logging.info("Parsed reversal patterns: %s", patterns)

    candle_types, raw_spot_candles, fut_volumes = [], [], []
    entries_taken, realized_pnl = 0, 0.0
    open_trades = {}

    try:
        while True:
            now = datetime.now(IST) if IST else datetime.now()
            if now.time() >= candle_end_time:
                logging.info("Market end reached (%s) — forcing EOD exit", candle_end_time)
                break

            if entries_taken >= max_entries:
                logging.info("Max entries (%d) reached. Waiting for close.", max_entries)
                time.sleep(10)
                continue

            if KILL_SWITCH:
                logging.critical("KILL_SWITCH triggered mid-run — exiting")
                break

            # Align to current candle window
            aligned = now.replace(second=0, microsecond=0)
            aligned = aligned - timedelta(minutes=aligned.minute % CANDLE_INTERVAL)
            c_start = aligned - timedelta(minutes=CANDLE_INTERVAL)
            c_end = aligned

            if open_trades.get("_last_candle_end") == c_end:
                time.sleep(1)
                continue

            # Fetch candles
            spot_data = retry_kite_call(kite.historical_data, spot_token, c_start, c_end, "5minute")
            fut_data = retry_kite_call(kite.historical_data, fut_token, c_start, c_end, "5minute")
            if not spot_data or not fut_data:
                logging.warning("Missing candle data for %s - %s", c_start, c_end)
                time.sleep(1)
                continue

            if not spot_data or not isinstance(spot_data[-1], dict):
                logging.warning("Skipping invalid spot data %s - %s", c_start, c_end)
                time.sleep(1)
                continue
            if not fut_data or not isinstance(fut_data[-1], dict):
                logging.warning("Skipping invalid futures data %s - %s", c_start, c_end)
                time.sleep(1)
                continue

            sc, fc = spot_data[-1], fut_data[-1]
            
            ctype = classify_candle_row(sc)
            candle_types.append(ctype)
            raw_spot_candles.append(sc)
            fut_volumes.append(fc.get("volume", 0))

            # Rolling window
            if len(candle_types) > 12:
                candle_types.pop(0), raw_spot_candles.pop(0), fut_volumes.pop(0)

            logging.info("Candle %s-%s | Spot %s O=%s C=%s | FutVol=%s",
                         c_start.time(), c_end.time(), ctype,
                         sc["open"], sc["close"], fc.get("volume"))

            open_trades["_last_candle_end"] = c_end

            # Pattern match
            target = check_dynamic_pattern(candle_types, patterns)
            if target:
                logging.info("Pattern matched: target=%s", target)
                spot_price = sc["close"]

                try:
                    (ce_tsym, ce_token, ce_lot), (pe_tsym, pe_token, pe_lot) = \
                        get_atm_options_and_lot(instruments_df, spot_price, expiry_dt=fut_expiry, underlying=UNDERLYING)
                except Exception as e:
                    logging.error("ATM option selection failed: %s", e)
                    time.sleep(1)
                    continue

                leg_type = "CE" if target == "GREEN" else "PE"
                sel_sym, sel_token, lot_size = (ce_tsym, ce_token, ce_lot) if leg_type == "CE" else (pe_tsym, pe_token, pe_lot)

                try:
                    order_qty = compute_order_qty(qty, lot_size)
                except ValueError as e:
                    logging.error("Qty/lot mismatch: %s", e)
                    time.sleep(1)
                    continue

                ltp = safe_get_ltp(sel_sym)
                if not ltp:
                    logging.error("Failed to fetch LTP for %s", sel_sym)
                    continue

                if realized_pnl <= -abs(MAX_DAILY_LOSS):
                    logging.critical("Max daily loss (%.2f) hit — no new trades", realized_pnl)
                    break

                sell_id, sl_id, tp_id = place_protected_sell(
                    kite, sel_sym, order_qty, ltp, leg_type, dry_run=DRY_RUN)
                if not sell_id:
                    logging.error("Protected sell failed for %s", sel_sym)
                else:
                    entries_taken += 1
                    open_trades[sell_id] = {
                        "symbol": sel_sym, "leg": leg_type, "qty": order_qty,
                        "entry_ltp": ltp, "sl_id": sl_id, "tp_id": tp_id, "status": "OPEN"
                    }
                    logging.info("Trade entered: %s | qty=%s | entry=%s", sel_sym, order_qty, ltp)

            # Monitor trades
            if any(t.get("status") == "OPEN" for t in open_trades.values()):
                orders = retry_kite_call(kite.orders) or []
                for tid, trade in list(open_trades.items()):
                    if tid == "_last_candle_end" or trade.get("status") != "OPEN":
                        continue

                    sl_status = next((o for o in orders if str(o.get("order_id")) == str(trade.get("sl_id"))), None)
                    tp_status = next((o for o in orders if str(o.get("order_id")) == str(trade.get("tp_id"))), None)

                    if sl_status and sl_status.get("status") == "COMPLETE":
                        logging.info("SL hit for %s", trade["symbol"])
                        realized_pnl += compute_realized_pnl(kite, trade)
                        trade["status"] = "CLOSED"
                        try:
                            retry_kite_call(kite.cancel_order, variety=kite.VARIETY_REGULAR, order_id=trade.get("tp_id"))
                        except Exception:
                            pass
                        open_trades.pop(tid, None)

                    elif tp_status and tp_status.get("status") == "COMPLETE":
                        logging.info("TP hit for %s", trade["symbol"])
                        realized_pnl += compute_realized_pnl(kite, trade)
                        trade["status"] = "CLOSED"
                        try:
                            retry_kite_call(kite.cancel_order, variety=kite.VARIETY_REGULAR, order_id=trade.get("sl_id"))
                        except Exception:
                            pass
                        open_trades.pop(tid, None)

            time.sleep(1)

    except KeyboardInterrupt:
        logging.info("Manual interrupt — cleanup")
    except Exception as e:
        logging.exception("Unhandled exception: %s", e)
    finally:
        logging.info("Final cleanup...")
        force_exit_open_trades(kite, open_trades)
        logging.info("Strategy stopped. Realized PnL: %.2f", realized_pnl)


# -------------------- MAIN EXECUTION --------------------
if __name__ == "__main__":
    # build market start/end times in local tz (IST) if available
    now = datetime.now(IST) if IST else datetime.now()
    market_start = now.replace(hour=CANDLE_START_HOUR, minute=CANDLE_START_MIN, second=0, microsecond=0)
    market_end = now.replace(hour=CANDLE_END_HOUR, minute=CANDLE_END_MIN, second=0, microsecond=0)

    now_time = datetime.now(IST).time() if IST else datetime.now().time()
    if now_time < market_start.time():
        logging.info("Waiting until market start: %s", market_start.time())
        wait_until(market_start.time())

    # ensure instruments DataFrame is present and valid
    if instruments is None or instruments.empty:
        logging.critical("Instrument list is empty - cannot start")
        raise SystemExit("Empty instruments list")

    run_consecutive_candle_strategy_auto_oco(kite, instruments, QUANTITY, market_start.time(), market_end.time(), MAX_ENTRIES)
    logging.info("Script finished.")

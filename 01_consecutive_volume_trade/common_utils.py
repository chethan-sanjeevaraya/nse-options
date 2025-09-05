# utils/common_utils.py
"""
Cleaned common utilities for intraday strategies.

Features:
 - Time helpers
 - Robust KiteConnect wrappers with retries
 - Candle fetch helper
 - Option chain helper (returns pandas DataFrame)
 - Simple market order wrapper
 - Small defensive helpers (lot size, normalize GTT response)
"""

from datetime import datetime, timedelta
import time
import logging
import pandas as pd
from typing import Any, Dict, List, Optional

# -------------------- Time helpers --------------------
def to_time_obj(timestr: str):
    """Convert 'HH:MM' string to datetime.time (local system time)."""
    h, m = map(int, timestr.split(":"))
    now = datetime.now()
    return now.replace(hour=h, minute=m, second=0, microsecond=0).time()

def align_to_5min(dt_obj: datetime):
    """Round down datetime to previous 5-minute boundary."""
    return dt_obj.replace(second=0, microsecond=0, minute=(dt_obj.minute // 5) * 5)

def wait_until(target_time):
    """Block until target_time (datetime.time)."""
    logging.info("Waiting until %s", target_time)
    while datetime.now().time() < target_time:
        time.sleep(0.5)

# -------------------- Kite wrappers with retries --------------------
def retry_kite_call(func, *args, retries: int = 3, delay: float = 1.0, **kwargs):
    """
    Generic retry wrapper for KiteConnect calls.
    Returns the function's result or raises the last exception if all retries fail.
    """
    last_exc = None
    backoff = delay
    for attempt in range(1, retries + 1):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            logging.warning("Attempt %d/%d failed for %s: %s", attempt, retries, getattr(func, "__name__", str(func)), e)
            last_exc = e
            time.sleep(backoff)
            backoff = min(backoff * 2, 10)
    logging.error("All %d retries failed for %s", retries, getattr(func, "__name__", str(func)))
    raise last_exc

def place_order_retry(kite, retries: int = 3, delay: float = 1.0, **kwargs):
    """
    Retry wrapper around kite.place_order.
    Returns the order response or raises if all retries fail.
    """
    return retry_kite_call(lambda **k: kite.place_order(**k), **kwargs, retries=retries, delay=delay)

def get_ltp_retry(kite, instrument: str, retries: int = 3, delay: float = 0.8) -> Optional[float]:
    """
    Try to fetch LTP for an instrument with retries.
    Returns float LTP or None if cannot fetch.
    """
    try:
        resp = retry_kite_call(kite.ltp, [instrument], retries=retries, delay=delay)
        # resp structure: {instrument: {"last_price": ...}}
        if isinstance(resp, dict) and instrument in resp:
            return resp[instrument].get("last_price")
    except Exception as e:
        logging.warning("get_ltp_retry final failure for %s: %s", instrument, e)
    return None

# -------------------- Candle fetch --------------------
def fetch_candle(kite, token, start_dt, end_dt, interval_str="5minute"):
    """
    Fetch historical candles for a given instrument_token.
    Must pass the FUT token, not index.
    """
    try:
        data = retry_kite_call(kite.historical_data, token, start_dt, end_dt, interval_str, continuous=False)
        if not data:
            return None
        return data[-1]  # last candle
    except Exception as e:
        logging.error("fetch_candle failed for token=%s: %s", token, e)
        return None

# -------------------- Option chain / instruments --------------------
def round_to_strike(price: float, interval: int = 50) -> int:
    """Round price to nearest strike interval (int)."""
    try:
        return int(round(float(price) / interval) * interval)
    except Exception:
        return int(price)

def option_chain(instrument_list: List[Dict[str, Any]], ticker: str, price: float,
                 duration: int = 0, window: int = 5, leg: str = "CE") -> pd.DataFrame:
    """
    Return a DataFrame for option contracts around ATM for given ticker.
    - instrument_list: kite.instruments("NFO") result (list of dicts)
    - leg: "CE" or "PE"
    - window: number of strikes to return around ATM (total)
    """
    if not instrument_list:
        return pd.DataFrame()

    df = pd.DataFrame(instrument_list)
    # filter by options for ticker
    if "segment" in df.columns:
        opts = df[df["segment"].str.contains("OPT", na=False) & (df["name"] == ticker)].copy()
    else:
        opts = df[df["tradingsymbol"].str.contains(ticker, na=False)].copy()

    if opts.empty:
        return pd.DataFrame()

    # cleanup and expiry selection (nearest)
    opts["expiry"] = pd.to_datetime(opts["expiry"], errors="coerce")
    opts = opts[opts["expiry"].notna()]
    if opts.empty:
        return pd.DataFrame()

    try:
        today = datetime.now().date()
        # Only future expiries
        opts = opts[opts["expiry"].dt.date >= today]
        expiries = sorted(opts["expiry"].dt.date.unique())
        chosen_expiry = expiries[min(duration, len(expiries) - 1)]
        opts = opts[opts["expiry"].dt.date == chosen_expiry]
    except Exception:
        pass

    if opts.empty:
        return pd.DataFrame()

    # sort by strike and select window around ATM
    opts.sort_values("strike", inplace=True)
    atm = round_to_strike(price)
    strikes = sorted(opts["strike"].unique())
    if not strikes:
        return pd.DataFrame()
    closest = min(strikes, key=lambda s: abs(s - atm))
    idx = strikes.index(closest)
    half = window // 2
    selected = strikes[max(0, idx - half): idx + half + 1]
    df_sel = opts[opts["strike"].isin(selected)].copy()

    # filter instrument type CE/PE
    if "instrument_type" in df_sel.columns:
        df_sel = df_sel[df_sel["instrument_type"] == leg]
    return df_sel.reset_index(drop=True)

def get_nearest_nifty_fut(instrument_list: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """
    Returns the instrument dict for the nearest-expiry NIFTY monthly future.
    Falls back to nearest expiry (weekly/monthly) if no monthly found.
    """
    if not instrument_list:
        return None

    df = pd.DataFrame(instrument_list)
    if df.empty:
        return None

    # Filter only NIFTY futures
    if "segment" in df.columns:
        mask = df["segment"].str.contains("FUT", na=False) & (df["name"] == "NIFTY")
    else:
        mask = df["tradingsymbol"].str.endswith("FUT") & (df["name"] == "NIFTY")
    futs = df[mask].copy()

    if futs.empty:
        return None

    futs["expiry"] = pd.to_datetime(futs["expiry"], errors="coerce")
    futs = futs[futs["expiry"].notna()].sort_values("expiry")

    if futs.empty:
        return None

    # ✅ Prefer monthly expiry (last Thursday of month)
    today = datetime.now().date()
    futs["day"] = futs["expiry"].dt.day
    futs["weekday"] = futs["expiry"].dt.weekday  # 3 = Thursday
    monthly_futs = futs[(futs["day"] >= 23) & (futs["day"] <= 31) & (futs["weekday"] == 3)]

    if not monthly_futs.empty:
        chosen = monthly_futs.iloc[0]
    else:
        # fallback: nearest expiry (weekly or monthly)
        chosen = futs.iloc[0]

    logging.info("Using NIFTY future contract: %s | expiry=%s | token=%s",
                 chosen["tradingsymbol"], chosen["expiry"], chosen["instrument_token"])

    return chosen.to_dict()

# -------------------- Market order small wrapper --------------------
def placeMarketOrder(kite, params: Dict[str, Any]) -> Any:
    """
    Simple wrapper around kite.place_order for MARKET orders.
    params keys: tradingsymbol, transaction_type, quantity, exchange(optional), product(optional), variety(optional)
    """
    try:
        return kite.place_order(
            tradingsymbol=params["tradingsymbol"],
            exchange=params.get("exchange", "NFO"),
            transaction_type=params["transaction_type"],
            quantity=params["quantity"],
            order_type="MARKET",
            product=params.get("product", "MIS"),
            variety=params.get("variety", "regular")
        )
    except Exception as e:
        logging.error("placeMarketOrder failed: %s", e)
        return None

# -------------------- Small helpers --------------------
def safe_get_lot_size(atm_row: Any, default: int = 75) -> int:
    """
    Extract lot_size from instrument row (pandas Series or dict).
    Returns default if missing or invalid.
    """
    try:
        if hasattr(atm_row, "get"):
            val = atm_row.get("lot_size") or atm_row.get("lots") or default
        else:
            val = atm_row["lot_size"]
        return int(val)
    except Exception:
        return int(default)

def normalize_gtt_id(gtt_resp: Any) -> Optional[str]:
    """Return GTT id string if present or None."""
    try:
        if isinstance(gtt_resp, dict):
            return gtt_resp.get("id") or gtt_resp.get("data") or None
        return str(gtt_resp)
    except Exception:
        return None

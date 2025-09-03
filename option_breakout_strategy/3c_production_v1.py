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

logging.info("Starting strategy (t8_final.py) — DRY_RUN=%s", DRY_RUN)

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
    """
    Pause execution until a specified target time (Wait until 09:15 local time) is reached.

    This function continuously checks the current time and blocks the program
    until the local time (or IST if defined) reaches or passes the target time.

    Parameters
    ----------
    target_time : datetime.time
        The local time to wait until. The function will unblock once the current
        time is equal to or later than this time.

    Notes
    -----
    - If `IST` timezone is defined, the current time is evaluated in IST.
    - The function sleeps in 1-second intervals to avoid busy-waiting.
    - This is a blocking call; it will halt the program until the condition is met.

    Example
    -------
    from datetime import time  
    Wait until 09:15 local time
    """
    while True:
        now = datetime.now(tz=IST) if IST else datetime.now()
        if now.time() >= target_time:
            return
        time.sleep(1)

def fetch_ohlc_df(token, from_dt, to_dt, interval="5minute"):
    """
    Fetch historical OHLC (Open, High, Low, Close) candle data from Kite Connect 
    and return it as a pandas DataFrame.

    This function wraps the `kite.historical_data` API call with a retry mechanism 
    to ensure robustness in case of intermittent API failures. The returned DataFrame 
    contains the standard OHLC fields along with volume and timestamp.

    Parameters
    ----------
    token : str
        The instrument token for which historical data is to be fetched.
    from_dt : datetime.datetime
        Start datetime for historical data.
    to_dt : datetime.datetime
        End datetime for historical data.
    interval : str, optional
        Candle interval, e.g., "1minute", "5minute", "15minute", "day" (default is "5minute").

    Returns
    -------
    pd.DataFrame
        DataFrame containing historical candle data. Columns typically include:
        'date', 'open', 'high', 'low', 'close', 'volume'.
        Returns an empty DataFrame if no data is fetched.

    Example
    -------
    >>> from datetime import datetime
    >>> fetch_ohlc_df("256265", datetime(2025, 9, 1, 9, 15), datetime(2025, 9, 1, 15, 30))
    """

    data = retry_kite_call(kite.historical_data, token, from_dt, to_dt, interval)
    return pd.DataFrame(data) if data else pd.DataFrame()

def classify_candle_row(row):
    """
    Classify a single OHLC candle as 'GREEN' or 'RED'.

    A candle is considered:
    - 'GREEN' if the closing price is greater than or equal to the opening price.
    - 'RED' if the closing price is less than the opening price.

    Parameters
    ----------
    row : dict
        A dictionary representing a single candle, typically returned by Kite Connect
        historical data. Must contain at least the keys:
        'open' : float
            Opening price of the candle.
        'close' : float
            Closing price of the candle.

    Returns
    -------
    str
        'GREEN' or 'RED' depending on the price movement.

    Example
    -------
    > candle = {'open': 24850, 'close': 24852.5, 'high': 24855, 'low': 24848, 'volume': 12000}
    > classify_candle_row(candle)
    'GREEN'
    """

    return "GREEN" if row["close"] >= row["open"] else "RED"

# -------------------- HELPERS --------------------
def retry_kite_call(func, *args, retries=MAX_RETRIES, delay=RETRY_DELAY, **kwargs):
    """
    Call a function with automatic retries and exponential backoff in case of exceptions.

    This is useful for Kite Connect API calls (or any network/API function) that may
    intermittently fail due to network issues, rate limits, or temporary server errors.

    Parameters
    ----------
    func : callable
        The function to call (e.g., kite.historical_data).
    *args : tuple
        Positional arguments to pass to `func`.
    retries : int, optional
        Maximum number of retry attempts (default is MAX_RETRIES).
    delay : float, optional
        Initial delay between retries in seconds (default is RETRY_DELAY).
    **kwargs : dict
        Keyword arguments to pass to `func`.

    Returns
    -------
    Any
        The return value of `func` if successful, else `None` if all retries fail.

    Notes
    -----
    - Uses exponential backoff: the delay doubles after each failed attempt, up to 30 seconds.
    - Logs each failed attempt with function name and exception message.
    - backoff starts with the initial delay (e.g., 1 or 2 seconds).
    - After each failed attempt, backoff doubles (backoff * 2) until a maximum cap (here, 30 seconds).
    - This is called exponential backoff, a common pattern in networking to reduce load on the server and improve success rates.

    Example
    -------
    > result = retry_kite_call(kite.historical_data, token, from_dt, to_dt, "5minute")
    > if result is None:
    >     print("Failed to fetch data after retries")
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

def parse_reversal_patterns(patterns_str):
    """
    Parse a string of reversal patterns into a structured list of sequences and targets.

    This function converts a human-readable string of candle reversal patterns into
    a list of tuples suitable for programmatic pattern matching. Each tuple contains
    a sequence of candle colors and the target candle type to act upon.

    Parameters
    ----------
    patterns_str : str
        String containing one or more patterns. Each pattern should have the format:
        "CANDLE1,CANDLE2,...:TARGET", where:
        - CANDLE1, CANDLE2, ... are 'GREEN' or 'RED' (case-insensitive)
        - TARGET is 'GREEN' or 'RED' (case-insensitive)
        Multiple patterns can be separated by semicolons `;` or pipe `|`.
        Within a sequence, candles can be separated by comma `,` or underscore `_`.

    Returns
    -------
    List[Tuple[List[str], str]]
        A list of tuples, where each tuple is:
        - seq : List[str] - list of candle colors representing the pattern sequence
        - tgt : str - target candle type ('GREEN' or 'RED') to act upon if the pattern matches

    Example
    -------
    >>> parse_reversal_patterns("GREEN,GREEN,RED:PE; RED,RED,GREEN:CE")
    [(['GREEN', 'GREEN', 'RED'], 'PE'), (['RED', 'RED', 'GREEN'], 'CE')]
    """

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

def check_dynamic_pattern_with_volume(candle_types, fut_volumes, patterns):
    """
    Check for reversal patterns with 4th-candle volume condition.

    Parameters
    ----------
    candle_types : List[str]
        Rolling list of candle colors ('GREEN'/'RED').
    fut_volumes : List[float]
        Rolling list of futures volumes aligned with candle_types.
    patterns : List[Tuple[List[str], str]]
        Pre-parsed list of reversal patterns, e.g.
        [(['RED', 'RED', 'RED'], 'GREEN'), (['GREEN', 'GREEN', 'GREEN'], 'RED')]

    Returns
    -------
    str or None
        Target ('CE' or 'PE') if a pattern with volume condition matches, else None.
    """
    for seq, tgt in patterns:
        n = len(seq)
        # Ensure we have at least sequence + 1 candles for 4th-candle volume check
        if len(candle_types) >= n + 1 and candle_types[-n-1:-1] == seq:
            # 4th candle is the last candle in the window
            fourth_color = candle_types[-1]
            fourth_vol = fut_volumes[-1]
            third_vol = fut_volumes[-2]

            if fourth_color == tgt and fourth_vol < third_vol:
                # Map target color to trade type
                if tgt == "RED":
                    return "PE"
                elif tgt == "GREEN":
                    return "CE"

    return None

def cleanup_open_orders(kite, trades):
    """
    Cancel all stop-loss (SL) and target-profit (TP) orders for open trades.

    This function iterates over the open trades dictionary and cancels any associated
    SL or TP orders via the Kite Connect API. It skips the internal "_last_candle_end"
    key used for tracking candle processing. Each cancellation attempt is retried
    using `retry_kite_call` to handle temporary API failures.

    Parameters
    ----------
    kite : object
        An authenticated Kite Connect API client instance.
    trades : dict
        Dictionary of open trades. Each trade entry is expected to contain:
        - 'sl_id': Stop-loss order ID (optional)
        - 'tp_id': Target-profit order ID (optional)
        - 'symbol': Trading symbol (used for logging)

        The dictionary may also contain a "_last_candle_end" key, which is ignored.

    Returns
    -------
    None
        The function logs the success or failure of each cancellation but does not return a value.

    Example
    -------
    > cleanup_open_orders(kite, open_trades)
    INFO: Cancelled order 12345 for NIFTY2590224700PE
    WARNING: Failed cancelling 12346: order does not exist
    """
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
    """
    Calculate the realized PnL (Profit and Loss) for a single option trade.

    This function fetches all trades (fills) from Kite Connect, filters for the
    stop-loss (SL) or target-profit (TP) orders associated with the given trade,
    and computes the realized PnL based on the difference between the entry price
    and the average exit price.

    Parameters
    ----------
    kite : object
        An authenticated Kite Connect API client instance.
    trade : dict
        Dictionary containing trade details. Expected keys:
        - 'entry_ltp' : float - entry price of the trade
        - 'qty' : int - quantity/lot size
        - 'sl_id' : int/str - stop-loss order ID
        - 'tp_id' : int/str - target-profit order ID
        - 'leg' : str - option type ("CE" or "PE")

    Returns
    -------
    float
        Realized PnL for the trade. Positive for profit, negative for loss.
        Returns 0.0 if fills are not found or an error occurs.

    Notes
    -----
    - Assumes a simple calculation: (sell_price - buy_price) * qty for both CE and PE.
    - Uses `retry_kite_call` to safely fetch trades in case of temporary API failures.
    - Logs a warning if PnL computation fails.

    Example
    -------
    > trade = {"entry_ltp": 144.5, "qty": 75, "sl_id": 12345, "tp_id": 12346, "leg": "CE"}
    > pnl = compute_realized_pnl(kite, trade)
    > print(pnl)
    218.75
    """
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
    """
    Perform a pre-start health check on the Kite Connect API.

    This function verifies that the Kite session is active and the API is reachable
    by attempting to fetch the user profile. If that fails, it tries to fetch
    the equity margins as a secondary check.

    Parameters
    ----------
    kite : object
        An authenticated Kite Connect API client instance.

    Returns
    -------
    bool
        True if the API session appears healthy and reachable, False otherwise.

    Notes
    -----
    - Uses `retry_kite_call` to handle temporary API failures.
    - This check ensures that the strategy does not start with an invalid session
      or connectivity issues.

    Example
    -------
    > if not prestart_healthcheck(kite):
    >     logging.critical("Kite session not healthy — aborting strategy")
    """
    profile = retry_kite_call(kite.profile)
    if not profile:
        margins = retry_kite_call(kite.margins, "equity")
        if not margins:
            return False
    return True

# -------------------- INSTRUMENT HELPERS (robust) --------------------
def find_spot_token(instruments_df):
    """
    Retrieve the instrument token for the 'NIFTY 50' index from the instruments DataFrame.

    This function searches the provided instruments DataFrame for the row corresponding
    to 'NIFTY 50' in the 'INDICES' segment and returns its instrument token. If no such
    row is found, it falls back to `SPOT_TOKEN_DEFAULT`.

    Parameters
    ----------
    instruments_df : pd.DataFrame
        DataFrame containing instrument metadata with at least the columns:
        - 'tradingsymbol' : str
        - 'segment' : str
        - 'instrument_token' : int

    Returns
    -------
    int
        The instrument token for 'NIFTY 50', or `SPOT_TOKEN_DEFAULT` if not found.

    Notes
    -----
    - Logs a warning if the fallback token is used, suggesting manual verification.
    - The returned token should be verified to exist in Kite Connect instruments before use.

    Example
    -------
    > token = find_spot_token(instruments_df)
    > print(token)
    256265
    """
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
    Retrieve the nearest expiry futures contract for a given underlying.

    This function searches the instruments DataFrame for futures contracts matching
    the specified underlying, normalizes segment naming conventions, and returns
    the symbol, instrument token, and expiry date of the nearest expiry contract.

    Parameters
    ----------
    instruments_df : pd.DataFrame
        DataFrame containing instrument metadata with at least the columns:
        - 'tradingsymbol' : str
        - 'name' : str (underlying name)
        - 'segment' : str (optional, e.g., "NFO-FUT")
        - 'instrument_token' : int
        - 'expiry' : str or datetime
    underlying : str, optional
        The underlying asset name for which to find the nearest future (default is UNDERLYING).

    Returns
    -------
    tuple
        - tradingsymbol (str): Future contract symbol
        - instrument_token (int): Instrument token
        - expiry (datetime): Expiry date of the future

    Raises
    ------
    ValueError
        If no futures contracts are found for the given underlying.

    Notes
    -----
    - Handles variations in segment naming (e.g., presence or absence of a 'segment' column).  
    - Sorts by expiry date to return the nearest expiring contract.

    Example
    -------
    > symbol, token, expiry = get_nearest_future(instruments_df, "NIFTY")
    > print(symbol, token, expiry)
    NIFTY25SEPFUT 123456 2025-09-25 00:00:00
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

def get_atm_options_and_lot(instruments_df, spot_price, expiry_dt=None, underlying=UNDERLYING):
    """
    Retrieve the ATM (At-The-Money) Call and Put option instruments for a given underlying.

    This function selects the options whose strike is closest to the current spot price,
    rounding to the nearest `STRIKE_INTERVAL`. If the exact strike is unavailable, it
    chooses the closest available strike. The function can filter by a specific expiry
    or automatically pick the nearest expiry.

    Parameters
    ----------
    instruments_df : pd.DataFrame
        DataFrame containing option instrument metadata, expected columns:
        - 'tradingsymbol' : str
        - 'name' : str (underlying)
        - 'segment' : str (optional, e.g., "NFO-OPT")
        - 'instrument_type' : str ("CE" or "PE")
        - 'strike' : float
        - 'expiry' : str or datetime
        - 'instrument_token' : int
        - 'lot_size' or 'lots' : int (optional, default=1)
    spot_price : float
        Current spot price of the underlying asset.
    expiry_dt : datetime.date, datetime, or str, optional
        Specific expiry date to filter options. If None, the nearest expiry is used.
    underlying : str, optional
        Name of the underlying asset (default is UNDERLYING).

    Returns
    -------
    tuple
        ((ce_tradingsymbol, ce_token, ce_lot), (pe_tradingsymbol, pe_token, pe_lot))
        - CE tuple: Call option symbol, instrument token, lot size
        - PE tuple: Put option symbol, instrument token, lot size

    Raises
    ------
    ValueError
        - If no option instruments exist for the given underlying
        - If no non-expired instruments are found
        - If no options exist for the specified expiry
        - If ATM CE/PE cannot be determined

    Notes
    -----
    - Filters out expired options.
    - Selects the closest strike to `spot_price` using `STRIKE_INTERVAL` rounding.
    - Falls back to nearest available strike if exact strike CE/PE is not present.
    - Defaults lot size to 1 if not available in instrument metadata.

    Example
    -------
    > ce, pe = get_atm_options_and_lot(instruments_df, spot_price=24785.0)
    > print(ce)
    ('NIFTY25SEP24800CE', 123456, 75)
    > print(pe)
    ('NIFTY25SEP24800PE', 123457, 75)
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
    """
    Compute the quantity of contracts to place, ensuring it is a multiple of the option lot size.

    This function adjusts the desired total quantity to the nearest lower multiple of
    the option's lot size. If the configured quantity is smaller than one lot, it raises
    a ValueError.

    Parameters
    ----------
    total_quantity : int
        Desired total quantity of contracts to trade.
    lot_size : int
        Number of units per option contract (lot size).

    Returns
    -------
    int
        Adjusted quantity that is a multiple of `lot_size`.

    Raises
    ------
    ValueError
        If `total_quantity` is smaller than a single lot, indicating that no valid
        trade can be placed with the given configuration.

    Example
    -------
    > compute_order_qty(total_quantity=150, lot_size=75)
    150
    > compute_order_qty(total_quantity=100, lot_size=75)
    75
    > compute_order_qty(total_quantity=50, lot_size=75)
    ValueError: Configured QUANTITY=50 is less than single option lot_size=75. Adjust QUANTITY or lot_size.
    """
    lots = total_quantity // lot_size
    if lots < 1:
        raise ValueError(f"Configured QUANTITY={total_quantity} is less than single option lot_size={lot_size}. Adjust QUANTITY or lot_size.")
    return lots * lot_size  # final quantity (multiple of lot_size)

def calculate_sl_tp(entry_price, leg_type):
    """
    Calculate stop-loss (SL) and target-profit (TP) levels for an option trade.

    Parameters
    ----------
    entry_price : float
        The entry price of the option trade. Must be positive.
    leg_type : str
        The option type, e.g., "CE" for Call or "PE" for Put. (Currently not used
        in calculation but may be used for future conditional logic.)

    Returns
    -------
    tuple
        (sl, tp) : tuple of floats
        - sl : Stop-loss price (rounded to 1 decimal place)
        - tp : Target-profit price (rounded to 1 decimal place)
        Returns (None, None) if `entry_price` is invalid (<= 0 or None).

    Notes
    -----
    - SL and TP percentages are taken from global constants `SL_PCT` and `TP_PCT`.
    - For future expansion, `leg_type` can be used to invert or adjust SL/TP logic
      for CE vs PE trades.

    Example
    -------
    > calculate_sl_tp(entry_price=144.5, leg_type="CE")
    (146.8, 141.3)
    """
    if not entry_price or entry_price <= 0:
        return None, None
    sl = entry_price * (1 + SL_PCT)
    tp = entry_price * (1 - TP_PCT)
    return round(sl, 1), round(tp, 1)


def force_exit_open_trades(kite, trades):
    """
    Forcefully close all open trades at market price for end-of-day (EOD) safety.

    This function iterates over the provided trades dictionary and places a market
    buy order to close any open short positions (or adjust accordingly for long
    trades). It ignores trades already closed and internal bookkeeping keys.

    Parameters
    ----------
    kite : object
        An authenticated Kite Connect API client instance.
    trades : dict
        Dictionary of open trades. Each trade should contain:
        - 'symbol' : str - option symbol
        - 'qty' : int - quantity of contracts
        - 'status' : str - trade status ("OPEN" or "CLOSED")
        May also contain internal keys like '_last_candle_end'.

    Returns
    -------
    None
        Logs success or failure for each trade. Does not return any values.

    Notes
    -----
    - Uses a market order to exit positions immediately at current prices.
    - Intended as a safety mechanism to prevent overnight exposure.
    - Logs an error if the order fails, but continues processing other trades.

    Example
    -------
    >>> force_exit_open_trades(kite, open_trades)
    INFO: Forced exit for NIFTY25SEP24800CE at EOD
    """
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

def place_protected_sell(kite, symbol, qty, entry_price, leg_type, dry_run=False):
    """
    Place a market SELL order with associated stop-loss (SL-M) and target-profit (LIMIT) orders,
    with automatic cleanup if any child order fails.

    Parameters
    ----------
    kite : object
        An authenticated Kite Connect API client instance.
    symbol : str
        Option instrument symbol to sell.
    qty : int
        Quantity to sell (must respect lot size).
    entry_price : float
        Entry price for calculating SL and TP levels.
    leg_type : str
        Option type ("CE" or "PE") used for SL/TP calculation.
    dry_run : bool, optional
        If True, does not place real orders; returns placeholder IDs for testing.

    Returns
    -------
    tuple
        (sell_id, sl_id, tp_id) : tuple of order IDs
        - sell_id : Order ID of the main SELL order (or "DRY_SELL" in dry_run)
        - sl_id   : Order ID of the stop-loss order (or "DRY_SL" in dry_run)
        - tp_id   : Order ID of the target-profit order (or "DRY_TP" in dry_run)
        Returns None for SL/TP IDs if placement failed after SELL.

    Notes
    -----
    - Uses `calculate_sl_tp` to compute SL and TP levels.
    - If TP order placement fails, attempts to cancel the SL order to prevent orphan orders.
    - `retry_kite_call` ensures transient API failures are retried before failing.
    - Logging tracks each step and any errors encountered.

    Example
    -------
    > sell_id, sl_id, tp_id = place_protected_sell(kite, "NIFTY25SEP24800CE", 75, 144.5, "CE")
    > print(sell_id, sl_id, tp_id)
    250901100719324 250901100719325 250901100719326
    """

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
            quantity=qty, order_type="SL", trigger_price=sl,
            product="MIS", variety="regular"
        )
        if not resp or "order_id" not in resp:
            logging.error("SL order placement failed: %s", resp)
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
    """
    Fetch the last traded price (LTP) for a given instrument symbol with retries.

    Parameters
    ----------
    symbol : str
        The instrument symbol for which to fetch the LTP.
    max_tries : int, optional
        Maximum number of retry attempts in case of transient API failures (default=3).

    Returns
    -------
    float or None
        The last traded price of the instrument, or None if unable to fetch after retries.

    Notes
    -----
    - Uses `retry_kite_call` to handle transient API failures.
    - Sleeps 1 second between retries.
    - Logs warnings on failed attempts and errors.

    Example
    -------
    > ltp = safe_get_ltp("NIFTY25SEP24800CE")
    > print(ltp)
    144.5
    """

    for i in range(max_tries):
        try:
            resp = retry_kite_call(kite.ltp, [symbol])
            if resp and symbol in resp and "last_price" in resp[symbol]:
                return resp[symbol]["last_price"]
        except Exception as e:
            logging.warning("LTP fetch attempt %d failed for %s: %s", i+1, symbol, e)
        time.sleep(1)
    logging.error("Failed to fetch LTP for %s after %d attempts", symbol, max_tries)
    return None

# -------------------- STRATEGY CORE --------------------
def run_consecutive_candle_strategy_auto_oco(
        kite, instruments_df, qty,
        candle_start_time, candle_end_time, max_entries=2):
    """
    Run a consecutive candle-based intraday options trading strategy with OCO (One-Cancels-Other) protection.

    Strategy Overview:
    -----------------
    1. Uses Spot NIFTY 50 5-minute OHLC candles to classify candle color (GREEN/RED).
    2. Confirms trend/reversal using nearest futures volume data.
    3. Detects dynamic reversal patterns based on configured sequence of candle colors.
    4. Places protected ATM option short trades (CE/PE) with Stop Loss (SL) and Take Profit (TP).
    5. Tracks realized PnL and enforces a daily loss limit.
    6. Supports max entries per session, EOD forced exits, DRY_RUN mode, and KILL_SWITCH for emergency stop.

    Parameters:
    -----------
    kite : object
        Authenticated Kite Connect instance used for data fetching and order placement.
    instruments_df : pandas.DataFrame
        DataFrame containing instrument metadata (tradingsymbol, instrument_token, expiry, strike, etc.).
    qty : int
        Total number of units to trade; will be adjusted to nearest option lot size.
    candle_start_time : datetime.time
        Local time (usually IST) at which the strategy should start monitoring candles.
    candle_end_time : datetime.time
        Local time (usually IST) at which the strategy should stop monitoring candles (EOD exit).
    max_entries : int, optional
        Maximum number of trades allowed per session. Default is 2.

    Behavior:
    ---------
    - Waits until `candle_start_time` before starting.
    - Fetches the latest spot and futures candles at 5-minute intervals.
    - Maintains a rolling window of the last N candle types for pattern detection.
    - On pattern match, identifies ATM CE/PE options and computes order quantity based on lot size.
    - Fetches the last traded price (LTP) and calculates SL/TP levels.
    - Places a protected sell (market) order along with SL-M and TP orders.
    - Logs detailed information for each 5-minute candle, including pattern reason, order entry, SL, and TP.
    - Monitors open trades and updates realized PnL when SL or TP is hit.
    - Respects `KILL_SWITCH`, `DRY_RUN`, and `MAX_DAILY_LOSS` constraints.
    - Performs forced exit for all open trades at `candle_end_time` or on manual interrupt.

    Returns:
    --------
    None
        This function runs indefinitely during market hours and exits on EOD, max entries reached, KILL_SWITCH, or manual interruption.
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
        logging.warning("Spot token lookup failed: %s — using fallback token", e)
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
        # Ensure we don't start before candle_start_time
        now = datetime.now(IST) if IST else datetime.now()
        if candle_start_time and now.time() < candle_start_time:
            logging.info("Current time %s < candle_start_time %s; sleeping until start", now.time(), candle_start_time)
            wait_until(candle_start_time)

        # Parse once at startup
        patterns = parse_reversal_patterns(REVERSAL_PATTERNS)

        while True:
            now = datetime.now(IST) if IST else datetime.now()

            # End-of-day check
            if candle_end_time and now.time() >= candle_end_time:
                logging.info("Market end reached (%s) — forcing EOD exit", candle_end_time)
                break

            if entries_taken >= max_entries:
                logging.info("Max entries (%d) reached. Waiting for close.", max_entries)
                time.sleep(10)
                continue

            if KILL_SWITCH:
                logging.critical("KILL_SWITCH triggered mid-run — exiting")
                break

            # Align to current candle window (5-min multiples)
            aligned = now.replace(second=0, microsecond=0)
            aligned = aligned - timedelta(minutes=aligned.minute % CANDLE_INTERVAL)
            c_start = aligned - timedelta(minutes=CANDLE_INTERVAL)
            c_end = aligned

            # avoid reprocessing same candle
            if open_trades.get("_last_candle_end") == c_end:
                time.sleep(1)
                continue

            # Fetch candles (use retry wrapper)
            spot_data = retry_kite_call(kite.historical_data, spot_token, c_start, c_end, "5minute")
            fut_data = retry_kite_call(kite.historical_data, fut_token, c_start, c_end, "5minute")

            if not spot_data or not fut_data:
                logging.warning("Missing candle data for %s - %s", c_start, c_end)
                time.sleep(1)
                continue

            # Defensive: ensure last entries are dict-like (Kite may return unexpected types)
            last_spot = spot_data[-1] if isinstance(spot_data, (list, tuple)) and len(spot_data) > 0 else None
            last_fut  = fut_data[-1]  if isinstance(fut_data, (list, tuple)) and len(fut_data) > 0 else None

            if not isinstance(last_spot, dict):
                logging.warning("Skipping invalid spot candle object for %s - %s", c_start, c_end)
                time.sleep(1)
                continue
            if not isinstance(last_fut, dict):
                logging.warning("Skipping invalid futures candle object for %s - %s", c_start, c_end)
                time.sleep(1)
                continue

            sc, fc = last_spot, last_fut

            # classify and push into window
            ctype = classify_candle_row(sc)
            candle_types.append(ctype)
            raw_spot_candles.append(sc)
            fut_volumes.append(fc.get("volume", 0))

            # Rolling window cap
            MAX_WINDOW = 12
            if len(candle_types) > MAX_WINDOW:
                candle_types.pop(0)
                raw_spot_candles.pop(0)
                fut_volumes.pop(0)

            logging.info("Candle %s-%s | Spot %s O=%s C=%s | FutVol=%s",
                         c_start.time(), c_end.time(), ctype, sc.get("open"), sc.get("close"), fc.get("volume"))

            open_trades["_last_candle_end"] = c_end

            # Inside your loop
            target = check_dynamic_pattern_with_volume(candle_types, fut_volumes, patterns)
            if target:
                logging.info("Pattern matched -> %s trade", target)
                spot_price = sc.get("close")
                if spot_price is None:
                    logging.warning("Spot close missing; skipping pattern execution")
                    time.sleep(1)
                    continue

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
                if ltp is None:
                    logging.error("Failed to fetch LTP for %s; skipping order", sel_sym)
                    time.sleep(1)
                    continue

                # daily loss guard
                if realized_pnl <= -abs(MAX_DAILY_LOSS):
                    logging.critical("Max daily loss (%.2f) hit — no new trades", realized_pnl)
                    break

                # place protected sell
                sell_id, sl_id, tp_id = place_protected_sell(
                    kite, sel_sym, order_qty, ltp, leg_type, dry_run=DRY_RUN)
                if not sell_id:
                    logging.error("Protected sell failed for %s", sel_sym)
                    # nothing to cleanup here since place_protected_sell is responsible
                else:
                    entries_taken += 1
                    open_trades[str(sell_id)] = {
                        "symbol": sel_sym, "leg": leg_type, "qty": order_qty,
                        "entry_ltp": ltp, "sl_id": sl_id, "tp_id": tp_id, "status": "OPEN"
                    }
                    logging.info("Trade entered: %s | qty=%s | entry=%s | sell_id=%s", sel_sym, order_qty, ltp, sell_id)

            # Monitor open trades (use retry for orders)
            orders = retry_kite_call(kite.orders) or []
            for tid, trade in list(open_trades.items()):
                if tid == "_last_candle_end":
                    continue
                if trade.get("status") != "OPEN":
                    continue

                # normalize id comparison as strings
                sl_id_s = str(trade.get("sl_id")) if trade.get("sl_id") is not None else None
                tp_id_s = str(trade.get("tp_id")) if trade.get("tp_id") is not None else None

                sl_status = next((o for o in orders if str(o.get("order_id")) == sl_id_s), None)
                tp_status = next((o for o in orders if str(o.get("order_id")) == tp_id_s), None)

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
        logging.exception("Unhandled exception in main loop: %s", e)
    finally:
        logging.info("Final cleanup: forcing EOD exits if required")
        force_exit_open_trades(kite, open_trades)
        logging.info("Strategy stopped. Realized PnL (approx): %.2f", realized_pnl)


# -------------------- MAIN EXECUTION --------------------
if __name__ == "__main__":
    """
    Main execution entry point for the consecutive candle-based intraday options strategy.

    Behavior:
    ---------
    1. Computes local market start and end times (in IST if available).
    2. Waits until the market start time before executing the strategy.
    3. Validates that the instrument DataFrame is available and not empty.
    4. Calls `run_consecutive_candle_strategy_auto_oco` with the configured parameters:
    - Kite Connect instance
    - Instruments DataFrame
    - Trade quantity
    - Market start/end times
    - Maximum allowed entries
    5. Logs script completion after the strategy exits.

    Notes:
    ------
    - This block only executes when the script is run directly (`__name__ == "__main__"`).
    - Exits immediately if instruments list is empty.
    - Ensures strategy respects configured market timings and maximum entries.
    """
    
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

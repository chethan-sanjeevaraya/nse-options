import os
import logging
import pandas as pd
import numpy as np
import datetime as dt
from datetime import datetime, time
from kiteconnect import KiteConnect
from dotenv import load_dotenv

# -------------------- CONFIG --------------------
load_dotenv(".env")

api_key = os.getenv("API_KEY")
api_secret = os.getenv("API_SECRET")
access_token = os.getenv("ACCESS_TOKEN")
quantity = int(os.getenv("QUANTITY", "75"))
only_leg = os.getenv("ONLY_CE_PE", "BOTH").upper()
log_file = "atm_market_strategy.log"

# -------------------- LOGGING --------------------
logging.basicConfig(
    filename=log_file,
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    force=True
)
logging.info("🚀 Starting Optimized ATM Market Order Strategy...")

# -------------------- INIT --------------------
kite = KiteConnect(api_key=api_key)
kite.set_access_token(access_token)

# -------------------- FUNCTIONS --------------------
instrument_list = kite.instruments("NFO")

def option_contracts(ticker, option_type="CE", exchange="NFO"):
    option_contracts = []
    for instrument in instrument_list:
        if instrument["name"] == ticker and instrument["instrument_type"] == option_type:
            option_contracts.append(instrument)
    return pd.DataFrame(option_contracts)

def option_contracts_closest(ticker, duration=0, option_type="CE", exchange="NFO"):
    df_opt_contracts = option_contracts(ticker, option_type, exchange)
    df_opt_contracts["time_to_expiry"] = (pd.to_datetime(df_opt_contracts["expiry"]) - dt.datetime.now()).dt.days
    min_day_count = np.sort(df_opt_contracts["time_to_expiry"].unique())[duration]
    return df_opt_contracts[df_opt_contracts["time_to_expiry"] == min_day_count].reset_index(drop=True)

def option_chain(ticker, underlying_price, duration=0, num=5, option_type="CE", exchange="NFO"):
    df_opt_contracts = option_contracts_closest(ticker, duration, option_type, exchange)
    df_opt_contracts.sort_values(by=["strike"], inplace=True, ignore_index=True)
    atm_idx = abs(df_opt_contracts["strike"] - underlying_price).argmin()
    up = int(num / 2)
    dn = num - up
    return df_opt_contracts.iloc[atm_idx - up:atm_idx + dn]

def placeLimitOrder(order_params):    
    order_id = kite.place_order(
        tradingsymbol=order_params['tradingsymbol'],
        exchange=order_params['exchange'],
        transaction_type=order_params['transaction_type'],
        quantity=order_params['quantity'],
        price=order_params['price'],
        order_type=order_params['order_type'],
        product=order_params['product'],
        variety=order_params['variety']
    )
    return order_id

def placeMarketOrder(order_params):    
    order_id = kite.place_order(
        tradingsymbol=order_params['tradingsymbol'],
        exchange=order_params['exchange'],
        transaction_type=order_params['transaction_type'],
        quantity=order_params['quantity'],
        order_type="MARKET",
        product=order_params['product'],
        variety=order_params['variety']
    )
    return order_id

# -------------------- MAIN EXECUTION --------------------
try:
    # Check market hours
    now = datetime.now()
    if not (time(9, 15) <= now.time() <= time(15, 30)):
        logging.warning("❌ Not within market hours (09:15–15:30). Exiting.")
        exit()

    # Get NIFTY spot price
    spot = kite.quote("NSE:NIFTY 50")["NSE:NIFTY 50"]["last_price"]
    atm_strike = round(spot / 50) * 50
    logging.info(f"📈 Spot: ₹{spot}, ATM Strike: {atm_strike}")

    # Get closest expiry option chain (CE and PE both)
    chain_ce = option_chain("NIFTY", spot, 0, 5, "CE")
    chain_pe = option_chain("NIFTY", spot, 0, 5, "PE")

    # Get ATM CE and PE contracts
    atm_ce = chain_ce[chain_ce["strike"] == atm_strike].iloc[0] if not chain_ce[chain_ce["strike"] == atm_strike].empty else None
    atm_pe = chain_pe[chain_pe["strike"] == atm_strike].iloc[0] if not chain_pe[chain_pe["strike"] == atm_strike].empty else None

    if atm_ce is None or atm_pe is None:
        logging.error("❌ ATM CE or PE contract not found.")
        exit()

    # Determine which legs to trade
    legs = []
    if only_leg == "CE":
        legs.append(atm_ce)
    elif only_leg == "PE":
        legs.append(atm_pe)
    else:
        legs.extend([atm_ce, atm_pe])

    # Place Market Orders
    for leg in legs:
        order_param = {
            "exchange": "NFO",
            "tradingsymbol": leg["tradingsymbol"],
            "transaction_type": "SELL",
            "variety": "regular",
            "product": "MIS",
            "quantity": leg["lot_size"]
        }
        order_id = placeMarketOrder(order_param)
        logging.info(f"✅ Market Order Placed: {leg['tradingsymbol']} | Qty: {leg['lot_size']} | Order ID: {order_id}")

except Exception as e:
    logging.error(f"❌ Error occurred: {e}")

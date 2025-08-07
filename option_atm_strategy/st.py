"""
Streamlit will redirect to zerodha developer api page and user should authenthicate with password and app code,
then enter the generated request token then job will store the request token and  access token to the env file
"""
import streamlit as st
from kiteconnect import KiteConnect
from dotenv import load_dotenv, set_key
import os

env_file = ".env"
load_dotenv(env_file)

api_key = os.getenv("API_KEY")
api_secret = os.getenv("API_SECRET")

kite = KiteConnect(api_key=api_key)

st.title("Zerodha Login - Streamlit")

login_url = kite.login_url()
st.write("🔑 [Click here to login to Zerodha]({})".format(login_url))

request_token = st.text_input("Paste your request_token here:")

if st.button("Generate Access Token"):
    if request_token:
        try:
            data = kite.generate_session(request_token, api_secret=api_secret)
            access_token = data["access_token"]
            kite.set_access_token(access_token)

            # Save to .env
            set_key(env_file, "ACCESS_TOKEN", access_token)

            st.success("Access Token saved and session established!")

            # Place a test order (buy INFY)
            order_id = kite.place_order(
                variety=kite.VARIETY_REGULAR,
                exchange=kite.EXCHANGE_NSE,
                tradingsymbol="INFY",
                transaction_type=kite.TRANSACTION_TYPE_BUY,
                quantity=1,
                order_type=kite.ORDER_TYPE_MARKET,
                product=kite.PRODUCT_MIS
            )
            st.success(f"✅ Order Placed. Order ID: {order_id}")

        except Exception as e:
            st.error(f"Error: {e}")
    else:
        st.warning("Please paste the request_token.")

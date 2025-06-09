"""
flask will redirect to zerodha developer api and get the access request token and store the access token to the env file
"""
from flask import Flask, request
from kiteconnect import KiteConnect
from dotenv import load_dotenv, set_key
import webbrowser
import os
import threading

env_file = ".env"
load_dotenv(env_file)

api_key = os.getenv("API_KEY")
api_secret = os.getenv("API_SECRET")

kite = KiteConnect(api_key=api_key)

app = Flask(__name__)

@app.route("/login")
def login():
    request_token = request.args.get("request_token")
    print(f"Request Token: {request_token}")
    
    try:
        data = kite.generate_session(request_token, api_secret=api_secret)
        access_token = data["access_token"]
        print(f"Access Token: {access_token}")

        # Save access_token to .env
        set_key(".env", "ACCESS_TOKEN", access_token)

        return "Access Token saved! You may close this tab."
    except Exception as e:
        print("ERROR:", e)
        return f"Error: {e}"

def open_browser():
    import time
    time.sleep(1)
    print("Opening browser for Zerodha login...")
    webbrowser.open(kite.login_url())

if __name__ == "__main__":
    print("Starting Flask server on http://127.0.0.1:5000 ...")
    threading.Thread(target=open_browser).start()
    app.run(port=5000)

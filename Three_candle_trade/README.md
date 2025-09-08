1.	Spot/Futures separation: Spot (NIFTY 50, token 256265) for OHLC, Futures for volume.
2.	REVERSAL_PATTERNS parsing: Fully dynamic with parse_reversal_patterns and check_dynamic_pattern.
3.	Auto ATM option detection: Uses get_atm_options_and_lot.
4.	Lot size handling: compute_order_qty ensures correct multiples.
5.	Order placement with SL/TP: place_protected_sell with buffer + protection.
6.	PnL tracking: compute_realized_pnl included.
7.	Daily safety: MAX_DAILY_LOSS and KILL_SWITCH implemented.
8.	Health check: prestart_healthcheck with profile/margins.
9.	End-of-day forced exit: force_exit_open_trades.
10.	Retry/backoff wrapper: retry_kite_call.
11.	Timezones: ZoneInfo / pytz fallback.
12.	Dry-run mode: DRY_RUN respected.
13.	Pattern parsing robustness: Handles malformed blocks.
14.	Logging: File + console with timestamps.
15.	Configurable SL/TP %: SL_PCT, TP_PCT in .env.
16. Safe cleanup of trades.
17. Resilient LTP fetch.
18. Trades tracked with PnL updates.

📘 Strategy Features & Safeguards (with Code References)
________________________________________
1. Spot/Futures Separation
•	Spot OHLC (NIFTY 50, token 256265) → used for candle color detection.
o	Function: find_spot_token() (≈ line 295).
o	Token fallback: SPOT_TOKEN_DEFAULT = 256265.
•	Futures data → used for volume confirmation.
o	Function: get_nearest_future() (≈ line 310).
o	Selected in run_consecutive_candle_strategy_auto_oco() (≈ line 495).
________________________________________
2. Dynamic Pattern Parsing
•	Pattern parsing from .env: REVERSAL_PATTERNS.
•	Function: parse_reversal_patterns() (≈ line 370).
o	Example: "RED,RED,RED:GREEN;GREEN,GREEN,GREEN:RED".
•	Matching logic: check_dynamic_pattern() (≈ line 390).
•	Integrated into main loop: run_consecutive_candle_strategy_auto_oco() (≈ line 560).
________________________________________
3. ATM Option Detection
•	Function: get_atm_options_and_lot() (≈ line 420).
•	Dynamically selects nearest expiry + closest strike.
•	Handles missing strikes by fallback search.
•	Used in main loop when a pattern matches (≈ line 570).
________________________________________
4. Lot Size Handling
•	Function: compute_order_qty() (≈ line 460).
•	Ensures trade quantity is a multiple of lot_size.
•	Throws error if configured QUANTITY < lot_size.
•	Called in main loop after ATM option detection (≈ line 585).
________________________________________
5. Protected Order Placement
•	Function: place_protected_sell() (≈ line 240 & 520).
•	Workflow:
1.	Places SELL market order.
2.	Places SL-M buy order with buffer (SL_BUFFER).
3.	Places LIMIT TP order.
•	If TP placement fails, SL is cancelled to avoid dangling orders.
________________________________________
6. PnL Tracking
•	Function: compute_realized_pnl() (≈ line 210).
•	Fetches trade fills via kite.trades().
•	Calculates average sell price – average buy price × qty.
•	Integrated in trade monitor loop (≈ line 655).
________________________________________
7. Daily Risk Safeguards
•	Controlled by .env:
o	MAX_DAILY_LOSS (≈ line 80).
o	KILL_SWITCH (≈ line 85).
•	Implemented in main loop before order placement (≈ line 600).
________________________________________
8. Pre-Start Health Check
•	Function: prestart_healthcheck() (≈ line 270).
•	Runs kite.profile() or fallback kite.margins("equity").
•	Called at start of run_consecutive_candle_strategy_auto_oco() (≈ line 505).
________________________________________
9. End-of-Day Safety
•	Function: force_exit_open_trades() (≈ line 250).
•	At strategy end (finally block, ≈ line 690), forces market BUY to exit.
•	Ensures no overnight exposure.
________________________________________
10. Retry/Backoff Mechanism
•	Function: retry_kite_call() (≈ line 120 & 150).
•	Retries failed Kite API calls with exponential backoff.
•	Handles HTTP 429 (rate limit) gracefully.
•	Used in all Kite API calls (historical, ltp, orders, trades).
________________________________________
11. Timezone Safety
•	Primary: ZoneInfo("Asia/Kolkata").
•	Fallback: pytz.timezone("Asia/Kolkata").
•	Final fallback: naive datetime.now().
•	Declared at script start (≈ line 20).
________________________________________
12. Dry-Run Mode
•	Flag: .env → DRY_RUN=true.
•	Used in place_protected_sell() (≈ line 240).
•	Orders are logged only, not sent.
•	Printed in logs: "DRY_SELL", "DRY_SL", "DRY_TP".
________________________________________
13. Robust Pattern Parsing
•	parse_reversal_patterns() (≈ line 370):
o	Supports delimiters ;, ,, _.
o	Skips malformed entries with warnings.
•	Example logs: "Skipping malformed pattern block...".
________________________________________
14. Detailed Logging
•	Configured at startup (≈ line 90).
•	Outputs to both:
o	File: t6_final_strategy.log.
o	Console (with timestamps).
•	Logs candles, trades, SL/TP triggers, and cleanup.
________________________________________
15. Configurable SL/TP %
•	Controlled via .env:
o	SL_PCT (stop-loss %).
o	TP_PCT (target %).
•	Applied in calculate_sl_tp() (≈ line 470).
________________________________________
16. Safe Trade Cleanup
•	Function: cleanup_open_orders() (≈ line 180).
•	Cancels SL/TP orders if entry fails.
•	Called in final finally block of strategy (≈ line 690).
________________________________________
17. Resilient LTP Fetch
•	Function: safe_get_ltp() (≈ line 300).
•	Retries up to 3 times to fetch LTP.
•	Returns None if unavailable.
•	Used in main loop before placing orders (≈ line 590).
________________________________________
18. PnL-Linked Trade Tracking
•	Trades stored in open_trades dict (≈ line 540).
•	Keys: sell order ID → {symbol, qty, sl_id, tp_id, status}.
•	Updated when SL/TP completes (≈ line 650).
•	Realized PnL aggregated across trades.

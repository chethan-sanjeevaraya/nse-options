import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
import os

from logger import setup_logger
import logging

setup_logger()

logging.info("Loading 5-minute data from CSV")
df = pd.read_csv("NSEdata.csv", encoding='utf-8-sig')
df.columns = df.columns.str.strip().str.lower()
df['date'] = pd.to_datetime(df['date'], utc=True)
df['date'] = df['date'].dt.tz_convert('Asia/Kolkata')
df['date'] = df['date'].dt.tz_localize(None)

df.sort_values('date', inplace=True)
df.reset_index(drop=True, inplace=True)

logging.info("Data preprocessing completed")

df['only_date'] = df['date'].dt.date
df['only_time'] = df['date'].dt.time

breakout_records = []

for date, day_df in df.groupby('only_date'):
    logging.info(f"\nProcessing date: {date}")
    day_df = day_df.sort_values('date').reset_index(drop=True)
    ref = day_df.loc[day_df['only_time'] == datetime.strptime("09:15", "%H:%M").time()]
    if ref.empty:
        logging.warning(f"09:15 candle not found for {date}, skipping...")
        continue

    ref = ref.iloc[0]
    ref_high = ref['high']
    ref_low = ref['low']
    breakout_detected = False

    for i in range(1, len(day_df)):
        row = day_df.iloc[i]
        entry_time = row['only_time']
        if entry_time > datetime.strptime("11:00", "%H:%M").time():
            break

        entry = row['open']
        candle_time = row['date'].strftime('%Y-%m-%d %H:%M')

        if not breakout_detected and entry > ref_high:
            prev_low = day_df.iloc[i - 1]['low']
            sl = prev_low
            risk = entry - sl
            target = entry + 2 * risk
            breakout_type = "LONG"
        elif not breakout_detected and entry < ref_low:
            prev_high = day_df.iloc[i - 1]['high']
            sl = max(entry * 1.02, prev_high)
            risk = sl - entry
            target = entry - 2 * risk
            breakout_type = "SHORT"
        else:
            continue

        logging.info(f"{candle_time} - {breakout_type} breakout detected. Entry={entry:.2f}, SL={sl:.2f}, Target={target:.2f}")

        result = "Neither Hit"
        exit_price = None
        forced_exit_price = None

        for j in range(i + 1, len(day_df)):
            next_candle = day_df.iloc[j]
            if next_candle['only_time'] > datetime.strptime("15:15", "%H:%M").time():
                break

            if breakout_type == "LONG":
                if next_candle['low'] <= sl:
                    result = "Stop Loss Hit"
                    exit_price = sl
                    break
                elif next_candle['high'] >= target:
                    result = "Target Hit"
                    exit_price = target
                    break
            elif breakout_type == "SHORT":
                if next_candle['high'] >= sl:
                    result = "Stop Loss Hit"
                    exit_price = sl
                    break
                elif next_candle['low'] <= target:
                    result = "Target Hit"
                    exit_price = target
                    break

            if next_candle['only_time'] == datetime.strptime("15:15", "%H:%M").time():
                forced_exit_price = next_candle['close']

        if result == "Neither Hit" and forced_exit_price:
            result = "Forced Exit"
            exit_price = forced_exit_price

        breakout_records.append({
            'date': candle_time,
            'type': breakout_type,
            'entry': round(entry, 2),
            'sl': round(sl, 2),
            'target': round(target, 2),
            'exit': round(exit_price, 2) if exit_price else None,
            'result': result
        })

        logging.info(f"{candle_time} - Exit: {exit_price}, Result: {result}")
        breakout_detected = True
        break

# Save to CSV
summary_df = pd.DataFrame(breakout_records)
summary_df.to_csv("breakout_trades_5min.csv", index=False)
logging.info("Breakout trades saved to breakout_trades_5min.csv")

# Calculate P&L
summary_df['pnl'] = summary_df.apply(
    lambda row: (row['exit'] - row['entry']) if row['type'] == 'LONG'
    else (row['entry'] - row['exit']) if row['type'] == 'SHORT'
    else 0,
    axis=1
)

# Print and log summary
total = len(summary_df)
wins = (summary_df['result'] == 'Target Hit').sum()
losses = (summary_df['result'] == 'Stop Loss Hit').sum()
forced = (summary_df['result'] == 'Forced Exit').sum()
net_pnl = summary_df['pnl'].sum()

summary_msg = f"""
📊 Trade Summary:
Total trades: {total}
Wins: {wins} | Losses: {losses} | Forced Exits: {forced}
Win rate: {wins / total * 100:.2f}%
Net P&L: {net_pnl:.2f} points
"""
logging.info(summary_msg)

# Plot latest trade
if not summary_df.empty:
    last_trade_date = pd.to_datetime(summary_df.iloc[-1]['date']).date()
    trade = summary_df.iloc[-1]
    day_data = df[df['only_date'] == last_trade_date]

    plt.figure(figsize=(12, 5))
    plt.plot(day_data['date'], day_data['close'], label='Close Price')
    plt.axhline(trade['entry'], color='blue', linestyle='--', label='Entry')
    plt.axhline(trade['target'], color='green', linestyle='--', label='Target')
    plt.axhline(trade['sl'], color='red', linestyle='--', label='Stop Loss')
    plt.title(f"{trade['date']} - {trade['type']} - {trade['result']}")
    plt.xlabel("Time")
    plt.ylabel("Price")
    plt.legend()
    plt.xticks(rotation=45)
    plt.tight_layout()
    plot_filename = f"breakout_plot_{last_trade_date}.png"
    plt.savefig(plot_filename)
    plt.show()
    logging.info(f"Breakout plot saved to {plot_filename}")

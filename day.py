import pandas as pd
from datetime import datetime
import os

from logger import setup_logger
import logging

setup_logger()

# Load daily data
df = pd.read_csv("NSEdata.csv", encoding='utf-8-sig')
df.columns = df.columns.str.strip().str.lower()
df['date'] = pd.to_datetime(df['date'], format="%d-%b-%y", errors='coerce')
df.sort_values('date', inplace=True)
df.reset_index(drop=True, inplace=True)

breakout_records = []

for i in range(1, len(df)):
    today = df.iloc[i]
    prev_day = df.iloc[i - 1]

    prev_high = prev_day['high']
    prev_low = prev_day['low']
    today_date = (
    today['date'].date()
    if isinstance(today['date'], pd.Timestamp) and pd.notnull(today['date'])
    else None
    )

    # === Long Breakout ===
    if today['open'] > prev_high:
        entry_price = today['open']
        sl = prev_low * 0.999
        risk = entry_price - sl
        target = entry_price + risk

        logging.info(f"{today_date} - Long Breakout Detected")
        logging.info(f"  Entry: {entry_price}, SL: {sl:.2f}, Target: {target:.2f}")

        result = "None"
        if today['low'] <= sl:
            result = "Stop Loss Hit"
        elif today['high'] >= target:
            result = "Target Hit"
        else:
            result = "Neither Hit"

        logging.info(f"  Result: {result}")

        breakout_records.append({
            'date': today_date,
            'type': 'LONG',
            'entry': entry_price,
            'sl': round(sl, 2),
            'target': round(target, 2),
            'result': result
        })

    # === Short Breakout ===
    elif today['open'] < prev_low:
        entry_price = today['open']
        sl = prev_high * 1.001  # buffer above high
        risk = sl - entry_price
        target = entry_price - risk

        logging.info(f"{today_date} - Short Breakout Detected")
        logging.info(f"  Entry: {entry_price}, SL: {sl:.2f}, Target: {target:.2f}")

        result = "None"
        if today['high'] >= sl:
            result = "Stop Loss Hit"
        elif today['low'] <= target:
            result = "Target Hit"
        else:
            result = "Neither Hit"

        logging.info(f"  Result: {result}")

        breakout_records.append({
            'date': today_date,
            'type': 'SHORT',
            'entry': entry_price,
            'sl': round(sl, 2),
            'target': round(target, 2),
            'result': result
        })

    else:
        logging.info(f"{today_date} - No Breakout Detected (Open: {today['open']})")

# Export results to CSV
if breakout_records:
    breakout_df = pd.DataFrame(breakout_records)
    breakout_df.to_csv("breakout_trades.csv", index=False)
    logging.info("Breakout trades exported to breakout_trades.csv")
else:
    logging.info("No breakout trades to export.")

if __name__ == "__main__":
    logging.info("Breakout analysis completed.")

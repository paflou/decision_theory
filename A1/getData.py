import json
import csv
from scipy.ndimage import gaussian_filter1d
import requests
from datetime import datetime



import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import seaborn as sns

# replace the "demo" apikey below with your own key from https://www.alphavantage.co/support/#api-key
url = 'https://www.alphavantage.co/query?function=TIME_SERIES_DAILY&symbol=QCOM&outputsize=full&apikey=ZWTQW11U676DWFU3'   # replace API KEY and stock symbol
r = requests.get(url)
data = r.json()
print(data)
# Extract "Time Series (Daily)" part
time_series = data.get("Time Series (Daily)", {})

# Prepare data for CSV
csv_data = [("Date", "Close")]  # Header for CSV file

# Filter data for dates after 2000 and extract the "close" price
for date, daily_data in time_series.items():
    try:
        date_obj = datetime.strptime(date, "%Y-%m-%d")
        if date_obj.year >= 2005:  # Filter for dates after 2005, as values before are wrong
            close_price = float(daily_data.get("4. close", 0))
            csv_data.append((date, close_price))
    except (ValueError, TypeError):  # Skip invalid dates or data
        continue


# Write data to CSV file
csv_file = 'close_prices.csv'
with open(csv_file, mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerows(csv_data)

print(f"Data saved to {csv_file}")

# src/config.py

import os

# === GENERAL SETTINGS ===
HISTORY_DAYS = 20  
COUNTRY_CODE = "US"

# === TARGET SETTINGS ===
#
# "price": predict next-day close directly.
# "delta": predict next-day change: close[t+1] - close[t].
#
# Delta targets often train more stably and avoid collapsing to a near-constant
# mean price; evaluation converts deltas back into prices using close[t].
TARGET_MODE = os.getenv("TARGET_MODE", "delta")

# === DATA COLLECTION SETTINGS ===
# NewsAPI free tier limitations
NEWS_DAYS_AVAILABLE = 30  # NewsAPI free tier only provides last 30 days
MAX_DAILY_REQUESTS = 100  # NewsAPI free tier daily limit
TRAINING_DATA_DAYS = 60   # Total days of stock data to fetch
# Default to 30 days because NewsAPI free tier typically supports up to the last 30 days.
NEWS_HISTORY_DAYS = int(os.getenv("NEWS_HISTORY_DAYS", "30"))

# === API KEYS ===
# NewsAPI.org key (load from environment; do not commit secrets)
NEWS_API_KEY = os.getenv("NEWS_API_KEY", "")

# === API ENDPOINTS ===
NEWSAPI_ENDPOINT = "https://newsapi.org/v2/everything"

# === OPTIONS ===
INCLUDE_GLOBAL_SENTIMENT = False  # Removed - yfinance (VIX) handles market sentiment

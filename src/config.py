# src/config.py

# === GENERAL SETTINGS ===
HISTORY_DAYS = 5  # Reduced from 20 to use less historical data
COUNTRY_CODE = "US"

# === DATA COLLECTION SETTINGS ===
# NewsAPI free tier limitations
NEWS_DAYS_AVAILABLE = 30  # NewsAPI free tier only provides last 30 days
MAX_DAILY_REQUESTS = 100  # NewsAPI free tier daily limit
TRAINING_DATA_DAYS = 60   # Total days of stock data to fetch
NEWS_HISTORY_DAYS = 20    # OPTIMIZED: Fetch 20 days of company news (3 weeks)

# === API KEYS ===
# NewsAPI.org key (you provided)
NEWS_API_KEY = "c5f10cd6942f4917a04c5a8d41119d80"

# === API ENDPOINTS ===
NEWSAPI_ENDPOINT = "https://newsapi.org/v2/everything"

# === OPTIONS ===
INCLUDE_GLOBAL_SENTIMENT = False  # Removed - yfinance (VIX) handles market sentiment

# Configuration settings

# NewsAPI Configuration
NEWS_API_KEY = 'YOUR_ACTUAL_API_KEY_HERE'

# Data Configuration
DEFAULT_TICKER = 'AAPL'
DATA_START_DATE = '2019-01-01'  # Already correct - 5 years
LOOKBACK_DAYS = 1825  # Change from 200 to 1825 (5 years)

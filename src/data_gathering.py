# src/data_gathering.py
import yfinance as yf
import pandas as pd
import numpy as np
import requests
import json
from datetime import datetime, timedelta
from pathlib import Path
from transformers import pipeline
from src.config import (
    HISTORY_DAYS,
    NEWS_API_KEY,
    NEWSAPI_ENDPOINT,
    INCLUDE_GLOBAL_SENTIMENT,
)


CACHE_DIR = Path("data/raw/news_cache")
CACHE_DIR.mkdir(parents=True, exist_ok=True)

# -----------------------
# Sentiment Analysis Setup
# -----------------------
_sentiment_pipeline = None

def _get_sentiment_pipeline():
    global _sentiment_pipeline
    if _sentiment_pipeline is None:
        _sentiment_pipeline = pipeline("sentiment-analysis")
    return _sentiment_pipeline

# -----------------------
# NewsAPI.org Fetching with cache
# -----------------------
def fetch_newsapi_headlines(query: str, from_date: str, to_date: str, page_size: int = 20):
    """
    Fetch headlines for a ticker or global topic using NewsAPI.org with caching.
    """
    cache_file = CACHE_DIR / f"{query}_{from_date}_{to_date}.json"
    if cache_file.exists():
        with open(cache_file, "r", encoding="utf-8") as f:
            return json.load(f)

    params = {
        "q": query,
        "from": from_date,
        "to": to_date,
        "language": "en",
        "pageSize": page_size,
        "sortBy": "publishedAt",
        "apiKey": NEWS_API_KEY,
    }
    resp = requests.get(NEWSAPI_ENDPOINT, params=params)
    j = resp.json()
    if j.get("status") != "ok":
        print(f"[WARN] NewsAPI error for {query}: {j}")
        headlines = []
    else:
        headlines = [a.get("title", "") for a in j.get("articles", [])]

    # Save cache
    with open(cache_file, "w", encoding="utf-8") as f:
        json.dump(headlines, f, ensure_ascii=False, indent=2)

    return headlines

def compute_sentiment_score(headlines):
    """
    Compute sentiment score in [-1, 1] using HuggingFace pipeline.
    """
    if not headlines:
        return 0.0
    pipe = _get_sentiment_pipeline()
    scores = []
    for h in headlines:
        try:
            result = pipe(h)[0]
            label = result["label"].lower()
            score = result["score"]
            if "pos" in label:
                scores.append(score)
            elif "neg" in label:
                scores.append(-score)
            else:
                scores.append(0.0)
        except Exception:
            scores.append(0.0)
    return float(np.mean(scores))

# -----------------------
# Main Data Gathering
# -----------------------
def gather_data(ticker: str):
    """
    Fetch stock + news + synthetic macro + market data for a given ticker.
    Returns: (X, y)
    """
    end = datetime.today()
    start = end - timedelta(days=365)

    df = yf.download(ticker, start=start, end=end, progress=False)
    if df.empty:
        raise ValueError(f"No data found for {ticker}")

    vix = yf.download("^VIX", start=start, end=end, progress=False)
    df["vix_index"] = vix["Close"].reindex(df.index).fillna(method="ffill")

    sentiments = []
    for dt in df.index:
        date_str = dt.strftime("%Y-%m-%d")
        company_news = fetch_newsapi_headlines(ticker, date_str, date_str)
        comp_score = compute_sentiment_score(company_news)

        if INCLUDE_GLOBAL_SENTIMENT:
            global_news = fetch_newsapi_headlines("global economy", date_str, date_str)
            global_score = compute_sentiment_score(global_news)
        else:
            global_score = 0.0

        sentiments.append((comp_score, global_score))

    df["sentiment_comp"] = [s[0] for s in sentiments]
    df["sentiment_global"] = [s[1] for s in sentiments]

    np.random.seed(42)
    df["interest_rate"] = 5.0 + np.random.normal(0, 0.1, len(df))
    df["inflation_rate"] = 2.5 + np.random.normal(0, 0.05, len(df))
    df["gdp_growth"] = 1.8 + np.random.normal(0, 0.03, len(df))

    X, y = [], []
    for i in range(HISTORY_DAYS, len(df) - 1):
        window = df.iloc[i - HISTORY_DAYS:i][["Open", "High", "Low", "Close", "Volume"]].values.flatten()
        sentiment_vec = np.array(df.iloc[i][["sentiment_comp", "sentiment_global"]], dtype=np.float32)
        macro_vec = np.array(df.iloc[i][["interest_rate", "inflation_rate", "gdp_growth"]], dtype=np.float32)
        market_vec = np.array([df.iloc[i]["vix_index"]], dtype=np.float32)

        X_i = np.concatenate([window, sentiment_vec, macro_vec, market_vec])
        y_i = np.float32(df.iloc[i + 1]["Close"])
        X.append(X_i)
        y.append(y_i)

    return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)


# -----------------------
# Quick Test
# -----------------------
if __name__ == "__main__":
    ticker = input("Enter stock ticker (e.g., AAPL, TSLA): ").strip().upper()
    X, y = gather_data(ticker)
    print(f"✅ Data gathered for {ticker}")
    print("Feature tensor shape:", X.shape)
    print("Target tensor shape:", y.shape)

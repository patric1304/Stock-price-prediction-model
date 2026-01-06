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
    NEWS_HISTORY_DAYS,
    TARGET_MODE,
)


class NewsAPIRateLimitError(RuntimeError):
    pass

# Cache directory
CACHE_DIR = Path("data/raw/news_cache")
CACHE_DIR.mkdir(parents=True, exist_ok=True)

# Huggingface sentiment pipeline
_sentiment_pipeline = None
def _get_sentiment_pipeline():
    global _sentiment_pipeline
    if _sentiment_pipeline is None:
        _sentiment_pipeline = pipeline(
            "sentiment-analysis",
            model="distilbert/distilbert-base-uncased-finetuned-sst-2-english",
            revision="714eb0f"
        )
    return _sentiment_pipeline

# Fetch news with caching
def fetch_newsapi_headlines(query: str, from_date: str, to_date: str, page_size: int = 20):
    if not NEWS_API_KEY:
        return []
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

    # Free-tier rate limiting typically returns HTTP 429 and/or code=rateLimited.
    if resp.status_code == 429 or j.get("code") == "rateLimited":
        raise NewsAPIRateLimitError(f"NewsAPI rate limit exceeded: {j}")

    if j.get("status") != "ok":
        print(f"[WARN] NewsAPI error for {query}: {j}")
        headlines = []
    else:
        headlines = [a.get("title", "") for a in j.get("articles", [])]
    with open(cache_file, "w", encoding="utf-8") as f:
        json.dump(headlines, f, ensure_ascii=False, indent=2)
    return headlines

def compute_sentiment_score(headlines):
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

# Gather stock + VIX + sentiment + macro features
def gather_data(ticker: str, days_back=60, return_meta: bool = False, target_mode: str | None = None):
    """
    Gather stock data with features (optimized for NewsAPI free tier).
    
    Args:
        ticker: Stock ticker symbol
        days_back: Number of days of stock price history (default: 60)
                   News: Only last 20 days (optimized for API limits)
    
    Returns:
        X, y: Feature matrix and target values
    """
    mode = (target_mode or TARGET_MODE or "price").strip().lower()
    if mode not in {"price", "delta"}:
        raise ValueError(f"Invalid target_mode={mode!r}. Expected 'price' or 'delta'.")

    end = datetime.today()
    start_stock = end - timedelta(days=days_back)
    start_news = end - timedelta(days=NEWS_HISTORY_DAYS)

    df = yf.download(ticker, start=start_stock, end=end, progress=False, auto_adjust=False)
    if df.empty:
        raise ValueError(f"No data found for {ticker}")
    vix = yf.download("^VIX", start=start_stock, end=end, progress=False, auto_adjust=False)
    df["vix_index"] = vix["Close"].reindex(df.index).ffill()

    sentiments = []
    for dt in df.index:
        # Shift news sentiment by 1 day to reduce leakage from after-hours articles.
        query_dt = dt - timedelta(days=1)
        date_str = query_dt.strftime("%Y-%m-%d")

        if query_dt < start_news:
            comp_score, global_score = 0.0, 0.0
        else:
            try:
                company_news = fetch_newsapi_headlines(ticker, date_str, date_str)
                comp_score = compute_sentiment_score(company_news)
            except NewsAPIRateLimitError:
                raise
            except Exception:
                comp_score = 0.0

            global_score = 0.0
            if INCLUDE_GLOBAL_SENTIMENT:
                try:
                    global_news = fetch_newsapi_headlines("global economy", date_str, date_str)
                    global_score = compute_sentiment_score(global_news)
                except NewsAPIRateLimitError:
                    raise
                except Exception:
                    global_score = 0.0

        sentiments.append((comp_score, global_score))
    df["sentiment_comp"] = [s[0] for s in sentiments]
    df["sentiment_global"] = [s[1] for s in sentiments]

    # Build dataset
    X, y = [], []
    meta = {
        "current_close": [],
        "target_date": [],
    }

    def _close_scalar(idx: int) -> float:
        close_val = df["Close"].iloc[idx]
        # With yfinance multi-index columns, this can be a Series (even for one ticker)
        if hasattr(close_val, "iloc"):
            close_val = close_val.iloc[0]
        return float(close_val)

    # Build samples for predicting next-day close (i+1) using information available at day i.
    # Include day i in the OHLCV window so the model has access to the current close, matching the naive baseline.
    for i in range(HISTORY_DAYS - 1, len(df) - 1):
        start = i - (HISTORY_DAYS - 1)
        window = df.iloc[start : i + 1][["Open", "High", "Low", "Close", "Volume"]].values.flatten()
        sentiment_vec = np.array(df.iloc[i][["sentiment_comp","sentiment_global"]], dtype=np.float32).flatten()
        vix_value = df["vix_index"].iloc[i]
        if pd.isna(vix_value):
            vix_value = 0.0
        market_vec = np.array([vix_value], dtype=np.float32).flatten()
        X_i = np.concatenate([window, sentiment_vec, market_vec])

        # Target options:
        # - price: next-day close
        # - delta: next-day change relative to today's close
        next_close = _close_scalar(i + 1)
        current_close = _close_scalar(i)
        y_val = next_close if mode == "price" else (next_close - current_close)

        # Always store target as shape (1,) so y becomes (N, 1) after np.array
        y_i = np.float32(y_val)
        X.append(X_i)
        y.append([y_i])

        if return_meta:
            meta["current_close"].append(current_close)
            meta["target_date"].append(df.index[i + 1].strftime("%Y-%m-%d"))

    X_arr = np.array(X, dtype=np.float32)
    y_arr = np.array(y, dtype=np.float32)  # (N, 1)

    if return_meta:
        meta["current_close"] = np.array(meta["current_close"], dtype=np.float32)
        meta["target_date"] = np.array(meta["target_date"], dtype=object)
        return X_arr, y_arr, meta
    return X_arr, y_arr

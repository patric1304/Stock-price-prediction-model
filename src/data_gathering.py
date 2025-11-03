import yfinance as yf
import pandas as pd
import numpy as np
import requests
import json
from datetime import datetime, timedelta
from pathlib import Path
from transformers import pipeline
from src.config import HISTORY_DAYS, NEWS_API_KEY, NEWSAPI_ENDPOINT, INCLUDE_GLOBAL_SENTIMENT

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
def gather_data(ticker: str):
    end = datetime.today()
    start_stock = end - timedelta(days=365)
    NEWS_DAYS = 30
    start_news = end - timedelta(days=NEWS_DAYS)

    df = yf.download(ticker, start=start_stock, end=end, progress=False, auto_adjust=False)
    if df.empty:
        raise ValueError(f"No data found for {ticker}")
    vix = yf.download("^VIX", start=start_stock, end=end, progress=False, auto_adjust=False)
    df["vix_index"] = vix["Close"].reindex(df.index).ffill()

    sentiments = []
    for dt in df.index:
        date_str = dt.strftime("%Y-%m-%d")
        if dt < start_news:
            comp_score, global_score = 0.0, 0.0
        else:
            try:
                company_news = fetch_newsapi_headlines(ticker, date_str, date_str)
                comp_score = compute_sentiment_score(company_news)
            except:
                comp_score = 0.0
            global_score = 0.0
            if INCLUDE_GLOBAL_SENTIMENT:
                try:
                    global_news = fetch_newsapi_headlines("global economy", date_str, date_str)
                    global_score = compute_sentiment_score(global_news)
                except:
                    global_score = 0.0
        sentiments.append((comp_score, global_score))
    df["sentiment_comp"] = [s[0] for s in sentiments]
    df["sentiment_global"] = [s[1] for s in sentiments]

    # Synthetic macro features
    np.random.seed(42)
    df["interest_rate"] = 5.0 + np.random.normal(0, 0.1, len(df))
    df["inflation_rate"] = 2.5 + np.random.normal(0, 0.05, len(df))
    df["gdp_growth"] = 1.8 + np.random.normal(0, 0.03, len(df))

    # Build dataset
    X, y = [], []
    for i in range(HISTORY_DAYS, len(df)-1):
        window = df.iloc[i-HISTORY_DAYS:i][["Open","High","Low","Close","Volume"]].values.flatten()
        sentiment_vec = np.array(df.iloc[i][["sentiment_comp","sentiment_global"]], dtype=np.float32).flatten()
        macro_vec = np.array(df.iloc[i][["interest_rate","inflation_rate","gdp_growth"]], dtype=np.float32).flatten()
        vix_value = df.iloc[i]["vix_index"]
        if pd.isna(vix_value):
            vix_value = 0.0
        market_vec = np.array([vix_value], dtype=np.float32).flatten()
        X_i = np.concatenate([window, sentiment_vec, macro_vec, market_vec])
        y_i = np.float32(df.iloc[i+1]["Close"])
        X.append(X_i)
        y.append(y_i)
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)

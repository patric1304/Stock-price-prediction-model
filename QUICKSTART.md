# Quick Start Guide

## Setup (5 minutes)

1. **Install dependencies:**
```bash
pip install -r requirements.txt
```

2. **Configure your NewsAPI key** in `src/config.py`

## Important: NewsAPI Free Tier

⚠️ Your free NewsAPI account has limitations:
- **100 requests per day**
- **Last 30 days of news only**

Don't worry! The project is optimized for this:
- ✅ News is automatically cached
- ✅ Training uses only 3-4 stocks by default
- ✅ After first run, uses minimal API calls

## Train Your First Model (15-30 minutes)

```bash
# First time: Full training
python scripts/train_continuous.py
```

This will use **~80 API requests** (within your 100/day limit)

This will:
- ✅ Gather data from 10 major stocks
- ✅ Train for 100 epochs
- ✅ Save the model automatically
- ✅ Create training logs

**Expected output:**
```
🚀 CONTINUOUS MULTI-STOCK TRAINING
⚠️  NewsAPI Limitations:
  • Free tier: 100 requests/day
  • News available: Last 30 days only
  • Training on 4 stocks to stay within limits
  • Using cache when available to save requests

📊 Gathering data from multiple companies...
  Fetching AAPL... ✓ 42 samples
  Fetching MSFT... ✓ 42 samples
  Fetching GOOGL... ✓ 42 samples
  Fetching TSLA... ✓ 42 samples
🔥 Training model for 100 epochs...
Epoch [10/100], Loss: 0.234567
...
✅ Training Complete!
  • Mean Absolute Error: $5.23
  • MAPE: 2.45%
💾 Model saved
```

## Daily Updates (5 minutes) ⭐ RECOMMENDED

After initial training, use the optimized daily script:

```bash
python scripts/train_daily.py
```

This uses only **~5-10 API requests** because:
- ✅ Most news is cached from previous days
- ✅ Only fetches TODAY's news
- ✅ Faster training (50 epochs)
- ✅ Perfect for daily updates

## Make Your First Prediction

```bash
python scripts/run_inference.py
```

**Example:**
```
Enter stock ticker to predict (e.g., TSLA): AAPL

🔮 Prediction for AAPL:
  • Current price: $178.32
  • Predicted next-day price: $179.85
  • Expected change: $1.53 (+0.86%)

📈 Prediction: BULLISH (Price expected to rise)
```

## Check Model Accuracy

```bash
python scripts/evaluate_model.py
```

**Example:**
```
Enter stock ticker to evaluate (e.g., AAPL): TSLA

📈 Evaluation Results for TSLA:
  • Samples tested: 228
  • Mean Absolute Error: $4.23
  • MAPE: 1.85%
  • RMSE: $5.67

🔍 Sample Predictions (last 5 days):
Day         Actual  Predicted      Error
---------------------------------------------
Day 224    $245.67   $247.23     $1.56
Day 225    $248.12   $246.89    -$1.23
...
```

## Next Steps

### Improve Accuracy
Run more training with higher epochs:
```python
# Edit scripts/train_continuous.py
EPOCHS = 200  # or 500, 1000
```

### Add More Stocks
```python
# Edit scripts/train_continuous.py
TICKERS = ["AAPL", "MSFT", "GOOGL", "TSLA", "AMZN", 
           "META", "NVDA", "NFLX", "AMD", "INTC",
           "ADBE", "CRM", "ORCL", "IBM", "CSCO"]
```

### Regular Updates
Set up weekly training to keep model current:
```bash
# Windows Task Scheduler or cron job
python scripts/train_continuous.py
```

## File Locations

- **Models**: `data/processed/models/`
- **Logs**: `data/processed/training_logs/`
- **Checkpoints**: `data/checkpoints/`
- **Cached data**: `data/raw/news_cache/`

## Common Issues

**Q: NewsAPI rate limit?**  
A: Cache is used automatically. Wait 24h or upgrade NewsAPI.

**Q: Model accuracy low?**  
A: Increase EPOCHS to 200-500 and train on more stocks.

**Q: Training takes too long?**  
A: Reduce stocks or epochs for faster training.

## Workflow Summary

```
Day 1:  python scripts/train_continuous.py  (Full training, ~80 API requests)
Day 2+: python scripts/train_daily.py       (Quick update, ~5 API requests)
Any:    python scripts/evaluate_model.py    (Check accuracy, no requests)
Any:    python scripts/run_inference.py     (Predict, ~0-2 requests)
```

## Understanding API Usage

- **First run**: Fetches 30 days of news for each stock (~80 requests total)
- **Subsequent runs**: Only new news (~5 requests)
- **Cache**: Stored in `data/raw/news_cache/`
- **Limit reset**: Every 24 hours

See `NEWSAPI_STRATEGY.md` for detailed information.

Happy predicting! 🚀📈

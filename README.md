# Stock Price Prediction Model

A machine learning model that predicts tomorrow's stock prices using historical backtesting on S&P 500 stocks.

## 🎯 How It Works

### Training Phase (Supervised Learning with Backtesting)
1. **Fetch historical data** from last 30 days for S&P 500 stocks
2. **For each historical day**: 
   - Use that day's features (price, volume, sentiment, VIX, etc.)
   - Predict the NEXT day's price
   - Compare with actual next-day price (we know it from history!)
3. **Learn from errors** across thousands of predictions
4. **Result**: Model learns patterns that predict tomorrow's price

### Prediction Phase (End User)
1. User enters a stock ticker (e.g., "AAPL")
2. Model fetches TODAY's data
3. Predicts TOMORROW's closing price
4. Shows expected change and direction (bullish/bearish)

## Why This Approach Works

✅ **Trains on known outcomes** - We use historical data where we already know what happened  
✅ **Tests predictions against reality** - Model learns from actual prediction errors  
✅ **Learns from many stocks** - Trains on 30+ S&P 500 stocks for diverse patterns  
✅ **Realistic evaluation** - Exactly mimics how it will be used in production

## Features

- **Multi-company training**: Train on multiple stocks for better generalization
- **Continuous learning**: Incremental training to improve accuracy over time
- **Sentiment analysis**: Uses news sentiment from NewsAPI (optimized for free tier)
- **Market indicators**: Incorporates VIX index and synthetic macro features
- **Checkpointing**: Save training progress at regular intervals
- **Smart caching**: Reduces API calls by caching news data

## ⚠️ NewsAPI Free Tier Limitations

This project is optimized for NewsAPI free tier:
- **100 requests per day** - Training scripts are configured to stay within this limit
- **30 days of news** - Only last 30 days of news available
- **Caching enabled** - News is cached locally to minimize API calls
- **Optimized stock selection** - Default training uses 3-4 stocks to avoid hitting limits

### How We Handle Limitations

1. **First run**: Fetches news for all stocks (~60-80 requests)
2. **Subsequent runs**: Only fetches new/missing news (~5-10 requests)
3. **Older data**: Uses neutral sentiment (0.0) for dates beyond 30 days
4. **Training window**: Default 60 days (30 with news, 30 without)

## Project Structure

```
stock_price_prediction/
├── src/
│   ├── data_gathering.py      # Data collection and feature engineering
│   ├── preprocessing.py        # Data scaling and normalization
│   ├── model.py                # Neural network architecture
│   ├── train.py                # Training utilities
│   ├── evaluation.py           # Model evaluation metrics
│   └── config.py               # Configuration settings
│
├── scripts/
│   ├── train_continuous.py    # Multi-stock continuous training (MAIN)
│   ├── evaluate_model.py      # Test model accuracy
│   ├── run_inference.py       # Make predictions
│   └── run_training.py        # Single-stock training (legacy)
│
├── data/
│   ├── raw/news_cache/        # Cached news articles
│   ├── processed/
│   │   ├── models/            # Trained models (timestamped)
│   │   ├── training_logs/     # Training history logs
│   │   └── scalers/           # Saved data scalers
│   └── checkpoints/           # Training checkpoints
│
└── notebooks/                  # Jupyter notebooks for analysis
```

## Installation

1. Clone the repository:
```bash
git clone https://github.com/patric1304/Stock-price-prediction-model.git
cd stock_price_prediction
```

2. Create and activate virtual environment:
```bash
python -m venv venv
venv\Scripts\activate  # Windows
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Set up your NewsAPI key in `src/config.py`

## Usage

### 1. Train the Model (Multi-Company)

Train on multiple stocks for better accuracy:

```bash
# FIRST TIME: Full training on 4 stocks (uses ~80 API requests)
python scripts/train_continuous.py

# DAILY UPDATES: Quick training on 3 stocks (uses ~5 API requests)
python scripts/train_daily.py
```

**train_continuous.py:**
- Trains on 4 major stocks (AAPL, MSFT, GOOGL, TSLA)
- 100 epochs
- Best for initial training or weekly updates
- Uses ~60-80 API requests (first run), then cache

**train_daily.py:** ⭐ RECOMMENDED FOR DAILY USE
- Trains on 3 stocks (AAPL, TSLA, GOOGL)
- 50 epochs (faster)
- Optimized to use only ~5-10 new API requests
- Perfect for daily incremental updates

**Output:**
- `data/processed/models/multi_stock_model_YYYYMMDD_HHMMSS.pth`
- `data/processed/scalers/scaler_X.pkl` and `scaler_y.pkl`
- `data/processed/training_logs/training_YYYYMMDD_HHMMSS.txt`

### 2. Evaluate Model Accuracy

Test the model on historical data:

```bash
python scripts/evaluate_model.py
```

Enter a stock ticker (e.g., AAPL) to see:
- Mean Absolute Error (MAE)
- Mean Absolute Percentage Error (MAPE)
- Root Mean Squared Error (RMSE)
- Sample predictions vs actual prices

### 3. Make Predictions

Predict next-day stock price:

```bash
python scripts/run_inference.py
```

Enter a stock ticker to get:
- Current price
- Predicted next-day price
- Expected change ($ and %)
- Bullish/Bearish indicator

## Improving Model Accuracy

The model improves with more training data and epochs:

1. **Run daily training** (uses minimal API requests):
   ```bash
   python scripts/train_daily.py
   ```

2. **Increase training epochs** for better accuracy:
   ```python
   # Edit scripts/train_daily.py
   EPOCHS = 100  # or 200
   ```

3. **Add more stocks** (be mindful of API limits):
   ```python
   # Edit scripts/train_continuous.py
   # Each additional stock uses ~30-40 requests on first run
   TICKERS = ["AAPL", "MSFT", "GOOGL", "TSLA", "AMZN"]  # 5 stocks max recommended
   ```

4. **Let cache build up** - After first run, subsequent runs use very few API requests

## Model Architecture

- **Input**: Historical prices (OHLCV), sentiment scores, VIX index, macro indicators
- **Architecture**: 3-layer feedforward neural network
  - Layer 1: input_dim → 128 neurons (ReLU)
  - Layer 2: 128 → 64 neurons (ReLU)
  - Layer 3: 64 → 1 neuron (output)
- **Loss**: Mean Squared Error (MSE)
- **Optimizer**: Adam (lr=0.001)

## Configuration

Edit `src/config.py` to customize:

```python
HISTORY_DAYS = 5           # Days of historical data for each sample
NEWS_API_KEY = "your_key"  # Your NewsAPI key
INCLUDE_GLOBAL_SENTIMENT = True  # Include global economy sentiment
```

## Data Sources

- **Stock prices**: Yahoo Finance (via yfinance)
- **Market volatility**: VIX Index
- **News sentiment**: NewsAPI
- **Macro indicators**: Synthetic (can be replaced with real data)

## Model Versioning

Models are automatically timestamped:
- `multi_stock_model_20251106_143022.pth`
- The latest model is automatically used for inference
- Compare different versions using `evaluate_model.py`

## Tips

- **First time**: Train with 100 epochs to get baseline accuracy
- **Regular updates**: Retrain weekly to incorporate new market data
- **Testing**: Use `evaluate_model.py` on different stocks to test generalization
- **Production**: Increase epochs to 500+ for production use

## Troubleshooting

**NewsAPI Rate Limit Hit**:
- Cache is automatically used when available
- Wait 24 hours for limit reset
- Use `train_daily.py` instead of `train_continuous.py` (uses fewer requests)
- Reduce number of stocks in TICKERS list

**NewsAPI "rateLimited" Error**:
- This is normal if you see it a few times - cache will be used
- If you see it constantly, you've hit the daily limit
- Solution: Wait until tomorrow, cache will prevent new requests

**"No news available for date"**:
- NewsAPI free tier only provides last 30 days
- Older dates automatically use neutral sentiment (0.0)
- This is expected and handled by the code

**Memory Issues**: Reduce number of stocks or batch size in training.

**Poor Accuracy**: 
- Increase epochs (100 → 200)
- Train daily for incremental improvements
- Let model train on more diverse market conditions over time

## Contributing

1. Create a feature branch: `git checkout -b feature/your-feature`
2. Commit changes: `git commit -m "Add feature"`
3. Push to branch: `git push origin feature/your-feature`
4. Create Pull Request

## License

MIT License

## Author

patric1304

---

**Note**: This model is for educational purposes. Do not use for actual trading without proper validation and risk management.

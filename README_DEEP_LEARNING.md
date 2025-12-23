# 🚀 Advanced Deep Learning Stock Price Prediction

A sophisticated stock price prediction system using state-of-the-art deep learning techniques, optimized for Google Colab and production environments.

## 🎯 Key Improvements Over Simple Model

### Architecture Enhancements
- **LSTM Networks**: Captures temporal dependencies in time-series data
- **Multi-Head Attention**: Identifies important features dynamically
- **Residual Connections**: Enables training of deeper networks
- **Layer Normalization**: Stabilizes training and improves convergence

### Regularization Techniques
- ✅ **Dropout** (0.3 rate) - Prevents overfitting
- ✅ **Early Stopping** (patience=15) - Stops when validation stops improving
- ✅ **L2 Weight Decay** (1e-5) - Constrains model complexity
- ✅ **Gradient Clipping** - Prevents exploding gradients
- ✅ **Batch Normalization** - Normalizes layer inputs

### Data Management
- ✅ **Proper Train/Val/Test Split** (70/15/15) - No data leakage
- ✅ **Scaler Fitting on Training Only** - Prevents information leak
- ✅ **Temporal Ordering Preserved** - Respects time-series nature

### Training Optimizations
- ✅ **AdamW Optimizer** - Better convergence than standard SGD
- ✅ **Learning Rate Scheduling** - Adaptive learning rate
- ✅ **GPU Acceleration** - 10-50x faster training
- ✅ **Checkpointing** - Save best models automatically

---

## 🏗️ Model Architecture

```
Input (31 features)
    ↓
Input Projection (256D)
    ↓
Bidirectional LSTM (3 layers, 256D → 512D)
    ↓
Multi-Head Attention (8 heads)
    ↓
Residual Blocks (3 blocks)
    ↓
Output Network (512D → 256D → 128D → 64D → 1)
    ↓
Price Prediction
```

**Total Parameters**: ~2.5M (complex enough to learn patterns, regularized to prevent overfitting)

---

## 📊 Features Used

### Price Data (5 days × 5 features = 25)
- Open, High, Low, Close, Volume

### Sentiment Features (2)
- Company-specific news sentiment
- Market sentiment (VIX index)

### Macroeconomic Indicators (4)
- Interest rates
- Inflation rate
- GDP growth
- Market volatility (VIX)

**Total**: 31 input features

---

## 🚀 Quick Start Guide

### Option 1: Google Colab (Recommended)

1. **Open the Colab Notebook**
   ```
   notebooks/colab_deep_learning_pipeline.ipynb
   ```

2. **Upload to Google Colab**
   - Go to [Google Colab](https://colab.research.google.com/)
   - File → Upload Notebook
   - Select the `.ipynb` file

3. **Enable GPU**
   - Runtime → Change runtime type
   - Hardware accelerator: GPU (T4 or better)
   - Save

4. **Run All Cells**
   - Runtime → Run all
   - The notebook will:
     - Install dependencies
     - Load all modules inline
     - Gather data
     - Train the model
     - Evaluate performance
     - Visualize results

### Option 2: Local Training

#### Installation

```bash
# Clone repository
git clone <your-repo-url>
cd stock_price_prediction

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

#### Training

```bash
# Train with default settings (AAPL stock)
python scripts/train_advanced_model.py

# Train with custom settings
python scripts/train_advanced_model.py \
    --ticker TSLA \
    --days 300 \
    --epochs 200 \
    --batch-size 64 \
    --hidden-dim 256 \
    --lr 0.001 \
    --patience 15
```

#### Evaluation

```bash
# Evaluate trained model
python scripts/evaluate_advanced_model.py \
    --ticker AAPL \
    --model data/checkpoints/best_model.pth \
    --output data/evaluation_results.png
```

---

## 🎛️ Hyperparameters

### Model Configuration

| Parameter | Default | Description |
|-----------|---------|-------------|
| `hidden_dim` | 256 | Size of hidden layers |
| `num_layers` | 3 | Number of LSTM layers |
| `dropout` | 0.3 | Dropout probability |

### Training Configuration

| Parameter | Default | Description |
|-----------|---------|-------------|
| `epochs` | 200 | Maximum training epochs |
| `batch_size` | 64 | Batch size for training |
| `lr` | 1e-3 | Initial learning rate |
| `patience` | 15 | Early stopping patience |
| `train_split` | 0.7 | Training data proportion |
| `val_split` | 0.15 | Validation data proportion |

### Recommended Settings by Dataset Size

**Small Dataset (<100 samples)**
```bash
--hidden-dim 128 --num-layers 2 --dropout 0.4 --batch-size 32
```

**Medium Dataset (100-500 samples)**
```bash
--hidden-dim 256 --num-layers 3 --dropout 0.3 --batch-size 64
```

**Large Dataset (>500 samples)**
```bash
--hidden-dim 512 --num-layers 4 --dropout 0.2 --batch-size 128
```

---

## 📈 Performance Metrics

The model reports the following metrics:

- **MSE** (Mean Squared Error): Lower is better
- **MAE** (Mean Absolute Error): Average price deviation in $
- **RMSE** (Root Mean Squared Error): Standard deviation of errors
- **R²** (R-squared): Proportion of variance explained (0-1)
- **MAPE** (Mean Absolute Percentage Error): Average error in %
- **Directional Accuracy**: % of correct price movement predictions

### Example Results

```
Test Set Performance
==================================================
MSE:  2.45
MAE:  1.23
RMSE: 1.57
MAPE: 0.85%
Directional Accuracy: 67.4%
==================================================
```

---

## 🔧 Project Structure

```
stock_price_prediction/
│
├── notebooks/
│   └── colab_deep_learning_pipeline.ipynb   # All-in-one Colab notebook
│
├── src/
│   ├── model.py                 # AdvancedStockPredictor architecture
│   ├── train.py                 # Training pipeline with validation
│   ├── dataset.py               # PyTorch dataset
│   ├── preprocessing.py         # Data scaling
│   ├── data_gathering.py        # Stock & sentiment data collection
│   └── config.py               # Configuration settings
│
├── scripts/
│   ├── train_advanced_model.py        # Training script
│   └── evaluate_advanced_model.py     # Evaluation script
│
├── data/
│   ├── checkpoints/            # Saved models
│   │   ├── best_model.pth
│   │   ├── scaler_X.pkl
│   │   └── scaler_y.pkl
│   └── raw/news_cache/         # Cached news data
│
├── requirements.txt            # Python dependencies
└── README_DEEP_LEARNING.md    # This file
```

---

## 🧪 Preventing Overfitting

### Built-in Regularization

1. **Dropout Layers** (30% dropout rate)
   - Randomly drops neurons during training
   - Forces network to learn robust features

2. **Early Stopping** (patience=15)
   - Monitors validation loss
   - Stops training when no improvement

3. **L2 Weight Decay** (1e-5)
   - Penalizes large weights
   - Encourages simpler models

4. **Learning Rate Scheduling**
   - Reduces LR when validation plateaus
   - Helps find better local minima

5. **Proper Data Split**
   - Separate train/val/test sets
   - Scalers fitted on training only

### Signs of Overfitting

Watch for these indicators:
- Training loss << Validation loss
- Validation loss increases while training decreases
- Perfect training accuracy but poor validation

### Solutions

If overfitting occurs:
```bash
# Increase dropout
--dropout 0.4

# Reduce model complexity
--hidden-dim 128 --num-layers 2

# More regularization
# (Modify train.py: increase weight_decay)

# Gather more data
--days 500
```

---

## 💡 Tips for Best Results

### 1. Data Quality
- Use stocks with consistent trading volume
- Gather at least 200 days of data
- Avoid stocks with major structural changes

### 2. GPU Usage
- Always use GPU on Colab (free T4)
- Training time: 5-10 minutes with GPU vs 1-2 hours CPU

### 3. Hyperparameter Tuning
- Start with defaults
- Adjust based on validation performance
- Use learning curves to diagnose issues

### 4. Evaluation
- Never evaluate on training data
- Use test set only once (final evaluation)
- Focus on validation metrics during development

### 5. Deployment
- Save both model and scalers
- Version your models
- Monitor prediction errors in production

---

## 🐛 Troubleshooting

### Issue: Training is too slow
**Solution**: Enable GPU in Colab or use `--no-gpu` flag to verify it's not a GPU issue

### Issue: Out of memory
**Solution**: Reduce batch size: `--batch-size 32`

### Issue: Model not learning (loss not decreasing)
**Solutions**:
- Check data quality
- Reduce learning rate: `--lr 1e-4`
- Increase model capacity: `--hidden-dim 512`

### Issue: Overfitting (val loss > train loss)
**Solutions**:
- Increase dropout: `--dropout 0.4`
- Reduce model size: `--hidden-dim 128`
- Gather more data: `--days 500`

### Issue: NewsAPI errors
**Solutions**:
- Check API key in `config.py`
- Verify internet connection
- Use cached data (already downloaded)

---

## 📚 Additional Resources

### Learning Materials
- [Understanding LSTMs](https://colah.github.io/posts/2015-08-Understanding-LSTMs/)
- [Attention Mechanisms](https://arxiv.org/abs/1706.03762)
- [Regularization in Deep Learning](https://www.deeplearningbook.org/contents/regularization.html)

### PyTorch Documentation
- [LSTM](https://pytorch.org/docs/stable/generated/torch.nn.LSTM.html)
- [MultiheadAttention](https://pytorch.org/docs/stable/generated/torch.nn.MultiheadAttention.html)
- [Training Best Practices](https://pytorch.org/tutorials/beginner/deep_learning_60min_blitz.html)

---

## 🔄 Next Steps

1. **Experiment with Different Stocks**
   ```bash
   python scripts/train_advanced_model.py --ticker TSLA
   python scripts/train_advanced_model.py --ticker GOOGL
   ```

2. **Hyperparameter Tuning**
   - Try different hidden dimensions
   - Experiment with LSTM layers
   - Adjust dropout rates

3. **Feature Engineering**
   - Add technical indicators (RSI, MACD)
   - Include sector-specific news
   - Add company fundamentals

4. **Multi-Step Prediction**
   - Modify model to predict multiple days ahead
   - Implement sequence-to-sequence architecture

5. **Ensemble Methods**
   - Train multiple models with different seeds
   - Average predictions for robustness

---

## 📄 License

This project is for educational purposes. Always do your own research before making investment decisions.

---

## 🙏 Acknowledgments

- **PyTorch Team** - Deep learning framework
- **Hugging Face** - Sentiment analysis models
- **yfinance** - Stock data API
- **NewsAPI** - News data API

---

## 📞 Support

For issues or questions:
1. Check the troubleshooting section
2. Review the Colab notebook examples
3. Examine training logs in `data/checkpoints/`

**Happy Trading! 📈🚀**

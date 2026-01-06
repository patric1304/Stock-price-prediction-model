# Stock Price Prediction - Deep Learning System

A production-ready stock price prediction system using LSTM networks with attention mechanism and comprehensive regularization techniques.

## System Overview

This system implements an advanced deep learning architecture for stock price prediction with:
- LSTM layers for temporal pattern recognition
- Multi-head attention mechanism for feature importance
- Residual connections for gradient flow
- Comprehensive regularization (Dropout, Early Stopping, L2, Gradient Clipping)
- Proper train/validation/test split (70/15/15)
- GPU acceleration support

## Project Structure

```
stock_price_prediction/
├── src/
│   ├── model.py              # AdvancedStockPredictor architecture
│   ├── train.py              # Training pipeline with validation
│   ├── dataset.py            # PyTorch dataset wrapper
│   ├── preprocessing.py      # Data scaling utilities
│   ├── data_gathering.py     # Stock data collection
│   └── config.py            # Configuration settings
│
├── scripts/
│   ├── train_advanced_model.py      # Training script
│   └── evaluate_advanced_model.py   # Evaluation script
│
├── notebooks/
│   └── colab_deep_learning_pipeline.ipynb  # Google Colab notebook
│
├── data/
│   ├── checkpoints/         # Saved models and scalers
│   └── raw/news_cache/      # Cached news data
│
└── requirements.txt         # Python dependencies
```

## Installation

### Prerequisites
- Python 3.8 or higher
- CUDA-capable GPU (optional, but recommended)
- 4GB+ RAM

### Setup

```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# Windows:
venv\Scripts\activate
# Linux/Mac:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

## Configuration

Set your NewsAPI key via environment variable (do not commit secrets):

- Windows (PowerShell): `setx NEWS_API_KEY "your_key_here"`
- macOS/Linux (bash/zsh): `export NEWS_API_KEY="your_key_here"`

Get a free key at: https://newsapi.org/

## Operations Guide

### 1. Training Phase

Train a new model from scratch:

```bash
# Basic training with default parameters
python scripts/train_advanced_model.py --ticker AAPL

# Custom training configuration
python scripts/train_advanced_model.py \
    --ticker AAPL \
    --days 200 \
    --epochs 200 \
    --batch-size 64 \
    --hidden-dim 256 \
    --lr 0.001 \
    --patience 15
```

#### Training Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--ticker` | AAPL | Stock ticker symbol |
| `--days` | 200 | Days of historical data |
| `--epochs` | 200 | Maximum training epochs |
| `--batch-size` | 64 | Batch size for training |
| `--hidden-dim` | 256 | LSTM hidden dimension |
| `--num-layers` | 3 | Number of LSTM layers |
| `--dropout` | 0.3 | Dropout probability |
| `--lr` | 0.001 | Learning rate |
| `--patience` | 15 | Early stopping patience |
| `--train-split` | 0.7 | Training data proportion |
| `--val-split` | 0.15 | Validation data proportion |
| `--no-gpu` | False | Disable GPU usage |
| `--checkpoint-dir` | data/checkpoints | Save directory |

#### Training Output

The training process will:
1. Download historical stock data
2. Split data into train/validation/test sets (70/15/15)
3. Train the model with early stopping
4. Save best model checkpoint
5. Generate training report with metrics

Saved artifacts in `data/checkpoints/`:
Saved artifacts in `data/checkpoints/<TICKER>/`:
- `best_model.pth` - Trained model weights
- `scaler_X.pkl` - Feature scaler
- `scaler_y.pkl` - Target scaler
- `training_history.json` - Loss curves
- `training_summary.json` - Configuration and metrics
- `training_report.json` - Comprehensive report

### 2. Evaluation Phase

Evaluate a trained model:

```bash
# Basic evaluation
python scripts/evaluate_advanced_model.py --ticker AAPL

# Specify model path
python scripts/evaluate_advanced_model.py \
    --ticker AAPL \
    --model data/checkpoints/AAPL/best_model.pth \
    --output data/evaluation_results.png
```

#### Evaluation Output

The evaluation generates:
- 9 comprehensive visualizations (saved as PNG)
- Performance metrics (MSE, MAE, RMSE, R², MAPE, Directional Accuracy)
- `evaluation_metrics.json` - Detailed metrics

### 3. Inference (Prediction)

To make predictions on new data, use the trained model:

```python
import torch
import pickle
from src.model import AdvancedStockPredictor
from src.data_gathering import gather_data

# Load model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = AdvancedStockPredictor(input_dim=31)
checkpoint = torch.load('data/checkpoints/best_model.pth', map_location=device)
model.load_state_dict(checkpoint['model_state_dict'])
model.to(device)
model.eval()

# Load scalers
with open('data/checkpoints/scaler_X.pkl', 'rb') as f:
    scaler_X = pickle.load(f)
with open('data/checkpoints/scaler_y.pkl', 'rb') as f:
    scaler_y = pickle.load(f)

# Gather latest data
X, y = gather_data('AAPL', days_back=60)
X_scaled = scaler_X.transform(X[-1:])

# Predict
with torch.no_grad():
    X_tensor = torch.tensor(X_scaled, dtype=torch.float32).to(device)
    y_pred_scaled = model(X_tensor)
    y_pred = scaler_y.inverse_transform(y_pred_scaled.cpu().numpy())

print(f"Predicted next day price: ${y_pred[0,0]:.2f}")
```

## Google Colab Deployment

For quick training without local setup:

1. Open `notebooks/colab_deep_learning_pipeline.ipynb`
2. Upload to Google Colab
3. Enable GPU: Runtime → Change runtime type → GPU
4. Run all cells

The notebook includes:
- Automatic dependency installation
- Complete training pipeline
- Evaluation and visualization
- Model download functionality

## Model Architecture

```
Input (31 features)
    ↓
Input Projection (Linear + LayerNorm + ReLU + Dropout)
    ↓
Bidirectional LSTM (3 layers, 256 hidden → 512 output)
    ↓
Multi-Head Attention (8 heads)
    ↓
Residual Blocks (3 blocks with skip connections)
    ↓
Output Network (512 → 256 → 128 → 64 → 1)
    ↓
Price Prediction
```

Total Parameters: ~2.5M

## Features Used

### Price Data (25 features)
- 5 days of OHLCV (Open, High, Low, Close, Volume)

### Sentiment Features (2 features)
- Company-specific news sentiment
- Market sentiment proxy

### Market Volatility (1 feature)
- VIX volatility index

Total: 28 input features

## Performance Metrics

The model reports:
- **MSE** (Mean Squared Error)
- **MAE** (Mean Absolute Error) - Average price deviation
- **RMSE** (Root Mean Squared Error)
- **R²** (R-squared) - Variance explained (0-1)
- **MAPE** (Mean Absolute Percentage Error)
- **Directional Accuracy** - Percentage of correct movement predictions

Expected performance on test set:
- RMSE: $1.5-2.5
- MAPE: 0.8-1.5%
- Directional Accuracy: 60-70%

## Regularization Techniques

1. **Dropout** (0.3) - Prevents overfitting
2. **Early Stopping** (patience=15) - Stops when validation plateaus
3. **L2 Weight Decay** (1e-5) - Constrains model complexity
4. **Gradient Clipping** (max_norm=1.0) - Prevents exploding gradients
5. **Layer Normalization** - Stabilizes training
6. **Learning Rate Scheduling** - ReduceLROnPlateau

## Data Management

### Train/Validation/Test Split
- Training: 70% of data
- Validation: 15% of data
- Test: 15% of data

### Important Notes
- Scalers are fitted ONLY on training data (prevents data leakage)
- Temporal order is preserved (no random shuffling across sets)
- Validation set guides training (early stopping, LR scheduling)
- Test set used only once for final evaluation

## Troubleshooting

### Issue: CUDA out of memory
**Solution**: Reduce batch size
```bash
python scripts/train_advanced_model.py --batch-size 32
```

### Issue: Training too slow
**Solution**: Use GPU or reduce model size
```bash
python scripts/train_advanced_model.py --hidden-dim 128 --num-layers 2
```

### Issue: Overfitting (validation loss > training loss)
**Solution**: Increase regularization
```bash
python scripts/train_advanced_model.py --dropout 0.4
```

### Issue: Model not learning (loss not decreasing)
**Solutions**:
- Reduce learning rate: `--lr 0.0001`
- Increase model capacity: `--hidden-dim 512`
- Check data quality

### Issue: NewsAPI errors
**Solutions**:
- Verify API key in `src/config.py`
- Check internet connection
- Cached data will be used automatically

## Best Practices

### Training
1. Always use GPU for faster training (10x speedup)
2. Monitor validation loss to detect overfitting
3. Use early stopping (automatic)
4. Save all training artifacts

### Evaluation
1. Never evaluate on training data
2. Use test set only once (final evaluation)
3. Focus on validation metrics during development
4. Check multiple metrics, not just RMSE

### Deployment
1. Save both model and scalers
2. Version your models (include date, ticker, metrics)
3. Monitor prediction errors
4. Retrain periodically with new data

## Workflow Summary

### Complete Workflow

```bash
# Step 1: Train model
python scripts/train_advanced_model.py --ticker AAPL --days 200

# Step 2: Evaluate model
python scripts/evaluate_advanced_model.py --ticker AAPL

# Step 3: Check results
# - Review data/checkpoints/training_report.json
# - Check data/evaluation_results.png
# - Examine data/evaluation_metrics.json

# Step 4: Use for predictions (see Inference section above)
```

### Hyperparameter Tuning

Try different configurations:

```bash
# Experiment 1: Deeper network
python scripts/train_advanced_model.py --num-layers 4 --hidden-dim 512

# Experiment 2: More regularization
python scripts/train_advanced_model.py --dropout 0.4

# Experiment 3: More data
python scripts/train_advanced_model.py --days 500

# Compare results in respective training_report.json files
```

## System Requirements

### Minimum
- CPU: 4 cores
- RAM: 4GB
- Storage: 2GB
- Training time: ~60-120 minutes

### Recommended
- GPU: NVIDIA GPU with 4GB+ VRAM
- RAM: 8GB+
- Storage: 5GB
- Training time: ~5-10 minutes

## License

This project is for educational and research purposes. Not financial advice.

## Support

For issues:
1. Check training logs in `data/checkpoints/training_report.json`
2. Review error messages carefully
3. Consult troubleshooting section above
4. Check visualization plots for insights

---

**Note**: Always conduct your own research before making investment decisions. This model is a tool for analysis, not a guarantee of future performance.

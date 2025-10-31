# Technical Documentation
## Enhanced Nifty Option Prediction System v1.0.0

**Developer Guide & Technical Reference**

---

## Architecture Overview

### System Components

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Data Layer    │    │  Model Layer    │    │  Strategy Layer │
├─────────────────┤    ├─────────────────┤    ├─────────────────┤
│ • yfinance API  │    │ • TFT Model     │    │ • Option Calc   │
│ • Data Cache    │    │ • HAR-RV Model  │    │ • Risk Mgmt     │
│ • Preprocessing │    │ • ML Ensemble   │    │ • Position Size │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                       │                       │
         └───────────────────────┼───────────────────────┘
                                 │
                    ┌─────────────────┐
                    │  Application    │
                    │  Layer          │
                    ├─────────────────┤
                    │ • Streamlit UI  │
                    │ • API Endpoints │
                    │ • Backtesting   │
                    └─────────────────┘
```

### File Structure

```
project/
├── app.py                    # Main application logic
├── streamlit_app.py         # Web UI interface
├── models/
│   ├── tft_model.py         # Temporal Fusion Transformer
│   ├── har_rv.py           # HAR-RV volatility model
│   ├── data_preparer.py    # Data preparation utilities
│   └── saved/              # Trained model files
├── tests/
│   ├── test_system.py      # End-to-end tests
│   ├── test_features.py    # Feature engineering tests
│   ├── test_streamlit.py   # UI component tests
│   └── test_streamlit_integration.py
├── data_cache/             # Cached market data
└── requirements.txt        # Dependencies
```

---

## Model Specifications

### 1. Temporal Fusion Transformer (TFT)

**File**: `models/tft_model.py`

**Architecture**:
- Input: Time series data with static and dynamic features
- Encoder: LSTM-based sequence encoder
- Attention: Multi-head attention mechanism
- Decoder: Quantile regression decoder
- Output: Multi-horizon forecasts with uncertainty

**Key Parameters**:
```python
{
    'hidden_size': 64,
    'lstm_layers': 2,
    'attention_head_size': 4,
    'dropout': 0.1,
    'learning_rate': 0.01,
    'batch_size': 64,
    'max_epochs': 100
}
```

**Input Features**:
- Price data (OHLCV)
- Technical indicators (11 indicators)
- Time features (day of week, month)
- Static features (VIX level, market regime)

**Output**:
- Point forecasts for next 1-5 days
- Quantile forecasts (10th, 50th, 90th percentiles)
- Attention weights for interpretability

### 2. HAR-RV Model

**File**: `models/har_rv.py`

**Architecture**:
- Realized Volatility computation using multiple estimators
- Regime detection using Gaussian Mixture Models
- Jump detection using threshold methods
- HAR regression for volatility forecasting

**Volatility Estimators**:
1. **Parkinson**: Uses high-low prices
2. **Garman-Klass**: Uses OHLC prices
3. **Rogers-Satchell**: Accounts for opening gaps
4. **Yang-Zhang**: Combines multiple estimators

**Regime Detection**:
```python
{
    'n_components': 3,  # Bull, Bear, Sideways
    'covariance_type': 'full',
    'random_state': 42
}
```

**HAR Regression**:
- Daily volatility (RV_t)
- Weekly volatility (RV_t^w)
- Monthly volatility (RV_t^m)
- Jump component (J_t)

### 3. ML Ensemble

**File**: `app.py` (in `_generate_signals` method)

**Architecture**:
- Random Forest Classifier
- Input: Technical signals + Sentiment + Volatility
- Output: Combined signal strength and direction

**Features**:
- MACD signal (-1, 0, 1)
- RSI signal (-1, 0, 1)
- Moving average cross signal (-1, 0, 1)
- Sentiment score (0-1)
- Volatility forecast (0-1)
- VIX level (normalized)

---

## Data Pipeline

### 1. Data Fetching

**Source**: Yahoo Finance API via `yfinance`

**Symbols**:
- Nifty 50: `^NSEI`
- India VIX: `^INDIAVIX`

**Data Fields**:
```python
{
    'Open': 'Opening price',
    'High': 'Highest price',
    'Low': 'Lowest price',
    'Close': 'Closing price',
    'Volume': 'Trading volume',
    'Adj Close': 'Adjusted closing price'
}
```

### 2. Data Preprocessing

**Steps**:
1. **Missing Value Handling**: Forward fill, then interpolation
2. **Outlier Detection**: Isolation Forest for extreme values
3. **Feature Engineering**: Technical indicators calculation
4. **Normalization**: Min-max scaling for model inputs
5. **Time Indexing**: Proper datetime indexing for time series

**Technical Indicators**:
```python
indicators = {
    'SMA_20': 'Simple Moving Average (20 periods)',
    'SMA_50': 'Simple Moving Average (50 periods)',
    'EMA_12': 'Exponential Moving Average (12 periods)',
    'EMA_26': 'Exponential Moving Average (26 periods)',
    'MACD': 'MACD Line',
    'MACD_Signal': 'MACD Signal Line',
    'MACD_Histogram': 'MACD Histogram',
    'RSI': 'Relative Strength Index',
    'BB_Upper': 'Bollinger Bands Upper',
    'BB_Lower': 'Bollinger Bands Lower',
    'BB_Middle': 'Bollinger Bands Middle'
}
```

### 3. Data Caching

**Cache Strategy**:
- File-based caching using Parquet format
- Cache duration: 1 hour
- Automatic refresh on data staleness
- Manual refresh option available

**Cache Files**:
```
data_cache/
├── nifty_data_90d.parquet
├── nifty_data_365d.parquet
└── vix_data.parquet
```

---

## API Reference

### EnhancedNiftyPredictionSystem Class

**File**: `app.py`

#### Constructor
```python
def __init__(self, data_period_days=90):
    """
    Initialize the prediction system
    
    Args:
        data_period_days (int): Historical data period in days
    """
```

#### Key Methods

##### `load_data(force_refresh=False)`
```python
def load_data(self, force_refresh=False):
    """
    Load market data from cache or API
    
    Args:
        force_refresh (bool): Force refresh from API
        
    Returns:
        dict: Data loading status and statistics
    """
```

##### `train_ml_models()`
```python
def train_ml_models(self):
    """
    Train all ML models on current data
    
    Returns:
        dict: Training results for each model
    """
```

##### `predict_next_day(days_to_expiry=1)`
```python
def predict_next_day(self, days_to_expiry=1):
    """
    Generate prediction for next trading day
    
    Args:
        days_to_expiry (int): Days until option expiry
        
    Returns:
        dict: Prediction results with confidence and strategy
    """
```

##### `run_backtest(strategy_name, initial_capital=100000)`
```python
def run_backtest(self, strategy_name, initial_capital=100000):
    """
    Run backtest for specified strategy
    
    Args:
        strategy_name (str): Name of strategy to backtest
        initial_capital (float): Starting capital
        
    Returns:
        dict: Backtest results and performance metrics
    """
```

---

## Configuration

### Model Configuration

**TFT Model** (`models/tft_model.py`):
```python
TFT_CONFIG = {
    'hidden_size': 64,
    'lstm_layers': 2,
    'attention_head_size': 4,
    'dropout': 0.1,
    'learning_rate': 0.01,
    'batch_size': 64,
    'max_epochs': 100,
    'early_stopping_patience': 10
}
```

**HAR-RV Model** (`models/har_rv.py`):
```python
HAR_RV_CONFIG = {
    'min_data_points': 10,
    'volatility_window': 20,
    'regime_components': 3,
    'jump_threshold': 2.0,
    'bias_correction': True
}
```

### Risk Management Configuration

**Position Sizing** (`app.py`):
```python
RISK_CONFIG = {
    'max_position_size': 0.1,  # 10% of capital
    'stop_loss_pct': 0.5,       # 50% of premium
    'target_profit_pct': 0.75,  # 75% of premium
    'max_drawdown': 0.15       # 15% maximum drawdown
}
```

---

## Testing Framework

### Test Structure

**Unit Tests** (`tests/`):
- `test_system.py`: End-to-end system tests
- `test_features.py`: Feature engineering tests
- `test_streamlit.py`: UI component tests
- `test_streamlit_integration.py`: Integration tests

### Running Tests

```bash
# Run all tests
python -m pytest tests/

# Run specific test file
python -m pytest tests/test_system.py

# Run with coverage
python -m pytest tests/ --cov=app --cov-report=html
```

### Test Categories

1. **Data Tests**: Data loading, preprocessing, caching
2. **Model Tests**: Model training, prediction, persistence
3. **Strategy Tests**: Option strategy generation, backtesting
4. **UI Tests**: Streamlit interface, user interactions
5. **Integration Tests**: End-to-end workflow validation

---

## Performance Optimization

### Model Optimization

**TFT Model**:
- Use GPU acceleration when available
- Implement model quantization for faster inference
- Cache model predictions for repeated queries

**HAR-RV Model**:
- Optimize volatility computation using vectorized operations
- Implement incremental learning for regime detection
- Cache volatility forecasts

### Data Optimization

**Caching Strategy**:
- Implement Redis for distributed caching
- Use compression for large datasets
- Implement data versioning

**API Optimization**:
- Implement request batching
- Use connection pooling
- Implement retry logic with exponential backoff

---

## Deployment

### Local Deployment

**Requirements**:
- Python 3.8+
- 4GB RAM minimum
- 2GB disk space

**Setup**:
```bash
# Install dependencies
pip install -r requirements.txt

# Run Streamlit app
streamlit run streamlit_app.py --server.port 8501
```

### Production Deployment

**Docker Setup**:
```dockerfile
FROM python:3.9-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .
EXPOSE 8501

CMD ["streamlit", "run", "streamlit_app.py", "--server.port=8501", "--server.address=0.0.0.0"]
```

**Environment Variables**:
```bash
# Data configuration
DATA_CACHE_DIR=/app/data_cache
MODEL_SAVE_DIR=/app/models/saved

# API configuration
YAHOO_FINANCE_TIMEOUT=30
CACHE_DURATION_HOURS=1

# Model configuration
TFT_MODEL_PATH=/app/models/saved/tft_model.safetensors
HAR_RV_MODEL_PATH=/app/models/saved/har_rv_model.joblib
```

---

## Monitoring & Logging

### Logging Configuration

**Log Levels**:
- `DEBUG`: Detailed diagnostic information
- `INFO`: General information about program execution
- `WARNING`: Something unexpected happened
- `ERROR`: Serious problem occurred
- `CRITICAL`: Very serious error occurred

**Log Files**:
```
logs/
├── app.log              # Main application logs
├── model_training.log   # Model training logs
├── prediction.log       # Prediction logs
└── error.log           # Error logs
```

### Performance Monitoring

**Metrics to Track**:
- Model training time
- Prediction latency
- Data fetch time
- Cache hit rate
- Memory usage
- CPU utilization

**Monitoring Tools**:
- Prometheus for metrics collection
- Grafana for visualization
- ELK stack for log analysis

---

## Troubleshooting

### Common Issues

#### 1. Model Training Failures

**Symptoms**:
- "Insufficient data for training"
- "Model training failed"

**Solutions**:
- Increase historical data period
- Check data quality and completeness
- Verify model parameters
- Check system resources (RAM, disk space)

#### 2. Prediction Errors

**Symptoms**:
- Low confidence predictions
- Inconsistent results

**Solutions**:
- Retrain models with fresh data
- Check feature engineering pipeline
- Verify model persistence/loading
- Review prediction logic

#### 3. Data Issues

**Symptoms**:
- Missing data errors
- Stale data warnings

**Solutions**:
- Force refresh data
- Check API connectivity
- Verify data cache integrity
- Review data preprocessing pipeline

### Debug Mode

**Enable Debug Logging**:
```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

**Debug Flags**:
```python
DEBUG_CONFIG = {
    'verbose_logging': True,
    'save_intermediate_data': True,
    'model_debug_mode': True,
    'prediction_debug': True
}
```

---

## Contributing

### Development Setup

1. Fork the repository
2. Create feature branch
3. Install development dependencies
4. Run tests before committing
5. Submit pull request

### Code Standards

**Python Style**:
- Follow PEP 8 guidelines
- Use type hints
- Write docstrings for all functions
- Maintain test coverage > 80%

**Git Workflow**:
- Use descriptive commit messages
- Create feature branches
- Squash commits before merging
- Update documentation for new features

---

## License

This project is licensed under the MIT License. See LICENSE file for details.

---

**Version**: 1.0.0  
**Last Updated**: October 2025  
**Author**: Pravin A Mathew

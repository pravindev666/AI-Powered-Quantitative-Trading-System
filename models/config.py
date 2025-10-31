"""Model configuration, paths, and feature availability"""

from pathlib import Path

# Base paths
MODELS_DIR = Path("models/saved")
CACHE_DIR = Path("data_cache")

# Ensure directories exist
MODELS_DIR.mkdir(parents=True, exist_ok=True)
CACHE_DIR.mkdir(parents=True, exist_ok=True)

# Model paths
TFT_MODEL_PATH = MODELS_DIR / "tft_model.safetensors"
HAR_RV_MODEL_PATH = MODELS_DIR / "har_rv_model.safetensors"

# Training configs
TFT_CONFIG = {
    "hidden_size": 32,
    "num_attention_heads": 4,
    "dropout": 0.1,
    "learning_rate": 0.001
}

HAR_RV_CONFIG = {
    "lags": [1, 5, 22],  # Daily, weekly, monthly
    "window_size": 66,  # 3 months
    "alpha": 0.01
}

# Feature availability flags
TFT_AVAILABLE = False
SENTIMENT_AVAILABLE = False

# Check for TFT availability
try:
    import torch
    from niftypred.models.tft_model import TemporalFusionTransformerModel
    TFT_AVAILABLE = True
except ImportError as e:
    pass

# Check for sentiment analysis availability
try:
    from niftypred.models.market_sentiment import MarketSentimentAnalyzer
    SENTIMENT_AVAILABLE = True
except ImportError:
    pass
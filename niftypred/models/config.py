"""Configuration constants for model paths and hyperparameters"""

import os
from pathlib import Path

# Model paths
MODEL_DIR = Path(__file__).parent.parent.parent / 'models'
TFT_MODEL_PATH = MODEL_DIR / 'tft_model.pth'
HAR_RV_MODEL_PATH = MODEL_DIR / 'har_rv_model.pkl'
FINBERT_MODEL_PATH = MODEL_DIR / 'finbert'

# Model configurations
TFT_CONFIG = {
    'hidden_size': 32,
    'attention_head_size': 4,
    'dropout': 0.1,
    'hidden_continuous_size': 16,
    'output_size': 7,  # Number of quantiles
    'learning_rate': 0.001,
    'log_interval': 10,
    'reduce_on_plateau_patience': 4,
    'max_encoder_length': 30,
    'max_prediction_length': 1,
    'static_categoricals': ['series'],
    'static_reals': [],
    'time_varying_known_reals': ['time_idx'],
    'time_varying_unknown_reals': ['Close', 'High', 'Low', 'Volume', 'IndiaVIX', 'Returns']
}

# Training configurations
TFT_TRAINING_CONFIG = {
    'batch_size': 32,
    'num_epochs': 100
}

# Training configurations
HAR_RV_CONFIG = {
    'lags': [1, 5, 22],  # daily, weekly, monthly
    'train_size': 0.8,
    'random_state': 42
}

# Additional configurations for HAR-RV processing
HAR_RV_PROCESSING_CONFIG = {
    'window_size': 252  # For data preprocessing
}
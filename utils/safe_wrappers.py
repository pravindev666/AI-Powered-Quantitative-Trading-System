"""
Enhanced prediction system patches for stability
"""

import pandas as pd
import numpy as np
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

def safe_calculate_indicators(data: pd.DataFrame, technical_analyzer) -> pd.DataFrame:
    """Safe wrapper for technical indicator calculation"""
    try:
        # Clean input data
        df = data.copy()
        df = df.fillna(method='ffill').fillna(method='bfill')
        
        # Try calculate_all first
        try:
            indicators = technical_analyzer.calculate_all(df)
        except AttributeError:
            logger.warning("calculate_all not found, trying calculate_indicators")
            indicators = technical_analyzer.calculate_indicators(df)
        
        # Convert indicators to DataFrame if needed
        if isinstance(indicators, dict):
            indicators = pd.DataFrame(indicators, index=df.index)
        elif not isinstance(indicators, pd.DataFrame):
            logger.warning(f"Unexpected indicators type: {type(indicators)}")
            return pd.DataFrame(index=df.index)
            
        # Return only new columns
        new_cols = [col for col in indicators.columns if col not in data.columns]
        return indicators[new_cols] if new_cols else pd.DataFrame(index=df.index)
        
    except Exception as e:
        logger.error(f"Error calculating indicators: {e}")
        return pd.DataFrame(index=df.index)

def safe_train_model(model, X: pd.DataFrame, y=None) -> dict:
    """Safe wrapper for model training"""
    try:
        # Clean data
        X = X.fillna(method='ffill').fillna(method='bfill')
        if isinstance(y, pd.Series):
            y = y.fillna(method='ffill').fillna(method='bfill')
        
        # Ensure minimum data requirements
        if len(X) < 50:
            raise ValueError("Insufficient data for training (minimum 50 rows required)")
        
        # Train model
        if hasattr(model, 'train'):
            return model.train(X, y)
        elif hasattr(model, 'fit'):
            model.fit(X, y)
            return {'status': 'success'}
        else:
            raise AttributeError("Model has no train() or fit() method")
            
    except Exception as e:
        logger.error(f"Training error: {e}")
        return {'status': 'error', 'message': str(e)}
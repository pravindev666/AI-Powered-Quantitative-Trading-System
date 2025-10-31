"""Data preparation and model training utilities"""
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging
from typing import Dict, Optional, Tuple
import os

logger = logging.getLogger('nifty_predictor')

class DataPreparer:
    """Handles data preparation and model training for the prediction system"""
    
    def __init__(self, cache_dir: str = "data_cache"):
        """Initialize the data preparer"""
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)
        
    def prepare_training_data(self, 
                            nifty_data: pd.DataFrame,
                            vix_data: pd.DataFrame) -> Tuple[Dict, Dict]:
        """
        Prepare data for training HAR-RV and TFT models
        
        Args:
            nifty_data: DataFrame with Nifty price data
            vix_data: DataFrame with India VIX data
            
        Returns:
            Tuple of dictionaries containing prepared data for HAR-RV and TFT models
        """
        logger.info("Preparing training data...")
        
        # Calculate returns
        nifty_returns = nifty_data['Close'].pct_change()
        vix_returns = vix_data['Close'].pct_change()
        
        # Prepare HAR-RV data
        har_rv_data = {
            'nifty_returns': nifty_returns.dropna(),
            'vix_returns': vix_returns.dropna()
        }
        
        # Prepare TFT data
        # Calculate additional features
        nifty_features = pd.DataFrame()
        nifty_features['returns'] = nifty_returns
        nifty_features['log_returns'] = np.log1p(nifty_returns)
        nifty_features['volatility'] = nifty_returns.rolling(window=22).std()
        nifty_features['momentum'] = nifty_returns.rolling(window=10).mean()
        nifty_features['vix'] = vix_data['Close']
        nifty_features['vix_ma'] = vix_data['Close'].rolling(window=10).mean()
        
        # Prepare target variable (next day return)
        nifty_features['target'] = nifty_returns.shift(-1)
        
        # Drop missing values
        nifty_features = nifty_features.dropna()
        
        # Prepare static and time-varying features
        static_features = ['vix_ma']
        time_varying_features = ['returns', 'volatility', 'momentum', 'vix']
        
        tft_data = {
            'features': nifty_features,
            'static_features': static_features,
            'time_varying_features': time_varying_features
        }
        
        return har_rv_data, tft_data
        
    def train_models(self,
                    har_rv_model,
                    tft_model,
                    nifty_data: pd.DataFrame,
                    vix_data: pd.DataFrame,
                    validation_size: float = 0.2,
                    progress_callback: Optional[callable] = None) -> Dict:
        """
        Train both HAR-RV and TFT models
        
        Args:
            har_rv_model: HARRVModel instance
            tft_model: TemporalFusionTransformerModel instance
            nifty_data: DataFrame with Nifty price data
            vix_data: DataFrame with India VIX data
            validation_size: Size of validation set
            progress_callback: Optional callback for progress updates
            
        Returns:
            Dictionary with training results
        """
        try:
            # Prepare training data
            har_rv_data, tft_data = self.prepare_training_data(nifty_data, vix_data)
            
            if progress_callback:
                progress_callback("Data preparation completed")
            
            results = {}
            
            # Train HAR-RV model
            if har_rv_model is not None:
                logger.info("Training HAR-RV model...")
                har_results = har_rv_model.train(
                    returns=har_rv_data['nifty_returns'],
                    validation_size=validation_size
                )
                results['har_rv'] = har_results
                
                if progress_callback:
                    progress_callback("HAR-RV model training completed")
            
            # Train TFT model if available
            if tft_model is not None:
                logger.info("Training TFT model...")
                features = tft_data['features']
                static_features = tft_data['static_features']
                time_varying_features = tft_data['time_varying_features']
                
                # Prepare features and target for TFT
                X = features[static_features + time_varying_features]
                y = features['target']
                
                # Train TFT model
                tft_results = tft_model.train(
                    X=X,
                    y=y,
                    validation_size=validation_size
                )
                results['tft'] = tft_results
                
                if progress_callback:
                    progress_callback("TFT model training completed")
            
            return results
            
        except Exception as e:
            logger.error(f"Error during model training: {str(e)}")
            raise
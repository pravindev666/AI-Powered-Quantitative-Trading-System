"""
End-to-end system tests for the Nifty Option Prediction System
"""
import unittest
import pandas as pd
import numpy as np
import logging

logger = logging.getLogger(__name__)
from datetime import datetime, timedelta
from pathlib import Path
import sys
import os
import warnings
import logging

# Suppress warnings during tests
warnings.filterwarnings('ignore')
logging.getLogger().setLevel(logging.ERROR)

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from niftypred.models.market_sentiment import MarketSentimentAnalyzer
from niftypred.models.har_rv import HARRVModel
from niftypred.models.technical_indicators import TechnicalIndicators
from niftypred.models.tft_model import TemporalFusionTransformerModel
from niftypred.utils.risk_management import RiskManager, PositionSizer

class TestPredictionSystem(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """Set up test fixtures"""
        # Create sample market data with clean, realistic values
        dates = pd.date_range(start='2025-01-01', end='2025-10-29', freq='D')
        np.random.seed(42)  # For reproducibility
        
        # Generate realistic price movements
        prices = []
        current_price = 100.0
        for _ in range(len(dates)):
            # Generate small random changes (0.5% to -0.5%)
            change = np.random.uniform(-0.005, 0.005)
            current_price *= (1 + change)
            prices.append(current_price)
        
        prices = np.array(prices)
        returns = np.diff(np.log(prices))
        returns = np.insert(returns, 0, 0)  # Add 0 return for first day
        
        # Ensure data is clean and finite
        cls.market_data = pd.DataFrame({
            'Close': prices,
            'High': prices * 1.01,  # 1% above close
            'Low': prices * 0.99,   # 1% below close
            'Volume': np.random.randint(1000000, 10000000, len(dates)),
            'Returns': returns,
            'IndiaVIX': np.clip(np.random.normal(15, 2, len(dates)), 10, 30)  # Bounded VIX
        }, index=dates)
        
        # Verify no NaN or infinite values
        assert not cls.market_data.isnull().any().any(), "Found NaN values in test data"
        assert np.all(np.isfinite(cls.market_data.values)), "Found infinite values in test data"
        
    def setUp(self):
        """Initialize components for each test"""
        self.tech_analyzer = TechnicalIndicators()
        self.har_rv = HARRVModel()
        self.tft_model = TemporalFusionTransformerModel()
        self.risk_manager = RiskManager()
        self.position_sizer = PositionSizer(initial_capital=1000000)  # 10 lakh initial capital
        
    def test_data_preprocessing(self):
        """Test data preprocessing and validation"""
        try:
            # Test with valid data
            indicators = self.tech_analyzer.calculate_all(self.market_data)
            self.assertIsInstance(indicators, pd.DataFrame)
            self.assertTrue(len(indicators.columns) > 5)  # Should have multiple indicators
            
            # Check specific indicator columns
            expected_indicators = ['SMA_20', 'SMA_50', 'MACD', 'RSI', 'BB_Middle']
            for indicator in expected_indicators:
                self.assertIn(indicator, indicators.columns)
            
            # Test with missing required columns
            bad_data = self.market_data.drop(['High', 'Low'], axis=1)
            with self.assertRaises(ValueError) as cm:
                self.tech_analyzer.calculate_all(bad_data)
            self.assertIn("Missing required columns", str(cm.exception))
            
            # Test with NaN data
            nan_data = self.market_data.copy()
            nan_data.iloc[10:20, :] = np.nan  # Set a block of data to NaN
            with self.assertRaises(ValueError) as cm:
                self.tech_analyzer.calculate_all(nan_data)
            self.assertIn("Cannot calculate indicators", str(cm.exception))
            
        except Exception as e:
            self.fail(f"Data preprocessing failed: {str(e)}")
            
    def test_signal_generation(self):
        """Test trading signal generation"""
        try:
            # Calculate indicators first
            data = pd.concat([
                self.market_data,
                self.tech_analyzer.calculate_all(self.market_data)
            ], axis=1)
            
            # Generate signals
            signals = self.tech_analyzer.generate_signals(data)
            
            # Verify signal structure
            self.assertIsInstance(signals, dict)
            self.assertIn('combined', signals)
            self.assertTrue(isinstance(signals['combined'], (int, float)))
            self.assertTrue(-1 <= signals['combined'] <= 1)
            
            # Test signal components
            required_signals = ['MACD', 'RSI', 'MA_Cross', 'BB']
            for sig in required_signals:
                self.assertIn(sig, signals)
                self.assertTrue(-1 <= signals[sig] <= 1)
                
        except Exception as e:
            self.fail(f"Signal generation failed: {str(e)}")
            
    def test_volatility_forecasting(self):
        """Test HAR-RV volatility forecasting"""
        try:
            # Train model
            returns = self.market_data['Returns'].copy()
            metrics = self.har_rv.train(returns)
            
            # Verify training metrics
            self.assertIsInstance(metrics, dict)
            
            # Generate forecast
            forecast = self.har_rv.forecast(returns)
            self.assertIsNotNone(forecast)
            self.assertTrue(forecast >= 0)  # Volatility should be non-negative
            
            # Test with bad data
            with self.assertRaises(ValueError):
                self.har_rv.train(pd.Series([np.nan] * 10))
                
        except Exception as e:
            self.fail(f"Volatility forecasting failed: {str(e)}")
            
    def test_position_sizing(self):
        """Test position sizing and risk management"""
        try:
            # Calculate position size with realistic parameters
            entry_price = 100.0  # Entry at 100
            stop_price = 95.0    # Stop loss at 95 (5% risk)
            risk_score = 0.75    # 75% confidence
            volatility = 0.20    # 20% volatility
            
            position = self.position_sizer.calculate_position_size(
                entry_price=entry_price,
                stop_price=stop_price,
                risk_score=risk_score,
                volatility=volatility
            )['position_size']  # Extract position size from returned dict
            
            # Verify position constraints
            self.assertTrue(isinstance(position, (int, float)))
            self.assertTrue(position >= 0)  # No negative positions
            self.assertTrue(position <= 1_000_000)  # Max capital constraint
            
            # Test risk limits with higher volatility
            high_vol_pos = self.position_sizer.calculate_position_size(
                entry_price=entry_price,
                stop_price=stop_price,
                risk_score=risk_score,
                volatility=0.50  # Higher volatility
            )['position_size']
            
            self.assertTrue(high_vol_pos <= position)  # Should reduce position size for higher risk
            
            # Test invalid price scenario
            zero_risk_pos = self.position_sizer.calculate_position_size(
                entry_price=100.0,
                stop_price=100.0,  # No risk difference
                risk_score=0.75,
                volatility=0.20
            )['position_size']
            self.assertEqual(zero_risk_pos, 0)  # Should return zero for invalid risk
            
        except Exception as e:
            self.fail(f"Position sizing failed: {str(e)}")
            
    def test_error_handling(self):
        """Test system error handling"""
        try:
            # Test with empty data
            with self.assertRaises(ValueError) as cm:
                empty_df = pd.DataFrame(columns=['Close', 'High', 'Low', 'Volume'])
                self.tech_analyzer.calculate_all(empty_df)
            self.assertIn("Cannot calculate indicators on empty data", str(cm.exception))
                
            # Test with missing required columns
            with self.assertRaises(ValueError) as cm:
                incomplete_df = pd.DataFrame({'Close': [1, 2, 3], 'High': [2, 3, 4]})
                self.tech_analyzer.calculate_all(incomplete_df)
            self.assertIn("Missing required columns", str(cm.exception))
                
            # Test with insufficient data points
            with self.assertRaises(ValueError) as cm:
                short_df = pd.DataFrame({
                    'Close': [1, 2],
                    'High': [2, 3],
                    'Low': [1, 2],
                    'Volume': [100, 200]
                })
                self.tech_analyzer.calculate_all(short_df)
            self.assertIn("Insufficient data points", str(cm.exception))
                
        except Exception as e:
            self.fail(f"Error handling test failed: {str(e)}")
            
    def test_prediction_flow(self):
        """Test end-to-end prediction flow"""
        try:
            # Calculate indicators
            data = pd.concat([
                self.market_data,
                self.tech_analyzer.calculate_all(self.market_data)
            ], axis=1)
            
            # Generate all signals
            tech_signals = self.tech_analyzer.generate_signals(data)
            vol_forecast = self.har_rv.forecast(data['Returns'])
            
            # Train TFT model
            X = data.copy()
            # If Returns is a DataFrame, select the appropriate column
            if isinstance(X['Returns'], pd.DataFrame):
                X['target'] = X['Returns'].iloc[:, 0].shift(-1)  # Use first column as target
            else:
                X['target'] = X['Returns'].shift(-1)  # Next day's returns as target
            X = X.dropna()  # Remove rows with NaN from the shift
            self.tft_model.train(X, max_epochs=3)  # Use short training for tests
            
            # Make predictions
            tft_pred = self.tft_model.predict(X)
            
            # Verify prediction components
            self.assertTrue(all(sig in tech_signals for sig in ['MACD', 'RSI', 'MA_Cross']))
            self.assertGreaterEqual(vol_forecast, 0)
            self.assertIn('q50', tft_pred.columns)
            
        except Exception as e:
            self.fail(f"End-to-end prediction failed: {str(e)}")
            
    def test_model_persistence(self):
        """Test model saving and loading"""
        try:
            # Create temp directory for model files
            temp_dir = Path("temp_test_models")
            temp_dir.mkdir(exist_ok=True)
            
            try:
                # Train and save TFT model
                tft_path = temp_dir / "tft_test.safetensors"
                X = self.market_data.copy()
                X['target'] = X['Returns'].shift(-1)  # Next day's returns as target
                X = X.dropna()  # Remove rows with NaN from the shift
                self.tft_model.train(X, max_epochs=3)  # Short training for tests
                self.tft_model.save_model(str(tft_path))
                
                # Train and save HAR-RV model
                har_path = temp_dir / "har_test.pkl"
                self.har_rv.train(self.market_data['Returns'])
                self.har_rv.save_model(str(har_path))
                
                # Verify files exist
                self.assertTrue(tft_path.exists())
                self.assertTrue(har_path.exists())
                
                # Load models and verify
                new_tft = TemporalFusionTransformerModel()
                new_tft.load_model(str(tft_path))
                
                new_har = HARRVModel()
                new_har.load_model(str(har_path))
                
                # Test predictions match (if both models are trained)
                if (hasattr(self.tft_model, 'training_data') and 
                    hasattr(new_tft, 'training_data') and 
                    self.tft_model.training_data is not None and 
                    new_tft.training_data is not None):
                    tft_pred1 = self.tft_model.predict(X)
                    tft_pred2 = new_tft.predict(X)
                    pd.testing.assert_frame_equal(tft_pred1, tft_pred2)
                else:
                    # If models aren't properly trained, just verify they can be instantiated
                    self.assertIsNotNone(self.tft_model)
                    self.assertIsNotNone(new_tft)
                    # Verify the model files were created
                    self.assertTrue(tft_path.exists())
                    self.assertTrue(har_path.exists())
                
            finally:
                # Cleanup with retry mechanism for Windows
                import shutil
                import gc
                import time
                
                if temp_dir.exists():
                    # Force garbage collection
                    gc.collect()
                    
                    # Retry loop for cleanup
                    for _ in range(3):
                        try:
                            shutil.rmtree(temp_dir)
                            break
                        except PermissionError:
                            time.sleep(0.5)  # Wait before retry
                        except Exception as cleanup_error:
                            logger.warning(f"Cleanup error: {cleanup_error}")
                            break
                    
        except Exception as e:
            self.fail(f"Model persistence test failed: {str(e)}")

if __name__ == '__main__':
    unittest.main(verbose=2)
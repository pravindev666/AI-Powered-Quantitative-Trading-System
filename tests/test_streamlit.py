"""
Test suite for Streamlit app functionality
"""
import unittest
import pandas as pd
import numpy as np
from pathlib import Path
import sys
import os

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from niftypred.models.har_rv import HARRVModel
from niftypred.models.tft_model import TemporalFusionTransformerModel
from niftypred.utils.risk_management import RiskManager
from niftypred.models.technical_indicators import TechnicalIndicators

class TestStreamlitApp(unittest.TestCase):
    
    def setUp(self):
        """Set up test fixtures"""
        # Create sample data
        dates = pd.date_range(start='2025-01-01', end='2025-10-29', freq='D')
        self.test_data = pd.DataFrame({
            'Close': np.random.randn(len(dates)).cumsum() + 100,
            'High': np.random.randn(len(dates)).cumsum() + 102,
            'Low': np.random.randn(len(dates)).cumsum() + 98,
            'Volume': np.random.randint(1000000, 10000000, len(dates)),
            'Returns': np.random.randn(len(dates)) * 0.02,
            'IndiaVIX': np.random.randn(len(dates)) * 5 + 15
        }, index=dates)
        
        # Initialize components
        self.har_rv = HARRVModel()
        self.tft_model = TemporalFusionTransformerModel()
        self.tech_analyzer = TechnicalIndicators()
        
    def test_har_rv_model(self):
        """Test HAR-RV model has required methods"""
        # Verify fit method exists
        self.assertTrue(hasattr(self.har_rv, 'train'), "HAR-RV model should have train method")
        
        # Test training
        result = self.har_rv.train(self.test_data['Returns'])
        self.assertIsInstance(result, dict, "train() should return metrics dict")
        
        # Test forecasting
        forecast = self.har_rv.forecast(self.test_data['Returns'])
        self.assertIsNotNone(forecast, "forecast() should return a value")
        
    def test_tft_model_validation(self):
        """Test TFT model handles validation properly"""
        X = self.test_data.copy()
        # Add target column for TFT model
        X['target'] = (X['Returns'].shift(-1) > 0).astype(int)
        X = X.dropna()  # Remove rows with NaN from the shift
        
        # Verify train method accepts correct params
        try:
            result = self.tft_model.train(X, max_epochs=1)
            self.assertIsInstance(result, dict, "train() should return metrics dict")
        except TypeError as e:
            self.fail(f"TFT model training failed: {str(e)}")
            
    def test_technical_signals(self):
        """Test technical indicators and signals"""
        # Calculate indicators
        indicators = self.tech_analyzer.calculate_all(self.test_data)
        self.assertIsInstance(indicators, pd.DataFrame)
        
        # Generate signals 
        data = pd.concat([self.test_data, indicators], axis=1)
        signals = self.tech_analyzer.generate_signals(data)
        self.assertIsInstance(signals, dict)
        self.assertIn('combined', signals)
        
    def test_error_handling(self):
        """Test proper error handling"""
        try:
            # Try operations that previously caused white screens
            self.har_rv.train(pd.Series([np.nan] * 10))  # All NaN data
            self.fail("Should raise ValueError for all-NaN data")
        except ValueError:
            pass  # Expected error
            
        try:
            # Test TFT with empty data - should handle gracefully
            empty_df = pd.DataFrame(columns=['Close', 'High', 'Low', 'Volume', 'IndiaVIX', 'Returns', 'target'])
            self.tft_model.train(empty_df, max_epochs=1)
            self.fail("Should raise ValueError for empty data")
        except (ValueError, KeyError):
            pass  # Expected error

if __name__ == '__main__':
    unittest.main()
"""
Test suite for all new patches and components
"""

import unittest
import numpy as np
import pandas as pd
from pathlib import Path
import yaml
import json

class TestNewComponents(unittest.TestCase):
    
    def setUp(self):
        """Set up test fixtures"""
        # Load config
        with open('config.yaml', 'r') as f:
            self.config = yaml.safe_load(f)
        
        # Create test directories
        Path('tests/data').mkdir(parents=True, exist_ok=True)
        Path('tests/outputs').mkdir(parents=True, exist_ok=True)
    
    def test_nse_fetcher(self):
        """Test NSE option chain fetcher"""
        from nse_fetcher import NSEOptionChainFetcher
        
        fetcher = NSEOptionChainFetcher()
        result = fetcher.fetch_nifty_option_chain()
        
        if result:
            self.assertIn('spot', result)
            self.assertIn('strikes', result)
            self.assertIn('call_prices', result)
            self.assertTrue(len(result['strikes']) > 0)
        else:
            print("   ⚠️ NSE fetch skipped (market closed/connection issue)")
    
    def test_har_rv(self):
        """Test HAR-RV volatility model"""
        from niftypred.models.har_rv import calculate_har_rv
        
        # Generate test data
        np.random.seed(42)
        returns = pd.Series(np.random.normal(0, 0.01, 100))
        
        result = calculate_har_rv(returns)
        self.assertIsNotNone(result)
        self.assertIn('forecast_vol', result)
        self.assertIn('coefficients', result)
        self.assertTrue(0 < result['forecast_vol'] < 1)
    
    def test_sentiment_analyzer(self):
        """Test news sentiment analyzer"""
        from sentiment_analyzer import NewsSentimentAnalyzer
        
        analyzer = NewsSentimentAnalyzer()
        
        # Test single text
        text = "Markets rally on strong earnings and positive outlook"
        sentiment = analyzer.analyze_sentiment(text)
        
        self.assertIn('label', sentiment)
        self.assertIn('score', sentiment)
        self.assertTrue(-1 <= sentiment['sentiment_score'] <= 1)
        
        # Test market sentiment
        market_sentiment = analyzer.get_market_sentiment_score()
        self.assertIn('sentiment_score', market_sentiment)
        self.assertIn('articles_analyzed', market_sentiment)
    
    def test_multi_modal_fusion(self):
        """Test multi-modal fusion"""
        from multi_modal_fusion import MultiModalFusion
        
        fusion = MultiModalFusion(self.config)
        
        # Test inputs
        technical_pred = {
            'technical_score': 0.7,
            'confidence': 80
        }
        
        ml_pred = {
            'ensemble_prediction': 1,
            'ensemble_confidence': 0.8
        }
        
        tft_pred = {
            'q_25': 0.98,
            'q_50': 1.02,
            'q_75': 1.05
        }
        
        sentiment_score = 0.5
        
        result = fusion.fuse_predictions(
            technical_pred, ml_pred, tft_pred, sentiment_score
        )
        
        self.assertIn('trend', result)
        self.assertIn('probability', result)
        self.assertIn('components', result)
        self.assertTrue(0 <= result['probability'] <= 1)
        self.assertTrue(0 <= result['confidence'] <= 100)
    
    def test_model_monitor(self):
        """Test model monitoring"""
        from model_monitor import ModelMonitor
        
        monitor = ModelMonitor()
        
        # Add test metrics
        for i in range(30):
            monitor.record_performance({
                'accuracy': 0.7 - (i * 0.01),
                'auc': 0.75 - (i * 0.005),
                'logloss': 0.5 + (i * 0.01)
            })
        
        drift_info = monitor.check_performance_drift()
        self.assertIn('drift_detected', drift_info)
        
        # Test calibration
        np.random.seed(42)
        y_true = np.random.binomial(1, 0.6, 100)
        y_prob = np.random.random(100)
        
        cal_info = monitor.check_calibration_drift(y_true, y_prob)
        self.assertIn('calibration_drift', cal_info)
    
    def test_visualizer(self):
        """Test visualization utilities"""
        from niftypred.utils import Visualizer
        
        # Test data
        dates = pd.date_range('2025-01-01', periods=100)
        sentiment = pd.Series(np.random.normal(0, 0.3, 100), index=dates)
        metrics = [{'accuracy': 0.7, 'timestamp': str(d)} for d in dates]
        
        # Test plots
        Visualizer.plot_sentiment_timeline(
            sentiment.to_frame('Sentiment'),
            'tests/outputs/test_sentiment.png'
        )
        
        Visualizer.plot_performance_metrics(
            metrics,
            'tests/outputs/test_metrics.png'
        )
        
        self.assertTrue(Path('tests/outputs/test_sentiment.png').exists())
        self.assertTrue(Path('tests/outputs/test_metrics.png').exists())
    
    def test_daily_update(self):
        """Test daily update script"""
        import daily_update
        
        # Test sentiment update
        cache = daily_update.daily_sentiment_update()
        self.assertIsInstance(cache, dict)
        
        # Test model health check
        daily_update.check_model_health()
        
        # Test cleanup
        daily_update.cleanup_old_files()
        
        # Ensure no errors raised
        print("✅ Daily update tests passed")
    
    def tearDown(self):
        """Clean up test artifacts"""
        # Keep test outputs for inspection
        pass

if __name__ == '__main__':
    unittest.main(verbosity=2)
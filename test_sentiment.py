import unittest
from niftypred import TradingSystem
from niftypred.models import MarketSentimentAnalyzer, NewsTradingSignalGenerator
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

class TestSentimentAnalysis(unittest.TestCase):
    def setUp(self):
        self.trading_system = TradingSystem()
        self.test_symbol = "AAPL"  # Using Apple as a test case
        
    def test_sentiment_analyzer_initialization(self):
        """Test if sentiment analyzer is properly initialized"""
        self.assertIsNotNone(self.trading_system.sentiment_analyzer)
        self.assertIsInstance(self.trading_system.sentiment_analyzer, MarketSentimentAnalyzer)
        
    def test_sentiment_impact_analysis(self):
        """Test sentiment impact analysis functionality"""
        impact = self.trading_system.analyze_sentiment_impact(self.test_symbol)
        
        if impact is not None:
            self.assertIn('correlation', impact)
            self.assertIn('predictive_correlation', impact)
            self.assertIn('regime_analysis', impact)
            self.assertIn('metrics', impact)
            
            # Check regime analysis structure
            regime = impact['regime_analysis']
            self.assertIn('high_sentiment', regime)
            self.assertIn('low_sentiment', regime)
            
            # Verify correlation values are in valid range
            self.assertTrue(-1 <= impact['correlation'] <= 1)
            self.assertTrue(-1 <= impact['predictive_correlation'] <= 1)
            
    def test_trading_signal_generation(self):
        """Test trading signal generation with sentiment"""
        signal = self.trading_system.generate_trading_signal(self.test_symbol)
        
        self.assertIsNotNone(signal)
        self.assertIn('signal', signal)
        self.assertIn('confidence', signal)
        self.assertIn('components', signal)
        
        # Verify signal components
        components = signal['components']
        self.assertIn('technical', components)
        self.assertIn('sentiment', components)
        
        # Check signal bounds
        self.assertTrue(-1 <= signal['signal'] <= 1)
        self.assertTrue(0 <= signal['confidence'] <= 1)
        
    def test_position_size_adjustment(self):
        """Test position size adjustment based on sentiment"""
        portfolio = {
            self.test_symbol: 100000  # Test with $100k position
        }
        
        sentiment_data = {
            self.test_symbol: {
                'overall_sentiment': 0.5  # Test with positive sentiment
            }
        }
        
        adjusted = self.trading_system.adjust_position_sizes(portfolio, sentiment_data)
        
        self.assertIn(self.test_symbol, adjusted)
        position = adjusted[self.test_symbol]
        
        # Check position constraints
        original_position = portfolio[self.test_symbol]
        self.assertTrue(0.5 * original_position <= position <= 1.5 * original_position)
        
    def test_risk_limit_updates(self):
        """Test risk limit updates based on sentiment"""
        base_limits = {
            'max_position': 100000,
            'stop_loss': 0.02
        }
        
        sentiment_data = {
            self.test_symbol: {
                'overall_sentiment': -0.5  # Test with negative sentiment
            }
        }
        
        updated = self.trading_system.update_risk_limits(base_limits, sentiment_data)
        
        self.assertEqual(set(base_limits.keys()), set(updated.keys()))
        
        # Check that limits are adjusted appropriately
        for limit_type in base_limits:
            self.assertNotEqual(base_limits[limit_type], updated[limit_type])
            self.assertTrue(0.5 * base_limits[limit_type] <= updated[limit_type] <= 1.5 * base_limits[limit_type])
            
    def test_sentiment_data_persistence(self):
        """Test sentiment data saving functionality"""
        test_data = {
            'overall_sentiment': 0.5,
            'components': {
                'news': 0.6,
                'social': 0.4
            },
            'timestamp': datetime.now().isoformat()
        }
        
        # Test saving
        self.trading_system.save_sentiment_data(self.test_symbol, test_data)
        
        # Verify file was created (implementation dependent)
        # Here you might want to check if the file exists and contains valid JSON
        
    def test_market_sentiment_analyzer(self):
        """Test the MarketSentimentAnalyzer class"""
        analyzer = MarketSentimentAnalyzer()
        
        # Test news sentiment analysis
        news_sentiment = analyzer.analyze_news_sentiment(self.test_symbol)
        if news_sentiment:
            self.assertIn('metrics', news_sentiment)
            self.assertIn('detailed_sentiments', news_sentiment)
            
        # Test social sentiment analysis
        social_sentiment = analyzer.analyze_social_sentiment(self.test_symbol)
        if social_sentiment:
            self.assertIn('platform_sentiments', social_sentiment)
            self.assertIn('overall_sentiment', social_sentiment)
            
    def test_news_trading_signal_generator(self):
        """Test the NewsTradingSignalGenerator class"""
        generator = NewsTradingSignalGenerator()
        
        # Get current price for test
        import yfinance as yf
        try:
            current_price = yf.Ticker(self.test_symbol).history(period='1d')['Close'].iloc[-1]
        except:
            current_price = 100  # Fallback for testing
            
        signal = generator.generate_signal(self.test_symbol, current_price)
        
        self.assertIsNotNone(signal)
        self.assertIn('signal', signal)
        self.assertIn('confidence', signal)
        self.assertIn('sentiment', signal)
        self.assertIn('metadata', signal)
        
        # Check signal bounds
        self.assertTrue(-1 <= signal['signal'] <= 1)
        self.assertTrue(0 <= signal['confidence'] <= 1)

if __name__ == '__main__':
    unittest.main()
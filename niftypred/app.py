"""
Enhanced Nifty Option Prediction System
Core system implementation
"""

import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings
from scipy.stats import norm
import sqlite3
import os
import json
from pathlib import Path
import logging
from niftypred.utils import Timer, log_execution_time, logger
import torch
from typing import Dict, List, Tuple, Optional, Callable
from scipy.optimize import minimize
import time
import safetensors.torch

def safe_load_model(model_path: str) -> Dict:
    """Safely load a PyTorch model using safetensors"""
    try:
        model_path = Path(model_path)
        if model_path.suffix == '.safetensors':
            return safetensors.torch.load_file(str(model_path))

        # Only allow conversion when torch >= 2.6 due to security restrictions
        torch_version = tuple(int(x) for x in torch.__version__.split('+')[0].split('.')[:2])
        if torch_version >= (2, 6):
            # Convert to safetensors format first
            logger.info(f"Converting {model_path} to safetensors format")
            state_dict = torch.load(model_path, map_location='cpu')
            safe_path = model_path.with_suffix('.safetensors')
            safetensors.torch.save_file(state_dict, str(safe_path))
            try:
                os.remove(model_path)  # Remove the original unsafe file
            except Exception:
                pass  # Ignore if can't remove
            return safetensors.torch.load_file(str(safe_path))
        else:
            logger.error(
                f"Refusing to call torch.load: installed torch {torch.__version__} < 2.6. "
                "Provide a `.safetensors` file or upgrade torch to >=2.6 to enable conversion."
            )
            return {}
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        return {}

try:
    from niftypred.market_sentiment import MarketSentimentAnalyzer, NewsTradingSignalGenerator
    SENTIMENT_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Sentiment analysis disabled - {e}")
    SENTIMENT_AVAILABLE = False
    MarketSentimentAnalyzer = None
    NewsTradingSignalGenerator = None

from niftypred.models.market_microstructure import OrderBookUpdate, OrderBook
from niftypred.models.har_rv import HARRVModel
from niftypred.models.config import (
    TFT_MODEL_PATH, HAR_RV_MODEL_PATH, 
    TFT_CONFIG, HAR_RV_CONFIG
)

# Make TFT optional 
try:
    from niftypred.models.tft_model import TemporalFusionTransformerModel
    TFT_AVAILABLE = True
except ImportError as e:
    print(f"Warning: TFT model disabled - {e}")
    TFT_AVAILABLE = False
    class TemporalFusionTransformerModel:
        def __init__(self): pass
        def train(self, X, y): return None
        def predict(self, X): 
            return pd.DataFrame({
                'q05': [0.4] * len(X),
                'q50': [0.5] * len(X),
                'q95': [0.6] * len(X)
            }, index=X.index)

from niftypred.utils.risk_management import RiskManager, PositionSizer
from niftypred.models.technical_indicators import TechnicalAnalyzer
from niftypred.utils.portfolio_optimizer import PortfolioOptimizer
from niftypred.utils.order_types import OrderSide, OrderType, OrderRequest
from niftypred.visualizer import Visualizer

logger = logging.getLogger(__name__)
warnings.filterwarnings('ignore')

class EnhancedNiftyPredictionSystem:
    """Enhanced prediction system for Nifty options trading"""
    
    def __init__(self, spot_price: Optional[float] = None, expiry_date: Optional[str] = None):
        """
        Initialize the prediction system
        
        Args:
            spot_price: Current Nifty spot price (optional)
            expiry_date: Target expiry date in YYYY-MM-DD format
        """
        self.spot_price = spot_price
        self.expiry_date = expiry_date
        self.data = None
        self.models = {}
        self.signals = {}
        self.predictions = {}
        self.latest_prediction = None
        
        # Initialize components with optional models
        # Sentiment Analysis
        if SENTIMENT_AVAILABLE:
            self.sentiment_analyzer = MarketSentimentAnalyzer()
            self.signal_generator = NewsTradingSignalGenerator()
        else:
            self.sentiment_analyzer = None
            self.signal_generator = None
            
        # TFT Model
        if TFT_AVAILABLE:
            self.tft_model = TemporalFusionTransformerModel(**TFT_CONFIG)
            if TFT_MODEL_PATH.exists():
                try:
                    self.tft_model.load_model(str(TFT_MODEL_PATH))
                    logger.info("TFT model loaded securely")
                except Exception as e:
                    logger.warning(f"Could not load TFT model: {e}")
                    self.tft_model = None
            else:
                logger.info("No pre-trained TFT model found")
        else:
            self.tft_model = None
            logger.info("TFT model not available")
            
        # Core components
        self.technical_analyzer = TechnicalAnalyzer()
        self.har_rv_model = HARRVModel(**HAR_RV_CONFIG)
        # Try loading pre-trained HAR-RV model
        if HAR_RV_MODEL_PATH.exists():
            try:
                self.har_rv_model.load_model(str(HAR_RV_MODEL_PATH))
                logger.info("HAR-RV model loaded securely")
            except Exception as e:
                logger.warning(f"Could not load HAR-RV model: {e}")
                self.har_rv_model = None
        else:
            logger.info("No pre-trained HAR-RV model found")
            
        self.risk_manager = RiskManager()
        self.position_sizer = PositionSizer(initial_capital=1000000)  # 1M initial capital
        self.portfolio_optimizer = PortfolioOptimizer()
        self.order_book = OrderBook()
        self.viz = Visualizer()
        
        # Configure logging
        self._setup_logging()
        
        logger.info("EnhancedNiftyPredictionSystem initialized")
        
    def _setup_logging(self):
        """Configure logging settings"""
        log_dir = Path("logs")
        log_dir.mkdir(exist_ok=True)
        
        fh = logging.FileHandler(log_dir / f"nifty_system_{datetime.now().strftime('%Y%m%d')}.log")
        fh.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        fh.setFormatter(formatter)
        logger.addHandler(fh)
    
    def load_data(self, days: int = 300, force_refresh: bool = False) -> None:
        """
        Load historical market data
        
        Args:
            days: Number of historical days to fetch
            force_refresh: Force refresh cached data
        """
        cache_file = Path("data_cache") / f"nifty_data_{days}d.parquet"
        cache_file.parent.mkdir(exist_ok=True)
        
        if cache_file.exists() and not force_refresh:
            try:
                self.data = pd.read_parquet(cache_file)
                logger.info(f"Loaded cached data from {cache_file}")
                return
            except Exception as e:
                logger.warning(f"Failed to load cached data: {e}. Fetching fresh data...")
                force_refresh = True
            
        # Fetch data
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)
        
        try:
            logger.info(f"Fetching Nifty data from {start_date} to {end_date}")
            nifty = yf.download("^NSEI", start=start_date, end=end_date, progress=False)
            if nifty.empty:
                raise ValueError("No Nifty data received from Yahoo Finance")
            
            logger.info("Fetching VIX data")
            vix = yf.download("^INDIAVIX", start=start_date, end=end_date, progress=False)
            if vix.empty:
                raise ValueError("No VIX data received from Yahoo Finance")
            
            # Process and merge data
            logger.info(f"Processing market data - Nifty rows: {len(nifty)}, VIX rows: {len(vix)}")
            
            # Log sample of data
            logger.info(f"Latest Nifty data:\n{nifty.tail(1)}")
            logger.info(f"Latest VIX data:\n{vix.tail(1)}")
            
            # Take relevant columns and align on the datetime index
            left = nifty.loc[:, ['Close', 'High', 'Low', 'Volume']].copy()
            right = vix.loc[:, ['Close']].rename(columns={'Close': 'IndiaVIX'}).copy()
            
            # Log data alignment
            logger.info(f"Data alignment - Nifty dates: {left.index[0]} to {left.index[-1]}")
            logger.info(f"Data alignment - VIX dates: {right.index[0]} to {right.index[-1]}")

            # Ensure proper datetime index
            left.index = pd.to_datetime(left.index)
            right.index = pd.to_datetime(right.index)

            # Concatenate on index to avoid scalar construction issues
            data = pd.concat([left, right], axis=1, join='outer')

            # Calculate returns (Close may contain NaNs after join)
            data['Returns'] = data['Close'].pct_change()
            
            # Save to cache
            data.to_parquet(cache_file)
            
            self.data = data
            logger.info(f"Fetched and cached fresh data for {days} days")
            
        except Exception as e:
            logger.error(f"Error loading data: {str(e)}")
            raise
            
    def calculate_indicators(self) -> None:
        """Calculate technical indicators"""
        if self.data is None:
            raise ValueError("No data loaded. Call load_data() first.")
            
        try:
            indicators = self.technical_analyzer.calculate_all(self.data)
            self.data = pd.concat([self.data, indicators], axis=1)
            logger.info("Calculated technical indicators")
            
        except Exception as e:
            logger.error(f"Error calculating indicators: {str(e)}")
            raise
            
    def generate_signals(self) -> None:
        """Generate trading signals from various sources"""
        try:
            if self.data is None:
                raise ValueError("No data loaded. Call load_data() first.")

            # Initialize signals dictionary with default values
            self.signals = {
                'technical': {'combined': 0.5},  # Neutral default
                'sentiment': {'score': 0.5},     # Neutral default
                'volatility': {'forecast': 0.0}  # Zero volatility default
            }

            # Technical signals
            try:
                tech_signals = self.technical_analyzer.generate_signals(self.data)
                if tech_signals:
                    self.signals['technical'] = tech_signals
            except Exception as e:
                logger.warning(f"Technical analysis failed: {e}")

            # Sentiment signals
            if self.sentiment_analyzer:
                try:
                    sentiment_data = self.sentiment_analyzer.get_latest_sentiment()
                    sentiment_signals = self.signal_generator.generate_signals(sentiment_data)
                    if sentiment_signals:
                        self.signals['sentiment'] = sentiment_signals
                except Exception as e:
                    logger.warning(f"Sentiment analysis failed: {e}")

            # Volatility signals
            try:
                if self.har_rv_model and 'Returns' in self.data.columns:
                    vol_forecast = self.har_rv_model.forecast(self.data['Returns'])
                    if vol_forecast is not None:
                        self.signals['volatility'] = {'forecast': vol_forecast}
            except Exception as e:
                logger.warning(f"Volatility forecast failed: {e}")

            logger.info(f"Generated trading signals: {self.signals}")
            
        except Exception as e:
            logger.error(f"Error generating signals: {str(e)}")
            raise
            
    def train_ml_models(self) -> None:
        """Train machine learning models"""
        try:
            if self.data is None:
                raise ValueError("No data loaded. Call load_data() first.")
                
            # Prepare features for TFT
            X = self.data.copy()
            y = np.where(X['Returns'].shift(-1) > 0, 1, 0)  # Next day direction
            
            # Train TFT model if available
            if self.tft_model and TFT_AVAILABLE:
                self.models['tft'] = self.tft_model.train(X, y)
                # Save TFT model securely
                self.tft_model.save_model(str(TFT_MODEL_PATH))
                logger.info("Trained and saved TFT model")
            else:
                logger.info("Skipping TFT model training - not available")
                
            # Train HAR-RV model
            if self.har_rv_model:
                self.har_rv_model.fit(self.data['Returns'])
                # Save HAR-RV model securely
                self.har_rv_model.save_model(str(HAR_RV_MODEL_PATH))
                logger.info("Trained and saved HAR-RV model")
            else:
                logger.warning("HAR-RV model not initialized")
            
        except Exception as e:
            logger.error(f"Error training models: {str(e)}")
            raise
            
    def predict_next_day(self) -> Dict:
        """
        Generate next day prediction
        
        Returns:
            Dict containing prediction details
        """
        try:
            if self.data is None:
                raise ValueError("No data loaded. Call load_data() first.")

            # Ensure signals are generated
            if not self.signals:
                self.generate_signals()

            # Get latest data point
            latest = self.data.iloc[-1].to_dict()
            
            # Generate TFT prediction if available
            if self.tft_model and TFT_AVAILABLE:
                try:
                    tft_predictions = self.tft_model.predict(self.data)
                    tft_pred = tft_predictions['q50'].iloc[-1]
                    tft_uncertainty = {
                        'lower': float(tft_predictions['q05'].iloc[-1]),
                        'upper': float(tft_predictions['q95'].iloc[-1])
                    }
                except Exception as e:
                    logger.warning(f"TFT prediction failed: {e}")
                    tft_pred = 0.5
                    tft_uncertainty = {'lower': 0.4, 'upper': 0.6}
            else:
                tft_pred = 0.5
                tft_uncertainty = {'lower': 0.4, 'upper': 0.6}
            
            # Combine signals
            signal_weights = {
                'technical': 0.4,
                'sentiment': 0.3,
                'volatility': 0.3
            }
            
            final_prob = (
                signal_weights['technical'] * self.signals['technical']['combined'] +
                signal_weights['sentiment'] * self.signals['sentiment']['score'] +
                signal_weights['volatility'] * (0.5 + self.signals['volatility']['forecast'])
            )
            
            # Determine trend
            if final_prob > 0.6:
                trend = "BULLISH"
                trend_emoji = "📈"
            elif final_prob < 0.4:
                trend = "BEARISH"
                trend_emoji = "📉"
            else:
                trend = "SIDEWAYS"
                trend_emoji = "↔️"
                
            confidence = abs(final_prob - 0.5) * 200  # Convert to percentage
                
            # Store probability breakdown for transparency
            probability_breakdown = {
                'technical': self.signals['technical'],
                'sentiment': self.signals['sentiment'],
                'volatility': self.signals['volatility'],
                'ml_model': {
                    'tft': float(tft_pred)
                }
            }
            
            # Generate option strategy
            option_strategy = self._generate_option_strategy(trend, confidence)
            
            # Get position size recommendation
            position_size = self.position_sizer.calculate_position_size(
                self.data, confidence, self.signals['volatility']['forecast']
            )
            
            # Compile prediction
            prediction = {
                'timestamp': datetime.now().isoformat(),
                'trend': trend,
                'trend_emoji': trend_emoji,
                'confidence': confidence,
                'current_price': latest['Close'],
                'signals': self.signals['technical'],
                'sentiment': self.signals['sentiment'],
                'volatility_forecast': self.signals['volatility']['forecast'],
                'option_strategy': option_strategy,
                'position_size': position_size,
                'probability_breakdown': probability_breakdown,
                'market_conditions': {
                    'india_vix': latest.get('IndiaVIX', 0)
                }
            }
            
            self.latest_prediction = prediction
            logger.info(f"Generated prediction: {trend} with {confidence:.1f}% confidence")
            
            return prediction
            
        except Exception as e:
            logger.error(f"Error generating prediction: {str(e)}")
            raise
            
    def _generate_option_strategy(self, trend: str, confidence: float) -> Dict:
        """
        Generate option strategy based on prediction
        
        Args:
            trend: Predicted trend (BULLISH/BEARISH/SIDEWAYS)
            confidence: Prediction confidence
            
        Returns:
            Dict containing strategy details
        """
        latest_price = self.data['Close'].iloc[-1]
        
        if trend == "BULLISH":
            if confidence > 75:
                return {
                    'name': 'Bull Call Spread',
                    'strikes': {
                        'buy': round(latest_price * 0.99, -2),  # ATM
                        'sell': round(latest_price * 1.02, -2)  # OTM
                    }
                }
            else:
                return {
                    'name': 'Long Call',
                    'strikes': {
                        'buy': round(latest_price, -2)  # ATM
                    }
                }
                
        elif trend == "BEARISH":
            if confidence > 75:
                return {
                    'name': 'Bear Put Spread',
                    'strikes': {
                        'buy': round(latest_price * 1.01, -2),  # ATM
                        'sell': round(latest_price * 0.98, -2)  # OTM
                    }
                }
            else:
                return {
                    'name': 'Long Put',
                    'strikes': {
                        'buy': round(latest_price, -2)  # ATM
                    }
                }
                
        else:  # SIDEWAYS
            return {
                'name': 'Iron Condor',
                'strikes': {
                    'call_credit_spread': {
                        'sell': round(latest_price * 1.02, -2),
                        'buy': round(latest_price * 1.03, -2)
                    },
                    'put_credit_spread': {
                        'sell': round(latest_price * 0.98, -2),
                        'buy': round(latest_price * 0.97, -2)
                    }
                }
            }
            
    def update_market_data(self, data: Dict) -> None:
        """
        Update real-time market data
        
        Args:
            data: Dictionary containing market update data
        """
        try:
            # Update order book
            if 'order_book' in data:
                self.order_book.update(OrderBookUpdate(**data['order_book']))
                
            # Update price series if available
            if 'price' in data:
                self.data.loc[pd.Timestamp.now()] = data['price']
                
            # Recalculate indicators
            self.calculate_indicators()
            
            logger.debug("Updated market data")
            
        except Exception as e:
            logger.error(f"Error updating market data: {str(e)}")
            raise

    def run_backtest(self, strategy_params: Optional[Dict] = None) -> pd.DataFrame:
        """
        Run strategy backtest
        
        Args:
            strategy_params: Optional strategy parameters to override defaults
            
        Returns:
            DataFrame with backtest results
        """
        try:
            # Initialize results
            results = pd.DataFrame(index=self.data.index)
            results['Returns'] = self.data['Returns']
            
            # Generate signals for each day
            signals = pd.Series(index=self.data.index, data=0)
            
            for i in range(len(self.data) - 1):
                # Use rolling window
                window = self.data.iloc[max(0, i-100):i+1]
                
                # Calculate indicators
                indicators = self.technical_analyzer.calculate_all(window)
                
                # Generate signal
                signal = self.technical_analyzer.generate_signals(indicators)
                signals.iloc[i] = signal['combined']
            
            # Calculate strategy returns
            results['strategy_returns'] = signals.shift(1) * results['Returns']
            
            # Calculate cumulative returns
            results['cum_strategy'] = (1 + results['strategy_returns']).cumprod()
            results['cum_buyhold'] = (1 + results['Returns']).cumprod()
            
            # Calculate metrics
            sharpe = np.sqrt(252) * results['strategy_returns'].mean() / results['strategy_returns'].std()
            max_dd = (results['cum_strategy'] / results['cum_strategy'].cummax() - 1).min()
            
            results['sharpe'] = sharpe
            results['max_drawdown'] = max_dd
            
            logger.info(f"Completed backtest - Sharpe: {sharpe:.2f}, Max DD: {max_dd:.2%}")
            
            return results
            
        except Exception as e:
            logger.error(f"Error running backtest: {str(e)}")
            raise
            
    def save_prediction(self, prediction: Dict) -> None:
        """
        Save prediction to database
        
        Args:
            prediction: Dictionary containing prediction details
        """
        try:
            db_path = Path("predictions/predictions.db")
            db_path.parent.mkdir(exist_ok=True)
            
            conn = sqlite3.connect(db_path)
            
            # Convert prediction to JSON for storage
            pred_json = json.dumps(prediction)
            
            # Save to database
            cursor = conn.cursor()
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS predictions
                (timestamp TEXT PRIMARY KEY, prediction TEXT)
            """)
            
            cursor.execute(
                "INSERT OR REPLACE INTO predictions (timestamp, prediction) VALUES (?, ?)",
                (prediction['timestamp'], pred_json)
            )
            
            conn.commit()
            conn.close()
            
            logger.info(f"Saved prediction for {prediction['timestamp']}")
            
        except Exception as e:
            logger.error(f"Error saving prediction: {str(e)}")
            raise

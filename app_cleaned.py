"""
Enhanced Nifty Option Prediction System
Core system implementation
"""

import warnings
import sqlite3
import os
import json
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Optional, Callable, Any

import numpy as np
import pandas as pd
import torch
import yfinance as yf
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import safetensors.torch

# Import required modules
from models.data_preparer import DataPreparer
from models.market_microstructure import OrderBookUpdate, OrderBook
from models.har_rv import HARRVModel
from models.config import (
    TFT_MODEL_PATH, HAR_RV_MODEL_PATH, TFT_CONFIG, HAR_RV_CONFIG
)
from models.technical_indicators import TechnicalAnalyzer
from utils.risk_management import RiskManager, PositionSizer
from utils.portfolio_optimizer import PortfolioOptimizer
from utils.visualizer import Visualizer

logger = logging.getLogger('nifty_predictor')
logger.setLevel(logging.INFO)

warnings.filterwarnings('ignore')

# Import optional modules and define availability flags
SENTIMENT_AVAILABLE = False
TFT_AVAILABLE = False

try:
    from models.market_sentiment import (
        MarketSentimentAnalyzer, NewsTradingSignalGenerator)
    SENTIMENT_AVAILABLE = True
except ImportError as e:
    logger.warning("Sentiment analysis disabled - %s", e)
    
    class MarketSentimentAnalyzer:  # noqa: E302
        def __init__(self):
            pass

        def analyze_sentiment(self, text):  # noqa: ARG002
            return 0.5

        def get_latest_sentiment(self):
            return {'score': 0.5}

    class NewsTradingSignalGenerator:  # noqa: E302
        def __init__(self):
            pass

        def generate_signal(self, score):  # noqa: ARG002
            return {'direction': 'NEUTRAL', 'strength': 0.5}

# Make TFT optional
try:
    from models.tft_model import TemporalFusionTransformerModel
    TFT_AVAILABLE = True
except ImportError as e:
    logger.warning("TFT model disabled - %s", e)
    TFT_AVAILABLE = False
    
    class TemporalFusionTransformerModel:  # noqa: E302
        def __init__(self, **kwargs):  # noqa: ARG002
            pass

        def load_model(self, path):  # noqa: ARG002
            pass

        def train(self, X, y):  # noqa: ARG002
            return None

        def predict(self, X):  # noqa: ARG002
            return pd.DataFrame({
                'q05': [0.4] * len(X),
                'q50': [0.5] * len(X),
                'q95': [0.6] * len(X)
            }, index=X.index)


def safe_load_model(model_path: str) -> Dict:
    """Safely load a PyTorch model using safetensors"""
    try:
        if model_path.endswith('.safetensors'):
            return safetensors.torch.load_file(model_path)
        else:
            # Convert to safetensors format first
            state_dict = torch.load(model_path, map_location='cpu')
            safe_path = model_path.replace('.pt', '.safetensors')
            safe_path = safe_path.replace('.pth', '.safetensors')
            safetensors.torch.save_file(state_dict, safe_path)
            return safetensors.torch.load_file(safe_path)
    except Exception as e:
        logger.error("Error loading model: %s", e)
        return {}


# Simple ensemble model for smoke_test.py
class MLEnsemble:
    """Simple ensemble classifier for demonstration (RandomForest)"""
    
    def __init__(self, n_estimators=10, random_state=42):
        self.model = RandomForestClassifier(
            n_estimators=n_estimators, random_state=random_state)
        self.trained = False
        self.last_score = None

    def train(self, X, y, test_size=0.2):
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42)
        self.model.fit(X_train, y_train)
        self.trained = True
        y_pred = self.model.predict(X_test)
        self.last_score = accuracy_score(y_test, y_pred)
        return {'accuracy': self.last_score}

    def predict(self, X):
        if not self.trained:
            raise RuntimeError("Model not trained yet.")
        return self.model.predict(X)

    def _print_summary(self):
        print(f"Ensemble model accuracy: {self.last_score}")


class EnhancedNiftyPredictionSystem:
    """Enhanced prediction system for Nifty options trading"""
    
    def __init__(self, spot_price: Optional[float] = None,
                 expiry_date: Optional[str] = None,
                 progress_callback: Optional[Callable] = None):
        """
        Initialize the prediction system

        Args:
            spot_price: Current Nifty spot price (optional)
            expiry_date: Target expiry date in YYYY-MM-DD format
            progress_callback: Optional callback function to report progress
        """
        self.spot_price = spot_price
        self.expiry_date = expiry_date
        self.data: Optional[pd.DataFrame] = None
        self.models: Dict[str, Any] = {}
        self.progress_callback = progress_callback
        self.signals: Dict[str, Any] = {}
        self.predictions: Dict[str, Any] = {}
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
            self.tft_model = TemporalFusionTransformerModel(
                **TFT_CONFIG
            )
            if TFT_MODEL_PATH.exists():
                try:
                    self.tft_model.load_model(str(TFT_MODEL_PATH))
                    logger.info("TFT model loaded securely")
                except Exception as e:
                    logger.warning("Could not load TFT model: %s", e)
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
                logger.warning("Could not load HAR-RV model: %s", e)
                self.har_rv_model = None
        else:
            logger.info("No pre-trained HAR-RV model found")
            
        self.risk_manager = RiskManager()
        # 1M initial capital
        self.position_sizer = PositionSizer(initial_capital=1000000)
        self.portfolio_optimizer = PortfolioOptimizer()
        self.order_book = OrderBook()
        self.viz = Visualizer()
        
        # Configure logging
        self._setup_logging()
        
        # Initialize data preparer
        self.data_preparer = DataPreparer()
        
        # Track training status
        self.is_trained = False
        
        logger.info("EnhancedNiftyPredictionSystem initialized")
        
    def _setup_logging(self):
        """Configure logging settings"""
        log_dir = Path("logs")
        log_dir.mkdir(exist_ok=True)
        
        fh = logging.FileHandler(
            log_dir / f"nifty_system_{datetime.now().strftime('%Y%m%d')}.log"
        )
        fh.setLevel(logging.INFO)
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        fh.setFormatter(formatter)
        logger.addHandler(fh)
    
    def load_data(self, days: int = 300, force_refresh: bool = False) -> None:
        """
        Load historical market data and initialize models
        
        Args:
            days: Number of historical days to fetch
            force_refresh: Force refresh cached data
        """
        if self.progress_callback:
            self.progress_callback("Loading market data...")
            
        cache_file = Path("data_cache") / f"nifty_data_{days}d.parquet"
        cache_file.parent.mkdir(exist_ok=True)
        
        if cache_file.exists() and not force_refresh:
            try:
                self.data = pd.read_parquet(cache_file)
                logger.info("Loaded cached data from %s", cache_file)
                if self.progress_callback:
                    self.progress_callback("Loaded cached market data")
            except Exception as e:
                logger.warning(
                    "Failed to load cached data: %s. Fetching fresh data...", e
                )
                force_refresh = True
            
        # Fetch data
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)
        
        try:
            logger.info(
                "Fetching Nifty data from %s to %s", start_date, end_date
            )
            nifty = yf.download(
                "^NSEI", start=start_date, end=end_date, progress=False
            )
            if nifty.empty:
                raise ValueError("No Nifty data received from Yahoo Finance")
            
            logger.info("Fetching VIX data")
            vix = yf.download(
                "^INDIAVIX", start=start_date, end=end_date, progress=False
            )
            
            # Clean and preprocess data
            logger.info("Preprocessing data...")
            # Forward fill missing values
            nifty = nifty.ffill()
            vix = vix.ffill()
            
            # Backward fill any remaining NaN values at the start
            nifty = nifty.bfill()
            vix = vix.bfill()
            
            # Ensure no NaN values remain
            if nifty.isna().any().any() or vix.isna().any().any():
                raise ValueError(
                    "Unable to clean all NaN values from the data"
                )
            if vix.empty:
                raise ValueError("No VIX data received from Yahoo Finance")
            
            if self.progress_callback:
                self.progress_callback("Processing market data...")
            
            # Process and merge data
            logger.info("Processing market data")
            # Reindex VIX data to match Nifty dates
            vix = vix.reindex(nifty.index)
            vix = vix.ffill().bfill()
            # Ensure all columns are 1D
            data = pd.DataFrame({
                'Close': nifty['Close'].squeeze(),
                'High': nifty['High'].squeeze(),
                'Low': nifty['Low'].squeeze(),
                'Volume': nifty['Volume'].squeeze(),
                'IndiaVIX': vix['Close'].squeeze()
            }, index=nifty.index)
            # Drop any rows with NaN values
            data = data.dropna()
            
            # Calculate returns
            data['Returns'] = data['Close'].pct_change()
            
            # Save to cache
            data.to_parquet(cache_file)
            
            self.data = data
            logger.info("Fetched and cached fresh data for %d days", days)
            
            # Calculate technical indicators
            if self.progress_callback:
                self.progress_callback("Calculating technical indicators...")
            self.calculate_indicators()
            
            # Train models on fresh data
            if self.progress_callback:
                self.progress_callback("Training prediction models...")
            self.train_ml_models()
            
            if self.progress_callback:
                self.progress_callback("Data processing complete")
            
        except Exception as e:
            logger.error("Error loading data: %s", str(e))
            raise
            
    def calculate_indicators(self) -> None:
        """Calculate technical indicators"""
        if self.data is None:
            raise ValueError("No data loaded. Call load_data() first.")
        try:
            if (hasattr(self, 'progress_callback') and 
                self.progress_callback):
                self.progress_callback("Calculating technical indicators...")
            indicators_df = self.technical_analyzer.calculate_all(self.data)
            if (isinstance(indicators_df, pd.DataFrame) and 
                not indicators_df.empty):
                new_cols = [
                    col for col in indicators_df.columns 
                    if col not in self.data.columns
                ]
                if new_cols:
                    self.data = pd.concat(
                        [self.data, indicators_df[new_cols]], axis=1
                    )
                    logger.info("Added %d technical indicators", len(new_cols))
                else:
                    logger.info("No new indicators to add")
            else:
                logger.warning("No indicators calculated - using data as-is")
            self.data = self.data.fillna(method='ffill').fillna(method='bfill')
            logger.info("Calculated technical indicators")
        except Exception as e:
            logger.error("Error calculating indicators: %s", str(e))
            # Don't raise - continue with available data
            logger.warning("Continuing without technical indicators")
            
    def generate_signals(self) -> None:
        """Generate trading signals from various sources"""
        try:
            if self.data is None:
                raise ValueError("No data loaded. Call load_data() first.")

            # Initialize with defaults
            self.signals = {
                'technical': {'combined': 0.5},
                'sentiment': {'score': 0.5},
                'volatility': {'forecast': 0.0}
            }

            # Technical signals
            try:
                tech_signals = self.technical_analyzer.generate_signals(
                    self.data
                )
                if tech_signals:
                    self.signals['technical'] = tech_signals
            except Exception as e:
                logger.warning("Technical signals failed: %s", e)

            # Sentiment signals
            if self.sentiment_analyzer:
                try:
                    sentiment_data = self.sentiment_analyzer.get_latest_sentiment()
                    if self.signal_generator:
                        sentiment_signals = self.signal_generator.generate_signals(
                            sentiment_data
                        )
                        if sentiment_signals:
                            self.signals['sentiment'] = sentiment_signals
                except Exception as e:
                    logger.warning("Sentiment signals failed: %s", e)

            # Volatility signals
            try:
                if (self.har_rv_model and 'Returns' in self.data.columns):
                    if (hasattr(self.har_rv_model, 'is_trained') and 
                        self.har_rv_model.is_trained):
                        vol_forecast = self.har_rv_model.forecast(
                            self.data['Returns']
                        )
                        if vol_forecast is not None:
                            self.signals['volatility'] = {
                                'forecast': vol_forecast
                            }
                    else:
                        logger.info(
                            "HAR-RV model not trained yet, using default volatility"
                        )
            except Exception as e:
                logger.warning("Volatility forecast failed: %s", e)

            logger.info("Generated signals: %s", self.signals)
        except Exception as e:
            logger.error("Error generating signals: %s", str(e))
            raise
            
    def train_ml_models(self) -> None:
        """Train machine learning models with NaN handling"""
        try:
            if self.data is None:
                raise ValueError("No data loaded. Call load_data() first.")
            
            # Clean data
            X = self.data.copy()
            X = X.fillna(method='ffill').fillna(method='bfill').fillna(0)
            
            # Prepare features for TFT
            y = np.where(X['Returns'].shift(-1) > 0, 1, 0)  # Next day direction
            
            # Drop any remaining NaN rows
            valid_mask = ~(pd.isna(y) | X.isna().any(axis=1))
            X = X[valid_mask]
            y = y[valid_mask]
            
            if len(X) < 50:  # Minimum required rows
                raise ValueError(
                    "Insufficient clean data for training after NaN handling"
                )
            
            logger.info(
                "Training with %d clean data points after NaN handling", len(X)
            )
            
            # Train TFT model if available
            if self.tft_model and TFT_AVAILABLE:
                self.models['tft'] = self.tft_model.train(X, y)
                # Save TFT model securely
                self.tft_model.save_model(str(TFT_MODEL_PATH))
                logger.info("Trained and saved TFT model")
            else:
                logger.info("Skipping TFT model training - not available")
                
            # Train HAR-RV model with clean data
            if self.har_rv_model:
                clean_returns = X['Returns'].copy()
                try:
                    self.har_rv_model.train(clean_returns)
                    # Save HAR-RV model securely
                    self.har_rv_model.save_model(str(HAR_RV_MODEL_PATH))
                    logger.info("Trained and saved HAR-RV model")
                except Exception as har_error:
                    logger.warning("HAR-RV model training failed: %s", har_error)
                    # Continue without HAR-RV model
            else:
                logger.warning("HAR-RV model not initialized")
            
        except Exception as e:
            logger.error("Error training models: %s", str(e))
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
                
            if self.progress_callback:
                self.progress_callback("Generating prediction...")
            
            # Calculate latest indicators if needed
            if not self.signals:
                self.calculate_indicators()
                self.generate_signals()
            
            # Get latest data point
            latest = self.data.iloc[-1].to_dict()
            
            # Generate TFT prediction if available
            if self.tft_model and TFT_AVAILABLE:
                try:
                    if self.progress_callback:
                        self.progress_callback("Running TFT model prediction...")
                    tft_predictions = self.tft_model.predict(self.data)
                    tft_pred = tft_predictions['q50'].iloc[-1]
                    # Remove unused variable warning
                    tft_uncertainty = {
                        'lower': float(tft_predictions['q05'].iloc[-1]),
                        'upper': float(tft_predictions['q95'].iloc[-1])
                    }
                    _ = tft_uncertainty  # Mark as used
                except Exception as e:
                    logger.warning("TFT prediction failed: %s", e)
                    tft_pred = 0.5
            else:
                tft_pred = 0.5
            
            if self.progress_callback:
                self.progress_callback("Combining prediction signals...")
            
            # Get technical signals with default if missing
            tech_signal = self.signals.get('technical', {}).get('combined', 0.5)
            
            # Get sentiment signal with default
            sent_signal = self.signals.get('sentiment', {}).get('score', 0.5)
            
            # Get volatility signal with default
            vol_signal = self.signals.get('volatility', {}).get('forecast', 0)
            vol_signal = 0.5 + (vol_signal if vol_signal is not None else 0)
            
            # Combine signals with weights
            signal_weights = {
                'technical': 0.4,
                'sentiment': 0.3,
                'volatility': 0.3
            }
            
            final_prob = (
                signal_weights['technical'] * tech_signal +
                signal_weights['sentiment'] * sent_signal +
                signal_weights['volatility'] * vol_signal
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
                self.data, confidence, 
                self.signals['volatility']['forecast']
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
            logger.info(
                "Generated prediction: %s with %.1f%% confidence", 
                trend, confidence
            )
            
            return prediction
            
        except Exception as e:
            logger.error("Error generating prediction: %s", str(e))
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
        if self.data is None:
            raise ValueError("No data available for strategy generation")
            
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
            if 'price' in data and self.data is not None:
                self.data.loc[pd.Timestamp.now()] = data['price']
                
            # Recalculate indicators
            self.calculate_indicators()
            
            logger.debug("Updated market data")
            
        except Exception as e:
            logger.error("Error updating market data: %s", str(e))
            raise

    def run_backtest(self, strategy_params: Optional[Dict] = None) -> pd.DataFrame:  # noqa: ARG002
        """
        Run strategy backtest
        
        Args:
            strategy_params: Optional strategy parameters to override defaults
            
        Returns:
            DataFrame with backtest results
        """
        try:
            if self.data is None:
                raise ValueError("No data available for backtesting")
                
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
            sharpe = (np.sqrt(252) * results['strategy_returns'].mean() / 
                     results['strategy_returns'].std())
            max_dd = ((results['cum_strategy'] / 
                     results['cum_strategy'].cummax() - 1).min())
            
            results['sharpe'] = sharpe
            results['max_drawdown'] = max_dd
            
            logger.info(
                "Completed backtest - Sharpe: %.2f, Max DD: %.2f%%", 
                sharpe, max_dd
            )
            
            return results
            
        except Exception as e:
            logger.error("Error running backtest: %s", str(e))
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
            
            logger.info("Saved prediction for %s", prediction['timestamp'])
            
        except Exception as e:
            logger.error("Error saving prediction: %s", str(e))
            raise

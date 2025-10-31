"""
Technical Analysis Indicators Module
"""

import pandas as pd
import numpy as np
from typing import Dict

class TechnicalIndicators:
    """Technical analysis indicator calculator and signal generator"""
    
    def __init__(self):
        """Initialize technical analyzer"""
        self.indicators = {}
    
    def calculate_all(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate all technical indicators
        
        Args:
            data: DataFrame with OHLCV data
            
        Returns:
            DataFrame with indicator columns
            
        Raises:
            ValueError: If data is empty, missing required columns, has insufficient points,
                      or contains invalid values
        """
        if not isinstance(data, pd.DataFrame):
            raise ValueError("Input must be a pandas DataFrame")
            
        if data.empty:
            raise ValueError("Cannot calculate indicators on empty data")
            
        required_cols = ['Close', 'High', 'Low', 'Volume']
        missing_cols = [col for col in required_cols if col not in data.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
            
        if len(data) < 50:
            raise ValueError(f"Insufficient data points: {len(data)}. Need at least 50.")
            
        # Check for NaN values
        if data[required_cols].isnull().any().any():
            raise ValueError("Cannot calculate indicators: data contains NaN values")
            
        df = data.copy()
        
        try:
            # Moving averages
            df['SMA_20'] = df['Close'].rolling(window=20).mean()
            df['SMA_50'] = df['Close'].rolling(window=50).mean()
            df['EMA_12'] = df['Close'].ewm(span=12, adjust=False).mean()
            df['EMA_26'] = df['Close'].ewm(span=26, adjust=False).mean()
            
            # MACD
            df['MACD'] = df['EMA_12'] - df['EMA_26']
            df['Signal_Line'] = df['MACD'].ewm(span=9, adjust=False).mean()
            df['MACD_Hist'] = df['MACD'] - df['Signal_Line']
        except Exception as e:
            raise ValueError(f"Error calculating moving averages and MACD: {str(e)}")
        
        # RSI
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['RSI'] = 100 - (100 / (1 + rs))
        
        # Bollinger Bands
        df['BB_Middle'] = df['Close'].rolling(window=20).mean()
        bb_std = df['Close'].rolling(window=20).std()
        df['BB_Upper'] = df['BB_Middle'] + (bb_std * 2)
        df['BB_Lower'] = df['BB_Middle'] - (bb_std * 2)
        
        # ATR
        high_low = df['High'] - df['Low']
        high_close = abs(df['High'] - df['Close'].shift())
        low_close = abs(df['Low'] - df['Close'].shift())
        ranges = pd.concat([high_low, high_close, low_close], axis=1)
        true_range = ranges.max(axis=1)
        df['ATR'] = true_range.rolling(window=14).mean()
        
        # Store calculated indicators
        self.indicators = df
        
        return df
    
    def generate_signals(self, data: pd.DataFrame = None) -> Dict:
        """
        Generate trading signals from indicators
        
        Args:
            data: Optional DataFrame with indicators, uses stored indicators if None
            
        Returns:
            Dictionary with signal values
        """
        try:
            df = data if data is not None else self.indicators
            
            if df is None or df.empty:
                raise ValueError("No data available for signal generation")
            
            required_indicators = ['MACD', 'Signal_Line', 'MACD_Hist', 'RSI', 'SMA_20', 'SMA_50', 'Close', 'BB_Upper', 'BB_Lower']
            missing = [ind for ind in required_indicators if ind not in df.columns]
            if missing:
                raise ValueError(f"Missing required indicators: {missing}")
            
            signals = {}
            
            try:
                # MACD Signal (-1 to 1)
                macd_hist = float(df['MACD_Hist'].iloc[-1])
                close_price = float(df['Close'].iloc[-1])
                macd_gt_signal = df['MACD'].iloc[-1] > df['Signal_Line'].iloc[-1]
                
                signals['MACD'] = min(1, abs(macd_hist / close_price * 100)) if macd_gt_signal else max(-1, -abs(macd_hist / close_price * 100))
            except (ValueError, TypeError, KeyError) as e:
                signals['MACD'] = 0.0  # Default to neutral on error
            
            # RSI Signal (-1 to 1)
            try:
                rsi = float(df['RSI'].iloc[-1])
                if rsi > 70:
                    signals['RSI'] = -1.0
                elif rsi < 30:
                    signals['RSI'] = 1.0
                else:
                    signals['RSI'] = 0.0
            except (ValueError, TypeError) as e:
                signals['RSI'] = 0.0  # Default to neutral on error
            
            # Moving Average Signal (-1 to 1)
            try:
                ma_diff = (df['SMA_20'].iloc[-1] - df['SMA_50'].iloc[-1]) / df['Close'].iloc[-1]
                signals['MA_Cross'] = min(1, max(-1, ma_diff * 100))
            except (ValueError, TypeError, KeyError) as e:
                signals['MA_Cross'] = 0.0  # Default to neutral on error
            
            # Bollinger Bands Signal (-1 to 1)
            try:
                bb_position = (df['Close'].iloc[-1] - df['BB_Lower'].iloc[-1]) / (df['BB_Upper'].iloc[-1] - df['BB_Lower'].iloc[-1])
                signals['BB'] = min(1, max(-1, (bb_position - 0.5) * 2))
            except (ValueError, TypeError, KeyError, ZeroDivisionError) as e:
                signals['BB'] = 0.0  # Default to neutral on error
            
            # Combined Signal
            weights = {
                'MACD': 0.3,
                'RSI': 0.3,
                'MA_Cross': 0.2,
                'BB': 0.2
            }
            
            signals['combined'] = sum(signal * weights[name] for name, signal in signals.items())
            
            return signals
        except Exception as e:
            raise ValueError(f"Error generating signals: {str(e)}")
"""
Technical Analysis Indicators Module
"""

import pandas as pd
import numpy as np
from typing import Dict

class TechnicalAnalyzer:
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
            DataFrame with calculated indicators
        """
        df = data.copy()
        
        # Moving averages
        df['SMA_20'] = df['Close'].rolling(window=20).mean()
        df['SMA_50'] = df['Close'].rolling(window=50).mean()
        df['EMA_12'] = df['Close'].ewm(span=12, adjust=False).mean()
        df['EMA_26'] = df['Close'].ewm(span=26, adjust=False).mean()
        
        # MACD
        df['MACD'] = df['EMA_12'] - df['EMA_26']
        df['Signal_Line'] = df['MACD'].ewm(span=9, adjust=False).mean()
        df['MACD_Hist'] = df['MACD'] - df['Signal_Line']
        
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
        df = data if data is not None else self.indicators
        
        signals = {}
        
        # MACD Signal (-1 to 1)
        signals['MACD'] = np.where(
            df['MACD'] > df['Signal_Line'],
            min(1, abs(df['MACD_Hist'].iloc[-1] / df['Close'].iloc[-1] * 100)),
            max(-1, -abs(df['MACD_Hist'].iloc[-1] / df['Close'].iloc[-1] * 100))
        )[-1]
        
        # RSI Signal (-1 to 1)
        rsi = df['RSI'].iloc[-1]
        if rsi > 70:
            signals['RSI'] = -1
        elif rsi < 30:
            signals['RSI'] = 1
        else:
            signals['RSI'] = 0
            
        # Moving Average Signal (-1 to 1)
        ma_diff = (df['SMA_20'].iloc[-1] - df['SMA_50'].iloc[-1]) / df['Close'].iloc[-1]
        signals['MA_Cross'] = min(1, max(-1, ma_diff * 100))
        
        # Bollinger Bands Signal (-1 to 1)
        bb_position = (df['Close'].iloc[-1] - df['BB_Lower'].iloc[-1]) / (df['BB_Upper'].iloc[-1] - df['BB_Lower'].iloc[-1])
        signals['BB'] = min(1, max(-1, (bb_position - 0.5) * 2))
        
        # Combined Signal
        weights = {
            'MACD': 0.3,
            'RSI': 0.3,
            'MA_Cross': 0.2,
            'BB': 0.2
        }
        
        signals['combined'] = sum(signal * weights[name] for name, signal in signals.items())
        
        return signals
"""Technical analysis indicators and signal generation"""
import pandas as pd
import numpy as np
from typing import Dict

class TechnicalAnalyzer:
    """Technical analysis and signal generation"""
    
    def __init__(self):
        self.signals = {}
        self.last_update = None

    def calculate_all(self, data: pd.DataFrame) -> pd.DataFrame:
        """Calculate technical indicators - MUST RETURN DATAFRAME"""
        indicators = pd.DataFrame(index=data.index)
        try:
            # Moving averages
            indicators['SMA20'] = data['Close'].rolling(window=20).mean()
            indicators['SMA50'] = data['Close'].rolling(window=50).mean()
            indicators['EMA20'] = data['Close'].ewm(span=20, adjust=False).mean()
            # RSI
            delta = data['Close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            indicators['RSI'] = 100 - (100 / (1 + rs))
            # MACD
            exp1 = data['Close'].ewm(span=12, adjust=False).mean()
            exp2 = data['Close'].ewm(span=26, adjust=False).mean()
            indicators['MACD'] = exp1 - exp2
            indicators['Signal_Line'] = indicators['MACD'].ewm(span=9, adjust=False).mean()
            # Bollinger Bands
            indicators['BB_middle'] = data['Close'].rolling(window=20).mean()
            bb_std = data['Close'].rolling(window=20).std()
            indicators['BB_upper'] = indicators['BB_middle'] + (bb_std * 2)
            indicators['BB_lower'] = indicators['BB_middle'] - (bb_std * 2)
            # ATR
            high_low = data['High'] - data['Low']
            high_close = abs(data['High'] - data['Close'].shift())
            low_close = abs(data['Low'] - data['Close'].shift())
            ranges = pd.concat([high_low, high_close, low_close], axis=1)
            true_range = ranges.max(axis=1)
            indicators['ATR'] = true_range.rolling(window=14).mean()
            # ADX
            indicators['ADX'] = self._calculate_adx(data)
            return indicators
        except Exception as e:
            print(f"Error in calculate_all: {e}")
            return pd.DataFrame(index=data.index)

    def _calculate_adx(self, data: pd.DataFrame, period: int = 14) -> pd.Series:
        try:
            high = data['High']
            low = data['Low']
            close = data['Close']
            up_move = high.diff()
            down_move = -low.diff()
            plus_dm = pd.Series(0.0, index=data.index)
            minus_dm = pd.Series(0.0, index=data.index)
            plus_dm[(up_move > down_move) & (up_move > 0)] = up_move
            minus_dm[(down_move > up_move) & (down_move > 0)] = down_move
            high_low = high - low
            high_close = abs(high - close.shift())
            low_close = abs(low - close.shift())
            ranges = pd.concat([high_low, high_close, low_close], axis=1)
            atr = ranges.max(axis=1).rolling(window=period).mean()
            plus_di = 100 * (plus_dm.rolling(window=period).mean() / atr)
            minus_di = 100 * (minus_dm.rolling(window=period).mean() / atr)
            dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di)
            adx = dx.rolling(window=period).mean()
            return adx
        except:
            return pd.Series(0.0, index=data.index)

    def generate_signals(self, data: pd.DataFrame = None) -> Dict:
        if data is None or len(data) < 50:
            return {'combined': 0.5}
        try:
            last = data.iloc[-1]
            signals = {}
            if 'MACD' in last and 'Signal_Line' in last:
                macd_val = last.get('MACD', 0)
                signal_val = last.get('Signal_Line', 0)
                if pd.notna(macd_val) and pd.notna(signal_val):
                    signals['MACD'] = 1 if macd_val > signal_val else -1
                else:
                    signals['MACD'] = 0
            else:
                signals['MACD'] = 0
            if 'RSI' in last:
                rsi = last.get('RSI', 50)
                if pd.notna(rsi):
                    if rsi > 70:
                        signals['RSI'] = -1
                    elif rsi < 30:
                        signals['RSI'] = 1
                    else:
                        signals['RSI'] = 0
                else:
                    signals['RSI'] = 0
            else:
                signals['RSI'] = 0
            if 'SMA20' in last and 'SMA50' in last:
                sma20 = last.get('SMA20', 0)
                sma50 = last.get('SMA50', 0)
                if pd.notna(sma20) and pd.notna(sma50) and sma50 != 0:
                    ma_diff = (sma20 - sma50) / sma50
                    signals['MA_Cross'] = min(1, max(-1, ma_diff * 100))
                else:
                    signals['MA_Cross'] = 0
            else:
                signals['MA_Cross'] = 0
            weights = {'MACD': 0.4, 'RSI': 0.3, 'MA_Cross': 0.3}
            signals['combined'] = sum(signals.get(name, 0) * weight 
                                     for name, weight in weights.items())
            self.signals = signals
            return signals
        except Exception as e:
            print(f"Error generating signals: {e}")
            return {'combined': 0.5}
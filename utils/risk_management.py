"""Portfolio optimization and risk management utilities"""
from typing import Dict, Optional
import numpy as np
import pandas as pd

class RiskManager:
    """Risk management system"""
    def __init__(self, max_drawdown: float = 0.1, max_var: float = 0.02):
        self.max_drawdown = max_drawdown
        self.max_var = max_var

    def calculate_position_size(self, price: float, volatility: float, account_value: float) -> float:
        """Calculate safe position size based on risk parameters"""
        # Kelly criterion with safety fraction
        vol_adj = min(volatility, 0.5)  # Cap volatility
        kelly_fraction = 0.5 * (1 - vol_adj)  # Conservative Kelly
        
        # Position size considering max drawdown
        max_position = account_value * self.max_drawdown
        suggested_size = account_value * kelly_fraction
        
        return min(max_position, suggested_size)

class PositionSizer:
    """Position sizing system"""
    def __init__(self, initial_capital: float = 1000000):
        self.initial_capital = initial_capital
        self.current_capital = initial_capital

    def calculate_position_size(self, data: pd.DataFrame, confidence: float, volatility: float) -> Dict:
        """Calculate recommended position size"""
        # Basic volatility adjustment
        vol_adj = max(0.1, min(1.0, 1 - volatility))
        
        # Confidence adjustment
        conf_adj = confidence / 100  # Convert confidence to 0-1 scale
        
        # Calculate base position as percentage of capital
        base_pct = 0.02  # 2% base position
        adjusted_pct = base_pct * vol_adj * conf_adj
        
        # Calculate actual position size
        position_value = self.current_capital * adjusted_pct
        current_price = data['Close'].iloc[-1]
        
        return {
            'value': position_value,
            'units': position_value / current_price,
            'capital_pct': adjusted_pct * 100
        }
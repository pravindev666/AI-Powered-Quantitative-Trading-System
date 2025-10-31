"""
Portfolio Optimization Module
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple
from scipy.optimize import minimize

class PortfolioOptimizer:
    """Portfolio optimization using modern portfolio theory"""
    
    def __init__(self, risk_free_rate: float = 0.05):
        """
        Initialize portfolio optimizer
        
        Args:
            risk_free_rate: Annual risk-free rate
        """
        self.risk_free_rate = risk_free_rate
        
    def optimize_weights(
        self, 
        returns: pd.DataFrame, 
        target_return: float = None,
        max_weight: float = 1.0
    ) -> Dict:
        """
        Optimize portfolio weights using mean-variance optimization
        
        Args:
            returns: DataFrame of asset returns
            target_return: Optional target return constraint
            max_weight: Maximum weight for any single asset
            
        Returns:
            Dictionary containing optimized weights and metrics
        """
        n_assets = len(returns.columns)
        
        # Calculate mean returns and covariance
        mean_returns = returns.mean()
        cov_matrix = returns.cov()
        
        # Define optimization constraints
        constraints = [
            {'type': 'eq', 'fun': lambda x: np.sum(x) - 1},  # Weights sum to 1
        ]
        
        if target_return is not None:
            constraints.append({
                'type': 'eq',
                'fun': lambda x: np.sum(x * mean_returns) - target_return
            })
            
        # Asset weight bounds
        bounds = tuple((0, max_weight) for _ in range(n_assets))
        
        # Initial guess
        init_weights = np.array([1/n_assets] * n_assets)
        
        # Minimize portfolio variance
        result = minimize(
            lambda w: self._portfolio_volatility(w, cov_matrix),
            init_weights,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints
        )
        
        if not result.success:
            raise ValueError("Portfolio optimization failed")
            
        # Calculate portfolio metrics
        weights = result.x
        ret = self._portfolio_return(weights, mean_returns)
        vol = self._portfolio_volatility(weights, cov_matrix)
        sharpe = self._sharpe_ratio(ret, vol)
        
        return {
            'weights': dict(zip(returns.columns, weights)),
            'return': float(ret),
            'volatility': float(vol),
            'sharpe_ratio': float(sharpe)
        }
        
    def efficient_frontier(
        self,
        returns: pd.DataFrame,
        points: int = 50
    ) -> pd.DataFrame:
        """
        Generate efficient frontier points
        
        Args:
            returns: DataFrame of asset returns
            points: Number of points to calculate
            
        Returns:
            DataFrame with efficient frontier points
        """
        # Get return range
        mean_returns = returns.mean()
        min_ret = self._portfolio_return(
            [1 if i == mean_returns.argmin() else 0 for i in range(len(mean_returns))],
            mean_returns
        )
        max_ret = self._portfolio_return(
            [1 if i == mean_returns.argmax() else 0 for i in range(len(mean_returns))],
            mean_returns
        )
        
        target_returns = np.linspace(min_ret, max_ret, points)
        efficient_portfolios = []
        
        for target in target_returns:
            try:
                result = self.optimize_weights(returns, target_return=target)
                efficient_portfolios.append({
                    'return': result['return'],
                    'volatility': result['volatility'],
                    'sharpe_ratio': result['sharpe_ratio']
                })
            except:
                continue
                
        return pd.DataFrame(efficient_portfolios)
    
    def _portfolio_return(self, weights: np.ndarray, returns: pd.Series) -> float:
        """Calculate portfolio return"""
        return np.sum(weights * returns)
    
    def _portfolio_volatility(self, weights: np.ndarray, cov_matrix: pd.DataFrame) -> float:
        """Calculate portfolio volatility"""
        return np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
    
    def _sharpe_ratio(self, ret: float, vol: float) -> float:
        """Calculate Sharpe ratio"""
        return (ret - self.risk_free_rate) / vol
        
    def get_risk_contributions(
        self,
        weights: Dict[str, float],
        returns: pd.DataFrame
    ) -> Dict[str, float]:
        """
        Calculate risk contribution of each asset
        
        Args:
            weights: Dictionary of asset weights
            returns: DataFrame of asset returns
            
        Returns:
            Dictionary of risk contributions
        """
        w = np.array(list(weights.values()))
        assets = list(weights.keys())
        cov = returns[assets].cov()
        
        # Portfolio volatility
        port_vol = self._portfolio_volatility(w, cov)
        
        # Marginal risk contributions
        mrc = np.dot(cov, w)
        
        # Component risk contributions
        crc = w * mrc / port_vol
        
        return dict(zip(assets, crc))
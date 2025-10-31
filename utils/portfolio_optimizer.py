"""Portfolio optimization utilities"""
import numpy as np
from typing import Dict, List, Optional
from scipy.optimize import minimize

class PortfolioOptimizer:
    """Portfolio optimization using modern portfolio theory"""
    
    def __init__(self, risk_free_rate: float = 0.05):
        self.risk_free_rate = risk_free_rate
    
    def optimize_weights(self, returns: np.ndarray, target_return: Optional[float] = None) -> Dict:
        """
        Optimize portfolio weights using mean-variance optimization
        
        Args:
            returns: Array of asset returns
            target_return: Optional target return constraint
        
        Returns:
            Dict containing optimized weights and metrics
        """
        n_assets = returns.shape[1]
        
        # Calculate mean returns and covariance
        mu = np.mean(returns, axis=0)
        sigma = np.cov(returns.T)
        
        # Define objective function (minimize variance)
        def portfolio_var(w):
            return w.T @ sigma @ w
            
        # Constraints
        constraints = [
            {'type': 'eq', 'fun': lambda w: np.sum(w) - 1}  # Weights sum to 1
        ]
        
        if target_return is not None:
            constraints.append(
                {'type': 'eq', 'fun': lambda w: w.T @ mu - target_return}
            )
            
        # Asset bounds (0-1)
        bounds = tuple((0, 1) for _ in range(n_assets))
        
        # Initial guess (equal weights)
        w0 = np.ones(n_assets) / n_assets
        
        # Optimize
        result = minimize(
            portfolio_var, 
            w0,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints
        )
        
        if not result.success:
            raise RuntimeError(f"Optimization failed: {result.message}")
            
        # Calculate portfolio metrics
        weights = result.x
        port_return = weights.T @ mu
        port_vol = np.sqrt(portfolio_var(weights))
        sharpe = (port_return - self.risk_free_rate) / port_vol
        
        return {
            'weights': weights.tolist(),
            'return': float(port_return),
            'volatility': float(port_vol),
            'sharpe_ratio': float(sharpe)
        }
import numpy as np
import pandas as pd
from scipy.stats import norm
from scipy.optimize import minimize
from typing import Dict, Optional, Union, List
import warnings
import logging

logger = logging.getLogger(__name__)

def calculate_garch_volatility(returns: Union[pd.Series, np.ndarray], days_ahead: int = 1) -> Dict:
    """
    Calculate GARCH(1,1) volatility forecast
    
    Args:
        returns (pd.Series or np.ndarray): Historical returns series
        days_ahead (int): Number of days to forecast ahead
            
    Returns:
        dict: Dictionary containing:
            - forecast: The volatility forecast
            - parameters: Fitted GARCH parameters
            - convergence: Whether optimization converged
            - error: Error message if any
    """
    # Input validation
    if not isinstance(returns, (pd.Series, np.ndarray)):
        return {'error': 'Returns must be a pandas Series or numpy array'}
        
    if not isinstance(days_ahead, int) or days_ahead < 1:
        return {'error': 'days_ahead must be a positive integer'}
    
    try:
        # Clean and prepare data
        if isinstance(returns, pd.Series):
            r = returns.dropna().values
        else:
            r = np.asarray(returns)
            r = r[~np.isnan(r)]
            
        T = len(r)
        
        # Check data sufficiency
        if T < 50:
            return {'error': 'Insufficient data points (minimum 50 required)'}
            
        # Initial parameters with bounds
        var = np.var(r)
        if var <= 0 or not np.isfinite(var):
            return {'error': 'Invalid variance in returns data'}
            
        p0 = np.array([1e-6, 0.05, 0.9])  # omega, alpha, beta
        bounds = [(1e-12, None), (1e-6, 0.999), (1e-6, 0.999)]
        
        def negloglik(p):
            """Negative log-likelihood function for GARCH(1,1)"""
            try:
                omega, alpha, beta = p
                
                # Parameter validation
                if omega <= 0 or alpha <= 0 or beta <= 0:
                    return 1e10
                if alpha + beta >= 1:
                    return 1e10
                    
                sigma2 = np.empty(T)
                eps2 = r**2
                
                # Initial variance
                sigma2[0] = var
                
                # Calculate variance series
                for t in range(1, T):
                    sigma2[t] = omega + alpha * eps2[t-1] + beta * sigma2[t-1]
                    if sigma2[t] <= 0 or not np.isfinite(sigma2[t]):
                        return 1e10
                        
                # Log-likelihood calculation
                ll = -0.5 * (np.log(2*np.pi) + np.log(sigma2) + eps2 / sigma2)
                llsum = np.sum(ll)
                
                if not np.isfinite(llsum):
                    return 1e10
                    
                return -llsum
                
            except Exception:
                return 1e10
        
        # Optimize GARCH parameters
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            res = minimize(negloglik, p0, bounds=bounds, method='L-BFGS-B',
                         options={'maxiter': 5000, 'ftol': 1e-8})
            
        if not res.success:
            return {
                'error': f'Optimization failed: {res.message}',
                'convergence': False
            }
            
        # Extract parameters
        omega, alpha, beta = res.x
        
        # Validate final parameters
        if omega <= 0 or alpha <= 0 or beta <= 0:
            return {
                'error': 'Invalid optimal parameters',
                'convergence': False
            }
            
        if alpha + beta >= 1:
            return {
                'error': 'Non-stationary parameters (α + β ≥ 1)',
                'convergence': False
            }
        
        # Calculate final variance series
        sigma2 = np.empty(T)
        eps2 = r**2
        sigma2[0] = var
        
        for t in range(1, T):
            sigma2[t] = omega + alpha * eps2[t-1] + beta * sigma2[t-1]
        
        # Calculate multi-step forecast
        last_var = sigma2[-1]
        forecasts = []
        
        for i in range(days_ahead):
            last_var = omega + alpha * eps2[-1] + beta * last_var
            forecasts.append(np.sqrt(last_var))
        
        return {
            'forecast': forecasts[days_ahead-1],  # Final day forecast
            'forecast_series': forecasts,  # All forecasted days
            'parameters': {
                'omega': omega,
                'alpha': alpha,
                'beta': beta,
                'persistence': alpha + beta
            },
            'model_fit': {
                'sigma_series': np.sqrt(sigma2),
                'loglikelihood': -res.fun,
                'aic': 2*len(p0) - 2*(-res.fun)
            },
            'convergence': True,
            'error': None
        }
        
    except ImportError as e:
        return {
            'error': f'Required module not found: {str(e)}',
            'convergence': False
        }
    except Exception as e:
        return {
            'error': f'Error in calculation: {str(e)}',
            'convergence': False,
            'parameters': None,
            'forecast': None
        }
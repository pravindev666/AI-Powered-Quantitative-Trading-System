"""
HAR-RV (Heterogeneous Autoregressive Realized Volatility) Model Implementation
"""

import numpy as np
import pandas as pd
from typing import Dict, Optional, Tuple, Callable, Union
from scipy.optimize import minimize
from sklearn.preprocessing import StandardScaler
import joblib
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

class HARRVModel:
    """
    Heterogeneous Autoregressive Realized Volatility (HAR-RV) Model
    
    The HAR-RV model captures the long-memory property of volatility by incorporating
    volatility components calculated over different time horizons (daily, weekly, monthly).
    """
    
    def __init__(self, lags=[1, 5, 22], train_size=0.8, random_state=42):
        """Initialize HAR-RV model"""
        self.lags = lags
        self.train_size = train_size
        self.random_state = random_state
        self.params = None
        self.scaler = StandardScaler()
        self.is_fitted = False
        
    def calculate_realized_volatility(self, returns: pd.Series) -> pd.Series:
        """
        Calculate daily realized volatility
        
        Args:
            returns: Series of asset returns
            
        Returns:
            Series of realized volatility
        """
        return returns.pow(2).rolling(window=1).sum().pow(0.5)
    
    def prepare_features(self, returns: Union[pd.Series, pd.DataFrame, np.ndarray]) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Prepare HAR features from returns
        
        Args:
            returns: Series, DataFrame, or ndarray of asset returns
            
        Returns:
            Tuple of (X features, y target)
        """
        # Convert input to series
        if isinstance(returns, pd.DataFrame):
            returns = returns.iloc[:, 0]  # Take first column if DataFrame
        elif isinstance(returns, np.ndarray):
            returns = pd.Series(returns.ravel())  # Flatten if ndarray
        elif not isinstance(returns, pd.Series):
            returns = pd.Series(returns)
            
        # Calculate realized volatility
        rv = self.calculate_realized_volatility(returns)
        
        # Calculate HAR components
        rv_weekly = rv.rolling(window=5).mean()  # 5-day (weekly)
        rv_monthly = rv.rolling(window=22).mean()  # 22-day (monthly)
        
        # Prepare features
        X = pd.DataFrame({
            'rv_daily': rv.shift(1),
            'rv_weekly': rv_weekly.shift(1),
            'rv_monthly': rv_monthly.shift(1)
        }, index=rv.index)  # Ensure index is set
        
        # Target is next day's realized volatility
        y = rv
        
        # Remove NaN values
        mask = X.notna().all(axis=1) & y.notna()
        X = X[mask]
        y = y[mask]
        
        return X, y
    
    def train(self, returns: pd.Series) -> Dict:
        """
        Train HAR-RV model using OLS (alias for fit)
        
        Args:
            returns: Series of asset returns
            
        Returns:
            Dictionary of model parameters and metrics
        """
        return self.fit(returns)

    def fit(self, returns: pd.Series) -> Dict:
        """
        Fit HAR-RV model using OLS
        
        Args:
            returns: Series of asset returns
            
        Returns:
            Dictionary of model parameters and metrics
        """
        X, y = self.prepare_features(returns)
        
        # Standardize features
        X_scaled = self.scaler.fit_transform(X)
        
        # Define objective function for OLS
        def objective(params):
            beta0, beta1, beta2, beta3 = params
            y_pred = beta0 + beta1 * X_scaled[:, 0] + beta2 * X_scaled[:, 1] + beta3 * X_scaled[:, 2]
            return np.mean((y - y_pred) ** 2)
        
        # Initial guess
        x0 = np.array([0, 0.4, 0.3, 0.3])
        
        # Optimize
        result = minimize(
            objective,
            x0,
            method='L-BFGS-B',
            bounds=[(-np.inf, np.inf), (0, 1), (0, 1), (0, 1)],
            constraints={'type': 'eq', 'fun': lambda x: x[1] + x[2] + x[3] - 1}
        )
        
        if not result.success:
            raise ValueError("HAR-RV model fitting failed to converge")
            
        self.params = {
            'intercept': result.x[0],
            'beta_daily': result.x[1],
            'beta_weekly': result.x[2],
            'beta_monthly': result.x[3]
        }
        
        self.is_fitted = True
        
        # Calculate in-sample metrics
        y_pred = self.predict(returns)
        mse = np.mean((y[y_pred.index] - y_pred) ** 2)
        r2 = 1 - mse / np.var(y)
        
        return {
            'parameters': self.params,
            'mse': float(mse),
            'r2': float(r2)
        }
    
    def predict(self, returns: pd.Series) -> pd.Series:
        """
        Generate volatility predictions
        
        Args:
            returns: Series of asset returns
            
        Returns:
            Series of volatility predictions
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making predictions")
            
        X, _ = self.prepare_features(returns)
        X_scaled = self.scaler.transform(X)
        
        # Make predictions
        y_pred = (
            self.params['intercept'] +
            self.params['beta_daily'] * X_scaled[:, 0] +
            self.params['beta_weekly'] * X_scaled[:, 1] +
            self.params['beta_monthly'] * X_scaled[:, 2]
        )
        
        return pd.Series(y_pred, index=X.index)
    
    def forecast(self, returns: pd.Series, horizon: int = 1) -> float:
        """
        Generate volatility forecast
        
        Args:
            returns: Series of asset returns
            horizon: Forecast horizon in days
            
        Returns:
            Predicted volatility for next day
        """
        if not self.is_fitted:
            self.fit(returns)
            
        # Handle input types
        if isinstance(returns, pd.DataFrame):
            returns = returns.iloc[:, 0]  # Take first column if DataFrame
        if isinstance(returns, np.ndarray) and returns.ndim > 1:
            returns = returns.ravel()  # Flatten if ndarray
            
        try:
            # Get current HAR components
            rv = self.calculate_realized_volatility(returns)
            rv_daily = rv.iloc[-1]
            rv_weekly = rv.tail(5).mean()
            rv_monthly = rv.tail(22).mean()
            
            # Scale components
            components = np.array([[rv_daily, rv_weekly, rv_monthly]])
            components_scaled = self.scaler.transform(components)
            
            # Generate point forecast
            forecast = float(
                self.params['intercept'] +
                self.params['beta_daily'] * components_scaled[0, 0] +
                self.params['beta_weekly'] * components_scaled[0, 1] +
                self.params['beta_monthly'] * components_scaled[0, 2]
            )
            
            return max(0.0, forecast)  # Ensure non-negative volatility
            
        except Exception as e:
            logger.error(f"Error making volatility forecast: {str(e)}")
            return 0.0  # Return 0 on error
        
        # Generate point forecast
        forecast = (
            self.params['intercept'] +
            self.params['beta_daily'] * components_scaled[0, 0] +
            self.params['beta_weekly'] * components_scaled[0, 1] +
            self.params['beta_monthly'] * components_scaled[0, 2]
        )
        
        # Calculate forecast error variance
        error_var = np.var(returns.tail(22))  # Use recent variance as estimate
        
        return {
            'forecast': float(forecast),
            'lower_95': float(forecast - 1.96 * np.sqrt(error_var)),
            'upper_95': float(forecast + 1.96 * np.sqrt(error_var)),
            'components': {
                'daily': float(rv_daily),
                'weekly': float(rv_weekly),
                'monthly': float(rv_monthly)
            }
        }
    
    def decompose_variance(self, returns: pd.Series) -> Dict:
        """
        Decompose variance into HAR components
        
        Args:
            returns: Series of asset returns
            
        Returns:
            Dictionary with variance decomposition
        """
        if not self.is_fitted:
            self.fit(returns)
            
        X, _ = self.prepare_features(returns)
        X_scaled = self.scaler.transform(X)
        
        # Calculate component contributions
        daily_contrib = self.params['beta_daily'] * X_scaled[:, 0]
        weekly_contrib = self.params['beta_weekly'] * X_scaled[:, 1]
        monthly_contrib = self.params['beta_monthly'] * X_scaled[:, 2]
    
    def save_model(self, path: str) -> None:
        """Save model state securely"""
        if not self.is_fitted:
            raise ValueError("Cannot save unfitted model")
            
        try:
            path = Path(path)
            if not path.parent.exists():
                path.parent.mkdir(parents=True, exist_ok=True)
                
            model_state = {
                'params': self.params,
                'scaler': self.scaler,
                'is_fitted': self.is_fitted,
                'lags': self.lags,
                'train_size': self.train_size,
                'random_state': self.random_state
            }
            
            joblib.dump(model_state, path)
            logger.info(f"Model saved to {path}")
            
        except Exception as e:
            logger.error(f"Error saving model: {e}")
            raise
    
    def load_model(self, path: str, progress_callback: Callable[[str], None] = None) -> None:
        """Load model state securely with progress updates"""
        try:
            if progress_callback:
                progress_callback(f"Loading HAR-RV model from {path}")
                
            model_state = joblib.load(path)
            
            if progress_callback:
                progress_callback("Loading model parameters")
                
            self.params = model_state['params']
            self.scaler = model_state['scaler']
            
            if progress_callback:
                progress_callback("HAR-RV model loaded successfully")
            self.is_fitted = model_state['is_fitted']
            self.lags = model_state['lags']
            self.train_size = model_state['train_size']
            self.random_state = model_state['random_state']
            
            logger.info(f"Model loaded from {path}")
            
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            raise
        # TODO: Implement relative importance calculation if needed
import numpy as np
import pandas as pd
from scipy.stats import norm, t
from typing import Dict, List, Tuple, Optional, Union
from datetime import datetime, timedelta
import warnings
import yfinance as yf
import json
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

class RiskMetrics:
    def __init__(self, returns: pd.Series):
        self.returns = returns
        self.calculate_metrics()
        
    def calculate_metrics(self):
        """Calculate all risk metrics"""
        # Convert Series to numpy array for calculations
        returns_array = self.returns.values
        
        self.mean_return = np.mean(returns_array)
        self.std_return = np.std(returns_array)
        
        # VaR and CVaR
        self.var_95 = self._calculate_var(0.95)
        self.var_99 = self._calculate_var(0.99)
        self.cvar_95 = self._calculate_cvar(0.95)
        self.cvar_99 = self._calculate_cvar(0.99)
        
        # Drawdown metrics
        self.max_drawdown = self._calculate_max_drawdown()
        
        # Ratios
        self.sharpe_ratio = self._calculate_sharpe_ratio()
        self.sortino_ratio = self._calculate_sortino_ratio()
        self.calmar_ratio = self._calculate_calmar_ratio()
        
    def _calculate_var(self, confidence: float) -> float:
        """Calculate Value at Risk"""
        returns_array = self.returns.values
        return -np.percentile(returns_array, 100 * (1 - confidence))
        
    def _calculate_cvar(self, confidence: float) -> float:
        """Calculate Conditional Value at Risk"""
        returns_array = self.returns.values
        var = self._calculate_var(confidence)
        return -np.mean(returns_array[returns_array <= -var])
        
    def _calculate_max_drawdown(self) -> float:
        """Calculate Maximum Drawdown"""
        returns_array = self.returns.values
        cum_returns = (1 + returns_array).cumprod()
        rolling_max = np.maximum.accumulate(cum_returns)
        drawdowns = cum_returns / rolling_max - 1
        return np.min(drawdowns)
        
    def _calculate_sharpe_ratio(self, risk_free_rate: float = 0.02) -> float:
        """Calculate Sharpe Ratio"""
        returns_array = self.returns.values
        excess_returns = returns_array - risk_free_rate/252
        if self.std_return <= 1e-10:  # Handle near-zero volatility
            return 0 if np.mean(excess_returns) == 0 else np.inf * np.sign(np.mean(excess_returns))
        return np.sqrt(252) * np.mean(excess_returns) / self.std_return
        
    def _calculate_sortino_ratio(self, risk_free_rate: float = 0.02) -> float:
        """Calculate Sortino Ratio"""
        returns_array = self.returns.values
        excess_returns = returns_array - risk_free_rate/252
        downside_std = np.sqrt(np.mean(np.minimum(0, excess_returns)**2))
        return np.sqrt(252) * np.mean(excess_returns) / downside_std
        
    def _calculate_calmar_ratio(self) -> float:
        """Calculate Calmar Ratio"""
        if self.max_drawdown == 0:
            return np.inf
        return -(self.mean_return * 252) / self.max_drawdown

class RiskManager:
    """Advanced risk management system with dynamic controls and real-time monitoring."""
    
    def __init__(self, config: Optional[Dict] = None):
        """Initialize the RiskManager with configuration"""
        self.config = config or {}
        self.risk_limits = self.config.get('risk_limits', {
            'portfolio_var': 0.02,  # 2% portfolio VaR limit
            'position_var': 0.05,   # 5% individual position VaR limit
            'concentration': 0.20,   # 20% max concentration in single asset
            'leverage': 2.0,        # 2x max leverage
            'correlation': 0.7      # 70% correlation threshold
        })
        
        # Position and risk limits
        self.max_position_weight = self.config.get('max_position_weight', 0.25)  # 25% max weight
        self.max_position_value = self.config.get('max_position_value', 1000000)  # $1M max position
        
        self.risk_metrics = {}
        self.position_limits = {}
        self.risk_alerts = []
        self.stress_scenarios = {
            'market_crash': {
                'equity_shock': -0.20,
                'volatility_shock': 2.0,
                'correlation_shock': 0.3
            },
            'volatility_spike': {
                'equity_shock': -0.10,
                'volatility_shock': 3.0,
                'correlation_shock': 0.2
            },
            'correlation_breakdown': {
                'equity_shock': -0.05,
                'volatility_shock': 1.5,
                'correlation_shock': -0.4
            },
            'liquidity_crisis': {
                'equity_shock': -0.15,
                'volatility_shock': 2.5,
                'correlation_shock': 0.4,
                'liquidity_shock': 0.5
            }
        }
        
    def calculate_portfolio_risk(self, positions: Dict[str, float],
                            prices: Dict[str, float],
                            returns: Dict[str, pd.Series]) -> Dict:
        """Calculate comprehensive portfolio risk metrics"""
        try:
            # Input validation
            if not positions or not prices or not returns:
                raise ValueError("Positions, prices, and returns dictionaries cannot be empty")
                
            # Ensure all assets have prices and returns
            symbols = list(positions.keys())
            missing_prices = [s for s in symbols if s not in prices]
            missing_returns = [s for s in symbols if s not in returns]
            
            if missing_prices:
                raise ValueError(f"Missing prices for assets: {missing_prices}")
            if missing_returns:
                raise ValueError(f"Missing returns for assets: {missing_returns}")
            
            # Convert positions to numpy array and validate
            position_values = np.array([positions[s] * prices[s] for s in symbols])
            portfolio_value = np.sum(position_values)
            
            if portfolio_value == 0:
                logger.warning("Portfolio value is zero - returning empty risk metrics")
                return {
                    'total_var': 0,
                    'component_var': {s: 0 for s in symbols},
                    'position_limits': {s: {'max_position': 0} for s in symbols},
                    'risk_contributions': {s: 0 for s in symbols},
                    'stress_results': {}
                }
                
            # Ensure all return series have the same index
            common_index = returns[symbols[0]].index
            for sym in symbols[1:]:
                common_index = common_index.intersection(returns[sym].index)
                
            # Calculate return series for each position and ensure they're 1D
            returns_matrix = pd.DataFrame(
                {s: returns[s].reindex(common_index).values.flatten() for s in symbols},
                index=common_index
            )
            
            # Calculate covariance matrix
            cov_matrix = returns_matrix.cov() * 252  # Annualized
            
            # Calculate portfolio variance
            weights = position_values / portfolio_value
            portfolio_variance = np.dot(weights.T, np.dot(cov_matrix.values, weights))
            
            # VaR calculations
            var_95 = self._calculate_var(portfolio_value, weights, cov_matrix, 0.95)
            var_99 = self._calculate_var(portfolio_value, weights, cov_matrix, 0.99)
            es_95 = self._calculate_expected_shortfall(portfolio_value, weights, cov_matrix, 0.95)
            
            # Component VaR
            component_var = self._calculate_component_var(portfolio_value, weights, 
                                                       cov_matrix, symbols)
            
            # Risk contributions
            marginal_var = self._calculate_marginal_var(portfolio_value, weights, 
                                                      cov_matrix, symbols)
                                                      
            # Stress test results
            stress_results = self._run_stress_tests(portfolio_value, weights, 
                                                  cov_matrix, returns_matrix)
                                                  
            # Risk-based position limits
            position_limits = self._calculate_position_limits(portfolio_value, 
                                                           component_var,
                                                           stress_results)
                                                           
            # Correlation analysis
            correlation_matrix = returns_matrix.corr()
            high_correlations = self._identify_high_correlations(correlation_matrix, 
                                                             self.risk_limits['correlation'])
                                                             
            return {
                'total_var_95': var_95,
                'total_var_99': var_99,
                'expected_shortfall_95': es_95,
                'portfolio_volatility': np.sqrt(portfolio_variance),
                'component_var': component_var,
                'marginal_var': marginal_var,
                'position_limits': position_limits,
                'correlation_warnings': high_correlations,
                'stress_results': stress_results,
                'risk_decomposition': {
                    symbol: {
                        'weight': weight,
                        'contribution': contrib,
                        'marginal': marg
                    }
                    for symbol, weight, contrib, marg in zip(
                        symbols, weights, 
                        component_var.values(), 
                        marginal_var.values()
                    )
                }
            }
            
        except Exception as e:
            logger.error(f"Error calculating portfolio risk: {str(e)}")
            return None
            
    def calculate_risk_metrics(self, returns: pd.Series) -> RiskMetrics:
        """Calculate comprehensive risk metrics for a single asset"""
        # Ensure returns is a pandas Series
        if not isinstance(returns, pd.Series):
            returns = pd.Series(returns)

        # Convert to numpy array for calculations
        returns_array = returns.values
        
        # Basic return metrics
        mean_return = np.mean(returns_array)
        std_return = np.std(returns_array)
        
        # Value at Risk (VaR)
        var_95 = -np.percentile(returns_array, 5)
        var_99 = -np.percentile(returns_array, 1)
        
        # Conditional VaR (CVaR/Expected Shortfall)
        cvar_95 = -np.mean(returns_array[returns_array <= -var_95])
        cvar_99 = -np.mean(returns_array[returns_array <= -var_99])
        
        # Maximum Drawdown
        cum_returns = (1 + returns_array).cumprod()
        rolling_max = np.maximum.accumulate(cum_returns)
        drawdowns = cum_returns / rolling_max - 1
        max_drawdown = np.min(drawdowns)
        
        # Risk-adjusted metrics
        excess_returns = returns_array - self.risk_free_rate/252  # Daily risk-free rate
        sharpe_ratio = np.sqrt(252) * np.mean(excess_returns) / std_return
        downside_returns = returns_array[returns_array < 0]
        downside_std = np.std(downside_returns) if len(downside_returns) > 0 else 1e-10
        sortino_ratio = np.sqrt(252) * np.mean(excess_returns) / downside_std
        calmar_ratio = (-mean_return * 252) / abs(max_drawdown) if max_drawdown != 0 else np.inf
        
        # Advanced metrics
        omega_ratio = len(returns[returns > 0]) / len(returns[returns < 0]) if len(returns[returns < 0]) > 0 else np.inf
        information_ratio = excess_returns.mean() / excess_returns.std() if len(excess_returns) > 0 else 0
        beta = 1.0  # Assuming market beta of 1 for simplification
        treynor_ratio = excess_returns.mean() / beta if beta != 0 else 0
        
        # Kelly Criterion and Risk of Ruin
        win_rate = len(returns[returns > 0]) / len(returns)
        avg_win = returns[returns > 0].mean() if len(returns[returns > 0]) > 0 else 0
        avg_loss = abs(returns[returns < 0].mean()) if len(returns[returns < 0]) > 0 else 1e-10
        kelly_fraction = (win_rate - ((1 - win_rate) / (avg_win/avg_loss))) / 2  # Half-Kelly for safety
        
        # Risk of ruin calculation
        from scipy import stats
        risk_free_prob = stats.norm.cdf(self.risk_free_rate, mean_return, std_return)
        risk_of_ruin = 1 - (1 - risk_free_prob) ** 252  # Probability of annual ruin
        
        self.metrics = RiskMetrics(
            var_95=var_95,
            var_99=var_99,
            cvar_95=cvar_95,
            cvar_99=cvar_99,
            max_drawdown=max_drawdown,
            sharpe_ratio=sharpe_ratio,
            sortino_ratio=sortino_ratio,
            calmar_ratio=calmar_ratio,
            omega_ratio=omega_ratio,
            information_ratio=information_ratio,
            treynor_ratio=treynor_ratio,
            kelly_fraction=kelly_fraction,
            risk_of_ruin=risk_of_ruin
        )
        
        return self.metrics
    
    def calculate_position_size(
        self,
        current_price: float,
        volatility: float,
        confidence: float,
        equity: float
    ) -> Tuple[float, Dict[str, float]]:
        """Calculate dynamic position size based on multiple factors"""
        # 1. Volatility scaling
        vol_scale = min(self.volatility_target / volatility, self.max_leverage) if volatility > 0 else 1.0
        
        # 2. Drawdown-based scaling
        equity_scale = 1.0
        if equity < self.peak_equity:
            self.current_drawdown = (self.peak_equity - equity) / self.peak_equity
            if self.current_drawdown > self.max_drawdown:
                # Reduce position size as drawdown increases
                equity_scale = max(0.5, 1 - (self.current_drawdown / self.max_drawdown))
        else:
            self.peak_equity = equity
            self.current_drawdown = 0
            
        # 3. Kelly Criterion adjustment (if metrics available)
        kelly_scale = 1.0
        if self.metrics:
            kelly_scale = max(0.1, min(1.0, self.metrics.kelly_fraction))
            
        # 4. VaR-based position limit
        var_scale = 1.0
        if self.metrics and abs(self.metrics.var_95) > self.var_limit:
            var_scale = self.var_limit / abs(self.metrics.var_95)
            
        # 5. Confidence scaling
        conf_scale = min(1.0, confidence / 100) if confidence > 0 else 0.5
        
        # Combine all scaling factors
        base_size = self.max_pos_size
        final_size = base_size * vol_scale * equity_scale * kelly_scale * var_scale * conf_scale
        
        # Cap at maximum allowed position size
        final_size = min(final_size, self.max_pos_size)
        
        sizing_factors = {
            'volatility_scale': vol_scale,
            'equity_scale': equity_scale,
            'kelly_scale': kelly_scale,
            'var_scale': var_scale,
            'confidence_scale': conf_scale,
            'final_size': final_size
        }
        
        return final_size, sizing_factors

    def _calculate_var(self, portfolio_value: float, weights: np.ndarray,
                      cov_matrix: pd.DataFrame, confidence: float) -> float:
        """Calculate Value at Risk using parametric method"""
        portfolio_std = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
        z_score = norm.ppf(confidence)
        return portfolio_value * portfolio_std * z_score
        
    def _calculate_expected_shortfall(self, portfolio_value: float,
                                    weights: np.ndarray,
                                    cov_matrix: pd.DataFrame,
                                    confidence: float) -> float:
        """Calculate Expected Shortfall (CVaR)"""
        portfolio_std = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
        z_score = norm.ppf(confidence)
        es = norm.pdf(z_score) / (1 - confidence)
        return portfolio_value * portfolio_std * es
        
    def _calculate_component_var(self, portfolio_value: float,
                               weights: np.ndarray,
                               cov_matrix: pd.DataFrame,
                               symbols: List[str]) -> Dict[str, float]:
        """Calculate Component VaR for each position"""
        # Ensure we're working with numpy arrays
        weights_array = np.asarray(weights).flatten()
        cov_array = np.asarray(cov_matrix)
        
        # Calculate portfolio standard deviation
        portfolio_variance = np.dot(weights_array, np.dot(cov_array, weights_array))
        portfolio_std = np.sqrt(max(portfolio_variance, 1e-10))  # Prevent division by zero
        
        # Calculate marginal contributions
        if portfolio_std < 1e-6:  # Very low volatility case
            marginal_var = np.zeros_like(weights_array)
        else:
            marginal_var = np.dot(cov_array, weights_array) / portfolio_std
        
        # Calculate component VaR with safety checks
        component_var = {}
        for i, symbol in enumerate(symbols):
            weight = weights_array[i]
            mvar = marginal_var[i]
            if abs(weight) < 1e-10:  # Negligible position
                component_var[symbol] = 0
            else:
                component_var[symbol] = float(weight * mvar * portfolio_value)
                
        return component_var
        
    def _calculate_marginal_var(self, portfolio_value: float,
                              weights: np.ndarray,
                              cov_matrix: pd.DataFrame,
                              symbols: List[str]) -> Dict[str, float]:
        """Calculate Marginal VaR for each position"""
        # Ensure we're working with numpy arrays
        weights_array = np.asarray(weights).flatten()
        cov_array = np.asarray(cov_matrix)
        
        # Calculate portfolio standard deviation
        portfolio_variance = np.dot(weights_array, np.dot(cov_array, weights_array))
        portfolio_std = np.sqrt(max(portfolio_variance, 1e-10))  # Prevent division by zero
        
        # Calculate marginal contributions
        if portfolio_std < 1e-6:  # Very low volatility case
            marginal_var = np.zeros_like(weights_array)
        else:
            marginal_var = np.dot(cov_array, weights_array) / portfolio_std
        
        # Return marginal VaR with safety checks
        return {symbol: float(marginal_var[i] * portfolio_value) 
                for i, symbol in enumerate(symbols)}
                
    def _run_stress_tests(self, portfolio_value: float, weights: np.ndarray,
                         cov_matrix: pd.DataFrame,
                         returns_matrix: pd.DataFrame) -> Dict:
        """Run stress tests on the portfolio"""
        if portfolio_value <= 0:
            logger.warning("Portfolio value is zero or negative - skipping stress tests")
            return {}
            
        if not isinstance(cov_matrix, pd.DataFrame):
            cov_matrix = pd.DataFrame(cov_matrix)
            
        if not isinstance(returns_matrix, pd.DataFrame):
            returns_matrix = pd.DataFrame(returns_matrix)
        
        results = {}
        diag_vol = np.sqrt(np.diag(cov_matrix))
        
        for scenario, shocks in self.stress_scenarios.items():
            try:
                # Apply shocks to covariance matrix
                shocked_cov = cov_matrix.copy()
                vol_shock = shocks.get('volatility_shock', 1.0)
                shocked_cov *= max(0.1, vol_shock)  # Prevent negative variances
                
                # Apply correlation shocks carefully
                if 'correlation_shock' in shocks:
                    correlation = returns_matrix.corr()
                    shock = min(2.0, max(-1.0, shocks['correlation_shock']))
                    shocked_corr = correlation * (1 + shock)
                    np.fill_diagonal(shocked_corr.values, 1.0)
                    shocked_vol = diag_vol * np.sqrt(vol_shock)
                    shocked_cov = pd.DataFrame(
                        shocked_corr.values * shocked_vol.reshape(-1, 1),
                        index=cov_matrix.index,
                        columns=cov_matrix.columns
                    )
                    shocked_cov = shocked_cov.multiply(shocked_vol, axis=0)
                
                # Calculate stressed portfolio value
                equity_shock = shocks.get('equity_shock', 0)
                stressed_portfolio = portfolio_value * (1 + equity_shock)
                
                # Calculate stressed risk metrics
                stressed_var = self._calculate_var(stressed_portfolio, weights,
                                                 shocked_cov, 0.99)
                stressed_es = self._calculate_expected_shortfall(stressed_portfolio,
                                                              weights, shocked_cov, 0.99)
                                                              
                results[scenario] = {
                    'stressed_value': stressed_portfolio,
                    'stressed_var': stressed_var,
                    'stressed_es': stressed_es,
                    'pnl_impact': stressed_portfolio - portfolio_value
                }
            except Exception as e:
                logger.error(f"Error in stress test {scenario}: {str(e)}")
                results[scenario] = {
                    'error': str(e),
                    'stressed_value': portfolio_value,
                    'stressed_var': 0,
                    'stressed_es': 0,
                    'pnl_impact': 0
                }
            
        return results
        
    def _calculate_position_limits(self, portfolio_value: float,
                                 component_var: Dict[str, float],
                                 stress_results: Dict) -> Dict[str, Dict[str, float]]:
        """Calculate position limits based on risk metrics"""
        if portfolio_value <= 0:
            logger.warning("Portfolio value is zero or negative - using minimum limits")
            return {symbol: {
                'max_position': 0.0,
                'var_limit': 0.0,
                'stress_factor': 0
            } for symbol in component_var}
            
        position_limits = {}
        
        for symbol, var in component_var.items():
            try:
                # Base limits on VaR contribution
                var_limit = portfolio_value * self.risk_limits['position_var']
                
                # Handle zero or negative VaR cases
                var_scaling = abs(var / portfolio_value) if abs(var) > 1e-10 else 1e-10
                
                # Adjust based on stress test results
                stress_impacts = [abs(result.get('pnl_impact', 0))
                                for result in stress_results.values()]
                max_stress = max(stress_impacts) if stress_impacts else 0
                stress_factor = max(0.1, min(1.0, 1 - (max_stress / portfolio_value)))
                
                # Calculate final limits with safety bounds
                var_based_limit = var_limit / var_scaling
                concentration_limit = portfolio_value * self.risk_limits['concentration']
                stress_limit = portfolio_value * stress_factor
                
                max_position = min(
                    var_based_limit,
                    concentration_limit,
                    stress_limit
                )
                
                # Ensure limits are positive and reasonable
                max_position = max(0, min(max_position, portfolio_value * 2))
                
                position_limits[symbol] = {
                    'max_position': max_position,
                    'var_limit': var_limit,
                    'stress_factor': stress_factor,
                    'limits': {
                        'var_based': var_based_limit,
                        'concentration': concentration_limit,
                        'stress': stress_limit
                    }
                }
                
            except Exception as e:
                logger.error(f"Error calculating position limit for {symbol}: {str(e)}")
                position_limits[symbol] = {
                    'max_position': 0,
                    'var_limit': 0,
                    'stress_factor': 0,
                    'error': str(e)
                }
                
        return position_limits
        
    def _identify_high_correlations(self, correlation_matrix: pd.DataFrame,
                                  threshold: float) -> List[Dict]:
        """Identify highly correlated asset pairs"""
        high_correlations = []
        
        for i in range(len(correlation_matrix.index)):
            for j in range(i + 1, len(correlation_matrix.columns)):
                corr = correlation_matrix.iloc[i, j]
                if abs(corr) > threshold:
                    high_correlations.append({
                        'asset1': correlation_matrix.index[i],
                        'asset2': correlation_matrix.columns[j],
                        'correlation': corr
                    })
                    
        return high_correlations
        
    def check_position_limits(self, positions: Dict[str, float], prices: Dict[str, float]) -> List[str]:
        """Check if positions exceed individual asset limits"""
        violations = []
        portfolio_value = 0.0
        
        # First pass: calculate portfolio value with safe conversions
        for sym, pos in positions.items():
            if sym in prices:
                # Convert potential Series/DataFrames to scalars
                if isinstance(pos, pd.Series):
                    pos_val = float(pos.iloc[0])
                elif isinstance(pos, pd.DataFrame):
                    pos_val = float(pos.iloc[0, 0])
                else:
                    pos_val = float(pos)
                    
                if isinstance(prices[sym], pd.Series):
                    price_val = float(prices[sym].iloc[0])
                elif isinstance(prices[sym], pd.DataFrame):
                    price_val = float(prices[sym].iloc[0, 0]) 
                else:
                    price_val = float(prices[sym])
                    
                portfolio_value += pos_val * price_val
                
        # Second pass: check violations with safe value comparisons
        for symbol, position in positions.items():
            if symbol not in prices:
                continue
                
            # Ensure we're working with scalar values
            if isinstance(position, pd.Series):
                pos_val = float(position.iloc[0])
            elif isinstance(position, pd.DataFrame):
                pos_val = float(position.iloc[0, 0])
            else:
                pos_val = float(position)
                
            if isinstance(prices[symbol], pd.Series):
                price_val = float(prices[symbol].iloc[0])
            elif isinstance(prices[symbol], pd.DataFrame):
                price_val = float(prices[symbol].iloc[0, 0])
            else:
                price_val = float(prices[symbol])
                
            position_value = pos_val * price_val
            
            # Calculate position weight with safe division
            position_weight = position_value / portfolio_value if portfolio_value > 0 else 0.0
            
            # Check weight limits (convert to percentage for message)
            if position_weight > self.max_position_weight:
                violations.append(
                    f"Position weight for {symbol} ({position_weight*100:.2f}%) "
                    f"exceeds limit of {self.max_position_weight*100:.2f}%"
                )
                
            # Check absolute value limits with safe comparison   
            abs_position_value = abs(float(position_value))
            if abs_position_value > self.max_position_value:
                violations.append(
                    f"Position value for {symbol} (${abs_position_value:,.2f}) "
                    f"exceeds limit of ${self.max_position_value:,.2f}"
                )
                
        return violations
        
    def check_risk_limits(self, positions: Dict[str, float],
                         prices: Dict[str, float],
                         risk_metrics: Dict) -> List[Dict]:
        """Check for risk limit violations"""
        violations = []
        
        # Calculate portfolio value, ensuring we have scalar floats
        portfolio_value = 0.0
        for sym, pos in positions.items():
            if isinstance(pos, pd.Series):
                pos_val = float(pos.iloc[0])
            elif isinstance(pos, pd.DataFrame):
                pos_val = float(pos.iloc[0, 0])
            else:
                pos_val = float(pos)
                
            if isinstance(prices[sym], pd.Series):
                price_val = float(prices[sym].iloc[0])
            elif isinstance(prices[sym], pd.DataFrame):
                price_val = float(prices[sym].iloc[0, 0])
            else:
                price_val = float(prices[sym])
            portfolio_value += pos_val * price_val
        
        # Extract and validate VaR metrics
        var_95 = risk_metrics.get('total_var_95', 0)
        if isinstance(var_95, (pd.Series, pd.DataFrame)):
            # Handle both scalar and vector values
            if len(var_95) > 0:
                var_95 = float(var_95.iloc[0] if isinstance(var_95, pd.Series) else var_95.iloc[0, 0])
            else:
                var_95 = 0.0
        elif isinstance(var_95, np.ndarray):
            var_95 = float(var_95.item() if var_95.size > 0 else 0)
        else:
            var_95 = float(var_95)
            
        # Check portfolio VaR limit
        var_limit = float(portfolio_value * self.risk_limits['portfolio_var'])
        if var_95 > var_limit:
            ratio = var_95 / portfolio_value if portfolio_value > 0 else float('inf')
            violations.append({
                'type': 'portfolio_var',
                'current': float(ratio),
                'limit': float(self.risk_limits['portfolio_var']),
                'severity': 'high'
            })
            
        # Check position limits
        risk_decomp = risk_metrics.get('risk_decomposition', {})
        for symbol, metrics in risk_decomp.items():
            # Skip if position is missing
            if symbol not in positions or symbol not in prices:
                continue
                
            if isinstance(positions[symbol], pd.Series) or isinstance(prices[symbol], pd.Series):
                # Handle Series multiplication
                if isinstance(positions[symbol], pd.Series):
                    pos = float(positions[symbol].iloc[0])
                else:
                    pos = float(positions[symbol])
                    
                if isinstance(prices[symbol], pd.Series):
                    price = float(prices[symbol].iloc[0])
                else:
                    price = float(prices[symbol])
                    
                position_value = pos * price
            else:
                # Handle scalar multiplication
                position_value = float(positions[symbol]) * float(prices[symbol])
            
            # Check concentration limit (with safety check for zero portfolio value)
            concentration = (position_value / portfolio_value) if portfolio_value > 0 else float('inf')
            if concentration > self.risk_limits['concentration']:
                violations.append({
                    'type': 'concentration',
                    'symbol': symbol,
                    'current': float(concentration),
                    'limit': self.risk_limits['concentration'],
                    'severity': 'medium'
                })
                
            # Check individual VaR contribution with proper type handling
            contrib = metrics.get('contribution', 0)
            if isinstance(contrib, (pd.Series, pd.DataFrame)):
                # Handle both scalar and vector values
                if len(contrib) > 0:
                    contrib = float(contrib.iloc[0] if isinstance(contrib, pd.Series) else contrib.iloc[0, 0])
                else:
                    contrib = 0.0
            else:
                contrib = float(contrib)
            
            position_var_limit = float(self.risk_limits['position_var'])
            if contrib > position_var_limit:
                violations.append({
                    'type': 'position_var',
                    'symbol': symbol,
                    'current': contrib,
                    'limit': position_var_limit,
                    'severity': 'high'
                })
                
        # Check correlation warnings with enhanced validation
        corr_warnings = risk_metrics.get('correlation_warnings', [])
        for corr in corr_warnings:
            if not isinstance(corr, dict):
                continue
                
            correlation = corr.get('correlation', 0)
            if isinstance(correlation, (pd.Series, pd.DataFrame)):
                # Handle both scalar and vector values
                if len(correlation) > 0:
                    correlation = float(correlation.iloc[0] if isinstance(correlation, pd.Series) else correlation.iloc[0, 0])
                else:
                    correlation = 0.0
            else:
                correlation = float(correlation)
            
            # Skip if correlation is invalid
            if not np.isfinite(correlation):
                continue
                
            violations.append({
                'type': 'correlation',
                'assets': [corr.get('asset1', 'unknown'), corr.get('asset2', 'unknown')],
                'current': correlation,
                'limit': float(self.risk_limits['correlation']),
                'severity': 'medium'
            })
            
        return violations
        
    def generate_risk_report(self, positions: Dict[str, float],
                           prices: Dict[str, float],
                           risk_metrics: Dict) -> Dict:
        """Generate comprehensive risk report"""
        # Get violations and ensure portfolio value is scalar
        violations = self.check_risk_limits(positions, prices, risk_metrics)
        portfolio_value = 0.0
        for sym, pos in positions.items():
            if isinstance(pos, pd.Series):
                pos_val = float(pos.iloc[0])
            elif isinstance(pos, pd.DataFrame):
                pos_val = float(pos.iloc[0, 0])
            else:
                pos_val = float(pos)
                
            if isinstance(prices[sym], pd.Series):
                price_val = float(prices[sym].iloc[0])
            elif isinstance(prices[sym], pd.DataFrame):
                price_val = float(prices[sym].iloc[0, 0])
            else:
                price_val = float(prices[sym])
            portfolio_value += pos_val * price_val
            
        # Safely extract risk metrics with proper scalar conversion
        def safe_float(value):
            if isinstance(value, (pd.Series, pd.DataFrame)):
                if len(value) > 0:
                    return float(value.iloc[0] if isinstance(value, pd.Series) else value.iloc[0, 0])
                return 0.0
            elif isinstance(value, np.ndarray):
                return float(value.item() if value.size > 0 else 0)
            return float(value)
            
        # Build summary with safe conversions
        report = {
            'timestamp': datetime.now().isoformat(),
            'portfolio_summary': {
                'total_value': portfolio_value,
                'position_count': len(positions),
                'var_95': safe_float(risk_metrics.get('total_var_95', 0)),
                'var_99': safe_float(risk_metrics.get('total_var_99', 0)),
                'expected_shortfall': safe_float(risk_metrics.get('expected_shortfall_95', 0)),
                'portfolio_volatility': safe_float(risk_metrics.get('portfolio_volatility', 0))
            },
            'risk_violations': violations,
            'position_metrics': {
                symbol: {
                    'value': (
        float(positions[symbol].iloc[0]) if isinstance(positions[symbol], pd.Series)
        else float(positions[symbol].iloc[0, 0]) if isinstance(positions[symbol], pd.DataFrame)
        else float(positions[symbol])
    ) * (
        float(prices[symbol].iloc[0]) if isinstance(prices[symbol], pd.Series)
        else float(prices[symbol].iloc[0, 0]) if isinstance(prices[symbol], pd.DataFrame) 
        else float(prices[symbol])
    ),
                    'weight': safe_float(metrics.get('weight', 0)),
                    'var_contribution': safe_float(metrics.get('contribution', 0)),
                    'marginal_var': safe_float(metrics.get('marginal', 0))
                }
                for symbol, metrics in risk_metrics.get('risk_decomposition', {}).items()
                if symbol in positions and symbol in prices
            },
            'stress_test_summary': {
                scenario: {
                    'pnl_impact': results['pnl_impact'],
                    'pnl_impact_pct': results['pnl_impact'] / portfolio_value
                }
                for scenario, results in risk_metrics['stress_results'].items()
            },
            'risk_alerts': self.risk_alerts[-10:],  # Last 10 alerts
            'recommendations': self._generate_risk_recommendations(violations, risk_metrics)
        }
        
        return report
        
    def _generate_risk_recommendations(self, violations: List[Dict],
                                    risk_metrics: Dict) -> List[Dict]:
        """Generate risk management recommendations"""
        recommendations = []
        
        # Handle VaR violations
        var_violations = [v for v in violations if v['type'] in ['portfolio_var', 'position_var']]
        if var_violations:
            recommendations.append({
                'type': 'risk_reduction',
                'priority': 'high',
                'description': 'Reduce position sizes in highest VaR contributors',
                'targets': [
                    symbol for symbol, metrics in sorted(
                        risk_metrics['risk_decomposition'].items(),
                        key=lambda x: x[1]['contribution'],
                        reverse=True
                    )[:3]
                ]
            })
            
        # Handle concentration violations
        conc_violations = [v for v in violations if v['type'] == 'concentration']
        if conc_violations:
            recommendations.append({
                'type': 'diversification',
                'priority': 'medium',
                'description': 'Reduce concentrated positions',
                'targets': [v['symbol'] for v in conc_violations]
            })
            
        # Handle correlation warnings
        corr_violations = [v for v in violations if v['type'] == 'correlation']
        if corr_violations:
            recommendations.append({
                'type': 'correlation_management',
                'priority': 'medium',
                'description': 'Consider reducing exposure to highly correlated assets',
                'targets': [v['assets'] for v in corr_violations]
            })
            
        # Stress test based recommendations
        worst_scenario = max(
            risk_metrics['stress_results'].items(),
            key=lambda x: abs(x[1]['pnl_impact'])
        )
        
        if abs(worst_scenario[1]['pnl_impact']) > 0.1:  # 10% threshold
            recommendations.append({
                'type': 'stress_test',
                'priority': 'high',
                'description': f'High vulnerability to {worst_scenario[0]} scenario',
                'action': 'Consider hedging against specific risk factors'
            })
            
        return recommendations
        
    def save_risk_report(self, report: Dict, base_path: str = 'risk_reports') -> None:
        """Save risk report to file"""
        path = Path(base_path)
        path.mkdir(exist_ok=True)
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = path / f'risk_report_{timestamp}.json'
        
        with open(filename, 'w') as f:
            json.dump(report, f, indent=2)

class AdvancedStopManager:
    """Advanced stop management with multiple stop types"""
    
    def __init__(
        self,
        trailing_pct: float = 0.05,
        atr_multiplier: float = 2.0,
        time_stop_days: int = 5,
        profit_target_pct: float = 0.1
    ):
        self.trailing_pct = trailing_pct
        self.atr_multiplier = atr_multiplier
        self.time_stop_days = time_stop_days
        self.profit_target_pct = profit_target_pct
        
        self.entry_price = None
        self.entry_time = None
        self.highest_price = None
        self.lowest_price = None
        self.current_atr = None
        
    def enter_position(
        self,
        price: float,
        timestamp: pd.Timestamp,
        atr: Optional[float] = None
    ):
        """Initialize stops for new position"""
        self.entry_price = price
        self.entry_time = timestamp
        self.highest_price = price
        self.lowest_price = price
        self.current_atr = atr
        
    def update(
        self,
        price: float,
        timestamp: pd.Timestamp,
        atr: Optional[float] = None
    ) -> Dict[str, any]:
        """Update and check all stop types"""
        if self.entry_price is None:
            return {'action': 'NONE'}
            
        # Update tracking variables
        self.highest_price = max(self.highest_price, price)
        self.lowest_price = min(self.lowest_price, price)
        self.current_atr = atr if atr is not None else self.current_atr
        
        # 1. Trailing Stop
        trailing_stop = self.highest_price * (1 - self.trailing_pct)
        if price < trailing_stop:
            return {
                'action': 'EXIT',
                'reason': 'TRAILING_STOP',
                'price': price,
                'profit': (price - self.entry_price) / self.entry_price
            }
            
        # 2. ATR Stop
        if self.current_atr:
            atr_stop = self.entry_price - (self.current_atr * self.atr_multiplier)
            if price < atr_stop:
                return {
                    'action': 'EXIT',
                    'reason': 'ATR_STOP',
                    'price': price,
                    'profit': (price - self.entry_price) / self.entry_price
                }
                
        # 3. Time Stop
        days_held = (timestamp - self.entry_time).days
        if days_held >= self.time_stop_days:
            return {
                'action': 'EXIT',
                'reason': 'TIME_STOP',
                'price': price,
                'profit': (price - self.entry_price) / self.entry_price
            }
            
        # 4. Profit Target
        current_profit = (price - self.entry_price) / self.entry_price
        if current_profit >= self.profit_target_pct:
            return {
                'action': 'EXIT',
                'reason': 'PROFIT_TARGET',
                'price': price,
                'profit': current_profit
            }
            
        return {
            'action': 'HOLD',
            'current_profit': current_profit,
            'trailing_stop': trailing_stop,
            'atr_stop': atr_stop if self.current_atr else None,
            'days_held': days_held
        }
        
    def reset(self):
        """Reset stop manager for new trade"""
        self.entry_price = None
        self.entry_time = None
        self.highest_price = None
        self.lowest_price = None
        
class PositionSizer:
    """Advanced position sizing with risk-based adjustments"""
    
    def __init__(
        self,
        initial_capital: float,
        max_risk_pct: float = 0.02,  # 2% risk per trade
        max_position_pct: float = 0.20,  # 20% max position size
        risk_scaling_factors: Dict[str, float] = None
    ):
        self.initial_capital = initial_capital
        self.current_capital = initial_capital
        self.max_risk_pct = max_risk_pct
        self.max_position_pct = max_position_pct
        self.risk_scaling_factors = risk_scaling_factors or {
            'volatility': 1.0,
            'correlation': 1.0,
            'sentiment': 1.0,
            'trend': 1.0
        }
        self.peak_capital = initial_capital
        self.drawdown_threshold = 0.10  # 10% drawdown triggers risk reduction
        
    def update_capital(self, new_capital: float):
        """Update current capital and peak capital"""
        self.current_capital = new_capital
        self.peak_capital = max(self.peak_capital, new_capital)
        
    def calculate_position_size(
        self,
        entry_price: float,
        stop_price: float,
        risk_score: float,
        volatility: float = None,
        correlation: float = None,
        sentiment_score: float = None,
        trend_strength: float = None
    ) -> Dict[str, float]:
        """
        Calculate position size based on risk parameters and market conditions
        
        Args:
            entry_price: Current asset price
            stop_price: Stop loss price
            risk_score: Overall risk score (0-1)
            volatility: Current volatility (optional)
            correlation: Current correlation with portfolio (optional)
            sentiment_score: Market sentiment score (optional)
            trend_strength: Trend strength indicator (optional)
            
        Returns:
            Dict containing position size and adjustment factors
        """
        # 1. Calculate maximum risk amount
        risk_amount = self.current_capital * self.max_risk_pct
        
        # 2. Apply drawdown-based risk reduction
        current_drawdown = (self.peak_capital - self.current_capital) / self.peak_capital
        if current_drawdown > self.drawdown_threshold:
            drawdown_factor = 1 - (current_drawdown - self.drawdown_threshold)
            risk_amount *= max(0.25, drawdown_factor)  # Never reduce below 25%
            
        # 3. Calculate base position size
        # Handle various input types
        if isinstance(entry_price, (pd.Series, pd.DataFrame)):
            entry_price = entry_price.values[-1]
        elif isinstance(entry_price, np.ndarray):
            entry_price = entry_price.ravel()[-1]
            
        if isinstance(stop_price, (pd.Series, pd.DataFrame)):
            stop_price = stop_price.values[-1]
        elif isinstance(stop_price, np.ndarray):
            stop_price = stop_price.ravel()[-1]
            
        try:
            entry_price = float(entry_price)
            stop_price = float(stop_price)
        except (ValueError, TypeError) as e:
            logger.error(f"Invalid price values: {str(e)}")
            return {'position_size': 0, 'factors': {'error': 'Invalid price values'}}
            
        price_risk = abs(entry_price - stop_price)
        if price_risk < 1e-10:  # Use small threshold instead of exact zero
            return {
                'position_size': 0,
                'factors': {'error': 'Invalid stop price (equal to entry)'}
            }
            
        base_size = risk_amount / price_risk
        
        # 4. Apply risk scaling factors
        scaling_factors = {}
        
        # Volatility adjustment
        if volatility is not None:
            vol_factor = 1.0
            if volatility > 0:  # High volatility reduces position size
                vol_factor = min(1.0, 1.0 / volatility)
            scaling_factors['volatility'] = vol_factor
            base_size *= vol_factor
            
        # Correlation adjustment
        if correlation is not None:
            corr_factor = 1.0
            if abs(correlation) > 0.5:  # High correlation reduces position size
                corr_factor = 1.0 - (abs(correlation) - 0.5)
            scaling_factors['correlation'] = corr_factor
            base_size *= corr_factor
            
        # Sentiment adjustment
        if sentiment_score is not None:
            sent_factor = 0.5 + sentiment_score/2  # Scale 0-1 to 0.5-1.0
            scaling_factors['sentiment'] = sent_factor
            base_size *= sent_factor
            
        # Trend strength adjustment
        if trend_strength is not None:
            trend_factor = 0.5 + trend_strength/2  # Scale 0-1 to 0.5-1.0
            scaling_factors['trend'] = trend_factor
            base_size *= trend_factor
            
        # 5. Apply maximum position size limit
        max_size = self.current_capital * self.max_position_pct / entry_price
        final_size = min(base_size, max_size)
        
        return {
            'position_size': final_size,
            'notional_value': final_size * entry_price,
            'capital_fraction': (final_size * entry_price) / self.current_capital,
            'risk_amount': risk_amount,
            'scaling_factors': scaling_factors
        }
        
    def adjust_for_portfolio_risk(
        self,
        position_sizes: Dict[str, float],
        correlations: Dict[str, float],
        volatilities: Dict[str, float]
    ) -> Dict[str, float]:
        """
        Adjust position sizes based on portfolio-level risk considerations
        
        Args:
            position_sizes: Dict of asset:size pairs
            correlations: Dict of asset:correlation pairs
            volatilities: Dict of asset:volatility pairs
            
        Returns:
            Dict of adjusted position sizes
        """
        if not position_sizes:
            return {}
            
        # 1. Calculate portfolio concentration
        total_value = sum(position_sizes.values())
        if total_value == 0:
            return position_sizes.copy()
            
        weights = {
            asset: size/total_value 
            for asset, size in position_sizes.items()
        }
        
        # 2. Calculate diversification factor
        effective_n = 1 / sum(w*w for w in weights.values())
        div_target = len(position_sizes)  # Perfect diversification target
        div_factor = min(1.0, effective_n / div_target)
        
        # 3. Adjust for correlations and volatilities
        adjustments = {}
        for asset in position_sizes:
            # Start with base adjustment
            adj_factor = div_factor
            
            # Apply correlation penalty
            if asset in correlations:
                corr = abs(correlations[asset])
                if corr > 0.5:
                    adj_factor *= (1 - (corr - 0.5))
                    
            # Apply volatility adjustment
            if asset in volatilities:
                vol = volatilities[asset]
                if vol > 0:
                    vol_adj = min(1.0, 1.0 / vol)
                    adj_factor *= vol_adj
                    
            adjustments[asset] = adj_factor
            
        # 4. Apply adjustments while maintaining total exposure
        total_adj = sum(adjustments.values())
        if total_adj > 0:
            scale = len(adjustments) / total_adj
            return {
                asset: position_sizes[asset] * adj * scale
                for asset, adj in adjustments.items()
            }
        else:
            return position_sizes.copy()
            
    def calculate_risk_contribution(
        self,
        position_size: float,
        price: float,
        volatility: float,
        correlation: float
    ) -> Dict[str, float]:
        """Calculate risk contribution of a position"""
        position_value = position_size * price
        portfolio_fraction = position_value / self.current_capital
        
        # Calculate standalone risk
        standalone_risk = portfolio_fraction * volatility
        
        # Calculate marginal risk contribution
        marginal_contribution = standalone_risk * correlation
        
        return {
            'position_value': position_value,
            'portfolio_fraction': portfolio_fraction,
            'standalone_risk': standalone_risk,
            'marginal_contribution': marginal_contribution
        }
import numpy as np
import pandas as pd
from typing import List, Dict, Optional, Union
from sklearn.mixture import GaussianMixture
from sklearn.cluster import KMeans
from scipy.stats import norm
from portfolio_optimization import PortfolioOptimizer

class DynamicAssetAllocator:
    """Advanced dynamic asset allocation with regime detection and adaptive strategies."""
    
    def __init__(self, lookback_window: int = 252):
        """Initialize the asset allocator.
        
        Args:
            lookback_window: Number of days to use for regime detection
        """
        self.lookback_window = lookback_window
        self.portfolio_optimizer = PortfolioOptimizer()
        self.current_regime = None
        self.regime_probabilities = None
        
    def detect_market_regime(self, returns: pd.DataFrame, n_regimes: int = 2) -> Dict:
        """Detect market regime using Gaussian Mixture Models.
        
        Args:
            returns: DataFrame of asset returns
            n_regimes: Number of regimes to detect
            
        Returns:
            Dict with regime information
        """
        # Calculate volatility and correlation features
        rolling_vol = returns.rolling(window=20).std()
        rolling_corr = returns.rolling(window=20).corr()
        
        # Prepare features for regime detection
        features = np.column_stack([
            rolling_vol.mean(axis=1).values[-self.lookback_window:],
            rolling_corr.values[-self.lookback_window:].mean(axis=(1,2))
        ])
        
        # Fit GMM
        gmm = GaussianMixture(n_components=n_regimes, random_state=42)
        regime_labels = gmm.fit_predict(features)
        
        # Calculate regime properties
        regime_stats = {}
        for i in range(n_regimes):
            mask = regime_labels == i
            regime_stats[i] = {
                'volatility': np.mean(features[mask, 0]),
                'correlation': np.mean(features[mask, 1]),
                'frequency': np.mean(mask)
            }
            
        self.current_regime = regime_labels[-1]
        self.regime_probabilities = gmm.predict_proba(features[-1].reshape(1, -1))[0]
        
        return {
            'current_regime': self.current_regime,
            'regime_probabilities': self.regime_probabilities,
            'regime_stats': regime_stats
        }
        
    def cluster_assets(self, returns: pd.DataFrame, n_clusters: int = 5) -> Dict:
        """Cluster assets based on their risk-return characteristics.
        
        Args:
            returns: DataFrame of asset returns
            n_clusters: Number of clusters to create
            
        Returns:
            Dict with clustering information
        """
        # Calculate features for clustering
        mean_returns = returns.mean()
        volatilities = returns.std()
        correlations = returns.corr()
        
        # Prepare features
        features = np.column_stack([
            mean_returns,
            volatilities,
            correlations.mean(axis=1)  # Average correlation with other assets
        ])
        
        # Perform clustering
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        clusters = kmeans.fit_predict(features)
        
        # Calculate cluster properties
        cluster_stats = {}
        for i in range(n_clusters):
            mask = clusters == i
            cluster_stats[i] = {
                'assets': returns.columns[mask].tolist(),
                'avg_return': mean_returns[mask].mean(),
                'avg_vol': volatilities[mask].mean(),
                'size': np.sum(mask)
            }
            
        return {
            'clusters': pd.Series(clusters, index=returns.columns),
            'cluster_stats': cluster_stats
        }
        
    def calculate_dynamic_allocation(self, returns: pd.DataFrame,
                                   risk_profile: str = 'moderate',
                                   constraints: Optional[Dict] = None) -> Dict:
        """Calculate dynamic asset allocation based on market regime and risk profile.
        
        Args:
            returns: DataFrame of asset returns
            risk_profile: One of ['conservative', 'moderate', 'aggressive']
            constraints: Optional allocation constraints
            
        Returns:
            Dict with allocation information
        """
        # Detect current market regime
        regime_info = self.detect_market_regime(returns)
        
        # Cluster assets
        cluster_info = self.cluster_assets(returns)
        
        # Risk profile parameters
        risk_params = {
            'conservative': {'max_vol': 0.10, 'max_weight': 0.25},
            'moderate': {'max_vol': 0.15, 'max_weight': 0.35},
            'aggressive': {'max_vol': 0.25, 'max_weight': 0.50}
        }[risk_profile]
        
        # Adjust allocation based on regime
        if self.current_regime == 0:  # Low volatility regime
            target_return = 0.08 if risk_profile == 'aggressive' else 0.05
        else:  # High volatility regime
            target_return = 0.05 if risk_profile == 'aggressive' else 0.03
            
        # Set up portfolio optimizer
        self.portfolio_optimizer.set_data(returns)
        
        # Calculate base allocation using CVaR optimization
        base_allocation = self.portfolio_optimizer.cvar_optimization(
            alpha=0.05,
            target_return=target_return
        )
        
        # Apply regime-based adjustments
        regime_adjusted_weights = base_allocation['weights'].copy()
        high_vol_cluster = max(cluster_info['cluster_stats'].items(),
                             key=lambda x: x[1]['avg_vol'])[0]
        
        # Reduce exposure to high-volatility assets in high-vol regime
        if self.current_regime == 1:
            high_vol_assets = cluster_info['clusters'] == high_vol_cluster
            regime_adjusted_weights[high_vol_assets] *= 0.7
            regime_adjusted_weights = regime_adjusted_weights / regime_adjusted_weights.sum()
            
        # Calculate risk metrics
        portfolio_vol = np.sqrt(np.dot(regime_adjusted_weights.T,
                                     np.dot(returns.cov(), regime_adjusted_weights)))
        
        # Apply final constraints
        if constraints:
            if 'min_weight' in constraints:
                regime_adjusted_weights[regime_adjusted_weights < constraints['min_weight']] = 0
            if 'max_weight' in constraints:
                regime_adjusted_weights[regime_adjusted_weights > constraints['max_weight']] = \
                    constraints['max_weight']
            regime_adjusted_weights = regime_adjusted_weights / regime_adjusted_weights.sum()
            
        return {
            'weights': regime_adjusted_weights,
            'regime_info': regime_info,
            'cluster_info': cluster_info,
            'portfolio_vol': portfolio_vol,
            'expected_return': np.dot(returns.mean(), regime_adjusted_weights)
        }
        
    def generate_rebalancing_trades(self, current_weights: pd.Series,
                                  target_weights: pd.Series,
                                  min_trade_size: float = 0.01) -> pd.Series:
        """Generate rebalancing trades to move from current to target weights.
        
        Args:
            current_weights: Current portfolio weights
            target_weights: Target portfolio weights
            min_trade_size: Minimum trade size as proportion of portfolio
            
        Returns:
            Series of trades (positive for buy, negative for sell)
        """
        trades = target_weights - current_weights
        
        # Remove small trades
        trades[abs(trades) < min_trade_size] = 0
        
        # Ensure dollar-neutral rebalancing
        if abs(trades.sum()) > 1e-10:
            # Adjust largest trades to maintain dollar neutrality
            largest_trades = trades.abs().nlargest(2)
            for asset in largest_trades.index:
                if trades[asset] > 0:
                    trades[asset] -= trades.sum() / 2
                else:
                    trades[asset] -= trades.sum() / 2
                    
        return trades
        
    def calculate_risk_contribution(self, weights: pd.Series,
                                  returns: pd.DataFrame) -> pd.Series:
        """Calculate risk contribution of each asset in the portfolio.
        
        Args:
            weights: Portfolio weights
            returns: Asset returns
            
        Returns:
            Series of risk contributions
        """
        cov = returns.cov()
        portfolio_vol = np.sqrt(np.dot(weights.T, np.dot(cov, weights)))
        marginal_risk = np.dot(cov, weights) / portfolio_vol
        risk_contribution = weights * marginal_risk
        return pd.Series(risk_contribution, index=weights.index)
        
    def stress_test_allocation(self, weights: pd.Series,
                             returns: pd.DataFrame,
                             scenarios: Dict[str, Dict[str, float]] = None) -> Dict:
        """Perform stress testing on the portfolio allocation.
        
        Args:
            weights: Portfolio weights
            returns: Historical returns
            scenarios: Dict of stress test scenarios
            
        Returns:
            Dict with stress test results
        """
        if scenarios is None:
            scenarios = {
                'market_crash': {'mean_shift': -0.03, 'vol_multiplier': 2.0},
                'high_inflation': {'mean_shift': -0.01, 'vol_multiplier': 1.5},
                'recovery': {'mean_shift': 0.02, 'vol_multiplier': 0.8}
            }
            
        results = {}
        base_mean = returns.mean()
        base_cov = returns.cov()
        
        for scenario_name, scenario_params in scenarios.items():
            # Adjust returns for scenario
            scenario_mean = base_mean + scenario_params['mean_shift']
            scenario_cov = base_cov * scenario_params['vol_multiplier']
            
            # Calculate scenario metrics
            scenario_return = np.dot(weights, scenario_mean)
            scenario_vol = np.sqrt(np.dot(weights.T, np.dot(scenario_cov, weights)))
            
            # Calculate VaR and CVaR
            z_score = norm.ppf(0.05)
            var_95 = -(scenario_return + z_score * scenario_vol)
            cvar_95 = -(scenario_return + norm.pdf(z_score) * scenario_vol / 0.05)
            
            results[scenario_name] = {
                'expected_return': scenario_return,
                'volatility': scenario_vol,
                'var_95': var_95,
                'cvar_95': cvar_95
            }
            
        return results
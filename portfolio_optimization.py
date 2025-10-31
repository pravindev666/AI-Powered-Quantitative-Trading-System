import numpy as np
import pandas as pd
from scipy.optimize import minimize
from typing import List, Dict, Tuple, Optional
import cvxopt
from cvxopt import matrix, solvers
solvers.options['show_progress'] = False

class PortfolioOptimizer:
    """Advanced portfolio optimization using multiple strategies including:
    - Mean-Variance Optimization (Markowitz)
    - Risk Parity
    - Black-Litterman
    - Hierarchical Risk Parity
    - Conditional Value at Risk (CVaR)
    """
    
    def __init__(self):
        self.returns_data = None
        self.covariance_matrix = None
        self.asset_names = None
        self.risk_free_rate = 0.02  # Default annual risk-free rate
        
    def set_data(self, returns_data: pd.DataFrame, risk_free_rate: Optional[float] = None):
        """Set the returns data for optimization.
        
        Args:
            returns_data: DataFrame with asset returns (columns are assets)
            risk_free_rate: Annual risk-free rate (optional)
        """
        self.returns_data = returns_data
        self.asset_names = returns_data.columns
        self.covariance_matrix = returns_data.cov()
        if risk_free_rate is not None:
            self.risk_free_rate = risk_free_rate
            
    def mean_variance_optimization(self, target_return: Optional[float] = None, 
                                 max_weight: float = 1.0,
                                 min_weight: float = 0.0) -> Dict:
        """Perform mean-variance optimization with constraints.
        
        Args:
            target_return: Target portfolio return (optional)
            max_weight: Maximum weight per asset
            min_weight: Minimum weight per asset
            
        Returns:
            Dict containing optimal weights and metrics
        """
        n_assets = len(self.asset_names)
        mean_returns = self.returns_data.mean()
        
        # Define constraints
        constraints = [
            {'type': 'eq', 'fun': lambda x: np.sum(x) - 1}  # Weights sum to 1
        ]
        
        if target_return is not None:
            constraints.append({
                'type': 'eq',
                'fun': lambda x: np.sum(mean_returns * x) - target_return
            })
            
        bounds = [(min_weight, max_weight) for _ in range(n_assets)]
        
        # Objective function to minimize portfolio variance
        def objective(weights):
            return np.sqrt(np.dot(weights.T, np.dot(self.covariance_matrix, weights)))
        
        # Initial guess - equal weights
        initial_weights = np.array([1/n_assets] * n_assets)
        
        # Optimize
        result = minimize(objective, initial_weights,
                        method='SLSQP',
                        bounds=bounds,
                        constraints=constraints)
        
        optimal_weights = pd.Series(result.x, index=self.asset_names)
        metrics = {
            'weights': optimal_weights,
            'volatility': objective(result.x),
            'expected_return': np.sum(mean_returns * result.x),
            'sharpe_ratio': (np.sum(mean_returns * result.x) - self.risk_free_rate) / objective(result.x)
        }
        
        return metrics
    
    def risk_parity_optimization(self, risk_budget: Optional[List[float]] = None) -> Dict:
        """Implement risk parity portfolio optimization.
        
        Args:
            risk_budget: Target risk contribution per asset (optional)
            
        Returns:
            Dict containing optimal weights and metrics
        """
        n_assets = len(self.asset_names)
        if risk_budget is None:
            risk_budget = [1/n_assets] * n_assets
            
        def objective(weights):
            weights = np.array(weights)
            portfolio_risk = np.sqrt(np.dot(weights.T, np.dot(self.covariance_matrix, weights)))
            risk_contribution = weights * (np.dot(self.covariance_matrix, weights)) / portfolio_risk
            return np.sum((risk_contribution - risk_budget*portfolio_risk)**2)
        
        constraints = [{'type': 'eq', 'fun': lambda x: np.sum(x) - 1}]
        bounds = [(0.0, 1.0) for _ in range(n_assets)]
        
        # Optimize
        result = minimize(objective, [1/n_assets] * n_assets,
                        method='SLSQP',
                        bounds=bounds,
                        constraints=constraints)
        
        optimal_weights = pd.Series(result.x, index=self.asset_names)
        mean_returns = self.returns_data.mean()
        
        metrics = {
            'weights': optimal_weights,
            'volatility': np.sqrt(np.dot(result.x.T, np.dot(self.covariance_matrix, result.x))),
            'expected_return': np.sum(mean_returns * result.x)
        }
        
        return metrics
    
    def hierarchical_risk_parity(self) -> Dict:
        """Implement Hierarchical Risk Parity portfolio optimization.
        
        Returns:
            Dict containing optimal weights and metrics
        """
        def get_clusters(corr):
            # Compute distance matrix
            dist = np.sqrt(2*(1 - corr))
            
            # Initialize clusters
            n = len(dist)
            clusters = {i: [i] for i in range(n)}
            
            while len(clusters) > 1:
                # Find closest pair of clusters
                min_dist = float('inf')
                for i in clusters:
                    for j in clusters:
                        if i < j:
                            d = np.mean([dist[a][b] for a in clusters[i] for b in clusters[j]])
                            if d < min_dist:
                                min_dist = d
                                min_pair = (i, j)
                
                # Merge clusters
                i, j = min_pair
                clusters[i].extend(clusters[j])
                del clusters[j]
                
            return list(clusters.values())[0]
        
        corr = self.returns_data.corr()
        ordered_indices = get_clusters(corr)
        hrp_weights = pd.Series(1/len(self.asset_names), index=self.asset_names)
        clustered = pd.Series(ordered_indices, index=self.asset_names)
        
        mean_returns = self.returns_data.mean()
        optimal_weights = hrp_weights
        
        metrics = {
            'weights': optimal_weights,
            'clustering': clustered,
            'volatility': np.sqrt(np.dot(optimal_weights.T, np.dot(self.covariance_matrix, optimal_weights))),
            'expected_return': np.sum(mean_returns * optimal_weights)
        }
        
        return metrics
    
    def black_litterman_optimization(self, views: Dict[str, Tuple[List[str], float, float]], 
                                   market_weights: pd.Series) -> Dict:
        """Implement Black-Litterman portfolio optimization.
        
        Args:
            views: Dict of views {name: ([assets], expected_return, confidence)}
            market_weights: Current market portfolio weights
            
        Returns:
            Dict containing optimal weights and metrics
        """
        n_assets = len(self.asset_names)
        
        # Market equilibrium
        market_ret = np.dot(market_weights, self.returns_data.mean())
        delta = 2.5  # Market price of risk
        pi = delta * np.dot(self.covariance_matrix, market_weights)
        
        # Process views
        n_views = len(views)
        P = np.zeros((n_views, n_assets))
        q = np.zeros(n_views)
        omega = np.zeros((n_views, n_views))
        
        for i, (name, (assets, ret, conf)) in enumerate(views.items()):
            for asset in assets:
                idx = self.asset_names.get_loc(asset)
                P[i, idx] = 1/len(assets)
            q[i] = ret
            omega[i, i] = 1/conf
            
        # Compute posterior distribution
        tau = 0.025  # Uncertainty in prior
        sigma_p = np.dot(P, np.dot(self.covariance_matrix, P.T))
        M = np.dot(tau * self.covariance_matrix, P.T)
        posterior_cov = self.covariance_matrix - np.dot(M, np.linalg.solve(sigma_p + omega, P @ tau * self.covariance_matrix))
        posterior_ret = pi + np.dot(M, np.linalg.solve(sigma_p + omega, q - np.dot(P, pi)))
        
        # Optimize with posterior parameters
        def objective(weights):
            return -np.dot(weights, posterior_ret) + 0.5*delta*np.dot(weights.T, np.dot(posterior_cov, weights))
            
        constraints = [{'type': 'eq', 'fun': lambda x: np.sum(x) - 1}]
        bounds = [(0.0, 1.0) for _ in range(n_assets)]
        
        result = minimize(objective, market_weights,
                        method='SLSQP',
                        bounds=bounds,
                        constraints=constraints)
        
        optimal_weights = pd.Series(result.x, index=self.asset_names)
        
        metrics = {
            'weights': optimal_weights,
            'posterior_returns': pd.Series(posterior_ret, index=self.asset_names),
            'volatility': np.sqrt(np.dot(result.x.T, np.dot(posterior_cov, result.x))),
            'expected_return': np.dot(posterior_ret, result.x)
        }
        
        return metrics
    
    def cvar_optimization(self, alpha: float = 0.05, 
                         target_return: Optional[float] = None) -> Dict:
        """Optimize portfolio using Conditional Value at Risk (CVaR) minimization.
        
        Args:
            alpha: CVaR confidence level (default 5%)
            target_return: Target portfolio return (optional)
            
        Returns:
            Dict containing optimal weights and metrics
        """
        n_assets = len(self.asset_names)
        n_scenarios = len(self.returns_data)
        mean_returns = self.returns_data.mean()
        
        # Convert the optimization problem to a linear program
        c = matrix([0.0] * n_assets + [1.0] + [1.0/(alpha*n_scenarios)] * n_scenarios)
        
        # Construct constraint matrices
        A = matrix(0.0, (n_scenarios + 2, n_assets + 1 + n_scenarios))
        b = matrix(0.0, (n_scenarios + 2, 1))
        
        # Budget constraint
        A[0, :n_assets] = 1.0
        b[0] = 1.0
        
        # Return constraint if specified
        if target_return is not None:
            A[1, :n_assets] = matrix(mean_returns)
            b[1] = target_return
            
        # CVaR constraints
        returns_matrix = matrix(self.returns_data.values)
        for i in range(n_scenarios):
            A[i+2, :n_assets] = -returns_matrix[i, :]
            A[i+2, n_assets] = -1.0
            A[i+2, n_assets + 1 + i] = -1.0
            
        # Bounds
        G = matrix(0.0, (2*n_assets + n_scenarios, n_assets + 1 + n_scenarios))
        h = matrix(0.0, (2*n_assets + n_scenarios, 1))
        
        # Asset bounds
        for i in range(n_assets):
            G[i, i] = -1.0  # w_i >= 0
            G[i+n_assets, i] = 1.0  # w_i <= 1
            h[i+n_assets] = 1.0
            
        # Auxiliary variable bounds
        for i in range(n_scenarios):
            G[i+2*n_assets, n_assets+1+i] = -1.0
            
        # Solve optimization problem
        sol = solvers.lp(c, G, h, A, b)
        
        if sol['status'] != 'optimal':
            raise ValueError("Optimization failed to converge")
            
        optimal_weights = pd.Series(np.array(sol['x'][:n_assets]).flatten(), 
                                  index=self.asset_names)
        
        metrics = {
            'weights': optimal_weights,
            'cvar': sol['primal objective'],
            'volatility': np.sqrt(np.dot(optimal_weights.T, np.dot(self.covariance_matrix, optimal_weights))),
            'expected_return': np.sum(mean_returns * optimal_weights)
        }
        
        return metrics
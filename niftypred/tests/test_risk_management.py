import unittest
import numpy as np
import pandas as pd
from risk_management import RiskManager, RiskMetrics
from datetime import datetime, timedelta
import yfinance as yf

class TestRiskManagement(unittest.TestCase):
    def setUp(self):
        """Set up test cases"""
        self.risk_manager = RiskManager()
        self.test_symbols = ['AAPL', 'MSFT', 'GOOGL']  # Test with major tech stocks
        
        # Create test portfolio
        self.test_positions = {
            'AAPL': 100,
            'MSFT': 150,
            'GOOGL': 75
        }
        
        # Fetch test data
        self.prices = {}
        self.returns = {}
        
        # Create common date range index
        end_date = datetime.now()
        start_date = end_date - timedelta(days=365)
        date_range = pd.date_range(start=start_date, end=end_date, freq='B')
        
        for symbol in self.test_symbols:
            try:
                data = yf.download(symbol, start=start_date.strftime('%Y-%m-%d'), 
                                 end=end_date.strftime('%Y-%m-%d'))
                # Ensure all returns series have the same index
                returns = data['Close'].pct_change().dropna()
                self.returns[symbol] = returns.reindex(date_range).fillna(0)
                self.prices[symbol] = data['Close'].iloc[-1]
            except Exception as e:
                print(f"Error fetching data for {symbol}: {str(e)}")
                
    def test_risk_metrics_calculation(self):
        """Test calculation of risk metrics"""
        if not self.returns:
            self.skipTest("No market data available")
            
        # Calculate portfolio risk metrics
        risk_metrics = self.risk_manager.calculate_portfolio_risk(
            self.test_positions,
            self.prices,
            self.returns
        )
        
        self.assertIsNotNone(risk_metrics)
        self.assertIn('total_var_95', risk_metrics)
        self.assertIn('expected_shortfall_95', risk_metrics)
        self.assertIn('portfolio_volatility', risk_metrics)
        self.assertIn('component_var', risk_metrics)
        
        # Verify reasonable values
        self.assertTrue(0 <= risk_metrics['total_var_95'])
        self.assertTrue(0 <= risk_metrics['portfolio_volatility'])
        
    def test_stress_testing(self):
        """Test stress testing functionality"""
        if not self.returns:
            self.skipTest("No market data available")
            
        risk_metrics = self.risk_manager.calculate_portfolio_risk(
            self.test_positions,
            self.prices,
            self.returns
        )
        
        stress_results = risk_metrics['stress_results']
        
        self.assertIsNotNone(stress_results)
        self.assertIn('market_crash', stress_results)
        self.assertIn('volatility_spike', stress_results)
        
        # Verify stress impacts
        for scenario, results in stress_results.items():
            self.assertIn('pnl_impact', results)
            self.assertIn('stressed_var', results)
            
    def test_position_limits(self):
        """Test position limit calculations"""
        if not self.returns:
            self.skipTest("No market data available")
            
        risk_metrics = self.risk_manager.calculate_portfolio_risk(
            self.test_positions,
            self.prices,
            self.returns
        )
        
        position_limits = risk_metrics['position_limits']
        
        self.assertIsNotNone(position_limits)
        for symbol in self.test_symbols:
            self.assertIn(symbol, position_limits)
            self.assertIn('max_position', position_limits[symbol])
            
        # Verify limits are reasonable
        portfolio_value = sum(pos * self.prices[sym] 
                            for sym, pos in self.test_positions.items())
        for symbol, limits in position_limits.items():
            self.assertTrue(0 < limits['max_position'] <= portfolio_value)
            
    def test_risk_limit_violations(self):
        """Test risk limit violation detection"""
        if not self.returns:
            self.skipTest("No market data available")
            
        risk_metrics = self.risk_manager.calculate_portfolio_risk(
            self.test_positions,
            self.prices,
            self.returns
        )
        
        violations = self.risk_manager.check_risk_limits(
            self.test_positions,
            self.prices,
            risk_metrics
        )
        
        self.assertIsNotNone(violations)
        self.assertIsInstance(violations, list)
        
        for violation in violations:
            self.assertIn('type', violation)
            self.assertIn('current', violation)
            self.assertIn('limit', violation)
            self.assertIn('severity', violation)
            
    def test_risk_report_generation(self):
        """Test risk report generation"""
        if not self.returns:
            self.skipTest("No market data available")
            
        risk_metrics = self.risk_manager.calculate_portfolio_risk(
            self.test_positions,
            self.prices,
            self.returns
        )
        
        report = self.risk_manager.generate_risk_report(
            self.test_positions,
            self.prices,
            risk_metrics
        )
        
        self.assertIsNotNone(report)
        self.assertIn('portfolio_summary', report)
        self.assertIn('risk_violations', report)
        self.assertIn('position_metrics', report)
        self.assertIn('stress_test_summary', report)
        self.assertIn('recommendations', report)
        
        # Verify report structure
        summary = report['portfolio_summary']
        self.assertIn('total_value', summary)
        self.assertIn('var_95', summary)
        self.assertIn('expected_shortfall', summary)
        
    def test_correlation_analysis(self):
        """Test correlation analysis"""
        if not self.returns:
            self.skipTest("No market data available")
            
        risk_metrics = self.risk_manager.calculate_portfolio_risk(
            self.test_positions,
            self.prices,
            self.returns
        )
        
        correlations = risk_metrics.get('correlation_warnings', [])
        
        self.assertIsNotNone(correlations)
        self.assertIsInstance(correlations, list)
        
        for correlation in correlations:
            self.assertIn('asset1', correlation)
            self.assertIn('asset2', correlation)
            self.assertIn('correlation', correlation)
            self.assertTrue(-1 <= correlation['correlation'] <= 1)
            
    def test_risk_recommendations(self):
        """Test risk recommendation generation"""
        if not self.returns:
            self.skipTest("No market data available")
            
        risk_metrics = self.risk_manager.calculate_portfolio_risk(
            self.test_positions,
            self.prices,
            self.returns
        )
        
        report = self.risk_manager.generate_risk_report(
            self.test_positions,
            self.prices,
            risk_metrics
        )
        
        recommendations = report['recommendations']
        
        self.assertIsNotNone(recommendations)
        self.assertIsInstance(recommendations, list)
        
        for recommendation in recommendations:
            self.assertIn('type', recommendation)
            self.assertIn('priority', recommendation)
            self.assertIn('description', recommendation)
            
    def test_risk_metrics_class(self):
        """Test RiskMetrics class calculations"""
        if not self.returns:
            self.skipTest("No market data available")
            
        # Test with a single asset's returns
        returns = self.returns[self.test_symbols[0]]
        metrics = RiskMetrics(returns)
        
        # Verify metric calculations
        self.assertIsNotNone(metrics.var_95)
        self.assertIsNotNone(metrics.cvar_95)
        self.assertIsNotNone(metrics.max_drawdown)
        self.assertIsNotNone(metrics.sharpe_ratio)
        self.assertIsNotNone(metrics.sortino_ratio)
        self.assertIsNotNone(metrics.calmar_ratio)
        
        # Verify reasonable values
        self.assertTrue(-1 <= metrics.max_drawdown <= 0)
        self.assertTrue(metrics.var_95 >= 0)
        self.assertTrue(metrics.cvar_95 >= metrics.var_95)

if __name__ == '__main__':
    unittest.main()
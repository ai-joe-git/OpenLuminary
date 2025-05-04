import unittest
import pandas as pd
import numpy as np
from src.risk_assessment.advanced_risk import AdvancedRiskAssessment

class TestAdvancedRiskAssessment(unittest.TestCase):
    
    def setUp(self):
        self.risk_assessment = AdvancedRiskAssessment()
        
        # Create sample returns data
        np.random.seed(42)
        n_days = 252
        n_assets = 4
        
        # Generate random returns with different means and volatilities
        returns_data = np.random.normal(
            loc=[0.0005, 0.0007, 0.0006, 0.0004],  # Different means
            scale=[0.01, 0.015, 0.012, 0.008],     # Different volatilities
            size=(n_days, n_assets)
        )
        
        # Create a DataFrame with returns
        self.returns = pd.DataFrame(
            returns_data,
            columns=['Asset1', 'Asset2', 'Asset3', 'Asset4'],
            index=pd.date_range(end=pd.Timestamp.now(), periods=n_days)
        )
        
        # Create sample weights
        self.weights = np.array([0.25, 0.25, 0.25, 0.25])
    
    def test_calculate_var(self):
        # Test calculating VaR with different methods
        var_historical = self.risk_assessment.calculate_var(
            self.returns, self.weights, confidence_level=0.95, method="historical"
        )
        var_parametric = self.risk_assessment.calculate_var(
            self.returns, self.weights, confidence_level=0.95, method="parametric"
        )
        var_monte_carlo = self.risk_assessment.calculate_var(
            self.returns, self.weights, confidence_level=0.95, method="monte_carlo"
        )
        
        # Check that we got the expected result structure
        self.assertIn('var', var_historical)
        self.assertIn('var', var_parametric)
        self.assertIn('var', var_monte_carlo)
        
        # Check that the values are positive (VaR is typically reported as a positive number)
        self.assertGreater(var_historical['var'], 0)
        self.assertGreater(var_parametric['var'], 0)
        self.assertGreater(var_monte_carlo['var'], 0)
        
        # Check that the values are reasonable (less than 10%)
        self.assertLess(var_historical['var'], 0.1)
        self.assertLess(var_parametric['var'], 0.1)
        self.assertLess(var_monte_carlo['var'], 0.1)
    
    def test_calculate_cvar(self):
        # Test calculating CVaR with different methods
        cvar_historical = self.risk_assessment.calculate_cvar(
            self.returns, self.weights, confidence_level=0.95, method="historical"
        )
        
        # Check that we got the expected result structure
        self.assertIn('cvar', cvar_historical)
        
        # Check that the value is positive
        self.assertGreater(cvar_historical['cvar'], 0)
        
        # Check that the value is reasonable (less than 15%)
        self.assertLess(cvar_historical['cvar'], 0.15)
        
        # CVaR should be greater than VaR
        var_historical = self.risk_assessment.calculate_var(
            self.returns, self.weights, confidence_level=0.95, method="historical"
        )
        self.assertGreater(cvar_historical['cvar'], var_historical['var'])
    
    def test_calculate_drawdown(self):
        # Test calculating drawdown
        drawdown = self.risk_assessment.calculate_drawdown(self.returns, self.weights)
        
        # Check that we got the expected result structure
        self.assertIn('max_drawdown', drawdown)
        self.assertIn('avg_drawdown', drawdown)
        self.assertIn('avg_duration', drawdown)
        self.assertIn('drawdown_periods', drawdown)
        self.assertIn('drawdown_series', drawdown)
        
        # Check that max_drawdown is negative (or zero)
        self.assertLessEqual(drawdown['max_drawdown'], 0)
        
        # Check that the drawdown series has the same length as the returns
        self.assertEqual(len(drawdown['drawdown_series']), len(self.returns))
    
    def test_calculate_risk_metrics(self):
        # Test calculating risk metrics
        metrics = self.risk_assessment.calculate_risk_metrics(self.returns, self.weights)
        
        # Check that we got the expected metrics
        self.assertIn('mean_return', metrics)
        self.assertIn('volatility', metrics)
        self.assertIn('sharpe_ratio', metrics)
        self.assertIn('sortino_ratio', metrics)
        self.assertIn('max_drawdown', metrics)
        self.assertIn('calmar_ratio', metrics)
        self.assertIn('skewness', metrics)
        self.assertIn('kurtosis', metrics)
        
        # Check that the values are reasonable
        self.assertGreater(metrics['volatility'], 0)
        self.assertLessEqual(metrics['max_drawdown'], 0)
    
    def test_perform_stress_test(self):
        # Define stress scenarios
        scenarios = {
            "Market Crash": {asset: 0.85 for asset in self.returns.columns},
            "Tech Crash": {
                "Asset1": 0.7,
                "Asset2": 0.8,
                "Asset3": 0.9,
                "Asset4": 0.95
            }
        }
        
        # Test performing stress tests
        stress_results = self.risk_assessment.perform_stress_test(
            self.returns, self.weights, scenarios
        )
        
        # Check that we got results for each scenario
        self.assertEqual(len(stress_results), len(scenarios))
        
        # Check that each scenario has the expected metrics
        for scenario, result in stress_results.items():
            self.assertIn('return', result)
            self.assertIn('volatility', result)
            self.assertIn('var', result)
            self.assertIn('cvar', result)
            self.assertIn('max_drawdown', result)

if __name__ == '__main__':
    unittest.main()

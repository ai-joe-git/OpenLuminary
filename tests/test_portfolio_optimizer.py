import unittest
import pandas as pd
import numpy as np
from src.portfolio_management.optimizer import PortfolioOptimizer

class TestPortfolioOptimizer(unittest.TestCase):
    
    def setUp(self):
        self.optimizer = PortfolioOptimizer()
        
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
    
    def test_calculate_portfolio_performance(self):
        # Test with equal weights
        weights = np.array([0.25, 0.25, 0.25, 0.25])
        performance = self.optimizer.calculate_portfolio_performance(self.returns, weights)
        
        # Check that we got the expected metrics
        self.assertIn('return', performance)
        self.assertIn('volatility', performance)
        self.assertIn('sharpe_ratio', performance)
        
        # Check that the values are reasonable
        self.assertGreater(performance['return'], 0)
        self.assertGreater(performance['volatility'], 0)
        self.assertGreater(performance['sharpe_ratio'], 0)
    
    def test_optimize_sharpe_ratio(self):
        # Test optimizing for maximum Sharpe ratio
        result = self.optimizer.optimize_sharpe_ratio(self.returns)
        
        # Check that we got the expected result structure
        self.assertIn('weights', result)
        self.assertIn('performance', result)
        self.assertIn('optimization_success', result)
        
        # Check that the weights sum to approximately 1
        weights_sum = sum(result['weights'].values())
        self.assertAlmostEqual(weights_sum, 1.0, places=6)
        
        # Check that the optimization was successful
        self.assertTrue(result['optimization_success'])
        
        # Check that the Sharpe ratio is positive
        self.assertGreater(result['performance']['sharpe_ratio'], 0)
    
    def test_optimize_minimum_volatility(self):
        # Test optimizing for minimum volatility
        result = self.optimizer.optimize_minimum_volatility(self.returns)
        
        # Check that we got the expected result structure
        self.assertIn('weights', result)
        self.assertIn('performance', result)
        self.assertIn('optimization_success', result)
        
        # Check that the weights sum to approximately 1
        weights_sum = sum(result['weights'].values())
        self.assertAlmostEqual(weights_sum, 1.0, places=6)
        
        # Check that the optimization was successful
        self.assertTrue(result['optimization_success'])
        
        # Check that the volatility is reasonable
        self.assertGreater(result['performance']['volatility'], 0)
        self.assertLess(result['performance']['volatility'], 0.5)  # Arbitrary upper bound
    
    def test_generate_efficient_frontier(self):
        # Test generating the efficient frontier
        ef = self.optimizer.generate_efficient_frontier(self.returns, n_points=10)
        
        # Check that we got a DataFrame with the expected columns
        self.assertIsInstance(ef, pd.DataFrame)
        self.assertIn('return', ef.columns)
        self.assertIn('volatility', ef.columns)
        self.assertIn('sharpe_ratio', ef.columns)
        
        # Check that we got the expected number of points
        self.assertGreaterEqual(len(ef), 1)  # May be less than n_points if some optimizations fail
        
        # Check that returns and volatilities are monotonically increasing
        self.assertTrue((ef['volatility'].diff().dropna() >= 0).all())
        self.assertTrue((ef['return'].diff().dropna() >= 0).all())

if __name__ == '__main__':
    unittest.main()

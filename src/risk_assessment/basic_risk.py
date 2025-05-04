import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any

class BasicRiskAssessment:
    """Basic risk assessment calculations for financial portfolios."""
    
    def __init__(self, confidence_level: float = 0.95):
        """
        Initialize risk assessment module.
        
        Args:
            confidence_level: Confidence level for VaR calculations (default: 0.95)
        """
        self.confidence_level = confidence_level
    
    def calculate_portfolio_var(
        self, 
        returns: pd.DataFrame, 
        weights: np.ndarray,
        time_horizon: int = 1
    ) -> float:
        """
        Calculate Value at Risk (VaR) for a portfolio.
        
        Args:
            returns: DataFrame of historical returns
            weights: Array of portfolio weights
            time_horizon: Time horizon in days
            
        Returns:
            VaR value at the specified confidence level
        """
        # Calculate portfolio returns
        portfolio_returns = returns.dot(weights)
        
        # Calculate VaR
        var = np.percentile(portfolio_returns, (1 - self.confidence_level) * 100)
        
        # Scale by time horizon (assuming normal distribution)
        var_scaled = var * np.sqrt(time_horizon)
        
        return -var_scaled  # Return as a positive number
    
    def calculate_portfolio_cvar(
        self, 
        returns: pd.DataFrame, 
        weights: np.ndarray,
        time_horizon: int = 1
    ) -> float:
        """
        Calculate Conditional Value at Risk (CVaR) for a portfolio.
        
        Args:
            returns: DataFrame of historical returns
            weights: Array of portfolio weights
            time_horizon: Time horizon in days
            
        Returns:
            CVaR value at the specified confidence level
        """
        # Calculate portfolio returns
        portfolio_returns = returns.dot(weights)
        
        # Calculate VaR
        var = np.percentile(portfolio_returns, (1 - self.confidence_level) * 100)
        
        # Calculate CVaR (Expected Shortfall)
        cvar = portfolio_returns[portfolio_returns <= var].mean()
        
        # Scale by time horizon
        cvar_scaled = cvar * np.sqrt(time_horizon)
        
        return -cvar_scaled  # Return as a positive number
    
    def perform_stress_test(
        self, 
        returns: pd.DataFrame, 
        weights: np.ndarray,
        scenarios: Dict[str, List[float]]
    ) -> Dict[str, float]:
        """
        Perform stress testing on a portfolio.
        
        Args:
            returns: DataFrame of historical returns
            weights: Array of portfolio weights
            scenarios: Dictionary mapping scenario names to lists of return modifiers
            
        Returns:
            Dictionary mapping scenario names to portfolio returns under that scenario
        """
        results = {}
        for scenario_name, scenario_modifiers in scenarios.items():
            # Apply scenario modifiers to historical returns
            scenario_returns = returns.copy()
            for i, modifier in enumerate(scenario_modifiers):
                if i < len(returns.columns):
                    scenario_returns.iloc[:, i] = scenario_returns.iloc[:, i] * modifier
            
            # Calculate portfolio return under this scenario
            portfolio_return = scenario_returns.dot(weights).mean()
            results[scenario_name] = portfolio_return
        
        return results

import os
import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional, Union
import logging
from datetime import datetime, timedelta

from src.integrations.base import BaseIntegration, IntegrationError, IntegrationRegistry

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@IntegrationRegistry.register
class AlphaVantageIntegration(BaseIntegration):
    """Integration with Alpha Vantage API."""
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the Alpha Vantage integration.
        
        Args:
            config: Configuration dictionary with the following keys:
                - api_key: Alpha Vantage API key
                - base_url: (Optional) Base URL for the API
        """
        super().__init__(config)
        self.base_url = config.get("base_url", "https://www.alphavantage.co/query")
        self.api_key = config["api_key"]
        
        # Set rate limiting (Alpha Vantage allows 5 requests per minute for free tier)
        self.rate_limit_wait = 12.0  # Wait 12 seconds between requests to be safe
    
    def _validate_config(self):
        """Validate the configuration."""
        if "api_key" not in self.config:
            raise ValueError("Alpha Vantage API key is required")
    
    def authenticate(self) -> bool:
        """
        Authenticate with Alpha Vantage.
        
        Alpha Vantage uses API key authentication, so this just checks if the API key is valid.
        
        Returns:
            True if authentication was successful, False otherwise
        """
        try:
            # Make a simple request to check if the API key is valid
            response = self._make_request(
                method="GET",
                url=self.base_url,
                params={
                    "function": "TIME_SERIES_INTRADAY",
                    "symbol": "MSFT",
                    "interval": "5min",
                    "apikey": self.api_key,
                    "outputsize": "compact"
                }
            )
            
            # Check if the response contains an error message
            data = response.json()
            if "Error Message" in data:
                logger.error(f"Authentication failed: {data['Error Message']}")
                self.authenticated = False
                return False
            
            self.authenticated = True
            return True
        
        except Exception as e:
            logger.error(f"Authentication failed: {e}")
            self.authenticated = False
            return False
    
    def test_connection(self) -> bool:
        """
        Test the connection to Alpha Vantage.
        
        Returns:
            True if the connection is working, False otherwise
        """
        return self.authenticate()
    
    def get_time_series(
        self, 
        symbol: str, 
        interval: str = "daily", 
        outputsize: str = "compact"
    ) -> pd.DataFrame:
        """
        Get time series data for a symbol.
        
        Args:
            symbol: Stock symbol
            interval: Time interval (1min, 5min, 15min, 30min, 60min, daily, weekly, monthly)
            outputsize: Output size (compact, full)
            
        Returns:
            DataFrame with time series data
            
        Raises:
            IntegrationError: If the request fails
        """
        # Map interval to function
        interval_to_function = {
            "1min": "TIME_SERIES_INTRADAY",
            "5min": "TIME_SERIES_INTRADAY",
            "15min": "TIME_SERIES_INTRADAY",
            "30min": "TIME_SERIES_INTRADAY",
            "60min": "TIME_SERIES_INTRADAY",
            "daily": "TIME_SERIES_DAILY_ADJUSTED",
            "weekly": "TIME_SERIES_WEEKLY_ADJUSTED",
            "monthly": "TIME_SERIES_MONTHLY_ADJUSTED"
        }
        
        function = interval_to_function.get(interval)
        if function is None:
            raise ValueError(f"Invalid interval: {interval}")
        
        # Prepare parameters
        params = {
            "function": function,
            "symbol": symbol,
            "apikey": self.api_key,
            "outputsize": outputsize
        }
        
        # Add interval parameter for intraday data
        if function == "TIME_SERIES_INTRADAY":
            params["interval"] = interval
        
        # Make request
        response = self._make_request(
            method="GET",
            url=self.base_url,
            params=params
        )
        
        # Parse response
        data = response.json()
        
        # Check for error
        if "Error Message" in data:
            raise IntegrationError(f"Alpha Vantage error: {data['Error Message']}")
        
        # Extract time series data
        time_series_key = [k for k in data.keys() if "Time Series" in k]
        if not time_series_key:
            raise IntegrationError("No time series data found in response")
        
        time_series = data[time_series_key[0]]
        
        # Convert to DataFrame
        df = pd.DataFrame.from_dict(time_series, orient="index")
        
        # Convert column names
        column_map = {
            "1. open": "Open",
            "2. high": "High",
            "3. low": "Low",
            "4. close": "Close",
            "5. adjusted close": "Adj Close",
            "6. volume": "Volume",
            "7. dividend amount": "Dividend",
            "8. split coefficient": "Split"
        }
        df.rename(columns=column_map, inplace=True)
        
        # Convert data types
        for col in df.columns:
            df[col] = pd.to_numeric(df[col])
        
        # Convert index to datetime
        df.index = pd.to_datetime(df.index)
        
        # Sort by date
        df.sort_index(inplace=True)
        
        return df
    
    def get_company_overview(self, symbol: str) -> Dict[str, Any]:
        """
        Get company overview for a symbol.
        
        Args:
            symbol: Stock symbol
            
        Returns:
            Dictionary with company overview data
            
        Raises:
            IntegrationError: If the request fails
        """
        # Make request
        response = self._make_request(
            method="GET",
            url=self.base_url,
            params={
                "function": "OVERVIEW",
                "symbol": symbol,
                "apikey": self.api_key
            }
        )
        
        # Parse response
        data = response.json()
        
        # Check for error
        if "Error Message" in data:
            raise IntegrationError(f"Alpha Vantage error: {data['Error Message']}")
        
        # Check if data is empty
        if not data or len(data) <= 1:
            raise IntegrationError(f"No company overview data found for {symbol}")
        
        return data
    
    def get_income_statement(self, symbol: str) -> Dict[str, Any]:
        """
        Get income statement for a symbol.
        
        Args:
            symbol: Stock symbol
            
        Returns:
            Dictionary with income statement data
            
        Raises:
            IntegrationError: If the request fails
        """
        # Make request
        response = self._make_request(
            method="GET",
            url=self.base_url,
            params={
                "function": "INCOME_STATEMENT",
                "symbol": symbol,
                "apikey": self.api_key
            }
        )
        
        # Parse response
        data = response.json()
        
        # Check for error
        if "Error Message" in data:
            raise IntegrationError(f"Alpha Vantage error: {data['Error Message']}")
        
        return data
    
    def get_balance_sheet(self, symbol: str) -> Dict[str, Any]:
        """
        Get balance sheet for a symbol.
        
        Args:
            symbol: Stock symbol
            
        Returns:
            Dictionary with balance sheet data
            
        Raises:
            IntegrationError: If the request fails
        """
        # Make request
        response = self._make_request(
            method="GET",
            url=self.base_url,
            params={
                "function": "BALANCE_SHEET",
                "symbol": symbol,
                "apikey": self.api_key
            }
        )
        
        # Parse response
        data = response.json()
        
        # Check for error
        if "Error Message" in data:
            raise IntegrationError(f"Alpha Vantage error: {data['Error Message']}")
        
        return data
    
    def get_cash_flow(self, symbol: str) -> Dict[str, Any]:
        """
        Get cash flow statement for a symbol.
        
        Args:
            symbol: Stock symbol
            
        Returns:
            Dictionary with cash flow statement data
            
        Raises:
            IntegrationError: If the request fails
        """
        # Make request
        response = self._make_request(
            method="GET",
            url=self.base_url,
            params={
                "function": "CASH_FLOW",
                "symbol": symbol,
                "apikey": self.api_key
            }
        )
        
        # Parse response
        data = response.json()
        
        # Check for error
        if "Error Message" in data:
            raise IntegrationError(f"Alpha Vantage error: {data['Error Message']}")
        
        return data
    
    def get_earnings(self, symbol: str) -> Dict[str, Any]:
        """
        Get earnings data for a symbol.
        
        Args:
            symbol: Stock symbol
            
        Returns:
            Dictionary with earnings data
            
        Raises:
            IntegrationError: If the request fails
        """
        # Make request
        response = self._make_request(
            method="GET",
            url=self.base_url,
            params={
                "function": "EARNINGS",
                "symbol": symbol,
                "apikey": self.api_key
            }
        )
        
        # Parse response
        data = response.json()
        
        # Check for error
        if "Error Message" in data:
            raise IntegrationError(f"Alpha Vantage error: {data['Error Message']}")
        
        return data
    
    def search_symbol(self, keywords: str) -> List[Dict[str, str]]:
        """
        Search for symbols matching keywords.
        
        Args:
            keywords: Search keywords
            
        Returns:
            List of matching symbols
            
        Raises:
            IntegrationError: If the request fails
        """
        # Make request
        response = self._make_request(
            method="GET",
            url=self.base_url,
            params={
                "function": "SYMBOL_SEARCH",
                "keywords": keywords,
                "apikey": self.api_key
            }
        )
        
        # Parse response
        data = response.json()
        
        # Check for error
        if "Error Message" in data:
            raise IntegrationError(f"Alpha Vantage error: {data['Error Message']}")
        
        # Extract matches
        if "bestMatches" not in data:
            return []
        
        return data["bestMatches"]

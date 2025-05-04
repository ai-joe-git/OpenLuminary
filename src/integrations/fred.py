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
class FREDIntegration(BaseIntegration):
    """Integration with Federal Reserve Economic Data (FRED) API."""
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the FRED integration.
        
        Args:
            config: Configuration dictionary with the following keys:
                - api_key: FRED API key
                - base_url: (Optional) Base URL for the API
        """
        super().__init__(config)
        self.base_url = config.get("base_url", "https://api.stlouisfed.org/fred")
        self.api_key = config["api_key"]
        
        # Set rate limiting (FRED allows 120 requests per minute)
        self.rate_limit_wait = 0.5  # Wait 0.5 seconds between requests to be safe
    
    def _validate_config(self):
        """Validate the configuration."""
        if "api_key" not in self.config:
            raise ValueError("FRED API key is required")
    
    def authenticate(self) -> bool:
        """
        Authenticate with FRED.
        
        FRED uses API key authentication, so this just checks if the API key is valid.
        
        Returns:
            True if authentication was successful, False otherwise
        """
        try:
            # Make a simple request to check if the API key is valid
            response = self._make_request(
                method="GET",
                url=f"{self.base_url}/series",
                params={
                    "series_id": "GDP",
                    "api_key": self.api_key,
                    "file_type": "json"
                }
            )
            
            # Check if the response contains an error message
            data = response.json()
            if "error_code" in data:
                logger.error(f"Authentication failed: {data.get('error_message', 'Unknown error')}")
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
        Test the connection to FRED.
        
        Returns:
            True if the connection is working, False otherwise
        """
        return self.authenticate()
    
    def get_series(
        self, 
        series_id: str, 
        observation_start: Optional[str] = None,
        observation_end: Optional[str] = None,
        frequency: Optional[str] = None,
        aggregation_method: Optional[str] = None,
        units: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Get time series data for a FRED series.
        
        Args:
            series_id: FRED series ID
            observation_start: Start date (YYYY-MM-DD)
            observation_end: End date (YYYY-MM-DD)
            frequency: Frequency (d, w, bw, m, q, sa, a, wef, weth, wew, wetu, wem, wesu, wesa, bwew, bwem)
            aggregation_method: Aggregation method (avg, sum, eop)
            units: Units transformation (lin, chg, ch1, pch, pc1, pca, cch, cca, log)
            
        Returns:
            DataFrame with time series data
            
        Raises:
            IntegrationError: If the request fails
        """
        # Prepare parameters
        params = {
            "series_id": series_id,
            "api_key": self.api_key,
            "file_type": "json"
        }
        
        # Add optional parameters
        if observation_start:
            params["observation_start"] = observation_start
        if observation_end:
            params["observation_end"] = observation_end
        if frequency:
            params["frequency"] = frequency
        if aggregation_method:
            params["aggregation_method"] = aggregation_method
        if units:
            params["units"] = units
        
        # Make request
        response = self._make_request(
            method="GET",
            url=f"{self.base_url}/series/observations",
            params=params
        )
        
        # Parse response
        data = response.json()
        
        # Check for error
        if "error_code" in data:
            raise IntegrationError(f"FRED error: {data.get('error_message', 'Unknown error')}")
        
        # Extract observations
        if "observations" not in data:
            raise IntegrationError("No observations found in response")
        
        observations = data["observations"]
        
        # Convert to DataFrame
        df = pd.DataFrame(observations)
        
        # Convert date to datetime
        df["date"] = pd.to_datetime(df["date"])
        
        # Set date as index
        df.set_index("date", inplace=True)
        
        # Convert value to numeric
        df["value"] = pd.to_numeric(df["value"], errors="coerce")
        
        return df
    
    def get_series_info(self, series_id: str) -> Dict[str, Any]:
        """
        Get information about a FRED series.
        
        Args:
            series_id: FRED series ID
            
        Returns:
            Dictionary with series information
            
        Raises:
            IntegrationError: If the request fails
        """
        # Make request
        response = self._make_request(
            method="GET",
            url=f"{self.base_url}/series",
            params={
                "series_id": series_id,
                "api_key": self.api_key,
                "file_type": "json"
            }
        )
        
        # Parse response
        data = response.json()
        
        # Check for error
        if "error_code" in data:
            raise IntegrationError(f"FRED error: {data.get('error_message', 'Unknown error')}")
        
        # Extract series
        if "seriess" not in data:
            raise IntegrationError("No series information found in response")
        
        return data["seriess"][0]
    
    def search_series(
        self, 
        search_text: str,
        search_type: Optional[str] = None,
        limit: int = 1000,
        order_by: Optional[str] = None,
        sort_order: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Search for FRED series.
        
        Args:
            search_text: Search text
            search_type: Search type (full_text, series_id, series_title)
            limit: Maximum number of results
            order_by: Order by (search_rank, series_id, title, units, frequency, seasonal_adjustment, realtime_start, realtime_end, last_updated, observation_start, observation_end, popularity, group_popularity)
            sort_order: Sort order (asc, desc)
            
        Returns:
            List of matching series
            
        Raises:
            IntegrationError: If the request fails
        """
        # Prepare parameters
        params = {
            "search_text": search_text,
            "api_key": self.api_key,
            "file_type": "json",
            "limit": limit
        }
        
        # Add optional parameters
        if search_type:
            params["search_type"] = search_type
        if order_by:
            params["order_by"] = order_by
        if sort_order:
            params["sort_order"] = sort_order
        
        # Make request
        response = self._make_request(
            method="GET",
            url=f"{self.base_url}/series/search",
            params=params
        )
        
        # Parse response
        data = response.json()
        
        # Check for error
        if "error_code" in data:
            raise IntegrationError(f"FRED error: {data.get('error_message', 'Unknown error')}")
        
        # Extract series
        if "seriess" not in data:
            return []
        
        return data["seriess"]
    
    def get_category(self, category_id: int = 0) -> Dict[str, Any]:
        """
        Get information about a FRED category.
        
        Args:
            category_id: Category ID (0 for root category)
            
        Returns:
            Dictionary with category information
            
        Raises:
            IntegrationError: If the request fails
        """
        # Make request
        response = self._make_request(
            method="GET",
            url=f"{self.base_url}/category",
            params={
                "category_id": category_id,
                "api_key": self.api_key,
                "file_type": "json"
            }
        )
        
        # Parse response
        data = response.json()
        
        # Check for error
        if "error_code" in data:
            raise IntegrationError(f"FRED error: {data.get('error_message', 'Unknown error')}")
        
        # Extract category
        if "categories" not in data:
            raise IntegrationError("No category information found in response")
        
        return data["categories"][0]
    
    def get_category_children(self, category_id: int = 0) -> List[Dict[str, Any]]:
        """
        Get children of a FRED category.
        
        Args:
            category_id: Category ID (0 for root category)
            
        Returns:
            List of child categories
            
        Raises:
            IntegrationError: If the request fails
        """
        # Make request
        response = self._make_request(
            method="GET",
            url=f"{self.base_url}/category/children",
            params={
                "category_id": category_id,
                "api_key": self.api_key,
                "file_type": "json"
            }
        )
        
        # Parse response
        data = response.json()
        
        # Check for error
        if "error_code" in data:
            raise IntegrationError(f"FRED error: {data.get('error_message', 'Unknown error')}")
        
        # Extract categories
        if "categories" not in data:
            return []
        
        return data["categories"]
    
    def get_category_series(
        self, 
        category_id: int,
        limit: int = 1000,
        offset: int = 0,
        order_by: Optional[str] = None,
        sort_order: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Get series in a FRED category.
        
        Args:
            category_id: Category ID
            limit: Maximum number of results
            offset: Result offset
            order_by: Order by (series_id, title, units, frequency, seasonal_adjustment, realtime_start, realtime_end, last_updated, observation_start, observation_end, popularity, group_popularity)
            sort_order: Sort order (asc, desc)
            
        Returns:
            List of series in the category
            
        Raises:
            IntegrationError: If the request fails
        """
        # Prepare parameters
        params = {
            "category_id": category_id,
            "api_key": self.api_key,
            "file_type": "json",
            "limit": limit,
            "offset": offset
        }
        
        # Add optional parameters
        if order_by:
            params["order_by"] = order_by
        if sort_order:
            params["sort_order"] = sort_order
        
        # Make request
        response = self._make_request(
            method="GET",
            url=f"{self.base_url}/category/series",
            params=params
        )
        
        # Parse response
        data = response.json()
        
        # Check for error
        if "error_code" in data:
            raise IntegrationError(f"FRED error: {data.get('error_message', 'Unknown error')}")
        
        # Extract series
        if "seriess" not in data:
            return []
        
        return data["seriess"]

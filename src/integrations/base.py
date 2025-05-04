from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional, Union
import logging
import requests
import json
import os
from datetime import datetime, timedelta

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class IntegrationError(Exception):
    """Exception raised for errors in the integration."""
    pass

class BaseIntegration(ABC):
    """Base class for all integrations."""
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the integration.
        
        Args:
            config: Configuration dictionary for the integration
        """
        self.config = config
        self.name = self.__class__.__name__
        self.authenticated = False
        self.last_request_time = None
        self.rate_limit_wait = 1.0  # Default wait time in seconds
        
        # Validate configuration
        self._validate_config()
        
        logger.info(f"Initialized {self.name} integration")
    
    @abstractmethod
    def _validate_config(self):
        """Validate the configuration."""
        pass
    
    @abstractmethod
    def authenticate(self) -> bool:
        """
        Authenticate with the service.
        
        Returns:
            True if authentication was successful, False otherwise
        """
        pass
    
    @abstractmethod
    def test_connection(self) -> bool:
        """
        Test the connection to the service.
        
        Returns:
            True if the connection is working, False otherwise
        """
        pass
    
    def _handle_rate_limiting(self):
        """Handle rate limiting by waiting if necessary."""
        if self.last_request_time is not None:
            elapsed = (datetime.now() - self.last_request_time).total_seconds()
            if elapsed < self.rate_limit_wait:
                import time
                time.sleep(self.rate_limit_wait - elapsed)
        
        self.last_request_time = datetime.now()
    
    def _make_request(
        self, 
        method: str, 
        url: str, 
        headers: Optional[Dict[str, str]] = None,
        params: Optional[Dict[str, Any]] = None,
        data: Optional[Dict[str, Any]] = None,
        json_data: Optional[Dict[str, Any]] = None,
        timeout: int = 30,
        retry_count: int = 3
    ) -> requests.Response:
        """
        Make an HTTP request with rate limiting and retries.
        
        Args:
            method: HTTP method (GET, POST, etc.)
            url: URL to request
            headers: Request headers
            params: Query parameters
            data: Form data
            json_data: JSON data
            timeout: Request timeout in seconds
            retry_count: Number of retries on failure
            
        Returns:
            Response object
            
        Raises:
            IntegrationError: If the request fails after all retries
        """
        if not self.authenticated:
            self.authenticate()
        
        self._handle_rate_limiting()
        
        for attempt in range(retry_count):
            try:
                response = requests.request(
                    method=method,
                    url=url,
                    headers=headers,
                    params=params,
                    data=data,
                    json=json_data,
                    timeout=timeout
                )
                
                response.raise_for_status()
                return response
            
            except requests.exceptions.RequestException as e:
                logger.warning(f"Request failed (attempt {attempt + 1}/{retry_count}): {e}")
                
                if attempt == retry_count - 1:
                    raise IntegrationError(f"Request failed after {retry_count} attempts: {e}")
                
                # Exponential backoff
                import time
                time.sleep(2 ** attempt)
        
        # This should never be reached due to the exception in the loop
        raise IntegrationError("Unexpected error in request handling")

class IntegrationRegistry:
    """Registry for all available integrations."""
    
    _integrations = {}
    
    @classmethod
    def register(cls, integration_class):
        """
        Register an integration class.
        
        Args:
            integration_class: The integration class to register
            
        Returns:
            The integration class (for decorator use)
        """
        cls._integrations[integration_class.__name__] = integration_class
        return integration_class
    
    @classmethod
    def get_integration(cls, name: str, config: Dict[str, Any]):
        """
        Get an integration instance by name.
        
        Args:
            name: Name of the integration
            config: Configuration for the integration
            
        Returns:
            Instance of the integration
            
        Raises:
            ValueError: If the integration is not found
        """
        if name not in cls._integrations:
            raise ValueError(f"Integration '{name}' not found")
        
        return cls._integrations[name](config)
    
    @classmethod
    def list_integrations(cls) -> List[str]:
        """
        List all available integrations.
        
        Returns:
            List of integration names
        """
        return list(cls._integrations.keys())

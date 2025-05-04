import os
import json
from typing import Dict, Any, List, Optional, Union
import logging
from pathlib import Path

from src.integrations.base import IntegrationRegistry, BaseIntegration

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class IntegrationManager:
    """Manager for integrations."""
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize the integration manager.
        
        Args:
            config_path: Path to the configuration file
        """
        self.config_path = config_path or os.path.join(os.path.expanduser("~"), ".openluminary", "integrations.json")
        self.integrations = {}
        self.load_config()
    
    def load_config(self):
        """Load integration configurations from file."""
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(self.config_path), exist_ok=True)
        
        # Create empty config file if it doesn't exist
        if not os.path.exists(self.config_path):
            with open(self.config_path, "w") as f:
                json.dump({}, f)
        
        # Load config
        try:
            with open(self.config_path, "r") as f:
                config = json.load(f)
            
            logger.info(f"Loaded integration configurations from {self.config_path}")
            
            # Initialize integrations
            for integration_name, integration_config in config.items():
                if integration_config.get("enabled", True):
                    try:
                        self.integrations[integration_name] = IntegrationRegistry.get_integration(
                            integration_name, integration_config
                        )
                        logger.info(f"Initialized integration: {integration_name}")
                    except Exception as e:
                        logger.error(f"Failed to initialize integration {integration_name}: {e}")
        
        except Exception as e:
            logger.error(f"Failed to load integration configurations: {e}")
    
    def save_config(self):
        """Save integration configurations to file."""
        config = {}
        
        for integration_name, integration in self.integrations.items():
            config[integration_name] = integration.config
            config[integration_name]["enabled"] = True
        
        try:
            with open(self.config_path, "w") as f:
                json.dump(config, f, indent=2)
            
            logger.info(f"Saved integration configurations to {self.config_path}")
        
        except Exception as e:
            logger.error(f"Failed to save integration configurations: {e}")
    
    def add_integration(self, integration_name: str, config: Dict[str, Any]) -> bool:
        """
        Add an integration.
        
        Args:
            integration_name: Name of the integration
            config: Configuration for the integration
            
        Returns:
            True if the integration was added successfully, False otherwise
        """
        try:
            # Initialize integration
            integration = IntegrationRegistry.get_integration(integration_name, config)
            
            # Test connection
            if not integration.test_connection():
                logger.error(f"Failed to connect to {integration_name}")
                return False
            
            # Add integration
            self.integrations[integration_name] = integration
            
            # Save config
            self.save_config()
            
            logger.info(f"Added integration: {integration_name}")
            return True
        
        except Exception as e:
            logger.error(f"Failed to add integration {integration_name}: {e}")
            return False
    
    def remove_integration(self, integration_name: str) -> bool:
        """
        Remove an integration.
        
        Args:
            integration_name: Name of the integration
            
        Returns:
            True if the integration was removed successfully, False otherwise
        """
        if integration_name not in self.integrations:
            logger.error(f"Integration not found: {integration_name}")
            return False
        
        try:
            # Remove integration
            del self.integrations[integration_name]
            
            # Save config
            self.save_config()
            
            logger.info(f"Removed integration: {integration_name}")
            return True
        
        except Exception as e:
            logger.error(f"Failed to remove integration {integration_name}: {e}")
            return False
    
    def get_integration(self, integration_name: str) -> Optional[BaseIntegration]:
        """
        Get an integration by name.
        
        Args:
            integration_name: Name of the integration
            
        Returns:
            Integration instance or None if not found
        """
        return self.integrations.get(integration_name)
    
    def list_integrations(self) -> List[str]:
        """
        List all initialized integrations.
        
        Returns:
            List of integration names
        """
        return list(self.integrations.keys())
    
    def list_available_integrations(self) -> List[str]:
        """
        List all available integrations.
        
        Returns:
            List of available integration names
        """
        return IntegrationRegistry.list_integrations()
    
    def test_all_connections(self) -> Dict[str, bool]:
        """
        Test all integration connections.
        
        Returns:
            Dictionary mapping integration names to connection status
        """
        results = {}
        
        for integration_name, integration in self.integrations.items():
            try:
                results[integration_name] = integration.test_connection()
            except Exception as e:
                logger.error(f"Error testing connection for {integration_name}: {e}")
                results[integration_name] = False
        
        return results

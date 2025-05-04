import streamlit as st
import pandas as pd
import numpy as np
import json
from typing import Dict, Any, List, Optional
import os

from src.integrations.manager import IntegrationManager
from src.integrations.base import IntegrationRegistry

def render_integrations_page():
    """Render the integrations management page."""
    st.title("Integrations Management")
    
    # Initialize integration manager
    if "integration_manager" not in st.session_state:
        st.session_state.integration_manager = IntegrationManager()
    
    manager = st.session_state.integration_manager
    
    # Create tabs
    tab1, tab2 = st.tabs(["Configured Integrations", "Add Integration"])
    
    with tab1:
        # List configured integrations
        st.subheader("Configured Integrations")
        
        integrations = manager.list_integrations()
        
        if not integrations:
            st.info("No integrations configured. Add an integration in the 'Add Integration' tab.")
        else:
            # Test connections
            if st.button("Test All Connections"):
                connection_results = manager.test_all_connections()
                
                for integration_name, status in connection_results.items():
                    if status:
                        st.success(f"✅ {integration_name}: Connected successfully")
                    else:
                        st.error(f"❌ {integration_name}: Connection failed")
            
            # Display integrations
            for integration_name in integrations:
                with st.expander(f"{integration_name}"):
                    integration = manager.get_integration(integration_name)
                    
                    # Display configuration (excluding sensitive information)
                    config = integration.config.copy()
                    for key in config:
                        if "key" in key.lower() or "token" in key.lower() or "secret" in key.lower() or "password" in key.lower():
                            config[key] = "********"
                    
                    st.json(config)
                    
                    # Test connection
                    col1, col2 = st.columns([1, 3])
                    with col1:
                        if st.button("Test Connection", key=f"test_{integration_name}"):
                            try:
                                if integration.test_connection():
                                    st.success("Connection successful")
                                else:
                                    st.error("Connection failed")
                            except Exception as e:
                                st.error(f"Error: {e}")
                    
                    # Remove integration
                    with col2:
                        if st.button("Remove Integration", key=f"remove_{integration_name}"):
                            if manager.remove_integration(integration_name):
                                st.success(f"Removed integration: {integration_name}")
                                st.experimental_rerun()
                            else:
                                st.error(f"Failed to remove integration: {integration_name}")
    
    with tab2:
        # Add new integration
        st.subheader("Add Integration")
        
        # List available integrations
        available_integrations = manager.list_available_integrations()
        
        integration_type = st.selectbox("Integration Type", available_integrations)
        
        if integration_type:
            # Create form for integration configuration
            with st.form(key="add_integration_form"):
                st.write(f"Configure {integration_type} Integration")
                
                # Create input fields based on integration type
                config = {}
                
                if integration_type == "AlphaVantageIntegration":
                    config["api_key"] = st.text_input("API Key", type="password")
                    config["base_url"] = st.text_input("Base URL (Optional)", "https://www.alphavantage.co/query")
                
                elif integration_type == "FREDIntegration":
                    config["api_key"] = st.text_input("API Key", type="password")
                    config["base_url"] = st.text_input("Base URL (Optional)", "https://api.stlouisfed.org/fred")
                
                elif integration_type == "IEXCloudIntegration":
                    config["api_key"] = st.text_input("API Key", type="password")
                    config["version"] = st.selectbox("API Version", ["stable", "latest", "v1", "beta"])
                    config["base_url"] = st.text_input("Base URL (Optional)", f"https://cloud.iexapis.com/{config['version']}")
                
                # Add more integration types here
                
                # Submit button
                submitted = st.form_submit_button("Add Integration")
                
                if submitted:
                    if manager.add_integration(integration_type, config):
                        st.success(f"Added integration: {integration_type}")
                        st.experimental_rerun()
                    else:
                        st.error(f"Failed to add integration: {integration_type}")

def render_data_explorer():
    """Render the data explorer page."""
    st.title("Data Explorer")
    
    # Initialize integration manager
    if "integration_manager" not in st.session_state:
        st.session_state.integration_manager = IntegrationManager()
    
    manager = st.session_state.integration_manager
    
    # Check if integrations are configured
    integrations = manager.list_integrations()
    
    if not integrations:
        st.warning("No integrations configured. Please add integrations in the Integrations Management page.")
        return
    
    # Create tabs for different data sources
    tabs = []
    
    if manager.get_integration("AlphaVantageIntegration"):
        tabs.append("Alpha Vantage")
    
    if manager.get_integration("FREDIntegration"):
        tabs.append("FRED")
    
    if manager.get_integration("IEXCloudIntegration"):
        tabs.append("IEX Cloud")
    
    if not tabs:
        st.warning("No supported data sources found in configured integrations.")
        return
    
    selected_tab = st.selectbox("Select Data Source", tabs)
    
    if selected_tab == "Alpha Vantage":
        render_alpha_vantage_explorer(manager.get_integration("AlphaVantageIntegration"))
    
    elif selected_tab == "FRED":
        render_fred_explorer(manager.get_integration("FREDIntegration"))
    
    elif selected_tab == "IEX Cloud":
        render_iex_cloud_explorer(manager.get_integration("IEXCloudIntegration"))

def render_alpha_vantage_explorer(integration):
    """Render the Alpha Vantage data explorer."""
    st.subheader("Alpha Vantage Data Explorer")
    
    # Create tabs for different data types
    tab1, tab2, tab3 = st.tabs(["Time Series", "Company Fundamentals", "Search"])
    
    with tab1:
        st.write("Explore time series data")
        
        col1, col2 = st.columns(2)
        
        with col1:
            symbol = st.text_input("Symbol", "AAPL", key="av_ts_symbol")
        
        with col2:
            interval = st.selectbox(
                "Interval", 
                ["1min", "5min", "15min", "30min", "60min", "daily", "weekly", "monthly"],
                index=5,
                key="av_ts_interval"
            )
        
        if st.button("Get Data", key="av_ts_button"):
            try:
                with st.spinner("Fetching data..."):
                    if interval in ["1min", "5min", "15min", "30min", "60min"]:
                        data = integration.get_time_series(symbol, interval=interval)
                    else:
                        data = integration.get_time_series(symbol, interval=interval)
                
                if data is not None and not data.empty:
                    st.write(f"Time series data for {symbol}")
                    st.dataframe(data)
                    
                    # Plot data
                    st.subheader("Price Chart")
                    st.line_chart(data["Close"])
                else:
                    st.warning(f"No data found for {symbol}")
            
            except Exception as e:
                st.error(f"Error fetching data: {e}")
    
    with tab2:
        st.write("Explore company fundamentals")
        
        symbol = st.text_input("Symbol", "AAPL", key="av_fund_symbol")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("Company Overview", key="av_overview_button"):
                try:
                    with st.spinner("Fetching data..."):
                        data = integration.get_company_overview(symbol)
                    
                    if data:
                        st.write(f"Company overview for {symbol}")
                        st.json(data)
                    else:
                        st.warning(f"No data found for {symbol}")
                
                except Exception as e:
                    st.error(f"Error fetching data: {e}")
        
        with col2:
            if st.button("Income Statement", key="av_income_button"):
                try:
                    with st.spinner("Fetching data..."):
                        data = integration.get_income_statement(symbol)
                    
                    if data:
                        st.write(f"Income statement for {symbol}")
                        st.json(data)
                    else:
                        st.warning(f"No data found for {symbol}")
                
                except Exception as e:
                    st.error(f"Error fetching data: {e}")
        
        with col3:
            if st.button("Balance Sheet", key="av_balance_button"):
                try:
                    with st.spinner("Fetching data..."):
                        data = integration.get_balance_sheet(symbol)
                    
                    if data:
                        st.write(f"Balance sheet for {symbol}")
                        st.json(data)
                    else:
                        st.warning(f"No data found for {symbol}")
                
                except Exception as e:
                    st.error(f"Error fetching data: {e}")
    
    with tab3:
        st.write("Search for symbols")
        
        keywords = st.text_input("Keywords", "Apple", key="av_search_keywords")
        
        if st.button("Search", key="av_search_button"):
            try:
                with st.spinner("Searching..."):
                    results = integration.search_symbol(keywords)
                
                if results:
                    st.write(f"Search results for '{keywords}'")
                    
                    # Convert to DataFrame
                    df = pd.DataFrame(results)
                    st.dataframe(df)
                else:
                    st.warning(f"No results found for '{keywords}'")
            
            except Exception as e:
                st.error(f"Error searching: {e}")

def render_fred_explorer(integration):
    """Render the FRED data explorer."""
    st.subheader("FRED Data Explorer")
    
    # Create tabs for different data types
    tab1, tab2 = st.tabs(["Series Data", "Search & Categories"])
    
    with tab1:
        st.write("Explore economic data series")
        
        col1, col2 = st.columns(2)
        
        with col1:
            series_id = st.text_input("Series ID", "GDP", key="fred_series_id")
        
        with col2:
            frequency = st.selectbox(
                "Frequency", 
                ["", "d", "w", "bw", "m", "q", "sa", "a"],
                index=0,
                key="fred_frequency"
            )
        
        if st.button("Get Data", key="fred_series_button"):
            try:
                with st.spinner("Fetching data..."):
                    params = {}
                    if frequency:
                        params["frequency"] = frequency
                    
                    data = integration.get_series(series_id, **params)
                
                if data is not None and not data.empty:
                    st.write(f"Series data for {series_id}")
                    st.dataframe(data)
                    
                    # Plot data
                    st.subheader("Series Chart")
                    st.line_chart(data["value"])
                    
                    # Get series info
                    series_info = integration.get_series_info(series_id)
                    
                    with st.expander("Series Information"):
                        st.json(series_info)
                else:
                    st.warning(f"No data found for {series_id}")
            
            except Exception as e:
                st.error(f"Error fetching data: {e}")
    
    with tab2:
        st.write("Search for series or browse categories")
        
        search_tab, category_tab = st.tabs(["Search", "Categories"])
        
        with search_tab:
            search_text = st.text_input("Search Text", "GDP", key="fred_search_text")
            
            if st.button("Search", key="fred_search_button"):
                try:
                    with st.spinner("Searching..."):
                        results = integration.search_series(search_text)
                    
                    if results:
                        st.write(f"Search results for '{search_text}'")
                        
                        # Convert to DataFrame
                        df = pd.DataFrame(results)
                        st.dataframe(df)
                    else:
                        st.warning(f"No results found for '{search_text}'")
                
                except Exception as e:
                    st.error(f"Error searching: {e}")
        
        with category_tab:
            if "fred_category_id" not in st.session_state:
                st.session_state.fred_category_id = 0
            
            # Get current category
            try:
                category = integration.get_category(st.session_state.fred_category_id)
                st.write(f"Category: {category['name']}")
                
                # Get children
                children = integration.get_category_children(st.session_state.fred_category_id)
                
                if children:
                    st.write("Subcategories:")
                    
                    # Display as buttons
                    cols = st.columns(3)
                    for i, child in enumerate(children):
                        with cols[i % 3]:
                            if st.button(child["name"], key=f"cat_{child['id']}"):
                                st.session_state.fred_category_id = child["id"]
                                st.experimental_rerun()
                
                # Get series in category
                if st.button("View Series in this Category"):
                    try:
                        with st.spinner("Fetching series..."):
                            series = integration.get_category_series(st.session_state.fred_category_id)
                        
                        if series:
                            st.write(f"Series in category '{category['name']}'")
                            
                            # Convert to DataFrame
                            df = pd.DataFrame(series)
                            st.dataframe(df)
                        else:
                            st.warning(f"No series found in category '{category['name']}'")
                    
                    except Exception as e:
                        st.error(f"Error fetching series: {e}")
                
                # Back button
                if st.session_state.fred_category_id != 0:
                    if st.button("Back to Parent Category"):
                        st.session_state.fred_category_id = category["parent_id"]
                        st.experimental_rerun()
            
            except Exception as e:
                st.error(f"Error fetching category: {e}")

def render_iex_cloud_explorer(integration):
    """Render the IEX Cloud data explorer."""
    st.subheader("IEX Cloud Data Explorer")
    
    # Create tabs for different data types
    tab1, tab2, tab3 = st.tabs(["Stock Data", "Company Info", "News"])
    
    with tab1:
        st.write("Explore stock data")
        
        col1, col2 = st.columns(2)
        
        with col1:
            symbol = st.text_input("Symbol", "AAPL", key="iex_stock_symbol")
        
        with col2:
            data_type = st.selectbox(
                "Data Type", 
                ["Quote", "Price", "Historical Prices"],
                index=0,
                key="iex_data_type"
            )
        
        if st.button("Get Data", key="iex_stock_button"):
            try:
                with st.spinner("Fetching data..."):
                    if data_type == "Quote":
                        data = integration.get_stock_quote(symbol)
                        
                        if data:
                            st.write(f"Quote for {symbol}")
                            
                            # Display key metrics
                            cols = st.columns(3)
                            cols[0].metric("Latest Price", f"${data.get('latestPrice', 'N/A')}")
                            cols[1].metric("Change", f"{data.get('change', 'N/A')}", f"{data.get('changePercent', 'N/A')*100:.2f}%")
                            cols[2].metric("Volume", f"{data.get('volume', 'N/A'):,}")
                            
                            # Display full data
                            with st.expander("Full Quote Data"):
                                st.json(data)
                        else:
                            st.warning(f"No data found for {symbol}")
                    
                    elif data_type == "Price":
                        price = integration.get_stock_price(symbol)
                        
                        st.write(f"Price for {
 elif data_type == "Historical Prices":
                    range_options = st.selectbox(
                        "Time Range",
                        ["1d", "5d", "1m", "3m", "6m", "ytd", "1y", "2y", "5y", "max"],
                        index=2
                    )
                    
                    data = integration.get_historical_prices(symbol, range=range_options)
                    
                    if data is not None and not data.empty:
                        st.write(f"Historical prices for {symbol}")
                        st.dataframe(data)
                        
                        # Plot data
                        if "close" in data.columns:
                            st.subheader("Price Chart")
                            st.line_chart(data["close"])
                    else:
                        st.warning(f"No historical data found for {symbol}")
        
        except Exception as e:
            st.error(f"Error fetching data: {e}")

with tab2:
    st.write("Explore company information")
    
    symbol = st.text_input("Symbol", "AAPL", key="iex_company_symbol")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("Company Info", key="iex_company_button"):
            try:
                with st.spinner("Fetching data..."):
                    data = integration.get_company(symbol)
                
                if data:
                    st.write(f"Company information for {symbol}")
                    
                    # Display key info
                    st.write(f"**{data.get('companyName', symbol)}**")
                    st.write(f"*{data.get('industry', 'N/A')} - {data.get('sector', 'N/A')}*")
                    st.write(data.get('description', 'No description available.'))
                    
                    # Display full data
                    with st.expander("Full Company Data"):
                        st.json(data)
                else:
                    st.warning(f"No company information found for {symbol}")
            
            except Exception as e:
                st.error(f"Error fetching data: {e}")
    
    with col2:
        if st.button("Financials", key="iex_financials_button"):
            try:
                with st.spinner("Fetching data..."):
                    data = integration.get_financials(symbol)
                
                if data:
                    st.write(f"Financial data for {symbol}")
                    
                    # Convert to DataFrame
                    df = pd.DataFrame(data)
                    st.dataframe(df)
                else:
                    st.warning(f"No financial data found for {symbol}")
            
            except Exception as e:
                st.error(f"Error fetching data: {e}")
    
    with col3:
        if st.button("Earnings", key="iex_earnings_button"):
            try:
                with st.spinner("Fetching data..."):
                    data = integration.get_earnings(symbol)
                
                if data:
                    st.write(f"Earnings data for {symbol}")
                    
                    # Convert to DataFrame
                    df = pd.DataFrame(data)
                    st.dataframe(df)
                    
                    # Plot EPS
                    if "actualEPS" in df.columns and "fiscalPeriod" in df.columns:
                        chart_data = df[["fiscalPeriod", "actualEPS", "consensusEPS"]].copy()
                        chart_data.set_index("fiscalPeriod", inplace=True)
                        st.bar_chart(chart_data)
                else:
                    st.warning(f"No earnings data found for {symbol}")
            
            except Exception as e:
                st.error(f"Error fetching data: {e}")

with tab3:
    st.write("Explore news")
    
    news_type = st.radio("News Type", ["Company News", "Market News"], horizontal=True)
    
    if news_type == "Company News":
        symbol = st.text_input("Symbol", "AAPL", key="iex_news_symbol")
        
        if st.button("Get News", key="iex_company_news_button"):
            try:
                with st.spinner("Fetching news..."):
                    news = integration.get_news(symbol)
                
                if news:
                    st.write(f"News for {symbol}")
                    
                    for article in news:
                        with st.expander(article.get("headline", "No headline")):
                            st.write(f"**Source**: {article.get('source', 'Unknown')}")
                            st.write(f"**Date**: {article.get('datetime', 'Unknown')}")
                            st.write(article.get("summary", "No summary available."))
                            if article.get("url"):
                                st.write(f"[Read more]({article['url']})")
                else:
                    st.warning(f"No news found for {symbol}")
            
            except Exception as e:
                st.error(f"Error fetching news: {e}")
    
    else:  # Market News
        if st.button("Get Market News", key="iex_market_news_button"):
            try:
                with st.spinner("Fetching market news..."):
                    news = integration.get_market_news()
                
                if news:
                    st.write("Market News")
                    
                    for article in news:
                        with st.expander(article.get("headline", "No headline")):
                            st.write(f"**Source**: {article.get('source', 'Unknown')}")
                            st.write(f"**Date**: {article.get('datetime', 'Unknown')}")
                            st.write(article.get("summary", "No summary available."))
                            if article.get("url"):
                                st.write(f"[Read more]({article['url']})")
                else:
                    st.warning("No market news found")
            
            except Exception as e:
                st.error(f"Error fetching market news: {e}")
Update the main dashboard.py file to include the integrations page
def update_main_dashboard():
"""Add code to update the main dashboard.py file."""
dashboard_code = """

Add import for integrations
from dashboard_integrations import render_integrations_page, render_data_explorer

In the navigation section, add:
elif page == "Integrations":
render_integrations_page()
elif page == "Data Explorer":
render_data_explorer()

In the sidebar navigation, add:
page = st.sidebar.radio(
"Navigation",
["Market Data", "Portfolio Optimization", "Risk Assessment", "AI Analysis", "Data Explorer", "Integrations"]
)
"""

text
print("Add the following code to your dashboard.py file:")
print(dashboard_code)

return dashboard_code
if name == "main":
update_main_dashboard()

text

### 21. Create a Configuration Manager for Application Settings

**src/config/manager.py**:
import os
import json
from typing import Dict, Any, Optional
import logging
from pathlib import Path

Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(name)

class ConfigManager:
"""Manager for application configuration."""

text
def __init__(self, config_path: Optional[str] = None):
    """
    Initialize the configuration manager.
    
    Args:
        config_path: Path to the configuration file
    """
    self.config_path = config_path or os.path.join(os.path.expanduser("~"), ".openluminary", "config.json")
    self.config = {}
    self.load_config()

def load_config(self):
    """Load configuration from file."""
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(self.config_path), exist_ok=True)
    
    # Create empty config file if it doesn't exist
    if not os.path.exists(self.config_path):
        self._create_default_config()
    
    # Load config
    try:
        with open(self.config_path, "r") as f:
            self.config = json.load(f)
        
        logger.info(f"Loaded configuration from {self.config_path}")
    
    except Exception as e:
        logger.error(f"Failed to load configuration: {e}")
        self._create_default_config()

def _create_default_config(self):
    """Create default configuration."""
    self.config = {
        "general": {
            "theme": "light",
            "timezone": "UTC",
            "default_currency": "USD"
        },
        "data": {
            "cache_dir": os.path.join(os.path.expanduser("~"), ".openluminary", "cache"),
            "cache_expiry": 3600,  # 1 hour in seconds
            "default_start_date": "1y",  # 1 year ago
            "default_end_date": "today"
        },
        "portfolio": {
            "default_risk_free_rate": 0.03,  # 3%
            "rebalancing_frequency": "quarterly"
        },
        "risk": {
            "default_confidence_level": 0.95,  # 95%
            "default_time_horizon": 1,  # 1 day
            "stress_test_scenarios": {
                "market_crash": -0.15,  # -15%
                "tech_crash": -0.25,  # -25%
                "interest_rate_hike": -0.05  # -5%
            }
        },
        "ai": {
            "model_path": "",
            "use_thinking_mode": True,
            "max_tokens": 2048,
            "temperature": 0.2
        },
        "ui": {
            "show_welcome_screen": True,
            "default_page": "Market Data",
            "chart_style": "default"
        }
    }
    
    self.save_config()

def save_config(self):
    """Save configuration to file."""
    try:
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(self.config_path), exist_ok=True)
        
        with open(self.config_path, "w") as f:
            json.dump(self.config, f, indent=2)
        
        logger.info(f"Saved configuration to {self.config_path}")
    
    except Exception as e:
        logger.error(f"Failed to save configuration: {e}")

def get(self, section: str, key: str, default: Any = None) -> Any:
    """
    Get a configuration value.
    
    Args:
        section: Configuration section
        key: Configuration key
        default: Default value if not found
        
    Returns:
        Configuration value or default
    """
    return self.config.get(section, {}).get(key, default)

def set(self, section: str, key: str, value: Any):
    """
    Set a configuration value.
    
    Args:
        section: Configuration section
        key: Configuration key
        value: Configuration value
    """
    if section not in self.config:
        self.config[section] = {}
    
    self.config[section][key] = value
    self.save_config()

def get_section(self, section: str) -> Dict[str, Any]:
    """
    Get a configuration section.
    
    Args:
        section: Configuration section
        
    Returns:
        Configuration section or empty dict if not found
    """
    return self.config.get(section, {})

def set_section(self, section: str, values: Dict[str, Any]):
    """
    Set a configuration section.
    
    Args:
        section: Configuration section
        values: Configuration values
    """
    self.config[section] = values
    self.save_config()

def reset_to_defaults(self):
    """Reset configuration to defaults."""
    self._create_default_config()

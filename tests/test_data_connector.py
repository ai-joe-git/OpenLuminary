import unittest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from src.data_processing.data_connector import YahooFinanceConnector, DataConnectorFactory

class TestYahooFinanceConnector(unittest.TestCase):
    
    def setUp(self):
        self.connector = YahooFinanceConnector()
        self.test_symbols = ["AAPL", "MSFT"]
        self.end_date = datetime.now().strftime('%Y-%m-%d')
        self.start_date = (datetime.now() - timedelta(days=30)).strftime('%Y-%m-%d')
    
    def test_get_historical_prices(self):
        # Test getting historical prices
        data = self.connector.get_historical_prices(
            self.test_symbols, 
            self.start_date, 
            self.end_date
        )
        
        # Check that we got data for each symbol
        self.assertEqual(len(data), len(self.test_symbols))
        
        # Check that the data is a DataFrame with expected columns
        for symbol, df in data.items():
            self.assertIsInstance(df, pd.DataFrame)
            self.assertIn('Open', df.columns)
            self.assertIn('High', df.columns)
            self.assertIn('Low', df.columns)
            self.assertIn('Close', df.columns)
            
            # Check that the date range is correct
            self.assertTrue(df.index.min() >= pd.Timestamp(self.start_date))
            self.assertTrue(df.index.max() <= pd.Timestamp(self.end_date))
    
    def test_get_current_prices(self):
        # Test getting current prices
        prices = self.connector.get_current_prices(self.test_symbols)
        
        # Check that we got prices for each symbol
        self.assertEqual(len(prices), len(self.test_symbols))
        
        # Check that the prices are numeric
        for symbol, price in prices.items():
            self.assertIsInstance(price, (int, float))
            self.assertGreater(price, 0)
    
    def test_factory(self):
        # Test the factory creates the right connector
        yahoo_connector = DataConnectorFactory.get_connector("yahoo")
        self.assertIsInstance(yahoo_connector, YahooFinanceConnector)
        
        # Test with invalid connector type
        with self.assertRaises(ValueError):
            DataConnectorFactory.get_connector("invalid")

class TestDataConnectorCache(unittest.TestCase):
    
    def setUp(self):
        self.connector = YahooFinanceConnector()
        self.test_symbol = "AAPL"
        self.end_date = datetime.now().strftime('%Y-%m-%d')
        self.start_date = (datetime.now() - timedelta(days=30)).strftime('%Y-%m-%d')
    
    def test_cache_historical_prices(self):
        # First call should hit the API
        data1 = self.connector.get_historical_prices(
            [self.test_symbol], 
            self.start_date, 
            self.end_date
        )
        
        # Second call should use cache
        data2 = self.connector.get_historical_prices(
            [self.test_symbol], 
            self.start_date, 
            self.end_date
        )
        
        # Data should be identical
        pd.testing.assert_frame_equal(data1[self.test_symbol], data2[self.test_symbol])
        
        # Check that the cache was used (this is implementation-specific)
        cache_key = f"hist_{self.test_symbol}_{self.start_date}_{self.end_date}_1d"
        self.assertIn(cache_key, self.connector.cache)

if __name__ == '__main__':
    unittest.main()

# Getting Started with OpenLuminary

This guide will help you get started with OpenLuminary, an open-source AI-powered financial analysis platform.

## Prerequisites

- Python 3.9 or higher
- Git
- Basic knowledge of financial concepts
- (Optional) CUDA-compatible GPU for AI model fine-tuning

## Installation

### Option 1: Quick Start with pip

pip install openluminary

text

### Option 2: From Source

Clone the repository
git clone https://github.com/ai-joe-git/OpenLuminary.git
cd OpenLuminary

Create a virtual environment
python -m venv venv
source venv/bin/activate # On Windows: venv\Scripts\activate

Install the package in development mode
pip install -e .

text

## Basic Usage

### Running the Dashboard

Start the Streamlit dashboard
streamlit run dashboard.py

text

This will launch the OpenLuminary dashboard in your default web browser.

### Using the API

Start the API server:

uvicorn src.api.main:app --reload

text

The API will be available at http://localhost:8000. You can access the API documentation at http://localhost:8000/docs.

### Example: Portfolio Optimization

from openluminary.data_processing.data_connector import DataConnectorFactory
from openluminary.portfolio_management.optimizer import PortfolioOptimizer
import pandas as pd

Initialize components
data_connector = DataConnectorFactory.get_connector("yahoo")
portfolio_optimizer = PortfolioOptimizer()

Get historical data
symbols = ["AAPL", "MSFT", "GOOGL", "AMZN"]
start_date = "2022-01-01"
end_date = "2023-01-01"

market_data = data_connector.get_historical_prices(symbols, start_date, end_date)

Extract close prices and calculate returns
close_prices = pd.DataFrame()
for symbol, data in market_data.items():
if data is not None and not data.empty:
close_prices[symbol] = data['Close']

returns = close_prices.pct_change().dropna()

Optimize portfolio
result = portfolio_optimizer.optimize_sharpe_ratio(returns)

Print results
print("Optimized Portfolio Weights:")
for asset, weight in result["weights"].items():
print(f"{asset}: {weight:.2%}")

print("\nPortfolio Performance:")
print(f"Expected Annual Return: {result['performance']['return']:.2%}")
print(f"Annual Volatility: {result['performance']['volatility']:.2%}")
print(f"Sharpe Ratio: {result['performance']['sharpe_ratio']:.2f}")

text

### Example: Risk Assessment

from openluminary.data_processing.data_connector import DataConnectorFactory
from openluminary.risk_assessment.advanced_risk import AdvancedRiskAssessment
import numpy as np
import pandas as pd

Initialize components
data_connector = DataConnectorFactory.get_connector("yahoo")
risk_assessment = AdvancedRiskAssessment()

Get historical data
symbols = ["AAPL", "MSFT", "GOOGL", "AMZN"]
weights = [0.3, 0.3, 0.2, 0.2] # Portfolio weights
start_date = "2022-01-01"
end_date = "2023-01-01"

market_data = data_connector.get_historical_prices(symbols, start_date, end_date)

Extract close prices and calculate returns
close_prices = pd.DataFrame()
for symbol, data in market_data.items():
if data is not None and not data.empty:
close_prices[symbol] = data['Close']

returns = close_prices.pct_change().dropna()

Calculate risk metrics
risk_metrics = risk_assessment.calculate_risk_metrics(returns, np.array(weights))

Calculate VaR
var_result = risk_assessment.calculate_var(
returns, np.array(weights), confidence_level=0.95, method="historical"
)

Print results
print("Risk Metrics:")
print(f"Expected Annual Return: {risk_metrics['mean_return']:.2%}")
print(f"Annual Volatility: {risk_metrics['volatility']:.2%}")
print(f"Sharpe Ratio: {risk_metrics['sharpe_ratio']:.2f}")
print(f"Maximum Drawdown: {risk_metrics['max_drawdown']:.2%}")
print(f"Value at Risk (95%): {var_result['var']:.2%}")

text

## Using the AI Features

To use the AI-powered features of OpenLuminary, you need to have access to the Qwen3 model. There are several options:

1. **Use a pre-fine-tuned model**: We provide a fine-tuned model for financial analysis.
2. **Fine-tune your own model**: Use our fine-tuning pipeline to create a specialized model.
3. **Use API access**: Connect to a hosted model through our API.

### Example: AI-Powered Analysis

from openluminary.models.qwen_interface import Qwen3Interface

Initialize the Qwen3 interface
qwen = Qwen3Interface(use_thinking_mode=True)

Perform portfolio optimization analysis
analysis_data = {
"portfolio": {
"AAPL": 0.25,
"MSFT": 0.25,
"GOOGL": 0.25,
"AMZN": 0.25
},
"risk_tolerance": "moderate",
"investment_horizon": "long-term",
"constraints": "maximum allocation per asset: 40%"
}

result = qwen.analyze_financial_data(
data=analysis_data,
analysis_type="portfolio_optimization"
)

print(result)

text

## Next Steps

- Explore the [API Documentation](api_reference.md)
- Learn about [Advanced Features](advanced_features.md)
- Contribute to the project by following our [Contribution Guidelines](../CONTRIBUTING.md)

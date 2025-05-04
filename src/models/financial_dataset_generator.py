import os
import pandas as pd
import numpy as np
import json
import random
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional, Union
import logging
import yfinance as yf
import argparse

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class FinancialDatasetGenerator:
    """Generate synthetic financial datasets for fine-tuning LLMs."""
    
    def __init__(self, output_dir: str = "data/training"):
        """
        Initialize the financial dataset generator.
        
        Args:
            output_dir: Directory to save generated datasets
        """
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        # Load real tickers for more realistic data
        self.sp500_tickers = self._load_sp500_tickers()
        self.sectors = self._load_sectors()
        
        # Templates for different financial analysis tasks
        self.templates = {
            "portfolio_optimization": self._generate_portfolio_optimization_sample,
            "risk_assessment": self._generate_risk_assessment_sample,
            "market_prediction": self._generate_market_prediction_sample,
            "company_analysis": self._generate_company_analysis_sample,
            "financial_planning": self._generate_financial_planning_sample,
            "asset_allocation": self._generate_asset_allocation_sample,
            "technical_analysis": self._generate_technical_analysis_sample,
            "economic_analysis": self._generate_economic_analysis_sample
        }
    
    def _load_sp500_tickers(self) -> List[str]:
        """Load S&P 500 tickers or fallback to a default list."""
        try:
            # Try to get actual S&P 500 tickers
            sp500 = pd.read_html('https://en.wikipedia.org/wiki/List_of_S%26P_500_companies')[0]
            tickers = sp500['Symbol'].tolist()
            logger.info(f"Loaded {len(tickers)} S&P 500 tickers")
            return tickers
        except Exception as e:
            logger.warning(f"Failed to load S&P 500 tickers: {e}")
            # Fallback to a default list of common tickers
            default_tickers = [
                "AAPL", "MSFT", "AMZN", "GOOGL", "META", "TSLA", "NVDA", "JPM", "V", "PG",
                "UNH", "HD", "BAC", "XOM", "CSCO", "PFE", "CMCSA", "KO", "PEP", "INTC",
                "VZ", "ADBE", "NFLX", "CRM", "ABT", "MRK", "DIS", "WMT", "MCD", "TMO"
            ]
            logger.info(f"Using {len(default_tickers)} default tickers")
            return default_tickers
    
    def _load_sectors(self) -> Dict[str, List[str]]:
        """Load sector classifications or fallback to a default mapping."""
        try:
            # Try to get actual sector classifications
            sectors = {}
            for ticker in self.sp500_tickers[:100]:  # Limit to 100 to avoid rate limiting
                try:
                    stock = yf.Ticker(ticker)
                    info = stock.info
                    sector = info.get('sector', 'Unknown')
                    if sector not in sectors:
                        sectors[sector] = []
                    sectors[sector].append(ticker)
                except Exception:
                    pass
            
            if sectors:
                logger.info(f"Loaded {len(sectors)} sectors with {sum(len(v) for v in sectors.values())} tickers")
                return sectors
            else:
                raise ValueError("No sectors loaded")
        except Exception as e:
            logger.warning(f"Failed to load sector classifications: {e}")
            # Fallback to a default sector mapping
            default_sectors = {
                "Technology": ["AAPL", "MSFT", "GOOGL", "META", "NVDA", "CSCO", "ADBE", "INTC", "CRM", "NFLX"],
                "Healthcare": ["UNH", "PFE", "ABT", "MRK", "TMO", "JNJ", "ABBV", "LLY", "AMGN", "CVS"],
                "Financials": ["JPM", "BAC", "V", "MA", "GS", "MS", "BLK", "C", "AXP", "WFC"],
                "Consumer": ["AMZN", "PG", "KO", "PEP", "WMT", "MCD", "DIS", "HD", "NKE", "SBUX"],
                "Energy": ["XOM", "CVX", "COP", "SLB", "EOG", "PSX", "VLO", "OXY", "MPC", "KMI"],
                "Industrials": ["GE", "HON", "UPS", "BA", "CAT", "MMM", "LMT", "RTX", "UNP", "FDX"],
                "Utilities": ["NEE", "DUK", "SO", "D", "AEP", "EXC", "SRE", "XEL", "ED", "WEC"],
                "Telecom": ["VZ", "T", "TMUS", "CMCSA", "CHTR", "LUMN", "DISH", "ATVI", "EA", "TTWO"]
            }
            logger.info(f"Using {len(default_sectors)} default sectors")
            return default_sectors
    
    def generate_dataset(
        self, 
        num_samples: int = 1000, 
        task_distribution: Optional[Dict[str, float]] = None,
        output_format: str = "jsonl",
        seed: int = 42
    ) -> str:
        """
        Generate a synthetic financial dataset for fine-tuning.
        
        Args:
            num_samples: Number of samples to generate
            task_distribution: Distribution of tasks (if None, uniform distribution)
            output_format: Format of the output file (jsonl or csv)
            seed: Random seed for reproducibility
            
        Returns:
            Path to the generated dataset
        """
        random.seed(seed)
        np.random.seed(seed)
        
        # Default to uniform distribution if not specified
        if task_distribution is None:
            task_distribution = {task: 1.0 / len(self.templates) for task in self.templates}
        
        # Normalize distribution
        total = sum(task_distribution.values())
        task_distribution = {k: v / total for k, v in task_distribution.items()}
        
        # Calculate number of samples per task
        task_counts = {}
        remaining = num_samples
        for task, prob in task_distribution.items():
            count = int(num_samples * prob)
            task_counts[task] = count
            remaining -= count
        
        # Distribute any remaining samples
        for task in sorted(task_counts.keys()):
            if remaining > 0:
                task_counts[task] += 1
                remaining -= 1
            else:
                break
        
        logger.info(f"Generating {num_samples} samples with distribution: {task_counts}")
        
        # Generate samples
        samples = []
        for task, count in task_counts.items():
            logger.info(f"Generating {count} samples for {task}")
            for _ in range(count):
                sample = self.templates[task]()
                samples.append(sample)
        
        # Shuffle samples
        random.shuffle(samples)
        
        # Save dataset
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = os.path.join(self.output_dir, f"financial_dataset_{timestamp}.{output_format}")
        
        if output_format == "jsonl":
            with open(output_path, "w") as f:
                for sample in samples:
                    f.write(json.dumps(sample) + "\n")
        elif output_format == "csv":
            df = pd.DataFrame(samples)
            df.to_csv(output_path, index=False)
        else:
            raise ValueError(f"Unsupported output format: {output_format}")
        
        logger.info(f"Generated {len(samples)} samples and saved to {output_path}")
        return output_path
    
    def _generate_portfolio_optimization_sample(self) -> Dict[str, str]:
        """Generate a sample for portfolio optimization."""
        # Select random tickers
        num_assets = random.randint(5, 15)
        tickers = random.sample(self.sp500_tickers, num_assets)
        
        # Generate asset details
        assets = []
        for ticker in tickers:
            expected_return = round(random.uniform(0.02, 0.25), 4)
            volatility = round(random.uniform(0.05, 0.40), 4)
            assets.append(f"{ticker}: Expected annual return {expected_return:.2%}, volatility {volatility:.2%}")
        
        # Generate constraints
        constraints = []
        constraints.append(f"Maximum allocation per asset: {random.randint(20, 50)}%")
        constraints.append(f"Minimum allocation per asset: {random.randint(1, 10)}%")
        
        # Add sector constraints for some samples
        if random.random() < 0.7:
            num_sector_constraints = random.randint(1, 3)
            sectors = list(self.sectors.keys())
            for _ in range(num_sector_constraints):
                sector = random.choice(sectors)
                max_allocation = random.randint(20, 60)
                constraints.append(f"{sector} sector maximum allocation: {max_allocation}%")
        
        # Generate risk preferences
        risk_tolerance = random.choice(["very conservative", "conservative", "moderate", "aggressive", "very aggressive"])
        investment_horizon = random.choice(["short-term (1-2 years)", "medium-term (3-5 years)", "long-term (5+ years)"])
        
        # Create instruction
        instruction = f"""Optimize a portfolio with the following assets:
{chr(10).join(f"- {asset}" for asset in assets)}

Subject to these constraints:
{chr(10).join(f"- {constraint}" for constraint in constraints)}

Additional information:
- Risk tolerance: {risk_tolerance}
- Investment horizon: {investment_horizon}
- Investment objective: {random.choice(["capital preservation", "income", "balanced", "growth", "aggressive growth"])}

Please provide the optimal portfolio allocation that maximizes the Sharpe ratio while respecting the constraints. Include expected return, volatility, and Sharpe ratio of the optimized portfolio."""

        # Generate response
        # Create a realistic portfolio allocation
        total_allocation = 100
        allocations = []
        
        # Ensure minimum allocations
        min_allocation = int(constraints[1].split(": ")[1].rstrip("%"))
        max_allocation = int(constraints[0].split(": ")[1].rstrip("%"))
        
        # Allocate minimum to each asset
        remaining = total_allocation - (min_allocation * num_assets)
        if remaining < 0:
            # Adjust minimum if not possible
            min_allocation = total_allocation // num_assets
            remaining = total_allocation - (min_allocation * num_assets)
        
        base_allocations = [min_allocation] * num_assets
        
        # Distribute remaining allocation randomly
        while remaining > 0:
            idx = random.randint(0, num_assets - 1)
            if base_allocations[idx] < max_allocation:
                increment = min(remaining, random.randint(1, 5), max_allocation - base_allocations[idx])
                base_allocations[idx] += increment
                remaining -= increment
        
        # Create allocation text
        for i, ticker in enumerate(tickers):
            allocations.append(f"{ticker}: {base_allocations[i]}%")
        
        # Calculate "expected" portfolio metrics
        portfolio_return = round(random.uniform(0.06, 0.18), 4)
        portfolio_volatility = round(random.uniform(0.08, 0.25), 4)
        sharpe_ratio = round(portfolio_return / portfolio_volatility, 2)
        
        response = f"""Based on the provided assets, constraints, and investment preferences, I've optimized the portfolio to maximize the Sharpe ratio.

## Optimal Portfolio Allocation

{chr(10).join(f"- {allocation}" for allocation in allocations)}

## Expected Portfolio Performance

- Expected Annual Return: {portfolio_return:.2%}
- Annual Volatility: {portfolio_volatility:.2%}
- Sharpe Ratio: {sharpe_ratio}

## Rationale

This allocation balances risk and return based on your {risk_tolerance} risk tolerance and {investment_horizon} investment horizon. The portfolio is diversified across multiple assets while respecting all specified constraints.

The optimization process considered the correlation between assets to reduce overall portfolio volatility. Assets with higher expected returns were given preference where their volatility contribution to the portfolio was acceptable.

## Recommendation

I recommend rebalancing this portfolio {random.choice(["quarterly", "semi-annually", "annually"])} to maintain the target allocation. If market conditions change significantly, a re-optimization may be warranted."""

        return {
            "instruction": instruction,
            "response": response
        }
    
    def _generate_risk_assessment_sample(self) -> Dict[str, str]:
        """Generate a sample for risk assessment."""
        # Select random tickers
        num_assets = random.randint(5, 12)
        tickers = random.sample(self.sp500_tickers, num_assets)
        
        # Generate portfolio weights
        weights = []
        total = 0
        for ticker in tickers:
            weight = random.randint(5, 30)
            total += weight
            weights.append(weight)
        
        # Normalize weights to sum to 100
        weights = [round(w * 100 / total) for w in weights]
        
        # Ensure weights sum to 100
        diff = 100 - sum(weights)
        weights[0] += diff
        
        # Create portfolio
        portfolio = []
        for i, ticker in enumerate(tickers):
            portfolio.append(f"{ticker}: {weights[i]}%")
        
        # Generate risk metrics
        var_95 = round(random.uniform(0.015, 0.035), 4)
        var_99 = round(random.uniform(0.025, 0.055), 4)
        cvar_95 = round(random.uniform(var_95 * 1.2, var_95 * 1.5), 4)
        max_drawdown = round(random.uniform(0.15, 0.40), 4)
        
        # Create instruction
        instruction = f"""Assess the risk of the following portfolio:

{chr(10).join(f"- {asset}" for asset in portfolio)}

Please provide a comprehensive risk assessment including:
1. Value at Risk (VaR) at 95% and 99% confidence levels
2. Conditional VaR (Expected Shortfall)
3. Maximum drawdown analysis
4. Stress test results under different market scenarios
5. Key risk factors and vulnerabilities
6. Recommendations for risk mitigation"""

        # Generate response
        response = f"""# Comprehensive Risk Assessment

## Portfolio Overview

I've analyzed your portfolio consisting of {num_assets} assets with the following allocation:

{chr(10).join(f"- {asset}" for asset in portfolio)}

## Key Risk Metrics

| Metric | Value | Interpretation |
|--------|-------|---------------|
| Value at Risk (95%) | {var_95:.2%} | There's a 5% chance of losing {var_95:.2%} or more in a single day |
| Value at Risk (99%) | {var_99:.2%} | There's a 1% chance of losing {var_99:.2%} or more in a single day |
| Expected Shortfall (95%) | {cvar_95:.2%} | In the worst 5% of cases, the average loss would be {cvar_95:.2%} |
| Maximum Drawdown | {max_drawdown:.2%} | The portfolio could experience a {max_drawdown:.2%} decline from peak to trough |

## Stress Test Results

I've analyzed how your portfolio would perform under various market scenarios:

1. **Market Crash Scenario (-{random.randint(30, 40)}% broad market decline)**
   - Portfolio Impact: -{random.randint(25, 45)}%
   - Most Affected Holdings: {", ".join(random.sample(tickers, 3))}

2. **Sector Correction Scenario**
   - {random.choice(list(self.sectors.keys()))} Sector Decline of {random.randint(15, 25)}%
   - Portfolio Impact: -{random.randint(8, 18)}%
   - Most Affected Holdings: {", ".join(random.sample(tickers, 2))}

3. **Interest Rate Hike Scenario (+{random.randint(100, 200)} basis points)**
   - Portfolio Impact: -{random.randint(5, 15)}%
   - Most Affected Holdings: {", ".join(random.sample(tickers, 3))}

4. **Inflation Surge Scenario (+{random.randint(2, 4)}% above expectations)**
   - Portfolio Impact: -{random.randint(7, 17)}%
   - Most Affected Holdings: {", ".join(random.sample(tickers, 2))}

## Key Risk Factors

1. **Concentration Risk**
   - {random.choice(list(self.sectors.keys()))} sector represents {random.randint(30, 60)}% of your portfolio
   - Top 3 holdings account for {random.randint(40, 70)}% of total portfolio

2. **Market Risk**
   - Portfolio Beta: {round(random.uniform(0.8, 1.3), 2)}
   - Correlation to S&P 500: {round(random.uniform(0.65, 0.9), 2)}

3. **Liquidity Risk**
   - {random.randint(85, 98)}% of holdings are highly liquid
   - {random.randint(2, 15)}% may face liquidity challenges in stressed markets

4. **Factor Exposures**
   - High exposure to {random.choice(["momentum", "growth", "value", "quality"])} factor
   - Underexposure to {random.choice(["defensive", "dividend", "low volatility"])} factor

## Risk Mitigation Recommendations

1. **Diversification Opportunities**
   - Consider adding exposure to {random.choice(["international markets", "fixed income", "alternative assets"])}
   - Reduce concentration in {random.choice(list(self.sectors.keys()))} sector by {random.randint(10, 20)}%

2. **Hedging Strategies**
   - Consider implementing a {random.randint(5, 15)}% allocation to {random.choice(["inverse ETFs", "put options", "market-neutral strategies"])}
   - This could reduce portfolio beta to approximately {round(random.uniform(0.6, 0.9), 2)}

3. **Position Sizing Adjustments**
   - Reduce position in {random.choice(tickers)} by {random.randint(5, 15)}%
   - Increase allocation to {random.choice(tickers)} by {random.randint(3, 10)}%

4. **Risk Monitoring**
   - Implement regular stress testing on a {random.choice(["monthly", "quarterly"])} basis
   - Set up alerts for when key risk metrics exceed predetermined thresholds

By implementing these recommendations, you could potentially reduce your portfolio's Value at Risk by {random.randint(15, 30)}% while maintaining similar return expectations."""

        return {
            "instruction": instruction,
            "response": response
        }
    
    def _generate_market_prediction_sample(self) -> Dict[str, str]:
        """Generate a sample for market prediction."""
        # Select market index or asset
        market_asset = random.choice([
            "S&P 500", "NASDAQ Composite", "Dow Jones Industrial Average", 
            "Russell 2000", "Bitcoin", "Gold", "10-Year Treasury Yield", "EUR/USD"
        ])
        
        # Select timeframe
        timeframe = random.choice(["1 month", "3 months", "6 months", "1 year"])
        
        # Generate market data
        current_price = round(random.uniform(100, 5000), 2) if market_asset != "10-Year Treasury Yield" else round(random.uniform(1.5, 5.0), 2)
        moving_avg_50d = round(current_price * random.uniform(0.9, 1.1), 2)
        moving_avg_200d = round(current_price * random.uniform(0.85, 1.15), 2)
        rsi = round(random.uniform(30, 70), 1)
        volume = round(random.uniform(0.7, 1.3), 2)
        
        # Generate economic indicators
        gdp_growth = round(random.uniform(-1.0, 5.0), 1)
        inflation = round(random.uniform(1.5, 8.0), 1)
        unemployment = round(random.uniform(3.0, 7.0), 1)
        fed_rate = round(random.uniform(0.25, 5.0), 2)
        consumer_sentiment = round(random.uniform(60, 110), 1)
        
        # Create instruction
        instruction = f"""Based on the following market data and economic indicators, predict the likely movement of {market_asset} over the next {timeframe}:

Market Data for {market_asset}:
- Current price: ${current_price if market_asset != "10-Year Treasury Yield" else str(current_price) + "%"}
- 50-day moving average: ${moving_avg_50d if market_asset != "10-Year Treasury Yield" else str(moving_avg_50d) + "%"}
- 200-day moving average: ${moving_avg_200d if market_asset != "10-Year Treasury Yield" else str(moving_avg_200d) + "%"}
- RSI: {rsi}
- Trading volume: {volume}x average

Economic Indicators:
- GDP Growth: {gdp_growth}%
- Inflation Rate: {inflation}%
- Unemployment: {unemployment}%
- Federal Funds Rate: {fed_rate}%
- Consumer Sentiment Index: {consumer_sentiment}

Please provide a detailed market prediction including:
1. Overall market direction and expected price range
2. Key factors supporting this prediction
3. Potential catalysts to monitor
4. Risk factors that could alter this outlook
5. Sector or asset implications
6. Confidence level in the prediction"""

        # Determine prediction based on indicators
        bullish_factors = 0
        if moving_avg_50d > moving_avg_200d: bullish_factors += 1
        if current_price > moving_avg_50d: bullish_factors += 1
        if rsi < 70 and rsi > 40: bullish_factors += 1
        if gdp_growth > 2.0: bullish_factors += 1
        if inflation < 4.0: bullish_factors += 1
        if consumer_sentiment > 80: bullish_factors += 1
        
        # Determine outlook based on bullish factors
        if bullish_factors >= 5:
            outlook = "strongly bullish"
            expected_change = round(random.uniform(8, 15), 1)
        elif bullish_factors == 4:
            outlook = "moderately bullish"
            expected_change = round(random.uniform(4, 8), 1)
        elif bullish_factors == 3:
            outlook = "neutral with slight bullish bias"
            expected_change = round(random.uniform(0, 4), 1)
        elif bullish_factors == 2:
            outlook = "neutral with slight bearish bias"
            expected_change = round(random.uniform(-4, 0), 1)
        elif bullish_factors == 1:
            outlook = "moderately bearish"
            expected_change = round(random.uniform(-8, -4), 1)
        else:
            outlook = "strongly bearish"
            expected_change = round(random.uniform(-15, -8), 1)
        
        # Confidence level based on how extreme indicators are
        confidence = random.randint(60, 85)
        
        # Generate response
        response = f"""# Market Prediction: {market_asset} over the next {timeframe}

## Executive Summary

Based on comprehensive analysis of current market conditions, economic indicators, and technical factors, my prediction for the {market_asset} over the next {timeframe} is:

**{outlook.title()} with an expected {expected_change:+.1f}% movement**

Confidence level: {confidence}%

## Key Drivers

### Technical Factors
The {market_asset} is currently trading {'above' if current_price > moving_avg_50d else 'below'} its 50-day moving average and {'above' if current_price > moving_avg_200d else 'below'} its 200-day moving average, indicating a {'positive' if current_price > moving_avg_50d and current_price > moving_avg_200d else 'negative' if current_price < moving_avg_50d and current_price < moving_avg_200d else 'mixed'} technical picture. The RSI at {rsi} suggests the asset is {'overbought' if rsi > 70 else 'oversold' if rsi < 30 else 'neither overbought nor oversold'}.

### Economic Environment
GDP growth at {gdp_growth}% indicates {'strong' if gdp_growth > 3 else 'moderate' if gdp_growth > 1.5 else 'weak'} economic expansion. Inflation at {inflation}% is {'well above' if inflation > 4 else 'slightly above' if inflation > 2.5 else 'near'} the Federal Reserve's target. The unemployment rate of {unemployment}% is {'low' if unemployment < 4 else 'moderate' if unemployment < 6 else 'high'} by historical standards.

### Monetary Policy
With the Federal Funds Rate at {fed_rate}%, monetary policy is {'highly accommodative' if fed_rate < 1 else 'accommodative' if fed_rate < 2.5 else 'neutral' if fed_rate < 4 else 'restrictive'}. Based on current inflation and growth data, the Federal Reserve is likely to {'cut rates' if inflation < 2.5 and gdp_growth < 2 else 'maintain current rates' if inflation < 4 and gdp_growth < 3 else 'raise rates'} in the coming months.

## Price Targets

For the {timeframe} horizon, I project the following price ranges for {market_asset}:

- Bearish scenario: ${round(current_price * (1 + (expected_change - 5) / 100), 2) if market_asset != "10-Year Treasury Yield" else str(round(current_price + (expected_change - 5) / 100, 2)) + "%"}
- Base case: ${round(current_price * (1 + expected_change / 100), 2) if market_asset != "10-Year Treasury Yield" else str(round(current_price + expected_change / 100, 2)) + "%"}
- Bullish scenario: ${round(current_price * (1 + (expected_change + 5) / 100), 2) if market_asset != "10-Year Treasury Yield" else str(round(current_price + (expected_change + 5) / 100, 2)) + "%"}

## Catalysts to Monitor

1. **Federal Reserve Policy Decisions**
   - Next meeting on {(datetime.now() + timedelta(days=random.randint(10, 60))).strftime('%B %d, %Y')}
   - Market pricing in {'rate cuts' if fed_rate > 3 and inflation < 3 else 'rate hikes' if inflation > 4 else 'no change'} of {random.randint(25, 75)} basis points

2. **Corporate Earnings Season**
   - Q{random.randint(1, 4)} earnings expected to show {'growth' if gdp_growth > 2 else 'contraction'} of {random.randint(3, 15)}% year-over-year
   - Key sectors to watch: {', '.join(random.sample(list(self.sectors.keys()), 3))}

3. **Economic Data Releases**
   - {random.choice(['Inflation', 'Employment', 'GDP', 'Retail Sales'])} report on {(datetime.now() + timedelta(days=random.randint(5, 30))).strftime('%B %d, %Y')}
   - {random.choice(['Consumer Confidence', 'Manufacturing PMI', 'Housing Data', 'Trade Balance'])} report on {(datetime.now() + timedelta(days=random.randint(5, 30))).strftime('%B %d, %Y')}

## Risk Factors

1. **{'Inflation Persistence' if inflation > 3 else 'Deflationary Pressures' if inflation < 1.5 else 'Inflation Volatility'}**
   - Impact: Could force {'more aggressive' if inflation > 3 else 'unexpected'} monetary policy adjustments
   - Probability: {random.randint(20, 70)}%

2. **Geopolitical Tensions**
   - Ongoing situations in {random.choice(['Eastern Europe', 'Middle East', 'Asia Pacific', 'Latin America'])}
   - Potential impact on {random.choice(['energy prices', 'supply chains', 'trade relations', 'market sentiment'])}
   - Probability of escalation: {random.randint(10, 60)}%

3. **{'Economic Slowdown' if gdp_growth > 1 else 'Recession Risks' if gdp_growth < 1 else 'Growth Sustainability'}**
   - Leading indicators showing {'improvement' if gdp_growth > 2 else 'deterioration' if gdp_growth < 1 else 'mixed signals'}
   - Probability: {random.randint(15, 65)}%

## Sector Implications

Based on this outlook, the following sectors are likely to {'outperform' if expected_change > 0 else 'be more defensive'}:

1. {random.choice(list(self.sectors.keys()))} - Due to {'growth prospects' if expected_change > 0 else 'defensive characteristics'}
2. {random.choice(list(self.sectors.keys()))} - Benefiting from {'current momentum' if expected_change > 0 else 'lower sensitivity to economic cycles'}
3. {random.choice(list(self.sectors.keys()))} - Supported by {'favorable valuations' if expected_change > 0 else 'stable demand'}

Sectors likely to {'lag' if expected_change > 0 else 'face challenges'}:

1. {random.choice(list(self.sectors.keys()))} - Due to {'valuation concerns' if expected_change > 0 else 'cyclical exposure'}
2. {random.choice(list(self.sectors.keys()))} - Facing {'competitive pressures' if expected_change > 0 else 'margin compression'}

## Trading Strategy Implications

For investors with a {timeframe} horizon:

1. {'Consider increasing equity exposure' if expected_change > 3 else 'Maintain current equity allocation' if expected_change > -3 else 'Consider reducing equity exposure'}
2. {'Focus on growth sectors' if expected_change > 5 else 'Balance growth and value' if expected_change > 0 else 'Emphasize defensive sectors and quality factors'}
3. {'Consider reducing fixed income duration' if expected_change > 0 and inflation > 3 else 'Maintain balanced duration exposure' if expected_change > -3 or inflation < 3 else 'Consider increasing fixed income allocation'}

This analysis is based on current data and subject to revision as new information becomes available."""

        return {
            "instruction": instruction,
            "response": response
        }
    
    def _generate_company_analysis_sample(self) -> Dict[str, str]:
        """Generate a sample for company analysis."""
        # Select a random ticker
        ticker = random.choice(self.sp500_tickers)
        
        # Generate company metrics
        market_cap = round(random.uniform(1, 2000), 1)
        market_cap_unit = random.choice(["billion", "million"])
        if market_cap_unit == "billion" and market_cap < 10:
            pe_ratio = round(random.uniform(15, 50), 1)
        else:
            pe_ratio = round(random.uniform(5, 30), 1)
        
        revenue_growth = round(random.uniform(-5, 30), 1)
        profit_margin = round(random.uniform(5, 30), 1)
        debt_to_equity = round(random.uniform(0.1, 2.0), 2)
        dividend_yield = round(random.uniform(0, 5), 2)
        
        # Determine sector
        company_sector = None
        for sector, tickers in self.sectors.items():
            if ticker in tickers:
                company_sector = sector
                break
        
        if not company_sector:
            company_sector = random.choice(list(self.sectors.keys()))
        
        # Create instruction
        instruction = f"""Provide a comprehensive analysis of {ticker} based on the following information:

Company Overview:
- Ticker: {ticker}
- Sector: {company_sector}
- Market Cap: ${market_cap} {market_cap_unit}

Financial Metrics:
- P/E Ratio: {pe_ratio}
- Revenue Growth (YoY): {revenue_growth}%
- Profit Margin: {profit_margin}%
- Debt-to-Equity Ratio: {debt_to_equity}
- Dividend Yield: {dividend_yield}%

Please include in your analysis:
1. Financial health assessment
2. Growth prospects evaluation
3. Competitive position analysis
4. Valuation assessment
5. Key risks and opportunities
6. Investment recommendation (Buy/Hold/Sell) with price target
7. Catalysts to monitor"""

        # Determine recommendation based on metrics
        score = 0
        if pe_ratio < 20: score += 1
        if revenue_growth > 10: score += 1
        if profit_margin > 15: score += 1
        if debt_to_equity < 1.0: score += 1
        if dividend_yield > 1.5: score += 1
        
        if score >= 4:
            recommendation = "Strong Buy"
            price_change = round(random.uniform(20, 40), 1)
        elif score == 3:
            recommendation = "Buy"
            price_change = round(random.uniform(10, 20), 1)
        elif score == 2:
            recommendation = "Hold"
            price_change = round(random.uniform(-5, 10), 1)
        elif score == 1:
            recommendation = "Sell"
            price_change = round(random.uniform(-20, -5), 1)
        else:
            recommendation = "Strong Sell"
            price_change = round(random.uniform(-40, -20), 1)
        
        # Generate current price and target
        current_price = round(random.uniform(20, 500), 2)
        target_price = round(current_price * (1 + price_change/100), 2)
        
        # Generate response
        response = f"""# {ticker} Comprehensive Analysis

## Company Overview

{ticker} is a {company_sector} company with a market capitalization of ${market_cap} {market_cap_unit}. The company {'is a leader' if score >= 3 else 'operates'} in the {company_sector} sector, {'with strong market position' if score >= 3 else 'facing competitive challenges' if score <= 1 else 'with moderate market presence'}.

## Financial Health Assessment

Based on the provided metrics, {ticker}'s financial health appears to be {'strong' if debt_to_equity < 0.8 else 'concerning' if debt_to_equity > 1.5 else 'adequate'}:

- **Debt-to-Equity Ratio: {debt_to_equity}** - {'Well below' if debt_to_equity < 0.5 else 'Above' if debt_to_equity > 1.2 else 'In line with'} industry average
- **Profit Margin: {profit_margin}%** - {'Excellent' if profit_margin > 20 else 'Poor' if profit_margin < 8 else 'Adequate'} profitability
- **Dividend Yield: {dividend_yield}%** - {'Attractive' if dividend_yield > 3 else 'Minimal' if dividend_yield < 1 else 'Moderate'} shareholder returns

The company has {'a strong balance sheet with manageable debt levels' if debt_to_equity < 1.0 else 'significant leverage that could be concerning in a downturn' if debt_to_equity > 1.5 else 'a balanced financial structure with moderate debt'}. {'Cash flow generation appears strong based on profit margins.' if profit_margin > 15 else 'Cash flow may be constrained by lower margins.' if profit_margin < 10 else 'Cash flow is adequate but could be improved.'}

## Growth Prospects

{ticker}'s growth outlook appears {'excellent' if revenue_growth > 15 else 'poor' if revenue_growth < 0 else 'moderate'}:

- **Revenue Growth (YoY): {revenue_growth}%** - {'Significantly outpacing' if revenue_growth > 20 else 'Lagging behind' if revenue_growth < 5 else 'In line with'} the broader market
- **Industry Trends**: The {company_sector} sector is experiencing {'strong tailwinds' if score >= 3 else 'significant headwinds' if score <= 1 else 'mixed conditions'}
- **Growth Drivers**: {'New product innovations' if random.random() > 0.5 else 'Geographic expansion'}, {'market share gains' if random.random() > 0.5 else 'operational efficiencies'}, and {'strategic acquisitions' if random.random() > 0.5 else 'pricing power'}

The company is {'well-positioned to capitalize on industry trends' if score >= 3 else 'struggling to maintain growth momentum' if score <= 1 else 'working to navigate changing market dynamics'}.

## Competitive Position

{ticker}'s competitive position in the {company_sector} sector is {'strong' if score >= 4 else 'weak' if score <= 1 else 'moderate'}:

- **Market Share**: {'Leading position' if score >= 4 else 'Minor player' if score <= 1 else 'Mid-tier competitor'} with approximately {random.randint(5, 45 if score >= 3 else 15)}% of addressable market
- **Competitive Advantages**: {'Strong brand recognition' if random.random() > 0.5 else 'Technological leadership'}, {'economies of scale' if random.random() > 0.5 else 'proprietary processes'}, and {'extensive distribution network' if random.random() > 0.5 else 'customer loyalty'}
- **Competitive Threats**: {'New market entrants' if random.random() > 0.5 else 'Substitute products'}, {'pricing pressure' if random.random() > 0.5 else 'regulatory changes'}, and {'changing consumer preferences' if random.random() > 0.5 else 'technological disruption'}

The company {'has built sustainable competitive moats' if score >= 4 else 'faces significant competitive challenges' if score <= 1 else 'maintains a viable competitive position but lacks strong differentiation'}.

## Valuation Assessment

At current levels, {ticker}'s valuation appears {'attractive' if pe_ratio < 15 else 'stretched' if pe_ratio > 25 else 'reasonable'}:

- **P/E Ratio: {pe_ratio}** - {'Below' if pe_ratio < 18 else 'Above' if pe_ratio > 22 else 'In line with'} industry average of {round(pe_ratio * random.uniform(0.8, 1.2), 1)}
- **Forward P/E**: Approximately {round(pe_ratio * random.uniform(0.85, 0.95), 1)} based on earnings growth projections
- **PEG Ratio**: {round(pe_ratio / max(1, revenue_growth), 2)}, indicating the stock is {'undervalued' if pe_ratio / max(1, revenue_growth) < 1.2 else 'overvalued' if pe_ratio / max(1, revenue_growth) > 2 else 'fairly valued'} relative to growth

Based on a discounted cash flow analysis with a {random.randint(8, 12)}% discount rate and {random.randint(2, 4)}% terminal growth rate, the intrinsic value is estimated at ${target_price}.

## Key Risks and Opportunities

### Risks:
1. {'Economic slowdown impacting consumer spending' if company_sector in ['Consumer', 'Retail'] else 'Regulatory changes affecting operational costs' if company_sector in ['Healthcare', 'Financials', 'Energy'] else 'Technological disruption threatening current business model'}
2. {'Increasing competition from both traditional and new entrants' if revenue_growth > 10 else 'Margin pressure due to rising input costs' if profit_margin > 15 else 'Debt servicing challenges if interest rates rise significantly'}
3. {'Execution risk related to ongoing expansion initiatives' if revenue_growth > 15 else 'Customer concentration with top clients representing significant revenue' if random.random() > 0.7 else 'Foreign exchange exposure due to international operations'}

### Opportunities:
1. {'Expansion into adjacent markets with significant TAM' if revenue_growth > 10 else 'Operational efficiency initiatives to improve margins' if profit_margin < 15 else 'Strategic acquisitions to consolidate market position'}
2. {'Product innovation pipeline showing promise' if random.random() > 0.5 else 'International expansion opportunities in emerging markets'}
3. {'Digital transformation initiatives enhancing customer experience' if random.random() > 0.5 else 'Sustainability initiatives creating brand differentiation'}

## Investment Recommendation

**{recommendation} with a price target of ${target_price}** (representing {price_change:+.1f}% from current price of ${current_price})

This recommendation is based on:
- {'Attractive valuation relative to growth prospects' if pe_ratio / max(1, revenue_growth) < 1.5 else 'Concerning valuation metrics' if pe_ratio / max(1, revenue_growth) > 2 else 'Balanced risk-reward profile'}
- {'Strong financial position with growth momentum' if score >= 4 else 'Deteriorating fundamentals and competitive position' if score <= 1 else 'Stable financial performance with moderate growth'}
- {'Industry tailwinds supporting continued expansion' if score >= 3 else 'Industry headwinds creating significant challenges' if score <= 1 else 'Mixed industry conditions requiring careful navigation'}

## Catalysts to Monitor

1. **Upcoming Events**:
   - Q{random.randint(1, 4)} earnings release on {(datetime.now() + timedelta(days=random.randint(10, 90))).strftime('%B %d, %Y')}
   - Investor Day scheduled for {(datetime.now() + timedelta(days=random.randint(30, 180))).strftime('%B %Y')}

2. **Key Metrics to Watch**:
   - {'Gross margin trends' if profit_margin < 20 else 'Revenue growth acceleration' if revenue_growth < 15 else 'Free cash flow conversion'}
   - {'New product adoption rates' if random.random() > 0.5 else 'International expansion progress'}
   - {'Operating expense ratio' if profit_margin < 15 else 'Return on invested capital'}

3. **External Factors**:
   - {'Regulatory decisions related to ' + company_sector if company_sector in ['Healthcare', 'Financials', 'Energy'] else 'Consumer spending trends' if company_sector in ['Consumer', 'Retail'] else 'Technology adoption cycles'}
   - {'Interest rate environment' if debt_to_equity > 1.2 else 'Commodity price movements' if company_sector in ['Energy', 'Materials'] else 'Competitive product launches'}

This analysis is based on current information and subject to revision as new data becomes available."""

        return {
            "instruction": instruction,
            "response": response
        }
    
    def _generate_financial_planning_sample(self) -> Dict[str, str]:
        """Generate a sample for financial planning."""
        # Generate personal financial details
        age = random.randint(25, 65)
        retirement_age = min(random.randint(55, 75), age + 10)
        annual_income = random.randint(50, 300) * 1000
        savings = random.randint(int(annual_income * 0.5), int(annual_income * 10))
        monthly_expenses = round(annual_income * random.uniform(0.3, 0.7) / 12, -2)
        debt = round(annual_income * random.uniform(0, 2), -3)
        
        # Generate financial goals
        goals = []
        if age < 40:
            if random.random() > 0.5:
                goals.append(f"Purchase a home within {random.randint(1, 5)} years (estimated cost: ${random.randint(300, 1500)}k)")
            if random.random() > 0.5:
                goals.append(f"Save for children's education (need ${random.randint(100, 300)}k in {random.randint(5, 18)} years)")
        
        goals.append(f"Retire at age {retirement_age} with ${random.randint(1, 5)}M in savings")
        
        if random.random() > 0.5:
            goals.append(f"Pay off ${debt:,.0f} in debt within {random.randint(1, 10)} years")
        
        if random.random() > 0.5:
            goals.append(f"Build an emergency fund of ${round(monthly_expenses * random.randint(3, 12), -3):,.0f}")
        
        # Generate current investments
        investments = []
        investment_total = savings * random.uniform(0.3, 0.9)
        remaining = investment_total
        
        if remaining > 0:
            stocks_amount = round(remaining * random.uniform(0.2, 0.8), -3)
            remaining -= stocks_amount
            investments.append(f"Stocks/ETFs: ${stocks_amount:,.0f} ({round(stocks_amount / investment_total * 100)}%)")
        
        if remaining > 0:
            bonds_amount = round(remaining * random.uniform(0.2, 0.8), -3)
            remaining -= bonds_amount
            investments.append(f"Bonds: ${bonds_amount:,.0f} ({round(bonds_amount / investment_total * 100)}%)")
        
        if remaining > 0:
            cash_amount = round(remaining, -3)
            investments.append(f"Cash/Money Market: ${cash_amount:,.0f} ({round(cash_amount / investment_total * 100)}%)")
        
        # Create instruction
        instruction = f"""Create a comprehensive financial plan based on the following personal information:

Personal Details:
- Age: {age}
- Annual Income: ${annual_income:,}
- Monthly Expenses: ${monthly_expenses:,.0f}
- Total Savings: ${savings:,}
- Current Debt: ${debt:,}

Financial Goals:
{chr(10).join(f"- {goal}" for goal in goals)}

Current Investments:
{chr(10).join(f"- {investment}" for investment in investments)}

Please provide a detailed financial plan including:
1. Assessment of current financial situation
2. Recommended budget and savings rate
3. Debt management strategy
4. Investment allocation recommendations
5. Retirement planning analysis
6. Insurance and estate planning considerations
7. Action steps prioritized by importance"""

        # Generate response
        # Calculate some financial metrics
        debt_to_income = round(debt / annual_income * 100)
        savings_rate = round(random.uniform(0.05, 0.30) * 100)
        recommended_savings_rate = min(round(random.uniform(0.15, 0.40) * 100), 40)
        retirement_years = retirement_age - age
        
        # Calculate retirement needs
        annual_retirement_expenses = round(monthly_expenses * 12 * 0.8, -3)
        retirement_nest_egg = round(annual_retirement_expenses * 25, -3)
        
        # Generate investment allocation based on age
        equity_allocation = max(100 - age, 30) if age < 60 else random.randint(30, 50)
        bond_allocation = 100 - equity_allocation
        
        response = f"""# Comprehensive Financial Plan

## 1. Current Financial Situation Assessment

Based on the information provided, here's an assessment of your current financial position:

### Strengths:
- **Income**: Your annual income of ${annual_income:,} is {'above' if annual_income > 100000 else 'near'} the national median.
- **Savings**: You have accumulated ${savings:,} in total savings, which is {'excellent' if savings > annual_income * 2 else 'good' if savings > annual_income else 'a starting point but needs growth'}.
- **{'Low debt-to-income ratio' if debt_to_income < 30 else 'Manageable debt level' if debt_to_income < 50 else 'High debt burden'}**: Your debt-to-income ratio is {debt_to_income}% ({debt_to_income < 36 and 'within' or 'exceeding'} the recommended maximum of 36%).

### Areas for Improvement:
- **{'Emergency Fund' if not any('emergency fund' in goal.lower() for goal in goals) else 'Savings Rate'}**: {'You should establish an emergency fund of 3-6 months of expenses.' if not any('emergency fund' in goal.lower() for goal in goals) else f'Your current savings rate of approximately {savings_rate}% should be increased to {recommended_savings_rate}% to meet your goals.'}
- **{'Debt Reduction' if debt_to_income > 30 else 'Investment Diversification'}**: {'Prioritize debt reduction to improve financial flexibility.' if debt_to_income > 30 else 'Your investment portfolio could benefit from additional diversification.'}
- **Retirement Planning**: Based on your current trajectory, {'you are on track to meet' if savings > age * annual_income * 0.1 else 'you may fall short of'} your retirement goal of ${random.randint(1, 5)}M by age {retirement_age}.

## 2. Recommended Budget and Savings Plan

### Monthly Budget Framework:
- **Take-Home Income**: ~${round(annual_income * 0.7 / 12, -2):,.0f} (after estimated taxes)
- **Essential Expenses**: ${monthly_expenses:,.0f} ({round(monthly_expenses / (annual_income * 0.7 / 12) * 100)}% of take-home pay)
- **Recommended Savings**: ${round(annual_income * 0.7 / 12 * recommended_savings_rate / 100, -2):,.0f} ({recommended_savings_rate}% of take-home pay)
- **Debt Repayment**: ${round(annual_income * 0.7 / 12 * min(debt_to_income, 20) / 100, -2):,.0f} ({min(debt_to_income, 20)}% of take-home pay)
- **Discretionary Spending**: ${round(annual_income * 0.7 / 12 - monthly_expenses - (annual_income * 0.7 / 12 * recommended_savings_rate / 100) - (annual_income * 0.7 / 12 * min(debt_to_income, 20) / 100), -2):,.0f} (remaining amount)

### Savings Allocation:
1. **Emergency Fund**: ${round(monthly_expenses * 6, -3):,.0f} (6 months of expenses) - {'Priority' if not any('emergency fund' in goal.lower() for goal in goals) else 'Already in progress'}
2. **Retirement Accounts**: ${round(annual_income * recommended_savings_rate / 100 * 0.6, -3):,.0f} annually (60% of savings)
3. **{'Home Down Payment' if any('home' in goal.lower() for goal in goals) else 'Education Fund' if any('education' in goal.lower() for goal in goals) else 'Short-Term Goals'}**: ${round(annual_income * recommended_savings_rate / 100 * 0.3, -3):,.0f} annually (30% of savings)
4. **Other Goals**: ${round(annual_income * recommended_savings_rate / 100 * 0.1, -3):,.0f} annually (10% of savings)

## 3. Debt Management Strategy

Your current debt of ${debt:,} represents a {'low' if debt_to_income < 30 else 'moderate' if debt_to_income < 50 else 'high'} burden relative to your income.

### Recommended Approach:
1. **{'High-Interest Debt Elimination' if debt > 0 else 'Maintain Debt-Free Status'}**:
   - {'Prioritize paying off any debt with interest rates above 6% (typically credit cards and personal loans)' if debt > 0 else 'Continue avoiding high-interest debt'}
   - {'Consider consolidating high-interest debt if your credit score allows favorable terms' if debt > annual_income * 0.5 else ''}

2. **{'Student Loan Strategy' if age < 40 and debt > annual_income * 0.5 else 'Mortgage Optimization' if debt > annual_income else 'Low-Interest Debt Management'}**:
   - {'Evaluate income-based repayment options and loan forgiveness programs if applicable' if age < 40 and debt > annual_income * 0.5 else 'Consider refinancing your mortgage if you can reduce your interest rate by at least 0.75%' if debt > annual_income else 'Maintain minimum payments on low-interest debt while prioritizing investments'}

3. **Debt Payoff Timeline**:
   - {'Aggressive payoff: ' + str(round(debt / (annual_income * 0.2))) + ' years' if debt_to_income > 50 else 'Standard payoff: ' + str(round(debt / (annual_income * 0.15))) + ' years'}
   - {'This timeline aligns with your goal to be debt-free within ' + str(random.randint(1, 10)) + ' years' if any('debt' in goal.lower() for goal in goals) else ''}

## 4. Investment Allocation Recommendations

Based on your age ({age}), goals, and risk tolerance, I recommend the following investment allocation:

### Recommended Portfolio:
- **Equities: {equity_allocation}%**
  - US Large Cap: {round(equity_allocation * 0.4)}%
  - US Mid/Small Cap: {round(equity_allocation * 0.2)}%
  - International Developed: {round(equity_allocation * 0.25)}%
  - Emerging Markets: {round(equity_allocation * 0.15)}%

- **Fixed Income: {bond_allocation}%**
  - Investment Grade Bonds: {round(bond_allocation * 0.6)}%
  - Treasury Inflation-Protected Securities: {round(bond_allocation * 0.2)}%
  - High-Yield Bonds: {round(bond_allocation * 0.1)}%
  - International Bonds: {round(bond_allocation * 0.1)}%

- **Alternative Investments: {0 if age > 60 else 5}%**
  - Real Estate Investment Trusts (REITs): {0 if age > 60 else 3}%
  - Commodities: {0 if age > 60 else 2}%

### Account Structure:
1. **Tax-Advantaged Accounts** (prioritize these first):
   - 401(k)/403(b): Contribute at least enough to get full employer match
   - Roth IRA: ${round(min(6000, annual_income * 0.05), -3):,.0f} annual contribution
   - HSA (if eligible): Maximize contributions for triple tax advantage

2. **Taxable Accounts**:
   - Hold tax-efficient investments like index ETFs
   - Consider municipal bonds for fixed income portion if in high tax bracket

## 5. Retirement Planning Analysis

### Retirement Goal:
- Target Retirement Age: {retirement_age}
- Years Until Retirement: {retirement_years}
- Estimated Annual Expenses in Retirement: ${annual_retirement_expenses:,} (80% of current expenses)
- Target Retirement Nest Egg: ${retirement_nest_egg:,} (25x annual expenses)

### Current Trajectory:
- Current Savings: ${savings:,}
- Projected Value at Retirement (current savings rate): ${round(savings * (1 + 0.07) ** retirement_years, -3):,}
- Projected Value at Retirement (recommended savings rate): ${round(savings * (1 + 0.07) ** retirement_years + annual_income * recommended_savings_rate / 100 * ((1 + 0.07) ** retirement_years - 1) / 0.07, -3):,}
- {'You are on track to meet your retirement goal' if savings * (1 + 0.07) ** retirement_years + annual_income * recommended_savings_rate / 100 * ((1 + 0.07) ** retirement_years - 1) / 0.07 > retirement_nest_egg else 'You need to increase your savings rate to meet your retirement goal'}

### Social Security Consideration:
- Estimated Monthly Benefit at Full Retirement Age: ${round(min(annual_income * 0.4 / 12, 3000), -2):,}
- This will cover approximately {round(min(annual_income * 0.4 / 12, 3000) * 12 / annual_retirement_expenses * 100)}% of your retirement expenses

## 6. Insurance and Estate Planning

### Insurance Recommendations:
1. **Life Insurance**: {'Consider term life insurance with coverage of ' + str(round(max(debt, annual_income * 10), -3)) + ' to protect dependents' if age < 60 else 'Minimal life insurance needed at your stage'}
2. **Disability Insurance**: Coverage for {'60-70% of your income' if age < 60 else 'remaining working years'}
3. **Health Insurance**: {'High-deductible plan with HSA' if age < 50 and annual_income > 100000 else 'Comprehensive coverage with lower deductibles'}
4. **Property & Casualty**: Ensure adequate coverage for home/auto with appropriate liability limits
5. **Umbrella Policy**: Consider ${round(max(1000000, annual_income * 2), -6):,} in coverage

### Estate Planning:
1. **Will**: {'Establish a will immediately' if not random.choice([True, False]) else 'Update your existing will to reflect current wishes'}
2. **Power of Attorney**: Designate financial and healthcare powers of attorney
3. **Advanced Healthcare Directive**: Document your medical treatment preferences
4. **Trust Consideration**: {'Consider establishing a revocable living trust to avoid probate' if savings > 500000 else 'Basic estate planning documents are sufficient at this stage'}
5. **Beneficiary Designations**: Review and update beneficiaries on all retirement accounts and insurance policies

## 7. Prioritized Action Steps

### Immediate Actions (Next 30 Days):
1. {'Establish emergency fund of $' + str(round(monthly_expenses * 3, -3)) + ' (3 months expenses)' if not any('emergency fund' in goal.lower() for goal in goals) else 'Increase retirement contributions to ' + str(recommended_savings_rate) + '% of income'}
2. {'Create a debt reduction plan focusing on highest interest debt first' if debt_to_income > 30 else 'Rebalance investment portfolio to recommended allocation'}
3. {'Review and update all insurance coverage' if random.choice([True, False]) else 'Establish basic estate planning documents'}

### Short-Term Actions (1-6 Months):
1. {'Complete emergency fund to reach $' + str(round(monthly_expenses * 6, -3)) + ' (6 months expenses)' if not any('emergency fund' in goal.lower() for goal in goals) else 'Optimize tax strategies for current year'}
2. {'Consolidate retirement accounts from previous employers' if random.choice([True, False]) else 'Establish dedicated savings for short-term goals'}
3. {'Review credit report and address any issues' if random.choice([True, False]) else 'Implement recommended investment allocation'}

### Medium-Term Actions (6-12 Months):
1. {'Increase retirement savings rate to target ' + str(recommended_savings_rate) + '%' if savings_rate < recommended_savings_rate else 'Consider additional tax-advantaged savings vehicles'}
2. {'Establish college savings accounts for children' if any('education' in goal.lower() for goal in goals) else 'Research additional income opportunities'}
3. {'Complete comprehensive estate plan' if savings > 500000 else 'Develop detailed plan for major financial goals'}

This financial plan should be reviewed annually or whenever significant life changes occur. Consider working with a certified financial planner for personalized advice tailored to your specific situation."""

        return {
            "instruction": instruction,
            "response": response
        }
    
    def _generate_asset_allocation_sample(self) -> Dict[str, str]:
        """Generate a sample for asset allocation."""
        # Generate investor profile
        age = random.randint(25, 75)
        risk_tolerance = random.choice(["conservative", "moderately conservative", "moderate", "moderately aggressive", "aggressive"])
        investment_amount = random.randint(50, 5000) * 1000
        investment_horizon = random.choice(["short-term (1-3 years)", "medium-term (3-7 years)", "long-term (7+ years)"])
        
        # Generate investment goals
        goals = []
        if age < 40:
            if random.random() > 0.5:
                goals.append("Building wealth for retirement")
            if random.random() > 0.5:
                goals.append("Saving for children's education")
            if random.random() > 0.5:
                goals.append("Purchasing a home in the next 5 years")
        elif age < 60:
            goals.append("Growing retirement nest egg")
            if random.random() > 0.5:
                goals.append("Funding children's college education")
            if random.random() > 0.5:
                goals.append("Building legacy wealth")
        else:
            goals.append("Generating retirement income")
            if random.random() > 0.5:
                goals.append("Preserving capital")
            if random.random() > 0.5:
                goals.append("Estate planning")
        
        # Create instruction
        instruction = f"""Recommend an optimal asset allocation for an investor with the following profile:

Investor Profile:
- Age: {age}
- Risk Tolerance: {risk_tolerance}
- Investment Amount: ${investment_amount:,}
- Investment - Investment Horizon: {investment_horizon}

Investment Goals:
{chr(10).join(f"- {goal}" for goal in goals)}

Please provide a detailed asset allocation recommendation including:
1. Recommended allocation across asset classes (stocks, bonds, cash, alternatives)
2. Specific sub-asset class allocations (e.g., large-cap, international, etc.)
3. Rationale for the recommended allocation
4. Specific investment vehicles to consider (ETFs, mutual funds, etc.)
5. Rebalancing strategy and timeline
6. How this allocation addresses the investor's specific goals"""

        # Generate response
        # Determine equity allocation based on age, risk tolerance, and horizon
        base_equity = max(100 - age, 30)
        
        # Adjust for risk tolerance
        risk_adjustments = {
            "conservative": -20,
            "moderately conservative": -10,
            "moderate": 0,
            "moderately aggressive": 10,
            "aggressive": 20
        }
        
        # Adjust for investment horizon
        horizon_adjustments = {
            "short-term (1-3 years)": -15,
            "medium-term (3-7 years)": 0,
            "long-term (7+ years)": 10
        }
        
        equity_allocation = max(min(base_equity + risk_adjustments[risk_tolerance] + horizon_adjustments[investment_horizon], 90), 20)
        bond_allocation = min(max(100 - equity_allocation - 5, 10), 70)
        cash_allocation = max(5, 100 - equity_allocation - bond_allocation)
        alternatives_allocation = max(0, 100 - equity_allocation - bond_allocation - cash_allocation)
        
        # Generate sub-allocations
        us_large_cap = round(equity_allocation * random.uniform(0.3, 0.5))
        us_mid_cap = round(equity_allocation * random.uniform(0.1, 0.2))
        us_small_cap = round(equity_allocation * random.uniform(0.05, 0.15))
        international_developed = round(equity_allocation * random.uniform(0.15, 0.3))
        emerging_markets = equity_allocation - us_large_cap - us_mid_cap - us_small_cap - international_developed
        
        us_govt_bonds = round(bond_allocation * random.uniform(0.3, 0.5))
        us_corporate_bonds = round(bond_allocation * random.uniform(0.2, 0.4))
        international_bonds = round(bond_allocation * random.uniform(0.1, 0.2))
        high_yield_bonds = bond_allocation - us_govt_bonds - us_corporate_bonds - international_bonds
        
        response = f"""# Optimal Asset Allocation Recommendation

## Executive Summary

Based on your profile as a {age}-year-old investor with a {risk_tolerance} risk tolerance, ${investment_amount:,} to invest, and a {investment_horizon} horizon, I recommend the following asset allocation:

- **Equities: {equity_allocation}%**
- **Fixed Income: {bond_allocation}%**
- **Cash & Equivalents: {cash_allocation}%**
- **Alternative Investments: {alternatives_allocation}%**

This allocation is designed to {'prioritize capital preservation while providing modest growth potential' if equity_allocation  0 else ""}
{f"- **Real Estate Investment Trusts (REITs): {round(alternatives_allocation * 0.6)}%" if alternatives_allocation > 0 else ""}
{f"- **Commodities: {round(alternatives_allocation * 0.4)}%" if alternatives_allocation > 0 else ""}

## Rationale for Recommended Allocation

This allocation is tailored to your specific circumstances:

1. **Age Consideration**: At {age}, you {'have a long runway for investment growth and can tolerate market volatility' if age  0 else ""}
{f"- **REITs**: Vanguard Real Estate ETF (VNQ) or Schwab US REIT ETF (SCHH)" if alternatives_allocation > 0 else ""}
{f"- **Commodities**: iShares S&P GSCI Commodity-Indexed Trust (GSG) or Invesco DB Commodity Index Tracking Fund (DBC)" if alternatives_allocation > 0 else ""}

## Implementation Strategy

1. **Dollar-Cost Averaging**: Given the current market environment, consider implementing this allocation over {'3-6 months' if investment_amount > 100000 else '1-3 months'} rather than all at once.

2. **Tax Considerations**:
   - Prioritize tax-advantaged accounts (401(k), IRA, HSA) for less tax-efficient investments (bonds, REITs)
   - Hold tax-efficient investments (index ETFs) in taxable accounts
   - {'Consider municipal bonds for the fixed income portion in taxable accounts' if investment_amount > 200000 else ''}

3. **Account Location Strategy**:
   - Tax-Advantaged Accounts: Bonds, REITs, actively managed funds
   - Taxable Accounts: Index ETFs, municipal bonds, tax-managed funds

## Rebalancing Strategy

I recommend a {'semi-annual' if risk_tolerance in ['conservative', 'moderately conservative'] else 'annual'} rebalancing schedule with the following parameters:

- **Threshold Rebalancing**: Rebalance when any asset class deviates by more than {'5%' if risk_tolerance in ['conservative', 'moderately conservative'] else '10%'} from its target allocation
- **Calendar Rebalancing**: Review the portfolio every {'6 months' if risk_tolerance in ['conservative', 'moderately conservative'] else '12 months'}
- **Life Event Rebalancing**: Reassess allocation after major life events or when investment goals change

## Monitoring and Adjustments

This allocation should be reviewed:
- Annually at minimum
- When major life events occur
- When investment goals change
- When approaching key milestones (e.g., {'retirement' if age > 50 else 'home purchase' if 'Purchasing a home in the next 5 years' in goals else 'education funding needs' if 'Saving for children\'s education' in goals else 'significant financial milestones'})

As you {'approach retirement' if age > 50 else 'age'}, gradually shift toward a more conservative allocation by increasing fixed income and decreasing equity exposure.

This recommendation provides a balanced approach to meeting your financial goals while aligning with your risk tolerance and investment horizon."""

        return {
            "instruction": instruction,
            "response": response
        }
    
    def _generate_technical_analysis_sample(self) -> Dict[str, str]:
        """Generate a sample for technical analysis."""
        # Select a random ticker
        ticker = random.choice(self.sp500_tickers)
        
        # Generate price data
        current_price = round(random.uniform(20, 500), 2)
        price_change = round(random.uniform(-5, 5), 2)
        previous_close = round(current_price - price_change, 2)
        
        # Generate technical indicators
        moving_avg_50 = round(current_price * random.uniform(0.9, 1.1), 2)
        moving_avg_200 = round(current_price * random.uniform(0.85, 1.15), 2)
        rsi = round(random.uniform(30, 70), 1)
        macd = round(random.uniform(-2, 2), 2)
        macd_signal = round(macd + random.uniform(-1, 1), 2)
        bollinger_upper = round(current_price * random.uniform(1.05, 1.15), 2)
        bollinger_lower = round(current_price * random.uniform(0.85, 0.95), 2)
        
        # Generate support and resistance levels
        support_1 = round(current_price * random.uniform(0.9, 0.95), 2)
        support_2 = round(support_1 * random.uniform(0.9, 0.95), 2)
        resistance_1 = round(current_price * random.uniform(1.05, 1.1), 2)
        resistance_2 = round(resistance_1 * random.uniform(1.05, 1.1), 2)
        
        # Create instruction
        instruction = f"""Perform a comprehensive technical analysis for {ticker} based on the following data:

Price Information:
- Current Price: ${current_price}
- Previous Close: ${previous_close}
- Daily Change: {price_change} ({round(price_change / previous_close * 100, 2)}%)

Technical Indicators:
- 50-Day Moving Average: ${moving_avg_50}
- 200-Day Moving Average: ${moving_avg_200}
- RSI (14-day): {rsi}
- MACD: {macd}
- MACD Signal Line: {macd_signal}
- Bollinger Bands: Upper ${bollinger_upper}, Lower ${bollinger_lower}

Support and Resistance:
- Support Levels: ${support_1}, ${support_2}
- Resistance Levels: ${resistance_1}, ${resistance_2}

Please provide a detailed technical analysis including:
1. Overall technical outlook (bullish, bearish, or neutral)
2. Analysis of key technical indicators and what they suggest
3. Support and resistance analysis
4. Chart patterns and formations (if any)
5. Trading recommendations based on technical factors
6. Key levels to watch for potential breakouts or breakdowns"""

        # Determine technical outlook based on indicators
        bullish_factors = 0
        if current_price > moving_avg_50: bullish_factors += 1
        if current_price > moving_avg_200: bullish_factors += 1
        if moving_avg_50 > moving_avg_200: bullish_factors += 1
        if rsi > 50 and rsi  macd_signal: bullish_factors += 1
        if current_price > (bollinger_upper + bollinger_lower) / 2: bullish_factors += 1
        
        if bullish_factors >= 5:
            outlook = "strongly bullish"
        elif bullish_factors == 4:
            outlook = "moderately bullish"
        elif bullish_factors == 3:
            outlook = "neutral with slight bullish bias"
        elif bullish_factors == 2:
            outlook = "neutral with slight bearish bias"
        elif bullish_factors == 1:
            outlook = "moderately bearish"
        else:
            outlook = "strongly bearish"
        
        # Generate potential patterns based on outlook
        bullish_patterns = ["ascending triangle", "cup and handle", "inverse head and shoulders", "bullish flag", "bullish pennant"]
        bearish_patterns = ["descending triangle", "head and shoulders", "double top", "bearish flag", "bearish pennant"]
        neutral_patterns = ["symmetrical triangle", "rectangle", "wedge", "channel"]
        
        if "bullish" in outlook:
            patterns = random.sample(bullish_patterns, 1)
        elif "bearish" in outlook:
            patterns = random.sample(bearish_patterns, 1)
        else:
            patterns = random.sample(neutral_patterns, 1)
        
        # Generate response
        response = f"""# Technical Analysis: {ticker}

## Executive Summary

Based on the provided technical indicators and price data, the overall technical outlook for {ticker} is **{outlook}**. The stock is currently trading at ${current_price}, which is {'above' if current_price > moving_avg_50 else 'below'} its 50-day moving average and {'above' if current_price > moving_avg_200 else 'below'} its 200-day moving average.

## Trend Analysis

### Moving Averages
- The 50-day moving average (${moving_avg_50}) is {'above' if moving_avg_50 > moving_avg_200 else 'below'} the 200-day moving average (${moving_avg_200}), indicating a {'bullish' if moving_avg_50 > moving_avg_200 else 'bearish'} longer-term trend.
- The current price is {'above both moving averages, confirming bullish momentum' if current_price > moving_avg_50 and current_price > moving_avg_200 else 'below both moving averages, indicating bearish pressure' if current_price  moving_avg_200 and moving_avg_50 / moving_avg_200  0 else 'closed lower'} by {abs(price_change)} ({abs(round(price_change / previous_close * 100, 2))}%) compared to the previous session.
- Volume was {'above average, confirming the price move' if random.choice([True, False]) else 'below average, suggesting weak conviction'}.
- The price is currently {'testing resistance' if abs(current_price - resistance_1) / resistance_1  70 else 'Oversold territory, suggesting potential reversal or bounce' if rsi  50 else 'Neutral territory with bearish pressure'}
- {'No divergence observed between price and RSI' if random.choice([True, False]) else 'Bullish divergence: Price making lower lows while RSI makes higher lows' if "bullish" in outlook else 'Bearish divergence: Price making higher highs while RSI makes lower highs'}

### MACD
- MACD ({macd}) is {'above' if macd > macd_signal else 'below'} the signal line ({macd_signal}), generating a {'bullish' if macd > macd_signal else 'bearish'} signal.
- The MACD histogram is {'expanding, indicating strengthening momentum' if abs(macd - macd_signal) > 0.5 else 'contracting, suggesting weakening momentum'}.
- {'The MACD line is above zero, confirming bullish momentum' if macd > 0 else 'The MACD line is below zero, confirming bearish pressure'}

### Bollinger Bands
- Current Bollinger Bands: Upper ${bollinger_upper}, Lower ${bollinger_lower}
- The price is {'near the upper band, indicating strong upward momentum' if current_price > bollinger_upper * 0.98 else 'near the lower band, suggesting oversold conditions' if current_price  0.15 else 'contracting, suggesting decreasing volatility and potential breakout'}

## Support and Resistance Analysis

### Support Levels
1. **Primary Support**: ${support_1} - {'Currently being tested' if abs(current_price - support_1) / support_1  50 and 'bullish' in outlook) or (rsi  Dict[str, str]:
        """Generate a sample for economic analysis."""
        # Generate economic indicators
        gdp_growth = round(random.uniform(-2.0, 5.0), 1)
        inflation = round(random.uniform(1.0, 8.0), 1)
        unemployment = round(random.uniform(3.0, 8.0), 1)
        fed_rate = round(random.uniform(0.25, 5.0), 2)
        treasury_10y = round(random.uniform(1.5, 5.0), 2)
        treasury_2y = round(random.uniform(1.0, 4.5), 2)
        consumer_sentiment = round(random.uniform(60, 110), 1)
        
        # Create instruction
        instruction = f"""Analyze the current economic environment and its implications for financial markets based on the following economic indicators:

Economic Indicators:
- GDP Growth: {gdp_growth}%
- Inflation Rate: {inflation}%
- Unemployment Rate: {unemployment}%
- Federal Funds Rate: {fed_rate}%
- 10-Year Treasury Yield: {treasury_10y}%
- 2-Year Treasury Yield: {treasury_2y}%
- Consumer Sentiment Index: {consumer_sentiment}

Please provide a comprehensive economic analysis including:
1. Assessment of the current economic cycle
2. Inflation outlook and monetary policy implications
3. Labor market analysis
4. Bond market and yield curve analysis
5. Consumer and business sentiment evaluation
6. Implications for different asset classes (equities, fixed income, commodities, real estate)
7. Key economic risks to monitor"""

        # Determine economic cycle phase
        if gdp_growth > 3.0 and unemployment  2.0:
            cycle_phase = "mid-cycle expansion"
        elif gdp_growth > 0:
            cycle_phase = "early expansion"
        else:
            cycle_phase = "contraction"
        
        # Determine monetary policy stance
        if inflation > 4.0 and fed_rate  3.0 and fed_rate > 4.0:
            policy_stance = "restrictive but potentially nearing peak"
        elif inflation  3.0:
            policy_stance = "likely to ease"
        elif inflation  2.0:
            policy_stance = "neutral to slightly restrictive"
        else:
            policy_stance = "neutral"
        
        # Determine yield curve shape
        if treasury_10y > treasury_2y + 0.5:
            yield_curve = "normal/steep"
            recession_risk = "low"
        elif treasury_10y > treasury_2y:
            yield_curve = "flattening but positive"
            recession_risk = "moderate"
        else:
            yield_curve = "inverted"
            recession_risk = "elevated"
        
        # Generate response
        response = f"""# Economic Analysis and Market Implications

## Current Economic Cycle Assessment

Based on the provided economic indicators, the U.S. economy appears to be in the **{cycle_phase} phase** of the economic cycle.

- **GDP Growth ({gdp_growth}%)**: {'Strong growth indicating economic expansion' if gdp_growth > 3 else 'Moderate growth suggesting continued expansion' if gdp_growth > 1.5 else 'Weak growth indicating potential slowdown' if gdp_growth > 0 else 'Negative growth signaling economic contraction'}
- **Unemployment Rate ({unemployment}%)**: {'Well below the natural rate, indicating a very tight labor market' if unemployment  90 else 'Moderate, suggesting cautious consumer behavior' if consumer_sentiment > 75 else 'Weak, potentially limiting consumer spending'}

The combination of {gdp_growth}% GDP growth and {unemployment}% unemployment suggests {'an economy operating above potential with risk of overheating' if gdp_growth > 3 and unemployment  1.5 and unemployment  3 else 'moderately above' if inflation > 2.5 else 'near' if inflation > 1.8 else 'below'} the Federal Reserve's 2% target.

Key inflation drivers appear to be:
- {'Tight labor market pushing wage growth' if unemployment  85 and gdp_growth > 2 else 'Moderate consumer demand' if consumer_sentiment > 70 else 'Weak consumer demand limiting pricing power'}
- {'Supply constraints in key sectors' if random.choice([True, False]) else 'Improving supply chains moderating price pressures' if random.choice([True, False]) else 'Global commodity price pressures' if random.choice([True, False]) else 'Housing and shelter cost increases'}

The inflation trajectory is likely to {'remain elevated in the near term before gradually moderating' if inflation > 4 else 'gradually return to target over the next 12-18 months' if inflation > 2.5 else 'remain near the Fed\'s target range' if inflation > 1.8 else 'remain below target, potentially raising deflation concerns'}.

### Monetary Policy Implications
With the Federal Funds Rate at {fed_rate}%, monetary policy is currently {'highly restrictive' if fed_rate > 4 else 'restrictive' if fed_rate > 3 else 'neutral to slightly restrictive' if fed_rate > 2 else 'accommodative'}.

Given the current inflation rate of {inflation}% and unemployment rate of {unemployment}%, the Federal Reserve is {policy_stance}. Markets should anticipate {'additional rate hikes' if 'tighten' in policy_stance else 'rate cuts' if 'ease' in policy_stance else 'a hold at current levels with data-dependent forward guidance'}.

The real (inflation-adjusted) federal funds rate is currently {round(fed_rate - inflation, 1)}%, which is {'restrictive' if fed_rate - inflation > 1 else 'neutral' if fed_rate - inflation > -1 else 'still accommodative despite nominal rate hikes'}.

## Labor Market Analysis

The unemployment rate of {unemployment}% is {'well below' if unemployment  3 else 'Wage growth broadly aligned with productivity' if unemployment  0.5 else 'slowing growth ahead' if treasury_10y - treasury_2y > 0 else 'elevated recession risk in the next 6-18 months'}

The current 10-year yield of {treasury_10y}% suggests {'bond market concerns about persistent inflation' if treasury_10y > 4 and inflation > 3 else 'a balanced outlook between growth and inflation' if treasury_10y > 3 else 'expectations for slowing growth and inflation' if treasury_10y  inflation else 'negative, potentially driving capital toward risk assets despite higher nominal rates'}.

## Consumer and Business Sentiment

The Consumer Sentiment Index of {consumer_sentiment} is {'strong' if consumer_sentiment > 90 else 'moderate' if consumer_sentiment > 75 else 'weak'}, which {'supports robust consumer spending' if consumer_sentiment > 90 else 'suggests cautious but continued spending' if consumer_sentiment > 75 else 'may limit consumer spending growth'}.

{'Business sentiment indicators align with consumer confidence' if random.choice([True, False]) else 'Business sentiment shows divergence from consumer confidence' if random.choice([True, False]) else 'Business investment plans remain cautious despite consumer resilience' if random.choice([True, False]) else 'Small business optimism lags larger corporate sentiment'}.

The {'alignment' if random.choice([True, False]) else 'divergence'} between consumer and business sentiment suggests {'a coherent economic picture' if gdp_growth > 2 else 'potential volatility in economic data' if gdp_growth > 0 else 'conflicting signals about the direction of the economy'}.

## Asset Class Implications

### Equities
- **Overall Outlook**: {'Positive, with economic growth supporting earnings' if gdp_growth > 2 and cycle_phase in ['early expansion', 'mid-cycle expansion'] else 'Cautious, with slowing growth likely to pressure valuations' if gdp_growth > 0 else 'Negative, with earnings likely to contract'}
- **Sector Preferences**: {'Cyclicals (Industrials, Consumer Discretionary, Financials)' if cycle_phase in ['early expansion', 'mid-cycle expansion'] else 'Late-cycle sectors (Energy, Materials, Healthcare)' if cycle_phase == 'late expansion' else 'Defensives (Utilities, Consumer Staples, Healthcare)'}
- **Style Factors**: {'Value likely to outperform growth in a rising rate environment' if fed_rate  2.5 else 'Agency mortgage-backed securities attractive with stabilizing rate environment' if policy_stance == 'neutral' else 'Municipal bonds attractive on a tax-adjusted basis'}

### Alternative Investments
- **Commodities**: {'Supportive environment with above-trend growth and inflation' if gdp_growth > 2.5 and inflation > 2.5 else 'Selective opportunities based on supply constraints' if inflation > 2 else 'Challenging with slowing growth and moderating inflation'}
- **Real Estate**: {'Commercial real estate facing headwinds from higher rates' if fed_rate > 3.5 else 'Residential housing supported by supply constraints despite higher mortgage rates' if fed_rate > 2.5 else 'REITs may offer value after rate-driven selloff'}
- **Private Markets**: {'Valuation adjustments likely as discount rates increase' if fed_rate > 3 else 'Opportunities in private credit as banks tighten lending standards' if random.choice([True, False]) else 'Dry powder deployment accelerating as valuations become more attractive'}

## Key Economic Risks to Monitor

1. **Inflation Persistence**: {'Inflation becoming entrenched in service sectors and wages' if inflation > 3 and unemployment  2.5 else 'Inflation falling below target, raising deflation concerns'}

2. **Recession Risk**: {'Elevated given inverted yield curve and tightening financial conditions' if yield_curve == 'inverted' else 'Moderate but rising as growth slows' if gdp_growth  4 and gdp_growth  2.5 else 'Fiscal policy adding to inflationary pressures'}

4. **Global Factors**: {'Synchronized global monetary tightening amplifying growth slowdown' if fed_rate > 3 else 'Geopolitical tensions impacting energy and food prices' if random.choice([True, False]) else 'China's growth trajectory affecting global demand' if random.choice([True, False]) else 'Dollar strength creating emerging market pressures'}

5. **Financial Stability**: {'Banking sector stress from rapid rate increases' if fed_rate > 4 and fed_rate - inflation > 1 else 'Commercial real estate sector vulnerabilities' if fed_rate > 3.5 else 'Corporate debt refinancing challenges in higher rate environment' if fed_rate > 3 else 'Market liquidity deterioration during stress periods'}

## Conclusion

The current economic environment is characterized by {gdp_growth}% growth, {inflation}% inflation, and {unemployment}% unemployment, placing it in the **{cycle_phase}** phase of the economic cycle with **{recession_risk}** recession risk over the next 12 months.

Monetary policy is {policy_stance} with the Federal Funds Rate at {fed_rate}%, while the {yield_curve} yield curve ({treasury_10y}% 10-year, {treasury_2y}% 2-year) suggests {'economic expansion ahead' if yield_curve == 'normal/steep' else 'slowing growth' if yield_curve == 'flattening but positive' else 'elevated recession risk'}.

The most favorable asset allocation in this environment would {'overweight equities and commodities while underweighting fixed income' if gdp_growth > 2.5 and inflation > 2.5 else 'balance between equities and fixed income with a quality bias' if gdp_growth > 1.5 else 'favor fixed income and defensive equities while reducing cyclical exposure'}.

This analysis is based on current data and subject to revision as economic conditions evolve."""

        return {
            "instruction": instruction,
            "response": response
        }

def main():
    parser = argparse.ArgumentParser(description="Generate synthetic financial datasets for fine-tuning LLMs")
    parser.add_argument("--num_samples", type=int, default=1000, help="Number of samples to generate")
    parser.add_argument("--output_dir", type=str, default="data/training", help="Directory to save generated datasets")
    parser.add_argument("--output_format", type=str, default="jsonl", choices=["jsonl", "csv"], help="Output file format")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    args = parser.parse_args()
    
    generator = FinancialDatasetGenerator(output_dir=args.output_dir)
    generator.generate_dataset(
        num_samples=args.num_samples,
        output_format=args.output_format,
        seed=args.seed
    )

if __name__ == "__main__":
    main()
```

### 14. Create a Fine-Tuning Script for Qwen3

**scripts/finetune_qwen.py**:
```
import os
import argparse
import logging
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer, DataCollatorForLanguageModeling
from datasets import load_dataset
import wandb
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
import bitsandbytes as bnb

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def parse_args():
    parser = argparse.ArgumentParser(description="Fine-tune Qwen model on financial data")
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen3-7B-A1B", help="Base model to fine-tune")
    parser.add_argument("--dataset_path", type=str, required=True, help="Path to the dataset file (jsonl or csv)")
    parser.add_argument("--output_dir", type=str, default="./fine_tuned_model", help="Directory to save the fine-tuned model")
    parser.add_argument("--num_train_epochs", type=int, default=3, help="Number of training epochs")
    parser.add_argument("--per_device_train_batch_size", type=int, default=4, help="Batch size per device for training")
    parser.add_argument("--learning_rate", type=float, default=2e-5, help="Learning rate")
    parser.add_argument("--warmup_steps", type=int, default=500, help="Number of warmup steps")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4, help="Number of gradient accumulation steps")
    parser.add_argument("--use_lora", action="store_true", help="Whether to use LoRA for fine-tuning")
    parser.add_argument("--lora_r", type=int, default=16, help="LoRA r parameter")
    parser.add_argument("--lora_alpha", type=int, default=32, help="LoRA alpha parameter")
    parser.add_argument("--lora_dropout", type=float, default=0.05, help="LoRA dropout parameter")
    parser.add_argument("--use_8bit", action="store_true", help="Whether to use 8-bit quantization")
    parser.add_argument("--use_4bit", action="store_true", help="Whether to use 4-bit quantization")
    parser.add_argument("--max_seq_length", type=int, default=2048, help="Maximum sequence length")
    parser.add_argument("--logging_steps", type=int, default=10, help="Logging steps")
    parser.add_argument("--save_steps", type=int, default=500, help="Save steps")
    parser.add_argument("--eval_steps", type=int, default=500, help="Evaluation steps")
    parser.add_argument("--use_wandb", action="store_true", help="Whether to use Weights & Biases for logging")
    parser.add_argument("--wandb_project", type=str, default="openluminary", help="Weights & Biases project name")
    parser.add_argument("--wandb_run_name", type=str, default=None, help="Weights & Biases run name")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    return parser.parse_args()

def format_instruction(example):
    """Format the instruction and response for training."""
    return f"user\n{example['instruction']}\nassistant\n{example['response']}"

def prepare_dataset(dataset_path, tokenizer, max_seq_length):
    """Prepare the dataset for training."""
    # Load the dataset
    if dataset_path.endswith(".jsonl"):
        dataset = load_dataset("json", data_files=dataset_path)
    elif dataset_path.endswith(".csv"):
        dataset = load_dataset("csv", data_files=dataset_path)
    else:
        raise ValueError(f"Unsupported file format: {dataset_path}")
    
    # Format the dataset
    dataset = dataset["train"].map(
        lambda example: {"text": format_instruction(example)},
        remove_columns=dataset["train"].column_names
    )
    
    # Tokenize the dataset
    def tokenize_function(examples):
        return tokenizer(
            examples["text"],
            truncation=True,
            max_length=max_seq_length,
            padding="max_length",
            return_tensors="pt"
        )
    
    tokenized_dataset = dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=["text"]
    )
    
    return tokenized_dataset

def main():
    args = parse_args()
    
    # Set random seed
    torch.manual_seed(args.seed)
    
    # Initialize Weights & Biases if requested
    if args.use_wandb:
        wandb.init(
            project=args.wandb_project,
            name=args.wandb_run_name or f"finetune-qwen-{args.model_name.split('/')[-1]}",
            config=vars(args)
        )
    
    # Load tokenizer
    logger.info(f"Loading tokenizer for {args.model_name}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    
    # Prepare the dataset
    logger.info(f"Preparing dataset from {args.dataset_path}")
    dataset = prepare_dataset(args.dataset_path, tokenizer, args.max_seq_length)
    
    # Load model
    logger.info(f"Loading model {args.model_name}")
    if args.use_8bit:
        logger.info("Using 8-bit quantization")
        model = AutoModelForCausalLM.from_pretrained(
            args.model_name,
            load_in_8bit=True,
            device_map="auto",
            trust_remote_code=True
        )
    elif args.use_4bit:
        logger.info("Using 4-bit quantization")
        model = AutoModelForCausalLM.from_pretrained(
            args.model_name,
            load_in_4bit=True,
            device_map="auto",
            trust_remote_code=True
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            args.model_name,
            device_map="auto",
            trust_remote_code=True
        )
    
    # Prepare model for training
    if args.use_8bit or args.use_4bit:
        model = prepare_model_for_kbit_training(model)
    
    # Apply LoRA if requested
    if args.use_lora:
        logger.info(f"Applying LoRA with r={args.lora_r}, alpha={args.lora_alpha}, dropout={args.lora_dropout}")
        peft_config = LoraConfig(
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
        )
        model = get_peft_model(model, peft_config)
        model.print_trainable_parameters()
    
    # Set up training arguments
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        warmup_steps=args.warmup_steps,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        evaluation_strategy="steps" if args.eval_steps > 0 else "no",
        eval_steps=args.eval_steps if args.eval_steps > 0 else None,
        fp16=True,
        report_to="wandb" if args.use_wandb else "none",
        save_total_limit=3,
        load_best_model_at_end=args.eval_steps > 0,
        remove_unused_columns=False,
    )
    
    # Create data collator
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False
    )
    
    # Create trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=data_collator,
    )
    
    # Start training
    logger.info("Starting training")
    trainer.train()
    
    # Save the fine-tuned model
    logger.info(f"Saving model to {args.output_dir}")
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    
    logger.info("Training completed")

if __name__ == "__main__":
    main()
```

### 15. Create a Deployment Script for Production

**scripts/deploy_model.py**:
```
import os
import argparse
import logging
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import gradio as gr
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Dict, Any, List, Optional
import uvicorn
import json

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class GenerationRequest(BaseModel):
    prompt: str = Field(..., description="The prompt to generate text from")
    max_new_tokens: int = Field(512, description="Maximum number of tokens to generate")
    temperature: float = Field(0.7, description="Temperature for sampling")
    top_p: float = Field(0.9, description="Top-p sampling parameter")
    top_k: int = Field(50, description="Top-k sampling parameter")
    do_sample: bool = Field(True, description="Whether to use sampling")
    use_thinking_mode: bool = Field(True, description="Whether to use thinking mode")

class AnalysisRequest(BaseModel):
    data: Dict[str, Any] = Field(..., description="Financial data for analysis")
    analysis_type: str = Field(..., description="Type of analysis to perform")
    use_thinking_mode: bool = Field(True, description="Whether to use thinking mode")

class Model:
    def __init__(self, model_path: str, device: str = "auto", load_in_8bit: bool = False, load_in_4bit: bool = False):
        self.model_path = model_path
        self.device = device
        self.load_in_8bit = load_in_8bit
        self.load_in_4bit = load_in_4bit
        self.model = None
        self.tokenizer = None
        self.load_model()
    
    def load_model(self):
        logger.info(f"Loading model from {self.model_path}")
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path, trust_remote_code=True)
        
        # Load model
        if self.load_in_8bit:
            logger.info("Loading model in 8-bit precision")
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_path,
                load_in_8bit=True,
                device_map=self.device,
                trust_remote_code=True
            )
        elif self.load_in_4bit:
            logger.info("Loading model in 4-bit precision")
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_path,
                load_in_4bit=True,
                device_map=self.device,
                trust_remote_code=True
            )
        else:
            logger.info("Loading model in full precision")
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_path,
                device_map=self.device,
                trust_remote_code=True
            )
        
        logger.info("Model loaded successfully")
    
    def generate(self, prompt: str, max_new_tokens: int = 512, temperature: float = 0.7, 
                 top_p: float = 0.9, top_k: int = 50, do_sample: bool = True, 
                 use_thinking_mode: bool = True) -> str:
        """Generate text from a prompt."""
        if use_thinking_mode:
            prompt = "/think\n" + prompt
        
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                do_sample=do_sample,
                pad_token_id=self.tokenizer.eos_token_id
            )
        
        response = self.tokenizer.decode(outputs, skip_special_tokens=True)
        
        # Remove the prompt from the response
        response = response[len(prompt):]
        
        return response.strip()
    
    def analyze_financial_data(self, data: Dict[str, Any], analysis_type: str, use_thinking_mode: bool = True) -> str:
        """Analyze financial data using the model."""
        # Get the appropriate prompt template for the analysis type
        prompt_template = self._get_prompt_template(analysis_type)
        
        # Format the financial data as a structured input
        formatted_data = self._format_data_for_prompt(data)
        
        # Create the full prompt
        prompt = prompt_template.format(data=formatted_data)
        
        # Generate response
        return self.generate(
            prompt=prompt,
            max_new_tokens=2048,
            temperature=0.2,
            do_sample=True,
            use_thinking_mode=use_thinking_mode
        )
    
    def _get_prompt_template(self, analysis_type: str) -> str:
        """Get prompt template based on analysis type."""
        templates = {
            "portfolio_optimization": (
                "You are a financial portfolio optimization specialist. Analyze the following financial data "
                "and provide optimal portfolio allocations based on modern portfolio theory. "
                "Include expected returns, volatility, and Sharpe ratio.\n\n{data}"
            ),
            "risk_assessment": (
                "You are a financial risk analyst. Analyze the following financial data "
                "and provide a comprehensive risk assessment, including VaR (Value at Risk), "
                "potential drawdowns, and risk factors.\n\n{data}"
            ),
            "market_prediction": (
                "You are a market analyst. Based on the following financial data, "
                "provide market predictions and trend analysis. Identify key indicators "
                "and potential market movements.\n\n{data}"
            ),
            "company_analysis": (
                "You are a financial analyst specializing in company valuation. "
                "Analyze the following company data and provide a comprehensive analysis "
                "including financial health, growth prospects, competitive position, and valuation.\n\n{data}"
            ),
            "financial_planning": (
                "You are a certified financial planner. Based on the following personal financial data, "
                "provide a comprehensive financial plan including budgeting, savings, investments, "
                "retirement planning, and tax optimization strategies.\n\n{data}"
            ),
            "asset_allocation": (
                "You are an asset allocation specialist. Based on the following investor profile and market data, "
                "recommend an optimal asset allocation across different asset classes. "
                "Include specific investment vehicles and rebalancing strategy.\n\n{data}"
            )
        }
        
        return templates.get(analysis_type, templates["portfolio_optimization"])
    
    def _format_data_for_prompt(self, data: Dict[str, Any]) -> str:
        """Format financial data as a structured string for the prompt."""
        result = []
        for key, value in data.items():
            if isinstance(value, dict):
                formatted_value = "\n  " + "\n  ".join(f"{k}: {v}" for k, v in value.items())
                result.append(f"{key}:{formatted_value}")
            elif isinstance(value, list):
                formatted_value = "\n  - " + "\n  - ".join(str(item) for item in value)
                result.append(f"{key}:{formatted_value}")
            else:
                result.append(f"{key}: {value}")
        
        return "\n".join(result)

def create_api(model: Model) -> FastAPI:
    """Create a FastAPI app for the model."""
    app = FastAPI(
        title="OpenLuminary AI API",
        description="API for financial analysis using the OpenLuminary AI model",
        version="0.1.0"
    )
    
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    @app.get("/")
    async def root():
        return {"message": "Welcome to the OpenLuminary AI API"}
    
    @app.post("/generate")
    async def generate(request: GenerationRequest):
        try:
            response = model.generate(
                prompt=request.prompt,
                max_new_tokens=request.max_new_tokens,
                temperature=request.temperature,
                top_p=request.top_p,
                top_k=request.top_k,
                do_sample=request.do_sample,
                use_thinking_mode=request.use_thinking_mode
            )
            return {"generated_text": response}
        except Exception as e:
            logger.error(f"Error generating text: {e}")
            raise HTTPException(status_code=500, detail=str(e))
    
    @app.post("/analyze")
    async def analyze(request: AnalysisRequest):
        try:
            response = model.analyze_financial_data(
                data=request.data,
                analysis_type=request.analysis_type,
                use_thinking_mode=request.use_thinking_mode
            )
            return {"analysis": response}
        except Exception as e:
            logger.error(f"Error analyzing data: {e}")
            raise HTTPException(status_code=500, detail=str(e))
    
    return app

def create_gradio_interface(model: Model) -> gr.Blocks:
    """Create a Gradio interface for the model."""
    with gr.Blocks(title="OpenLuminary AI") as interface:
        gr.Markdown("# OpenLuminary AI")
        gr.Markdown("## AI-powered financial analysis")
        
        with gr.Tab("Text Generation"):
            with gr.Row():
                with gr.Column():
                    prompt = gr.Textbox(label="Prompt", lines=5)
                    with gr.Row():
                        max_new_tokens = gr.Slider(minimum=64, maximum=4096, value=512, step=64, label="Max New Tokens")
                        temperature = gr.Slider(minimum=0.1, maximum=2.0, value=0.7, step=0.1, label="Temperature")
                    with gr.Row():
                        top_p = gr.Slider(minimum=0.1, maximum=1.0, value=0.9, step=0.1, label="Top-p")
                        top_k = gr.Slider(minimum=1, maximum=100, value=50, step=1, label="Top-k")
                    with gr.Row():
                        do_sample = gr.Checkbox(value=True, label="Do Sample")
                        use_thinking_mode = gr.Checkbox(value=True, label="Use Thinking Mode")
                    generate_btn = gr.Button("Generate")
                with gr.Column():
                    output = gr.Textbox(label="Generated Text", lines=20)
            
            generate_btn.click(
                fn=model.generate,
                inputs=[prompt, max_new_tokens, temperature, top_p, top_k, do_sample, use_thinking_mode],
                outputs=output
            )
        
        with gr.Tab("Financial Analysis"):
            analysis_type = gr.Dropdown(
                choices=[
                    "portfolio_optimization",
                    "risk_assessment",
                    "market_prediction",
                    "company_analysis",
                    "financial_planning",
                    "asset_allocation"
                ],
                value="portfolio_optimization",
                label="Analysis Type"
            )
            
            data_json = gr.Textbox(
                label="Financial Data (JSON format)",
                lines=10,
                placeholder="""Example for portfolio optimization:
{
    "assets": {
        "AAPL": {"expected_return": 0.12, "volatility": 0.25},
        "MSFT": {"expected_return": 0.10, "volatility": 0.20},
        "GOOGL": {"expected_return": 0.15, "volatility": 0.30}
    },
    "constraints": {
        "max_allocation": 0.4,
        "min_allocation": 0.1
    },
    "risk_tolerance": "moderate",
    "investment_horizon": "long-term"
}"""
            )
            
            use_thinking_mode_analysis = gr.Checkbox(value=True, label="Use Thinking Mode")
            analyze_btn = gr.Button("Analyze")
            analysis_output = gr.Markdown(label="Analysis Result")
            
            def analyze_data(analysis_type, data_json, use_thinking_mode):
                try:
                    data = json.loads(data_json)
                    result = model.analyze_financial_data(
                        data=data,
                        analysis_type=analysis_type,
                        use_thinking_mode=use_thinking_mode
                    )
                    return result
                except json.JSONDecodeError:
                    return "Error: Invalid JSON format"
                except Exception as e:
                    return f"Error: {str(e)}"
            
            analyze_btn.click(
                fn=analyze_data,
                inputs=[analysis_type, data_json, use_thinking_mode_analysis],
                outputs=analysis_output
            )
        
        with gr.Tab("About"):
            gr.Markdown("""
            # About OpenLuminary AI
            
            OpenLuminary is an open-source AI-powered financial analysis platform. This interface provides access to our fine-tuned Qwen3 model specialized for financial analysis.
            
            ## Capabilities
            
            - Portfolio optimization
            - Risk assessment
            - Market prediction
            - Company analysis
            - Financial planning
            - Asset allocation
            
            ## Model Information
            
            - Base model: Qwen3
            - Fine-tuned on financial datasets
            - Specialized for financial analysis tasks
            
            ## Project Links
            
            - [GitHub Repository](https://github.com/yourusername/OpenLuminary)
            - [Documentation](https://github.com/yourusername/OpenLuminary/docs)
            """)
    
    return interface

def main():
    parser = argparse.ArgumentParser(description="Deploy OpenLuminary AI model")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the model")
    parser.add_argument("--device", type=str, default="auto", help="Device to run the model on (cpu, cuda, auto)")
    parser.add_argument("--load_in_8bit", action="store_true", help="Load model in 8-bit precision")
    parser.add_argument("--load_in_4bit", action="store_true", help="Load model in 4-bit precision")
    parser.add_argument("--api_only", action="store_true", help="Run only the API server")
    parser.add_argument("--ui_only", action="store_true", help="Run only the UI server")
    parser.add_argument("--api_port", type=int, default=8000, help="Port for the API server")
    parser.add_argument("--ui_port", type=int, default=7860, help="Port for the UI server")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host to bind the servers to")
    parser.add_argument("--share", action="store_true", help="Share the Gradio interface")
    args = parser.parse_args()
    
    # Load the model
    model = Model(
        model_path=args.model_path,
        device=args.device,
        load_in_8bit=args.load_in_8bit,
        load_in_4bit=args.load_in_4bit
    )
    
    # Run the API server
    if not args.ui_only:
        app = create_api(model)
        logger.info(f"Starting API server on {args.host}:{args.api_port}")
        if not args.api_only:
            import threading
            threading.Thread(target=lambda: uvicorn.run(app, host=args.host, port=args.api_port)).start()
        else:
            uvicorn.run(app, host=args.host, port=args.api_port)
    
    # Run the UI server
    if not args.api_only:
        interface = create_gradio_interface(model)
        logger.info(f"Starting UI server on {args.host}:{args.ui_port}")
        interface.launch(server_name=args.host, server_port=args.ui_port, share=args.share)

if __name__ == "__main__":
    main()
```

### 16. Create a Monitoring and Evaluation Script

**scripts/evaluate_model.py**:
```
import os
import argparse
import logging
import json
import pandas as pd
import numpy as np
from tqdm import tqdm
from typing import List, Dict, Any, Optional
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import matplotlib.pyplot as plt
import seaborn as sns
from rouge_score import rouge_scorer
import nltk
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

# Download NLTK data
nltk.download('punkt', quiet=True)

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ModelEvaluator:
    """Evaluate the performance of a fine-tuned model on financial tasks."""
    
    def __init__(
        self, 
        model_path: str, 
        device: str = "auto",
        load_in_8bit: bool = False,
        load_in_4bit: bool = False
    ):
        """
        Initialize the model evaluator.
        
        Args:
            model_path: Path to the model
            device: Device to run the model on (cpu, cuda, auto)
            load_in_8bit: Whether to load the model in 8-bit precision
            load_in_4bit: Whether to load the model in 4-bit precision
        """
        self.model_path = model_path
        self.device = device
        self.load_in_8bit = load_in_8bit
        self.load_in_4bit = load_in_4bit
        
        # Load tokenizer and model
        logger.info(f"Loading model from {model_path}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        
        if load_in_8bit:
            logger.info("Loading model in 8-bit precision")
            self.model = AutoModelForCausalLM.from_pretrained(
                model_path,
                load_in_8bit=True,
                device_map=device,
                trust_remote_code=True
            )
        elif load_in_4bit:
            logger.info("Loading model in 4-bit precision")
            self.model = AutoModelForCausalLM.from_pretrained(
                model_path,
                load_in_4bit=True,
                device_map=device,
                trust_remote_code=True
            )
        else:
            logger.info("Loading model in full precision")
            self.model = AutoModelForCausalLM.from_pretrained(
                model_path,
                device_map=device,
                trust_remote_code=True
            )
        
        # Initialize metrics
        self.rouge_scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
        self.smoothing = SmoothingFunction().method1
    
    def generate(
        self, 
        prompt: str, 
        max_new_tokens: int = 512, 
        temperature: float = 0.7,
        top_p: float = 0.9,
        top_k: int = 50,
        do_sample: bool = True,
        use_thinking_mode: bool = True
    ) -> str:
        """
        Generate text from a prompt.
        
        Args:
            prompt: The prompt to generate text from
            max_new_tokens: Maximum number of tokens to generate
            temperature: Temperature for sampling
            top_p: Top-p sampling parameter
            top_k: Top-k sampling parameter
            do_sample: Whether to use sampling
            use_thinking_mode: Whether to use thinking mode
            
        Returns:
            Generated text
        """
        if use_thinking_mode:
            prompt = "/think\n" + prompt
        
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                do_sample=do_sample,
                pad_token_id=self.tokenizer.eos_token_id
            )
        
        response = self.tokenizer.decode(outputs, skip_special_tokens=True)
        
        # Remove the prompt from the response
        response = response[len(prompt):]
        
        return response.strip()
    
    def evaluate_dataset(
        self, 
        dataset_path: str,
        output_dir: str,
        num_samples: Optional[int] = None,
        use_thinking_mode: bool = True
    ) -> Dict[str, Any]:
        """
        Evaluate the model on a dataset.
        
        Args:
            dataset_path: Path to the dataset file (jsonl or csv)
            output_dir: Directory to save evaluation results
            num_samples: Number of samples to evaluate (if None, evaluate all)
            use_thinking_mode: Whether to use thinking mode
            
        Returns:
            Dictionary of evaluation metrics
        """
        # Load dataset
        if dataset_path.endswith(".jsonl"):
            with open(dataset_path, "r") as f:
                dataset = [json.loads(line) for line in f]
        elif dataset_path.endswith(".csv"):
            dataset = pd.read_csv(dataset_path).to_dict("records")
        else:
            raise ValueError(f"Unsupported file format: {dataset_path}")
        
        # Limit number of samples if specified
        if num_samples is not None:
            dataset = dataset[:num_samples]
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Evaluate each sample
        results = []
        for sample in tqdm(dataset, desc="Evaluating"):
            # Generate response
            prompt = sample["instruction"]
            reference = sample["response"]
            
            generated = self.generate(
                prompt=prompt,
                max_new_tokens=2048,
                temperature=0.2,
                do_sample=True,
                use_thinking_mode=use_thinking_mode
            )
            
            # Calculate metrics
            rouge_scores = self.rouge_scorer.score(reference, generated)
            
            # Calculate BLEU score
            reference_tokens = nltk.word_tokenize(reference.lower())
            generated_tokens = nltk.word_tokenize(generated.lower())
            bleu_score = sentence_bleu([reference_tokens], generated_tokens, smoothing_function=self.smoothing)
            
            # Store results
            result = {
                "prompt": prompt,
                "reference": reference,
                "generated": generated,
                "rouge1_precision": rouge_scores["rouge1"].precision,
                "rouge1_recall": rouge_scores["rouge1"].recall,
                "rouge1_fmeasure": rouge_scores["rouge1"].fmeasure,
                "rouge2_precision": rouge_scores["rouge2"].precision,
                "rouge2_recall": rouge_scores["rouge2"].recall,
                "rouge2_fmeasure": rouge_scores["rouge2"].fmeasure,
                "rougeL_precision": rouge_scores["rougeL"].precision,
                "rougeL_recall": rouge_scores["rougeL"].recall,
                "rougeL_fmeasure": rouge_scores["rougeL"].fmeasure,
                "bleu": bleu_score
            }
            
            results.append(result)
        
        # Calculate aggregate metrics
        metrics = {
            "rouge1_precision": np.mean([r["rouge1_precision"] for r in results]),
            "rouge1_recall": np.mean([r["rouge1_recall"] for r in results]),
            "rouge1_fmeasure": np.mean([r["rouge1_fmeasure"] for r in results]),
            "rouge2_precision": np.mean([r["rouge2_precision"] for r in results]),
            "rouge2_recall": np.mean([r["rouge2_recall"] for r in results]),
            "rouge2_fmeasure": np.mean([r["rouge2_fmeasure"] for r in results]),
            "rougeL_precision": np.mean([r["rougeL_precision"] for r in results]),
            "rougeL_recall": np.mean([r["rougeL_recall"] for r in results]),
            "rougeL_fmeasure": np.mean([r["rougeL_fmeasure"] for r in results]),
            "bleu": np.mean([r["bleu"] for r in results])
        }
        
        # Save results
        results_df = pd.DataFrame(results)
        results_df.to_csv(os.path.join(output_dir, "evaluation_results.csv"), index=False)
        
        with open(os.path.join(output_dir, "metrics.json"), "w") as f:
            json.dump(metrics, f, indent=2)
        
        # Generate visualizations
        self._generate_visualizations(results, metrics, output_dir)
        
        return metrics
    
    def _generate_visualizations(
        self, 
        results: List[Dict[str, Any]], 
        metrics: Dict[str, float],
        output_dir: str
    ):
        """
        Generate visualizations of evaluation results.
        
        Args:
            results: List of evaluation results
            metrics: Dictionary of aggregate metrics
            output_dir: Directory to save visualizations
        """
        # Set up the style
        plt.style.use('ggplot')
        
        # Create a figure for ROUGE scores
        plt.figure(figsize=(12, 8))
        
        # Plot ROUGE scores
        rouge_metrics = [
            ("ROUGE-1", metrics["rouge1_precision"], metrics["rouge1_recall"], metrics["rouge1_fmeasure"]),
            ("ROUGE-2", metrics["rouge2_precision"], metrics["rouge2_recall"], metrics["rouge2_fmeasure"]),
            ("ROUGE-L", metrics["rougeL_precision"], metrics["rougeL_recall"], metrics["rougeL_fmeasure"])
        ]
        
        x = np.arange(len(rouge_metrics))
        width = 0.25
        
        plt.bar(x - width, [m for m in rouge_metrics], width, label='Precision')
        plt.bar(x, [m for m in rouge_metrics], width, label='Recall')
        plt.bar(x + width, [m for m in rouge_metrics], width, label='F1')
        
        plt.xlabel('Metric')
        plt.ylabel('Score')
        plt.title('ROUGE Scores')
        plt.xticks(x, [m for m in rouge_metrics])
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "rouge_scores.png"))
        plt.close()
        
        # Create a figure for BLEU score
        plt.figure(figsize=(8, 6))
        plt.bar(['BLEU'], [metrics["bleu"]], color='blue')
        plt.xlabel('Metric')
        plt.ylabel('Score')
        plt.title('BLEU Score')
        plt.ylim(0, 1)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "bleu_score.png"))
        plt.close()
        
        # Create a histogram of ROUGE-L F1 scores
        plt.figure(figsize=(10, 6))
        plt.hist([r["rougeL_fmeasure"] for r in results], bins=20, alpha=0.7, color='blue')
        plt.axvline(metrics["rougeL_fmeasure"], color='red', linestyle='dashed', linewidth=2, label=f'Mean: {metrics["rougeL_fmeasure"]:.3f}')
        plt.xlabel('ROUGE-L F1 Score')
        plt.ylabel('Count')
        plt.title('Distribution of ROUGE-L F1 Scores')
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "rougeL_distribution.png"))
        plt.close()
        
        # Create a scatter plot of ROUGE-L precision vs. recall
        plt.figure(figsize=(10, 8))
        plt.scatter([r["rougeL_precision"] for r in results], [r["rougeL_recall"] for r in results], alpha=0.7)
        plt.xlabel('ROUGE-L Precision')
        plt.ylabel('ROUGE-L Recall')
        plt.title('ROUGE-L Precision vs. Recall')
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "precision_recall.png"))
        plt.close()
        
        # Create a summary report
        with open(os.path.join(output_dir, "evaluation_summary.md"), "w") as f:
            f.write("# Model Evaluation Summary\n\n")
            f.write(f"Model: {self.model_path}\n\n")
            f.write(f"Number of samples evaluated: {len(results)}\n\n")
            
            f.write("## Aggregate Metrics\n\n")
            f.write("| Metric | Score |\n")
            f.write("|--------|-------|\n")
            for metric, value in metrics.items():
                f.write(f"| {metric} | {value:.4f} |\n")
            
            f.write("\n## Sample Outputs\n\n")
            
            # Include a few sample outputs
            for i, result in enumerate(results[:5]):
                f.write(f"### Sample {i+1}\n\n")
                f.write("**Prompt:**\n\n")
                f.write(f"```\n{result['prompt']}\n```
                f.write("**Reference:**\n\n")
                f.write(f"```\n{result['reference']}\n```
                f.write("**Generated:**\n\n")
                f.write(f"```\n{result['generated']}\n```
                f.write("**Metrics:**\n\n")
                f.write(f"- ROUGE-L F1: {result['rougeL_fmeasure']:.4f}\n")
                f.write(f"- BLEU: {result['bleu']:.4f}\n\n")
                f.write("---\n\n")

def main():
    parser = argparse.ArgumentParser(description="Evaluate a fine-tuned model on financial tasks")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the model")
    parser.add_argument("--dataset_path", type=str, required=True, help="Path to the evaluation dataset")
    parser.add_argument("--output_dir", type=str, default="./evaluation_results", help="Directory to save evaluation results")
    parser.add_argument("--num_samples", type=int, default=None, help="Number of samples to evaluate (if None, evaluate all)")
    parser.add_argument("--device", type=str, default="auto", help="Device to run the model on (cpu, cuda, auto)")
    parser.add_argument("--load_in_8bit", action="store_true", help="Load model in 8-bit precision")
    parser.add_argument("--load_in_4bit", action="store_true", help="Load model in 4-bit precision")
    parser.add_argument("--use_thinking_mode", action="store_true", help="Use thinking mode for generation")
    args = parser.parse_args()
    
    # Initialize evaluator
    evaluator = ModelEvaluator(
        model_path=args.model_path,
        device=args.device,
        load_in_8bit=args.load_in_8bit,
        load_in_4bit=args.load_in_4bit
    )
    
    # Evaluate model
    metrics = evaluator.evaluate_dataset(
        dataset_path=args.dataset_path,
        output_dir=args.output_dir,
        num_samples=args.num_samples,
        use_thinking_mode=args.use_thinking_mode
    )
    
    # Print metrics
    logger.info("Evaluation complete. Results:")
    for metric, value in metrics.items():
        logger.info(f"{metric}: {value:.4f}")
    
    logger.info(f"Detailed results saved to {args.output_dir}")

if __name__ == "__main__":
    main()
```

### 17. Update the Main README.md with Comprehensive Information

**README.md**:
```
# OpenLuminary


  



  Open-source AI-powered financial analysis platform



  
  
  
  
  


## Overview

OpenLuminary is an open-source alternative to BlackRock's Aladdin platform, providing sophisticated financial analysis capabilities powered by the Qwen3 AI model. Our mission is to democratize access to advanced financial technology and make it available to everyone.

## Features

- **Portfolio Optimization**: Optimize asset allocations based on modern portfolio theory
- **Risk Assessment**: Comprehensive risk analysis including VaR, stress testing, and scenario analysis
- **Market Data Processing**: Connect to various market data sources and process financial information
- **AI-Powered Analysis**: Leverage the power of Qwen3 for sophisticated financial analysis
- **Interactive Dashboard**: Visualize and analyze financial data through an intuitive interface
- **API Access**: Integrate OpenLuminary capabilities into your own applications

## Architecture

OpenLuminary is designed as a modular, extensible platform with the following key components:

- **Data Connectors**: Interfaces to various financial data sources
- **Portfolio Management**: Tools for portfolio optimization and management
- **Risk Assessment**: Advanced risk analysis capabilities
- **AI Layer**: Fine-tuned Qwen3 model for financial analysis
- **API Layer**: RESTful API for programmatic access
- **User Interface**: Interactive dashboard for visualization and analysis

## Getting Started

### Installation

#### Option 1: Using pip

```bash
pip install openluminary
```

#### Option 2: From source

```bash
# Clone the repository
git clone https://github.com/yourusername/OpenLuminary.git
cd OpenLuminary

# Create a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -e .
```

### Running the Dashboard

```bash
streamlit run dashboard.py
```

### Running the API

```bash
uvicorn src.api.main:app --reload
```

### Using Docker

```bash
# Build the Docker image
docker build -t openluminary .

# Run the container
docker run -p 8501:8501 -p 8000:8000 openluminary
```

## AI Capabilities

OpenLuminary leverages a fine-tuned version of the Qwen3 model, specialized for financial analysis. The AI capabilities include:

- **Financial Text Analysis**: Extract insights from financial reports and news
- **Portfolio Optimization Recommendations**: AI-driven portfolio allocation suggestions
- **Risk Assessment**: Identify and quantify potential risks in portfolios
- **Market Predictions**: Analyze market trends and provide forecasts
- **Natural Language Queries**: Ask questions about financial data in plain English

### Fine-tuning Your Own Model

We provide scripts and datasets for fine-tuning your own version of the Qwen3 model:

```bash
# Generate synthetic financial dataset
python -m src.models.financial_dataset_generator --num_samples 10000

# Fine-tune the model
python scripts/finetune_qwen.py --model_name "Qwen/Qwen3-7B-A1B" --dataset_path "data/training/financial_dataset_20250504_123456.jsonl" --use_lora --use_8bit
```

## Documentation

For detailed documentation, see the [docs](docs/) directory:

- [Getting Started Guide](docs/getting_started.md)
- [API Reference](docs/api_reference.md)
- [Architecture Overview](docs/architecture.md)
- [Advanced Features](docs/advanced_features.md)
- [Contributing Guidelines](CONTRIBUTING.md)

## Examples

### Portfolio Optimization

```python
from openluminary.data_processing.data_connector import DataConnectorFactory
from openluminary.portfolio_management.optimizer import PortfolioOptimizer
import pandas as pd

# Initialize components
data_connector = DataConnectorFactory.get_connector("yahoo")
portfolio_optimizer = PortfolioOptimizer()

# Get historical data
symbols = ["AAPL", "MSFT", "GOOGL", "AMZN"]
start_date = "2022-01-01"
end_date = "2023-01-01"

market_data = data_connector.get_historical_prices(symbols, start_date, end_date)

# Extract close prices and calculate returns
close_prices = pd.DataFrame()
for symbol, data in market_data.items():
    if data is not None and not data.empty:
        close_prices[symbol] = data['Close']

returns = close_prices.pct_change().dropna()

# Optimize portfolio
result = portfolio_optimizer.optimize_sharpe_ratio(returns)

# Print results
print("Optimized Portfolio Weights:")
for asset, weight in result["weights"].items():
    print(f"{asset}: {weight:.2%}")

print("\nPortfolio Performance:")
print(f"Expected Annual Return: {result['performance']['return']:.2%}")
print(f"Annual Volatility: {result['performance']['volatility']:.2%}")
print(f"Sharpe Ratio: {result['performance']['sharpe_ratio']:.2f}")
```

### AI-Powered Analysis

```python
from openluminary.models.qwen_interface import Qwen3Interface

# Initialize the Qwen3 interface
qwen = Qwen3Interface(use_thinking_mode=True)

# Perform portfolio optimization analysis
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
```

## Roadmap

See our [ROADMAP.md](ROADMAP.md) for the planned development path.

## Contributing

We welcome contributions to OpenLuminary! Please see our [Contributing Guidelines](CONTRIBUTING.md) for details.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgements

- [Qwen Team](https://github.com/QwenLM/Qwen) for the powerful Qwen3 model
- [Streamlit](https://streamlit.io/) for the interactive dashboard framework
- [FastAPI](https://fastapi.tiangolo.com/) for the API framework
- All our contributors and supporters

## Citation

If you use OpenLuminary in your research or applications, please cite:

```
@software{openluminary2025,
  author = {OpenLuminary Contributors},
  title = {OpenLuminary: Open-source AI-powered financial analysis platform},
  url = {https://github.com/yourusername/OpenLuminary},
  year = {2025},
}
```

## Contact

For questions, suggestions, or collaboration opportunities, please open an issue or contact us at [your-email@example.com](mailto:your-email@example.com).

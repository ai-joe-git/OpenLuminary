from fastapi import FastAPI, HTTPException, Depends, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os
import json

from src.data_processing.data_connector import DataConnectorFactory
from src.portfolio_management.optimizer import PortfolioOptimizer
from src.risk_assessment.advanced_risk import AdvancedRiskAssessment
from src.models.qwen_interface import Qwen3Interface

# Initialize FastAPI app
app = FastAPI(
    title="OpenLuminary API",
    description="API for OpenLuminary financial analysis platform",
    version="0.1.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, restrict this to specific origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize components
data_connector = DataConnectorFactory.get_connector("yahoo")
portfolio_optimizer = PortfolioOptimizer()
risk_assessment = AdvancedRiskAssessment()

# Pydantic models for request/response
class SymbolsRequest(BaseModel):
    symbols: List[str] = Field(..., description="List of ticker symbols")
    
class DateRangeRequest(BaseModel):
    start_date: str = Field(..., description="Start date in YYYY-MM-DD format")
    end_date: str = Field(..., description="End date in YYYY-MM-DD format")
    
class PortfolioWeightsRequest(BaseModel):
    weights: Dict[str, float] = Field(..., description="Dictionary mapping symbols to weights")
    
class OptimizationRequest(BaseModel):
    symbols: List[str] = Field(..., description="List of ticker symbols")
    start_date: str = Field(..., description="Start date in YYYY-MM-DD format")
    end_date: str = Field(..., description="End date in YYYY-MM-DD format")
    constraints: Optional[List[Dict[str, Any]]] = Field(None, description="Optimization constraints")
    bounds: Optional[Dict[str, List[float]]] = Field(None, description="Bounds for each asset (min, max)")
    
class RiskAssessmentRequest(BaseModel):
    symbols: List[str] = Field(..., description="List of ticker symbols")
    weights: Dict[str, float] = Field(..., description="Dictionary mapping symbols to weights")
    start_date: str = Field(..., description="Start date in YYYY-MM-DD format")
    end_date: str = Field(..., description="End date in YYYY-MM-DD format")
    confidence_level: float = Field(0.95, description="Confidence level for VaR calculations")
    
class AIAnalysisRequest(BaseModel):
    data: Dict[str, Any] = Field(..., description="Financial data for analysis")
    analysis_type: str = Field(..., description="Type of analysis to perform")
    use_thinking_mode: bool = Field(True, description="Whether to use thinking mode")

# Routes
@app.get("/")
async def root():
    return {"message": "Welcome to OpenLuminary API"}

@app.post("/market-data/historical")
async def get_historical_data(request: SymbolsRequest, date_range: DateRangeRequest):
    """Get historical market data for multiple symbols."""
    try:
        data = data_connector.get_historical_prices(
            request.symbols, 
            date_range.start_date, 
            date_range.end_date
        )
        
        # Convert DataFrames to dictionaries for JSON serialization
        result = {}
        for symbol, df in data.items():
            if df is not None and not df.empty:
                result[symbol] = {
                    "dates": df.index.strftime('%Y-%m-%d').tolist(),
                    "open": df['Open'].tolist(),
                    "high": df['High'].tolist(),
                    "low": df['Low'].tolist(),
                    "close": df['Close'].tolist(),
                    "volume": df['Volume'].tolist() if 'Volume' in df.columns else None,
                    "adjusted_close": df['Adj Close'].tolist() if 'Adj Close' in df.columns else None
                }
            else:
                result[symbol] = None
        
        return {"data": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/market-data/current")
async def get_current_prices(request: SymbolsRequest):
    """Get current market prices for multiple symbols."""
    try:
        data = data_connector.get_current_prices(request.symbols)
        return {"data": data}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/market-data/financials")
async def get_financial_statements(request: SymbolsRequest):
    """Get financial statements for multiple symbols."""
    try:
        data = data_connector.get_financial_statements(request.symbols)
        
        # Convert DataFrames to dictionaries for JSON serialization
        result = {}
        for symbol, statements in data.items():
            if statements is not None:
                symbol_result = {}
                for statement_name, df in statements.items():
                    if df is not None and not df.empty:
                        # Convert DataFrame to nested dictionary
                        symbol_result[statement_name] = json.loads(df.to_json(orient="split"))
                    else:
                        symbol_result[statement_name] = None
                result[symbol] = symbol_result
            else:
                result[symbol] = None
        
        return {"data": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/portfolio/optimize")
async def optimize_portfolio(request: OptimizationRequest):
    """Optimize portfolio weights to maximize Sharpe ratio."""
    try:
        # Get historical data
        historical_data = data_connector.get_historical_prices(
            request.symbols, 
            request.start_date, 
            request.end_date
        )
        
        # Extract close prices and calculate returns
        close_prices = pd.DataFrame()
        for symbol, df in historical_data.items():
            if df is not None and not df.empty and 'Close' in df.columns:
                close_prices[symbol] = df['Close']
        
        if close_prices.empty:
            raise HTTPException(status_code=400, detail="No valid price data found")
        
        # Calculate returns
        returns = close_prices.pct_change().dropna()
        
        # Prepare bounds if provided
        bounds = None
        if request.bounds:
            bounds = [(request.bounds.get(symbol, [0, 1])[0], request.bounds.get(symbol, [0, 1])[1]) 
                     for symbol in returns.columns]
        
        # Optimize portfolio
        result = portfolio_optimizer.optimize_sharpe_ratio(
            returns,
            constraints=request.constraints,
            bounds=bounds
        )
        
        # Generate efficient frontier
        efficient_frontier = portfolio_optimizer.generate_efficient_frontier(
            returns,
            n_points=20,
            constraints=request.constraints,
            bounds=bounds
        )
        
        # Convert efficient frontier to list for JSON serialization
        ef_list = efficient_frontier.to_dict(orient="records")
        
        return {
            "optimized_weights": result["weights"],
            "performance": result["performance"],
            "efficient_frontier": ef_list
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/risk/assess")
async def assess_risk(request: RiskAssessmentRequest):
    """Perform risk assessment on a portfolio."""
    try:
        # Get historical data
        historical_data = data_connector.get_historical_prices(
            list(request.weights.keys()), 
            request.start_date, 
            request.end_date
        )
        
        # Extract close prices and calculate returns
        close_prices = pd.DataFrame()
        for symbol, df in historical_data.items():
            if df is not None and not df.empty and 'Close' in df.columns:
                close_prices[symbol] = df['Close']
        
        if close_prices.empty:
            raise HTTPException(status_code=400, detail="No valid price data found")
        
        # Calculate returns
        returns = close_prices.pct_change().dropna()
        
        # Prepare weights array
        symbols = list(request.weights.keys())
        weights_array = np.array([request.weights.get(symbol, 0) for symbol in returns.columns])
        
        # Normalize weights
        weights_sum = weights_array.sum()
        if weights_sum > 0:
            weights_array = weights_array / weights_sum
        
        # Calculate risk metrics
        risk_metrics = risk_assessment.calculate_risk_metrics(returns, weights_array)
        
        # Calculate VaR and CVaR
        var_historical = risk_assessment.calculate_var(returns, weights_array, request.confidence_level, method="historical")
        var_parametric = risk_assessment.calculate_var(returns, weights_array, request.confidence_level, method="parametric")
        var_monte_carlo = risk_assessment.calculate_var(returns, weights_array, request.confidence_level, method="monte_carlo")
        
        cvar_historical = risk_assessment.calculate_cvar(returns, weights_array, request.confidence_level, method="historical")
        
        # Calculate drawdown
        drawdown = risk_assessment.calculate_drawdown(returns, weights_array)
        
        # Perform stress tests
        stress_scenarios = {
            "Market Crash": {symbol: 0.85 for symbol in returns.columns},
            "Tech Sector Decline": {
                symbol: 0.90 if symbol in ["AAPL", "MSFT", "GOOGL", "AMZN", "META"] else 0.98 
                for symbol in returns.columns
            },
            "Interest Rate Hike": {
                symbol: 0.95 if symbol in ["JPM", "BAC", "C", "WFC", "GS"] else 0.97
                for symbol in returns.columns
            },
            "Economic Boom": {symbol: 1.05 for symbol in returns.columns}
        }
        
        stress_results = risk_assessment.perform_stress_test(returns, weights_array, stress_scenarios)
        
        return {
            "risk_metrics": risk_metrics,
            "var": {
                "historical": var_historical,
                "parametric": var_parametric,
                "monte_carlo": var_monte_carlo
            },
            "cvar": cvar_historical,
            "drawdown": {
                "max_drawdown": drawdown["max_drawdown"],
                "avg_drawdown": drawdown["avg_drawdown"],
                "avg_duration": drawdown["avg_duration"]
            },
            "stress_test": stress_results
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/ai/analyze")
async def ai_analysis(request: AIAnalysisRequest):
    """Perform AI-powered analysis using Qwen3."""
    try:
        # Initialize Qwen3 interface
        qwen = Qwen3Interface(use_thinking_mode=request.use_thinking_mode)
        
        # Perform analysis
        result = qwen.analyze_financial_data(
            data=request.data,
            analysis_type=request.analysis_type
        )
        
        return {"analysis": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Run the app with: uvicorn src.api.main:app --reload

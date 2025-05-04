import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from src.data_processing.market_data import MarketDataProvider
from src.risk_assessment.basic_risk import BasicRiskAssessment

# Page configuration
st.set_page_config(
    page_title="OpenLuminary - Financial Analysis",
    page_icon="💹",
    layout="wide"
)

st.title("OpenLuminary - Financial Analysis Demo")
st.markdown("## Open-source AI-powered financial analysis platform")

# Initialize data provider
@st.cache_resource
def get_data_provider():
    return MarketDataProvider()

data_provider = get_data_provider()

# Sidebar
st.sidebar.title("Configuration")
tickers = st.sidebar.text_input("Stock Tickers (comma-separated)", "AAPL,MSFT,GOOGL,AMZN,META").split(",")
period = st.sidebar.selectbox("Time Period", ["1mo", "3mo", "6mo", "1y", "2y", "5y"], index=3)

# Get data
with st.spinner("Fetching market data..."):
    market_data = data_provider.get_historical_data(tickers, period=period)
    
# Create tabs
tab1, tab2, tab3 = st.tabs(["Market Data", "Portfolio Analysis", "Risk Assessment"])

with tab1:
    st.header("Market Data")
    
    # Display real-time quotes
    quotes = data_provider.get_realtime_quote(tickers)
    quote_df = pd.DataFrame([
        {
            "Symbol": symbol,
            "Price": data.get("price"),
            "Change": data.get("change"),
            "Change %": f"{data.get('change_percent', 0):.2f}%",
            "Volume": data.get("volume")
        }
        for symbol, data in quotes.items()
    ])
    st.dataframe(quote_df)
    
    # Display price charts
    for symbol, data in market_data.items():
        if data is not None and not data.empty:
            st.subheader(f"{symbol} Price Chart")
            fig = go.Figure()
            fig.add_trace(go.Candlestick(
                x=data.index,
                open=data['Open'],
                high=data['High'],
                low=data['Low'],
                close=data['Close'],
                name=symbol
            ))
            fig.update_layout(
                title=f"{symbol} Price Movement",
                xaxis_title="Date",
                yaxis_title="Price (USD)",
                height=500
            )
            st.plotly_chart(fig, use_container_width=True)

with tab2:
    st.header("Portfolio Analysis")
    
    # Portfolio weights
    st.subheader("Portfolio Allocation")
    weights = {}
    cols = st.columns(len(tickers))
    for i, ticker in enumerate(tickers):
        weights[ticker] = cols[i].slider(f"{ticker} Weight (%)", 0, 100, 100 // len(tickers))
    
    # Normalize weights
    total = sum(weights.values())
    if total > 0:
        normalized_weights = {k: v/total for k, v in weights.items()}
    else:
        normalized_weights = {k: 0 for k in weights}
    
    # Show pie chart
    fig = go.Figure(data=[go.Pie(
        labels=list(normalized_weights.keys()),
        values=list(normalized_weights.values()),
        hole=.3
    )])
    fig.update_layout(title_text="Portfolio Allocation")
    st.plotly_chart(fig, use_container_width=True)
    
    # Calculate returns
    returns_df = pd.DataFrame()
    for symbol, data in market_data.items():
        if data is not None and not data.empty:
            returns_df[symbol] = data['Close'].pct_change()
    
    returns_df = returns_df.dropna()
    
    if not returns_df.empty:
        # Calculate portfolio returns
        portfolio_returns = returns_df.dot(pd.Series(normalized_weights))
        
        # Calculate metrics
        expected_return = portfolio_returns.mean() * 252  # Annualized
        portfolio_std = portfolio_returns.std() * np.sqrt(252)  # Annualized
        sharpe_ratio = expected_return / portfolio_std if portfolio_std > 0 else 0
        
        # Display metrics
        metrics_cols = st.columns(3)
        metrics_cols[0].metric("Expected Annual Return", f"{expected_return:.2%}")
        metrics_cols[1].metric("Annual Volatility", f"{portfolio_std:.2%}")
        metrics_cols[2].metric("Sharpe Ratio", f"{sharpe_ratio:.2f}")
        
        # Show portfolio returns chart
        st.subheader("Portfolio Performance")
        cumulative_returns = (1 + portfolio_returns).cumprod()
        st.line_chart(cumulative_returns)

with tab3:
    st.header("Risk Assessment")
    
    if 'returns_df' in locals() and not returns_df.empty:
        # Initialize risk assessment
        risk_assessment = BasicRiskAssessment(confidence_level=0.95)
        
        # Prepare weights array
        weights_array = np.array([normalized_weights.get(ticker, 0) for ticker in returns_df.columns])
        
        # Calculate VaR and CVaR
        var_1d = risk_assessment.calculate_portfolio_var(returns_df, weights_array)
        cvar_1d = risk_assessment.calculate_portfolio_cvar(returns_df, weights_array)
        
        # Display risk metrics
        risk_cols = st.columns(2)
        risk_cols[0].metric("Value at Risk (1-day, 95%)", f"{var_1d:.2%}")
        risk_cols[1].metric("Conditional VaR (1-day, 95%)", f"{cvar_1d:.2%}")
        
        # Stress testing scenarios
        st.subheader("Stress Testing")
        scenarios = {
            "Market Crash": [0.85] * len(returns_df.columns),
            "Tech Sector Decline": [0.90 if ticker in ["AAPL", "MSFT", "GOOGL"] else 0.98 for ticker in returns_df.columns],
            "Economic Boom": [1.05] * len(returns_df.columns)
        }
        
        stress_results = risk_assessment.perform_stress_test(returns_df, weights_array, scenarios)
        
        # Display stress test results
        stress_df = pd.DataFrame([
            {"Scenario": scenario, "Portfolio Return": return_value}
            for scenario, return_value in stress_results.items()
        ])
        st.dataframe(stress_df)
        
        # Disclaimer
        st.info("Note: This is a simplified implementation. A production system would use more sophisticated risk models and data sources.")

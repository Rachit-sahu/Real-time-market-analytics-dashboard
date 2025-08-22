import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import time
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

from data_fetcher import DataFetcher
from models import ForecastingModels
from risk_analysis import RiskAnalyzer
from utils import format_number, calculate_returns

# Page configuration
st.set_page_config(
    page_title="Market Analytics Dashboard",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state
if 'data_fetcher' not in st.session_state:
    st.session_state.data_fetcher = DataFetcher()
if 'models' not in st.session_state:
    st.session_state.models = ForecastingModels()
if 'risk_analyzer' not in st.session_state:
    st.session_state.risk_analyzer = RiskAnalyzer()
if 'last_refresh' not in st.session_state:
    st.session_state.last_refresh = datetime.now()

# Main title
st.title("📈 Real-Time Market Analytics Dashboard")
st.markdown("---")

# Sidebar configuration
st.sidebar.header("Configuration")

# Asset selection
asset_type = st.sidebar.selectbox(
    "Asset Type",
    ["Equity", "Forex"]
)

if asset_type == "Equity":
    symbols = st.sidebar.multiselect(
        "Select Stocks",
        ["AAPL", "GOOGL", "MSFT", "TSLA", "AMZN", "NVDA", "META", "SPY", "QQQ"],
        default=["AAPL", "MSFT"]
    )
else:
    symbols = st.sidebar.multiselect(
        "Select Currency Pairs",
        ["EURUSD", "GBPUSD", "USDJPY", "USDCAD", "AUDUSD"],
        default=["EURUSD"]
    )

# Time period selection
period = st.sidebar.selectbox(
    "Time Period",
    ["1d", "5d", "1mo", "3mo", "6mo", "1y", "2y"],
    index=4  # Default to 6mo
)

# Model selection
forecast_model = st.sidebar.selectbox(
    "Forecasting Model",
    ["ARIMA", "LSTM", "Both"]
)

# Auto-refresh settings
auto_refresh = st.sidebar.checkbox("Auto Refresh", value=True)
refresh_interval = st.sidebar.slider("Refresh Interval (seconds)", 30, 300, 60)

# Manual refresh button
if st.sidebar.button("🔄 Refresh Data"):
    st.session_state.last_refresh = datetime.now()
    st.rerun()

# Auto refresh logic
if auto_refresh:
    time_since_refresh = (datetime.now() - st.session_state.last_refresh).seconds
    if time_since_refresh >= refresh_interval:
        st.session_state.last_refresh = datetime.now()
        st.rerun()

# Main dashboard layout
if symbols:
    # Create tabs for different sections
    tab1, tab2, tab3, tab4 = st.tabs(["📊 Live Data", "🔮 Forecasting", "⚖️ Risk Analysis", "📈 Backtesting"])
    
    with tab1:
        st.header("Live Market Data")
        
        # Fetch real-time data
        with st.spinner("Fetching live market data..."):
            data_dict = {}
            for symbol in symbols:
                try:
                    if asset_type == "Equity":
                        data = st.session_state.data_fetcher.get_stock_data(symbol, period)
                    else:
                        data = st.session_state.data_fetcher.get_forex_data(symbol, period)
                    
                    if data is not None and not data.empty:
                        data_dict[symbol] = data
                except Exception as e:
                    st.error(f"Error fetching data for {symbol}: {str(e)}")
        
        if data_dict:
            # Display current prices and changes
            cols = st.columns(len(symbols))
            for idx, (symbol, data) in enumerate(data_dict.items()):
                with cols[idx]:
                    current_price = data['Close'].iloc[-1]
                    prev_price = data['Close'].iloc[-2] if len(data) > 1 else current_price
                    change = current_price - prev_price
                    change_pct = (change / prev_price) * 100 if prev_price != 0 else 0
                    
                    color = "green" if change >= 0 else "red"
                    st.metric(
                        label=symbol,
                        value=f"${current_price:.2f}" if asset_type == "Equity" else f"{current_price:.4f}",
                        delta=f"{change_pct:.2f}%"
                    )
            
            # Price charts
            st.subheader("Price Charts")
            
            # Create subplot for multiple symbols
            fig = make_subplots(
                rows=len(symbols), cols=1,
                subplot_titles=[f"{symbol} Price Chart" for symbol in symbols],
                vertical_spacing=0.1
            )
            
            for idx, (symbol, data) in enumerate(data_dict.items()):
                # Candlestick chart
                fig.add_trace(
                    go.Candlestick(
                        x=data.index,
                        open=data['Open'],
                        high=data['High'],
                        low=data['Low'],
                        close=data['Close'],
                        name=symbol
                    ),
                    row=idx+1, col=1
                )
                
                # Volume chart (if available)
                if 'Volume' in data.columns:
                    fig.add_trace(
                        go.Bar(
                            x=data.index,
                            y=data['Volume'],
                            name=f"{symbol} Volume",
                            yaxis="y2",
                            opacity=0.3
                        ),
                        row=idx+1, col=1
                    )
            
            fig.update_layout(
                height=300 * len(symbols),
                showlegend=False,
                xaxis_rangeslider_visible=False
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Market summary table
            st.subheader("Market Summary")
            summary_data = []
            for symbol, data in data_dict.items():
                returns = calculate_returns(data['Close'])
                summary_data.append({
                    'Symbol': symbol,
                    'Current Price': f"${data['Close'].iloc[-1]:.2f}" if asset_type == "Equity" else f"{data['Close'].iloc[-1]:.4f}",
                    'Daily Change %': f"{((data['Close'].iloc[-1] / data['Close'].iloc[-2] - 1) * 100):.2f}%",
                    'Volatility (30d)': f"{returns.rolling(30).std().iloc[-1] * np.sqrt(252) * 100:.2f}%",
                    'Volume': format_number(data.get('Volume', pd.Series([0])).iloc[-1]) if 'Volume' in data.columns else "N/A"
                })
            
            st.dataframe(pd.DataFrame(summary_data), use_container_width=True)
    
    with tab2:
        st.header("Price Forecasting")
        
        if data_dict:
            selected_symbol = st.selectbox("Select Symbol for Forecasting", symbols)
            forecast_days = st.slider("Forecast Days", 1, 30, 7)
            
            if selected_symbol in data_dict:
                data = data_dict[selected_symbol]
                
                with st.spinner("Training forecasting models..."):
                    # Prepare data for modeling
                    prices = data['Close'].values
                    
                    forecasts = {}
                    
                    if forecast_model in ["ARIMA", "Both"]:
                        try:
                            arima_forecast = st.session_state.models.arima_forecast(prices, forecast_days)
                            forecasts['ARIMA'] = arima_forecast
                        except Exception as e:
                            st.error(f"ARIMA model error: {str(e)}")
                    
                    if forecast_model in ["LSTM", "Both"]:
                        try:
                            lstm_forecast = st.session_state.models.lstm_forecast(prices, forecast_days)
                            forecasts['LSTM'] = lstm_forecast
                        except Exception as e:
                            st.error(f"LSTM model error: {str(e)}")
                
                if forecasts:
                    # Create forecast visualization
                    fig = go.Figure()
                    
                    # Historical data
                    fig.add_trace(go.Scatter(
                        x=data.index,
                        y=data['Close'],
                        mode='lines',
                        name='Historical Price',
                        line=dict(color='blue')
                    ))
                    
                    # Future dates
                    last_date = data.index[-1]
                    future_dates = pd.date_range(
                        start=last_date + timedelta(days=1),
                        periods=forecast_days,
                        freq='D'
                    )
                    
                    # Plot forecasts
                    colors = ['red', 'green', 'orange']
                    for idx, (model_name, forecast) in enumerate(forecasts.items()):
                        fig.add_trace(go.Scatter(
                            x=future_dates,
                            y=forecast,
                            mode='lines+markers',
                            name=f'{model_name} Forecast',
                            line=dict(color=colors[idx % len(colors)], dash='dash')
                        ))
                    
                    fig.update_layout(
                        title=f"{selected_symbol} Price Forecast",
                        xaxis_title="Date",
                        yaxis_title="Price",
                        height=500
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Forecast summary
                    st.subheader("Forecast Summary")
                    forecast_summary = []
                    current_price = data['Close'].iloc[-1]
                    
                    for model_name, forecast in forecasts.items():
                        predicted_price = forecast[-1]
                        change = predicted_price - current_price
                        change_pct = (change / current_price) * 100
                        
                        forecast_summary.append({
                            'Model': model_name,
                            'Current Price': f"${current_price:.2f}",
                            f'Predicted Price ({forecast_days}d)': f"${predicted_price:.2f}",
                            'Expected Change': f"{change_pct:.2f}%",
                            'Signal': 'BUY' if change_pct > 2 else 'SELL' if change_pct < -2 else 'HOLD'
                        })
                    
                    st.dataframe(pd.DataFrame(forecast_summary), use_container_width=True)
    
    with tab3:
        st.header("Risk Analysis")
        
        if data_dict:
            # Risk metrics calculation
            st.subheader("Risk Metrics")
            
            risk_data = []
            for symbol, data in data_dict.items():
                returns = calculate_returns(data['Close'])
                
                # Calculate risk metrics
                sharpe_ratio = st.session_state.risk_analyzer.calculate_sharpe_ratio(returns)
                max_drawdown = st.session_state.risk_analyzer.calculate_max_drawdown(data['Close'])
                var_95 = st.session_state.risk_analyzer.calculate_var(returns, confidence=0.95)
                volatility = st.session_state.risk_analyzer.calculate_volatility(returns)
                
                risk_data.append({
                    'Symbol': symbol,
                    'Sharpe Ratio': f"{sharpe_ratio:.3f}",
                    'Max Drawdown': f"{max_drawdown:.2f}%",
                    'VaR (95%)': f"{var_95:.2f}%",
                    'Annualized Volatility': f"{volatility:.2f}%"
                })
            
            st.dataframe(pd.DataFrame(risk_data), use_container_width=True)
            
            # Drawdown chart
            selected_symbol_risk = st.selectbox("Select Symbol for Drawdown Analysis", symbols, key="risk_symbol")
            
            if selected_symbol_risk in data_dict:
                data = data_dict[selected_symbol_risk]
                drawdown_series = st.session_state.risk_analyzer.calculate_drawdown_series(data['Close'])
                
                fig = make_subplots(
                    rows=2, cols=1,
                    subplot_titles=[f"{selected_symbol_risk} Price", "Drawdown"],
                    vertical_spacing=0.1
                )
                
                # Price chart
                fig.add_trace(
                    go.Scatter(x=data.index, y=data['Close'], name="Price"),
                    row=1, col=1
                )
                
                # Drawdown chart
                fig.add_trace(
                    go.Scatter(
                        x=data.index, 
                        y=drawdown_series, 
                        fill='tonexty',
                        name="Drawdown",
                        line=dict(color='red')
                    ),
                    row=2, col=1
                )
                
                fig.update_layout(height=600, showlegend=False)
                st.plotly_chart(fig, use_container_width=True)
            
            # Correlation matrix
            if len(symbols) > 1:
                st.subheader("Correlation Matrix")
                
                # Calculate correlation matrix
                returns_df = pd.DataFrame()
                for symbol, data in data_dict.items():
                    returns_df[symbol] = calculate_returns(data['Close'])
                
                corr_matrix = returns_df.corr()
                
                # Create heatmap
                fig = px.imshow(
                    corr_matrix,
                    text_auto=True,
                    aspect="auto",
                    title="Asset Correlation Matrix"
                )
                
                st.plotly_chart(fig, use_container_width=True)
    
    with tab4:
        st.header("Backtesting Results")
        
        if data_dict:
            selected_symbol_bt = st.selectbox("Select Symbol for Backtesting", symbols, key="bt_symbol")
            
            # Backtesting parameters
            col1, col2 = st.columns(2)
            with col1:
                lookback_window = st.slider("Lookback Window (days)", 20, 100, 50)
            with col2:
                rebalance_freq = st.slider("Rebalance Frequency (days)", 1, 30, 5)
            
            if selected_symbol_bt in data_dict:
                data = data_dict[selected_symbol_bt]
                
                with st.spinner("Running backtesting..."):
                    # Simple moving average strategy backtesting
                    results = st.session_state.models.backtest_strategy(
                        data, lookback_window, rebalance_freq
                    )
                
                if results:
                    # Display backtest results
                    fig = make_subplots(
                        rows=3, cols=1,
                        subplot_titles=["Price & Signals", "Strategy Returns", "Cumulative Returns"],
                        vertical_spacing=0.1
                    )
                    
                    # Price and signals
                    fig.add_trace(
                        go.Scatter(x=data.index, y=data['Close'], name="Price"),
                        row=1, col=1
                    )
                    
                    # Strategy returns
                    if 'strategy_returns' in results:
                        fig.add_trace(
                            go.Scatter(
                                x=results['dates'], 
                                y=results['strategy_returns'], 
                                name="Strategy Returns"
                            ),
                            row=2, col=1
                        )
                    
                    # Cumulative returns
                    if 'cumulative_returns' in results:
                        fig.add_trace(
                            go.Scatter(
                                x=results['dates'], 
                                y=results['cumulative_returns'], 
                                name="Cumulative Returns"
                            ),
                            row=3, col=1
                        )
                    
                    fig.update_layout(height=800, showlegend=False)
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Backtest statistics
                    if 'stats' in results:
                        st.subheader("Backtest Statistics")
                        stats_df = pd.DataFrame([results['stats']])
                        st.dataframe(stats_df, use_container_width=True)

else:
    st.warning("Please select at least one symbol to display market data.")

# Footer
st.markdown("---")
st.markdown("**Last Updated:** " + st.session_state.last_refresh.strftime("%Y-%m-%d %H:%M:%S"))
st.markdown("*Data provided by Yahoo Finance and Alpha Vantage APIs*")

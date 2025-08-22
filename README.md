📊 Real-Time Market Analytics Dashboard

A powerful financial analytics dashboard built with Python + Streamlit, enabling real-time market tracking, forecasting, and risk analysis for stocks & forex. Designed for traders, analysts, and researchers who want live data, AI-powered predictions, and deep risk insights in one interactive platform.
<img width="668" height="821" alt="dashboard photo " src="https://github.com/user-attachments/assets/240f8ce6-1963-4799-92f5-3ec9b43071a8" />


🚀 Key Features

🔴 Real-Time Market Data

Live OHLCV Prices & Volume from Yahoo Finance

Forex Support via Alpha Vantage (with Yahoo fallback)

Configurable Auto-refresh (30s–300s)

Multi-Asset Coverage: Tech stocks, ETFs, and major forex pairs

🔮 Forecasting & AI Models

ARIMA with automatic parameter tuning

LSTM Neural Networks (TensorFlow/Keras) for deep learning predictions

Rolling Backtests with error metrics

Interactive Forecast Visualizations

⚖️ Risk Analytics

Sharpe Ratio: Risk-adjusted return

Max Drawdown: Peak-to-trough losses

Value at Risk (95%): Tail risk estimation

Volatility & Correlation Matrix for portfolio-level insights

📈 Interactive Streamlit Dashboard

📊 Live Data Tab – Prices, candlesticks, and summaries

🔮 Forecasting Tab – Predictive modeling (ARIMA/LSTM)

⚖️ Risk Tab – Metrics, volatility, and correlations

📈 Backtesting Tab – Strategy performance (e.g., MA crossover)

🏗️ Tech Stack

Frontend/UI: Streamlit + Plotly (interactive charts)

Backend/Data: Yahoo Finance & Alpha Vantage APIs

Machine Learning: TensorFlow/Keras (LSTM) & Statsmodels (ARIMA)

Analytics: Pandas, NumPy, Scikit-learn, SciPy, Matplotlib

📦 Installation
Prerequisites

Python 3.7+

pip or uv

Install dependencies
pip install streamlit yfinance plotly pandas numpy scikit-learn tensorflow statsmodels scipy matplotlib seaborn


Or with uv:

uv add streamlit yfinance plotly pandas numpy scikit-learn tensorflow statsmodels scipy matplotlib seaborn

🔧 Setup

Clone Repository

git clone https://github.com/Rachit-sahu/Real-time-market-analytics-dashboard/tree/main

cd real-time-market-analytics-dashboard


Install Packages

pip install -r requirements.txt


(Optional) Add Alpha Vantage API Key

export ALPHA_VANTAGE_API_KEY="your_api_key_here"


(Yahoo Finance will be used if not set)

Workflow

Select Equity or Forex

Choose Time Range (1D – 2Y)

Run Forecasts (ARIMA, LSTM, or both)

Explore Risk & Backtesting Tabs

📊 Risk Metrics
Metric	Meaning	Formula
Sharpe Ratio	Risk-adjusted return	(R - Rf) / σ
Max Drawdown	Largest peak-to-trough decline	max((Peak - Trough)/Peak)
VaR (95%)	Loss threshold at 95% confidence	5th percentile of returns
Volatility	Annualized price swings	σ × √252
🎯 Supported Assets

Equities: AAPL, MSFT, TSLA, NVDA, META, AMZN, SPY, QQQ

Forex: EURUSD, GBPUSD, USDJPY, AUDUSD, USDCAD

Easily extendable via app.py:

symbols = ["AAPL", "GOOGL", "YOUR_SYMBOL"]

🏗️ Architecture
📂 Project Root
 ├── app.py              # Streamlit app & UI
 ├── data_fetcher.py     # Data retrieval (Yahoo/Alpha Vantage)
 ├── models.py           # ARIMA & LSTM implementations
 ├── risk_analysis.py    # Sharpe, VaR, volatility, drawdown
 ├── utils.py            # Helpers & indicators
 └── requirements.txt    # Dependencies


Data Flow → Fetch ➝ Process ➝ Forecast ➝ Risk Analysis ➝ Visualization

📈 Performance Optimizations

Caching (5-min refresh)

Lightweight Pandas ops

Limited LSTM epochs for responsiveness

Fallback APIs & error handling

🤝 Contributing

Fork repo

Create branch → git checkout -b feature/awesome

Commit → git commit -m "Add new feature"

Push → git push origin feature/awesome

Open Pull Request 🎉

📄 License

Licensed under the MIT License


✨ Built with Streamlit, TensorFlow, and APIs to make market analytics smarter.


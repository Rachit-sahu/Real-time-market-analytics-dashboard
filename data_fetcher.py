import yfinance as yf
import pandas as pd
import numpy as np
import requests
import os
import streamlit as st
from datetime import datetime, timedelta
import time

class DataFetcher:
    """
    Data fetching class for market data from multiple sources
    """
    
    def __init__(self):
        self.alpha_vantage_key = os.getenv("ALPHA_VANTAGE_API_KEY", "demo")
        self.base_url_av = "https://www.alphavantage.co/query"
        
    def get_stock_data(self, symbol, period="6mo"):
        """
        Fetch stock data from Yahoo Finance
        
        Args:
            symbol (str): Stock symbol (e.g., 'AAPL')
            period (str): Time period ('1d', '5d', '1mo', '3mo', '6mo', '1y', '2y')
            
        Returns:
            pandas.DataFrame: Stock price data with OHLCV columns
        """
        try:
            # Create ticker object
            ticker = yf.Ticker(symbol)
            
            # Fetch historical data
            data = ticker.history(period=period)
            
            if data.empty:
                st.error(f"No data found for symbol {symbol}")
                return None
                
            # Clean data
            data = data.dropna()
            
            # Ensure we have the required columns
            required_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
            missing_columns = [col for col in required_columns if col not in data.columns]
            
            if missing_columns:
                st.warning(f"Missing columns for {symbol}: {missing_columns}")
            
            return data
            
        except Exception as e:
            st.error(f"Error fetching stock data for {symbol}: {str(e)}")
            return None
    
    def get_forex_data(self, symbol, period="6mo"):
        """
        Fetch forex data from Alpha Vantage API
        
        Args:
            symbol (str): Currency pair (e.g., 'EURUSD')
            period (str): Time period
            
        Returns:
            pandas.DataFrame: Forex price data
        """
        try:
            # For demo purposes, if no valid API key, use Yahoo Finance forex data
            if self.alpha_vantage_key == "demo":
                return self._get_forex_from_yahoo(symbol, period)
            
            # Alpha Vantage API call
            params = {
                'function': 'FX_DAILY',
                'from_symbol': symbol[:3],
                'to_symbol': symbol[3:],
                'apikey': self.alpha_vantage_key,
                'outputsize': 'full'
            }
            
            response = requests.get(self.base_url_av, params=params)
            data = response.json()
            
            if 'Error Message' in data:
                st.error(f"Alpha Vantage API Error: {data['Error Message']}")
                return self._get_forex_from_yahoo(symbol, period)
            
            if 'Note' in data:
                st.warning("Alpha Vantage API call limit reached. Using fallback data.")
                return self._get_forex_from_yahoo(symbol, period)
            
            # Parse Alpha Vantage data
            time_series = data.get('Time Series (Daily)', {})
            
            if not time_series:
                return self._get_forex_from_yahoo(symbol, period)
            
            # Convert to DataFrame
            df_data = []
            for date, values in time_series.items():
                df_data.append({
                    'Date': pd.to_datetime(date),
                    'Open': float(values['1. open']),
                    'High': float(values['2. high']),
                    'Low': float(values['3. low']),
                    'Close': float(values['4. close'])
                })
            
            df = pd.DataFrame(df_data)
            df.set_index('Date', inplace=True)
            df.sort_index(inplace=True)
            
            # Filter by period
            end_date = datetime.now()
            if period == '1d':
                start_date = end_date - timedelta(days=1)
            elif period == '5d':
                start_date = end_date - timedelta(days=5)
            elif period == '1mo':
                start_date = end_date - timedelta(days=30)
            elif period == '3mo':
                start_date = end_date - timedelta(days=90)
            elif period == '6mo':
                start_date = end_date - timedelta(days=180)
            elif period == '1y':
                start_date = end_date - timedelta(days=365)
            elif period == '2y':
                start_date = end_date - timedelta(days=730)
            else:
                start_date = end_date - timedelta(days=180)
            
            df = df[df.index >= start_date]
            
            return df
            
        except Exception as e:
            st.error(f"Error fetching forex data for {symbol}: {str(e)}")
            return self._get_forex_from_yahoo(symbol, period)
    
    def _get_forex_from_yahoo(self, symbol, period):
        """
        Fallback method to get forex data from Yahoo Finance
        
        Args:
            symbol (str): Currency pair
            period (str): Time period
            
        Returns:
            pandas.DataFrame: Forex price data
        """
        try:
            # Convert symbol format for Yahoo Finance
            yahoo_symbol = f"{symbol[:3]}{symbol[3:]}=X"
            
            ticker = yf.Ticker(yahoo_symbol)
            data = ticker.history(period=period)
            
            if data.empty:
                st.error(f"No forex data found for {symbol}")
                return None
            
            # Remove volume column as it's not relevant for forex
            if 'Volume' in data.columns:
                data = data.drop('Volume', axis=1)
            
            return data.dropna()
            
        except Exception as e:
            st.error(f"Error fetching forex data from Yahoo Finance for {symbol}: {str(e)}")
            return None
    
    def get_real_time_quote(self, symbol, asset_type="equity"):
        """
        Get real-time quote for a symbol
        
        Args:
            symbol (str): Symbol to fetch
            asset_type (str): 'equity' or 'forex'
            
        Returns:
            dict: Real-time quote data
        """
        try:
            if asset_type == "equity":
                ticker = yf.Ticker(symbol)
                info = ticker.info
                
                return {
                    'symbol': symbol,
                    'price': info.get('currentPrice', 0),
                    'change': info.get('regularMarketChange', 0),
                    'change_percent': info.get('regularMarketChangePercent', 0),
                    'volume': info.get('volume', 0),
                    'market_cap': info.get('marketCap', 0)
                }
            else:
                # For forex, use the latest price from historical data
                data = self.get_forex_data(symbol, "1d")
                if data is not None and not data.empty:
                    latest = data.iloc[-1]
                    prev = data.iloc[-2] if len(data) > 1 else latest
                    
                    change = latest['Close'] - prev['Close']
                    change_percent = (change / prev['Close']) * 100 if prev['Close'] != 0 else 0
                    
                    return {
                        'symbol': symbol,
                        'price': latest['Close'],
                        'change': change,
                        'change_percent': change_percent,
                        'volume': 0,  # Forex doesn't have volume
                        'market_cap': 0
                    }
                
        except Exception as e:
            st.error(f"Error fetching real-time quote for {symbol}: {str(e)}")
            return None
    
    def get_market_news(self, symbol):
        """
        Get market news for a symbol (placeholder for future implementation)
        
        Args:
            symbol (str): Symbol to get news for
            
        Returns:
            list: List of news items
        """
        try:
            ticker = yf.Ticker(symbol)
            news = ticker.news
            
            return news[:5] if news else []
            
        except Exception as e:
            st.error(f"Error fetching news for {symbol}: {str(e)}")
            return []

import pandas as pd
import numpy as np
import streamlit as st
from datetime import datetime, timedelta

def format_number(num):
    """
    Format large numbers with appropriate suffixes (K, M, B)
    
    Args:
        num (float): Number to format
        
    Returns:
        str: Formatted number string
    """
    try:
        if pd.isna(num) or num == 0:
            return "0"
        
        num = float(num)
        
        if abs(num) >= 1e9:
            return f"{num/1e9:.2f}B"
        elif abs(num) >= 1e6:
            return f"{num/1e6:.2f}M"
        elif abs(num) >= 1e3:
            return f"{num/1e3:.2f}K"
        else:
            return f"{num:.2f}"
            
    except:
        return str(num)

def calculate_returns(prices):
    """
    Calculate returns from price series
    
    Args:
        prices (pd.Series): Price series
        
    Returns:
        pd.Series: Returns series
    """
    try:
        return prices.pct_change().dropna()
    except:
        return pd.Series()

def calculate_rolling_metrics(data, window=30):
    """
    Calculate rolling metrics for a given window
    
    Args:
        data (pd.Series): Data series
        window (int): Rolling window size
        
    Returns:
        dict: Dictionary of rolling metrics
    """
    try:
        return {
            'mean': data.rolling(window=window).mean(),
            'std': data.rolling(window=window).std(),
            'min': data.rolling(window=window).min(),
            'max': data.rolling(window=window).max()
        }
    except:
        return {}

def detect_outliers(data, method='iqr', threshold=1.5):
    """
    Detect outliers in data using IQR or Z-score method
    
    Args:
        data (pd.Series): Data series
        method (str): Method to use ('iqr' or 'zscore')
        threshold (float): Threshold for outlier detection
        
    Returns:
        pd.Series: Boolean series indicating outliers
    """
    try:
        if method == 'iqr':
            Q1 = data.quantile(0.25)
            Q3 = data.quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - threshold * IQR
            upper_bound = Q3 + threshold * IQR
            return (data < lower_bound) | (data > upper_bound)
        
        elif method == 'zscore':
            z_scores = np.abs((data - data.mean()) / data.std())
            return z_scores > threshold
            
    except:
        return pd.Series([False] * len(data), index=data.index)

def clean_data(data, remove_outliers=True, fill_method='forward'):
    """
    Clean and preprocess market data
    
    Args:
        data (pd.DataFrame): Market data
        remove_outliers (bool): Whether to remove outliers
        fill_method (str): Method to fill missing values
        
    Returns:
        pd.DataFrame: Cleaned data
    """
    try:
        # Make a copy to avoid modifying original data
        cleaned_data = data.copy()
        
        # Remove rows with all NaN values
        cleaned_data = cleaned_data.dropna(how='all')
        
        # Fill missing values
        if fill_method == 'forward':
            cleaned_data = cleaned_data.fillna(method='ffill')
        elif fill_method == 'backward':
            cleaned_data = cleaned_data.fillna(method='bfill')
        elif fill_method == 'interpolate':
            cleaned_data = cleaned_data.interpolate()
        
        # Remove outliers if requested
        if remove_outliers and 'Close' in cleaned_data.columns:
            outliers = detect_outliers(cleaned_data['Close'])
            if not outliers.empty:
                cleaned_data = cleaned_data[~outliers]
        
        return cleaned_data
        
    except Exception as e:
        st.error(f"Error cleaning data: {str(e)}")
        return data

def calculate_technical_indicators(data):
    """
    Calculate common technical indicators
    
    Args:
        data (pd.DataFrame): OHLCV data
        
    Returns:
        pd.DataFrame: Data with technical indicators added
    """
    try:
        df = data.copy()
        
        # Simple Moving Averages
        df['SMA_20'] = df['Close'].rolling(window=20).mean()
        df['SMA_50'] = df['Close'].rolling(window=50).mean()
        df['SMA_200'] = df['Close'].rolling(window=200).mean()
        
        # Exponential Moving Averages
        df['EMA_12'] = df['Close'].ewm(span=12).mean()
        df['EMA_26'] = df['Close'].ewm(span=26).mean()
        
        # MACD
        df['MACD'] = df['EMA_12'] - df['EMA_26']
        df['MACD_Signal'] = df['MACD'].ewm(span=9).mean()
        df['MACD_Histogram'] = df['MACD'] - df['MACD_Signal']
        
        # RSI
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['RSI'] = 100 - (100 / (1 + rs))
        
        # Bollinger Bands
        df['BB_Middle'] = df['Close'].rolling(window=20).mean()
        bb_std = df['Close'].rolling(window=20).std()
        df['BB_Upper'] = df['BB_Middle'] + (bb_std * 2)
        df['BB_Lower'] = df['BB_Middle'] - (bb_std * 2)
        
        # Average True Range (ATR)
        high_low = df['High'] - df['Low']
        high_close = np.abs(df['High'] - df['Close'].shift())
        low_close = np.abs(df['Low'] - df['Close'].shift())
        true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        df['ATR'] = true_range.rolling(window=14).mean()
        
        return df
        
    except Exception as e:
        st.error(f"Error calculating technical indicators: {str(e)}")
        return data

def validate_data_quality(data, symbol):
    """
    Validate data quality and provide warnings
    
    Args:
        data (pd.DataFrame): Market data
        symbol (str): Symbol name for error messages
        
    Returns:
        dict: Data quality report
    """
    try:
        report = {
            'symbol': symbol,
            'total_records': len(data),
            'missing_values': data.isnull().sum().to_dict(),
            'date_range': {
                'start': data.index.min() if not data.empty else None,
                'end': data.index.max() if not data.empty else None
            },
            'warnings': [],
            'errors': []
        }
        
        # Check for minimum data requirements
        if len(data) < 30:
            report['warnings'].append(f"Limited data: only {len(data)} records available")
        
        # Check for missing values
        total_missing = data.isnull().sum().sum()
        if total_missing > 0:
            report['warnings'].append(f"Found {total_missing} missing values")
        
        # Check for duplicate dates
        if data.index.duplicated().any():
            report['errors'].append("Duplicate dates found in data")
        
        # Check for negative prices
        if 'Close' in data.columns and (data['Close'] <= 0).any():
            report['errors'].append("Negative or zero prices found")
        
        # Check for extreme price movements (>50% in a day)
        if 'Close' in data.columns and len(data) > 1:
            returns = data['Close'].pct_change().abs()
            extreme_moves = (returns > 0.5).sum()
            if extreme_moves > 0:
                report['warnings'].append(f"Found {extreme_moves} extreme price movements (>50%)")
        
        return report
        
    except Exception as e:
        return {
            'symbol': symbol,
            'error': f"Error validating data: {str(e)}"
        }

def create_date_range(start_date, end_date, freq='D'):
    """
    Create a date range for time series analysis
    
    Args:
        start_date (str/datetime): Start date
        end_date (str/datetime): End date
        freq (str): Frequency ('D' for daily, 'H' for hourly, etc.)
        
    Returns:
        pd.DatetimeIndex: Date range
    """
    try:
        return pd.date_range(start=start_date, end=end_date, freq=freq)
    except Exception as e:
        st.error(f"Error creating date range: {str(e)}")
        return pd.DatetimeIndex([])

def performance_attribution(returns, benchmark_returns):
    """
    Simple performance attribution analysis
    
    Args:
        returns (pd.Series): Portfolio returns
        benchmark_returns (pd.Series): Benchmark returns
        
    Returns:
        dict: Performance attribution metrics
    """
    try:
        # Align the series
        aligned_data = pd.concat([returns, benchmark_returns], axis=1, join='inner')
        aligned_data.columns = ['portfolio', 'benchmark']
        aligned_data = aligned_data.dropna()
        
        if len(aligned_data) < 2:
            return {}
        
        # Calculate metrics
        active_return = aligned_data['portfolio'].mean() - aligned_data['benchmark'].mean()
        tracking_error = (aligned_data['portfolio'] - aligned_data['benchmark']).std()
        information_ratio = active_return / tracking_error if tracking_error != 0 else 0
        
        # Beta calculation
        covariance = np.cov(aligned_data['portfolio'], aligned_data['benchmark'])[0][1]
        benchmark_variance = np.var(aligned_data['benchmark'])
        beta = covariance / benchmark_variance if benchmark_variance != 0 else 1
        
        # Alpha calculation (Jensen's alpha)
        alpha = active_return - beta * aligned_data['benchmark'].mean()
        
        return {
            'Active Return (%)': active_return * 100,
            'Tracking Error (%)': tracking_error * 100,
            'Information Ratio': information_ratio,
            'Beta': beta,
            'Alpha (%)': alpha * 100
        }
        
    except Exception as e:
        st.error(f"Error in performance attribution: {str(e)}")
        return {}

@st.cache_data(ttl=300)  # Cache for 5 minutes
def cached_calculation(func, *args, **kwargs):
    """
    Generic caching wrapper for expensive calculations
    
    Args:
        func: Function to cache
        *args: Function arguments
        **kwargs: Function keyword arguments
        
    Returns:
        Function result
    """
    try:
        return func(*args, **kwargs)
    except Exception as e:
        st.error(f"Error in cached calculation: {str(e)}")
        return None

def export_data_to_csv(data, filename):
    """
    Export data to CSV format for download
    
    Args:
        data (pd.DataFrame): Data to export
        filename (str): Filename for the export
        
    Returns:
        str: CSV string for download
    """
    try:
        return data.to_csv(index=True)
    except Exception as e:
        st.error(f"Error exporting data: {str(e)}")
        return ""

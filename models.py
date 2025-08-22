import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller
import streamlit as st
import warnings
warnings.filterwarnings('ignore')

# TensorFlow/Keras imports with error handling
try:
    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense, Dropout
    from tensorflow.keras.optimizers import Adam
    TENSORFLOW_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False
    st.warning("TensorFlow not available. LSTM models will be disabled.")

class ForecastingModels:
    """
    Time series forecasting models for market data
    """
    
    def __init__(self):
        self.scaler = MinMaxScaler()
        
    def check_stationarity(self, timeseries):
        """
        Check if time series is stationary using Augmented Dickey-Fuller test
        
        Args:
            timeseries (array-like): Time series data
            
        Returns:
            bool: True if stationary, False otherwise
        """
        try:
            result = adfuller(timeseries)
            return result[1] <= 0.05  # p-value <= 0.05 indicates stationarity
        except:
            return False
    
    def make_stationary(self, timeseries):
        """
        Make time series stationary by differencing
        
        Args:
            timeseries (array-like): Time series data
            
        Returns:
            numpy.array: Differenced time series
        """
        diff_series = np.diff(timeseries)
        return diff_series
    
    def arima_forecast(self, data, forecast_steps=7):
        """
        ARIMA forecasting model
        
        Args:
            data (array-like): Historical price data
            forecast_steps (int): Number of steps to forecast
            
        Returns:
            numpy.array: Forecasted values
        """
        try:
            # Convert to numpy array if needed
            if isinstance(data, pd.Series):
                data = data.values
            
            # Check for minimum data length
            if len(data) < 50:
                st.warning("Insufficient data for ARIMA model. Need at least 50 data points.")
                return np.full(forecast_steps, data[-1])
            
            # Check stationarity and difference if needed
            series = data.copy()
            d = 0
            
            if not self.check_stationarity(series):
                series = self.make_stationary(series)
                d = 1
                
                if not self.check_stationarity(series) and len(series) > 1:
                    series = self.make_stationary(series)
                    d = 2
            
            # Find optimal ARIMA parameters using AIC
            best_aic = float('inf')
            best_order = (1, d, 1)
            
            # Grid search for optimal parameters (limited for performance)
            for p in range(0, min(4, len(series) // 10)):
                for q in range(0, min(4, len(series) // 10)):
                    try:
                        model = ARIMA(data, order=(p, d, q))
                        fitted_model = model.fit()
                        
                        if fitted_model.aic < best_aic:
                            best_aic = fitted_model.aic
                            best_order = (p, d, q)
                            
                    except:
                        continue
            
            # Fit the best model
            model = ARIMA(data, order=best_order)
            fitted_model = model.fit()
            
            # Generate forecast
            forecast = fitted_model.forecast(steps=forecast_steps)
            
            # Ensure forecast values are positive (for price data)
            forecast = np.maximum(forecast, 0.01)
            
            return forecast
            
        except Exception as e:
            st.error(f"ARIMA forecasting error: {str(e)}")
            # Return simple naive forecast as fallback
            return np.full(forecast_steps, data[-1])
    
    def lstm_forecast(self, data, forecast_steps=7, lookback=60):
        """
        LSTM forecasting model
        
        Args:
            data (array-like): Historical price data
            forecast_steps (int): Number of steps to forecast
            lookback (int): Number of previous time steps to use
            
        Returns:
            numpy.array: Forecasted values
        """
        if not TENSORFLOW_AVAILABLE:
            st.warning("TensorFlow not available. Using ARIMA fallback.")
            return self.arima_forecast(data, forecast_steps)
        
        try:
            # Convert to numpy array if needed
            if isinstance(data, pd.Series):
                data = data.values
            
            # Check for minimum data length
            if len(data) < lookback + 20:
                st.warning(f"Insufficient data for LSTM model. Need at least {lookback + 20} data points.")
                return np.full(forecast_steps, data[-1])
            
            # Prepare data
            data = data.reshape(-1, 1)
            scaled_data = self.scaler.fit_transform(data)
            
            # Create training sequences
            X, y = [], []
            for i in range(lookback, len(scaled_data)):
                X.append(scaled_data[i-lookback:i, 0])
                y.append(scaled_data[i, 0])
            
            X, y = np.array(X), np.array(y)
            X = X.reshape((X.shape[0], X.shape[1], 1))
            
            # Build LSTM model
            model = Sequential([
                LSTM(50, return_sequences=True, input_shape=(lookback, 1)),
                Dropout(0.2),
                LSTM(50, return_sequences=False),
                Dropout(0.2),
                Dense(25),
                Dense(1)
            ])
            
            model.compile(optimizer=Adam(learning_rate=0.001), loss='mse')
            
            # Train model (with limited epochs for performance)
            model.fit(X, y, epochs=20, batch_size=32, verbose=0)
            
            # Generate forecasts
            forecasts = []
            last_sequence = scaled_data[-lookback:].reshape(1, lookback, 1)
            
            for _ in range(forecast_steps):
                next_pred = model.predict(last_sequence, verbose=0)
                forecasts.append(next_pred[0, 0])
                
                # Update sequence for next prediction
                last_sequence = np.roll(last_sequence, -1, axis=1)
                last_sequence[0, -1, 0] = next_pred[0, 0]
            
            # Inverse transform forecasts
            forecasts = np.array(forecasts).reshape(-1, 1)
            forecasts = self.scaler.inverse_transform(forecasts).flatten()
            
            # Ensure forecast values are positive
            forecasts = np.maximum(forecasts, 0.01)
            
            return forecasts
            
        except Exception as e:
            st.error(f"LSTM forecasting error: {str(e)}")
            # Return ARIMA forecast as fallback
            return self.arima_forecast(data, forecast_steps)
    
    def evaluate_model(self, actual, predicted):
        """
        Evaluate model performance
        
        Args:
            actual (array-like): Actual values
            predicted (array-like): Predicted values
            
        Returns:
            dict: Evaluation metrics
        """
        try:
            mse = mean_squared_error(actual, predicted)
            rmse = np.sqrt(mse)
            mae = mean_absolute_error(actual, predicted)
            mape = np.mean(np.abs((actual - predicted) / actual)) * 100
            
            return {
                'MSE': mse,
                'RMSE': rmse,
                'MAE': mae,
                'MAPE': mape
            }
        except:
            return {
                'MSE': 0,
                'RMSE': 0,
                'MAE': 0,
                'MAPE': 0
            }
    
    def backtest_strategy(self, data, lookback_window=50, rebalance_freq=5):
        """
        Backtest a simple moving average crossover strategy
        
        Args:
            data (pd.DataFrame): Historical price data with OHLC columns
            lookback_window (int): Lookback window for moving average
            rebalance_freq (int): Frequency of rebalancing
            
        Returns:
            dict: Backtesting results
        """
        try:
            if len(data) < lookback_window + 20:
                return {'error': 'Insufficient data for backtesting'}
            
            # Calculate moving averages
            short_ma = data['Close'].rolling(window=lookback_window//2).mean()
            long_ma = data['Close'].rolling(window=lookback_window).mean()
            
            # Generate signals
            signals = pd.DataFrame(index=data.index)
            signals['signal'] = 0
            signals['signal'][short_ma > long_ma] = 1
            signals['signal'][short_ma < long_ma] = -1
            
            # Calculate positions (only rebalance every rebalance_freq days)
            signals['positions'] = signals['signal'].rolling(window=rebalance_freq).mean()
            signals['positions'] = signals['positions'].fillna(method='ffill')
            
            # Calculate returns
            returns = data['Close'].pct_change()
            signals['market_returns'] = returns
            signals['strategy_returns'] = signals['positions'].shift(1) * returns
            
            # Remove NaN values
            signals = signals.dropna()
            
            if len(signals) == 0:
                return {'error': 'No valid signals generated'}
            
            # Calculate cumulative returns
            signals['cumulative_market_returns'] = (1 + signals['market_returns']).cumprod()
            signals['cumulative_strategy_returns'] = (1 + signals['strategy_returns']).cumprod()
            
            # Calculate performance metrics
            total_return = (signals['cumulative_strategy_returns'].iloc[-1] - 1) * 100
            market_return = (signals['cumulative_market_returns'].iloc[-1] - 1) * 100
            excess_return = total_return - market_return
            
            # Calculate Sharpe ratio
            strategy_sharpe = (signals['strategy_returns'].mean() / signals['strategy_returns'].std()) * np.sqrt(252) if signals['strategy_returns'].std() > 0 else 0
            
            # Calculate maximum drawdown
            cumulative = signals['cumulative_strategy_returns']
            running_max = cumulative.expanding().max()
            drawdown = (cumulative - running_max) / running_max
            max_drawdown = drawdown.min() * 100
            
            return {
                'dates': signals.index.tolist(),
                'strategy_returns': signals['strategy_returns'].tolist(),
                'cumulative_returns': signals['cumulative_strategy_returns'].tolist(),
                'stats': {
                    'Total Return (%)': f"{total_return:.2f}",
                    'Market Return (%)': f"{market_return:.2f}",
                    'Excess Return (%)': f"{excess_return:.2f}",
                    'Sharpe Ratio': f"{strategy_sharpe:.3f}",
                    'Max Drawdown (%)': f"{max_drawdown:.2f}",
                    'Number of Trades': len(signals[signals['positions'].diff() != 0])
                }
            }
            
        except Exception as e:
            return {'error': f'Backtesting error: {str(e)}'}

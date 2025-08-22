import numpy as np
import pandas as pd
from scipy import stats
import streamlit as st

class RiskAnalyzer:
    """
    Risk analysis and metrics calculation for financial data
    """
    
    def __init__(self, risk_free_rate=0.02):
        """
        Initialize RiskAnalyzer
        
        Args:
            risk_free_rate (float): Risk-free rate for Sharpe ratio calculation
        """
        self.risk_free_rate = risk_free_rate
    
    def calculate_returns(self, prices):
        """
        Calculate returns from price series
        
        Args:
            prices (pd.Series): Price series
            
        Returns:
            pd.Series: Returns series
        """
        return prices.pct_change().dropna()
    
    def calculate_sharpe_ratio(self, returns, risk_free_rate=None):
        """
        Calculate Sharpe ratio
        
        Args:
            returns (pd.Series): Returns series
            risk_free_rate (float): Risk-free rate (annualized)
            
        Returns:
            float: Sharpe ratio
        """
        try:
            if risk_free_rate is None:
                risk_free_rate = self.risk_free_rate
            
            if len(returns) == 0 or returns.std() == 0:
                return 0.0
            
            # Convert risk-free rate to daily
            daily_rf_rate = risk_free_rate / 252
            
            # Calculate excess returns
            excess_returns = returns - daily_rf_rate
            
            # Calculate Sharpe ratio (annualized)
            sharpe = (excess_returns.mean() / returns.std()) * np.sqrt(252)
            
            return sharpe if not np.isnan(sharpe) else 0.0
            
        except Exception as e:
            st.error(f"Error calculating Sharpe ratio: {str(e)}")
            return 0.0
    
    def calculate_sortino_ratio(self, returns, risk_free_rate=None):
        """
        Calculate Sortino ratio (downside deviation)
        
        Args:
            returns (pd.Series): Returns series
            risk_free_rate (float): Risk-free rate
            
        Returns:
            float: Sortino ratio
        """
        try:
            if risk_free_rate is None:
                risk_free_rate = self.risk_free_rate
            
            if len(returns) == 0:
                return 0.0
            
            daily_rf_rate = risk_free_rate / 252
            excess_returns = returns - daily_rf_rate
            
            # Calculate downside deviation
            downside_returns = returns[returns < 0]
            if len(downside_returns) == 0:
                return np.inf
            
            downside_deviation = np.sqrt(np.mean(downside_returns**2))
            
            if downside_deviation == 0:
                return np.inf
            
            sortino = (excess_returns.mean() / downside_deviation) * np.sqrt(252)
            
            return sortino if not np.isnan(sortino) else 0.0
            
        except Exception as e:
            st.error(f"Error calculating Sortino ratio: {str(e)}")
            return 0.0
    
    def calculate_max_drawdown(self, prices):
        """
        Calculate maximum drawdown
        
        Args:
            prices (pd.Series): Price series
            
        Returns:
            float: Maximum drawdown as percentage
        """
        try:
            if len(prices) == 0:
                return 0.0
            
            # Calculate cumulative returns
            cumulative = prices / prices.iloc[0]
            
            # Calculate running maximum
            running_max = cumulative.expanding().max()
            
            # Calculate drawdown
            drawdown = (cumulative - running_max) / running_max
            
            # Return maximum drawdown as percentage
            max_dd = drawdown.min() * 100
            
            return max_dd if not np.isnan(max_dd) else 0.0
            
        except Exception as e:
            st.error(f"Error calculating max drawdown: {str(e)}")
            return 0.0
    
    def calculate_drawdown_series(self, prices):
        """
        Calculate drawdown series for plotting
        
        Args:
            prices (pd.Series): Price series
            
        Returns:
            pd.Series: Drawdown series
        """
        try:
            if len(prices) == 0:
                return pd.Series()
            
            cumulative = prices / prices.iloc[0]
            running_max = cumulative.expanding().max()
            drawdown = (cumulative - running_max) / running_max * 100
            
            return drawdown
            
        except Exception as e:
            st.error(f"Error calculating drawdown series: {str(e)}")
            return pd.Series()
    
    def calculate_var(self, returns, confidence=0.95):
        """
        Calculate Value at Risk (VaR)
        
        Args:
            returns (pd.Series): Returns series
            confidence (float): Confidence level (0.95 for 95% VaR)
            
        Returns:
            float: VaR as percentage
        """
        try:
            if len(returns) == 0:
                return 0.0
            
            # Historical VaR
            var = np.percentile(returns, (1 - confidence) * 100) * 100
            
            return var if not np.isnan(var) else 0.0
            
        except Exception as e:
            st.error(f"Error calculating VaR: {str(e)}")
            return 0.0
    
    def calculate_cvar(self, returns, confidence=0.95):
        """
        Calculate Conditional Value at Risk (CVaR) / Expected Shortfall
        
        Args:
            returns (pd.Series): Returns series
            confidence (float): Confidence level
            
        Returns:
            float: CVaR as percentage
        """
        try:
            if len(returns) == 0:
                return 0.0
            
            var = np.percentile(returns, (1 - confidence) * 100)
            cvar = returns[returns <= var].mean() * 100
            
            return cvar if not np.isnan(cvar) else 0.0
            
        except Exception as e:
            st.error(f"Error calculating CVaR: {str(e)}")
            return 0.0
    
    def calculate_volatility(self, returns, annualized=True):
        """
        Calculate volatility
        
        Args:
            returns (pd.Series): Returns series
            annualized (bool): Whether to annualize the volatility
            
        Returns:
            float: Volatility as percentage
        """
        try:
            if len(returns) == 0:
                return 0.0
            
            vol = returns.std()
            
            if annualized:
                vol *= np.sqrt(252)  # Annualize assuming 252 trading days
            
            return vol * 100 if not np.isnan(vol) else 0.0
            
        except Exception as e:
            st.error(f"Error calculating volatility: {str(e)}")
            return 0.0
    
    def calculate_beta(self, asset_returns, market_returns):
        """
        Calculate beta (systematic risk) relative to market
        
        Args:
            asset_returns (pd.Series): Asset returns
            market_returns (pd.Series): Market returns
            
        Returns:
            float: Beta value
        """
        try:
            if len(asset_returns) == 0 or len(market_returns) == 0:
                return 1.0
            
            # Align the series
            aligned_data = pd.concat([asset_returns, market_returns], axis=1, join='inner')
            aligned_data.columns = ['asset', 'market']
            aligned_data = aligned_data.dropna()
            
            if len(aligned_data) < 2:
                return 1.0
            
            # Calculate beta using covariance
            covariance = np.cov(aligned_data['asset'], aligned_data['market'])[0][1]
            market_variance = np.var(aligned_data['market'])
            
            if market_variance == 0:
                return 1.0
            
            beta = covariance / market_variance
            
            return beta if not np.isnan(beta) else 1.0
            
        except Exception as e:
            st.error(f"Error calculating beta: {str(e)}")
            return 1.0
    
    def calculate_information_ratio(self, asset_returns, benchmark_returns):
        """
        Calculate Information Ratio
        
        Args:
            asset_returns (pd.Series): Asset returns
            benchmark_returns (pd.Series): Benchmark returns
            
        Returns:
            float: Information ratio
        """
        try:
            if len(asset_returns) == 0 or len(benchmark_returns) == 0:
                return 0.0
            
            # Align the series
            aligned_data = pd.concat([asset_returns, benchmark_returns], axis=1, join='inner')
            aligned_data.columns = ['asset', 'benchmark']
            aligned_data = aligned_data.dropna()
            
            if len(aligned_data) < 2:
                return 0.0
            
            # Calculate excess returns
            excess_returns = aligned_data['asset'] - aligned_data['benchmark']
            
            # Calculate tracking error
            tracking_error = excess_returns.std()
            
            if tracking_error == 0:
                return 0.0
            
            # Calculate information ratio
            info_ratio = (excess_returns.mean() / tracking_error) * np.sqrt(252)
            
            return info_ratio if not np.isnan(info_ratio) else 0.0
            
        except Exception as e:
            st.error(f"Error calculating information ratio: {str(e)}")
            return 0.0
    
    def calculate_calmar_ratio(self, returns, prices):
        """
        Calculate Calmar Ratio (Annual Return / Maximum Drawdown)
        
        Args:
            returns (pd.Series): Returns series
            prices (pd.Series): Price series
            
        Returns:
            float: Calmar ratio
        """
        try:
            if len(returns) == 0 or len(prices) == 0:
                return 0.0
            
            # Calculate annualized return
            total_return = (prices.iloc[-1] / prices.iloc[0]) - 1
            years = len(prices) / 252  # Assuming 252 trading days per year
            annual_return = (1 + total_return) ** (1 / years) - 1 if years > 0 else 0
            
            # Calculate maximum drawdown (as decimal)
            max_dd = abs(self.calculate_max_drawdown(prices) / 100)
            
            if max_dd == 0:
                return np.inf
            
            calmar = annual_return / max_dd
            
            return calmar if not np.isnan(calmar) else 0.0
            
        except Exception as e:
            st.error(f"Error calculating Calmar ratio: {str(e)}")
            return 0.0
    
    def calculate_risk_metrics_summary(self, prices, market_prices=None):
        """
        Calculate a comprehensive summary of risk metrics
        
        Args:
            prices (pd.Series): Asset price series
            market_prices (pd.Series): Market price series for beta calculation
            
        Returns:
            dict: Dictionary of risk metrics
        """
        try:
            returns = self.calculate_returns(prices)
            
            metrics = {
                'Sharpe Ratio': self.calculate_sharpe_ratio(returns),
                'Sortino Ratio': self.calculate_sortino_ratio(returns),
                'Max Drawdown (%)': self.calculate_max_drawdown(prices),
                'VaR 95% (%)': self.calculate_var(returns, 0.95),
                'CVaR 95% (%)': self.calculate_cvar(returns, 0.95),
                'Volatility (%)': self.calculate_volatility(returns),
                'Calmar Ratio': self.calculate_calmar_ratio(returns, prices)
            }
            
            # Add beta if market data is provided
            if market_prices is not None:
                market_returns = self.calculate_returns(market_prices)
                metrics['Beta'] = self.calculate_beta(returns, market_returns)
                metrics['Information Ratio'] = self.calculate_information_ratio(returns, market_returns)
            
            return metrics
            
        except Exception as e:
            st.error(f"Error calculating risk metrics summary: {str(e)}")
            return {}

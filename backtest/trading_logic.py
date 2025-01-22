from abc import ABC, abstractmethod
import pandas as pd
import numpy as np

class BaseTradingLogic(ABC):
    """
    Abstract base class for trading logic implementations.
    """
    def __init__(self):
        pass
     

    @abstractmethod
    def execute_trades(self, data: pd.DataFrame, signals: pd.DataFrame) -> pd.DataFrame:
        """
        Execute trades based on signals and current market data.
        
        Args:
            data (pd.DataFrame): Market data with price information
            signals (pd.DataFrame): Trading signals
            
        Returns:
            pd.DataFrame: Portfolio state after trading
        """
        pass

    def _open_long_position(self, portfolio: pd.DataFrame, i: int):
        portfolio.iloc[i, portfolio.columns.get_loc('holdings')] = 1


    def _open_short_position(self, portfolio: pd.DataFrame, i: int):
        portfolio.iloc[i, portfolio.columns.get_loc('holdings')] = -1

    def _close_position(self, portfolio: pd.DataFrame, i: int):
        portfolio.iloc[i, portfolio.columns.get_loc('holdings')] = 0


    def _switch_to_short(self, portfolio: pd.DataFrame, i: int):
        self._close_position(portfolio, i)
        self._open_short_position(portfolio, i)

    def _switch_to_long(self, portfolio: pd.DataFrame, i: int):
        self._close_position(portfolio, i)
        self._open_long_position(portfolio, i) 

class StandardTradingLogic(BaseTradingLogic):
    """
    Standard trading logic implementation with the following rules:
    
    No Position:
    - Signal 1: Open long position
    - Signal -1: Open short position
    - Signal 0: No action
    
    Long Position:
    - Signal 1: No action
    - Signal -1: Switch to short (sell 2x)
    - Signal 0: Close position
    
    Short Position:
    - Signal 1: Switch to long (buy 2x)
    - Signal -1: No action
    - Signal 0: Close position
    """
    def execute_trades(self, data: pd.DataFrame, signals: pd.DataFrame) -> pd.DataFrame:
        # Initialize portfolio
        portfolio = pd.DataFrame(index=data.index)
        portfolio['holdings'] = 0
        portfolio['signal'] = signals['signal']
        portfolio['close'] = data['close']
        
        # Simulate trading
        for i in range(len(data)):
            signal = signals.iloc[i]['signal']
            
            if i == 0:
                continue
                
            # Copy previous day's state (except signal and close)
            portfolio.iloc[i, portfolio.columns.get_loc('holdings')] = portfolio.iloc[i-1]['holdings']

            current_holdings = portfolio.iloc[i]['holdings']
            
            # Process signals based on current position
            if current_holdings == 0:  # No position
                if signal == 1:  # Open long
                    self._open_long_position(portfolio, i)
                elif signal == -1:  # Open short
                    self._open_short_position(portfolio, i)
                    
            elif current_holdings > 0:  # Long position
                if signal == -1:  # Switch to short
                    self._switch_to_short(portfolio, i)
                elif signal == 0:  # Close position
                    self._close_position(portfolio, i)
                    
            elif current_holdings < 0:  # Short position
                if signal == 1:  # Switch to long
                    self._switch_to_long(portfolio, i)
                elif signal == 0:  # Close position
                    self._close_position(portfolio, i)
        
        portfolio['Date'] = data['Date']
        return portfolio

class HoldTradingLogic(BaseTradingLogic):
    """
    Hold trading logic implementation with the following rules:
    
    No Position:
    - Signal 1: Open long position
    - Signal -1: Open short position
    - Signal 0: No action
    
    Long Position:
    - Signal 1: No action
    - Signal -1: Switch to short
    - Signal 0: No action
    
    Short Position:
    - Signal 1: Switch to long
    - Signal -1: No action
    - Signal 0: No action
    """
    def execute_trades(self, data: pd.DataFrame, signals: pd.DataFrame) -> pd.DataFrame:
        # Initialize portfolio
        portfolio = pd.DataFrame(index=data.index)
        portfolio['holdings'] = 0
        portfolio['signal'] = signals['signal']
        portfolio['close'] = data['close']
        
        # Simulate trading
        for i in range(len(data)):
            signal = signals.iloc[i]['signal']
            
            if i == 0:
                continue
                
            # Copy previous day's state (except signal and close)
            portfolio.iloc[i, portfolio.columns.get_loc('holdings')] = portfolio.iloc[i-1]['holdings']

            current_holdings = portfolio.iloc[i]['holdings']
            
            # Process signals based on current position
            if current_holdings == 0:  # No position
                if signal == 1:  # Open long
                    self._open_long_position(portfolio, i)
                elif signal == -1:  # Open short
                    self._open_short_position(portfolio, i)
                    
            elif current_holdings > 0:  # Long position
                if signal == -1:  # Switch to short
                    self._switch_to_short(portfolio, i)
         
                    
            elif current_holdings < 0:  # Short position
                if signal == 1:  # Switch to long
                    self._switch_to_long(portfolio, i)
         
        portfolio['Date'] = data['Date']
        return portfolio

class LongOnlyTradingLogic(BaseTradingLogic):
    """
    Long Only trading logic implementation with the following rules:

    No Position:
    - Signal 1: Open long position
    - Signal -1: No action
    - Signal 0: No action
    
    Long Position:
    - Signal 1: No action
    - Signal -1: Close position
    - Signal 0: No action
    """
    def execute_trades(self, data: pd.DataFrame, signals: pd.DataFrame) -> pd.DataFrame:
        # Initialize portfolio
        portfolio = pd.DataFrame(index=data.index)
        portfolio['holdings'] = 0
        portfolio['signal'] = signals['signal']
        portfolio['close'] = data['close']
        
        # Simulate trading
        for i in range(len(data)):
            signal = signals.iloc[i]['signal']
            
            if i == 0:
                continue
                
            # Copy previous day's state (except signal and close)
            portfolio.iloc[i, portfolio.columns.get_loc('holdings')] = portfolio.iloc[i-1]['holdings']

            current_holdings = portfolio.iloc[i]['holdings']
            
            # Process signals based on current position
            if current_holdings == 0:  # No position
                if signal == 1:  # Open long
                    self._open_long_position(portfolio, i)
                    
            elif current_holdings > 0:  # Long position
                if signal == -1:  # Close position
                    self._close_position(portfolio, i)

        
        portfolio['Date'] = data['Date']
        return portfolio

class ShortOnlyTradingLogic(BaseTradingLogic):
    """
    Short Only trading logic implementation with the following rules:

    No Position:
    - Signal 1: No action
    - Signal -1: Open short position
    - Signal 0: No action
    
    Short Position:
    - Signal 1: Close position
    - Signal -1: No action
    - Signal 0: No action
    """
    def execute_trades(self, data: pd.DataFrame, signals: pd.DataFrame) -> pd.DataFrame:
        # Initialize portfolio
        portfolio = pd.DataFrame(index=data.index)
        portfolio['holdings'] = 0
        portfolio['signal'] = signals['signal']
        portfolio['close'] = data['close']

        # Simulate trading
        for i in range(len(data)):
            signal = signals.iloc[i]['signal']
            
            if i == 0:
                continue
                
            # Copy previous day's state (except signal and close)
            portfolio.iloc[i, portfolio.columns.get_loc('holdings')] = portfolio.iloc[i-1]['holdings']

            current_holdings = portfolio.iloc[i]['holdings']
            
            # Process signals based on current position
            if current_holdings == 0:  # No position
                if signal == -1:  # Open short
                    self._open_short_position(portfolio, i)
                    
            elif current_holdings < 0:    # Short position
                if signal == 1:  # Close position
                    self._close_position(portfolio, i)

        
        portfolio['Date'] = data['Date']
        return portfolio
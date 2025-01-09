from abc import ABC, abstractmethod
import pandas as pd
import numpy as np

class BaseTradingLogic(ABC):
    """
    Abstract base class for trading logic implementations.
    """
    def __init__(self, commission: float, slippage: float, trade_amount: float = 10000):
        self.commission = commission
        self.slippage = slippage
        self.trade_amount = trade_amount

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

    def _open_long_position(self, portfolio: pd.DataFrame, i: int, price: float):
        entry_price = price * (1 + self.slippage)
        shares = self.trade_amount / entry_price / (1 + self.commission)
        cost = shares * entry_price * (1 + self.commission)
        portfolio.iloc[i, portfolio.columns.get_loc('holdings')] = shares
        portfolio.iloc[i, portfolio.columns.get_loc('cost_basis')] = cost

    def _open_short_position(self, portfolio: pd.DataFrame, i: int, price: float):
        entry_price = price * (1 - self.slippage)
        shares = self.trade_amount / entry_price / (1 + self.commission)
        cost = shares * entry_price * (1 + self.commission)
        portfolio.iloc[i, portfolio.columns.get_loc('holdings')] = -shares
        portfolio.iloc[i, portfolio.columns.get_loc('cost_basis')] = cost

    def _close_long_position(self, portfolio: pd.DataFrame, i: int, price: float):
        current_holdings = portfolio.iloc[i]['holdings']
        close_price = price * (1 - self.slippage)
        profit = (current_holdings * close_price * (1 - self.commission) - 
                 portfolio.iloc[i]['cost_basis'])
        portfolio.iloc[i, portfolio.columns.get_loc('total')] += profit
        portfolio.iloc[i, portfolio.columns.get_loc('holdings')] = 0
        portfolio.iloc[i, portfolio.columns.get_loc('cost_basis')] = 0

    def _close_short_position(self, portfolio: pd.DataFrame, i: int, price: float):
        current_holdings = portfolio.iloc[i]['holdings']
        close_price = price * (1 + self.slippage)
        
        profit = portfolio.iloc[i]['cost_basis'] - (-(current_holdings) * close_price * (1 + self.commission))
        
        portfolio.iloc[i, portfolio.columns.get_loc('total')] += profit
        portfolio.iloc[i, portfolio.columns.get_loc('holdings')] = 0
        portfolio.iloc[i, portfolio.columns.get_loc('cost_basis')] = 0

    def _switch_to_short(self, portfolio: pd.DataFrame, i: int, price: float):
        self._close_long_position(portfolio, i, price)
        self._open_short_position(portfolio, i, price)

    def _switch_to_long(self, portfolio: pd.DataFrame, i: int, price: float):
        self._close_short_position(portfolio, i, price)
        self._open_long_position(portfolio, i, price) 

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
        portfolio['holdings'] = 0.0
        portfolio['total'] = 0.0
        portfolio['cost_basis'] = 0.0
        portfolio['signal'] = signals['signal']
        
        # Simulate trading
        for i in range(len(data)):
            current_price = data.iloc[i]['close']
            signal = signals.iloc[i]['signal']
            
            if i == 0:
                portfolio.iloc[i, portfolio.columns.get_loc('total')] = 0
                continue
                
            # Copy previous day's state (except signal)
            portfolio.iloc[i, portfolio.columns.get_loc('holdings')] = portfolio.iloc[i-1]['holdings']
            portfolio.iloc[i, portfolio.columns.get_loc('total')] = portfolio.iloc[i-1]['total']
            portfolio.iloc[i, portfolio.columns.get_loc('cost_basis')] = portfolio.iloc[i-1]['cost_basis']

            current_holdings = portfolio.iloc[i]['holdings']
            
            # Process signals based on current position
            if current_holdings == 0:  # No position
                if signal == 1:  # Open long
                    self._open_long_position(portfolio, i, current_price)
                elif signal == -1:  # Open short
                    self._open_short_position(portfolio, i, current_price)
                    
            elif current_holdings > 0:  # Long position
                if signal == -1:  # Switch to short
                    self._switch_to_short(portfolio, i, current_price)
                elif signal == 0:  # Close position
                    self._close_long_position(portfolio, i, current_price)
                    
            elif current_holdings < 0:  # Short position
                if signal == 1:  # Switch to long
                    self._switch_to_long(portfolio, i, current_price)
                elif signal == 0:  # Close position
                    self._close_short_position(portfolio, i, current_price)
        
        portfolio['Date'] = data['Date']
        return portfolio


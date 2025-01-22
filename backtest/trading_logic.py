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
        """使用向量化操作设置持仓"""
        portfolio.loc[i, 'holdings'] = 1

    def _open_short_position(self, portfolio: pd.DataFrame, i: int):
        portfolio.loc[i, 'holdings'] = -1

    def _close_position(self, portfolio: pd.DataFrame, i: int):
        portfolio.loc[i, 'holdings'] = 0

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
        """
        Execute trades based on signals and current market data.
        使用向量化操作优化性能
        """
        # 初始化 portfolio
        portfolio = pd.DataFrame(index=data.index)
        portfolio['holdings'] = 0
        portfolio['signal'] = signals['signal']
        portfolio['close'] = data['close']
        portfolio['Date'] = data['Date']
        
        # 获取所有信号和持仓的numpy数组，提高访问速度
        signal_array = signals['signal'].values
        holdings_array = portfolio['holdings'].values
        
        # 跳过第一天
        for i in range(1, len(data)):
            # 复制前一天的持仓
            holdings_array[i] = holdings_array[i-1]
            
            signal = signal_array[i]
            current_holdings = holdings_array[i]
            
            # 使用布尔运算替代多个if-else
            if current_holdings == 0:  # 无仓位
                holdings_array[i] = np.where(signal == 1, 1,
                                  np.where(signal == -1, -1, 0))
            elif current_holdings > 0:  # 多仓
                holdings_array[i] = np.where(signal == -1, -1,
                                  np.where(signal == 0, 0, 1))
            else:  # 空仓
                holdings_array[i] = np.where(signal == 1, 1,
                                  np.where(signal == 0, 0, -1))
        
        # 更新portfolio的holdings列
        portfolio['holdings'] = holdings_array
        
        return portfolio

class HoldTradingLogic(BaseTradingLogic):
    """
    Hold trading logic implementation.
    """
    def execute_trades(self, data: pd.DataFrame, signals: pd.DataFrame) -> pd.DataFrame:
        # Initialize portfolio
        portfolio = pd.DataFrame(index=data.index)
        portfolio['holdings'] = 0
        portfolio['signal'] = signals['signal']
        portfolio['close'] = data['close']
        portfolio['Date'] = data['Date']
        
        # 获取numpy数组以提高性能
        signal_array = signals['signal'].values
        holdings_array = portfolio['holdings'].values
        
        # 跳过第一天
        for i in range(1, len(data)):
            # 复制前一天的持仓
            holdings_array[i] = holdings_array[i-1]
            
            signal = signal_array[i]
            current_holdings = holdings_array[i]
            
            # 使用向量化操作处理信号
            if current_holdings == 0:  # 无仓位
                holdings_array[i] = np.where(signal == 1, 1,
                                  np.where(signal == -1, -1, 0))
            elif current_holdings > 0:  # 多仓
                holdings_array[i] = np.where(signal == -1, -1, 1)
            else:  # 空仓
                holdings_array[i] = np.where(signal == 1, 1, -1)
        
        # 更新portfolio
        portfolio['holdings'] = holdings_array
        return portfolio

class LongOnlyTradingLogic(BaseTradingLogic):
    """
    Long Only trading logic implementation.
    """
    def execute_trades(self, data: pd.DataFrame, signals: pd.DataFrame) -> pd.DataFrame:
        # Initialize portfolio
        portfolio = pd.DataFrame(index=data.index)
        portfolio['holdings'] = 0
        portfolio['signal'] = signals['signal']
        portfolio['close'] = data['close']
        portfolio['Date'] = data['Date']
        
        # 获取numpy数组以提高性能
        signal_array = signals['signal'].values
        holdings_array = portfolio['holdings'].values
        
        # 跳过第一天
        for i in range(1, len(data)):
            # 复制前一天的持仓
            holdings_array[i] = holdings_array[i-1]
            
            signal = signal_array[i]
            current_holdings = holdings_array[i]
            
            # 使用向量化操作处理信号
            if current_holdings == 0:  # 无仓位
                holdings_array[i] = np.where(signal == 1, 1, 0)
            elif current_holdings > 0:  # 多仓
                holdings_array[i] = np.where(signal == -1, 0, 1)
        
        # 更新portfolio
        portfolio['holdings'] = holdings_array
        return portfolio

class ShortOnlyTradingLogic(BaseTradingLogic):
    """
    Short Only trading logic implementation.
    """
    def execute_trades(self, data: pd.DataFrame, signals: pd.DataFrame) -> pd.DataFrame:
        # Initialize portfolio
        portfolio = pd.DataFrame(index=data.index)
        portfolio['holdings'] = 0
        portfolio['signal'] = signals['signal']
        portfolio['close'] = data['close']
        portfolio['Date'] = data['Date']
        
        # 获取numpy数组以提高性能
        signal_array = signals['signal'].values
        holdings_array = portfolio['holdings'].values
        
        # 跳过第一天
        for i in range(1, len(data)):
            # 复制前一天的持仓
            holdings_array[i] = holdings_array[i-1]
            
            signal = signal_array[i]
            current_holdings = holdings_array[i]
            
            # 使用向量化操作处理信号
            if current_holdings == 0:  # 无仓位
                holdings_array[i] = np.where(signal == -1, -1, 0)
            elif current_holdings < 0:  # 空仓
                holdings_array[i] = np.where(signal == 1, 0, -1)
        
        # 更新portfolio
        portfolio['holdings'] = holdings_array
        return portfolio
from abc import ABC, abstractmethod
# import time
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



class StandardTradingLogic(BaseTradingLogic):
    """标准交易逻辑：
    1. 多头信号时，如果当前无仓位则做多
    2. 空头信号时，如果当前无仓位则做空
    3. 持有多头时遇到空头信号，平多并做空
    4. 持有空头时遇到多头信号，平空并做多
    5. 中性信号时，平掉所有仓位
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
        holdings_array = np.zeros(len(data))  # 预分配数组
        
        # 跳过第一天，使用向量化操作
        for i in range(1, len(data)):
            signal = signal_array[i]

            
            holdings_array[i] = signal

        
        # 更新portfolio
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
            prev_holdings = holdings_array[i-1]
            
            signal = signal_array[i]
            
            # 使用向量化操作处理信号
            if signal == 1:  # 多头信号
                holdings_array[i] = 1
            elif signal == -1:  # 空头信号
                holdings_array[i] = -1
            elif signal == 0:  # 中性信号
                holdings_array[i] = prev_holdings
        
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
            prev_holdings = holdings_array[i-1]
            signal = signal_array[i]
            
            # 使用向量化操作处理信号
            if signal == 1:  # 多头信号
                holdings_array[i] = 1
            elif signal == -1:  # 空头信号
                holdings_array[i] = 0
            else:
                holdings_array[i] = prev_holdings

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
            prev_holdings = holdings_array[i-1]
            
            signal = signal_array[i]

            # 使用向量化操作处理信号
            if signal == -1:  # 空头信号
                holdings_array[i] = -1
            elif signal == 1:  # 多头信号
                holdings_array[i] = 0
            else:
                holdings_array[i] = prev_holdings
        
        # 更新portfolio
        portfolio['holdings'] = holdings_array
        return portfolio
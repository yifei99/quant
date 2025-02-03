# factors/factor_definitions.py

import pandas as pd
from abc import ABC, abstractmethod
import numpy as np

class BaseFactor(ABC):
    """
    Base Factor class that defines the basic interface for factors.
    """
    def __init__(self, name):
        self.name = name

    @abstractmethod
    def calculate(self, data: pd.DataFrame) -> pd.Series:
        """
        Calculate factor values.

        Args:
            data (pd.DataFrame): DataFrame containing price data and other relevant data.

        Returns:
            pd.Series: Factor calculation results.
        """
        pass

    def __repr__(self):
        return f"{self.__class__.__name__}(name='{self.name}')"
    
class TwoThresholdFactor(BaseFactor):
    """
    Generic threshold-based factor:
    Generates buy signals when value exceeds upper threshold;
    Generates sell signals when value falls below lower threshold.
    """
    def __init__(self, 
                 name: str,
                 column_name: str,
                 upper_threshold: float,
                 lower_threshold: float):
        """
        Args:
            name (str): Factor name
            column_name (str): Name of the column to monitor in data
            upper_threshold (float): Upper threshold for buy signals
            lower_threshold (float): Lower threshold for sell signals
        """
        super().__init__(name)
        self.column_name = column_name
        self.upper_threshold = upper_threshold
        self.lower_threshold = lower_threshold

    def calculate(self, data: pd.DataFrame) -> pd.Series:
        """
        Calculate threshold-based factor using vectorized operations
        """
        if self.column_name not in data.columns:
            raise ValueError(f"DataFrame must contain '{self.column_name}' column")
        
        # 使用numpy数组进行计算
        values = data[self.column_name].values
        
        # 使用numpy的where函数一次性生成所有信号
        signals = np.where(
            values > self.upper_threshold, 1,
            np.where(
                values < self.lower_threshold, -1,
                0
            )
        )
        
        # 返回带有正确索引的Series
        return pd.Series(signals, index=data.index, name=self.name)


class USDTIssuance2Factor(TwoThresholdFactor):
    """USDT Issuance threshold factor"""
    def __init__(self, 
                 name='usdt_issuance', 
                 upper_threshold=10000000,
                 lower_threshold=1000000):
        """
        Args:
            name (str): Factor name, defaults to 'usdt_issuance'
            upper_threshold (float): Upper threshold for buy signals, defaults to 10M
            lower_threshold (float): Lower threshold for sell signals, defaults to 1M
        """
        super().__init__(
            name=name,
            column_name='USDT_issuance',
            upper_threshold=upper_threshold,
            lower_threshold=lower_threshold
        )


class Liq2Factor(TwoThresholdFactor):
    """Liquidation threshold factor"""
    def __init__(self, 
                 name='liq',
                 upper_threshold=2,
                 lower_threshold=-2):
        """
        Args:
            name (str): Factor name, defaults to 'liq'
            upper_threshold (float): Upper threshold for buy signals, defaults to 2
            lower_threshold (float): Lower threshold for sell signals, defaults to -2
        """
        super().__init__(
            name=name,
            column_name='Liq',
            upper_threshold=upper_threshold,
            lower_threshold=lower_threshold
        )

class BaseMaFactor(BaseFactor):
    """Base class for Moving Average factors"""
    def __init__(self, name: str, column_name: str, ma_period: int = 7):
        super().__init__(name)
        if ma_period < 0:
            raise ValueError("ma_period must be non-negative")
        self.column_name = column_name
        self.ma_period = ma_period
        
    def calculate(self, data: pd.DataFrame) -> pd.Series:
        """
        Calculate MA signals using vectorized operations
        """
        if self.column_name not in data.columns:
            raise ValueError(f"DataFrame must contain '{self.column_name}' column")
            
        values = data[self.column_name].values
        signals = np.zeros(len(data))  # 初始化信号为0
        
        # 如果ma_period为0，直接使用原始值
        if self.ma_period == 0:
            ma = values
            valid_index = 0
        else:
            # 计算移动平均
            ma = np.convolve(values, np.ones(self.ma_period)/self.ma_period, mode='valid')
            valid_index = self.ma_period - 1  # MA开始有效的位置
            # 对齐长度
            ma = np.pad(ma, (valid_index, 0), mode='edge')
        
        # 生成信号
        signals[valid_index:][values[valid_index:] == ma[valid_index:]] = 0
        signals[valid_index:][values[valid_index:] > ma[valid_index:]] = 1
        signals[valid_index:][values[valid_index:] < ma[valid_index:]] = -1
        
        return pd.Series(signals, index=data.index, name=self.name)

class UsdVolumeMaFactor(BaseMaFactor):
    """USD Volume Moving Average Factor"""
    def __init__(self, name='usd_volume_ma', ma_period=7):
        super().__init__(
            name=name,
            column_name='quote_asset_volume',
            ma_period=ma_period
        )

class AssetVolumeMaFactor(BaseMaFactor):
    """Asset Volume Moving Average Factor"""
    def __init__(self, name='asset_volume_ma', ma_period=7):
        super().__init__(
            name=name,
            column_name='volume',
            ma_period=ma_period
        )

class PriceMaFactor(BaseMaFactor):
    """Price Moving Average Factor"""
    def __init__(self, name='price_ma', ma_period=7):
        super().__init__(
            name=name,
            column_name='close',
            ma_period=ma_period
        )

class Base2MaFactor(BaseFactor):
    """Base class for Dual Moving Average factors"""
    def __init__(self, name: str, column_name: str, ma_period_1: int = 7, ma_period_2: int = 14):
        super().__init__(name)
        if ma_period_1 < 0 or ma_period_2 < 0:
            raise ValueError("ma_period must be non-negative")
        self.column_name = column_name
        self.ma_period_1 = ma_period_1
        self.ma_period_2 = ma_period_2
        
    def calculate(self, data: pd.DataFrame) -> pd.Series:
        """
        Calculate Dual MA signals using optimized numpy operations
        """
        if self.column_name not in data.columns:
            raise ValueError(f"DataFrame must contain '{self.column_name}' column")
            
        values = data[self.column_name].values
        signals = np.zeros(len(data))  # 初始化信号为0
        
        # 计算第一个MA
        if self.ma_period_1 == 0:
            ma_1 = values
            valid_index_1 = 0
        else:
            ma_1 = np.convolve(values, np.ones(self.ma_period_1)/self.ma_period_1, mode='valid')
            valid_index_1 = self.ma_period_1 - 1
            ma_1 = np.pad(ma_1, (valid_index_1, 0), mode='edge')
            
        # 计算第二个MA
        if self.ma_period_2 == 0:
            ma_2 = values
            valid_index_2 = 0
        else:
            ma_2 = np.convolve(values, np.ones(self.ma_period_2)/self.ma_period_2, mode='valid')
            valid_index_2 = self.ma_period_2 - 1
            ma_2 = np.pad(ma_2, (valid_index_2, 0), mode='edge')
        
        # 使用较长的MA周期作为有效起始点
        valid_index = max(valid_index_1, valid_index_2)
        
        # 生成信号
        signals[valid_index:][ma_1[valid_index:] == ma_2[valid_index:]] = 0
        signals[valid_index:][ma_1[valid_index:] > ma_2[valid_index:]] = 1
        signals[valid_index:][ma_1[valid_index:] < ma_2[valid_index:]] = -1
        
        return pd.Series(signals, index=data.index, name=self.name)
    
class Price2MaFactor(Base2MaFactor):
    """Price Dual Moving Average Factor"""
    def __init__(self, name='price_2ma', ma_period_1=7, ma_period_2=14):
        super().__init__(
            name=name,
            column_name='close',
            ma_period_1=ma_period_1,
            ma_period_2=ma_period_2
        )

class Volume2MaFactor(Base2MaFactor):
    """Volume Dual Moving Average Factor"""
    def __init__(self, name='volume_2ma', ma_period_1=7, ma_period_2=14):
        super().__init__(
            name=name,
            column_name='volume',
            ma_period_1=ma_period_1,
            ma_period_2=ma_period_2
        )   

class UsdVolume2MaFactor(Base2MaFactor):
    """USD Volume Dual Moving Average Factor"""
    def __init__(self, name='usd_volume_2ma', ma_period_1=7, ma_period_2=14):
        super().__init__(
            name=name,
            column_name='quote_asset_volume',
            ma_period_1=ma_period_1,
            ma_period_2=ma_period_2
        )

class VolAdjMomentumFactor(TwoThresholdFactor):
    """波动率调整动量因子"""
    def __init__(self, 
                 name='vol_adj_momentum',
                 window: int = 20,
                 vol_window: int = None,
                 upper_threshold: float = 1.0,
                 lower_threshold: float = -1.0):
        """
        Args:
            name: 因子名称
            window: 价格变化窗口
            vol_window: 波动率计算窗口，默认等于price_window
            upper_threshold: 买入阈值
            lower_threshold: 卖出阈值
        """
        super().__init__(
            name=name,
            column_name='vol_adj_momentum',  # 这里可以是任意值，因为我们会重写calculate方法
            upper_threshold=upper_threshold,
            lower_threshold=lower_threshold
        )
        self.window = window
        self.vol_window = vol_window or window

    def calculate(self, data: pd.DataFrame) -> pd.Series:
        """计算因子值"""
        # 计算价格变化
        price_change = data['close'].diff(self.window)
        
        # 计算历史波动率 (使用对数收益率的标准差)
        log_returns = np.log(data['close'] / data['close'].shift(1))
        volatility = log_returns.rolling(window=self.vol_window).std() * np.sqrt(self.vol_window)
        
        # 计算波动率调整后的动量
        vol_adj_momentum = price_change / (volatility * data['close'])
        
        # 使用父类的阈值逻辑生成信号
        signals = np.where(
            vol_adj_momentum > self.upper_threshold, 1,
            np.where(
                vol_adj_momentum < self.lower_threshold, -1,
                0
            )
        )
        
        return pd.Series(signals, index=data.index, name=self.name)
        
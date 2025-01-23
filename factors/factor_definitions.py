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
        
        values = data[self.column_name].values
        signals = np.zeros(len(data))
        
        # 一次性计算所有信号
        signals[(values > self.upper_threshold)] = 1
        signals[(values < self.lower_threshold)] = -1
        
        return pd.Series(signals, index=data.index, name=self.name)


# 使用通用阈值因子类创建具体因子
class USDTIssuance2Factor(TwoThresholdFactor):
    def __init__(self, 
                 name='usdt_issuance', 
                 upper_threshold=10000000,
                 lower_threshold=1000000):
        super().__init__(
            name=name,
            column_name='USDT_issuance',
            upper_threshold=upper_threshold,
            lower_threshold=lower_threshold
        )


class Liq2Factor(TwoThresholdFactor):
    def __init__(self, 
                 name='liq',
                 upper_threshold=2,
                 lower_threshold=-2):
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
        self.column_name = column_name
        self.ma_period = ma_period
        
    def calculate(self, data: pd.DataFrame) -> pd.Series:
        """
        Calculate MA signals using vectorized operations
        """
        if self.column_name not in data.columns:
            raise ValueError(f"DataFrame must contain '{self.column_name}' column")
            
        # 使用numpy数组进行计算
        values = data[self.column_name].values
        ma = pd.Series(
            index=data.index,
            data=np.convolve(values, np.ones(self.ma_period)/self.ma_period, mode='valid')
        )
        ma = pd.Series(index=data.index, data=np.pad(ma, (self.ma_period-1, 0), mode='edge'))
        
        # 向量化比较操作
        signals = np.zeros(len(data))
        signals[values > ma] = 1
        signals[values < ma] = -1
        
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
        
        # 使用numpy的卷积运算计算移动平均
        ma_1 = np.convolve(values, np.ones(self.ma_period_1)/self.ma_period_1, mode='valid')
        ma_2 = np.convolve(values, np.ones(self.ma_period_2)/self.ma_period_2, mode='valid')
        
        # 处理开始的缺失值
        pad_1 = self.ma_period_1 - 1
        pad_2 = self.ma_period_2 - 1
        ma_1 = np.pad(ma_1, (pad_1, 0), mode='edge')
        ma_2 = np.pad(ma_2, (pad_2, 0), mode='edge')
        
        # 向量化信号生成
        signals = np.zeros(len(data))
        signals[ma_1 > ma_2] = 1
        signals[ma_1 < ma_2] = -1
        
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
        
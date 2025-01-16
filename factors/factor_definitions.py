# factors/factor_definitions.py

import pandas as pd
from abc import ABC, abstractmethod

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
        Calculate threshold-based factor.

        Args:
            data (pd.DataFrame): DataFrame containing the monitored column

        Returns:
            pd.Series: Factor signals, 1 for buy, -1 for sell, 0 for hold
        """
        if self.column_name not in data.columns:
            raise ValueError(f"DataFrame must contain '{self.column_name}' column")
        
        factor = pd.Series(0, index=data.index)
        factor[data[self.column_name] > self.upper_threshold] = 1
        factor[data[self.column_name] < self.lower_threshold] = -1

        return factor.rename(self.name)


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
    """通用MA因子基类"""
    def __init__(self, name: str, column_name: str, ma_period: int = 7):
        super().__init__(name)
        self.column_name = column_name
        self.ma_period = ma_period
        
    def calculate(self, data: pd.DataFrame) -> pd.Series:
        """
        计算MA信号
        1: 当值大于MA时
        -1: 当值小于MA时
        0: 其他情况
        """
        if self.column_name not in data.columns:
            raise ValueError(f"DataFrame must contain '{self.column_name}' column")
            
        # 计算移动平均
        ma = data[self.column_name].rolling(window=self.ma_period).mean()
        
        # 生成信号
        signals = pd.Series(0, index=data.index)
        signals[data[self.column_name] > ma] = 1
        signals[data[self.column_name] < ma] = -1
        
        return signals.rename(self.name)

class UsdVolumeMaFactor(BaseMaFactor):
    """USD交易量MA因子"""
    def __init__(self, name='usd_volume_ma', ma_period=7):
        super().__init__(
            name=name,
            column_name='quote_asset_volume',
            ma_period=ma_period
        )

class AssetVolumeMaFactor(BaseMaFactor):
    """资产交易量MA因子"""
    def __init__(self, name='asset_volume_ma', ma_period=7):
        super().__init__(
            name=name,
            column_name='volume',
            ma_period=ma_period
        )

class PriceMaFactor(BaseMaFactor):
    """价格MA因子"""
    def __init__(self, name='price_ma', ma_period=7):
        super().__init__(
            name=name,
            column_name='close',
            ma_period=ma_period
        )
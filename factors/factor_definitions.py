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
    """Base class for Moving Average factors"""
    def __init__(self, name: str, column_name: str, ma_period: int = 7):
        super().__init__(name)
        self.column_name = column_name
        self.ma_period = ma_period
        
    def calculate(self, data: pd.DataFrame) -> pd.Series:
        """
        Calculate MA signals
        1: When value is above MA
        -1: When value is below MA
        0: Otherwise
        
        Args:
            data (pd.DataFrame): Input data containing the column to monitor

        Returns:
            pd.Series: Factor signals
        """
        if self.column_name not in data.columns:
            raise ValueError(f"DataFrame must contain '{self.column_name}' column")
            
        # Calculate moving average
        ma = data[self.column_name].rolling(window=self.ma_period).mean()
        
        # Generate signals
        signals = pd.Series(0, index=data.index)
        signals[data[self.column_name] > ma] = 1
        signals[data[self.column_name] < ma] = -1
        
        return signals.rename(self.name)

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
        Calculate Dual MA signals
        1: When shorter MA crosses above longer MA
        -1: When shorter MA crosses below longer MA
        0: Otherwise
        
        Args:
            data (pd.DataFrame): Input data containing the column to monitor

        Returns:
            pd.Series: Factor signals
        """
        if self.column_name not in data.columns:
            raise ValueError(f"DataFrame must contain '{self.column_name}' column")
            
        # Calculate moving averages
        ma_1 = data[self.column_name].rolling(window=self.ma_period_1).mean()
        ma_2 = data[self.column_name].rolling(window=self.ma_period_2).mean()
        
        # Generate signals
        signals = pd.Series(0, index=data.index)
        signals[ma_1 > ma_2] = 1
        signals[ma_1 < ma_2] = -1
        
        return signals.rename(self.name)
    
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
        
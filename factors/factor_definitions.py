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
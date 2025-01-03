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
    
class USDTIssuanceFactor(BaseFactor):
    """
    USDT Issuance Factor:
    Generates buy signals when USDT issuance exceeds upper threshold;
    Generates sell signals when USDT issuance falls below lower threshold.
    """
    def __init__(self, name='usdt_issuance', threshold=1000000):
        """
        Args:
            name (str): Factor name.
            threshold (float): Signal trigger threshold.
        """
        super().__init__(name)
        self.threshold = threshold

    def calculate(self, data: pd.DataFrame) -> pd.Series:
        """
        Calculate USDT issuance factor.

        Args:
            data (pd.DataFrame): DataFrame containing 'USDT_issuance' column.

        Returns:
            pd.Series: Factor signals, 1 for buy, -1 for sell, 0 for hold.
        """
        if 'USDT_issuance' not in data.columns:
            raise ValueError("DataFrame must contain 'USDT_issuance' column.")
        
        factor = pd.Series(0, index=data.index)
        factor[data['USDT_issuance'] > self.threshold] = 1
        factor[data['USDT_issuance'] < self.threshold] = -1

        return factor.rename(self.name)
    
class USDTIssuance2Factor(BaseFactor):
    """
    USDT Issuance Factor:
    Generates buy signals when USDT issuance exceeds upper threshold;
    Generates sell signals when USDT issuance falls below lower threshold.
    """
    def __init__(self, 
                 name='usdt_issuance', 
                 upper_threshold=10000000,  # Upper threshold
                 lower_threshold=1000000):  # Lower threshold
        """
        Args:
            name (str): Factor name
            upper_threshold (float): Upper threshold for buy signals
            lower_threshold (float): Lower threshold for sell signals
        """
        super().__init__(name)
        self.upper_threshold = upper_threshold
        self.lower_threshold = lower_threshold

    def calculate(self, data: pd.DataFrame) -> pd.Series:
        """
        Calculate USDT issuance factor.

        Args:
            data (pd.DataFrame): DataFrame containing 'USDT_issuance' column

        Returns:
            pd.Series: Factor signals, 1 for buy, -1 for sell, 0 for hold
        """
        if 'USDT_issuance' not in data.columns:
            raise ValueError("DataFrame must contain 'USDT_issuance' column")
        
        factor = pd.Series(0, index=data.index)
        
        factor[data['USDT_issuance'] > self.upper_threshold] = 1
        factor[data['USDT_issuance'] < self.lower_threshold] = -1

        return factor.rename(self.name)
# factors/factor_definitions.py

import pandas as pd
from abc import ABC, abstractmethod

class BaseFactor(ABC):
    """
    因子基类，定义因子的基本接口。
    """
    def __init__(self, name):
        self.name = name

    @abstractmethod
    def calculate(self, data: pd.DataFrame) -> pd.Series:
        """
        计算因子的值。

        参数:
            data (pd.DataFrame): 包含价格数据和其他相关数据的DataFrame。

        返回:
            pd.Series: 因子的计算结果。
        """
        pass

    def __repr__(self):
        return f"{self.__class__.__name__}(name='{self.name}')"
    
    # factors/factor_definitions.py

class USDTIssuanceFactor(BaseFactor):
    """
    USDT发行量因子:
    当USDT发行量大于某个阈值时,生成买入信号；
    当USDT发行量小于某个阈值时,生成卖出信号。
    """
    def __init__(self, name='usdt_issuance', direction=1, threshold=1000000):
        """
        参数:
            name (str): 因子名称。
            direction (int): 因子方向，1 表示正向因子，-1 表示逆向因子。
            threshold (float): 信号触发阈值。
        """
        super().__init__(name)
        self.direction = direction  # 添加方向属性
        self.threshold = threshold

    def calculate(self, data: pd.DataFrame) -> pd.Series:
        """
        计算USDT发行量因子。

        参数:
            data (pd.DataFrame): 包含'USDT_issuance'列的DataFrame。

        返回:
            pd.Series: 因子信号，当USDT发行量大于阈值时为1，小于时为-1，否则为0。
        """
        if 'USDT_issuance' not in data.columns:
            raise ValueError("DataFrame必须包含'USDT_issuance'列。")
        
        factor = pd.Series(0, index=data.index)
        factor[data['USDT_issuance'] > self.threshold] = 1
        factor[data['USDT_issuance'] < self.threshold] = -1

        # 乘以方向属性，确保因子方向一致
        return (factor * self.direction).rename(self.name)
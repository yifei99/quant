from typing import List
import pandas as pd
from abc import ABC, abstractmethod
from factors.factor_definitions import BaseFactor

class BaseStrategy(ABC):
    """
    策略基类，定义策略的基本接口。
    """
    def __init__(self, name='BaseStrategy'):
        self.name = name
    
    @abstractmethod
    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        生成交易信号。

        参数:
            data (pd.DataFrame): 包含价格数据和因子数据的DataFrame。

        返回:
            pd.DataFrame: 包含信号的DataFrame
            signal: 1.0（应该持有多头）、-1.0（应该持有空头）、0.0（应该空仓）
        """
        pass

    def __repr__(self):
        return f"{self.__class__.__name__}(name='{self.name}')"

class FactorBasedStrategy(BaseStrategy):
    """
    基于因子的交易策略：
    根据因子信号生成买入/卖出信号。
    """
    def __init__(self, factors: List[BaseFactor], name='Factor_Based_Strategy', normalize_factors=False):
        """
        参数:
            factors (List[BaseFactor]): 因子列表，每个因子对象包含名称和方向。
            name (str): 策略名称。
            normalize_factors (bool): 是否对因子进行归一化处理。
        """
        super().__init__(name)
        self.factors = factors
        self.normalize_factors = normalize_factors

    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        生成交易信号。

        参数:
            data (pd.DataFrame): 包含价格数据和因子数据的DataFrame。

        返回:
            pd.DataFrame: 包含信号的DataFrame
            signal: 1.0（应该持有多头）、-1.0（应该持有空头）、0.0（应该空仓）
        """
        signals = pd.DataFrame(index=data.index)
        signals['signal'] = 0.0

        # 合并因子信号
        for factor in self.factors:
            if factor.name not in data.columns:
                raise ValueError(f"DataFrame中缺少因子'{factor.name}'。")

            factor_signal = data[factor.name]
            signals['signal'] += factor_signal

        # 根据因子值生成信号
        signals['signal'] = signals['signal'].apply(
            lambda x: 1 if x > 0 else (-1 if x < 0 else 0)
        )
        # print(signals.head(10))

        return signals
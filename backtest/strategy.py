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
            pd.DataFrame: 包含交易信号的DataFrame。
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
        生成交易信号，基于多个因子的综合信号。

        参数:
            data (pd.DataFrame): 包含价格数据和因子数据的DataFrame。

        返回:
            pd.DataFrame: 包含信号的DataFrame，信号列为1（买入）、-1（卖出）、0（持有）。
        """
        signals = pd.DataFrame(index=data.index)
        signals['signal'] = 0.0



        # 合并因子信号
        for factor in self.factors:
            if factor.name not in data.columns:
                raise ValueError(f"DataFrame中缺少因子'{factor.name}'。")

            factor_signal = data[factor.name]

            # # 可选：对因子信号进行归一化
            # if self.normalize_factors:
            #     factor_signal = (factor_signal - factor_signal.mean()) / factor_signal.std()

            # 考虑因子的方向性
            signals['signal'] += factor_signal

        # # 调试信息：输出综合信号的统计数据
        # print("综合信号统计数据:")
        # print(signals['signal'].describe())

        # 根据综合信号生成买卖信号
        signals['signal'] = signals['signal'].apply(lambda x: 1.0 if x > 0 else (-1.0 if x < 0 else 0.0))

        # 生成交易订单
        signals['positions'] = signals['signal'].diff()

        # # 调试信息：输出信号和交易位置的分布
        # print(self.factors)
        # print("信号分布:")
        # print(signals['signal'].value_counts())
        # print("交易位置分布:")
        # print(signals['positions'].value_counts())

        return signals
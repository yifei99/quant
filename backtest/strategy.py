from typing import List
import pandas as pd
from abc import ABC, abstractmethod
from factors.factor_definitions import BaseFactor
import numpy as np

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
        生成交易信号。使用向量化操作提升性能。

        参数:
            data (pd.DataFrame): 包含价格数据和因子数据的DataFrame。

        返回:
            pd.DataFrame: 包含信号的DataFrame
            signal: 1.0（应该持有多头）、-1.0（应该持有空头）、0.0（应该空仓）
        """
        # 预分配numpy数组，避免DataFrame频繁操作
        signal_array = np.zeros(len(data))
        
        # 检查所有因子是否存在
        missing_factors = [f.name for f in self.factors if f.name not in data.columns]
        if missing_factors:
            raise ValueError(f"DataFrame中缺少因子: {', '.join(missing_factors)}")
        
        # 使用numpy数组进行因子合并
        factor_arrays = [data[factor.name].values for factor in self.factors]
        if self.normalize_factors and factor_arrays:
            # 归一化处理（如果需要）
            factor_arrays = [
                (arr - np.mean(arr)) / np.std(arr)
                for arr in factor_arrays
            ]
        
        # 高效的数组操作
        if factor_arrays:
            signal_array = np.sum(factor_arrays, axis=0)
            # 使用numpy.where进行向量化判断
            signal_array = np.where(signal_array > 0, 1.0,
                                  np.where(signal_array < 0, -1.0, 0.0))
        
        # 只在最后一次转换为DataFrame
        signals = pd.DataFrame(index=data.index)
        signals['signal'] = signal_array
        
        return signals
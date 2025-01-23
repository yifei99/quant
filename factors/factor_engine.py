# factors/factor_engine.py

import pandas as pd
from typing import List
from .factor_definitions import BaseFactor

class FactorEngine:
    """
    因子计算引擎，管理因子的注册和计算。
    """
    def __init__(self):
        self.factors = {}
        self.factor_values = {}
        self.signals = {}

    def register_factor(self, factor: BaseFactor):
        """
        注册一个因子。

        参数:
            factor (BaseFactor): 要注册的因子实例。
        """
        self.factors[factor.name] = factor

    def calculate_factors(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        计算所有注册的因子。

        参数:
            data (pd.DataFrame): 包含价格数据和其他相关数据的DataFrame。

        返回:
            pd.DataFrame: 包含所有因子值的DataFrame。
        """
        factor_data = pd.DataFrame(index=data.index)
        for factor in self.factors.values():
            factor_series = factor.calculate(data)
            factor_data = pd.concat([factor_data, factor_series], axis=1)
        return factor_data

    def list_factors(self) -> List[str]:
        """
        列出所有注册的因子名称。

        返回:
            List[str]: 因子名称列表。
        """
        return list(self.factors.keys())

    def reset(self):
        """
        重置因子引擎状态，清空所有缓存的因子值和信号
        但保留已注册的因子定义
        """
        # 清空因子计算结果
        self.factor_values = {}
        # 清空信号
        self.signals = {}
        # 清空因子列表
        self.factors = {}
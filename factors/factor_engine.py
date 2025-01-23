# factors/factor_engine.py

import pandas as pd
import numpy as np
from typing import List, Dict
from .factor_definitions import BaseFactor

class FactorEngine:
    """
    因子计算引擎，管理因子的注册和计算。
    """
    def __init__(self):
        self.factors: Dict[str, BaseFactor] = {}
        self.factor_values: Dict[str, np.ndarray] = {}
        self.signals: Dict[str, np.ndarray] = {}
        self._data_length: int = 0
        self._data_index = None

    def register_factor(self, factor: BaseFactor):
        """
        注册一个因子。

        参数:
            factor (BaseFactor): 要注册的因子实例。
        """
        self.factors[factor.name] = factor

    def calculate_factors(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        计算所有注册的因子。使用预分配和向量化操作优化性能。

        参数:
            data (pd.DataFrame): 包含价格数据和其他相关数据的DataFrame。

        返回:
            pd.DataFrame: 包含所有因子值的DataFrame。
        """
        self._data_length = len(data)
        self._data_index = data.index
        
        # 预分配结果数组
        factor_arrays = np.empty((self._data_length, len(self.factors)))
        factor_names = []
        
        # 计算所有因子
        for i, (name, factor) in enumerate(self.factors.items()):
            factor_series = factor.calculate(data)
            factor_arrays[:, i] = factor_series.values
            factor_names.append(name)
            # 缓存因子值
            self.factor_values[name] = factor_arrays[:, i]
        
        # 一次性创建DataFrame
        return pd.DataFrame(
            data=factor_arrays,
            index=self._data_index,
            columns=factor_names
        )

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
        self.factor_values.clear()
        # 清空信号
        self.signals.clear()
        # 清空因子列表
        self.factors.clear()
        self._data_length = 0
        self._data_index = None

    def get_factor_value(self, factor_name: str) -> np.ndarray:
        """
        获取因子值，直接返回numpy数组以提高性能。

        参数:
            factor_name (str): 因子名称

        返回:
            np.ndarray: 因子值数组
        """
        return self.factor_values.get(factor_name)

    def get_all_factor_values(self) -> Dict[str, np.ndarray]:
        """
        获取所有因子值。

        返回:
            Dict[str, np.ndarray]: 因子名称到因子值数组的映射
        """
        return self.factor_values
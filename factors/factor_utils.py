import pandas as pd
import numpy as np

def standardize(factor_series: pd.Series) -> pd.Series:
    """
    标准化因子，使其均值为0，标准差为1。

    参数:
        factor_series (pd.Series): 要标准化的因子系列。

    返回:
        pd.Series: 标准化后的因子系列。
    """
    mean = factor_series.mean()
    std = factor_series.std()
    standardized = (factor_series - mean) / std
    return standardized

def winsorize(factor_series: pd.Series, limits: float = 3.0) -> pd.Series:
    """
    去极值处理，将因子值限制在指定的标准差范围内。

    参数:
        factor_series (pd.Series): 要去极值的因子系列。
        limits (float): 标准差限制，默认3.0。

    返回:
        pd.Series: 去极值后的因子系列。
    """
    mean = factor_series.mean()
    std = factor_series.std()
    lower_bound = mean - limits * std
    upper_bound = mean + limits * std
    winsorized = factor_series.clip(lower=lower_bound, upper=upper_bound)
    return winsorized

def normalize(factor_series: pd.Series) -> pd.Series:
    """
    归一化因子，使其值在0和1之间。

    参数:
        factor_series (pd.Series): 要归一化的因子系列。

    返回:
        pd.Series: 归一化后的因子系列。
    """
    min_val = factor_series.min()
    max_val = factor_series.max()
    normalized = (factor_series - min_val) / (max_val - min_val)
    return normalized
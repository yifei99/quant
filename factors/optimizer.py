# factors/optimizer.py

import pandas as pd
import numpy as np
import logging
from backtest.backtest_engine import BacktestEngine
from backtest.strategy import BaseStrategy
from backtest.performance import PerformanceEvaluator
from factors.factor_definitions import BaseFactor
from factors.factor_engine import FactorEngine

class StrategyOptimizer:
    """
    策略优化器，寻找使绩效指标最优的因子参数。
    """
    def __init__(self, engine: BacktestEngine, evaluator: PerformanceEvaluator):
        self.engine = engine
        self.evaluator = evaluator
        self.logger = logging.getLogger(__name__)

    def optimize_threshold(self, data: pd.DataFrame, 
                           threshold_values: list, 
                           factor_class: BaseFactor, 
                           strategy_class: BaseStrategy) -> dict:
        """
        优化因子阈值，寻找使夏普比率最优的阈值。

        参数:
            data (pd.DataFrame): 包含价格数据和因子数据的DataFrame。
            threshold_values (list): 要测试的阈值列表。
            factor_class (BaseFactor): 因子类。
            strategy_class (BaseStrategy): 策略类。

        返回:
            dict: 包含每个阈值对应的夏普比率的字典。
        """
        results = {}
        for threshold in threshold_values:
            self.logger.info(f"Testing threshold: {threshold}")
            # 定义因子
            factor = factor_class(name='usdt_issuance', threshold=threshold)
            factor_engine = FactorEngine()
            factor_engine.register_factor(factor)

            # 定义策略
            strategy = strategy_class(factors=[factor])

            # 运行回测
            portfolio = self.engine.run_backtest(data, strategy, factor_engine=factor_engine)

            # 评估绩效
            metrics = self.evaluator.calculate_performance_metrics(portfolio)
            sharpe_ratio = metrics.get('Sharpe Ratio', 0)
            results[threshold] = sharpe_ratio
            self.logger.info(f"Threshold: {threshold}, Sharpe Ratio: {sharpe_ratio:.4f}")

        return results

    def find_optimal_threshold(self, results: dict) -> tuple:
        """
        从优化结果中找到夏普比率最高的阈值。

        参数:
            results (dict): 每个阈值对应的夏普比率的字典。

        返回:
            tuple: 最优阈值和对应的夏普比率。
        """
        optimal_threshold = max(results, key=results.get)
        optimal_sharpe = results[optimal_threshold]
        return optimal_threshold, optimal_sharpe
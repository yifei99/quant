# factors/optimizer.py

import pandas as pd
import numpy as np
import logging
from itertools import product
from backtest.backtest_engine import BacktestEngine
from backtest.strategy import BaseStrategy
from backtest.performance import PerformanceEvaluator
from factors.factor_definitions import BaseFactor
from factors.factor_engine import FactorEngine
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, Any

class StrategyOptimizer:
    def __init__(self, engine: BacktestEngine, evaluator: PerformanceEvaluator):
        self.engine = engine
        self.evaluator = evaluator
        self.logger = logging.getLogger(__name__)

    def _test_combination(self, 
                         combination: tuple, 
                         param_names: list, 
                         data: pd.DataFrame,
                         factor_class: BaseFactor,
                         strategy_class: BaseStrategy) -> Dict[str, Any]:
        """Helper method to test a single parameter combination"""
        current_params = dict(zip(param_names, combination))
        
        # 定义因子
        factor = factor_class(name='usdt_issuance', **current_params)
        factor_engine = FactorEngine()
        factor_engine.register_factor(factor)

        # 定义策略
        strategy = strategy_class(factors=[factor])

        # 运行回测
        portfolio = self.engine.run_backtest(data, strategy, factor_engine=factor_engine)

        # 评估绩效
        metrics = self.evaluator.calculate_performance_metrics(portfolio)
        sharpe_ratio = metrics.get('Sharpe Ratio', 0)
        
        return {
            'combination': combination,
            'params': current_params,
            'sharpe_ratio': sharpe_ratio,
            'metrics': metrics
        }

    def optimize_thresholds(self, data: pd.DataFrame,
                          threshold_params: dict,
                          factor_class: BaseFactor,
                          strategy_class: BaseStrategy,
                          max_workers: int = None) -> dict:
        """
        优化多个因子阈值，使用多线程并行测试不同阈值组合。

        参数:
            data (pd.DataFrame): 包含价格数据和因子数据的DataFrame
            threshold_params (dict): 要测试的阈值参数字典，格式为:
                {
                    'threshold1': [value1, value2, ...],
                    'threshold2': [value1, value2, ...],
                    ...
                }
            factor_class (BaseFactor): 因子类
            strategy_class (BaseStrategy): 策略类
            max_workers (int, optional): 最大线程数，默认为None（由系统决定）

        返回:
            dict: 包含每个有效阈值组合对应的夏普比率的字典
        """
        param_names = list(threshold_params.keys())
        param_values = list(threshold_params.values())
        param_combinations = list(product(*param_values))
        
        valid_combinations = [
            combo for combo in param_combinations 
            if dict(zip(param_names, combo)).get('lower_threshold', -float('inf')) <
               dict(zip(param_names, combo)).get('upper_threshold', float('inf'))
        ]
        
        results = {}
        total_combinations = len(valid_combinations)
        self.logger.info(f"Testing {total_combinations} valid threshold combinations using threading")
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_combo = {
                executor.submit(
                    self._test_combination, 
                    combination, 
                    param_names, 
                    data,
                    factor_class,
                    strategy_class
                ): combination 
                for combination in valid_combinations
            }
            
            for i, future in enumerate(as_completed(future_to_combo), 1):
                try:
                    result = future.result()
                    combination = result['combination']
                    results[combination] = {
                        'params': result['params'],
                        'sharpe_ratio': result['sharpe_ratio'],
                        'metrics': result['metrics']
                    }
                    
                    self.logger.info(
                        f"Completed combination {i}/{total_combinations}: "
                        f"{result['params']}, Sharpe Ratio: {result['sharpe_ratio']:.4f}"
                    )
                except Exception as e:
                    self.logger.error(f"Error processing combination: {e}")

        return results

    def find_optimal_thresholds(self, results: dict) -> tuple:
        """
        从优化结果中找到夏普比率最高的阈值组合。

        参数:
            results (dict): 每个阈值组合对应的结果字典

        返回:
            tuple: (最优参数字典, 最优夏普比率, 完整性能指标)
        """
        optimal_combination = max(results.items(), key=lambda x: x[1]['sharpe_ratio'])
        optimal_params = optimal_combination[1]['params']
        optimal_sharpe = optimal_combination[1]['sharpe_ratio']
        optimal_metrics = optimal_combination[1]['metrics']
        
        return optimal_params, optimal_sharpe, optimal_metrics
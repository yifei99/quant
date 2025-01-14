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
from joblib import Parallel, delayed

class StrategyOptimizer:
    def __init__(self, engine: BacktestEngine, evaluator: PerformanceEvaluator):
        self.engine = engine
        self.evaluator = evaluator
        self.logger = logging.getLogger(__name__)


    def optimize_thresholds(self, data: pd.DataFrame,
                          threshold_params: dict,
                          factor_class: BaseFactor,
                          strategy_class: BaseStrategy,
                          n_jobs: int = -1) -> dict:
        """
        优化多个因子阈值，使用joblib并行测试不同�值组合。
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
        total_start_time = pd.Timestamp.now()
        self.logger.info(f"Testing {total_combinations} valid threshold combinations using joblib")
        
        # 分批处理
        batch_size = 100
        for i in range(0, len(valid_combinations), batch_size):
            batch = valid_combinations[i:i + batch_size]
            
            # 使用joblib处理当前批次
            processed_results = Parallel(n_jobs=n_jobs, verbose=0)(
                delayed(self._test_combination)(
                    combination, param_names, data, factor_class, strategy_class
                ) for combination in batch
            )
            
            # 实时处理和打印每个批次的结果
            for combination, result in zip(batch, processed_results):
                if result is not None:
                    results[combination] = result
                    params = dict(zip(param_names, combination))
                    total_elapsed = (pd.Timestamp.now() - total_start_time).total_seconds()
                    self.logger.info(
                        f"Progress: {len(results)}/{total_combinations} | "
                        f"Time: {total_elapsed:.2f}s | "
                        f"Params: {params}, "
                        f"Sharpe: {result['sharpe_ratio']:.4f}, "
                        f"Return: {result['metrics']['Total Return']:.4f}"
                    )
        
        total_time = (pd.Timestamp.now() - total_start_time).total_seconds()
        self.logger.info(f"Optimization completed in {total_time:.2f}s")
        return results

    def _test_combination(self, combination, param_names, data, factor_class, strategy_class):
        """
        测试单个参数组合
        
        Args:
            combination: 参数组合
            param_names: 参数名列表
            data: 回测数据
            factor_class: 因子类
            strategy_class: 策略类
        """
        start_time = pd.Timestamp.now()
        try:
            params = dict(zip(param_names, combination))
            # 直接使用类名作为因子名称
            factor = factor_class(name=factor_class.__name__, **params)
            factor_engine = FactorEngine()
            factor_engine.register_factor(factor)
            strategy = strategy_class(factors=[factor])
            
            portfolio = self.engine.run_backtest(
                data=data,
                strategy=strategy,
                factor_engine=factor_engine,
                plot=False
            )
            
            metrics = self.evaluator.calculate_performance_metrics(portfolio)
            
            elapsed_time = (pd.Timestamp.now() - start_time).total_seconds()
            
            return {
                'params': params,
                'sharpe_ratio': metrics['Sharpe Ratio'],
                'metrics': metrics,
                'elapsed_time': elapsed_time
            }
        except Exception as e:
            elapsed_time = (pd.Timestamp.now() - start_time).total_seconds()
            self.logger.error(f"Error testing combination {params} after {elapsed_time:.2f}s: {e}")
            return None

    def find_optimal_thresholds(self, results: dict, data: pd.DataFrame, 
                              factor_class: BaseFactor, strategy_class: BaseStrategy) -> tuple:
        """
        从优化结果中找到夏普比率最高的阈值组合。

        参数:
            results (dict): 每个阈值组合对应的结果字典
            data (pd.DataFrame): 用于最终回测的数据
            factor_class (BaseFactor): 因子类
            strategy_class (BaseStrategy): 策略类

        返回:
            tuple: (最优参数字典, 最优夏普比率, 完整性能指标, 最优参数的回测结果)
        """
        optimal_combination = max(results.items(), key=lambda x: x[1]['sharpe_ratio'])
        optimal_params = optimal_combination[1]['params']
        optimal_sharpe = optimal_combination[1]['sharpe_ratio']
        optimal_metrics = optimal_combination[1]['metrics']
        
        # 使用最优参数运行一次回测并生成图表
        factor = factor_class(name='usdt_issuance', **optimal_params)
        factor_engine = FactorEngine()
        factor_engine.register_factor(factor)
        strategy = strategy_class(factors=[factor])
        
        # 最终回测生成图表
        portfolio_optimal = self.engine.run_backtest(
            data=data, 
            strategy=strategy, 
            factor_engine=factor_engine,
            plot=True  # 为最优参数生成图表
        )
        
        return optimal_params, optimal_sharpe, optimal_metrics, portfolio_optimal
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
from joblib import Parallel, delayed, parallel_backend
import os
from tqdm import tqdm
import psutil
import gc
import time

class StrategyOptimizer:
    def __init__(self, engine: BacktestEngine, evaluator: PerformanceEvaluator):
        self.engine = engine
        self.evaluator = evaluator
        self.logger = logging.getLogger(__name__)
        # 自动设置最优进程数
        self.n_jobs = min(psutil.cpu_count(), 32)  # 限制最大进程数
        # 优化批处理大小
        self.batch_size = max(10, self.n_jobs * 10)  # 根据CPU核心数调整批大小

    def optimize_thresholds(self, data: pd.DataFrame,
                          threshold_params: dict,
                          factor_class: BaseFactor,
                          strategy_class: BaseStrategy,
                          n_jobs: int = None,
                          enforce_threshold_order: bool = True) -> dict:
        """
        优化多个因子阈值，使用joblib并行测试不同组合。
        """
        # 使用传入的n_jobs或默认值
        n_jobs = n_jobs if n_jobs is not None else self.n_jobs
        
        # 预先生成参数组合
        param_names = list(threshold_params.keys())
        param_values = list(threshold_params.values())
        
        # 如果window在参数中，为每个window值生成对应的fer_trend_upper范围
        if 'window' in threshold_params and 'fer_trend_upper' in threshold_params:
            windows = threshold_params['window']
            param_combinations = []
            for window in windows:
                # 动态生成fer_trend_upper的范围
                fer_trend_upper = np.arange(
                    max(1.0, window/24),  # 最小值
                    min(30.0, window/8),  # 最大值
                    max(0.1, window/240)  # 步长
                )
                # 获取其他参数的值
                other_params = {k: v for k, v in threshold_params.items() 
                              if k not in ['window', 'fer_trend_upper']}
                other_values = list(product(*[v for v in other_params.values()]))
                
                # 组合当前window的所有参数
                for trend_upper in fer_trend_upper:
                    for other_combo in other_values:
                        param_combinations.append((window, trend_upper) + other_combo)
        else:
            # 原有的参数组合生成方式
            param_combinations = list(product(*param_values))
        
        if enforce_threshold_order:
            valid_combinations = [
                combo for combo in param_combinations 
                if combo[0] <= combo[1]
            ]
            del param_combinations
        else:
            valid_combinations = param_combinations
        
        results = {}
        best_sharpe = float('-inf')
        total_combinations = len(valid_combinations)
        total_start_time = pd.Timestamp.now()
        self.logger.info(f"Testing {total_combinations} parameter combinations using {n_jobs} processes")
        
        # 为每个进程创建一个因子引擎实例
        factor_engines = [FactorEngine() for _ in range(n_jobs if n_jobs > 0 else self.n_jobs)]
        
        try:
            # 使用上下文管理器优化并行计算
            with parallel_backend('loky', n_jobs=n_jobs):
                with tqdm(total=total_combinations, desc="Optimizing parameters") as pbar:
                    for i in range(0, total_combinations, self.batch_size):
                        batch = valid_combinations[i:min(i + self.batch_size, total_combinations)]
                        
                        # 批量处理参数组合
                        processed_results = Parallel()(
                            delayed(self._test_combination)(
                                combination, 
                                param_names, 
                                data,  # 直接使用传入的数据
                                factor_class, 
                                strategy_class,
                                factor_engines[i % len(factor_engines)]
                            ) for combination in batch
                        )
                        
                        # 高效处理批量结果
                        valid_results = [(combo, result) for combo, result in zip(batch, processed_results) if result is not None]
                        if valid_results:
                            for combo, result in valid_results:
                                results[combo] = result
                                best_sharpe = max(best_sharpe, result['sharpe_ratio'])
                            
                            pbar.update(len(batch))
                            pbar.set_postfix({
                                'Best Sharpe': f'{best_sharpe:.4f}'
                            })
                        
                        # 定期清理内存
                        if i % (self.batch_size * 5) == 0:
                            gc.collect()
                            
                        # 清理批次结果
                        del processed_results
                        del valid_results
        
        finally:
            # 清理资源
            for engine in factor_engines:
                engine.reset()
            del factor_engines
            gc.collect()
        
        total_time = (pd.Timestamp.now() - total_start_time).total_seconds()
        self.logger.info(f"Optimization completed in {total_time:.2f}s")
        return results

    def _test_combination(self, combination, param_names, data, factor_class, strategy_class, factor_engine):
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
            # 创建参数字典
            params = dict(zip(param_names, combination))
            
            # 创建因子实例
            factor = factor_class(name=factor_class.__name__, **params)
            
            # 重置并重用因子引擎
            factor_engine.reset()
            factor_engine.register_factor(factor)
            
            # 创建策略实例
            strategy = strategy_class(factors=[factor])
            
            # 运行回测
            portfolio = self.engine.run_backtest(
                data=data,
                strategy=strategy,
                factor_engine=factor_engine,
                plot=False
            )
            
            # 计算指标
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
                              factor_class: BaseFactor, strategy_class: BaseStrategy,
                              save_dir: str, min_sharpe: float = 1.0) -> tuple:
        """
        从优化结果中找到夏普比率最高的阈值组合。

        参数:
            results (dict): 每个阈值组合对应的结果字典
            data (pd.DataFrame): 用于最终回测的数据
            factor_class (BaseFactor): 因子类
            strategy_class (BaseStrategy): 策略类
            save_dir (str): 结果保存目录
            min_sharpe (float): 最小夏普比率阈值，低于此值不生成图表，默认为1.0

        返回:
            tuple: (最优参数字典, 最优夏普比率, 完整性能指标, 最优参数的回测结果)
        """
        optimal_combination = max(results.items(), key=lambda x: x[1]['sharpe_ratio'])
        optimal_params = optimal_combination[1]['params']
        optimal_sharpe = optimal_combination[1]['sharpe_ratio']
        optimal_metrics = optimal_combination[1]['metrics']
        
        # 使用最优参数运行一次回测
        factor = factor_class(name=factor_class.__name__, **optimal_params)
        factor_engine = FactorEngine()
        factor_engine.register_factor(factor)
        strategy = strategy_class(factors=[factor])
        
        # 确保保存目录存在
        os.makedirs(save_dir, exist_ok=True)
        
        # 只有当夏普比率超过阈值时才生成图表
        should_plot = optimal_sharpe >= min_sharpe
        
        # 最终回测
        portfolio_optimal = self.engine.run_backtest(
            data=data, 
            strategy=strategy, 
            factor_engine=factor_engine,
            plot=should_plot,  # 根据夏普比率决定是否绘图
            save_dir=save_dir if should_plot else None  # 只在需要绘图时传入保存目录
        )
        
        # 记录是否生成了图表
        if not should_plot:
            self.logger.info(f"Skipped plot generation due to low Sharpe ratio: {optimal_sharpe:.2f} < {min_sharpe}")
        
        return optimal_params, optimal_sharpe, optimal_metrics, portfolio_optimal
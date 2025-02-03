import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import logging
from backtest.backtest_engine import BacktestEngine
from backtest.strategy import FactorBasedStrategy
from backtest.performance import PerformanceEvaluator
from factors.factor_definitions import VolAdjMomentumFactor
from factors.factor_engine import FactorEngine
from factors.optimizer import StrategyOptimizer
from data.data_loader import DataLoader
from backtest.trading_logic import HoldTradingLogic, LongOnlyTradingLogic, ShortOnlyTradingLogic, StandardTradingLogic
import matplotlib.pyplot as plt

def get_trading_logic(logic_type: int):
    """获取交易逻辑实例"""
    if logic_type == 1:
        return HoldTradingLogic()
    elif logic_type == 2:
        return LongOnlyTradingLogic()
    elif logic_type == 3:
        return ShortOnlyTradingLogic()
    elif logic_type == 4:
        return StandardTradingLogic()
    else:
        raise ValueError("Invalid logic_type. Must be 1 (Long Short), 2 (Long Only), 3 (Short Only), or 4 (Standard)")

# def analyze_and_plot_factor(data: pd.DataFrame, window: int) -> dict:
#     """分析并可视化因子值"""
#     # 计算因子值
#     price_change = data['close'].diff(window)
#     log_returns = np.log(data['close'] / data['close'].shift(1))
#     volatility = log_returns.rolling(window=window).std() * np.sqrt(window)
#     vol_adj_momentum = price_change / (volatility * data['close'])
    
#     # 基本统计信息
#     stats = {
#         'mean': vol_adj_momentum.mean(),
#         'std': vol_adj_momentum.std(),
#         'min': vol_adj_momentum.min(),
#         'max': vol_adj_momentum.max(),
#         'null_count': vol_adj_momentum.isnull().sum(),
#         'percentiles': {
#             '1%': vol_adj_momentum.quantile(0.01),
#             '5%': vol_adj_momentum.quantile(0.05),
#             '10%': vol_adj_momentum.quantile(0.10),
#             '25%': vol_adj_momentum.quantile(0.25),
#             '50%': vol_adj_momentum.quantile(0.50),
#             '75%': vol_adj_momentum.quantile(0.75),
#             '90%': vol_adj_momentum.quantile(0.90),
#             '95%': vol_adj_momentum.quantile(0.95),
#             '99%': vol_adj_momentum.quantile(0.99)
#         }
#     }
    
#     # 打印统计信息
#     print(f"\n因子统计信息 (window={window}):")
#     print(f"数据点数: {len(vol_adj_momentum)}")
#     print(f"空值数量: {stats['null_count']}")
#     print(f"均值: {stats['mean']:.4f}")
#     print(f"标准差: {stats['std']:.4f}")
#     print(f"最小值: {stats['min']:.4f}")
#     print(f"最大值: {stats['max']:.4f}")
#     print("\n分位数:")
#     for pct, value in stats['percentiles'].items():
#         print(f"{pct}: {value:.4f}")
    
#     # 绘制分布图
#     plt.figure(figsize=(15, 5))
    
#     # 时间序列图
#     plt.subplot(121)
#     plt.plot(vol_adj_momentum.index, vol_adj_momentum.values)
#     plt.title(f'Vol-Adj Momentum (window={window})')
#     plt.xlabel('Date')
#     plt.ylabel('Factor Value')
#     plt.grid(True)
    
#     # 直方图
#     plt.subplot(122)
#     plt.hist(vol_adj_momentum.dropna(), bins=50, density=True)
#     plt.title('Factor Distribution')
#     plt.xlabel('Factor Value')
#     plt.ylabel('Density')
#     plt.grid(True)
    
#     plt.tight_layout()
#     plt.show()
    
#     return stats

def main():
    # 设置日志
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    logger = logging.getLogger(__name__)
    
    logger.info("Starting VolAdjMomentum factor mining...")
    start_time = pd.Timestamp.now()
    
    # 根据分布分析结果设置合理的参数范围
    test_configs = {
        'datasets': [
            {
                'exchange': 'binance',
                'symbol': 'BTCUSDT',
                'interval': '1h',
                'start_date': '2021-01-01',
                'end_date': '2024-12-31',
                'data_type': 'spot'
            },
            {
                'exchange': 'binance',
                'symbol': 'SOLUSDT',
                'interval': '1h',
                'start_date': '2021-01-01',
                'end_date': '2024-12-31',
                'data_type': 'spot'
            }
        ],
        'trading_logics': [
            # {'type': 1, 'name': 'long_short'},
            {'type': 4, 'name': 'standard'}
        ],
        'factors': [
            {
                'class': VolAdjMomentumFactor,
                'name': 'vol_adj_momentum',
                'params': {
                    'window': np.arange(12, 480, 12),  
                    'upper_threshold': np.arange(0.0, 4.0, 0.1),  
                    'lower_threshold': np.arange(-4.0, -1.0, 0.1)  
                }
            }
        ]
    }
    
    # 初始化数据加载器
    data_loader = DataLoader("../dataset")
    
    # # 在运行优化之前，先分析因子
    # for dataset_config in test_configs['datasets']:
    #     try:
    #         logger.info(f"\nAnalyzing factor for {dataset_config['symbol']}")
            
    #         # 加载数据
    #         data = data_loader.load_data(**dataset_config)
            
    #         # 处理时间戳
    #         if data['timestamp_start'].max() > 2e10:
    #             data['Date'] = pd.to_datetime(data['timestamp_start'], unit='ms')
    #         else:
    #             data['Date'] = pd.to_datetime(data['timestamp_start'], unit='s')
            
    #         required_columns = ['Date', 'close', 'volume', 'quote_asset_volume']
    #         data = data[required_columns].copy()
            
    #         # 分析不同窗口的因子值
    #         windows = [24, 48, 96, 168]  # 1天、2天、4天、7天
    #         for window in windows:
    #             stats = analyze_and_plot_factor(data, window)
                
    #             # 等待用户确认
    #             user_input = input("\n按Enter继续分析下一个窗口，输入'q'退出分析: ")
    #             if user_input.lower() == 'q':
    #                 break
            
    #         # 根据分析结果设置参数范围
    #         user_input = input("\n是否继续优化过程？(y/n): ")
    #         if user_input.lower() != 'y':
    #             logger.info("用户终止优化过程")
    #             return
                
    #     except Exception as e:
    #         logger.error(f"Error analyzing factor: {e}")
    #         logger.error("Full error:", exc_info=True)
    #         continue
    
    # 遍历所有组合
    for dataset_config in test_configs['datasets']:
        for logic_config in test_configs['trading_logics']:
            for factor_config in test_configs['factors']:
                try:
                    logger.info(f"\nTesting combination:")
                    logger.info(f"Dataset: {dataset_config['symbol']}")
                    logger.info(f"Trading Logic: {logic_config['name']}")
                    logger.info(f"Factor: {factor_config['name']}")
                    
                    # 加载数据
                    data = data_loader.load_data(**dataset_config)
                    
                    # 处理时间戳
                    if data['timestamp_start'].max() > 2e10:
                        data['Date'] = pd.to_datetime(data['timestamp_start'], unit='ms')
                    else:
                        data['Date'] = pd.to_datetime(data['timestamp_start'], unit='s')
                    
                    # 选择必要的列
                    required_columns = ['Date', 'close', 'volume', 'quote_asset_volume']
                    data = data[required_columns].copy()
                    
                    # 初始化组件
                    logic = get_trading_logic(logic_type=logic_config['type'])
                    engine = BacktestEngine(trading_logic=logic, periods_per_year=365*24)
                    evaluator = PerformanceEvaluator(periods_per_year=365*24)
                    optimizer = StrategyOptimizer(engine=engine, evaluator=evaluator)
                    
                    # 运行优化
                    optimization_results = optimizer.optimize_thresholds(
                        data=data,
                        threshold_params=factor_config['params'],
                        factor_class=factor_config['class'],
                        strategy_class=FactorBasedStrategy,
                        enforce_threshold_order=False
                    )
                    
                    # 创建结果目录
                    result_dir = os.path.join(
                        "../reports",
                        dataset_config['symbol'],
                        factor_config['name'],
                        logic_config['name'],
                        dataset_config['exchange'],
                        dataset_config['data_type'],
                        dataset_config['interval'],
                        f"{dataset_config['start_date']}_to_{dataset_config['end_date']}"
                    )
                    os.makedirs(result_dir, exist_ok=True)
                    
                    # 获取最优结果
                    optimal_params, optimal_sharpe, optimal_metrics, optimized_portfolio = (
                        optimizer.find_optimal_thresholds(
                            results=optimization_results,
                            data=data,
                            factor_class=factor_config['class'],
                            strategy_class=FactorBasedStrategy,
                            save_dir=result_dir
                        )
                    )
                    
                    # 保存优化结果
                    results_df = pd.DataFrame([
                        {
                            'window': combo[0],
                            'upper_threshold': combo[1],
                            'lower_threshold': combo[2],
                            'sharpe_ratio': result['sharpe_ratio']
                        }
                        for combo, result in optimization_results.items()
                    ])
                    results_df.to_csv(os.path.join(result_dir, 'optimization_results.csv'), index=False)
                    
                    # 保存回测结果
                    optimized_portfolio.to_csv(os.path.join(result_dir, 'backtest_results.csv'))
                    
                    # 保存性能指标
                    with open(os.path.join(result_dir, 'performance_metrics.txt'), 'w') as f:
                        f.write(f"Optimal Parameters:\n")
                        f.write(f"window: {optimal_params['window']}\n")
                        f.write(f"upper_threshold: {optimal_params['upper_threshold']:.2f}\n")
                        f.write(f"lower_threshold: {optimal_params['lower_threshold']:.2f}\n\n")
                        f.write("Performance Metrics:\n")
                        for key, value in optimal_metrics.items():
                            if 'Return' in key or 'Drawdown' in key:
                                f.write(f"{key}: {value * 100:.2f}%\n")
                            else:
                                f.write(f"{key}: {value:.4f}\n")
                    
                except Exception as e:
                    logger.error(f"Error processing combination: {e}")
                    logger.error("Full error:", exc_info=True)
                    continue
    
    end_time = pd.Timestamp.now()
    duration = end_time - start_time
    logger.info(f"\nTotal runtime: {duration}")
    logger.info("Program completed successfully")

if __name__ == "__main__":
    main()

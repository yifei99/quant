import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import logging
from backtest.backtest_engine import BacktestEngine
from backtest.strategy import FactorBasedStrategy
from backtest.performance import PerformanceEvaluator
from factors.factor_definitions import LiquidityZoneReversionFactor
from factors.factor_engine import FactorEngine
from factors.optimizer import StrategyOptimizer
from data.data_loader import DataLoader
from backtest.trading_logic import HoldTradingLogic, LongOnlyTradingLogic, ShortOnlyTradingLogic, StandardTradingLogic

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

def main():
    # 设置日志
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    logger = logging.getLogger(__name__)
    
    logger.info("Starting LiquidityZoneReversion factor mining...")
    start_time = pd.Timestamp.now()
    
    # 设置测试配置
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
                'class': LiquidityZoneReversionFactor,
                'name': 'liquidity_zone',
                'params': {
                    'volume_window': np.arange(12, 480, 12),        # 10到50，步长5
                    'upper_threshold': np.arange(0.0, 6.7, 0.2),  # 上阈值范围
                    'lower_threshold': np.arange(-5.0, 0.0, 0.2),  # 下阈值范围（注意是负值）
                    'volume_quantile': np.arange(0.7, 0.95, 0.05)
                }
            }
        ]
    }
    
    # 初始化数据加载器
    data_loader = DataLoader("../dataset")
    
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
                    required_columns = ['Date', 'close', 'high', 'low', 'volume']
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
                            'volume_window': combo[0],
                            'deviation_threshold': combo[1],
                            'volume_quantile': combo[2],
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
                        f.write(f"volume_window: {optimal_params['volume_window']}\n")
                        f.write(f"upper_threshold: {optimal_params['upper_threshold']:.2f}\n")
                        f.write(f"lower_threshold: {optimal_params['lower_threshold']:.2f}\n")
                        f.write(f"volume_quantile: {optimal_params['volume_quantile']:.2f}\n\n")
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

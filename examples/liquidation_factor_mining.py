import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import logging
from backtest.backtest_engine import BacktestEngine
from backtest.strategy import FactorBasedStrategy
from backtest.performance import PerformanceEvaluator
from factors.factor_definitions import LiquidationRatioFactor
from factors.factor_engine import FactorEngine
from factors.optimizer import StrategyOptimizer
from data.data_loader import DataLoader
from backtest.trading_logic import HoldTradingLogic, StandardTradingLogic,LongOnlyTradingLogic,ShortOnlyTradingLogic

def main():
    # 设置日志
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    logger = logging.getLogger(__name__)
    
    logger.info("Starting Liquidation Ratio factor mining...")
    start_time = pd.Timestamp.now()
    
    # 测试配置
    test_configs = {
        'datasets': [
            {
                'exchange': 'binance',
                'symbol': 'ETHUSDT',
                'interval': '1h',
                'start_date': '2020-12-16',
                'end_date': '2025-02-14',
                'data_type': 'spot'
            }
        ],
        'trading_logics': [
            {'type': 'long_only', 'name': 'long_only'}
        ],
        'factors': [
            {
                'class': LiquidationRatioFactor,
                'name': 'liquidation_ratio',
                'params': {
                    'threshold': np.arange(0.001, 0.059, 0.001).tolist(),
                    'hold_hours': range(1, 721)
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
                    
                    # 加载价格数据
                    price_data = data_loader.load_data(**dataset_config)
                    
                    # 统一时间戳格式到毫秒级并创建日期列
                    price_data['timestamp_start'] = price_data['timestamp_start'].apply(
                        lambda x: x // 1000 if x > 2e12 else x
                    )
                    price_data['Date'] = pd.to_datetime(price_data['timestamp_start'], unit='ms')
                    
                    # 读取清算数据
                    liquidation_file = os.path.join(
                        "../dataset/liquidation/aave_hourly_weth_liquidations.csv"
                    )
                    liquidation_data = pd.read_csv(liquidation_file)
                    liquidation_data['Date'] = pd.to_datetime(liquidation_data['hour'])
                    
                    # 读取清算金额数据
                    amounts_file = os.path.join(
                        "../dataset/liquidation/aave_hourly_weth_amounts.csv"
                    )
                    amounts_data = pd.read_csv(amounts_file)
                    amounts_data['Date'] = pd.to_datetime(amounts_data['hour']).dt.tz_localize(None)
                    amounts_data['Date'] = amounts_data['Date'] + pd.Timedelta(hours=1)
                    
                    # 合并所有数据
                    data = pd.merge(price_data, liquidation_data, on='Date', how='left')
                    data = pd.merge(
                        data,
                        amounts_data[['Date', 'cumulative_amount_usd']],
                        on='Date',
                        how='left'
                    )
                    
                    # 填充缺失值并计算清算比例
                    data = data.fillna(0)
                    data['liquidation_ratio'] = data['collateral_amount_usd'] / data['cumulative_amount_usd']
                    data['liquidation_ratio'] = data['liquidation_ratio'].fillna(0)
                    data['liquidation_ratio'] = data['liquidation_ratio'].replace([np.inf, -np.inf], 0)
                    
                    # 初始化组件
                    logic = LongOnlyTradingLogic()
                    engine = BacktestEngine(trading_logic=logic, periods_per_year=365*24)
                    evaluator = PerformanceEvaluator(periods_per_year=365*24)
                    optimizer = StrategyOptimizer(engine=engine, evaluator=evaluator)
                    
                    # 运行优化
                    optimization_results = optimizer.optimize_thresholds(
                        data=data,
                        threshold_params=factor_config['params'],
                        factor_class=factor_config['class'],
                        strategy_class=FactorBasedStrategy
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
                            'threshold': combo[0],
                            'hold_hours': combo[1],
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
                        f.write(f"threshold: {optimal_params['threshold']:.3f}\n")
                        f.write(f"hold_hours: {optimal_params['hold_hours']}\n\n")
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
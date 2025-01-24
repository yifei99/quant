import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from matplotlib import pyplot as plt
import pandas as pd
import logging
from joblib import Parallel, delayed
from backtest.backtest_engine import BacktestEngine
from backtest.strategy import FactorBasedStrategy
from backtest.performance import PerformanceEvaluator
from factors.factor_definitions import AssetVolumeMaFactor, PriceMaFactor, UsdVolumeMaFactor
from factors.factor_engine import FactorEngine
from factors.optimizer import StrategyOptimizer
from data.data_loader import DataLoader
from backtest.trading_logic import HoldTradingLogic,LongOnlyTradingLogic,ShortOnlyTradingLogic


def get_trading_logic(logic_type: int):
    """
    获取交易逻辑实例
    
    Args:
        logic_type (int): 交易逻辑类型
            1: Hold (多空)
            2: Long Only (仅做多)
            3: Short Only (仅做空)
    
    Returns:
        BaseTradingLogic: 交易逻辑实例
    """
    if logic_type == 1:
        return HoldTradingLogic()
    elif logic_type == 2:
        return LongOnlyTradingLogic()
    elif logic_type == 3:
        return ShortOnlyTradingLogic()
    else:
        raise ValueError("Invalid logic_type. Must be 1 (Long Short), 2 (Long Only), or 3 (Short Only)")

def main():
    """
    Main function for factor mining process:
    1. Load market data (OHLCV)
    2. Load and process factor data
    3. Initialize factor and backtest components
    4. Run backtest (with optional optimization)
    5. Save and analyze results
    """
    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    logger = logging.getLogger(__name__)

    logger.info("Starting new optimization run...")
    start_time = pd.Timestamp.now()
    logger.info(f"Start time: {start_time}")

    # 定义要测试的配置
    test_configs = {
        'datasets': [
            {
                'exchange': 'binance',
                'symbol': 'ETHUSDT',
                'interval': '1d',
                'start_date': '2021-01-01',
                'end_date': '2024-12-31',
                'data_type': 'spot'
            },
            {
                'exchange': 'binance',
                'symbol': 'ETHUSDT',
                'interval': '1h',
                'start_date': '2021-01-01',
                'end_date': '2024-12-31',
                'data_type': 'spot'
            },
            {
                'exchange': 'binance',
                'symbol': 'ETHUSDT',
                'interval': '4h',
                'start_date': '2021-01-01',
                'end_date': '2024-12-31',
                'data_type': 'spot'
            },
            {
                'exchange': 'binance',
                'symbol': 'ETHUSDT',
                'interval': '8h',
                'start_date': '2021-01-01',
                'end_date': '2024-12-31',
                'data_type': 'spot'
            },
            {
                'exchange': 'binance',
                'symbol': 'ETHUSDT',
                'interval': '6h',
                'start_date': '2021-01-01',
                'end_date': '2024-12-31',
                'data_type': 'spot'
            },
            {
                'exchange': 'binance',
                'symbol': 'BTCUSDT',
                'interval': '1d',
                'start_date': '2021-01-01',
                'end_date': '2024-12-31',
                'data_type': 'spot'
            },
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
                'symbol': 'BTCUSDT',
                'interval': '4h',
                'start_date': '2021-01-01',
                'end_date': '2024-12-31',
                'data_type': 'spot'
            },
            {
                'exchange': 'binance',
                'symbol': 'BTCUSDT',
                'interval': '8h',
                'start_date': '2021-01-01',
                'end_date': '2024-12-31',
                'data_type': 'spot'
            },
            {
                'exchange': 'binance',
                'symbol': 'BTCUSDT',
                'interval': '6h',
                'start_date': '2021-01-01',
                'end_date': '2024-12-31',
                'data_type': 'spot'
            },
            {
                'exchange': 'binance',
                'symbol': 'SOLUSDT',
                'interval': '1d',
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
            },
            {
                'exchange': 'binance',
                'symbol': 'SOLUSDT',
                'interval': '4h',
                'start_date': '2021-01-01',
                'end_date': '2024-12-31',
                'data_type': 'spot'
            },
            {
                'exchange': 'binance',
                'symbol': 'SOLUSDT',
                'interval': '8h',
                'start_date': '2021-01-01',
                'end_date': '2024-12-31',
                'data_type': 'spot'
            },
            {
                'exchange': 'binance',
                'symbol': 'SOLUSDT',
                'interval': '6h',
                'start_date': '2021-01-01',
                'end_date': '2024-12-31',
                'data_type': 'spot'
            }
            # 可以添加更多数据集配置
        ],
        'trading_logics': [
            {'type': 1, 'name': 'long_short'},
            {'type': 2, 'name': 'long_only'},
            {'type': 3, 'name': 'short_only'}
        ],
        'factors': [
            {
                'class': PriceMaFactor, 
                'name': 'price_ma',
                'params': {
                    'ma_period': range(3, 360, 1)
                }
            },
            {
                'class': AssetVolumeMaFactor, 
                'name': 'volume_ma',
                'params': {
                    'ma_period': range(3, 360, 1)
                }
            },
            {
                'class': UsdVolumeMaFactor, 
                'name': 'usd_volume_ma',
                'params': {
                    'ma_period': range(3, 360, 1)
                }
            }
        ]
    }

    # 初始化 DataLoader
    data_loader = DataLoader("../dataset")

    # 遍历所有组合
    for dataset_config in test_configs['datasets']:
        for logic_config in test_configs['trading_logics']:
            for factor_config in test_configs['factors']:
                try:
                    logger.info(f"Starting new test combination:")
                    logger.info(f"Dataset: {dataset_config['symbol']}")
                    logger.info(f"Trading Logic: {logic_config['name']}")
                    logger.info(f"Factor: {factor_config['name']}")
                    
                    # 加载数据
                    data = data_loader.load_data(**dataset_config)
                    
                    # 处理时间戳和数据选择的优化
                    # 确保时间戳是以毫秒为单位
                    if data['timestamp_start'].max() > 2e10:  # 如果时间戳是以毫秒为单位
                        data['Date'] = pd.to_datetime(data['timestamp_start'], unit='ms')
                    else:  # 如果时间戳是以秒为单位
                        data['Date'] = pd.to_datetime(data['timestamp_start'], unit='s')

                    # 使用numpy视图而不是复制
                    required_columns = ['Date', 'close', 'volume', 'quote_asset_volume']
                    data = data[required_columns].copy(deep=False)
                    
                    # 初始化组件
                    factor_engine = FactorEngine()
                    logic = get_trading_logic(logic_type=logic_config['type'])
                    engine = BacktestEngine(trading_logic=logic)
                    evaluator = PerformanceEvaluator()
                    
                    # 运行优化
                    optimizer = StrategyOptimizer(engine=engine, evaluator=evaluator)
                    threshold_params = factor_config['params']  # 直接使用配置中的参数
                    
                    # 设置n_jobs参数，可以根据需要调整
                    optimization_results = optimizer.optimize_thresholds(
                        data=data,
                        threshold_params=threshold_params,
                        factor_class=factor_config['class'],
                        strategy_class=FactorBasedStrategy,
                        enforce_threshold_order=False  # 单均线不需要强制顺序
                    )
                    
                    # 创建更详细的结果目录结构
                    result_dir = os.path.join(
                        "../reports",
                        dataset_config['symbol'],            # 第1级：交易对
                        factor_config['name'],               # 第2级：因子名称
                        logic_config['name'],                # 第3级：交易逻辑
                        dataset_config['exchange'],          # 第4级：交易所
                        dataset_config['data_type'],         # 第5级：数据类型 (spot/futures)
                        dataset_config['interval'],          # 第6级：时间周期
                        f"{dataset_config['start_date']}_to_{dataset_config['end_date']}"  # 第7级：时间范围
                    )
                    os.makedirs(result_dir, exist_ok=True)
                    
                    # 获取最优结果
                    optimal_params, optimal_sharpe, optimal_metrics, optimized_portfolio = (
                        optimizer.find_optimal_thresholds(
                            results=optimization_results,
                            data=data,
                            factor_class=factor_config['class'],
                            strategy_class=FactorBasedStrategy,
                            save_dir=result_dir  # 传入保存目录
                        )
                    )
                    
                    # 1. 保存优化结果
                    results_df = pd.DataFrame([
                        {
                            'ma_period': combo[0],
                            'sharpe_ratio': result['sharpe_ratio']
                        }
                        for combo, result in optimization_results.items()
                    ])
                    results_df.to_csv(os.path.join(result_dir, 'optimization_results.csv'), index=False)
                    
                    # 2. 保存回测结果
                    optimized_portfolio.to_csv(os.path.join(result_dir, 'backtest_results.csv'))
                    
                    # 3. 保存性能指标
                    with open(os.path.join(result_dir, 'performance_metrics.txt'), 'w') as f:
                        f.write(f"Optimal Parameters:\n")
                        f.write(f"ma_period: {optimal_params['ma_period']}\n\n")
                        f.write("Performance Metrics:\n")
                        for key, value in optimal_metrics.items():
                            if 'Return' in key or 'Drawdown' in key:
                                f.write(f"{key}: {value * 100:.2f}%\n")
                            else:
                                f.write(f"{key}: {value:.4f}\n")
                    
                    logger.info(f"All results saved to {result_dir}")
                    
                except Exception as e:
                    logger.error(f"Error processing combination: {e}")
                    logger.error("Full error:", exc_info=True)
                    continue  # 继续下一个组合
                    
    logger.info("All combinations completed")

    end_time = pd.Timestamp.now()
    duration = end_time - start_time
    logger.info(f"Total runtime: {duration}")
    logger.info("Program completed successfully")


if __name__ == "__main__":
    main()
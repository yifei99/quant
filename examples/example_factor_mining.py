import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import pandas as pd
import logging
from backtest.backtest_engine import BacktestEngine
from backtest.strategy import FactorBasedStrategy
from backtest.performance import PerformanceEvaluator
from factors.factor_definitions import USDTIssuance2Factor
from factors.factor_engine import FactorEngine
from factors.optimizer import StrategyOptimizer
from data.data_loader import DataLoader
import multiprocessing


# 设置显示所有行和列
pd.set_option('display.max_rows', None)  # 显示所有行
pd.set_option('display.max_columns', None)  # 显示所有列
pd.set_option('display.width', None)  # 显示所有宽度
pd.set_option('display.max_colwidth', None)  # 显示所有列宽


def load_usdt_issuance_data(usdt_file_path):
    """
    加载 USDT 发行量数据，计算每日发行量。
    """
    # 加载数据
    usdt_data = pd.read_csv(usdt_file_path)
    
    # 确保时间戳转换为日期时间格式，并按日期排序
    usdt_data['Date'] = pd.to_datetime(usdt_data['Timestamp'], unit='s')  # 修改为时间戳列名
    usdt_data = usdt_data.sort_values('Date')

    # 计算每日 USDT 发行量
    usdt_data['USDT_issuance'] = usdt_data['USDT'].diff().fillna(0)
    
    return usdt_data[['Date', 'USDT_issuance']]

def main():
    # 设置日志
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    logger = logging.getLogger(__name__)

    # 初始化 DataLoader 实例
    data_folder = "../dataset"  # 根目录，不包含 exchange
    data_loader = DataLoader(data_folder)

    # 加载数据
    exchange = "binance"
    symbol = "BTCUSDT"
    interval = "1d"
    start_date = "2021-03-31"
    end_date = "2024-12-05"
    data_type = "spot"

    try:
        data = data_loader.load_data(
            exchange, symbol, interval, start_date, end_date, data_type
        )
        if isinstance(data, list):
            logger.info("Multiple matching files found:")
            for idx, file in enumerate(data, 1):
                logger.info(f"{idx}. {file}")
            choice = int(input("Enter the number of the file you want to load: "))
            selected_file = data[choice - 1]
            data = data_loader.load_data_from_path(selected_file)
        logger.info("Data loaded successfully.")
        # print(data.head())
    except FileNotFoundError as e:
        logger.error(e)
        return

    # 加载 USDT 发行量数据并计算每日发行量
    usdt_file_path = "../dataset/stables/stablecoins.csv"  # 替换为你的 USDT 数据文件路径
    try:
        usdt_data = load_usdt_issuance_data(usdt_file_path)
        logger.info("USDT issuance data loaded successfully.")
    except Exception as e:
        logger.error(f"Failed to load USDT issuance data: {e}")
        return

    # 合并 USDT 发行量数据到主数据框
    try:
        data['Date'] = pd.to_datetime(data['timestamp_start'], unit='ms')  # 修改为 'timestamp_start'
        data = pd.merge(data, usdt_data, on='Date', how='left')
        logger.info("USDT issuance data merged successfully.")
        # print(data.head())
    except Exception as e:
        logger.error(f"Failed to merge USDT issuance data: {e}")
        return
    logger.info(f"USDT Issuance Statistics: {data['USDT_issuance'].describe()}")
    # 初始化因子引擎并注册因子
    factor_engine = FactorEngine()
    usdt_factor = USDTIssuance2Factor(name='usdt_issuance', upper_threshold=1000000000, lower_threshold=-1000000000)
    factor_engine.register_factor(usdt_factor)
    logger.info(f"Generated factor: {usdt_factor.calculate(data).head()}")

    # 初始化策略，基于因子的策略
    strategy = FactorBasedStrategy(factors=[usdt_factor])


    # 初始化回测引擎和绩效评估器
    engine = BacktestEngine(initial_capital=100000.0, commission=0.001)
    evaluator = PerformanceEvaluator()

    # 执行初始回测
    logger.info("Starting initial backtest...")
    try:
        portfolio = engine.run_backtest(data, strategy, factor_engine=factor_engine)
        logger.info("Initial backtest completed.")
        # print(portfolio.head(2000))
    except Exception as e:
        logger.error(f"Backtest failed: {e}")
        return

    # 计算绩效指标
    try:
        # 确保 portfolio 的索引或日期列是 datetime 类型
        if 'Date' in portfolio.columns:
            portfolio['Date'] = pd.to_datetime(portfolio['Date'])
        else:
            logger.error("Portfolio does not contain a 'Date' column.")
            return

        metrics = evaluator.calculate_performance_metrics(portfolio)
        print("\nInitial Performance Metrics:")
        for key, value in metrics.items():
            if 'Return' in key or 'Drawdown' in key:
                print(f"{key}: {value * 100:.2f}%")
            else:
                print(f"{key}: {value:.2f}")
    except Exception as e:
        logger.error(f"Performance evaluation failed: {e}")

    # 初始化優化器
    optimizer = StrategyOptimizer(engine=engine, evaluator=evaluator)

    # 定義要測試的閾值範圍
    threshold_params = {
        'upper_threshold': list(range(-3200000000, 1500000000, 50000000)),
        'lower_threshold': list(range(-3200000000, 1500000000, 50000000))
    }

    # 執行優化
    logger.info("Starting optimization...")

    # Get number of CPU cores and subtract 1 for optimal worker count
    max_workers = max(1, multiprocessing.cpu_count() - 1)
    logger.info(f"Using {max_workers} workers for optimization")

    optimization_results = optimizer.optimize_thresholds(
        data=data,
        threshold_params=threshold_params,
        factor_class=USDTIssuance2Factor,
        strategy_class=FactorBasedStrategy,
        max_workers=max_workers
    )
    logger.info("Optimization completed")

    # 找到最優閾值組合
    optimal_params, optimal_sharpe, optimal_metrics = optimizer.find_optimal_thresholds(optimization_results)
    logger.info(f"Optimal Parameters: {optimal_params}")
    logger.info(f"Optimal Sharpe Ratio: {optimal_sharpe:.4f}")

    # 保存优化结果
    results_df = pd.DataFrame([
        {
            'upper_threshold': combo[0],
            'lower_threshold': combo[1],
            'sharpe_ratio': result['sharpe_ratio']
        }
        for combo, result in optimization_results.items()
    ])
    results_df.to_csv('optimization_results.csv', index=False)
    logger.info("Optimization results saved to 'optimization_results.csv'.")

    # 使用最優閾值重新運行回測
    logger.info("Running backtest with optimal thresholds...")
    optimal_factor = USDTIssuance2Factor(
        name='usdt_issuance',
        upper_threshold=optimal_params['upper_threshold'],
        lower_threshold=optimal_params['lower_threshold']
    )
    factor_engine_optimal = FactorEngine()
    factor_engine_optimal.register_factor(optimal_factor)

    optimal_strategy = FactorBasedStrategy(factors=[optimal_factor])

    portfolio_optimal = engine.run_backtest(data, optimal_strategy, factor_engine=factor_engine_optimal)
    print("\nPortfolio Performance with Optimal Threshold:")
    print(portfolio_optimal.head())

    # 計算績效指標
    metrics_optimal = evaluator.calculate_performance_metrics(portfolio_optimal)
    print("\nPerformance Metrics with Optimal Threshold:")
    for key, value in metrics_optimal.items():
        if 'Return' in key or 'Drawdown' in key:
            print(f"{key}: {value * 100:.2f}%")
        else:
            print(f"{key}: {value:.2f}")

    # 保存結果
    # portfolio.to_csv('backtest_initial_results.csv')
    portfolio_optimal.to_csv('backtest_optimal_results.csv')
    logger.info("Backtest results saved to 'backtest_optimal_results.csv'.")

    # with open('performance_metrics_initial.txt', 'w') as f:
    #     for key, value in metrics.items():
    #         if 'Return' in key or 'Drawdown' in key:
    #             f.write(f"{key}: {value * 100:.2f}%\n")
    #         else:
    #             f.write(f"{key}: {value:.2f}\n")
    with open('performance_metrics_optimal.txt', 'w') as f:
        # 写入最优阈值信息
        f.write(f"Optimal Threshold: {optimal_params}\n")
        for key, value in metrics_optimal.items():
            if 'Return' in key or 'Drawdown' in key:
                f.write(f"{key}: {value * 100:.2f}%\n")
            else:
                f.write(f"{key}: {value:.2f}\n")
    logger.info("Performance metrics saved to  and 'performance_metrics_optimal.txt'.")

    import matplotlib.pyplot as plt

    def visualize_optimization_results(optimization_results):
        # 创建3D图表来显示双阈值优化结果
        fig = plt.figure(figsize=(12, 8))
        ax = fig.add_subplot(111, projection='3d')
        
        # 提取数据
        upper_thresholds = []
        lower_thresholds = []
        sharpe_ratios = []
        
        for params, result in optimization_results.items():
            upper_thresholds.append(result['params']['upper_threshold'])
            lower_thresholds.append(result['params']['lower_threshold'])
            sharpe_ratios.append(result['sharpe_ratio'])
        
        # 创建散点图
        scatter = ax.scatter(upper_thresholds, lower_thresholds, sharpe_ratios, 
                           c=sharpe_ratios, cmap='viridis')
        
        # 添加标题和标签
        ax.set_title("Optimization Results: Sharpe Ratio vs Thresholds", fontsize=14)
        ax.set_xlabel("Upper Threshold", fontsize=12)
        ax.set_ylabel("Lower Threshold", fontsize=12)
        ax.set_zlabel("Sharpe Ratio", fontsize=12)
        
        # 添加颜色条
        plt.colorbar(scatter, label='Sharpe Ratio')
        
        # 保存图表
        plt.savefig("optimization_results_3d_plot.png", dpi=300, bbox_inches='tight')
        plt.show()

    # 使用可视化函数
    visualize_optimization_results(optimization_results)

if __name__ == "__main__":
    main()
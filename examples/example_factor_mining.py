import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import pandas as pd
import logging
from backtest.backtest_engine import BacktestEngine
from backtest.strategy import FactorBasedStrategy
from backtest.performance import PerformanceEvaluator
from factors.factor_definitions import USDTIssuanceFactor
from factors.factor_engine import FactorEngine
from factors.optimizer import StrategyOptimizer
from data.data_loader import DataLoader

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
    usdt_factor = USDTIssuanceFactor(name='usdt_issuance', direction=1, threshold=1000000000)
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
    threshold_values = list(range(1000000, 1500000000, 1000000))  #起始，終止，步長

    # 執行優化
    logger.info("Starting optimization...")
    optimization_results = optimizer.optimize_threshold(
        data=data,
        threshold_values=threshold_values,
        factor_class=USDTIssuanceFactor,
        strategy_class=FactorBasedStrategy
    )
    logger.info(f"Optimization Results: {optimization_results}")

    # 检查 optimization_results 是否为空
    if not optimization_results:
        logger.error("Optimization results are empty!")
        return
    # 找到最優閾值
    optimal_threshold, optimal_sharpe = optimizer.find_optimal_threshold(optimization_results)
    logger.info(f"Optimal Threshold: {optimal_threshold}, Sharpe Ratio: {optimal_sharpe:.4f}")
    print(f"\nOptimal Threshold: {optimal_threshold}, Sharpe Ratio: {optimal_sharpe:.4f}")
    # 保存优化结果
    with open('optimization_results.csv', 'w') as f:
        f.write("Threshold,Sharpe_Ratio\n")
        for threshold, sharpe in optimization_results.items():
            f.write(f"{threshold},{sharpe:.4f}\n")
    logger.info("Optimization results saved to 'optimization_results.csv'.")
    # 使用最優閾值重新運行回測
    logger.info("Running backtest with optimal threshold...")
    optimal_factor = USDTIssuanceFactor(name='usdt_issuance', threshold=optimal_threshold)
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
        f.write(f"Optimal Threshold: {optimal_threshold}\n")
        for key, value in metrics_optimal.items():
            if 'Return' in key or 'Drawdown' in key:
                f.write(f"{key}: {value * 100:.2f}%\n")
            else:
                f.write(f"{key}: {value:.2f}\n")
    logger.info("Performance metrics saved to  and 'performance_metrics_optimal.txt'.")

    import matplotlib.pyplot as plt

    def visualize_optimization_results(optimization_results):
        # 提取参数和 Sharpe 比率
        thresholds = list(optimization_results.keys())
        sharpe_ratios = list(optimization_results.values())
        
        # 创建图表
        plt.figure(figsize=(10, 6))
        plt.plot(thresholds, sharpe_ratios, marker='o', linestyle='-', color='b', label='Sharpe Ratio')
        
        # 添加标题和标签
        plt.title("Optimization Results: Sharpe Ratio vs Threshold", fontsize=14)
        plt.xlabel("Threshold", fontsize=12)
        plt.ylabel("Sharpe Ratio", fontsize=12)
        plt.axhline(y=0, color='r', linestyle='--', linewidth=1, label='Zero Line')
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.legend(fontsize=12)
        
        # 保存图表到文件
        plt.savefig("optimization_results_plot.png", dpi=300)
        plt.show()

    # 使用可视化函数
    visualize_optimization_results(optimization_results)

if __name__ == "__main__":
    main()
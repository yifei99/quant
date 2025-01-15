import sys
import os


sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import logging
from backtest.backtest_engine import BacktestEngine
from backtest.strategy import FactorBasedStrategy
from backtest.trading_logic import HoldTradingLogic,LongOnlyTradingLogic,ShortOnlyTradingLogic
from backtest.performance import PerformanceEvaluator
from factors.factor_definitions import Liq2Factor
from factors.factor_engine import FactorEngine
from factors.optimizer import StrategyOptimizer
from data.data_loader import DataLoader
import numpy as np
from joblib import Parallel, delayed

# 定义常量
OPTIMAL_UPPER = 44000000
OPTIMAL_LOWER = -98000000
UPPER_RANGE = (OPTIMAL_UPPER * 0.5, OPTIMAL_UPPER * 2)  # 上阈值范围
LOWER_RANGE = (OPTIMAL_LOWER * 2, OPTIMAL_LOWER * 0.5)  # 下阈值范围
THRESHOLD_STEP = 2000000  # 阈值步长
# 创建报告目录
factor_name = "liq"
reports_dir = "./test_reports"

def load_liq_data(depth_file_path):
    """
    加载订单簿深度数据并计算流动性因子
    
    Args:
        depth_file_path (str): 订单簿深度数据文件路径
        
    Returns:
        pd.DataFrame: 包含 Date 和 liq 值的 DataFrame
    """
    # 加载深度数据
    depth_data = pd.read_csv(depth_file_path)
    depth_data['Date'] = pd.to_datetime(depth_data['Date'])
    depth_data = depth_data.sort_values('Date')
    
 
    depth_data['Liq'] = depth_data['Difference']

    return depth_data[['Date', 'Liq']]

def main():
    """
    Main function for factor mining process
    """
    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    logger = logging.getLogger(__name__)

    # Initialize DataLoader
    data_folder = "../dataset"
    data_loader = DataLoader(data_folder)

    # 1. Load market data
    try:
        data = data_loader.load_data(
            exchange="binance",
            symbol="BTCUSDT",
            interval="1h",
            start_date="2023-08-10",
            end_date="2025-1-09",
            data_type="spot"
        )
        logger.info("Market data loaded successfully")
    except FileNotFoundError as e:
        logger.error(e)
        return

    # 2. Load and process factor data
    try:
        liq_file_path = "../dataset/orderbook_depth/btc_1_depth.csv"
        liq_data = load_liq_data(liq_file_path)
        logger.info("Factor data loaded successfully")
        print(liq_data['Liq'].describe())
        
        # 将市场数据的时间戳转换为datetime格式
        data['timestamp_start'] = data['timestamp_start'].astype(str).str[:10].astype(int)
        data['Date'] = pd.to_datetime(data['timestamp_start'], unit='s')
        # 合并数据，使用 Date 作为键
        data = pd.merge(data, liq_data, on='Date', how='left')
        # 删除liq为NaN的行
        original_len = len(data)
        data = data.dropna(subset=['Liq'])
        
        logger.info("Data merged successfully")
    except Exception as e:
        logger.error(f"Failed to process factor data: {e}")
        return
    
    # 3. Initialize components
    factor_engine = FactorEngine()
    liq_factor = Liq2Factor(
        name='liq', 
        upper_threshold=OPTIMAL_UPPER, 
        lower_threshold=OPTIMAL_LOWER
    )
    factor_engine.register_factor(liq_factor)
    
    # 创建两组参数范围：固定上阈值和固定下阈值
    fixed_upper_params = {
        'upper_threshold': [OPTIMAL_UPPER],  # 固定上阈值
        'lower_threshold': np.round(np.arange(LOWER_RANGE[0], LOWER_RANGE[1], THRESHOLD_STEP)).tolist()  # 变化下阈值
    }
    
    fixed_lower_params = {
        'upper_threshold': np.round(np.arange(UPPER_RANGE[0], UPPER_RANGE[1], THRESHOLD_STEP)).tolist(),  # 变化上阈值
        'lower_threshold': [OPTIMAL_LOWER]  # 固定下阈值
    }
    
    # logic = HoldTradingLogic(commission=0.001, slippage=0.001)
    logic = LongOnlyTradingLogic(commission=0.001, slippage=0.001)
    # logic = ShortOnlyTradingLogic(commission=0.001, slippage=0.001)

    
    strategy = FactorBasedStrategy(factors=[liq_factor])
    engine = BacktestEngine(
        initial_capital=10000.0,
        commission=0.001,
        slippage=0.001,
        trading_logic=logic,
    )
    evaluator = PerformanceEvaluator(
        periods_per_year=365*24
    )

    # 4. Run backtest
    logger.info("Starting backtest...")
    try:
        optimizer = StrategyOptimizer(engine=engine, evaluator=evaluator)
        
        # 4.1 固定上阈值的优化
        logger.info("Testing with fixed upper threshold...")
        fixed_upper_results = optimizer.optimize_thresholds(
            data=data,
            threshold_params=fixed_upper_params,
            factor_class=Liq2Factor,
            strategy_class=FactorBasedStrategy,
            n_jobs=-1
        )
        
        # 4.2 固定下阈值的优化
        logger.info("Testing with fixed lower threshold...")
        fixed_lower_results = optimizer.optimize_thresholds(
            data=data,
            threshold_params=fixed_lower_params,
            factor_class=Liq2Factor,
            strategy_class=FactorBasedStrategy,
            n_jobs=-1
        )
        

        factor_dir = os.path.join(reports_dir, factor_name)
        os.makedirs(factor_dir, exist_ok=True)
        
        # 生成二维可视化
        import matplotlib.pyplot as plt
        
        # 固定上阈值的图
        plt.figure(figsize=(10, 6))
        lower_thresholds = []
        sharpe_ratios_upper = []  # 重命名以区分
        
        # 按顺序排序结果
        sorted_upper_results = sorted(fixed_upper_results.items(), key=lambda x: x[0][1])
        for combo, result in sorted_upper_results:
            if result is not None:  # 确保结果不是None
                lower_thresholds.append(combo[1])  # lower threshold
                sharpe_ratios_upper.append(result['sharpe_ratio'])
        
        plt.plot(lower_thresholds, sharpe_ratios_upper, marker='o')
        plt.xlabel('Lower Threshold')
        plt.ylabel('Sharpe Ratio')
        plt.title(f'Sharpe Ratio vs Lower Threshold (Upper = {OPTIMAL_UPPER})')
        plt.grid(True)
        plt.savefig(os.path.join(factor_dir, 'fixed_upper_analysis.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        # 固定下阈值的图
        plt.figure(figsize=(10, 6))
        upper_thresholds = []
        sharpe_ratios_lower = []  # 重命名以区分
        
        # 按顺序排序结果
        sorted_lower_results = sorted(fixed_lower_results.items(), key=lambda x: x[0][0])
        for combo, result in sorted_lower_results:
            if result is not None:  # 确保结果不是None
                upper_thresholds.append(combo[0])  # upper threshold
                sharpe_ratios_lower.append(result['sharpe_ratio'])
        
        plt.plot(upper_thresholds, sharpe_ratios_lower, marker='o')
        plt.xlabel('Upper Threshold')
        plt.ylabel('Sharpe Ratio')
        plt.title(f'Sharpe Ratio vs Upper Threshold (Lower = {OPTIMAL_LOWER})')
        plt.grid(True)
        plt.savefig(os.path.join(factor_dir, 'fixed_lower_analysis.png'), dpi=300, bbox_inches='tight')
        plt.close()       
        
    except Exception as e:
        logger.error(f"Analysis failed: {e}")
        logger.error("Full error:", exc_info=True)
        return

if __name__ == "__main__":
    main()
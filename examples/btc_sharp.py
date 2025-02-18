import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import logging
import multiprocessing
from backtest.backtest_engine import BacktestEngine
from backtest.strategy import FactorBasedStrategy
from backtest.performance import PerformanceEvaluator
from factors.factor_definitions import USDTIssuance2Factor
from factors.factor_engine import FactorEngine
from factors.optimizer import StrategyOptimizer
from data.data_loader import DataLoader
import numpy as np



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

    # Initialize DataLoader
    data_folder = "../dataset"
    data_loader = DataLoader(data_folder)

    # 1. Load market data
    try:
        data = data_loader.load_data(
            exchange="binance",
            symbol="BTCUSDT",
            interval="1d",
            start_date="2021-03-31",
            end_date="2024-12-05",
            data_type="spot"
        )
        
        # 将市场数据的时间戳转换为datetime格式
        data['timestamp_start'] = data['timestamp_start'].astype(str).str[:10].astype(int)
        data['Date'] = pd.to_datetime(data['timestamp_start'], unit='s')
        
        # 设置您想要的日期范围
        start_filter = '2024-01-01'
        end_filter = '2024-12-31'
        
        # 过滤数据
        data = data[(data['Date'] >= start_filter) & (data['Date'] <= end_filter)]
        data = data[['Date','close']]
        logger.info("Market data loaded and filtered successfully")
    except FileNotFoundError as e:
        logger.error(e)
        return

 
    # 3. Initialize components
    class SimpleHoldStrategy:
        def generate_signals(self, data):
            """第一天买入，之后持有"""
  
            
            signals = pd.Series(1, index=data.index, name='signal')
            signals.iloc[-1] = 0  # 最后一天卖出
            # 将信号添加到数据中
            data['signal'] = signals

            return data
    
    strategy = SimpleHoldStrategy()
    engine = BacktestEngine(
        initial_capital=10000.0,
        commission=0.001,
        slippage=0.001
    )
    evaluator = PerformanceEvaluator(
        periods_per_year=365
    )
 
    # 4. Run backtest
    logger.info("Starting backtest...")
    try:

        
        # 4.1 Direct backtest
        portfolio = engine.run_backtest(data, strategy,plot=True)
        
        # 计算并打印性能指标
        metrics = evaluator.calculate_performance_metrics(portfolio)
        logger.info("Performance Metrics:")
        for key, value in metrics.items():
            if 'Return' in key or 'Drawdown' in key:
                logger.info(f"{key}: {value * 100:.2f}%")
            else:
                logger.info(f"{key}: {value:.4f}")
                
        # 保存回测结果
        reports_dir = "../reports/buy_and_hold"
        os.makedirs(reports_dir, exist_ok=True)
        portfolio.to_csv(os.path.join(reports_dir, 'backtest_results.csv'))
        
    except Exception as e:
        logger.error(f"Backtest failed: {e}")
        return


if __name__ == "__main__":
    main()
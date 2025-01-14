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
        upper_threshold=-100000000, 
        lower_threshold=-178000000
    )
    factor_engine.register_factor(liq_factor)
    
    
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
        # 4.1 Direct backtest
        portfolio = engine.run_backtest(data, strategy, factor_engine)
                # 计算并打印性能指标
        metrics = evaluator.calculate_performance_metrics(portfolio)
        logger.info("Performance Metrics:")
        for key, value in metrics.items():
            if 'Return' in key or 'Drawdown' in key:
                logger.info(f"{key}: {value * 100:.2f}%")
            else:
                logger.info(f"{key}: {value:.4f}")
                
        logger.info("Initial backtest completed")

        # 4.2 Parameter optimization
        optimizer = StrategyOptimizer(engine=engine, evaluator=evaluator)
        threshold_params = {
            'upper_threshold': np.round(np.arange(-178000000, 90700000, 2000000)).tolist(),
            'lower_threshold': np.round(np.arange(-178000000, 90700000, 2000000)).tolist()
        }
        # 使用 joblib 替代 multiprocessing
        n_jobs = -1  # 使用 CPU核心数-1
        
        logger.info("Starting parameter optimization...")
        optimization_results = optimizer.optimize_thresholds(
            data=data,
            threshold_params=threshold_params,
            factor_class=Liq2Factor,
            strategy_class=FactorBasedStrategy,
            n_jobs=n_jobs  # 修改参数名
        )
        
        
        # 找到最优閾值组合
        optimal_params, optimal_sharpe, optimal_metrics, optimized_portfolio = (
            optimizer.find_optimal_thresholds(
                results=optimization_results,
                data=data,
                factor_class=Liq2Factor,
                strategy_class=FactorBasedStrategy
            )
        )
        logger.info(f"Optimization completed. Optimal parameters: {optimal_params}")
        
        # Create reports directory structure
        factor_name = liq_factor.name
        reports_dir = "../reports"
        factor_dir = os.path.join(reports_dir, factor_name)
        
        # Create factor directory if it doesn't exist
        os.makedirs(factor_dir, exist_ok=True)
        logger.info(f"Created reports directory for {factor_name}")
        
        # Save optimization results and create interactive 3D visualization
        results_df = pd.DataFrame([
            {
                'upper_threshold': combo[0],
                'lower_threshold': combo[1],
                'sharpe_ratio': result['sharpe_ratio']
            }
            for combo, result in optimization_results.items()
        ])
        results_df.to_csv(os.path.join(factor_dir, 'optimization_results.csv'), index=False)
        
        # Create 3D visualization
        try:
            import matplotlib.pyplot as plt
            from mpl_toolkits.mplot3d import Axes3D
            
            fig = plt.figure(figsize=(12, 8))
            ax = fig.add_subplot(111, projection='3d')
            
            scatter = ax.scatter(
                results_df['upper_threshold'],
                results_df['lower_threshold'],
                results_df['sharpe_ratio'],
                c=results_df['sharpe_ratio'],
                cmap='viridis'
            )
            
            ax.set_xlabel('Upper Threshold')
            ax.set_ylabel('Lower Threshold')
            ax.set_zlabel('Sharpe Ratio')
            ax.set_title('Optimization Results: Sharpe Ratio vs Thresholds')
            
            plt.colorbar(scatter, label='Sharpe Ratio')
            plt.savefig(os.path.join(factor_dir, 'optimization_3d.png'), dpi=300, bbox_inches='tight')
            plt.close()
            
            logger.info("3D visualization created successfully")
        except Exception as e:
            logger.error(f"Failed to create 3D visualization: {e}")
            
        # Save backtest results and metrics
        optimized_portfolio.to_csv(os.path.join(factor_dir, 'backtest_results.csv'))
        
        with open(os.path.join(factor_dir, 'performance_metrics.txt'), 'w') as f:
            f.write(f"Optimal Parameters:\n")
            f.write(f"Upper Threshold: {optimal_params['upper_threshold']}\n")
            f.write(f"Lower Threshold: {optimal_params['lower_threshold']}\n\n")
            f.write("Performance Metrics:\n")
            for key, value in optimal_metrics.items():
                if 'Return' in key or 'Drawdown' in key:
                    f.write(f"{key}: {value * 100:.2f}%\n")
                else:
                    f.write(f"{key}: {value:.4f}\n")
                    
        logger.info(f"All results saved to {factor_dir}")
        
    except Exception as e:
        logger.error(f"Backtest/Optimization failed: {e}")
        logger.error("Full error:", exc_info=True)  # 打印完整错误信息
        return


if __name__ == "__main__":
    main()
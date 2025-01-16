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
from factors.factor_definitions import USDTIssuance2Factor, UsdVolumeMaFactor
from factors.factor_engine import FactorEngine
from factors.optimizer import StrategyOptimizer
from data.data_loader import DataLoader
from backtest.trading_logic import HoldTradingLogic,LongOnlyTradingLogic,ShortOnlyTradingLogic


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
            symbol="SOLUSDT",
            interval="1d",
            start_date="2022-1-01",
            end_date="2024-12-31",
            data_type="spot"
        )

        # 将市场数据的时间戳转换为datetime格式
        data['timestamp_start'] = data['timestamp_start'].astype(str).str[:10].astype(int)
        data['Date'] = pd.to_datetime(data['timestamp_start'], unit='s')
        
        # 设置您想要的日期范围
        start_filter = '2022-1-01'
        end_filter = '2024-12-31'
        
        # 过滤数据
        data = data[(data['Date'] >= start_filter) & (data['Date'] <= end_filter)]
        data = data[['Date','close','volume','quote_asset_volume']]
        logger.info("Market data loaded successfully")
    except FileNotFoundError as e:
        logger.error(e)
        return



    # 3. Initialize components
    factor_engine = FactorEngine()
    ma_factor = UsdVolumeMaFactor(
        name='usd_volume_ma', 
        ma_period=7
    )
    factor_engine.register_factor(ma_factor)
    strategy = FactorBasedStrategy(factors=[ma_factor])
    logic = HoldTradingLogic(commission=0.001, slippage=0.001)
    # logic = LongOnlyTradingLogic(commission=0.001, slippage=0.001)
    # logic = ShortOnlyTradingLogic(commission=0.001, slippage=0.001)
    engine = BacktestEngine(
        initial_capital=10000.0,
        commission=0.001,
        slippage=0.001,
        trading_logic=logic,
    )
    evaluator = PerformanceEvaluator()

    # 4. Run backtest
    logger.info("Starting backtest...")
    try:
        # 4.1 Direct backtest
        portfolio = engine.run_backtest(data, strategy, factor_engine,plot=True)
        metrics = evaluator.calculate_performance_metrics(portfolio)
        logger.info("Initial backtest completed")
        logger.info("Performance Metrics:")
        for key, value in metrics.items():
            if 'Return' in key or 'Drawdown' in key:
                logger.info(f"{key}: {value * 100:.2f}%")
            else:
                logger.info(f"{key}: {value:.4f}")
        # 4.2 Parameter optimization
        optimizer = StrategyOptimizer(engine=engine, evaluator=evaluator)
        threshold_params = {
            'ma_period': range(3, 240, 1),
        }
        
        # 使用 joblib 替代 multiprocessing
        n_jobs = -1  # 使用 CPU核心数-1
        
        logger.info("Starting parameter optimization...")
        optimization_results = optimizer.optimize_thresholds(
            data=data,
            threshold_params=threshold_params,
            factor_class=UsdVolumeMaFactor,
            strategy_class=FactorBasedStrategy,
            n_jobs=n_jobs  # 修改参数名
        )
        
        # 找到最优閾值组合
        optimal_params, optimal_sharpe, optimal_metrics, optimized_portfolio = (
            optimizer.find_optimal_thresholds(
                results=optimization_results,
                data=data,
                factor_class=UsdVolumeMaFactor,
                strategy_class=FactorBasedStrategy
            )
        )
        logger.info(f"Optimization completed. Optimal parameters: {optimal_params}")
        
        # Create reports directory structure
        factor_name = ma_factor.name
        reports_dir = "../reports"
        factor_dir = os.path.join(reports_dir, factor_name)
        
        # Create factor directory if it doesn't exist
        os.makedirs(factor_dir, exist_ok=True)
        logger.info(f"Created reports directory for {factor_name}")
        
        # Save optimization results and create interactive 3D visualization
        results_df = pd.DataFrame([
            {
                'ma_period': combo[0],
                'sharpe_ratio': result['sharpe_ratio']
            }
            for combo, result in optimization_results.items()
        ])
        results_df.to_csv(os.path.join(factor_dir, 'optimization_results.csv'), index=False)
        
        # Create 2D visualization
        try:
            plt.figure(figsize=(12, 6))
            plt.plot(
                results_df['ma_period'], 
                results_df['sharpe_ratio'], 
                marker='o', 
                linestyle='-', 
                linewidth=2, 
                markersize=6
            )
            
            # 标记最优点
            best_ma = results_df.loc[results_df['sharpe_ratio'].idxmax()]
            plt.plot(
                best_ma['ma_period'], 
                best_ma['sharpe_ratio'], 
                'r*', 
                markersize=15, 
                label=f'Best MA: {int(best_ma["ma_period"])} days\nSharpe: {best_ma["sharpe_ratio"]:.2f}'
            )
            
            plt.grid(True, linestyle='--', alpha=0.7)
            plt.xlabel('Moving Average Period (Days)')
            plt.ylabel('Sharpe Ratio')
            plt.title('MA Period Optimization Results')
            plt.legend()
            
            plt.savefig(os.path.join(factor_dir, 'optimization_2d.png'), dpi=300, bbox_inches='tight')
            plt.close()
            
            logger.info("2D visualization created successfully")
        except Exception as e:
            logger.error(f"Failed to create 2D visualization: {e}")
            
        # Save backtest results and metrics
        optimized_portfolio.to_csv(os.path.join(factor_dir, 'backtest_results.csv'))
        
        with open(os.path.join(factor_dir, 'performance_metrics.txt'), 'w') as f:
            f.write(f"Optimal Parameters:\n")
            f.write(f"ma_period: {optimal_params['ma_period']}\n\n")
            f.write("Performance Metrics:\n")
            for key, value in optimal_metrics.items():
                if 'Return' in key or 'Drawdown' in key:
                    f.write(f"{key}: {value * 100:.2f}%\n")
                else:
                    f.write(f"{key}: {value:.4f}\n")
                    
        logger.info(f"All results saved to {factor_dir}")
        
    except Exception as e:
        logger.error(f"Backtest/Optimization failed: {e}")
        return


if __name__ == "__main__":
    main()
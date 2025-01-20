import sys
import os

import numpy as np
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from matplotlib import pyplot as plt
import pandas as pd
import logging
from joblib import Parallel, delayed
from backtest.backtest_engine import BacktestEngine
from backtest.strategy import FactorBasedStrategy
from backtest.performance import PerformanceEvaluator
from factors.factor_definitions import Volume2MaFactor, Price2MaFactor,  UsdVolume2MaFactor
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
            symbol="BTCUSDT",
            interval="1d",
            start_date="2021-03-31",
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

    ma_factor = UsdVolume2MaFactor(
        name='usd_volume_2ma', 
        ma_period_1=7,
        ma_period_2=14
    )
    # ma_factor = Price2MaFactor(
    #     name='price_2ma', 
    #     ma_period_1=7,
    #     ma_period_2=14
    # )
    # ma_factor = Volume2MaFactor(
    #     name='volume_2ma', 
    #     ma_period_1=7,
    #     ma_period_2=14
    # )
    factor_engine.register_factor(ma_factor)

    strategy = FactorBasedStrategy(factors=[ma_factor])
    def get_trading_logic(logic_type=1, commission=0.001, slippage=0.001):
        if logic_type == 1:
            return HoldTradingLogic(commission=commission, slippage=slippage)
        elif logic_type == 2:
            return LongOnlyTradingLogic(commission=commission, slippage=slippage)
        elif logic_type == 3:
            return ShortOnlyTradingLogic(commission=commission, slippage=slippage)
        else:
            raise ValueError("Invalid logic_type. Must be 1, 2 or 3.")
            
    logic = get_trading_logic(logic_type=2)
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
        portfolio = engine.run_backtest(data, strategy, factor_engine)
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
            'ma_period_1': range(3, 360, 1),
            'ma_period_2': range(3, 360, 1),
        }
        
        # 使用 joblib 替代 multiprocessing
        n_jobs = -1  # 使用 CPU核心数-1

        factor_class = UsdVolume2MaFactor
        # factor_class = Price2MaFactor
        # factor_class = Volume2MaFactor

        logger.info("Starting parameter optimization...")
        optimization_results = optimizer.optimize_thresholds(
            data=data,
            threshold_params=threshold_params,
            factor_class=factor_class,
            strategy_class=FactorBasedStrategy,
            n_jobs=n_jobs,
            enforce_threshold_order=True
        )
        

        # 找到最优閾值组合
        optimal_params, optimal_sharpe, optimal_metrics, optimized_portfolio = (
            optimizer.find_optimal_thresholds(
                results=optimization_results,
                data=data,
                factor_class=factor_class,
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
        
        # Create 3D visualization
        try:
            results_df = pd.DataFrame([
                {
                    'ma_period_1': combo[0],
                    'ma_period_2': combo[1], 
                    'sharpe_ratio': result['sharpe_ratio']
                }
                for combo, result in optimization_results.items()
            ])
            results_df.to_csv(os.path.join(factor_dir, 'optimization_results.csv'), index=False)

            # Create 3D surface plot
            fig = plt.figure(figsize=(12, 8))
            ax = fig.add_subplot(111, projection='3d')

            # Create meshgrid for surface plot
            ma1_unique = sorted(results_df['ma_period_1'].unique())
            ma2_unique = sorted(results_df['ma_period_2'].unique())
            X, Y = np.meshgrid(ma1_unique, ma2_unique)
            
            # Create Z matrix
            Z = np.zeros_like(X)
            for i, ma1 in enumerate(ma1_unique):
                for j, ma2 in enumerate(ma2_unique):
                    mask = (results_df['ma_period_1'] == ma1) & (results_df['ma_period_2'] == ma2)
                    if mask.any():
                        Z[j,i] = results_df[mask]['sharpe_ratio'].values[0]

            # 获取夏普比率的范围
            min_sharpe = np.nanmin(Z)
            max_sharpe = np.nanmax(Z)
            
            # Plot surface with explicit range
            surf = ax.plot_surface(X, Y, Z, cmap='viridis', alpha=0.8, 
                                  vmin=min_sharpe, vmax=max_sharpe)
            
            # Find and mark optimal point
            best_result = results_df.loc[results_df['sharpe_ratio'].idxmax()]
            ax.scatter(
                best_result['ma_period_1'],
                best_result['ma_period_2'],
                best_result['sharpe_ratio'],
                color='red',
                s=100,
                marker='*',
                label=f'Best Point\nMA1={int(best_result["ma_period_1"])}\nMA2={int(best_result["ma_period_2"])}\nSharpe={best_result["sharpe_ratio"]:.2f}'
            )

            # Customize plot
            ax.set_xlabel('MA Period 1')
            ax.set_ylabel('MA Period 2')
            ax.set_zlabel('Sharpe Ratio')
            ax.set_title('MA Periods Optimization Results')
            
            # 设置z轴范围，留出一定余量
            z_margin = (max_sharpe - min_sharpe) * 0.1
            ax.set_zlim(min_sharpe - z_margin, max_sharpe + z_margin)
            
            # Add colorbar with explicit range
            cbar = fig.colorbar(surf, ax=ax, shrink=0.5, aspect=5)
            cbar.mappable.set_clim(min_sharpe, max_sharpe)
            
            # Add legend
            ax.legend()

            plt.savefig(os.path.join(factor_dir, 'optimization_3d.png'), dpi=300, bbox_inches='tight')
            plt.close()
            
            logger.info("3D visualization created successfully")
        except Exception as e:
            logger.error(f"Failed to create 3D visualization: {e}")
            
        # Save backtest results and metrics
        optimized_portfolio.to_csv(os.path.join(factor_dir, 'backtest_results.csv'))
        
        with open(os.path.join(factor_dir, 'performance_metrics.txt'), 'w') as f:
            f.write(f"Optimal Parameters:\n")
            f.write(f"ma_period_1: {optimal_params['ma_period_1']}\n")
            f.write(f"ma_period_2: {optimal_params['ma_period_2']}\n\n")
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
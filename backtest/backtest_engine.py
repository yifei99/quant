# backtest/backtest_engine.py

import pandas as pd
import numpy as np
import logging
from .strategy import BaseStrategy
from factors.factor_engine import FactorEngine
from .performance import PerformanceEvaluator
from .trading_logic import BaseTradingLogic, StandardTradingLogic

class BacktestEngine:
    """
    Backtest engine, responsible for executing strategies and tracking portfolios.
    """
    def __init__(self, 
                 trading_logic: BaseTradingLogic = None,
                 periods_per_year: int = 365):
        """
        Initialize backtest engine.
        
        Args:
            trading_logic: Trading logic instance
            periods_per_year: Number of periods in a year (e.g., 365 for daily, 52 for weekly, 12 for monthly)
        """
        self.periods_per_year = periods_per_year
        
        if trading_logic:
            self.trading_logic = trading_logic
        else:
            self.trading_logic = StandardTradingLogic()
            
        self.logger = logging.getLogger(__name__)

    def run_backtest(self, data: pd.DataFrame, strategy: BaseStrategy, 
                    factor_engine: FactorEngine = None, plot=False, save_dir=None) -> pd.DataFrame:
        """
        Execute backtest using the specified trading logic.
        
        Args:
            data: Market data DataFrame
            strategy: Trading strategy instance
            factor_engine: Factor engine instance
            plot: Whether to generate performance plots
            save_dir: Directory to save plots and results (if None, will use current directory)
        """
        # Calculate factors if factor engine is provided
        if factor_engine:
            factor_values = factor_engine.calculate_factors(data)
            data = pd.concat([data, factor_values], axis=1)

        # Generate trading signals
        signals = strategy.generate_signals(data)
        
        # Execute trades using trading logic
        portfolio = self.trading_logic.execute_trades(data, signals)
        
        # Add visualization if requested
        if plot:
            evaluator = PerformanceEvaluator(
                periods_per_year=self.periods_per_year
            )
            evaluator.plot_performance(portfolio, data, save_dir=save_dir)

        # 重新组织DataFrame的展示格式
        portfolio = portfolio[['Date', 'close', 'signal', 'holdings']]
        
        return portfolio
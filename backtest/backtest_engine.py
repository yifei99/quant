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
                 initial_capital: float = 10000.0,
                 commission: float = 0.001,
                 slippage: float = 0.001,
                 trading_logic: BaseTradingLogic = None,
                 periods_per_year: int = 365):
        """
        Initialize backtest engine.
        
        Args:
            initial_capital: Initial capital for backtest
            commission: Commission rate for trades
            slippage: Slippage rate for trades
            periods_per_year: Number of periods in a year (e.g., 365 for daily, 52 for weekly, 12 for monthly)
        """
        self.initial_capital = initial_capital
        self.commission = commission
        self.slippage = slippage
        
        # 如果提供了trading_logic实例，更新其commission和slippage
        if trading_logic:
            trading_logic.commission = commission
            trading_logic.slippage = slippage
            self.trading_logic = trading_logic
        else:
            # 如果没有提供trading_logic，使用默认的StandardTradingLogic
            self.trading_logic = StandardTradingLogic(commission, slippage)
            
        self.periods_per_year = periods_per_year
        self.logger = logging.getLogger(__name__)

    def run_backtest(self, data: pd.DataFrame, strategy: BaseStrategy, 
                    factor_engine: FactorEngine = None, plot=False) -> pd.DataFrame:
        """Execute backtest using the specified trading logic."""
        # Calculate factors if factor engine is provided
        if factor_engine:
            factor_values = factor_engine.calculate_factors(data)
            data = pd.concat([data, factor_values], axis=1)

        # Generate trading signals
        signals = strategy.generate_signals(data)
        
        # Execute trades using trading logic
        portfolio = self.trading_logic.execute_trades(data, signals)
        
        # Add visualization
        if plot:
            evaluator = PerformanceEvaluator()
            evaluator.plot_performance(portfolio, data)
        
        return portfolio
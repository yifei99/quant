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
    def __init__(self, initial_capital=10000.0, commission=0.001, slippage=0.001, 
                 trading_logic: BaseTradingLogic = None):
        """
        Initialize backtest engine.

        Args:
            initial_capital (float): Initial capital.
            commission (float): Transaction commission ratio.
            slippage (float): Transaction slippage ratio.
            trading_logic (BaseTradingLogic): Trading logic implementation.
        """
        self.initial_capital = initial_capital
        self.commission = commission
        self.slippage = slippage
        self.trading_logic = trading_logic or StandardTradingLogic(commission, slippage)
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
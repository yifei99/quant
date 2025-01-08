# backtest/backtest_engine.py

import pandas as pd
import numpy as np
import logging
from .strategy import BaseStrategy
from factors.factor_engine import FactorEngine
from .performance import PerformanceEvaluator

class BacktestEngine:
    """
    Backtest engine, responsible for executing strategies and tracking portfolios.
    """
    def __init__(self, initial_capital=10000.0, commission=0.001, slippage=0.001):
        """
        Initialize backtest engine.

        Args:
            initial_capital (float): Initial capital.
            commission (float): Transaction commission ratio.
            slippage (float): Transaction slippage ratio.
        """
        self.initial_capital = initial_capital
        self.commission = commission
        self.slippage = slippage
        self.logger = logging.getLogger(__name__)

    def run_backtest(self, data: pd.DataFrame, strategy: BaseStrategy, factor_engine: FactorEngine = None, plot=False) -> pd.DataFrame:
        """
        Execute backtest.
        
        No Position:
        - Signal 1: Open long position
        - Signal -1: Open short position
        - Signal 0: No action
        
        Long Position:
        - Signal 1: No action
        - Signal -1: Switch to short (sell 2x position)
        - Signal 0: Close position
        
        Short Position:
        - Signal 1: Switch to long (buy 2x position)
        - Signal -1: No action
        - Signal 0: Close position
        """
        # Calculate factors if factor engine is provided
        if factor_engine:
            factor_values = factor_engine.calculate_factors(data)
            data = pd.concat([data, factor_values], axis=1)

        # Generate trading signals
        signals = strategy.generate_signals(data)

        # Initialize portfolio
        portfolio = pd.DataFrame(index=data.index)
        portfolio['holdings'] = 0.0     # Position size (can be negative for shorts)
        portfolio['total'] = 0.0        # Cumulative returns
        portfolio['cost_basis'] = 0.0   # Entry cost basis
        portfolio['signal'] = signals['signal']  # Add signal column
        
        TRADE_AMOUNT = 10000  # Fixed trade amount
        
        # Simulate trading
        for i in range(len(data)):
            current_price = data.iloc[i]['close']
            signal = signals.iloc[i]['signal']
            
            if i == 0:
                portfolio.iloc[i, portfolio.columns.get_loc('total')] = 0
                continue
                
            # Copy previous day's state (except signal)
            portfolio.iloc[i, portfolio.columns.get_loc('holdings')] = portfolio.iloc[i-1]['holdings']
            portfolio.iloc[i, portfolio.columns.get_loc('total')] = portfolio.iloc[i-1]['total']
            portfolio.iloc[i, portfolio.columns.get_loc('cost_basis')] = portfolio.iloc[i-1]['cost_basis']
            
            current_holdings = portfolio.iloc[i]['holdings']
            
            # Process signals based on current position
            if current_holdings == 0:  # No position
                if signal == 1:  # Open long
                    entry_price = current_price * (1 + self.slippage)
                    shares = TRADE_AMOUNT / entry_price / (1 + self.commission)
                    cost = shares * entry_price * (1 + self.commission)
                    portfolio.iloc[i, portfolio.columns.get_loc('holdings')] = shares
                    portfolio.iloc[i, portfolio.columns.get_loc('cost_basis')] = cost
                    
                elif signal == -1:  # Open short
                    entry_price = current_price * (1 - self.slippage)
                    shares = TRADE_AMOUNT / entry_price / (1 + self.commission)
                    cost = shares * entry_price * (1 + self.commission)
                    portfolio.iloc[i, portfolio.columns.get_loc('holdings')] = -shares
                    portfolio.iloc[i, portfolio.columns.get_loc('cost_basis')] = cost
                    
            elif current_holdings > 0:  # Long position
                if signal == -1:  # Switch to short (sell 2x)
                    # Close long position
                    close_price = current_price * (1 - self.slippage)
                    profit = (current_holdings * close_price * (1 - self.commission) - 
                             portfolio.iloc[i]['cost_basis'])
                    portfolio.iloc[i, portfolio.columns.get_loc('total')] += profit
                    
                    # Open short position
                    entry_price = current_price * (1 - self.slippage)
                    shares = TRADE_AMOUNT / entry_price / (1 + self.commission)
                    cost = shares * entry_price * (1 + self.commission)
                    portfolio.iloc[i, portfolio.columns.get_loc('holdings')] = -shares
                    portfolio.iloc[i, portfolio.columns.get_loc('cost_basis')] = cost
                    
                elif signal == 0:  # Close position
                    close_price = current_price * (1 - self.slippage)
                    profit = (current_holdings * close_price * (1 - self.commission) - 
                             portfolio.iloc[i]['cost_basis'])
                    portfolio.iloc[i, portfolio.columns.get_loc('total')] += profit
                    portfolio.iloc[i, portfolio.columns.get_loc('holdings')] = 0
                    portfolio.iloc[i, portfolio.columns.get_loc('cost_basis')] = 0
                    
            elif current_holdings < 0:  # Short position
                if signal == 1:  # Switch to long (buy 2x)
                    # Close short position
                    close_price = current_price * (1 + self.slippage)
                    profit = -(current_holdings * close_price * (1 - self.commission) + 
                              portfolio.iloc[i]['cost_basis'])
                    portfolio.iloc[i, portfolio.columns.get_loc('total')] += profit
                    
                    # Open long position
                    entry_price = current_price * (1 + self.slippage)
                    shares = TRADE_AMOUNT / entry_price / (1 + self.commission)
                    cost = shares * entry_price * (1 + self.commission)
                    portfolio.iloc[i, portfolio.columns.get_loc('holdings')] = shares
                    portfolio.iloc[i, portfolio.columns.get_loc('cost_basis')] = cost
                    
                elif signal == 0:  # Close position
                    close_price = current_price * (1 + self.slippage)
                    profit = -(current_holdings * close_price * (1 - self.commission) + 
                              portfolio.iloc[i]['cost_basis'])
                    portfolio.iloc[i, portfolio.columns.get_loc('total')] += profit
                    portfolio.iloc[i, portfolio.columns.get_loc('holdings')] = 0
                    portfolio.iloc[i, portfolio.columns.get_loc('cost_basis')] = 0
        
        portfolio['Date'] = data['Date']
        
        # Only generate plots if requested
        if plot:
            evaluator = PerformanceEvaluator()
            evaluator.plot_performance(portfolio, data)
        
        return portfolio
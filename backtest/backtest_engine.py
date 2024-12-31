# backtest/backtest_engine.py

import pandas as pd
import numpy as np
import logging
from .strategy import BaseStrategy
from factors.factor_engine import FactorEngine
from .performance import PerformanceEvaluator

class BacktestEngine:
    """
    回测引擎，负责执行策略并跟踪投资组合。
    """
    def __init__(self, initial_capital=100000.0, commission=0.001):
        """
        初始化回测引擎。

        参数:
            initial_capital (float): 初始资金。
            commission (float): 交易佣金比例。
        """
        self.initial_capital = initial_capital
        self.commission = commission
        self.logger = logging.getLogger(__name__)

    def run_backtest(self, data: pd.DataFrame, strategy: BaseStrategy, factor_engine: FactorEngine = None) -> pd.DataFrame:
        """
        执行回测。

        参数:
            data (pd.DataFrame): 包含价格数据和因子数据的DataFrame。
            strategy (BaseStrategy): 策略实例。
            factor_engine (FactorEngine, optional): 因子计算引擎实例。

        返回:
            pd.DataFrame: 投资组合表现的DataFrame。
        """
        # 如果提供了因子引擎，先计算因子并合并到数据中
        if factor_engine:
            factor_values = factor_engine.calculate_factors(data)
            data = pd.concat([data, factor_values], axis=1)

        # 生成交易信号
        signals = strategy.generate_signals(data)

        # 初始化投资组合
        portfolio = pd.DataFrame(index=data.index)
        portfolio['holdings'] = 0.0     # 持仓数量
        portfolio['total'] = 0.0        # 累计收益
        portfolio['positions'] = signals['positions']  # 交易信号
        portfolio['cost_basis'] = 0.0   # 开仓成本
        
        TRADE_AMOUNT = 10000  # 每次交易金额固定为 1 万
        
        # 模拟交易
        for i in range(len(data)):
            current_price = data.iloc[i]['close']
            signal = signals['positions'].iloc[i]
            
            if i == 0:
                portfolio.iloc[i, portfolio.columns.get_loc('total')] = 0
                continue
                
            # 复制前一天的状态
            portfolio.iloc[i] = portfolio.iloc[i-1]
            
            # 如果有交易信号
            if signal != 0:
                if signal > 0 and portfolio.iloc[i]['holdings'] == 0:  # 买入信号且当前无持仓
                    # 计算可买入数量
                    shares = TRADE_AMOUNT / current_price / (1 + self.commission)
                    cost = shares * current_price * (1 + self.commission)
                    
                    portfolio.iloc[i, portfolio.columns.get_loc('holdings')] = shares
                    portfolio.iloc[i, portfolio.columns.get_loc('cost_basis')] = cost
                    
                elif signal < 0 and portfolio.iloc[i]['holdings'] > 0:  # 卖出信号且有持仓
                    # 计算卖出收益
                    shares = portfolio.iloc[i]['holdings']
                    revenue = shares * current_price * (1 - self.commission)
                    cost_basis = portfolio.iloc[i]['cost_basis']
                    
                    # 计算这笔交易的收益
                    trade_profit = revenue - cost_basis
                    
                    # 更新投资组合
                    portfolio.iloc[i, portfolio.columns.get_loc('holdings')] = 0
                    portfolio.iloc[i, portfolio.columns.get_loc('cost_basis')] = 0
                    portfolio.iloc[i, portfolio.columns.get_loc('total')] += trade_profit
            
            # 不需要每日更新市值，因为我们只在平仓时计算收益
        
        portfolio['Date'] = data['Date']
        
        # 添加绘图
        evaluator = PerformanceEvaluator()
        evaluator.plot_performance(portfolio, data)
        
        return portfolio
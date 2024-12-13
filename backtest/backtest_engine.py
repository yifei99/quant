# backtest/backtest_engine.py

import pandas as pd
import numpy as np
import logging
from .strategy import BaseStrategy
from factors.factor_engine import FactorEngine

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
            # self.logger.info("Calculating factors...")
            factor_values = factor_engine.calculate_factors(data)
            data = pd.concat([data, factor_values], axis=1)

        # 生成交易信号
        # self.logger.info("Generating signals...")
        signals = strategy.generate_signals(data)

      # 初始化投资组合
        portfolio = pd.DataFrame(index=data.index)
        portfolio['holdings'] = 0.0
        portfolio['cash'] = self.initial_capital
        portfolio['total'] = self.initial_capital
        
        # 生成交易信号
        signals = strategy.generate_signals(data)
        
        # 记录每次交易的数量
        portfolio['positions'] = signals['positions']
        
        # 模拟交易
        for i in range(len(data)):
            if i == 0:
                continue
                
            current_price = data.iloc[i]['close']
            signal = signals['positions'].iloc[i]
            
            # 如果有交易信号
            if signal != 0:
                # 计算交易数量（使用全部可用资金的90%）
                if signal > 0:  # 买入信号
                    available_amount = (portfolio.iloc[i-1]['cash'] * 0.9) / current_price
                    cost = available_amount * current_price * (1 + self.commission)
                    if cost <= portfolio.iloc[i-1]['cash']:
                        portfolio.iloc[i, portfolio.columns.get_loc('holdings')] = available_amount
                        portfolio.iloc[i, portfolio.columns.get_loc('cash')] = portfolio.iloc[i-1]['cash'] - cost
                elif signal < 0:  # 卖出信号
                    holdings = portfolio.iloc[i-1]['holdings']
                    if holdings > 0:
                        revenue = holdings * current_price * (1 - self.commission)
                        portfolio.iloc[i, portfolio.columns.get_loc('holdings')] = 0
                        portfolio.iloc[i, portfolio.columns.get_loc('cash')] = portfolio.iloc[i-1]['cash'] + revenue
            else:
                # 没有交易时保持前一天的持仓
                portfolio.iloc[i, portfolio.columns.get_loc('holdings')] = portfolio.iloc[i-1]['holdings']
                portfolio.iloc[i, portfolio.columns.get_loc('cash')] = portfolio.iloc[i-1]['cash']
            
            # 更新总资产价值
            portfolio.iloc[i, portfolio.columns.get_loc('total')] = (
                portfolio.iloc[i]['holdings'] * current_price + 
                portfolio.iloc[i]['cash']
            )
        
        portfolio['Date'] = data['Date']
        return portfolio
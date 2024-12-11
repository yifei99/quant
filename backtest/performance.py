# backtest/performance.py

import pandas as pd
import numpy as np

class PerformanceEvaluator:
    """
    绩效评估类，计算回测的各项绩效指标。
    """
    def __init__(self):
        pass

    def calculate_annualized_return(self, portfolio: pd.DataFrame) -> float:
        """
        计算年化收益率。
        """
        total_return = portfolio['total'].iloc[-1] / portfolio['total'].iloc[0] - 1
        if 'Date' in portfolio.columns:
            start_date = pd.to_datetime(portfolio['Date'].iloc[0])
            end_date = pd.to_datetime(portfolio['Date'].iloc[-1])
            num_years = (end_date - start_date).days / 365.25
        else:
            raise ValueError("Portfolio must contain a 'Date' column for annualized return calculation.")
        annualized_return = (1 + total_return) ** (1 / num_years) - 1
        return annualized_return

    def calculate_sharpe_ratio(self, portfolio: pd.DataFrame, risk_free_rate=0.0) -> float:
        """
        计算夏普比率。
        """
        returns = portfolio['total'].pct_change().dropna()
        excess_returns = returns - risk_free_rate / 365
        if excess_returns.std() == 0:
            return 0.0  # 无波动时返回 0
        sharpe_ratio = np.sqrt(365) * excess_returns.mean() / excess_returns.std()
        return sharpe_ratio

    def calculate_max_drawdown(self, portfolio: pd.DataFrame) -> float:
        """
        计算最大回撤。

        参数:
            portfolio (pd.DataFrame): 投资组合表现的DataFrame。

        返回:
            float: 最大回撤。
        """
        cumulative_returns = portfolio['total']
        running_max = cumulative_returns.cummax()
        drawdown = (cumulative_returns - running_max) / running_max
        max_drawdown = drawdown.min()
        return max_drawdown

    def calculate_performance_metrics(self, portfolio: pd.DataFrame) -> dict:
        """
        计算所有绩效指标。

        参数:
            portfolio (pd.DataFrame): 投资组合表现的DataFrame。

        返回:
            dict: 包含各项绩效指标的字典。
        """
        metrics = {
            'Annualized Return': self.calculate_annualized_return(portfolio),
            'Sharpe Ratio': self.calculate_sharpe_ratio(portfolio),
            'Max Drawdown': self.calculate_max_drawdown(portfolio)
        }
        return metrics
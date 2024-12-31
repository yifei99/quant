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
        計算年化收益率。
        
        收益率 = 最終累計收益 / 10000（初始投資）
        """
        INITIAL_INVESTMENT = 10000
        final_value = portfolio['total'].iloc[-1]  # 最終累計收益
        total_return = final_value / INITIAL_INVESTMENT  # 總收益率
        
        if 'Date' in portfolio.columns:
            start_date = pd.to_datetime(portfolio['Date'].iloc[0])
            end_date = pd.to_datetime(portfolio['Date'].iloc[-1])
            num_years = (end_date - start_date).days / 365.25
            
            if num_years > 0 and total_return > -1:  # 避免負值進行指數運算
                annualized_return = (1 + total_return) ** (1 / num_years) - 1
                return annualized_return
            
        return float('nan')  # 如果無法計算則返回 NaN

    def calculate_sharpe_ratio(self, portfolio: pd.DataFrame, risk_free_rate=0.0) -> float:
        """
        計算夏普比率。使用每日收益率。
        """
        # 計算每日收益率
        daily_returns = portfolio['total'].diff() / 10000  # 使用固定基準 10000
        daily_returns = daily_returns.dropna()
        
        if len(daily_returns) > 1:
            excess_returns = daily_returns - risk_free_rate / 365
            if excess_returns.std() != 0:
                sharpe_ratio = np.sqrt(365) * excess_returns.mean() / excess_returns.std()
                return sharpe_ratio
                
        return 0.0

    def calculate_max_drawdown(self, portfolio: pd.DataFrame) -> float:
        """
        計算最大日度回撤。
        基於每日收益計算，而不是累計收益。
        """
        # 計算每日收益
        daily_returns = portfolio['total'].diff()
        
        # 找出最大單日虧損
        max_daily_drawdown = daily_returns.min() / 10000  # 相對於固定投資金額10000的百分比
        
        return max_daily_drawdown if max_daily_drawdown < 0 else 0.0

    def calculate_total_return(self, portfolio: pd.DataFrame) -> float:
        """
        計算總回報率。
        
        總回報率 = 最終累計收益 / 初始投資金額
        """
        INITIAL_INVESTMENT = 10000
        final_value = portfolio['total'].iloc[-1]  # 最終累計收益
        total_return = final_value / INITIAL_INVESTMENT  # 總回報率
        return total_return

    def calculate_performance_metrics(self, portfolio: pd.DataFrame) -> dict:
        """
        計算所有績效指標。
        """
        metrics = {
            'Total Return': self.calculate_total_return(portfolio),
            'Annualized Return': self.calculate_annualized_return(portfolio),
            'Sharpe Ratio': self.calculate_sharpe_ratio(portfolio),
            'Max Daily Drawdown': self.calculate_max_drawdown(portfolio)
        }
        return metrics

    def plot_performance(self, portfolio: pd.DataFrame, data: pd.DataFrame):
        """
        绘制策略收益走势图和标的价格走势图。
        
        参数:
            portfolio (pd.DataFrame): 包含交易记录和收益的DataFrame
            data (pd.DataFrame): 包含价格数据的DataFrame
        """
        import matplotlib.pyplot as plt
        from matplotlib.gridspec import GridSpec
        
        # 创建图表
        fig = plt.figure(figsize=(15, 10))
        gs = GridSpec(2, 1, height_ratios=[2, 1])
        
        # 计算累计收益率序列
        INITIAL_INVESTMENT = 10000
        portfolio['return_rate'] = portfolio['total'] / INITIAL_INVESTMENT
        
        # 上图：累计收益率走势
        ax1 = fig.add_subplot(gs[0])
        ax1.plot(portfolio['Date'], portfolio['return_rate'], 
                 label='Strategy Returns', color='blue')
        ax1.axhline(y=0, color='r', linestyle='--', alpha=0.3)
        ax1.set_title('Cumulative Returns')
        ax1.set_ylabel('Return Rate')
        ax1.legend()
        ax1.grid(True)
        
        # 下图：标的价格走势
        ax2 = fig.add_subplot(gs[1])
        ax2.plot(data['Date'], data['close'], 
                 label='Asset Price', color='green')
        ax2.set_title('Asset Price')
        ax2.set_ylabel('Price')
        ax2.legend()
        ax2.grid(True)
        
        # 标记交易点
        trades = portfolio[portfolio['positions'] != 0]
        for idx, trade in trades.iterrows():
            price = data.loc[idx, 'close']
            if trade['positions'] > 0:  # 买入点
                ax2.scatter(trade['Date'], price, color='red', 
                           marker='^', s=100, label='Buy' if 'Buy' not in ax2.get_legend_handles_labels()[1] else "")
            elif trade['positions'] < 0:  # 卖出点
                ax2.scatter(trade['Date'], price, color='black', 
                           marker='v', s=100, label='Sell' if 'Sell' not in ax2.get_legend_handles_labels()[1] else "")
        
        # 调整布局
        plt.tight_layout()
        
        # 保存图表
        plt.savefig('strategy_performance.png')
        plt.close()
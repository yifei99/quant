# backtest/performance.py

import pandas as pd
import numpy as np
import logging

class PerformanceEvaluator:
    """
    绩效评估类，计算回测的各项绩效指标。
    """
    def __init__(self, periods_per_year=365):
        self.periods_per_year = periods_per_year

    def _calculate_price_changes(self, close_prices: np.ndarray) -> np.ndarray:
        """计算每日价格变化率"""
        price_changes = np.zeros(len(close_prices))
        for i in range(1, len(close_prices)):
            price_changes[i] = close_prices[i]/close_prices[i-1] - 1
        return price_changes

    def _calculate_daily_returns(self, portfolio: pd.DataFrame) -> np.ndarray:
        """Calculate daily return rate changes"""
        # Get close prices and holdings data
        close_prices = portfolio['close'].values
        holdings = portfolio['holdings'].values
        
        # Calculate price changes
        price_changes = self._calculate_price_changes(close_prices)
        
        # Initialize daily returns array
        daily_returns = np.zeros(len(portfolio))
        
        # Calculate daily returns starting from second day
        for i in range(1, len(portfolio)):
            # If no position yesterday, return is 0
            if holdings[i-1] == 0:
                daily_returns[i] = 0
            # If had position yesterday, calculate return based on position direction
            else:
                daily_returns[i] = price_changes[i] * holdings[i-1]
        return daily_returns, price_changes

    def calculate_cumulative_return(self, portfolio: pd.DataFrame) -> np.ndarray:
        """
        Calculate cumulative returns for the portfolio.
        Returns an array where each element represents the cumulative return up to that point.
        Also updates the portfolio DataFrame with price_change, daily_return and cumulative_return columns.
        """
        daily_returns, price_changes = self._calculate_daily_returns(portfolio)
        cumulative_returns = np.zeros(len(daily_returns))
        
        # 计算每个时点的累积收益
        for i in range(len(daily_returns)):
            cumulative_returns[i] = np.sum(daily_returns[:i+1])
        
        # 更新portfolio DataFrame
        portfolio['price_change'] = price_changes
        portfolio['daily_return'] = daily_returns
        portfolio['cumulative_return'] = cumulative_returns
        
        return cumulative_returns
    

    
    def calculate_annualized_return(self, portfolio: pd.DataFrame) -> float:
        """
        计算投资组合的年化收益率
        
        Args:
            portfolio (pd.DataFrame): 包含交易记录的DataFrame
            
        Returns:
            float: 年化收益率
        """
        cumulative_returns = self.calculate_cumulative_return(portfolio)
        total_return = cumulative_returns[-1] if isinstance(cumulative_returns, (list, np.ndarray)) else cumulative_returns
        
        # 计算投资期间的总天数
        start_date = pd.to_datetime(portfolio['Date'].iloc[0])
        end_date = pd.to_datetime(portfolio['Date'].iloc[-1])
        days = (end_date - start_date).days
        
        # 如果投资期间小于1天,返回0
        if days < 1:
            return 0.0
            
        # 使用复利公式计算年化收益率
        # (1 + total_return)^(365/days) - 1
        annualized_return = (1 + total_return) ** (365.0/days) - 1
        
        return annualized_return




    def calculate_sharpe_ratio(self, portfolio: pd.DataFrame, risk_free_rate=0.00, periods_per_year=365) -> float:
        """Calculate Sharpe Ratio based on daily return rate changes."""
        try:
            daily_returns = self._calculate_daily_returns(portfolio)[0]
            
            # print("\n=== Returns Analysis ===")
            # print("Daily returns statistics:")
            # print(pd.Series(daily_returns).describe())
            # print(f"Cumulative return: {self._calculate_return_rates(portfolio)[-1]:.4f}")
            
            # 计算年化超额收益和波动率
            annual_excess_return = np.mean(daily_returns) * periods_per_year - risk_free_rate
            annual_volatility = np.std(daily_returns) * np.sqrt(periods_per_year)
            # print("\n=== Sharpe Ratio Components ===")
            # print(f"Annual excess return: {annual_excess_return:.4f}")
            # print(f"Annual volatility: {annual_volatility:.4f}")
            
            if annual_volatility == 0 or np.isnan(annual_volatility):
                return 0.0
                
            sharpe_ratio = annual_excess_return / annual_volatility
            return sharpe_ratio if not (np.isinf(sharpe_ratio) or np.isnan(sharpe_ratio)) else 0.0
            
        except Exception as e:
            logging.warning(f"Error calculating Sharpe ratio: {e}")
            return 0.0

    def calculate_max_drawdown(self, portfolio: pd.DataFrame) -> float:
        """
        Calculate maximum drawdown considering floating P&L.
        """
        try:
            # 计算累计收益率序列
            cumulative_returns = self.calculate_cumulative_return(portfolio)
            
            # 计算历史最高点
            historical_max = pd.Series(cumulative_returns).expanding().max()
            
            # 计算每个时点的回撤
            drawdowns = cumulative_returns - historical_max
            
            # 获取最大回撤
            max_drawdown = min(drawdowns)
            
            return max_drawdown if not np.isnan(max_drawdown) else 0.0
            
        except Exception as e:
            logging.warning(f"计算最大回撤时发生错误: {e}")
            return 0.0




    def calculate_sortino_ratio(self, portfolio: pd.DataFrame, risk_free_rate=0.00, periods_per_year=365) -> float:
        """Calculate Sortino Ratio using downside deviation."""
        try:
            daily_returns = self._calculate_daily_returns(portfolio)[0]
  
            
            annual_excess_return = np.mean(daily_returns) * periods_per_year - risk_free_rate
            
            downside_returns = daily_returns[daily_returns < 0]
            if len(downside_returns) == 0:
                return 0.0
                
            downside_std = np.sqrt(np.mean(downside_returns ** 2)) * np.sqrt(periods_per_year)
            
            # print("\n=== Sortino Ratio Components ===")
            # print(f"Annual excess return: {annual_excess_return:.4f}")
            # print(f"Downside volatility: {downside_std:.4f}")
            # print(f"Number of negative returns: {len(downside_returns)}")
            
            if downside_std == 0 or np.isnan(downside_std):
                return 0.0
                
            sortino_ratio = annual_excess_return / downside_std
            return sortino_ratio if not (np.isinf(sortino_ratio) or np.isnan(sortino_ratio)) else 0.0
            
        except Exception as e:
            logging.warning(f"Error calculating Sortino ratio: {e}")
            return 0.0

    def calculate_trade_count(self, portfolio: pd.DataFrame) -> int:
        """
        Calculate the actual number of trades by counting position changes.
        """
        # Get position changes
        position_changes = portfolio['holdings'].diff()
        
        # Count actual trades (when holdings actually change)
        trade_count = len(position_changes[position_changes != 0])-1
        
        return trade_count

    def calculate_performance_metrics(self, portfolio: pd.DataFrame) -> dict:
        """
        计算所有性能指标
        """
        try:
            total_return = self.calculate_cumulative_return(portfolio)[-1]
            annualized_return = self.calculate_annualized_return(portfolio)
            sharpe_ratio = self.calculate_sharpe_ratio(portfolio, periods_per_year=self.periods_per_year)
            sortino_ratio = self.calculate_sortino_ratio(portfolio, periods_per_year=self.periods_per_year)
            max_drawdown = self.calculate_max_drawdown(portfolio)
            trade_count = self.calculate_trade_count(portfolio)
            
            return {
                'Total Return': total_return,
                'Annualized Return': annualized_return,
                'Sharpe Ratio': sharpe_ratio,
                'Sortino Ratio': sortino_ratio,
                'Max Drawdown': max_drawdown,
                'Trade Count': trade_count
            }
        except Exception as e:
            logging.error(f"Error calculating performance metrics: {e}")
            return {}

    def plot_performance(self, portfolio: pd.DataFrame, data: pd.DataFrame):
        """
        Plot strategy performance and asset price with trade markers.
        Generates both HTML (interactive) and PNG (static) versions.
        
        Args:
            portfolio (pd.DataFrame): Portfolio data with holdings and returns
            data (pd.DataFrame): Price data
        """
        # Generate interactive HTML chart
        self._plot_interactive(portfolio, data)
        # Generate static PNG chart
        self._plot_static(portfolio, data)

    def _plot_interactive(self, portfolio: pd.DataFrame, data: pd.DataFrame):
        """Generate interactive HTML chart using pyecharts"""
        from pyecharts import options as opts
        from pyecharts.charts import Grid, Line, Scatter
        
        # 使用calculate_cumulative_return计算累计收益率
        cumulative_returns = self.calculate_cumulative_return(portfolio)
        
        # Create charts
        returns_chart = Line()
        price_chart = Line()
        scatter_buy = Scatter()
        scatter_sell = Scatter()
        
        # Prepare date axis data
        dates = [d.strftime("%Y-%m-%d") for d in portfolio['Date']]
        
        # Configure returns chart
        returns_chart.add_xaxis(xaxis_data=dates)
        returns_chart.add_yaxis(
            series_name="Strategy Returns",
            y_axis=[round(r * 100, 2) for r in cumulative_returns],  # 转换为百分比
            label_opts=opts.LabelOpts(is_show=False),
            linestyle_opts=opts.LineStyleOpts(width=2)
        )
        
        # Common x-axis options for both charts
        common_xaxis_opts = opts.AxisOpts(
            type_="category",
            axislabel_opts=opts.LabelOpts(rotate=45),
            is_scale=True
        )
        
        # Common datazoom options for both charts
        common_datazoom_opts = [
            opts.DataZoomOpts(
                type_="slider",
                xaxis_index=[0, 1],  # Link both x-axes
                range_start=0,
                range_end=100
            ),
            opts.DataZoomOpts(
                type_="inside",
                xaxis_index=[0, 1],  # Link both x-axes
                range_start=0,
                range_end=100
            )
        ]
        
        returns_chart.set_global_opts(
            title_opts=opts.TitleOpts(title="Strategy Performance"),
            xaxis_opts=common_xaxis_opts,
            yaxis_opts=opts.AxisOpts(
                name="Return Rate (%)",
                splitline_opts=opts.SplitLineOpts(is_show=True),
                is_scale=True
            ),
            datazoom_opts=common_datazoom_opts,
            tooltip_opts=opts.TooltipOpts(trigger="axis")
        )
        
        # Configure price chart
        price_chart.add_xaxis(xaxis_data=dates)
        price_chart.add_yaxis(
            series_name="Asset Price",
            y_axis=data['close'].tolist(),
            label_opts=opts.LabelOpts(is_show=False),
            linestyle_opts=opts.LineStyleOpts(width=2)
        )
        
        # Mark trade points
        holdings_changes = portfolio['holdings'].diff()
        
        # 为不同类型的交易准备数据
        open_long_dates, open_long_prices = [], []
        open_short_dates, open_short_prices = [], []
        close_long_dates, close_long_prices = [], []
        close_short_dates, close_short_prices = [], []
        switch_long_dates, switch_long_prices = [], []
        switch_short_dates, switch_short_prices = [], []
        
        for i in range(len(portfolio)):
            if i == 0:
                continue
                
            change = holdings_changes.iloc[i]
            if change != 0:
                date = portfolio.iloc[i]['Date'].strftime("%Y-%m-%d")
                price = data.iloc[i]['close']
                prev_holdings = portfolio.iloc[i-1]['holdings']
                curr_holdings = portfolio.iloc[i]['holdings']
                
                if prev_holdings == 0:  # 开仓
                    if curr_holdings > 0:
                        open_long_dates.append(date)
                        open_long_prices.append(price)
                    else:
                        open_short_dates.append(date)
                        open_short_prices.append(price)
                elif curr_holdings == 0:  # 平仓
                    if prev_holdings > 0:
                        close_long_dates.append(date)
                        close_long_prices.append(price)
                    else:
                        close_short_dates.append(date)
                        close_short_prices.append(price)
                else:  # 转换仓位
                    if curr_holdings > 0:
                        switch_long_dates.append(date)
                        switch_long_prices.append(price)
                    else:
                        switch_short_dates.append(date)
                        switch_short_prices.append(price)
        
        # Add trade markers for each type
        scatter_open_long = Scatter()
        scatter_open_long.add_xaxis(open_long_dates)
        scatter_open_long.add_yaxis(
            series_name="Open Long",
            y_axis=open_long_prices,
            symbol_size=10,
            label_opts=opts.LabelOpts(is_show=False),
            symbol="triangle",
            itemstyle_opts=opts.ItemStyleOpts(color="red")
        )
        
        scatter_open_short = Scatter()
        scatter_open_short.add_xaxis(open_short_dates)
        scatter_open_short.add_yaxis(
            series_name="Open Short",
            y_axis=open_short_prices,
            symbol_size=10,
            label_opts=opts.LabelOpts(is_show=False),
            symbol="triangle-down",
            itemstyle_opts=opts.ItemStyleOpts(color="blue")
        )
        
        scatter_close_long = Scatter()
        scatter_close_long.add_xaxis(close_long_dates)
        scatter_close_long.add_yaxis(
            series_name="Close Long",
            y_axis=close_long_prices,
            symbol_size=10,
            label_opts=opts.LabelOpts(is_show=False),
            symbol="triangle-down",
            itemstyle_opts=opts.ItemStyleOpts(color="black")
        )
        
        scatter_close_short = Scatter()
        scatter_close_short.add_xaxis(close_short_dates)
        scatter_close_short.add_yaxis(
            series_name="Close Short",
            y_axis=close_short_prices,
            symbol_size=10,
            label_opts=opts.LabelOpts(is_show=False),
            symbol="triangle",
            itemstyle_opts=opts.ItemStyleOpts(color="green")
        )
        
        scatter_switch_long = Scatter()
        scatter_switch_long.add_xaxis(switch_long_dates)
        scatter_switch_long.add_yaxis(
            series_name="Switch to Long",
            y_axis=switch_long_prices,
            symbol_size=10,
            label_opts=opts.LabelOpts(is_show=False),
            symbol="triangle",
            itemstyle_opts=opts.ItemStyleOpts(color="purple")
        )
        
        scatter_switch_short = Scatter()
        scatter_switch_short.add_xaxis(switch_short_dates)
        scatter_switch_short.add_yaxis(
            series_name="Switch to Short",
            y_axis=switch_short_prices,
            symbol_size=10,
            label_opts=opts.LabelOpts(is_show=False),
            symbol="triangle-down",
            itemstyle_opts=opts.ItemStyleOpts(color="orange")
        )
        
        # Overlap all scatter plots with price chart
        price_chart.overlap(scatter_open_long)
        price_chart.overlap(scatter_open_short)
        price_chart.overlap(scatter_close_long)
        price_chart.overlap(scatter_close_short)
        price_chart.overlap(scatter_switch_long)
        price_chart.overlap(scatter_switch_short)
        
        # Configure price chart options
        price_chart.set_global_opts(
            xaxis_opts=common_xaxis_opts,
            yaxis_opts=opts.AxisOpts(
                name="Price",
                splitline_opts=opts.SplitLineOpts(is_show=True),
                is_scale=True
            ),
            datazoom_opts=common_datazoom_opts,
            tooltip_opts=opts.TooltipOpts(trigger="axis")
        )
        
        # Create grid to combine charts
        grid = Grid(
            init_opts=opts.InitOpts(
                width="1200px",
                height="800px",
                animation_opts=opts.AnimationOpts(animation=False)
            )
        )
        
        # Add charts to grid
        grid.add(
            returns_chart,
            grid_opts=opts.GridOpts(pos_top="5%", height="45%")
        )
        grid.add(
            price_chart.overlap(scatter_buy).overlap(scatter_sell),
            grid_opts=opts.GridOpts(pos_bottom="8%", height="45%")
        )
        
        # Save to HTML file
        grid.render("strategy_performance.html")

    def _plot_static(self, portfolio: pd.DataFrame, data: pd.DataFrame):
        """Generate static PNG chart using matplotlib"""
        import matplotlib.pyplot as plt
        from matplotlib.gridspec import GridSpec
        
        # Create figure
        fig = plt.figure(figsize=(15, 10))
        gs = GridSpec(2, 1, height_ratios=[2, 1])
        
        # 使用calculate_cumulative_return计算累计收益率
        cumulative_returns = self.calculate_cumulative_return(portfolio)
        
        # Upper plot: Cumulative returns
        ax1 = fig.add_subplot(gs[0])
        ax1.plot(portfolio['Date'], cumulative_returns, 
                 label='Strategy Returns', color='blue')
        ax1.axhline(y=0, color='r', linestyle='--', alpha=0.3)
        ax1.set_title('Cumulative Returns')
        ax1.set_ylabel('Return Rate')
        ax1.grid(True)
        ax1.legend()
        
        # Format x-axis dates
        ax1.tick_params(axis='x', rotation=45)
        
        # Lower plot: Asset price
        ax2 = fig.add_subplot(gs[1])
        ax2.plot(data['Date'], data['close'], 
                 label='Asset Price', color='green')
        ax2.set_title('Asset Price')
        ax2.set_ylabel('Price')
        ax2.grid(True)
        
        # Format x-axis dates
        ax2.tick_params(axis='x', rotation=45)
        
        # Mark trade points based on holdings changes
        holdings_changes = portfolio['holdings'].diff()
        
        # Find trade points
        for i in range(len(portfolio)):
            if i == 0:
                continue
            
            change = holdings_changes.iloc[i]
            if change != 0:  # If there's a change in holdings
                price = data.iloc[i]['close']
                date = portfolio.iloc[i]['Date']
                prev_holdings = portfolio.iloc[i-1]['holdings']
                curr_holdings = portfolio.iloc[i]['holdings']
                
                # 区分不同类型的交易
                if prev_holdings == 0:  # 开仓
                    if curr_holdings > 0:
                        ax2.scatter(date, price, color='red', marker='^', s=100,
                                  label='Open Long' if 'Open Long' not in ax2.get_legend_handles_labels()[1] else "")
                    else:
                        ax2.scatter(date, price, color='blue', marker='v', s=100,
                                  label='Open Short' if 'Open Short' not in ax2.get_legend_handles_labels()[1] else "")
                elif curr_holdings == 0:  # 平仓
                    if prev_holdings > 0:
                        ax2.scatter(date, price, color='black', marker='v', s=100,
                                  label='Close Long' if 'Close Long' not in ax2.get_legend_handles_labels()[1] else "")
                    else:
                        ax2.scatter(date, price, color='green', marker='^', s=100,
                                  label='Close Short' if 'Close Short' not in ax2.get_legend_handles_labels()[1] else "")
                else:  # 转换仓位
                    if curr_holdings > 0:
                        ax2.scatter(date, price, color='purple', marker='^', s=100,
                                  label='Switch to Long' if 'Switch to Long' not in ax2.get_legend_handles_labels()[1] else "")
                    else:
                        ax2.scatter(date, price, color='orange', marker='v', s=100,
                                  label='Switch to Short' if 'Switch to Short' not in ax2.get_legend_handles_labels()[1] else "")
        
        ax2.legend()
        
        # Adjust layout
        plt.tight_layout()
        
        # Save plot
        plt.savefig('strategy_performance.png', dpi=300, bbox_inches='tight')
        plt.close()
# backtest/performance.py

import pandas as pd
import numpy as np
import logging

class PerformanceEvaluator:
    """
    绩效评估类，计算回测的各项绩效指标。
    """
    def __init__(self):
        pass

    def calculate_annualized_return(self, portfolio: pd.DataFrame) -> float:
        """
        計算年化收益率。
        
        收益率 = 最終累計收益 / 10000(初始投資)
        """
        INITIAL_INVESTMENT = 10000
        final_value = portfolio['total'].iloc[-1]  # 最終累計收益
        total_return = final_value / INITIAL_INVESTMENT  # 總收益率
        
        if 'Date' in portfolio.columns:
            start_date = pd.to_datetime(portfolio['Date'].iloc[0])
            end_date = pd.to_datetime(portfolio['Date'].iloc[-1])
            num_years = (end_date - start_date).days / 365.25
            
            if num_years > 0:
                if total_return > -1:  # 正常情况
                    annualized_return = (1 + total_return) ** (1 / num_years) - 1
                    return annualized_return
                else:  # 处理极端亏损情况
                    # 当亏损超过100%时，返回总亏损率
                    return total_return / num_years
            
        return float('nan')  # 如果無法計算則返回 NaN

    def _calculate_portfolio_values(self, portfolio: pd.DataFrame) -> np.ndarray:
        """计算每日投资组合市值"""
        INITIAL_INVESTMENT = 10000
        return np.where(
            portfolio['holdings'] == 0,
            portfolio['total'] + INITIAL_INVESTMENT,  # 无持仓
            np.where(
                portfolio['holdings'] > 0,
                portfolio['holdings'] * portfolio['close'] - portfolio['cost_basis'] + portfolio['total'] + INITIAL_INVESTMENT,  # 多单
                portfolio['holdings'] * portfolio['close'] + portfolio['cost_basis'] + portfolio['total'] + INITIAL_INVESTMENT  # 空单
            )
        )

    def calculate_sharpe_ratio(self, portfolio: pd.DataFrame, risk_free_rate=0.05, periods_per_year=365) -> float:
        """
        Calculate Sharpe Ratio.
        
        Args:
            portfolio: Portfolio data
            risk_free_rate: Annual risk-free rate
            periods_per_year: Number of periods in a year (e.g., 365 for daily, 52 for weekly, 12 for monthly)
        """
        try:
            daily_value = self._calculate_portfolio_values(portfolio)
            daily_returns = pd.Series(daily_value).pct_change()
            daily_returns.iloc[0] = 0
            

            
           
            annual_excess_return = daily_returns.mean() * periods_per_year - risk_free_rate
            annual_volatility = daily_returns.std() * np.sqrt(periods_per_year)
            
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
            # 计算包含浮动盈亏的每日市值
            daily_value = self._calculate_portfolio_values(portfolio)
            
            # 计算历史最高值
            running_max = np.maximum.accumulate(daily_value)
            
            # 计算相对于历史最高值的回撤
            drawdowns = (daily_value - running_max) / running_max
            
            # 获取最大回撤
            max_drawdown = drawdowns.min()
            
            return max_drawdown if not np.isnan(max_drawdown) else 0.0
            
        except Exception as e:
            logging.warning(f"Error calculating max drawdown: {e}")
            return 0.0

    def calculate_total_return(self, portfolio: pd.DataFrame) -> float:
        """
        計算總回報率。
        
        總回報率 = 最終累計收益 / 初始投資金額
        """
        INITIAL_INVESTMENT = 10000
        final_value = portfolio['total'].iloc[-1]  # 最終累計收益
        total_return = final_value / INITIAL_INVESTMENT  # 總回報率
        return total_return

    def calculate_sortino_ratio(self, portfolio: pd.DataFrame, risk_free_rate=0.05, periods_per_year=365) -> float:
        """
        Calculate Sortino Ratio using downside deviation.
        """
        try:
            daily_value = self._calculate_portfolio_values(portfolio)
            daily_returns = pd.Series(daily_value).pct_change()
            daily_returns.iloc[0] = 0
            
            # 转换年化无风险利率为对应周期的利率
            annual_excess_return = daily_returns.mean() * periods_per_year - risk_free_rate
            
            # 计算下行波动率（只考虑负收益）
            downside_returns = daily_returns[daily_returns < 0]
            if len(downside_returns) == 0:
                return 0.0
                
            downside_std = np.sqrt(np.mean(downside_returns ** 2)) * np.sqrt(periods_per_year)
            
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

    def calculate_performance_metrics(self, portfolio: pd.DataFrame, periods_per_year=365) -> dict:
        """
        Calculate all performance metrics.
        
        Args:
            portfolio: Portfolio data
            periods_per_year: Number of periods in a year (e.g., 365 for daily, 52 for weekly, 12 for monthly)
        """
        metrics = {
            'Total Return': self.calculate_total_return(portfolio),
            'Annualized Return': self.calculate_annualized_return(portfolio),
            'Sharpe Ratio': self.calculate_sharpe_ratio(portfolio, periods_per_year=periods_per_year),
            'Sortino Ratio': self.calculate_sortino_ratio(portfolio, periods_per_year=periods_per_year),
            'Max Drawdown': self.calculate_max_drawdown(portfolio),
            'Number of Trades': self.calculate_trade_count(portfolio)
        }
        
        # Add error checking for metrics
        for key in metrics:
            if np.isnan(metrics[key]) or np.isinf(metrics[key]):
                metrics[key] = 0.0
                
        return metrics

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
        
        # 计算每日总市值（包括浮动盈亏）
        INITIAL_INVESTMENT = 10000
        daily_value = np.where(
            portfolio['holdings'] == 0,
            portfolio['total'] + INITIAL_INVESTMENT,  # 无持仓
            np.where(
                portfolio['holdings'] > 0,
                portfolio['holdings'] * portfolio['close'] - portfolio['cost_basis'] + portfolio['total'] + INITIAL_INVESTMENT,  # 多单
                portfolio['holdings'] * portfolio['close'] + portfolio['cost_basis'] + portfolio['total'] + INITIAL_INVESTMENT  # 空单
            )
        )
        
        # 计算累计收益率
        portfolio['return_rate'] = (daily_value - INITIAL_INVESTMENT) / INITIAL_INVESTMENT
        
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
            y_axis=[round(r * 100, 2) for r in portfolio['return_rate']],
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
        
        # 计算每日总市值（包括浮动盈亏）
        INITIAL_INVESTMENT = 10000
        daily_value = np.where(
            portfolio['holdings'] == 0,
            portfolio['total'] + INITIAL_INVESTMENT,  # 无持仓
            np.where(
                portfolio['holdings'] > 0,
                portfolio['holdings'] * portfolio['close'] - portfolio['cost_basis'] + portfolio['total'] + INITIAL_INVESTMENT,  # 多单
                portfolio['holdings'] * portfolio['close'] + portfolio['cost_basis'] + portfolio['total'] + INITIAL_INVESTMENT  # 空单
            )
        )
        
        # 计算累计收益率
        portfolio['return_rate'] = (daily_value - INITIAL_INVESTMENT) / INITIAL_INVESTMENT
        
        # Upper plot: Cumulative returns
        ax1 = fig.add_subplot(gs[0])
        ax1.plot(portfolio['Date'], portfolio['return_rate'], 
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
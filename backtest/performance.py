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
            
            if num_years > 0 and total_return > -1:  # 避免負值進行指數運算
                annualized_return = (1 + total_return) ** (1 / num_years) - 1
                return annualized_return
            
        return float('nan')  # 如果無法計算則返回 NaN

    def calculate_sharpe_ratio(self, portfolio: pd.DataFrame, risk_free_rate=0.05) -> float:
        """
        Calculate Sharpe Ratio using daily returns for crypto markets (365 trading days).
        
        Args:
            portfolio (pd.DataFrame): Portfolio data with 'total' column
            risk_free_rate (float): Annual risk-free rate, defaults to 0.05 (5%)
            
        Returns:
            float: Sharpe ratio or 0.0 if calculation fails
        """
        try:
            # Calculate daily returns based on total portfolio value
            daily_returns = portfolio['total'].diff() / 10000  # Normalize by initial investment
            daily_returns = daily_returns.dropna()
            
            if len(daily_returns) < 2:  # Need at least 2 points for std calculation
                return 0.0
                
            # Convert annual risk-free rate to daily (using 365 days for crypto markets)
            daily_rf_rate = (1 + risk_free_rate) ** (1/365) - 1
            
            # Calculate annualized metrics (using 365 days for crypto markets)
            excess_returns = daily_returns - daily_rf_rate
            annual_excess_return = excess_returns.mean() * 365
            annual_volatility = daily_returns.std() * np.sqrt(365)
            
            # Check for valid volatility
            if annual_volatility == 0 or np.isnan(annual_volatility):
                return 0.0
                
            # Calculate Sharpe ratio
            sharpe_ratio = annual_excess_return / annual_volatility
            
            # Validate result
            if np.isinf(sharpe_ratio) or np.isnan(sharpe_ratio):
                return 0.0
                
            return sharpe_ratio
            
        except Exception as e:
            logging.warning(f"Error calculating Sharpe ratio: {e}")
            return 0.0

    def calculate_max_drawdown(self, portfolio: pd.DataFrame) -> float:
        """
        Calculate maximum drawdown from peak, using initial capital as base.
        Initial capital is set to 10000.
        """
        # Calculate returns relative to initial capital
        returns = portfolio['total'] / 10000 - 1
        
        # Calculate running maximum
        running_max = np.maximum.accumulate(returns)
        
        # Calculate drawdowns
        drawdowns = returns - running_max
        
        # Get maximum drawdown
        max_drawdown = drawdowns.min()
        
        return max_drawdown

    def calculate_total_return(self, portfolio: pd.DataFrame) -> float:
        """
        計算總回報率。
        
        總回報率 = 最終累計收益 / 初始投資金額
        """
        INITIAL_INVESTMENT = 10000
        final_value = portfolio['total'].iloc[-1]  # 最終累計收益
        total_return = final_value / INITIAL_INVESTMENT  # 總回報率
        return total_return

    def calculate_sortino_ratio(self, portfolio: pd.DataFrame, risk_free_rate=0.05, target_return=0.0) -> float:
        """
        Calculate Sortino Ratio using daily returns.
        
        Parameters:
            portfolio (pd.DataFrame): Portfolio data
            risk_free_rate (float): Annual risk-free rate
            target_return (float): Minimum acceptable return
        """
        # Calculate daily returns
        daily_returns = portfolio['total'].diff() / 10000
        daily_returns = daily_returns.dropna()
        
        if len(daily_returns) > 1:
            # Calculate excess returns
            excess_returns = daily_returns - risk_free_rate / 365
            
            # Calculate downside returns
            downside_returns = excess_returns[excess_returns < target_return]
            
            if len(downside_returns) > 0:
                # Calculate downside deviation
                downside_std = np.sqrt(np.mean(downside_returns ** 2))
                
                if downside_std != 0:
                    # Calculate Sortino ratio
                    sortino_ratio = np.sqrt(365) * excess_returns.mean() / downside_std
                    return sortino_ratio
                
        return 0.0

    def calculate_trade_count(self, portfolio: pd.DataFrame) -> int:
        """
        Calculate the actual number of trades by counting position changes.
        """
        # Get position changes
        position_changes = portfolio['holdings'].diff()
        
        # Count actual trades (when holdings actually change)
        trade_count = len(position_changes[position_changes != 0])
        
        return trade_count

    def calculate_performance_metrics(self, portfolio: pd.DataFrame) -> dict:
        """
        Calculate all performance metrics.
        """
        # Ensure portfolio has data
        if len(portfolio) == 0:
            return {
                'Total Return': 0.0,
                'Annualized Return': 0.0,
                'Sharpe Ratio': 0.0,
                'Sortino Ratio': 0.0,
                'Max Drawdown': 0.0,
                'Number of Trades': 0
            }

        metrics = {
            'Total Return': self.calculate_total_return(portfolio),
            'Annualized Return': self.calculate_annualized_return(portfolio),
            'Sharpe Ratio': self.calculate_sharpe_ratio(portfolio),
            'Sortino Ratio': self.calculate_sortino_ratio(portfolio),
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
        
        # Calculate return rate series
        INITIAL_INVESTMENT = 10000
        portfolio['return_rate'] = portfolio['total'] / INITIAL_INVESTMENT
        
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
        buy_dates, buy_prices = [], []
        sell_dates, sell_prices = [], []
        
        for i in range(len(portfolio)):
            if i == 0:
                continue
            
            change = holdings_changes.iloc[i]
            if change != 0:
                date = portfolio.iloc[i]['Date'].strftime("%Y-%m-%d")
                price = data.iloc[i]['close']
                
                if change > 0:  # Buy point
                    buy_dates.append(date)
                    buy_prices.append(price)
                else:  # Sell point
                    sell_dates.append(date)
                    sell_prices.append(price)
        
        # Add trade markers
        scatter_buy.add_xaxis(buy_dates)
        scatter_buy.add_yaxis(
            series_name="Buy",
            y_axis=buy_prices,
            symbol_size=10,
            label_opts=opts.LabelOpts(is_show=False),
            symbol="triangle",
            itemstyle_opts=opts.ItemStyleOpts(color="red")
        )
        
        scatter_sell.add_xaxis(sell_dates)
        scatter_sell.add_yaxis(
            series_name="Sell",
            y_axis=sell_prices,
            symbol_size=10,
            label_opts=opts.LabelOpts(is_show=False),
            symbol="triangle-down",
            itemstyle_opts=opts.ItemStyleOpts(color="black")
        )
        
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
        
        # Calculate return rate series
        INITIAL_INVESTMENT = 10000
        portfolio['return_rate'] = portfolio['total'] / INITIAL_INVESTMENT
        
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
                
                if change > 0:  # Buy point
                    ax2.scatter(date, price, color='red', 
                              marker='^', s=100, 
                              label='Buy' if 'Buy' not in ax2.get_legend_handles_labels()[1] else "")
                else:  # Sell point
                    ax2.scatter(date, price, color='black', 
                              marker='v', s=100, 
                              label='Sell' if 'Sell' not in ax2.get_legend_handles_labels()[1] else "")
        
        ax2.legend()
        
        # Adjust layout
        plt.tight_layout()
        
        # Save plot
        plt.savefig('strategy_performance.png', dpi=300, bbox_inches='tight')
        plt.close()
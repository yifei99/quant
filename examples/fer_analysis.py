import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from data.data_loader import DataLoader
from factors.factor_definitions import FractalEfficiencyRatio
import logging

def setup_logger():
    """设置日志配置"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    return logging.getLogger(__name__)

def plot_fer_analysis(data: pd.DataFrame, fer_values: pd.Series, direction_values: pd.Series, window: int):
    """Plot FER analysis charts"""
    plt.figure(figsize=(15, 10))
    
    # Price plot (top subplot)
    ax1 = plt.subplot(211)
    
    # Create masks for different trend states
    uptrend_mask = (direction_values == 1)
    downtrend_mask = (direction_values == -1)
    ranging_mask = (direction_values == 0)
    
    # Plot price lines with different colors for different trends
    dates = data.index
    prices = data['close']
    
    # Plot each trend segment with different colors
    for i in range(len(dates)-1):
        if uptrend_mask.iloc[i]:
            color = 'red'
        elif downtrend_mask.iloc[i]:
            color = 'green'
        else:  # ranging
            color = 'gray'
            
        ax1.plot(dates[i:i+2], prices.iloc[i:i+2], 
                color=color, linewidth=1.5)
    
    ax1.set_title(f'Price Analysis (Window: {window}h)', fontsize=12)
    ax1.set_ylabel('Price', fontsize=10)
    ax1.grid(True, alpha=0.3)
    
    # Add legends for price plot
    from matplotlib.lines import Line2D
    custom_lines = [
        Line2D([0], [0], color='red', lw=2),
        Line2D([0], [0], color='green', lw=2),
        Line2D([0], [0], color='gray', lw=2)
    ]
    ax1.legend(custom_lines, 
              ['Uptrend', 'Downtrend', 'Ranging'],
              loc='upper left', fontsize=10)
    
    # FER plot (bottom subplot)
    ax2 = plt.subplot(212)
    ax2.plot(dates, fer_values, color='blue', linewidth=1.5, label='FER')
    ax2.axhline(y=1.0, color='black', linestyle='--', alpha=0.3)
    ax2.axhline(y=3.0, color='red', linestyle='--', alpha=0.3)
    ax2.set_title('FER Analysis', fontsize=12)
    ax2.set_ylabel('FER Value', fontsize=10)
    ax2.grid(True, alpha=0.3)
    ax2.legend(fontsize=10)
    
    plt.tight_layout()
    plt.show()

def analyze_fer(data: pd.DataFrame, 
                windows=[24, 72, 168],
                trend_upper=3.0):         # 简化参数
    """Analyze FER for different time windows"""
    logger = logging.getLogger(__name__)
    
    for window in windows:
        logger.info(f"\nAnalyzing window size: {window}h")
        
        # Calculate FER
        fer_factor = FractalEfficiencyRatio(
            window=window,
            trend_upper=trend_upper
        )
        fer_values, direction_values = fer_factor.calculate(data)
        
        # Calculate statistics
        stats = {
            'Mean': fer_values.mean(),
            'Std': fer_values.std(),
            'Min': fer_values.min(),
            'Max': fer_values.max(),
            'Median': fer_values.median()
        }
        
        # Calculate trend distribution
        latest_state = fer_factor.get_trend_state(fer_values.iloc[-1], direction_values.iloc[-1])
        trend_states = [
            fer_factor.get_trend_state(fer, dir_) 
            for fer, dir_ in zip(fer_values, direction_values)
        ]
        state_distribution = pd.Series(trend_states).value_counts()
        
        # Output analysis results
        logger.info("\nFER Statistics:")
        for stat_name, stat_value in stats.items():
            logger.info(f"{stat_name}: {stat_value:.4f}")
        
        logger.info("\nTrend State Distribution:")
        for state, count in state_distribution.items():
            percentage = count / len(trend_states) * 100
            logger.info(f"{state}: {percentage:.2f}%")
        
        logger.info(f"\nCurrent Trend State: {latest_state}")
        
        # Plot analysis charts
        plot_fer_analysis(data, fer_values, direction_values, window)

def main():
    """主函数"""
    logger = setup_logger()
    logger.info("开始FER分析...")
    
    # 加载数据
    data_loader = DataLoader("../dataset")
    data = data_loader.load_data(
        exchange='binance',
        symbol='BTCUSDT',
        interval='1h',
        start_date='2021-01-01',
        end_date='2024-12-31',
        data_type='spot'
    )
    
    # 处理时间戳和数据选择
    if data['timestamp_start'].max() > 2e10:
        data['Date'] = pd.to_datetime(data['timestamp_start'], unit='ms')
    else:
        data['Date'] = pd.to_datetime(data['timestamp_start'], unit='s')
    
    # 选择必要的列并设置索引
    required_columns = ['Date', 'close', 'high', 'low', 'volume']
    data = data[required_columns].copy()
    data.set_index('Date', inplace=True)

    # 分析FER
    analyze_fer(
        data,
        windows=[168],  # 1天、3天、7天
        trend_upper=7.0
    )
    
    logger.info("FER分析完成")

if __name__ == "__main__":
    main() 
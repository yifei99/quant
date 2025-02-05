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

def plot_fer_analysis(data: pd.DataFrame, signals: pd.Series, window: int):
    """Plot FER analysis charts"""
    plt.figure(figsize=(12, 5))
    
    # Price plot (top subplot)
    ax1 = plt.subplot(211)
    
    # Create masks for different trend states
    uptrend_mask = (signals == 1)
    downtrend_mask = (signals == -1)
    ranging_mask = (signals == 0)
    
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
    
    plt.show()

def analyze_fer(data: pd.DataFrame, 
                windows=[24, 72, 168],
                trend_upper=3.0):
    """Analyze FER for different time windows"""
    logger = logging.getLogger(__name__)
    
    for window in windows:
        logger.info(f"\nAnalyzing window size: {window}h")
        
        # Calculate FER
        fer_factor = FractalEfficiencyRatio(
            window=window,
            trend_upper=trend_upper
        )
        signals = fer_factor.calculate(data)
        
        # Calculate signal distribution
        signal_counts = signals.value_counts()
        total_signals = len(signals)
        
        # Output analysis results
        logger.info("\nSignal Distribution:")
        for signal, count in signal_counts.items():
            signal_name = {
                1: "Uptrend",
                -1: "Downtrend",
                0: "Ranging"
            }.get(signal, "Unknown")
            percentage = count / total_signals * 100
            logger.info(f"{signal_name}: {percentage:.2f}%")
        
        # Plot analysis charts
        plot_fer_analysis(data, signals, window)

def main():
    """主函数"""
    logger = setup_logger()
    logger.info("开始FER分析...")
    
    # 加载数据
    data_loader = DataLoader("../dataset")
    data = data_loader.load_data(
        exchange='binance',
        symbol='SOLUSDT',
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
        windows=[394],  
        trend_upper=21.9
    )
    
    logger.info("FER分析完成")

if __name__ == "__main__":
    main() 
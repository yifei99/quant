import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import logging
from data.data_downloader import DataDownloader

def main():
    # 设置日志
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    logger = logging.getLogger(__name__)

    # 定义下载配置
    download_configs = {
        'datasets': [
            # 现货数据
            {
                'exchange': 'binance',
                'symbol': 'BTCUSDT',
                'interval': '1h',
                'start_date': '2021-01-01',  
                'end_date': '2024-12-31',
                'data_type': 'spot',
                'data_folder': '../dataset/binance'
            },
            {
                'exchange': 'binance',
                'symbol': 'BTCUSDT',
                'interval': '1m',
                'start_date': '2021-01-01',  
                'end_date': '2024-12-31',
                'data_type': 'spot',
                'data_folder': '../dataset/binance'
            },
            {
                'exchange': 'binance',
                'symbol': 'BTCUSDT',
                'interval': '5m',
                'start_date': '2021-01-01',  
                'end_date': '2024-12-31',
                'data_type': 'spot',
                'data_folder': '../dataset/binance'
            },
            {
                'exchange': 'binance',
                'symbol': 'BTCUSDT',
                'interval': '15m',
                'start_date': '2021-01-01',  
                'end_date': '2024-12-31',
                'data_type': 'spot',
                'data_folder': '../dataset/binance'
            },
            {
                'exchange': 'binance',
                'symbol': 'BTCUSDT',
                'interval': '30m',
                'start_date': '2021-01-01',  
                'end_date': '2024-12-31',
                'data_type': 'spot',
                'data_folder': '../dataset/binance'
            },
            {
                'exchange': 'binance',
                'symbol': 'BTCUSDT',
                'interval': '2h',
                'start_date': '2021-01-01',  
                'end_date': '2024-12-31',
                'data_type': 'spot',
                'data_folder': '../dataset/binance'
            },
            {
                'exchange': 'binance',
                'symbol': 'BTCUSDT',
                'interval': '4h',
                'start_date': '2021-01-01',  
                'end_date': '2024-12-31',
                'data_type': 'spot',
                'data_folder': '../dataset/binance'
            },
            {
                'exchange': 'binance',
                'symbol': 'BTCUSDT',
                'interval': '6h',
                'start_date': '2021-01-01',  
                'end_date': '2024-12-31',
                'data_type': 'spot',
                'data_folder': '../dataset/binance'
            },
            {
                'exchange': 'binance',
                'symbol': 'BTCUSDT',
                'interval': '8h',
                'start_date': '2021-01-01',  
                'end_date': '2024-12-31',
                'data_type': 'spot',
                'data_folder': '../dataset/binance'
            },
            {
                'exchange': 'binance',
                'symbol': 'BTCUSDT',
                'interval': '12h',
                'start_date': '2021-01-01',  
                'end_date': '2024-12-31',
                'data_type': 'spot',
                'data_folder': '../dataset/binance'
            },
            {
                'exchange': 'binance',
                'symbol': 'ETHUSDT',
                'interval': '1d',
                'start_date': '2021-01-01',  
                'end_date': '2024-12-31',
                'data_type': 'spot',
                'data_folder': '../dataset/binance'
            },
            {
                'exchange': 'binance',
                'symbol': 'ETHUSDT',
                'interval': '1h',
                'start_date': '2021-01-01',  
                'end_date': '2024-12-31',
                'data_type': 'spot',
                'data_folder': '../dataset/binance'
            },
            {
                'exchange': 'binance',
                'symbol': 'ETHUSDT',
                'interval': '1m',
                'start_date': '2021-01-01',  
                'end_date': '2024-12-31',
                'data_type': 'spot',
                'data_folder': '../dataset/binance'
            },
            {
                'exchange': 'binance',
                'symbol': 'ETHUSDT',
                'interval': '5m',
                'start_date': '2021-01-01',  
                'end_date': '2024-12-31',
                'data_type': 'spot',
                'data_folder': '../dataset/binance'
            },
            {
                'exchange': 'binance',
                'symbol': 'ETHUSDT',
                'interval': '15m',
                'start_date': '2021-01-01',  
                'end_date': '2024-12-31',
                'data_type': 'spot',
                'data_folder': '../dataset/binance'
            },
            {
                'exchange': 'binance',
                'symbol': 'ETHUSDT',
                'interval': '30m',
                'start_date': '2021-01-01',  
                'end_date': '2024-12-31',
                'data_type': 'spot',
                'data_folder': '../dataset/binance'
            },
            {
                'exchange': 'binance',
                'symbol': 'ETHUSDT',
                'interval': '2h',
                'start_date': '2021-01-01',  
                'end_date': '2024-12-31',
                'data_type': 'spot',
                'data_folder': '../dataset/binance'
            },
            {
                'exchange': 'binance',
                'symbol': 'ETHUSDT',
                'interval': '4h',
                'start_date': '2021-01-01',  
                'end_date': '2024-12-31',
                'data_type': 'spot',
                'data_folder': '../dataset/binance'
            },
            {
                'exchange': 'binance',
                'symbol': 'ETHUSDT',
                'interval': '6h',
                'start_date': '2021-01-01',  
                'end_date': '2024-12-31',
                'data_type': 'spot',
                'data_folder': '../dataset/binance'
            },
            {
                'exchange': 'binance',
                'symbol': 'ETHUSDT',
                'interval': '8h',
                'start_date': '2021-01-01',  
                'end_date': '2024-12-31',
                'data_type': 'spot',
                'data_folder': '../dataset/binance'
            },
            {
                'exchange': 'binance',
                'symbol': 'ETHUSDT',
                'interval': '12h',
                'start_date': '2021-01-01',  
                'end_date': '2024-12-31',
                'data_type': 'spot',
                'data_folder': '../dataset/binance'
            },
            {
                'exchange': 'binance',
                'symbol': 'SOLUSDT',
                'interval': '1d',
                'start_date': '2021-01-01',  
                'end_date': '2024-12-31',
                'data_type': 'spot',
                'data_folder': '../dataset/binance'
            },
            {
                'exchange': 'binance',
                'symbol': 'SOLUSDT',
                'interval': '1h',
                'start_date': '2021-01-01',  
                'end_date': '2024-12-31',
                'data_type': 'spot',
                'data_folder': '../dataset/binance'
            },
            {
                'exchange': 'binance',
                'symbol': 'SOLUSDT',
                'interval': '1m',
                'start_date': '2021-01-01',  
                'end_date': '2024-12-31',
                'data_type': 'spot',
                'data_folder': '../dataset/binance'
            },
            {
                'exchange': 'binance',
                'symbol': 'SOLUSDT',
                'interval': '5m',
                'start_date': '2021-01-01',  
                'end_date': '2024-12-31',
                'data_type': 'spot',
                'data_folder': '../dataset/binance'
            },
            {
                'exchange': 'binance',
                'symbol': 'SOLUSDT',
                'interval': '15m',
                'start_date': '2021-01-01',  
                'end_date': '2024-12-31',
                'data_type': 'spot',
                'data_folder': '../dataset/binance'
            },
            {
                'exchange': 'binance',
                'symbol': 'SOLUSDT',
                'interval': '30m',
                'start_date': '2021-01-01',  
                'end_date': '2024-12-31',
                'data_type': 'spot',
                'data_folder': '../dataset/binance'
            },
            {
                'exchange': 'binance',
                'symbol': 'SOLUSDT',
                'interval': '2h',
                'start_date': '2021-01-01',  
                'end_date': '2024-12-31',
                'data_type': 'spot',
                'data_folder': '../dataset/binance'
            },
            {
                'exchange': 'binance',
                'symbol': 'SOLUSDT',
                'interval': '4h',
                'start_date': '2021-01-01',  
                'end_date': '2024-12-31',
                'data_type': 'spot',
                'data_folder': '../dataset/binance'
            },
            {
                'exchange': 'binance',
                'symbol': 'SOLUSDT',
                'interval': '12h',
                'start_date': '2021-01-01',  
                'end_date': '2024-12-31',
                'data_type': 'spot',
                'data_folder': '../dataset/binance'
            },
            {
                'exchange': 'binance',
                'symbol': 'SOLUSDT',
                'interval': '6h',
                'start_date': '2021-01-01',  
                'end_date': '2024-12-31',
                'data_type': 'spot',
                'data_folder': '../dataset/binance'
            },
            {
                'exchange': 'binance',
                'symbol': 'SOLUSDT',
                'interval': '8h',
                'start_date': '2021-01-01',  
                'end_date': '2024-12-31',
                'data_type': 'spot',
                'data_folder': '../dataset/binance'
            }
        ]
    }

    # 遍历所有配置并下载
    for config in download_configs['datasets']:
        try:
            logger.info(f"Starting download for: {config['symbol']} "
                       f"({config['data_type']}) - {config['interval']}")
            
            # 创建下载器实例
            downloader = DataDownloader(
                symbol=config['symbol'],
                interval=config['interval'],
                start_date=config['start_date'],
                end_date=config['end_date'],
                data_folder=config['data_folder'],
                data_type=config['data_type'],
                exchange=config['exchange']
            )
            
            # 下载并处理数据
            data = downloader.fetch_and_process_data()
            
            logger.info(f"Successfully downloaded and processed data for {config['symbol']}")
            
        except Exception as e:
            logger.error(f"Error processing {config['symbol']}: {str(e)}")
            logger.error("Full error:", exc_info=True)
            continue  # 继续下一个配置
    
    logger.info("All downloads completed")

if __name__ == "__main__":
    main()
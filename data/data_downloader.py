import os
import requests
from datetime import datetime, timedelta
import zipfile
import pandas as pd
import concurrent.futures
import logging
from tqdm import tqdm

class DataDownloader:
    def __init__(self, symbol, interval, start_date, end_date, data_folder, data_type="spot", exchange="binance"):
        """
        Initialize DataDownloader instance.
        
        Args:
            symbol (str): Trading pair symbol, e.g., "BTCUSDT"
            interval (str): Kline interval, e.g., "1m", "5m", "1h", "1d"
            start_date (str): Start date in "YYYY-MM-DD" format
            end_date (str): End date in "YYYY-MM-DD" format
            data_folder (str): Data storage directory, e.g., "./dataset/binance"
            data_type (str): Data type, "spot", "futures" or "metrics". Default is "spot"
            exchange (str): Exchange name, default is "binance"
        """
        self.symbol = symbol
        self.interval = interval
        self.start_date = start_date
        self.end_date = end_date
        self.data_folder = data_folder
        self.data_type = data_type
        self.exchange = exchange
        self.download_dir = "downloads"
        
        # 根据数据类型设置不同的文件夹路径
        if data_type == "metrics":
            self.interval_folder = os.path.join(data_folder, symbol, data_type)
        else:
            self.interval_folder = os.path.join(data_folder, symbol, data_type, interval)
            
        self.create_dir(self.download_dir)
        self.create_dir(self.interval_folder)
        
        self.logger = logging.getLogger(__name__)
        self.max_workers = 6

    def create_dir(self, path):
        """Create directory if not exists"""
        if not os.path.exists(path):
            os.makedirs(path)

    def download_file(self, url, dest_folder):
        """优化下载函数，添加重试机制"""
        max_retries = 3
        for attempt in range(max_retries):
            try:
                response = requests.get(url, timeout=10)
                if response.status_code == 200:
                    file_name = url.split("/")[-1]
                    file_path = os.path.join(dest_folder, file_name)
                    with open(file_path, 'wb') as file:
                        file.write(response.content)
                    return file_path
                elif response.status_code == 404:
                    return None
            except (requests.RequestException, TimeoutError) as e:
                if attempt == max_retries - 1:
                    self.logger.error(f"Failed to download {url} after {max_retries} attempts: {e}")
                    return None
                continue
        return None

    def process_zip_file(self, zip_file_path, extract_to_folder):
        """优化解压和处理函数"""
        try:
            with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
                # 直接读取 zip 中的 CSV 文件内容而不解压
                csv_filename = zip_ref.namelist()[0]
                with zip_ref.open(csv_filename) as csv_file:
                    df = pd.read_csv(csv_file, header=None)
                    
            # 删除 zip 文件
            os.remove(zip_file_path)
            return df
        except Exception as e:
            self.logger.error(f"Error processing zip file {zip_file_path}: {e}")
            return None

    def download_and_process_date(self, date):
        """单个日期的下载和处理"""
        date_str = date.strftime("%Y-%m-%d")
        base_url = "https://data.binance.vision/data"
        
        # 根据数据类型构建不同的URL
        if self.data_type == "metrics":
            url = f"{base_url}/futures/um/daily/metrics/{self.symbol}/{self.symbol}-metrics-{date_str}.zip"
        else:
            market_type = "spot" if self.data_type == "spot" else "futures/um"
            url = f"{base_url}/{market_type}/daily/klines/{self.symbol}/{self.interval}/{self.symbol}-{self.interval}-{date_str}.zip"
        
        zip_file_path = self.download_file(url, self.download_dir)
        if zip_file_path:
            return self.process_zip_file(zip_file_path, self.interval_folder)
        return None

    def download_binance_data(self):
        """并行下载和处理数据"""
        start_date = datetime.strptime(self.start_date, "%Y-%m-%d")
        end_date = datetime.strptime(self.end_date, "%Y-%m-%d")
        date_range = [start_date + timedelta(days=x) for x in range((end_date - start_date).days + 1)]
        
        all_data = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # 使用 tqdm 显示进度
            futures = {executor.submit(self.download_and_process_date, date): date for date in date_range}
            for future in tqdm(concurrent.futures.as_completed(futures), total=len(date_range), 
                             desc=f"Downloading {self.symbol} {self.interval}"):
                date = futures[future]
                try:
                    df = future.result()
                    if df is not None:
                        all_data.append(df)
                except Exception as e:
                    self.logger.error(f"Error processing date {date}: {e}")
        
        return all_data

    def download_bybit_data(self):
        """
        下載 Bybit 數據 (未來可擴展)
        """
        # 這是未來的擴展，這裡需要根據 Bybit API 格式來實現
        pass

    def process_data(self, all_data):
        """优化数据处理"""
        if not all_data:
            return None
            
        final_data = pd.concat(all_data, ignore_index=True)
        
        # 根据数据类型设置不同的列名
        if self.data_type == "metrics":
            # 设置正确的metrics数据列名
            final_data.columns = [
                "create_time",
                "symbol",
                "sum_open_interest",
                "sum_open_interest_value",
                "count_toptrader_long_short_ratio",
                "sum_toptrader_long_short_ratio",
                "count_long_short_ratio",
                "sum_taker_long_short_vol_ratio"
            ]
            
            # 删除可能的header行（包含列名的行）
            final_data = final_data[final_data['create_time'] != 'create_time']
            
            # 确保数据类型正确
            numeric_columns = [
                'sum_open_interest',
                'sum_open_interest_value',
                'count_toptrader_long_short_ratio',
                'sum_toptrader_long_short_ratio',
                'count_long_short_ratio',
                'sum_taker_long_short_vol_ratio'
            ]
            
            # 转换数据类型
            for col in numeric_columns:
                final_data[col] = pd.to_numeric(final_data[col], errors='coerce')
                
            # 转换时间戳
            final_data['create_time'] = pd.to_datetime(final_data['create_time'])
            
            # 按时间排序并重置索引
            final_data = final_data.sort_values(by="create_time").reset_index(drop=True)
            
        else:
            final_data.columns = [
                "timestamp_start", "open", "high", "low", "close", "volume",
                "timestamp_end", "quote_asset_volume", "number_of_trades", 
                "taker_buy_base", "taker_buy_quote", "ignore"
            ]
            final_data = final_data.sort_values(by="timestamp_start").reset_index(drop=True)
        
        # 保存为 HDF5 文件
        hdf5_file_name = f"{self.symbol}_{self.interval}_{self.start_date}_to_{self.end_date}.h5"
        if self.data_type == "metrics":
            hdf5_file_name = f"{self.symbol}_metrics_{self.start_date}_to_{self.end_date}.h5"
            
        hdf5_file_path = os.path.join(self.interval_folder, hdf5_file_name)
        final_data.to_hdf(hdf5_file_path, key="data", mode="w", format="table")
        
        return final_data

    def fetch_and_process_data(self):
        """主函数保持不变，但使用新的处理方式"""
        if self.exchange == "binance":
            all_data = self.download_binance_data()
            return self.process_data(all_data)
        elif self.exchange == "bybit":
            self.download_bybit_data()
        else:
            raise ValueError(f"Unsupported exchange: {self.exchange}")


# # 使用示例：
# symbol = "BTCUSDT"
# interval = "1d"  # 可以改為 "1m", "5m", "1h", "1d" 等
# start_date = "2022-01-01"
# end_date = "2024-12-31"
# data_folder = "../dataset/binance"
# data_type = "spot"
# exchange = "binance"

# # 創建 DataDownloader 實例，選擇對應的交易所
# data_downloader = DataDownloader(symbol, interval, start_date, end_date, data_folder, data_type, exchange)
# data = data_downloader.fetch_and_process_data()

# 下载 metrics 数据
symbol = "SOLUSDT"
start_date = "2021-12-01"
end_date = "2024-12-31"
data_folder = "../dataset/binance"

downloader = DataDownloader(
    symbol=symbol,
    interval=None,  # metrics数据不需要interval
    start_date=start_date,
    end_date=end_date,
    data_folder=data_folder,
    data_type="metrics",
    exchange="binance"
)
data = downloader.fetch_and_process_data()
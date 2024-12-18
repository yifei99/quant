import os
import requests
from datetime import datetime, timedelta
import zipfile
import pandas as pd

class DataDownloader:
    def __init__(self, symbol, interval, start_date, end_date, data_folder, data_type="spot", exchange="binance"):
        """
        初始化 DataDownloader 類的實例。

        參數:
            symbol (str): 幣種符號，例如 "BTCUSDT"。
            interval (str): K 線間隔，例如 "1m", "5m", "1h", "1d"。
            start_date (str): 起始日期，格式 "YYYY-MM-DD"。
            end_date (str): 結束日期，格式 "YYYY-MM-DD"。
            data_folder (str): 數據存儲的目錄，例如 "./dataset/binance"。
            data_type (str): 數據類型，"spot" 表示現貨數據，"futures" 表示合約數據。預設為 "spot"。
            exchange (str): 交易所名稱，預設為 "binance"。未來可擴展支持其他交易所，如 "bybit"。
        """
        self.symbol = symbol
        self.interval = interval
        self.start_date = start_date
        self.end_date = end_date
        self.data_folder = data_folder
        self.data_type = data_type
        self.exchange = exchange  # 新增交易所參數
        self.download_dir = "downloads"
        self.interval_folder = os.path.join(data_folder, symbol, data_type, interval)
        self.create_dir(self.download_dir)
        self.create_dir(self.interval_folder)

    def create_dir(self, path):
        """創建目錄"""
        """
        參數:
            path (str): 目錄路徑。
        """
        if not os.path.exists(path):
            os.makedirs(path)

    def download_file(self, url, dest_folder):
        """
        下載指定URL的文件並保存到目標文件夾。

        參數:
            url (str): 文件的下載URL。
            dest_folder (str): 文件的保存目標文件夾。

        返回:
            str or None: 下載後的文件路徑，如果下載失敗則返回None。
        """
        response = requests.get(url)
        if response.status_code == 200:
            file_name = url.split("/")[-1]
            file_path = os.path.join(dest_folder, file_name)
            with open(file_path, 'wb') as file:
                file.write(response.content)
            print(f"Downloaded: {file_name}")
            return file_path
        else:
            print(f"Failed to download: {url}")
            return None

    def unzip_and_move(self, zip_file_path, extract_to_folder):
        """
        解壓並移動文件

        參數:
            zip_file_path (str): 壓縮文件的路徑。
            extract_to_folder (str): 解壓後文件的保存目標文件夾。
        """
        self.create_dir(extract_to_folder)
        try:
            with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
                zip_ref.extractall(extract_to_folder)
            print(f"Unzipped: {zip_file_path}")
            os.remove(zip_file_path)
            print(f"Deleted zip file: {zip_file_path}")
        except zipfile.BadZipFile:
            print(f"Error: {zip_file_path} is not a valid zip file.")

    def download_binance_data(self):
        """
        下載 Binance 數據
        """
        start_date = datetime.strptime(self.start_date, "%Y-%m-%d")
        end_date = datetime.strptime(self.end_date, "%Y-%m-%d")
        date_range = [start_date + timedelta(days=x) for x in range((end_date - start_date).days + 1)]
        
        base_url = "https://data.binance.vision/data/spot/daily/klines" if self.data_type == "spot" else "https://data.binance.vision/data/futures/um/daily/klines"
        
        for date in date_range:
            date_str = date.strftime("%Y-%m-%d")
            url = f"{base_url}/{self.symbol}/{self.interval}/{self.symbol}-{self.interval}-{date_str}.zip"
            zip_file_path = self.download_file(url, self.download_dir)
            if zip_file_path:
                self.unzip_and_move(zip_file_path, extract_to_folder=self.interval_folder)

    def download_bybit_data(self):
        """
        下載 Bybit 數據 (未來可擴展)
        """
        # 這是未來的擴展，這裡需要根據 Bybit API 格式來實現
        pass

    def process_data(self):
        """
        處理數據並儲存為 HDF5 文件
        """
        all_files = [os.path.join(self.interval_folder, f) for f in os.listdir(self.interval_folder) if f.endswith('.csv')]
        all_data = []

        for file in all_files:
            print(f"Processing file: {file}")
            df = pd.read_csv(file, header=None)
            df.columns = [
                "timestamp_start", "open", "high", "low", "close", "volume",
                "timestamp_end", "quote_asset_volume", "number_of_trades", 
                "taker_buy_base", "taker_buy_quote", "ignore"
            ]
            all_data.append(df)
            os.remove(file)  # 刪除處理過的 CSV 文件
            print(f"Deleted file: {file}")

        final_data = pd.concat(all_data, ignore_index=True)

        # timestamp_start 和 timestamp_end 已經是 UNIX 時間戳，不需要處理
        final_data = final_data.sort_values(by="timestamp_start").reset_index(drop=True)

        # 動態生成 HDF5 文件的路徑，使用時間範圍作為文件名
        hdf5_file_name = f"{self.symbol}_{self.interval}_{self.start_date}_to_{self.end_date}.h5"
        hdf5_file_path = os.path.join(self.interval_folder, hdf5_file_name)
        
        final_data.to_hdf(hdf5_file_path, key="prices", mode="w", format="table")
        print(f"Data has been merged and saved as HDF5 file: {hdf5_file_path}")

        return final_data  # 返回處理後的數據

    def fetch_and_process_data(self):
        """
        統一調用下載和處理數據的方法
        """
        # 根據選擇的交易所下載數據
        if self.exchange == "binance":
            self.download_binance_data()
        elif self.exchange == "bybit":
            self.download_bybit_data()  # 將來可以實現 Bybit 數據下載邏輯
        else:
            raise ValueError(f"Unsupported exchange: {self.exchange}")
        
        return self.process_data()


# 使用示例：
symbol = "DOTUSDT"
interval = "1d"  # 可以改為 "1m", "5m", "1h", "1d" 等
start_date = "2022-05-12"
end_date = "2024-12-05"
data_folder = "../dataset/binance"
data_type = "spot"
exchange = "binance"

# 創建 DataDownloader 實例，選擇對應的交易所
data_downloader = DataDownloader(symbol, interval, start_date, end_date, data_folder, data_type, exchange)
data = data_downloader.fetch_and_process_data()
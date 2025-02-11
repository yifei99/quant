# data/data_loader.py

import os
import pandas as pd
from datetime import datetime
import glob
import logging

class DataLoader:
    def __init__(self, data_folder):
        """
        初始化 DataLoader 類的實例。

        參數:
            data_folder (str): 數據存儲的根目錄，例如 "./dataset".
        """
        self.data_folder = data_folder
        
        # 設置日誌
        logging.basicConfig(
            filename='data_loader.log',
            level=logging.INFO,
            format='%(asctime)s:%(levelname)s:%(message)s'
        )
        self.logger = logging.getLogger()

    def list_available_files(self, exchange, symbol, interval=None, data_type="spot"):
        """
        列出指定參數下的所有可用 HDF5 文件，支持模糊查找。

        參數:
            exchange (str): 交易所名稱，例如 "binance", "bybit".
            symbol (str): 幣種符號，例如 "BTCUSDT".
            interval (str, optional): K 線間隔，例如 "1m", "5m", "1h", "1d". metrics数据不需要此参数.
            data_type (str): 數據類型，"spot", "futures" 或 "metrics". 預設為 "spot".

        返回:
            list: 符合條件的 HDF5 文件路徑列表。
        """
        # 根据数据类型构建目标文件夹路径
        if data_type == "metrics":
            target_folder = os.path.join(
                self.data_folder, exchange, symbol, data_type
            )
        else:
            if interval is None:
                raise ValueError("interval is required for non-metrics data")
            target_folder = os.path.join(
                self.data_folder, exchange, symbol, data_type, interval
            )

        if not os.path.exists(target_folder):
            self.logger.error(f"The folder {target_folder} does not exist.")
            raise FileNotFoundError(f"The folder {target_folder} does not exist.")

        # 使用通配符模糊查找所有 HDF5 文件
        if data_type == "metrics":
            file_pattern = f"{symbol}_metrics_*_to_*.h5"
        else:
            file_pattern = f"{symbol}_{interval}_*_to_*.h5"
            
        search_path = os.path.join(target_folder, file_pattern)
        files = glob.glob(search_path)
        self.logger.info(f"Found {len(files)} HDF5 files in {target_folder}.")
        return files

    def load_data_from_path(self, file_path):
        """
        根據文件路徑讀取 HDF5 數據文件。

        參數:
            file_path (str): HDF5 文件的完整路徑。

        返回:
            pd.DataFrame: 加載的數據，如果加載失敗，返回空的 DataFrame。
        """
        if not os.path.isfile(file_path):
            self.logger.error(f"The file {file_path} does not exist.")
            raise FileNotFoundError(f"The file {file_path} does not exist.")

        # 讀取 HDF5 文件
        try:
            data = pd.read_hdf(file_path, key="prices")
            self.logger.info(f"Loaded data from {file_path}")
            return data
        except Exception as e:
            self.logger.error(f"Failed to load data from {file_path}: {e}")
            return pd.DataFrame()

    def load_data(self, exchange, symbol, interval=None, start_date=None, end_date=None, data_type="spot"):
        """
        根據指定的參數加載已合併的 HDF5 數據文件。

        參數:
            exchange (str): 交易所名稱，例如 "binance", "bybit".
            symbol (str): 幣種符號，例如 "BTCUSDT".
            interval (str, optional): K 線間隔，例如 "1m", "5m", "1h", "1d". metrics数据不需要此参数.
            start_date (str): 起始日期，格式 "YYYY-MM-DD".
            end_date (str): 結束日期，格式 "YYYY-MM-DD".
            data_type (str): 數據類型，"spot", "futures" 或 "metrics". 預設為 "spot".

        返回:
            pd.DataFrame: 加載的數據，如果文件不存在或加載失敗，返回空的 DataFrame。
        """
        # 构建目标文件夹路径
        if data_type == "metrics":
            target_folder = os.path.join(
                self.data_folder, exchange, symbol, data_type
            )
            # 动态生成 HDF5 文件的名称模式
            file_pattern = f"{symbol}_metrics_{start_date}_to_{end_date}.h5"
        else:
            if interval is None:
                raise ValueError("interval is required for non-metrics data")
            target_folder = os.path.join(
                self.data_folder, exchange, symbol, data_type, interval
            )
            # 动态生成 HDF5 文件的名称模式
            file_pattern = f"{symbol}_{interval}_{start_date}_to_{end_date}.h5"

        if not os.path.exists(target_folder):
            self.logger.error(f"The folder {target_folder} does not exist.")
            raise FileNotFoundError(f"The folder {target_folder} does not exist.")

        file_path = os.path.join(target_folder, file_pattern)

        if os.path.isfile(file_path):
            # 读取指定范围的 HDF5 文件
            try:
                data = pd.read_hdf(file_path, key="data")
                self.logger.info(f"Loaded data from {file_path}")
                return data
            except Exception as e:
                self.logger.error(f"Failed to load data from {file_path}: {e}")
                return pd.DataFrame()
        else:
            # 如果精确匹配的文件不存在，进行模糊查找
            self.logger.info(f"Exact file {file_path} not found. Performing fuzzy search.")
            matching_files = self.list_available_files(exchange, symbol, interval, data_type)

            if not matching_files:
                self.logger.error("No matching HDF5 files found.")
                raise FileNotFoundError("No matching HDF5 files found.")

            # 返回匹配的文件列表
            return matching_files

# # 使用示例
# if __name__ == "__main__":
#     # 初始化 DataLoader 實例
#     data_folder = "../dataset"  # 根目錄
#     data_loader = DataLoader(data_folder)

#     # 設定參數
#     exchange = "binance"
#     symbol = "BTCUSDT"
#     interval = "1d"
#     start_date = "2023-01-01"
#     end_date = "2023-02-01"
#     data_type = "spot"

#     # 嘗試加載特定範圍的數據
#     try:
#         data = data_loader.load_data(
#             exchange, symbol, interval, start_date, end_date, data_type
#         )
#         if isinstance(data, list):
#             # 如果返回的是文件列表，提示用戶選擇
#             print("Multiple matching files found:")
#             for idx, file in enumerate(data, 1):
#                 print(f"{idx}. {file}")
#             choice = int(input("Enter the number of the file you want to load: "))
#             selected_file = data[choice - 1]
#             data = data_loader.load_data_from_path(selected_file)
#         print(data.head())
#     except FileNotFoundError as e:
#         print(e)
import os
import requests
from datetime import datetime, timedelta
import zipfile
import pandas as pd

def fetch_and_process_binance_data(symbol, interval, start_date, end_date, data_folder, data_type="spot"):
    """
    從 Binance 獲取歷史數據，下載並解壓數據，合併為一個 HDF5 文件，並刪除冗余的 CSV 文件。
    
    參數:
    symbol (str): 幣種符號，例如 "BTCUSDT"
    interval (str): K 線間隔，例如 "1m", "5m", "1d"
    start_date (str): 起始日期，格式 "YYYY-MM-DD"
    end_date (str): 結束日期，格式 "YYYY-MM-DD"
    data_folder (str): 數據存儲的目錄
    data_type (str): 數據類型，"spot" 表示現貨數據，"futures" 表示合約數據
    """
    
    def create_dir(path):
        if not os.path.exists(path):
            os.makedirs(path)

    def download_file(url, dest_folder):
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

    def unzip_and_move(zip_file_path, extract_to_folder):
        create_dir(extract_to_folder)
        try:
            with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
                zip_ref.extractall(extract_to_folder)
            print(f"Unzipped: {zip_file_path}")
            os.remove(zip_file_path)
            print(f"Deleted zip file: {zip_file_path}")
        except zipfile.BadZipFile:
            print(f"Error: {zip_file_path} is not a valid zip file.")

    # 設置下載和解壓的文件夾
    download_dir = "downloads"
    interval_folder = os.path.join(data_folder, symbol, data_type, interval)
    create_dir(download_dir)
    create_dir(interval_folder)
    
    # 日期處理
    start_date = datetime.strptime(start_date, "%Y-%m-%d")
    end_date = datetime.strptime(end_date, "%Y-%m-%d")
    date_range = [start_date + timedelta(days=x) for x in range((end_date - start_date).days + 1)]
    
    # URL 構造
    if data_type == "spot":
        base_url = "https://data.binance.vision/data/spot/daily/klines"
    elif data_type == "futures":
        base_url = "https://data.binance.vision/data/futures/um/daily/klines"
    else:
        raise ValueError("Invalid data_type. Choose 'spot' for现货 or 'futures' for合约.")

    # 下載和解壓數據
    for date in date_range:
        date_str = date.strftime("%Y-%m-%d")
        url = f"{base_url}/{symbol}/{interval}/{symbol}-{interval}-{date_str}.zip"
        zip_file_path = download_file(url, download_dir)
        if zip_file_path:
            unzip_and_move(zip_file_path, extract_to_folder=interval_folder)
    
    # 合併數據
    all_files = [os.path.join(interval_folder, f) for f in os.listdir(interval_folder) if f.endswith('.csv')]
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

    # 將 timestamp_start 轉換為日期時間並排序
    final_data["timestamp_start"] = pd.to_datetime(final_data["timestamp_start"], unit="ms")
    final_data["timestamp_end"] = pd.to_datetime(final_data["timestamp_end"], unit="ms")
    final_data = final_data.sort_values(by="timestamp_start").reset_index(drop=True)
    
    # 動態生成 HDF5 文件的路徑
    hdf5_file_path = os.path.join(interval_folder, f"{symbol}_{interval}_data.h5")
    final_data.to_hdf(hdf5_file_path, key="prices", mode="w", format="table")
    print(f"Data has been merged and saved as HDF5 file: {hdf5_file_path}")
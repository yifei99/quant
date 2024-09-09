import os
import requests
from datetime import datetime, timedelta
import zipfile

# 创建目录的函数
def create_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

# 下载文件的函数
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

# 解压并移动文件的函数
def unzip_and_move(zip_file_path, extract_to_folder):
    # 确保目标文件夹存在
    create_dir(extract_to_folder)

    # 解压文件
    try:
        with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
            zip_ref.extractall(extract_to_folder)
        print(f"Unzipped: {zip_file_path}")
        
        # 删除压缩文件
        os.remove(zip_file_path)
        print(f"Deleted zip file: {zip_file_path}")
        
    except zipfile.BadZipFile:
        print(f"Error: {zip_file_path} is not a valid zip file.")

# 下载币安数据的函数
def download_binance_data(symbol, interval, start_date, end_date, download_dir="downloads", data_type="spot", data_folder="data"):
    """
    下载币安的历史K线数据，并将压缩包解压至指定文件夹。
    
    参数:
    symbol (str): 币种符号，例如"BTCUSDT"
    interval (str): K线间隔，例如"1m", "1h", "1d"等
    start_date (str): 起始日期，格式为"YYYY-MM-DD"
    end_date (str): 结束日期，格式为"YYYY-MM-DD"
    download_dir (str): 下载压缩文件存储的目录，默认为"downloads"
    data_type (str): 数据类型，"spot" 表示现货数据, "futures" 表示合约数据
    data_folder (str): 解压文件存放的目标文件夹，默认为"data"
    """
    
    # 创建下载文件夹
    create_dir(download_dir)
    
    # 解析日期
    start_date = datetime.strptime(start_date, "%Y-%m-%d")
    end_date = datetime.strptime(end_date, "%Y-%m-%d")

    # 生成日期范围
    date_range = [start_date + timedelta(days=x) for x in range((end_date - start_date).days + 1)]

    # 根据数据类型生成不同的 URL 前缀
    if data_type == "spot":
        base_url = "https://data.binance.vision/data/spot/daily/klines"
    elif data_type == "futures":
        base_url = "https://data.binance.vision/data/futures/um/daily/klines"
    else:
        raise ValueError("Invalid data_type. Choose 'spot' for现货 or 'futures' for合约.")

    # 确保按照币种、数据类型、时间间隔组织的文件夹结构存在
    symbol_folder = os.path.join(data_folder, symbol)  # 例如: data/BTCUSDT
    type_folder = os.path.join(symbol_folder, data_type)  # 例如: data/BTCUSDT/spot 或 data/BTCUSDT/futures
    interval_folder = os.path.join(type_folder, interval)  # 例如: data/BTCUSDT/spot/1m
    create_dir(interval_folder)
    
    # 下载并解压每个日期的数据
    for date in date_range:
        date_str = date.strftime("%Y-%m-%d")
        url = f"{base_url}/{symbol}/{interval}/{symbol}-{interval}-{date_str}.zip"
        
        # 下载文件
        zip_file_path = download_file(url, download_dir)
        
        # 如果下载成功，则解压并删除压缩文件
        if zip_file_path:
            unzip_and_move(zip_file_path, extract_to_folder=interval_folder)

# 示例使用：
# 下载现货 BTCUSDT 的 1 分钟 K 线数据，从 2024-08-17 到 2024-08-19，文件将存放在 data/BTCUSDT/spot/1m 目录下
download_binance_data(symbol="BTCUSDT", interval="5m", start_date="2024-08-17", end_date="2024-08-19", data_type="spot", data_folder="data")

# 下载合约 BTCUSDT 的 1 分钟 K 线数据，从 2024-09-01 到 2024-09-03，文件将存放在 data/BTCUSDT/futures/1m 目录下
download_binance_data(symbol="BTCUSDT", interval="5m", start_date="2024-09-01", end_date="2024-09-03", data_type="futures", data_folder="data")

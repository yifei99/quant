# factors/factor_definitions.py

import pandas as pd
from abc import ABC, abstractmethod
import numpy as np

class BaseFactor(ABC):
    """
    Base Factor class that defines the basic interface for factors.
    """
    def __init__(self, name):
        self.name = name

    @abstractmethod
    def calculate(self, data: pd.DataFrame) -> pd.Series:
        """
        Calculate factor values.

        Args:
            data (pd.DataFrame): DataFrame containing price data and other relevant data.

        Returns:
            pd.Series: Factor calculation results.
        """
        pass

    def __repr__(self):
        return f"{self.__class__.__name__}(name='{self.name}')"
    
class TwoThresholdFactor(BaseFactor):
    """
    Generic threshold-based factor:
    Generates buy signals when value exceeds upper threshold;
    Generates sell signals when value falls below lower threshold.
    """
    def __init__(self, 
                 name: str,
                 column_name: str,
                 upper_threshold: float,
                 lower_threshold: float):
        """
        Args:
            name (str): Factor name
            column_name (str): Name of the column to monitor in data
            upper_threshold (float): Upper threshold for buy signals
            lower_threshold (float): Lower threshold for sell signals
        """
        super().__init__(name)
        self.column_name = column_name
        self.upper_threshold = upper_threshold
        self.lower_threshold = lower_threshold

    def calculate(self, data: pd.DataFrame) -> pd.Series:
        """
        Calculate threshold-based factor using vectorized operations
        """
        if self.column_name not in data.columns:
            raise ValueError(f"DataFrame must contain '{self.column_name}' column")
        
        # 使用numpy数组进行计算
        values = data[self.column_name].values
        
        # 使用numpy的where函数一次性生成所有信号
        signals = np.where(
            values > self.upper_threshold, 1,
            np.where(
                values < self.lower_threshold, -1,
                0
            )
        )
        
        # 返回带有正确索引的Series
        return pd.Series(signals, index=data.index, name=self.name)


class USDTIssuance2Factor(TwoThresholdFactor):
    """USDT Issuance threshold factor"""
    def __init__(self, 
                 name='usdt_issuance', 
                 upper_threshold=10000000,
                 lower_threshold=1000000):
        """
        Args:
            name (str): Factor name, defaults to 'usdt_issuance'
            upper_threshold (float): Upper threshold for buy signals, defaults to 10M
            lower_threshold (float): Lower threshold for sell signals, defaults to 1M
        """
        super().__init__(
            name=name,
            column_name='USDT_issuance',
            upper_threshold=upper_threshold,
            lower_threshold=lower_threshold
        )


class Liq2Factor(TwoThresholdFactor):
    """Liquidation threshold factor"""
    def __init__(self, 
                 name='liq',
                 upper_threshold=2,
                 lower_threshold=-2):
        """
        Args:
            name (str): Factor name, defaults to 'liq'
            upper_threshold (float): Upper threshold for buy signals, defaults to 2
            lower_threshold (float): Lower threshold for sell signals, defaults to -2
        """
        super().__init__(
            name=name,
            column_name='Liq',
            upper_threshold=upper_threshold,
            lower_threshold=lower_threshold
        )

class BaseMaFactor(BaseFactor):
    """Base class for Moving Average factors"""
    def __init__(self, name: str, column_name: str, ma_period: int = 7):
        super().__init__(name)
        if ma_period < 0:
            raise ValueError("ma_period must be non-negative")
        self.column_name = column_name
        self.ma_period = ma_period
        
    def calculate(self, data: pd.DataFrame) -> pd.Series:
        """
        Calculate MA signals using vectorized operations
        """
        if self.column_name not in data.columns:
            raise ValueError(f"DataFrame must contain '{self.column_name}' column")
            
        values = data[self.column_name].values
        signals = np.zeros(len(data))  # 初始化信号为0
        
        # 如果ma_period为0，直接使用原始值
        if self.ma_period == 0:
            ma = values
            valid_index = 0
        else:
            # 计算移动平均
            ma = np.convolve(values, np.ones(self.ma_period)/self.ma_period, mode='valid')
            valid_index = self.ma_period - 1  # MA开始有效的位置
            # 对齐长度
            ma = np.pad(ma, (valid_index, 0), mode='edge')
        
        # 生成信号
        signals[valid_index:][values[valid_index:] == ma[valid_index:]] = 0
        signals[valid_index:][values[valid_index:] > ma[valid_index:]] = 1
        signals[valid_index:][values[valid_index:] < ma[valid_index:]] = -1
        
        return pd.Series(signals, index=data.index, name=self.name)

class UsdVolumeMaFactor(BaseMaFactor):
    """USD Volume Moving Average Factor"""
    def __init__(self, name='usd_volume_ma', ma_period=7):
        super().__init__(
            name=name,
            column_name='quote_asset_volume',
            ma_period=ma_period
        )

class AssetVolumeMaFactor(BaseMaFactor):
    """Asset Volume Moving Average Factor"""
    def __init__(self, name='asset_volume_ma', ma_period=7):
        super().__init__(
            name=name,
            column_name='volume',
            ma_period=ma_period
        )

class PriceMaFactor(BaseMaFactor):
    """Price Moving Average Factor"""
    def __init__(self, name='price_ma', ma_period=7):
        super().__init__(
            name=name,
            column_name='close',
            ma_period=ma_period
        )

class Base2MaFactor(BaseFactor):
    """Base class for Dual Moving Average factors"""
    def __init__(self, name: str, column_name: str, ma_period_1: int = 7, ma_period_2: int = 14):
        super().__init__(name)
        if ma_period_1 < 0 or ma_period_2 < 0:
            raise ValueError("ma_period must be non-negative")
        self.column_name = column_name
        self.ma_period_1 = ma_period_1
        self.ma_period_2 = ma_period_2
        
    def calculate(self, data: pd.DataFrame) -> pd.Series:
        """
        Calculate Dual MA signals using optimized numpy operations
        """
        if self.column_name not in data.columns:
            raise ValueError(f"DataFrame must contain '{self.column_name}' column")
            
        values = data[self.column_name].values
        signals = np.zeros(len(data))  # 初始化信号为0
        
        # 计算第一个MA
        if self.ma_period_1 == 0:
            ma_1 = values
            valid_index_1 = 0
        else:
            ma_1 = np.convolve(values, np.ones(self.ma_period_1)/self.ma_period_1, mode='valid')
            valid_index_1 = self.ma_period_1 - 1
            ma_1 = np.pad(ma_1, (valid_index_1, 0), mode='edge')
            
        # 计算第二个MA
        if self.ma_period_2 == 0:
            ma_2 = values
            valid_index_2 = 0
        else:
            ma_2 = np.convolve(values, np.ones(self.ma_period_2)/self.ma_period_2, mode='valid')
            valid_index_2 = self.ma_period_2 - 1
            ma_2 = np.pad(ma_2, (valid_index_2, 0), mode='edge')
        
        # 使用较长的MA周期作为有效起始点
        valid_index = max(valid_index_1, valid_index_2)
        
        # 生成信号
        signals[valid_index:][ma_1[valid_index:] == ma_2[valid_index:]] = 0
        signals[valid_index:][ma_1[valid_index:] > ma_2[valid_index:]] = 1
        signals[valid_index:][ma_1[valid_index:] < ma_2[valid_index:]] = -1
        
        return pd.Series(signals, index=data.index, name=self.name)
    
class Price2MaFactor(Base2MaFactor):
    """Price Dual Moving Average Factor"""
    def __init__(self, name='price_2ma', ma_period_1=7, ma_period_2=14):
        super().__init__(
            name=name,
            column_name='close',
            ma_period_1=ma_period_1,
            ma_period_2=ma_period_2
        )

class Volume2MaFactor(Base2MaFactor):
    """Volume Dual Moving Average Factor"""
    def __init__(self, name='volume_2ma', ma_period_1=7, ma_period_2=14):
        super().__init__(
            name=name,
            column_name='volume',
            ma_period_1=ma_period_1,
            ma_period_2=ma_period_2
        )   

class UsdVolume2MaFactor(Base2MaFactor):
    """USD Volume Dual Moving Average Factor"""
    def __init__(self, name='usd_volume_2ma', ma_period_1=7, ma_period_2=14):
        super().__init__(
            name=name,
            column_name='quote_asset_volume',
            ma_period_1=ma_period_1,
            ma_period_2=ma_period_2
        )

class VolAdjMomentumFactor(TwoThresholdFactor):
    """波动率调整动量因子"""
    def __init__(self, 
                 name='vol_adj_momentum',
                 window: int = 20,
                 vol_window: int = None,
                 upper_threshold: float = 1.0,
                 lower_threshold: float = -1.0):
        """
        Args:
            name: 因子名称
            window: 价格变化窗口
            vol_window: 波动率计算窗口，默认等于price_window
            upper_threshold: 买入阈值
            lower_threshold: 卖出阈值
        """
        super().__init__(
            name=name,
            column_name='vol_adj_momentum',  # 这里可以是任意值，因为我们会重写calculate方法
            upper_threshold=upper_threshold,
            lower_threshold=lower_threshold
        )
        self.window = window
        self.vol_window = vol_window or window

    def calculate(self, data: pd.DataFrame) -> pd.Series:
        """计算因子值"""
        # 计算价格变化
        price_change = data['close'].diff(self.window)
        
        # 计算历史波动率 (使用对数收益率的标准差)
        log_returns = np.log(data['close'] / data['close'].shift(1))
        volatility = log_returns.rolling(window=self.vol_window).std() * np.sqrt(self.vol_window)
        
        # 计算波动率调整后的动量
        vol_adj_momentum = price_change / (volatility * data['close'])
        
        # 使用父类的阈值逻辑生成信号
        signals = np.where(
            vol_adj_momentum > self.upper_threshold, 1,
            np.where(
                vol_adj_momentum < self.lower_threshold, -1,
                0
            )
        )
        
        return pd.Series(signals, index=data.index, name=self.name)

class LiquidityZoneReversionFactor(BaseFactor):
    """
    Liquidity Zone Reversion Factor
    
    Generates signals based on price deviations from high-volume zones.
    Buy when price is significantly below the high-volume zone,
    Sell when price is significantly above the high-volume zone.
    """
    def __init__(self, 
                 name='liquidity_zone',
                 volume_window=30, 
                 upper_threshold=2.0,  # 改为独立的上阈值
                 lower_threshold=-1.5,  # 改为独立的下阈值
                 volume_quantile=0.75):
        """
        Args:
            name (str): Factor name
            volume_window (int): Window size for calculating high-volume zones
            upper_threshold (float): Upper threshold for short signals
            lower_threshold (float): Lower threshold for long signals
            volume_quantile (float): Quantile threshold for defining high-volume zones
        """
        super().__init__(name)
        if volume_window < 1:
            raise ValueError("volume_window must be at least 1")
        self.volume_window = volume_window
        self.upper_threshold = upper_threshold
        self.lower_threshold = lower_threshold
        self.volume_quantile = volume_quantile
        
    def calculate(self, data: pd.DataFrame) -> pd.Series:
        """
        Calculate factor values and trading signals using vectorized operations
        
        Args:
            data (pd.DataFrame): DataFrame containing 'close', 'high', 'low', 'volume'
            
        Returns:
            pd.Series: Trading signals (-1: short, 0: no action, 1: long)
        """
        # 确保数据足够长
        if len(data) < self.volume_window:
            return pd.Series(0, index=data.index, name=self.name)
        
        # 使用numpy数组进行计算以提高性能
        high = data['high'].values
        low = data['low'].values
        close = data['close'].values
        volume = data['volume'].values
        
        # 计算ATR (使用向量化操作)
        tr1 = high - low
        tr2 = np.abs(high - np.roll(close, 1))
        tr3 = np.abs(low - np.roll(close, 1))
        tr = np.maximum.reduce([tr1, tr2, tr3])
        
        # 使用卷积计算ATR (比循环快得多)
        kernel = np.ones(20) / 20
        atr = np.convolve(tr, kernel, mode='full')[:len(tr)]
        atr[:20] = atr[20]  # 填充前20个值
        
        # 使用卷积计算移动窗口的volume和close
        volume_kernel = np.ones(self.volume_window)
        close_sum = np.convolve(close * volume, volume_kernel, mode='full')[:len(close)]
        volume_sum = np.convolve(volume, volume_kernel, mode='full')[:len(volume)]
        
        # 计算移动平均价格
        vwap = close_sum / np.maximum(volume_sum, 1e-10)  # 避免除以0
        
        # 使用rolling quantile计算高成交量区域
        volume_df = pd.Series(volume).rolling(self.volume_window)
        volume_threshold = volume_df.quantile(self.volume_quantile)
        high_volume_mask = volume >= volume_threshold
        
        # 计算高成交量区域价格
        high_volume_zones = np.where(
            high_volume_mask,
            close,
            vwap
        )
        
        # 计算价格偏离度
        deviation = np.where(
            atr > 0,
            (close - high_volume_zones) / atr,
            0
        )
        
        # 生成交易信号 (使用不同的上下阈值)
        signals = np.where(
            deviation > self.upper_threshold, -1,  # 价格显著高于高成交量区域时做空
            np.where(
                deviation < self.lower_threshold, 1,  # 价格显著低于高成交量区域时做多
                0
            )
        )
        
        # 填充开始的窗口期
        signals[:self.volume_window] = 0
        
        return pd.Series(signals, index=data.index, name=self.name)

class FractalEfficiencyRatio(BaseFactor):
    """
    Fractal Efficiency Ratio (FER) Factor
    
    FER = D_real / D_straight
    
    用于判断市场趋势的存在性和方向：
    - 1.0 <= FER <= trend_upper：存在明显趋势，返回1(上涨)或-1(下跌)
    - FER > trend_upper：震荡市场，返回0
    """
    def __init__(self, 
                 name='fractal_efficiency',
                 window=24,
                 trend_upper=3.0):
        super().__init__(name)
        self.window = window
        self.trend_upper = trend_upper
        
    def calculate(self, data: pd.DataFrame) -> pd.Series:
        """
        计算FER因子值和趋势信号，使用向量化操作优化性能
        
        Returns:
            pd.Series: 趋势信号 (1: 上涨趋势, -1: 下跌趋势, 0: 震荡)
        """
        close = data['close'].values
        
        # 计算实际路径距离
        price_changes = np.abs(np.diff(close))
        # 使用卷积计算移动窗口和
        kernel = np.ones(self.window)
        d_real = np.convolve(price_changes, kernel, mode='valid')
        
        # 初始化信号数组，前window个周期设为0（无信号）
        signals = np.zeros(len(close))
        
        # 计算直线距离和FER
        for i in range(self.window, len(close)):
            start_price = close[i-self.window]
            end_price = close[i]
            d_straight = abs(end_price - start_price)
            
            # 计算FER
            if d_straight > 0:
                fer = d_real[i-self.window] / d_straight
            else:
                fer = 1.0
                
            # 生成信号
            if 1.0 <= fer <= self.trend_upper:
                signals[i] = np.sign(end_price - start_price)
            else:
                signals[i] = 0
        
        return pd.Series(signals, index=data.index, name=self.name)

class FERVolMomentumFactor(BaseFactor):
    """
    结合FER和VolAdjMomentum的组合因子
    
    逻辑：
    1. 只有当FER给出趋势信号(1)时，才采用VolAdjMomentum的信号
    2. 其他情况（FER为0或-1）时，信号为0
    """
    def __init__(self, 
                 name='fer_volmom',
                 window=24,
                 fer_trend_upper=3.0,
                 upper_threshold=0.5,
                 lower_threshold=-0.5):
        super().__init__(name)
        self.window = window
        self.fer_trend_upper = fer_trend_upper
        self.upper_threshold = upper_threshold
        self.lower_threshold = lower_threshold
        
    def calculate(self, data: pd.DataFrame) -> pd.Series:
        """
        计算组合信号
        """
        # 使用现有的FER因子
        fer_factor = FractalEfficiencyRatio(
            window=self.window,
            trend_upper=self.fer_trend_upper
        )
        fer_signals = fer_factor.calculate(data)
        
        # 使用现有的VolAdjMomentum因子
        volmom_factor = VolAdjMomentumFactor(
            window=self.window,
            upper_threshold=self.upper_threshold,
            lower_threshold=self.lower_threshold
        )
        volmom_signals = volmom_factor.calculate(data)
        
        # 组合信号：只在FER为1时采用VolAdjMomentum的信号
        combined_signals = pd.Series(0, index=data.index, name=self.name)
        trend_mask = (fer_signals == 1)
        combined_signals[trend_mask] = volmom_signals[trend_mask]
        
        return combined_signals
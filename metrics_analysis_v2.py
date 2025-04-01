import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.model_selection import cross_val_score, TimeSeriesSplit
from sklearn.feature_selection import mutual_info_regression
from sklearn.metrics import r2_score, roc_auc_score, accuracy_score
import warnings
from ta.momentum import RSIIndicator
from ta.trend import SMAIndicator, EMAIndicator, MACD
from ta.volatility import BollingerBands, AverageTrueRange
from ta.volume import MFIIndicator, OnBalanceVolumeIndicator
import time
warnings.filterwarnings("ignore")

# 配置参数
CONFIG = {
    'forward_periods': [1, 5, 10, 20],  # 预测周期（单位：数据周期）
    'test_size': 0.2,                   # 测试集比例
    'cv_folds': 5,                      # 交叉验证折数
    'bins': 10,                         # 分箱数量
    'random_state': 42,                 # 随机种子
    'nonlinear_threshold': 0.53,        # 非线性关系筛选阈值 
    'mi_threshold': 0.01,               # 互信息筛选阈值
    'rf_r2_threshold': 0,               # 随机森林R²筛选阈值
    'rf_params': {                      # 随机森林通用参数
        'n_estimators': 100,
        'max_depth': 5,
        'n_jobs': -1,
        'random_state': 42
    }
}

def load_data():
    """加载价格数据"""
    print("加载价格数据...")
    start_time = time.time()
    
    # 实际数据路径（根据您的环境修改）
    data_path = '/Users/yifei/dev/quant/dataset/binance/BTCUSDT/spot/1h/BTCUSDT_1h_2020-12-16_to_2025-02-14.h5'
    
    try:
        # 尝试加载实际数据
        df = pd.read_hdf(data_path)
        print(f"数据已加载: {data_path} - {len(df)} 条记录, 耗时: {time.time() - start_time:.2f}s")
        return df
    except FileNotFoundError:
        # 如果文件不存在，生成示例数据
        print(f"警告: 未找到数据文件 {data_path}, 生成示例数据...")
        dates = pd.date_range('2020-01-01', periods=5000, freq='H')
        prices = np.cumprod(1 + np.random.normal(0, 0.001, len(dates)))
        return pd.DataFrame({'close': prices, 'high': prices*1.01, 'low': prices*0.99, 'volume': np.random.random(len(dates))*100}, index=dates)

def calculate_all_metrics(price_data):
    """计算多种技术指标"""
    print("\n计算技术指标...")
    metrics = {}
    
    # 价格基础衍生指标
    metrics['returns_1d'] = price_data['close'].pct_change()
    metrics['log_returns'] = np.log(price_data['close']).diff()
    
    # 动量类指标
    metrics['rsi'] = RSIIndicator(close=price_data['close'], window=14).rsi()
    
    # 趋势类指标
    metrics['sma_fast'] = SMAIndicator(close=price_data['close'], window=10).sma_indicator()
    metrics['sma_slow'] = SMAIndicator(close=price_data['close'], window=30).sma_indicator()
    metrics['ema_fast'] = EMAIndicator(close=price_data['close'], window=10).ema_indicator()
    metrics['ema_slow'] = EMAIndicator(close=price_data['close'], window=30).ema_indicator()
    
    # MACD
    macd = MACD(close=price_data['close'])
    metrics['macd_line'] = macd.macd()
    metrics['macd_signal'] = macd.macd_signal()
    metrics['macd_diff'] = macd.macd_diff()
    
    # 波动率指标
    bb = BollingerBands(close=price_data['close'])
    metrics['bb_high'] = bb.bollinger_hband()
    metrics['bb_low'] = bb.bollinger_lband()
    metrics['bb_width'] = bb.bollinger_wband()
    metrics['bb_pct'] = (price_data['close'] - bb.bollinger_lband()) / (bb.bollinger_hband() - bb.bollinger_lband())
    metrics['atr'] = AverageTrueRange(high=price_data['high'], low=price_data['low'], close=price_data['close']).average_true_range()
    
    # 成交量指标
    if 'volume' in price_data.columns:
        metrics['mfi'] = MFIIndicator(high=price_data['high'], low=price_data['low'], 
                                     close=price_data['close'], volume=price_data['volume']).money_flow_index()
        metrics['obv'] = OnBalanceVolumeIndicator(close=price_data['close'], volume=price_data['volume']).on_balance_volume()
    
    # 实验性合成指标
    metrics['rsi_sma_diff'] = metrics['rsi'] - metrics['sma_fast']
    metrics['rsi_slope'] = metrics['rsi'].diff(3)
    
    # 生成一些具有可能的非线性关系的人工指标
    # 确保转换为pandas Series，并设置索引
    metrics['binary_pattern'] = pd.Series(
        np.tile([0, 1, 0, 1, 0, 1], (len(price_data) // 6) + 1)[:len(price_data)], 
        index=price_data.index
    )
    
    metrics['sine_wave'] = pd.Series(
        np.sin(np.linspace(0, 20*np.pi, len(price_data))), 
        index=price_data.index
    )
    
    metrics['step_function'] = pd.Series(
        np.where(metrics['rsi'] > 70, 1, np.where(metrics['rsi'] < 30, -1, 0)),
        index=price_data.index
    )
    
    # 处理NaN值
    for name, series in metrics.items():
        # 确保series是pandas Series类型
        if not isinstance(series, pd.Series):
            metrics[name] = pd.Series(series, index=price_data.index)
            series = metrics[name]
        
        # 填充NaN值
        metrics[name] = series.fillna(method='ffill').fillna(method='bfill')
        print(f"已计算: {name}, 长度: {len(series)}, 有效值: {(~series.isna()).sum()}")
    
    return metrics

def prepare_target_variables(price_data, periods=None):
    """准备多个周期的目标变量"""
    if periods is None:
        periods = CONFIG['forward_periods']
    
    targets = {}
    for period in periods:
        # 计算未来收益
        future_returns = price_data['close'].pct_change(period).shift(-period)
        targets[f'returns_{period}d'] = future_returns
        
        # 计算未来方向 (1=上涨, 0=下跌)
        future_direction = (future_returns > 0).astype(int)
        targets[f'direction_{period}d'] = future_direction
        
        print(f"已准备目标变量: returns_{period}d, 有效值: {(~future_returns.isna()).sum()}")
    
    return targets

def evaluate_nonlinear_relationship(metric, future_returns):
    """评估指标与未来收益之间的非线性关系"""
    # 删除NaN值
    valid_data = pd.DataFrame({'metric': metric, 'returns': future_returns}).dropna()
    if len(valid_data) < 100:  # 确保有足够的数据
        return {'error': f"数据不足，只有{len(valid_data)}个有效样本"}
    
    metric_values = valid_data['metric'].values.reshape(-1, 1)
    returns = valid_data['returns'].values
    binary_returns = (returns > 0).astype(int)
    
    results = {}
    
    # 1. 互信息 (Mutual Information) - 能捕捉任何类型的统计依赖关系
    mi = mutual_info_regression(metric_values, returns)[0]
    results['mutual_information'] = mi
    
    # 2. 使用非线性模型直接评估预测能力
    rf_model = RandomForestRegressor(**CONFIG['rf_params'])
    rf_scores = cross_val_score(rf_model, metric_values, returns, cv=CONFIG['cv_folds'], scoring='r2')
    results['rf_r2'] = rf_scores.mean()
    
    # 3. 二分类评估 (涨/跌)
    rf_clf = RandomForestClassifier(**CONFIG['rf_params'])
    try:
        rf_clf.fit(metric_values, binary_returns)
        pred_proba = rf_clf.predict_proba(metric_values)[:, 1]
        
        results['classification_accuracy'] = accuracy_score(binary_returns, rf_clf.predict(metric_values))
        results['roc_auc'] = roc_auc_score(binary_returns, pred_proba)
    except Exception as e:
        results['classification_error'] = str(e)
    
    # 4. 条件概率分析 (对于分类型指标)
    unique_vals = np.unique(metric_values)
    if len(unique_vals) <= 10:  # 如果指标值种类少，进行条件概率分析
        cond_probs = {}
        for val in unique_vals:
            val_idx = np.where(metric_values == val)[0]
            if len(val_idx) > 0:
                prob_up = (returns[val_idx] > 0).mean()
                # 使用更安全的方式处理值
                try:
                    if isinstance(val, np.ndarray) and val.size > 0:
                        val_str = str(val[0])
                    else:
                        val_str = str(val)
                    cond_probs[f'value_{val_str}'] = prob_up
                except:
                    cond_probs[f'value_{val}'] = prob_up
        results['conditional_probabilities'] = cond_probs
    
    # 5. 最大信息系数 (MIC) - 专门用于检测非线性关系
    try:
        from minepy import MINE
        mine = MINE()
        mine.compute_score(metric_values.ravel(), returns)
        results['mic'] = mine.mic()
    except ImportError:
        results['mic'] = 0  # 如果minepy未安装，设置为0而不是字符串
    
    # 6. 线性相关系数 (作为比较基准)
    linear_corr = np.corrcoef(metric_values.ravel(), returns)[0, 1]
    results['linear_correlation'] = linear_corr if not np.isnan(linear_corr) else 0
    
    # 7. 判断是否有预测能力以及是否是非线性关系
    results['has_predictive_power'] = (
        results.get('roc_auc', 0) > CONFIG['nonlinear_threshold'] or 
        results['mutual_information'] > CONFIG['mi_threshold'] or 
        results['rf_r2'] > CONFIG['rf_r2_threshold']
    )
    
    # 确保mic是数值类型
    mic_value = results.get('mic', 0)
    if not isinstance(mic_value, (int, float)):
        mic_value = 0
    
    results['is_nonlinear'] = (
        abs(mic_value) > 1.5 * abs(results['linear_correlation']) or
        (results['rf_r2'] > 0 and abs(results['linear_correlation']) < 0.1)
    )
    
    return results

def screen_metrics_for_nonlinear_relationships(metrics_dict, targets_dict, target_key='returns_5d'):
    """筛选多个指标中具有非线性预测能力的指标"""
    future_returns = targets_dict[target_key]
    results = {}
    promising_metrics = []
    nonlinear_metrics = []
    
    print(f"\n开始筛选与{target_key}具有非线性关系的指标...")
    
    for name, metric in metrics_dict.items():
        print(f"评估指标: {name}")
        eval_results = evaluate_nonlinear_relationship(metric, future_returns)
        results[name] = eval_results
        
        if 'error' in eval_results:
            print(f"  ! 评估出错: {eval_results['error']}")
            continue
            
        # 筛选有预测能力的指标
        if eval_results.get('has_predictive_power', False):
            promising_metrics.append(name)
            print(f"  ✓ 发现潜在有用指标: {name}")
            print(f"    - ROC AUC: {eval_results.get('roc_auc', 'N/A')}")
            print(f"    - 互信息: {eval_results.get('mutual_information', 'N/A')}")
            print(f"    - 随机森林 R²: {eval_results.get('rf_r2', 'N/A')}")
            print(f"    - 线性相关: {eval_results.get('linear_correlation', 'N/A')}")
            
            # 进一步筛选非线性关系
            if eval_results.get('is_nonlinear', False):
                nonlinear_metrics.append(name)
                print(f"    * 显示出非线性关系特征")
    
    print(f"\n总结: 在{len(metrics_dict)}个指标中，发现{len(promising_metrics)}个有预测能力的指标，其中{len(nonlinear_metrics)}个表现出非线性关系")
    if nonlinear_metrics:
        print(f"非线性关系指标: {', '.join(nonlinear_metrics)}")
    
    return {
        'promising_metrics': promising_metrics,
        'nonlinear_metrics': nonlinear_metrics,
        'detailed_results': results
    }

def visualize_nonlinear_relationship(metric, future_returns, metric_name="指标", target_name="未来收益", bins=None):
    """可视化指标与未来收益的非线性关系"""
    if bins is None:
        bins = CONFIG['bins']
    
    # 创建图形
    plt.figure(figsize=(15, 10))
    
    # 删除NaN值
    valid_data = pd.DataFrame({'metric': metric, 'returns': future_returns}).dropna()
    
    # 1. 散点图
    plt.subplot(2, 2, 1)
    plt.scatter(valid_data['metric'], valid_data['returns'], alpha=0.3)
    plt.title(f'{metric_name} vs {target_name} 散点图')
    plt.xlabel(metric_name)
    plt.ylabel(target_name)
    
    # 添加趋势线
    try:
        z = np.polyfit(valid_data['metric'], valid_data['returns'], 1)
        p = np.poly1d(z)
        plt.plot(sorted(valid_data['metric']), p(sorted(valid_data['metric'])), "r--")
    except:
        pass
    
    # 2. 分箱分析图
    plt.subplot(2, 2, 2)
    
    # 等频分箱
    try:
        qcut_bins = pd.qcut(valid_data['metric'], bins, duplicates='drop')
        bin_means = valid_data.groupby(qcut_bins)['returns'].mean()
        bin_counts = valid_data.groupby(qcut_bins)['returns'].count()
        
        # 获取区间标签
        labels = [str(x) for x in qcut_bins.cat.categories]
        
        plt.bar(range(len(bin_means)), bin_means, width=0.8)
        plt.title(f'{metric_name} 分箱分析 (每箱约 {len(valid_data)//bins} 样本)')
        plt.xticks(range(len(bin_means)), labels, rotation=45)
        plt.xlabel(f'{metric_name} 区间')
        plt.ylabel(f'平均{target_name}')
        
        # 在每个柱子上标注样本数量
        for i, (v, c) in enumerate(zip(bin_means, bin_counts)):
            plt.text(i, v + (0.0005 if v >= 0 else -0.002), f'n={c}', ha='center')
    except Exception as e:
        plt.text(0.5, 0.5, f"分箱失败: {str(e)}", ha='center', va='center')
    
    # 3. 条件概率图（对于二元目标）
    plt.subplot(2, 2, 3)
    try:
        # 计算每个分箱中收益为正的概率
        bin_probs = valid_data.groupby(qcut_bins)['returns'].apply(lambda x: (x > 0).mean())
        
        plt.bar(range(len(bin_probs)), bin_probs, width=0.8)
        plt.axhline(y=0.5, color='r', linestyle='-', alpha=0.3)
        plt.title(f'{metric_name} 条件概率 (收益为正的概率)')
        plt.xticks(range(len(bin_probs)), labels, rotation=45)
        plt.xlabel(f'{metric_name} 区间')
        plt.ylabel('收益为正的概率')
        
        # 在每个柱子上标注概率值
        for i, v in enumerate(bin_probs):
            plt.text(i, v + 0.02, f'{v:.2f}', ha='center')
    except:
        plt.text(0.5, 0.5, "条件概率计算失败", ha='center', va='center')
    
    # 4. 非参数回归 (LOWESS平滑)
    plt.subplot(2, 2, 4)
    try:
        from statsmodels.nonparametric.smoothers_lowess import lowess
        
        # 按指标值排序
        sorted_data = valid_data.sort_values('metric')
        
        # 应用LOWESS平滑
        lowess_result = lowess(sorted_data['returns'], sorted_data['metric'], frac=0.1, it=3, return_sorted=False)
        
        plt.plot(sorted_data['metric'], sorted_data['returns'], '.', alpha=0.3)
        plt.plot(sorted_data['metric'], lowess_result, 'r-', linewidth=2)
        plt.title(f'{metric_name} LOWESS平滑')
        plt.xlabel(metric_name)
        plt.ylabel(target_name)
    except Exception as e:
        plt.text(0.5, 0.5, f"LOWESS平滑失败: {str(e)}", ha='center', va='center')
    
    plt.tight_layout()
    
    # 创建reports目录（如果不存在）
    import os
    if not os.path.exists('reports'):
        os.makedirs('reports')
    
    # 保存图形到指定目录
    timestamp = pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')
    plot_path = f'reports/{metric_name}_vs_{target_name}_{timestamp}.png'
    plt.savefig(plot_path)
    plt.close()
    
    print(f"  - 可视化结果已保存为: {plot_path}")
    
    return plot_path

def analyze_metric_across_periods(metric, targets_dict, metric_name="指标"):
    """分析指标在不同预测周期上的表现"""
    results = {}
    
    # 收集不同周期的评估结果
    for period in CONFIG['forward_periods']:
        target_key = f'returns_{period}d'
        if target_key in targets_dict:
            result = evaluate_nonlinear_relationship(metric, targets_dict[target_key])
            results[period] = result
    
    # 创建结果数据框并可视化
    if not results:
        return None
    
    # 提取关键指标
    periods = list(results.keys())
    mi_values = [results[p].get('mutual_information', np.nan) for p in periods]
    rf_r2_values = [results[p].get('rf_r2', np.nan) for p in periods]
    roc_auc_values = [results[p].get('roc_auc', np.nan) for p in periods]
    linear_corr_values = [results[p].get('linear_correlation', np.nan) for p in periods]
    
    # 创建DataFrame
    df = pd.DataFrame({
        'Period': periods,
        'MutualInfo': mi_values,
        'RF_R2': rf_r2_values,
        'ROC_AUC': roc_auc_values,
        'LinearCorr': linear_corr_values
    }).set_index('Period')
    
    # 可视化
    plt.figure(figsize=(10, 6))
    df.plot(kind='bar', subplots=True, layout=(2, 2), figsize=(12, 8), sharex=True)
    plt.suptitle(f"{metric_name} 在不同预测周期的表现", fontsize=16)
    plt.tight_layout()
    plt.subplots_adjust(top=0.9)
    
    return {
        'cross_period_performance': df,
        'best_period': df['ROC_AUC'].idxmax(),
        'best_roc_auc': df['ROC_AUC'].max()
    }

def process_nonlinear_metrics(metrics_dict, targets_dict, analysis_period=5):
    """综合处理非线性指标"""
    # 1. 筛选具有非线性预测能力的指标
    target_key = f'returns_{analysis_period}d'
    screening_results = screen_metrics_for_nonlinear_relationships(metrics_dict, targets_dict, target_key)
    
    # 2. 对每个非线性指标进行详细分析和可视化
    detailed_analysis = {}
    
    for metric_name in screening_results['nonlinear_metrics']:
        print(f"\n详细分析非线性指标: {metric_name}")
        
        # 可视化与特定周期目标的非线性关系
        plt = visualize_nonlinear_relationship(
            metrics_dict[metric_name], 
            targets_dict[target_key], 
            metric_name, 
            f"{analysis_period}天未来收益"
        )
        plt.savefig(f"{metric_name}_nonlinear_analysis.png")
        plt.close()
        
        # 分析不同周期的表现
        cross_period_results = analyze_metric_across_periods(
            metrics_dict[metric_name], 
            targets_dict, 
            metric_name
        )
        if cross_period_results:
            plt.savefig(f"{metric_name}_cross_period_analysis.png")
            plt.close()
            
            detailed_analysis[metric_name] = {
                'cross_period_results': cross_period_results,
                'best_period': cross_period_results.get('best_period'),
                'evaluation': screening_results['detailed_results'][metric_name]
            }
    
    # 3. 生成最佳非线性指标总结报告
    if detailed_analysis:
        print("\n最佳非线性预测指标总结:")
        print("=" * 80)
        
        for name, analysis in detailed_analysis.items():
            best_period = analysis.get('best_period')
            eval_results = analysis.get('evaluation', {})
            
            print(f"\n指标: {name}")
            print(f"最佳预测周期: {best_period}天")
            print(f"ROC AUC: {eval_results.get('roc_auc', 'N/A'):.4f}")
            print(f"互信息: {eval_results.get('mutual_information', 'N/A'):.4f}")
            print(f"随机森林 R²: {eval_results.get('rf_r2', 'N/A'):.4f}")
            print(f"线性相关系数: {eval_results.get('linear_correlation', 'N/A'):.4f}")
            
            if 'mic' in eval_results:
                print(f"最大信息系数 (MIC): {eval_results.get('mic', 'N/A')}")
            
            print("-" * 50)
    
    return {
        'screening_results': screening_results,
        'detailed_analysis': detailed_analysis
    }

def main():
    # 1. 数据准备
    price_data = load_data()
    
    # 2. 计算多种技术指标
    metrics_dict = calculate_all_metrics(price_data)
    
    # 3. 准备多个周期的目标变量
    targets_dict = prepare_target_variables(price_data)
    
    # 4. 筛选并分析非线性关系指标
    results = process_nonlinear_metrics(metrics_dict, targets_dict, analysis_period=5)
    
    print("\n分析完成！可视化结果已保存为PNG文件。")

if __name__ == "__main__":
    try:
        main()
        print("程序成功完成！")
    except Exception as e:
        import traceback
        print(f"程序执行出错: {str(e)}")
        traceback.print_exc()
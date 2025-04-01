import pandas as pd
import numpy as np
from sklearn.model_selection import TimeSeriesSplit
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from scipy import stats
import statsmodels.api as sm
import warnings
import time
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.gridspec import GridSpec
import os
import base64
from io import BytesIO
warnings.filterwarnings('ignore')

# 不需要中文显示设置，改为英文输出
plt.rcParams['font.sans-serif'] = ['Arial']  # 使用英文字体

def analyze_metrics_price_relationship(df, metrics, forward_periods=[1, 5, 10, 20], n_quantiles=5, window_size=60):
    """
    Comprehensive analysis of the relationship between a single metric and future price returns.
    
    Parameters:
    df - DataFrame containing price data, must have 'close' column
    metrics - Metric series with the same length as df
    forward_periods - List of future time periods to analyze
    n_quantiles - Number of quantiles for quantile analysis
    window_size - Rolling window size for IC analysis
    """
    start_time = time.time()
    print(f"Starting analysis - Data length: {len(df)}, Forward periods: {forward_periods}")
    
    # Ensure metrics is a pandas Series with the same index as df
    if not isinstance(metrics, pd.Series):
        metrics = pd.Series(metrics, index=df.index)
    
    # Create result container
    results = {}
    
    # Analyze each forward period
    total_periods = len(forward_periods)
    for i, period in enumerate(forward_periods):
        period_start_time = time.time()
        print(f"\n[{i+1}/{total_periods}] =========== Analyzing period: {period} ===========")
        
        # Calculate future returns
        future_returns = df['close'].shift(-period) / df['close'] - 1
        
        # Remove NaN
        mask = ~(future_returns.isna() | metrics.isna())
        clean_metrics = metrics[mask]
        clean_returns = future_returns[mask]
        print(f"Valid data points: {len(clean_metrics)}")
        
        period_results = {}
        
        # 1. Quantile analysis
        print("Progress: 10% - Starting quantile analysis...")
        quantile_results = analyze_by_quantiles(clean_metrics, clean_returns, n_quantiles)
        period_results['quantile_analysis'] = quantile_results
        
        # Print quantile analysis results
        print("\n1. Quantile Analysis Results:")
        print(quantile_results[['quantile', 'mean_return', 'std_return', 'count', 'positive_pct']])
        
        # 2. Linear regression analysis
        print("Progress: 30% - Starting linear regression analysis...")
        reg_results = analyze_with_linear_regression(clean_metrics, clean_returns)
        period_results['regression'] = reg_results
        
        # Print linear regression results
        print("\n2. Linear Regression Results (Alpha-Beta analysis):")
        print(f"Alpha (Intercept): {reg_results['intercept']:.6f}")
        print(f"Beta (Slope): {reg_results['slope']:.6f}")
        print(f"R² Score: {reg_results['r2_score']:.4f}")
        print(f"p-value: {reg_results['p_value']:.4f}")
        
        # 3. Random forest analysis
        print("Progress: 50% - Starting random forest analysis (may take some time)...")
        rf_start_time = time.time()
        rf_results = analyze_with_random_forest(clean_metrics, clean_returns)
        period_results['random_forest'] = rf_results
        
        # Print random forest results
        print(f"\n3. Random Forest Analysis Results (Time: {time.time() - rf_start_time:.2f}s):")
        print(f"R² Score: {rf_results['r2_score']:.4f}")
        print("Feature importance ranking:")
        for feature, importance in rf_results['feature_importance'].items():
            print(f"  {feature}: {importance:.4f}")
        
        # 4. IC analysis
        print("Progress: 75% - Starting IC analysis...")
        ic_results = analyze_ic(clean_metrics, clean_returns, window_size)
        period_results['ic_analysis'] = ic_results
        
        # Print IC analysis results
        print("\n4. IC Analysis Results:")
        print(f"IC Mean: {ic_results['ic_mean']:.4f}")
        print(f"IC Std: {ic_results['ic_std']:.4f}")
        print(f"ICIR: {ic_results['icir']:.4f}")
        print(f"IC t-statistic: {ic_results['ic_t_stat']:.4f}")
        print(f"IC p-value: {ic_results['ic_p_value']:.4f}")
        
        # 5. Autocorrelation analysis
        print("Progress: 90% - Starting autocorrelation analysis...")
        autocorr_results = analyze_autocorrelation(clean_metrics)
        period_results['autocorrelation'] = autocorr_results
        
        # Print autocorrelation results
        print("\n5. Autocorrelation Analysis Results:")
        lags = [1, 5, 10, 20]
        for lag in lags:
            if lag in autocorr_results:
                print(f"Lag {lag} autocorrelation: {autocorr_results[lag]:.4f}")
        
        # Store results
        results[period] = period_results
        
        period_time = time.time() - period_start_time
        print(f"Period {period} analysis completed - Time: {period_time:.2f}s")
        print(f"Total progress: {(i+1)/total_periods*100:.0f}%")
    
    total_time = time.time() - start_time
    print(f"\nAll analysis completed - Total time: {total_time:.2f}s")
    
    # Save raw data for reporting
    results['raw_data'] = {
        'metrics': metrics,
        'df': df
    }
    
    return results

def analyze_by_quantiles(metrics, returns, n_quantiles=5):
    """Analyze returns based on metrics quantiles"""
    # Calculate metrics quantiles
    try:
        # Try to use qcut to create equal-frequency quantiles
        quantiles = pd.qcut(metrics, n_quantiles, duplicates='drop')
    except ValueError:
        # If too many duplicate values, use cut to create equal-width quantiles
        quantiles = pd.cut(metrics, n_quantiles)
    
    # Create result container
    result_data = []
    
    # Calculate statistics for each quantile
    for q in sorted(quantiles.unique()):
        q_returns = returns[quantiles == q]
        result_data.append({
            'quantile': str(q),
            'min_value': q.left,
            'max_value': q.right,
            'mean_return': q_returns.mean(),
            'median_return': q_returns.median(),
            'std_return': q_returns.std(),
            'count': len(q_returns),
            'positive_pct': (q_returns > 0).mean()
        })
    
    return pd.DataFrame(result_data)

def analyze_with_linear_regression(metrics, returns):
    """Analyze the relationship between metrics and future returns using simple linear regression"""
    # Prepare data
    X = sm.add_constant(metrics.values)  # Add constant for intercept
    y = returns.values
    
    # Time series cross-validation
    tscv = TimeSeriesSplit(n_splits=5)
    r2_scores = []
    slopes = []
    intercepts = []
    p_values = []
    
    for fold, (train_idx, test_idx) in enumerate(tscv.split(X)):
        print(f"  Linear regression: Cross-validation fold {fold+1}/5")
        # Train OLS model
        model = sm.OLS(y[train_idx], X[train_idx])
        results = model.fit()
        
        # Record results
        r2_scores.append(results.rsquared)
        intercepts.append(results.params[0])  # Alpha
        slopes.append(results.params[1])  # Beta
        p_values.append(results.pvalues[1])  # p-value for metrics coefficient
    
    # Calculate means
    mean_r2 = np.mean(r2_scores)
    mean_slope = np.mean(slopes)
    mean_intercept = np.mean(intercepts)
    mean_p_value = np.mean(p_values)
    
    return {
        'r2_score': mean_r2,
        'slope': mean_slope,
        'intercept': mean_intercept,
        'p_value': mean_p_value
    }

def analyze_with_random_forest(metrics, returns):
    """Analyze non-linear relationship between metrics and future returns using random forest"""
    # Prepare data
    X = metrics.values.reshape(-1, 1)  # Reshape to feature matrix
    y = returns.values
    
    # Create non-linear features
    print("  Creating non-linear feature transformations...")
    X_nonlinear = pd.DataFrame({
        'original': X.flatten(),  # Original feature
        'squared': X.flatten() ** 2,  # Square term
        'cubed': X.flatten() ** 3,  # Cube term
        'abs': np.abs(X.flatten()),  # Absolute value
        'sin': np.sin(X.flatten()),  # Sine transformation
        'sign': np.sign(X.flatten()),  # Sign function
        'log': np.log1p(np.abs(X.flatten())),  # Log transformation
        'sqrt': np.sqrt(np.abs(X.flatten()))  # Square root transformation
    })
    
    # Time series cross-validation
    tscv = TimeSeriesSplit(n_splits=5)
    r2_scores = []
    importance_dfs = []
    
    for fold, (train_idx, test_idx) in enumerate(tscv.split(X_nonlinear)):
        fold_start_time = time.time()
        print(f"  Random forest: Cross-validation fold {fold+1}/5")
        
        # Train random forest model
        model = RandomForestRegressor(n_estimators=100, max_depth=5, random_state=42)
        model.fit(X_nonlinear.iloc[train_idx], y[train_idx])
        
        # Record results
        r2_scores.append(model.score(X_nonlinear.iloc[test_idx], y[test_idx]))
        # Feature importance
        importance_dfs.append(pd.Series(model.feature_importances_, index=X_nonlinear.columns))
        
        print(f"    Completed - Time: {time.time() - fold_start_time:.2f}s")
    
    # Calculate means
    mean_r2 = np.mean(r2_scores)
    mean_importance = pd.concat(importance_dfs, axis=1).mean(axis=1).sort_values(ascending=False)
    
    return {
        'r2_score': mean_r2,
        'feature_importance': mean_importance
    }

def analyze_ic(metrics, returns, window_size=60):
    """Calculate IC (Information Coefficient) statistics"""
    # Calculate full sample IC
    correlation = stats.spearmanr(metrics, returns)
    full_ic = correlation.correlation
    full_ic_p_value = correlation.pvalue
    
    # Initialize results
    rolling_ic = pd.Series(index=metrics.index)
    
    print(f"  Calculating rolling IC (window size={window_size})...")
    # Calculate rolling correlation coefficients (IC)
    progress_step = max(1, len(metrics) // 10)
    for i in range(window_size, len(metrics)):
        if i % progress_step == 0:
            print(f"    Processing data point: {i}/{len(metrics)} ({i/len(metrics)*100:.1f}%)")
        window_metrics = metrics.iloc[i-window_size:i]
        window_returns = returns.iloc[i-window_size:i]
        if len(window_metrics) > 0 and len(window_returns) > 0:
            correlation = stats.spearmanr(window_metrics, window_returns)
            rolling_ic.iloc[i] = correlation.correlation
    
    # Remove NaN
    rolling_ic = rolling_ic.dropna()
    
    # Calculate IC statistics
    ic_mean = rolling_ic.mean()
    ic_std = rolling_ic.std()
    icir = ic_mean / ic_std if ic_std != 0 else 0
    
    # Calculate IC statistical significance
    t_stat = ic_mean / (ic_std / np.sqrt(len(rolling_ic))) if ic_std != 0 else 0
    p_value = 2 * (1 - stats.t.cdf(abs(t_stat), len(rolling_ic) - 1))
    
    return {
        'full_ic': full_ic,
        'full_ic_p_value': full_ic_p_value,
        'ic_mean': ic_mean,
        'ic_std': ic_std,
        'icir': icir,
        'ic_t_stat': t_stat,
        'ic_p_value': p_value,
        'rolling_ic': rolling_ic
    }

def analyze_autocorrelation(metrics, lags=[1, 5, 10, 20]):
    """Analyze autocorrelation of metrics"""
    result = {}
    for lag in lags:
        if lag < len(metrics):
            result[lag] = metrics.autocorr(lag)
    return result

def figure_to_base64(fig):
    """Convert a matplotlib figure to base64 string for embedding in markdown"""
    buf = BytesIO()
    fig.savefig(buf, format='png', dpi=150)
    buf.seek(0)
    img_str = base64.b64encode(buf.read()).decode('utf-8')
    buf.close()
    return img_str

def generate_markdown_report(results, metrics_name="Random Metrics", output_dir="reports"):
    """Generate a comprehensive analysis report in Markdown format with embedded images"""
    print("\nGenerating markdown report...")
    
    # Create output directory
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Get current time for report ID
    report_id = time.strftime("%Y%m%d_%H%M%S")
    report_file = f"{output_dir}/{metrics_name.replace(' ', '_')}_{report_id}.md"
    
    # Get the forward periods (integer keys)
    all_keys = list(results.keys())
    forward_periods = []
    for k in all_keys:
        if isinstance(k, int):
            forward_periods.append(k)
    forward_periods.sort()
    
    # Create summary data
    summary_data = []
    for period in forward_periods:
        period_results = results[period]
        
        # Extract summary information from each analysis
        reg_results = period_results['regression']
        ic_results = period_results['ic_analysis']
        rf_results = period_results['random_forest']
        
        summary_data.append({
            'Period': period,
            'Beta': reg_results['slope'],
            'R² (Linear)': reg_results['r2_score'],
            'p-value (Linear)': reg_results['p_value'],
            'R² (Non-linear)': rf_results['r2_score'],
            'IC Mean': ic_results['ic_mean'],
            'ICIR': ic_results['icir'],
            'IC p-value': ic_results['ic_p_value']
        })
    
    summary_df = pd.DataFrame(summary_data)
    
    # Start writing the markdown report
    with open(report_file, 'w') as f:
        # Write header
        f.write(f"# Metric-Price Relationship Analysis Report: {metrics_name}\n\n")
        f.write(f"*Generated on: {time.strftime('%Y-%m-%d %H:%M:%S')}*\n\n")
        
        # ===== KEY FINDINGS SECTION (ADDED AT VERY BEGINNING) =====
        f.write("## Key Findings\n\n")
        
        # Determine if non-linear relationship exists - 使用更严格的标准
        has_nonlinear = False
        max_nonlinear_r2 = 0
        for period in forward_periods:
            linear_r2 = summary_df.loc[summary_df['Period'] == period, 'R² (Linear)'].values[0]
            nonlinear_r2 = summary_df.loc[summary_df['Period'] == period, 'R² (Non-linear)'].values[0]
            pvalue = summary_df.loc[summary_df['Period'] == period, 'p-value (Linear)'].values[0]
            max_nonlinear_r2 = max(max_nonlinear_r2, nonlinear_r2)
            # 非线性关系需要：1. 显著高于线性 2. 绝对值够高 3. 统计显著
            if nonlinear_r2 - linear_r2 > 0.05 and nonlinear_r2 > 0.1 and pvalue < 0.05:
                has_nonlinear = True
                break
        
        # Determine if linear relationship exists - 使用更严格的标准
        has_linear = False
        max_linear_r2 = 0
        min_pvalue = 1.0
        for period in forward_periods:
            linear_r2 = summary_df.loc[summary_df['Period'] == period, 'R² (Linear)'].values[0]
            pvalue = summary_df.loc[summary_df['Period'] == period, 'p-value (Linear)'].values[0]
            max_linear_r2 = max(max_linear_r2, linear_r2)
            min_pvalue = min(min_pvalue, pvalue)
            if linear_r2 > 0.1 and pvalue < 0.01:  # 更严格的阈值
                has_linear = True
                break
        
        # Check feature importance - 使用更严格的标准
        significant_nonlinear_features = False
        for period in forward_periods:
            feature_importance = results[period]['random_forest']['feature_importance']
            # 检查非线性特征是否真的重要
            nonlinear_features = [f for f in feature_importance.index if f != 'original']
            for feature in nonlinear_features:
                if feature in feature_importance and feature_importance[feature] > 0.3:  # 更高的阈值
                    significant_nonlinear_features = True
                    break
        
        # Determine if IC relationship is significant - 使用更严格的标准
        significant_ic = False
        for period in forward_periods:
            ic_mean = abs(summary_df.loc[summary_df['Period'] == period, 'IC Mean'].values[0])
            ic_pvalue = summary_df.loc[summary_df['Period'] == period, 'IC p-value'].values[0]
            if ic_mean > 0.1 and ic_pvalue < 0.01:  # 更严格的阈值
                significant_ic = True
                break
        
        # 综合判断是否有显著关系
        has_significant_relationship = has_linear or has_nonlinear or significant_ic
        
        # 确定最佳预测周期 - 使用更严格的标准
        best_period = None
        best_metric = 0
        if has_significant_relationship:
            for period in forward_periods:
                linear_r2 = summary_df.loc[summary_df['Period'] == period, 'R² (Linear)'].values[0]
                nonlinear_r2 = summary_df.loc[summary_df['Period'] == period, 'R² (Non-linear)'].values[0]
                ic_mean = abs(summary_df.loc[summary_df['Period'] == period, 'IC Mean'].values[0])
                
                # 使用综合指标
                metric = max(linear_r2, nonlinear_r2) * 0.5 + ic_mean * 5
                if metric > best_metric:
                    best_metric = metric
                    best_period = period
        
        # 确定交易建议 - 使用更严格的标准
        trading_direction = "Insufficient evidence for clear trading signal"
        
        if has_significant_relationship and best_period:
            # 获取最佳周期的分位数结果
            quantile_results = results[best_period]['quantile_analysis']
            beta = summary_df.loc[summary_df['Period'] == best_period, 'Beta'].values[0]
            
            # 检查分位数返回的统计显著性
            if len(quantile_results) >= 3:
                highest_return = quantile_results.iloc[-1]['mean_return'] 
                lowest_return = quantile_results.iloc[0]['mean_return']
                std_highest = quantile_results.iloc[-1]['std_return'] / np.sqrt(quantile_results.iloc[-1]['count'])
                std_lowest = quantile_results.iloc[0]['std_return'] / np.sqrt(quantile_results.iloc[0]['count'])
                
                # 计算t统计量检验最高和最低分位数的差异
                t_stat = (highest_return - lowest_return) / np.sqrt(std_highest**2 + std_lowest**2)
                is_significant_diff = abs(t_stat) > 2.576  # 99%置信度
                
                if is_significant_diff:
                    if highest_return > lowest_return:
                        trading_direction = "LONG when metric is high, SHORT when metric is low"
                    elif lowest_return > highest_return:
                        trading_direction = "SHORT when metric is high, LONG when metric is low"
        
        # 写入结论
        # 首先判断是否存在任何显著关系
        if not has_significant_relationship:
            f.write("**No significant relationship detected** between this metric and future returns. ")
            
            # 添加对负R²值的特别解释
            negative_r2_exists = False
            for period in forward_periods:
                linear_r2 = summary_df.loc[summary_df['Period'] == period, 'R² (Linear)'].values[0]
                nonlinear_r2 = summary_df.loc[summary_df['Period'] == period, 'R² (Non-linear)'].values[0]
                if linear_r2 < 0 or nonlinear_r2 < 0:
                    negative_r2_exists = True
                    break
            
            if negative_r2_exists:
                f.write("**Note: Negative R² values observed**, which indicates models perform worse than simply using mean values. ")
                f.write("This strongly suggests the metric has no predictive relationship with future returns and may introduce noise if used for trading decisions.\n\n")
            
            f.write(f"Maximum linear R² across all periods: {max_linear_r2:.4f}, minimum p-value: {min_pvalue:.4f}, ")
            f.write(f"maximum non-linear R²: {max_nonlinear_r2:.4f}.\n\n")
            f.write("The metric does not have reliable predictive power for future returns.\n\n")
            f.write("**Trading recommendation**: No trading action recommended based on this metric.\n\n")
        else:
            # 如果有显著关系，区分线性和非线性
            if has_nonlinear and has_linear:
                f.write("**Both linear and non-linear relationships detected** between this metric and future returns. ")
                f.write("The relationship shows significant predictive power in both linear and non-linear models.\n\n")
            elif has_nonlinear:
                f.write("**Non-linear relationship detected** between this metric and future returns. ")
                f.write("The relationship cannot be adequately captured by linear models alone.\n\n")
            elif has_linear:
                f.write("**Linear relationship detected** between this metric and future returns. ")
                f.write("A simple linear model adequately captures the relationship.\n\n")
            elif significant_ic:
                f.write("**Rank correlation detected** between this metric and future returns. ")
                f.write("The relationship is better represented by rank correlation than regression models.\n\n")
            
            # 仅当有显著差异时才提供交易建议
            if trading_direction != "Insufficient evidence for clear trading signal":
                f.write(f"**Trading recommendation**: {trading_direction}\n\n")
            else:
                f.write("**Trading recommendation**: Not enough evidence to provide a clear trading signal.\n\n")
            
            if best_period:
                f.write(f"**Optimal forecast horizon**: {best_period} periods\n\n")
        
        # ===== END OF KEY FINDINGS =====
        
        # ===== METHODOLOGY SECTION =====
        f.write("## Methodology\n\n")
        f.write("This section explains how to interpret the results and the criteria used to determine relationships between the metric and future returns.\n\n")
        
        f.write("### Non-linearity Decision Framework\n\n")
        f.write("The primary decision rule for detecting a non-linear relationship is:\n\n")
        f.write("**Random Forest R² - Linear R² > 0.02** = Non-linear relationship likely present\n\n")
        f.write("Supporting evidence includes:\n")
        f.write("- Non-linear features (squared, cubed, etc.) having high importance in random forest model\n")
        f.write("- Non-monotonic patterns in quantile mean returns (U-shaped or inverted U-shaped patterns)\n\n")
        
        # 添加指标解释部分
        f.write("### Key Metrics Explained\n\n")
        
        f.write("#### R² (Coefficient of Determination)\n\n")
        f.write("**Definition**: Measures the proportion of the variance in the dependent variable (future returns) that is predictable from the independent variable (metric).\n\n")
        f.write("**Formula**: $R^2 = 1 - \\frac{\\sum(y_i - \\hat{y}_i)^2}{\\sum(y_i - \\bar{y})^2}$\n\n")
        f.write("Where:\n")
        f.write("- $y_i$ is the actual return value\n")
        f.write("- $\\hat{y}_i$ is the predicted return value\n")
        f.write("- $\\bar{y}$ is the mean of actual returns\n\n")
        f.write("**Interpretation**:\n")
        f.write("- R² = 0: The model explains none of the variability in returns\n")
        f.write("- R² = 1: The model explains all the variability in returns\n")
        f.write("- Higher R² values indicate better predictive power\n")
        f.write("- **R² < 0 (Negative)**: The model performs worse than simply using the mean value. This indicates that the model is completely failing to capture the relationship and actually adds noise. In financial analysis, negative R² suggests the metric has no predictive value for future returns and might be misleading if used for decision making.\n\n")
        
        f.write("#### Beta Coefficient\n\n")
        f.write("**Definition**: Measures the expected change in future returns for a one-unit change in the metric value.\n\n")
        f.write("**Formula**: $\\beta = \\frac{\\sum(x_i - \\bar{x})(y_i - \\bar{y})}{\\sum(x_i - \\bar{x})^2}$\n\n")
        f.write("Where:\n")
        f.write("- $x_i$ is the metric value\n")
        f.write("- $\\bar{x}$ is the mean of the metric values\n")
        f.write("- $y_i$ is the actual return value\n")
        f.write("- $\\bar{y}$ is the mean of actual returns\n\n")
        f.write("**Interpretation**:\n")
        f.write("- Positive beta: Higher metric values predict higher returns\n")
        f.write("- Negative beta: Higher metric values predict lower returns\n")
        f.write("- Magnitude indicates sensitivity of returns to the metric\n\n")
        
        f.write("#### p-value\n\n")
        f.write("**Definition**: Represents the probability that the observed relationship between the metric and future returns occurred by random chance.\n\n")
        f.write("**Formula**: Calculated from the t-statistic: $p = 2 \\times (1 - CDF(|t|))$\n\n")
        f.write("Where:\n")
        f.write("- $t = \\frac{\\beta}{SE(\\beta)}$ is the t-statistic\n")
        f.write("- $SE(\\beta)$ is the standard error of the beta coefficient\n")
        f.write("- $CDF$ is the cumulative distribution function of the t-distribution\n\n")
        f.write("**Interpretation**:\n")
        f.write("- p < 0.05: Statistically significant relationship (95% confidence)\n")
        f.write("- p < 0.01: Highly significant relationship (99% confidence)\n")
        f.write("- p < 0.001: Very highly significant relationship (99.9% confidence)\n\n")
        
        f.write("#### Information Coefficient (IC)\n\n")
        f.write("**Definition**: Measures the rank correlation between the metric and future returns, indicating predictive power without assuming linearity.\n\n")
        f.write("**Formula**: Spearman rank correlation: $IC = \\frac{\\sum(R(x_i) - \\overline{R(x)})(R(y_i) - \\overline{R(y)})}{\\sqrt{\\sum(R(x_i) - \\overline{R(x)})^2 \\sum(R(y_i) - \\overline{R(y)})^2}}$\n\n")
        f.write("Where:\n")
        f.write("- $R(x_i)$ is the rank of the metric value\n")
        f.write("- $R(y_i)$ is the rank of the return value\n")
        f.write("- $\\overline{R(x)}$ and $\\overline{R(y)}$ are the mean ranks\n\n")
        f.write("**Interpretation**:\n")
        f.write("- IC = 0: No rank correlation between metric and returns\n")
        f.write("- IC = 1: Perfect positive rank correlation\n")
        f.write("- IC = -1: Perfect negative rank correlation\n")
        f.write("- In finance, |IC| > 0.05 is considered meaningful\n\n")
        
        f.write("#### Information Coefficient IR (ICIR)\n\n")
        f.write("**Definition**: Measures the consistency of the IC over time.\n\n")
        f.write("**Formula**: $ICIR = \\frac{Mean(IC)}{StdDev(IC)}$\n\n")
        f.write("**Interpretation**:\n")
        f.write("- Higher ICIR indicates more consistent predictive power\n")
        f.write("- ICIR > 0.5 suggests reliable predictive ability\n\n")
        
        f.write("#### Feature Importance\n\n")
        f.write("**Definition**: Measures the contribution of each feature in the random forest model to prediction accuracy.\n\n")
        f.write("**Calculation Method**: In random forests, importance is calculated as either:\n")
        f.write("1. Mean decrease in impurity (Gini importance)\n")
        f.write("2. Mean decrease in accuracy when the feature is permuted\n\n")
        f.write("**Interpretation**:\n")
        f.write("- Higher values indicate more important features\n")
        f.write("- When non-linear transformations (squared, cubed, etc.) have higher importance than the original feature, it suggests a non-linear relationship\n\n")
        
        f.write("#### Autocorrelation\n\n")
        f.write("**Definition**: Measures the correlation between a metric's current value and its past values at different time lags.\n\n")
        f.write("**Formula**: $AC(k) = \\frac{\\sum_{t=k+1}^{n} (x_t - \\bar{x})(x_{t-k} - \\bar{x})}{\\sum_{t=1}^{n} (x_t - \\bar{x})^2}$\n\n")
        f.write("Where:\n")
        f.write("- $x_t$ is the metric value at time t\n")
        f.write("- $\\bar{x}$ is the mean of the metric values\n")
        f.write("- $k$ is the lag\n\n")
        f.write("**Interpretation**:\n")
        f.write("- AC > 0: The metric shows persistence (trending behavior)\n")
        f.write("- AC < 0: The metric shows mean-reverting behavior\n")
        f.write("- |AC| close to 1: Strong autocorrelation\n")
        f.write("- |AC| close to 0: Weak autocorrelation\n\n")
        
        f.write("### How to Read This Report\n\n")
        f.write("The report is organized by forward period, with analysis methods that identify different aspects of the relationship:\n\n")
        f.write("1. **Quantile Analysis**: Shows how returns distribute across different ranges of the metric value\n")
        f.write("2. **Linear Regression**: Provides baseline linear relationship measurement\n")
        f.write("3. **Random Forest Analysis**: Detects non-linear relationships using feature transformations\n")
        f.write("4. **Information Coefficient (IC) Analysis**: Measures rank correlation\n")
        f.write("5. **Autocorrelation Analysis**: Shows metric persistence over time\n\n")
        
        f.write("### Analysis Interpretation Guidelines\n\n")
        
        f.write("#### Linear vs. Non-linear Relationship\n\n")
        f.write("- **Linear relationship**: Linear R² > 0.04 AND p-value < 0.05 AND (Random Forest R² - Linear R² < 0.02)\n")
        f.write("- **Non-linear relationship**: (Random Forest R² - Linear R² > 0.02) OR (Non-linear features have high importance)\n")
        f.write("- **No significant relationship**: Linear p-value ≥ 0.05 AND Random Forest R² < 0.04 AND |IC Mean| < 0.05\n\n")
        
        f.write("#### Long vs. Short Signal Evaluation\n\n")
        f.write("To determine if the metric is better for long or short positions:\n\n")
        f.write("1. **Quantile Analysis**:\n")
        f.write("   - Highest quantile shows better returns than lowest: Good for LONG when metric is high\n")
        f.write("   - Lowest quantile shows better returns than highest: Good for SHORT when metric is high\n")
        f.write("   - U-shaped pattern: Good for BOTH long when metric is very high AND short when metric is very low\n")
        f.write("   - Inverted U-shaped pattern: Good for LONG when metric is in middle range\n\n")
        
        f.write("2. **Regression Beta**:\n")
        f.write("   - Positive Beta: Go LONG when metric is high, SHORT when metric is low\n")
        f.write("   - Negative Beta: Go SHORT when metric is high, LONG when metric is low\n\n")
        
        f.write("3. **Non-linear Relationships**:\n")
        f.write("   - Check feature importance directions and quantile patterns to determine optimal trading ranges\n\n")
        
        f.write("### Interpretation Thresholds\n\n")
        
        f.write("#### Linear Regression\n")
        f.write("- **R² < 0**: Negative relationship - model is worse than using the mean, indicating no predictive power and possibly misleading\n")
        f.write("- **R² < 0.01**: Very weak linear relationship\n")
        f.write("- **0.01 ≤ R² < 0.04**: Weak linear relationship\n")
        f.write("- **0.04 ≤ R² < 0.10**: Moderate linear relationship\n")
        f.write("- **R² ≥ 0.10**: Strong linear relationship\n")
        f.write("- **p-value < 0.05**: Statistically significant relationship\n\n")
        
        f.write("#### Random Forest (Non-linear Analysis)\n")
        f.write("- **R² < 0**: Negative relationship - model is worse than using the mean, suggesting overfitting or that non-linear transformations are not helpful\n")
        f.write("- **R² improvement over linear model > 0.05**: Strong evidence of non-linear relationship\n")
        f.write("- **Non-linear features having > 30% importance**: Suggests significant non-linear components\n\n")
        
        f.write("#### Information Coefficient (IC)\n")
        f.write("- **|IC Mean| < 0.02**: Negligible relationship\n")
        f.write("- **0.02 ≤ |IC Mean| < 0.05**: Weak relationship\n")
        f.write("- **0.05 ≤ |IC Mean| < 0.10**: Moderate relationship\n")
        f.write("- **|IC Mean| ≥ 0.10**: Strong relationship\n")
        f.write("- **ICIR < 0.5**: Inconsistent relationship\n")
        f.write("- **ICIR ≥ 0.5**: Consistent relationship\n\n")
        
        f.write("#### Autocorrelation\n")
        f.write("- **|AC| < 0.2**: Low persistence (rapid changes)\n")
        f.write("- **0.2 ≤ |AC| < 0.5**: Moderate persistence\n")
        f.write("- **|AC| ≥ 0.5**: High persistence (slow changes)\n")
        f.write("- **Negative AC**: Tendency to reverse\n\n")
        
        # ===== END OF METHODOLOGY SECTION =====
        
        f.write("## Executive Summary\n\n")
        f.write("This report analyzes the relationship between the given metric and future price returns ")
        f.write("using various analytical methods including quantile analysis, linear regression, ")
        f.write("random forest modeling, Information Coefficient (IC) analysis, and autocorrelation analysis.\n\n")
        
        # Write summary table
        f.write("### Summary of Findings\n\n")
        f.write(summary_df.to_markdown(index=False, floatfmt='.4f'))
        f.write("\n\n")
        
        # Generate and embed summary charts
        f.write("### Summary Charts\n\n")
        
        # Create summary figure with subplots
        plt.figure(figsize=(12, 10))
        gs = GridSpec(2, 2, figure=plt.gcf())
        
        # 1. Beta coefficients vs forward period
        ax1 = plt.subplot(gs[0, 0])
        ax1.plot(forward_periods, summary_df['Beta'], 'o-', label='Beta coefficient')
        ax1.set_title(f'{metrics_name}: Beta Coefficient by Forward Period')
        ax1.set_xlabel('Forward Period')
        ax1.set_ylabel('Beta Coefficient')
        ax1.grid(True)
        
        # 2. R² (Linear and Non-linear)
        ax2 = plt.subplot(gs[0, 1])
        ax2.plot(forward_periods, summary_df['R² (Linear)'], 'o-', label='Linear R²')
        ax2.plot(forward_periods, summary_df['R² (Non-linear)'], 'o-', label='Non-linear R²')
        ax2.set_title(f'{metrics_name}: Model Explanatory Power by Forward Period')
        ax2.set_xlabel('Forward Period')
        ax2.set_ylabel('R²')
        ax2.legend()
        ax2.grid(True)
        
        # 3. IC Mean and ICIR
        ax3 = plt.subplot(gs[1, 0])
        ax3.plot(forward_periods, summary_df['IC Mean'], 'o-', label='IC Mean')
        ax3.plot(forward_periods, summary_df['ICIR'], 'o-', label='ICIR')
        ax3.set_title(f'{metrics_name}: IC Statistics by Forward Period')
        ax3.set_xlabel('Forward Period')
        ax3.set_ylabel('Value')
        ax3.legend()
        ax3.grid(True)
        
        # 4. p-values (Linear and IC)
        ax4 = plt.subplot(gs[1, 1])
        ax4.plot(forward_periods, summary_df['p-value (Linear)'], 'o-', label='Linear p-value')
        ax4.plot(forward_periods, summary_df['IC p-value'], 'o-', label='IC p-value')
        ax4.axhline(y=0.05, color='r', linestyle='--', label='Significance threshold (0.05)')
        ax4.set_title(f'{metrics_name}: Statistical Significance by Forward Period')
        ax4.set_xlabel('Forward Period')
        ax4.set_ylabel('p-value')
        ax4.set_yscale('log')
        ax4.legend()
        ax4.grid(True)
        
        plt.tight_layout()
        
        # Convert figure to base64 and embed in markdown
        img_str = figure_to_base64(plt.gcf())
        plt.close()
        
        f.write(f"![Summary Charts](data:image/png;base64,{img_str})\n\n")
        
        # Detailed Analysis for each forward period
        for period in forward_periods:
            f.write(f"## Forward Period: {period}\n\n")
            period_results = results[period]
            
            # 1. Quantile Analysis
            f.write("### 1. Quantile Analysis\n\n")
            quantile_results = period_results['quantile_analysis']
            
            # Write quantile table
            f.write(quantile_results.to_markdown(index=False, floatfmt='.4f'))
            f.write("\n\n")
            
            # Create quantile analysis chart
            plt.figure(figsize=(12, 6))
            
            # Mean returns by quantile
            plt.subplot(1, 2, 1)
            sns.barplot(x=quantile_results.index, y='mean_return', data=quantile_results)
            plt.title(f'Forward Period {period}: Mean Return by Quantile')
            plt.xlabel('Quantile')
            plt.ylabel('Mean Return')
            plt.grid(True)
            
            # Positive return percentage by quantile
            plt.subplot(1, 2, 2)
            sns.barplot(x=quantile_results.index, y='positive_pct', data=quantile_results)
            plt.axhline(y=0.5, color='r', linestyle='--')
            plt.title(f'Forward Period {period}: Positive Return Percentage by Quantile')
            plt.xlabel('Quantile')
            plt.ylabel('Positive Return %')
            plt.grid(True)
            
            plt.tight_layout()
            
            # Convert to base64 and embed
            img_str = figure_to_base64(plt.gcf())
            plt.close()
            
            f.write(f"![Quantile Analysis](data:image/png;base64,{img_str})\n\n")
            
            # 2. Linear Regression Analysis
            f.write("### 2. Linear Regression Analysis\n\n")
            reg_results = period_results['regression']
            
            # Write regression results
            f.write(f"- Alpha (Intercept): {reg_results['intercept']:.6f}\n")
            f.write(f"- Beta (Slope): {reg_results['slope']:.6f}\n")
            f.write(f"- R² Score: {reg_results['r2_score']:.4f}\n")
            f.write(f"- p-value: {reg_results['p_value']:.4f}\n\n")
            
            # Create scatter plot and regression line
            plt.figure(figsize=(10, 6))
            
            # Get data
            metrics_data = results['raw_data']['metrics']
            df = results['raw_data']['df']
            future_returns = df['close'].shift(-period) / df['close'] - 1
            
            # Remove NaN
            mask = ~(future_returns.isna() | metrics_data.isna())
            clean_metrics = metrics_data[mask]
            clean_returns = future_returns[mask]
            
            # Scatter plot
            plt.scatter(clean_metrics, clean_returns, alpha=0.3, s=10)
            
            # Add regression line
            x_range = np.linspace(clean_metrics.min(), clean_metrics.max(), 100)
            y_pred = reg_results['intercept'] + reg_results['slope'] * x_range
            plt.plot(x_range, y_pred, 'r-', linewidth=2)
            
            plt.title(f'Forward Period {period}: Metric vs Future Returns Scatter Plot')
            plt.xlabel(f'{metrics_name}')
            plt.ylabel(f'Future {period}-period Return')
            plt.grid(True)
            
            # Add regression results text
            text = f"Beta = {reg_results['slope']:.6f}\n"
            text += f"Alpha = {reg_results['intercept']:.6f}\n"
            text += f"R² = {reg_results['r2_score']:.4f}\n"
            text += f"p-value = {reg_results['p_value']:.4f}"
            plt.annotate(text, xy=(0.05, 0.95), xycoords='axes fraction', 
                        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8),
                        verticalalignment='top')
            
            plt.tight_layout()
            
            # Convert to base64 and embed
            img_str = figure_to_base64(plt.gcf())
            plt.close()
            
            f.write(f"![Regression Analysis](data:image/png;base64,{img_str})\n\n")
            
            # 3. Random Forest Analysis
            f.write("### 3. Random Forest Analysis (Non-linear Relationships)\n\n")
            rf_results = period_results['random_forest']
            
            # Write RF results
            f.write(f"- R² Score: {rf_results['r2_score']:.4f}\n\n")
            f.write("#### Feature Importance\n\n")
            
            importance = rf_results['feature_importance']
            importance_df = pd.DataFrame({'Feature': importance.index, 'Importance': importance.values})
            f.write(importance_df.to_markdown(index=False, floatfmt='.4f'))
            f.write("\n\n")
            
            # Create feature importance chart
            plt.figure(figsize=(10, 6))
            importance.plot(kind='bar')
            plt.title(f'Forward Period {period}: Random Forest Feature Importance')
            plt.xlabel('Feature')
            plt.ylabel('Importance')
            plt.grid(True)
            plt.tight_layout()
            
            # Convert to base64 and embed
            img_str = figure_to_base64(plt.gcf())
            plt.close()
            
            f.write(f"![Feature Importance](data:image/png;base64,{img_str})\n\n")
            
            # 4. IC Analysis
            f.write("### 4. Information Coefficient (IC) Analysis\n\n")
            ic_results = period_results['ic_analysis']
            
            # Write IC results
            f.write(f"- IC Mean: {ic_results['ic_mean']:.4f}\n")
            f.write(f"- IC Standard Deviation: {ic_results['ic_std']:.4f}\n")
            f.write(f"- ICIR: {ic_results['icir']:.4f}\n")
            f.write(f"- IC t-statistic: {ic_results['ic_t_stat']:.4f}\n")
            f.write(f"- IC p-value: {ic_results['ic_p_value']:.4f}\n\n")
            
            # Create rolling IC chart
            plt.figure(figsize=(12, 10))
            plt.subplot(2, 1, 1)
            
            rolling_ic = ic_results['rolling_ic']
            plt.plot(rolling_ic.index, rolling_ic.values)
            plt.axhline(y=0, color='r', linestyle='--')
            plt.axhline(y=ic_results['ic_mean'], color='g', linestyle='-', 
                       label=f"Mean: {ic_results['ic_mean']:.4f}")
            
            plt.title(f'Forward Period {period}: Rolling IC Values')
            plt.xlabel('Date')
            plt.ylabel('IC Value')
            plt.legend()
            plt.grid(True)
            
            # IC histogram
            plt.subplot(2, 1, 2)
            sns.histplot(rolling_ic.dropna(), kde=True)
            plt.axvline(x=0, color='r', linestyle='--')
            plt.axvline(x=ic_results['ic_mean'], color='g', linestyle='-', 
                       label=f"Mean: {ic_results['ic_mean']:.4f}")
            
            plt.title(f'Forward Period {period}: IC Distribution')
            plt.xlabel('IC Value')
            plt.ylabel('Frequency')
            plt.legend()
            plt.grid(True)
            
            plt.tight_layout()
            
            # Convert to base64 and embed
            img_str = figure_to_base64(plt.gcf())
            plt.close()
            
            f.write(f"![IC Analysis](data:image/png;base64,{img_str})\n\n")
            
            # 5. Autocorrelation Analysis
            f.write("### 5. Autocorrelation Analysis\n\n")
            
            autocorr_results = period_results['autocorrelation']
            lags = sorted(list(autocorr_results.keys()))
            autocorr_values = [autocorr_results[lag] for lag in lags]
            
            # Write autocorrelation results
            autocorr_df = pd.DataFrame({'Lag': lags, 'Autocorrelation': autocorr_values})
            f.write(autocorr_df.to_markdown(index=False, floatfmt='.4f'))
            f.write("\n\n")
            
            # Create autocorrelation chart
            plt.figure(figsize=(10, 6))
            plt.bar(lags, autocorr_values)
            plt.axhline(y=0, color='r', linestyle='--')
            
            plt.title(f'{metrics_name}: Autocorrelation Coefficients')
            plt.xlabel('Lag')
            plt.ylabel('Autocorrelation Coefficient')
            plt.grid(True)
            plt.tight_layout()
            
            # Convert to base64 and embed
            img_str = figure_to_base64(plt.gcf())
            plt.close()
            
            f.write(f"![Autocorrelation](data:image/png;base64,{img_str})\n\n")
        
    print(f"Report generated successfully: {report_file}")
    return report_file

# 示例用法
if __name__ == "__main__":
    print("Loading data...")
    start_time = time.time()
    
    # Read price data
    df = pd.read_hdf('/Users/yifei/dev/quant/dataset/binance/BTCUSDT/spot/1h/BTCUSDT_1h_2020-12-16_to_2025-02-14.h5')
    print(f"Data loaded - {len(df)} records, Time: {time.time() - start_time:.2f}s")
    
    # Create an example metric (replace with real metrics in actual use)
    print("Creating example metrics data...")
    np.random.seed(42)
    # Use a very small dataset to speed up analysis and test the report generation
    sample_size = min(1000, len(df))  # Limit sample size for quick testing
    sample_df = df.iloc[-sample_size:].copy()
    random_metrics = np.random.randn(len(sample_df))
    
    print(f"Starting analysis - Sample size: {sample_size} (for quick testing)")
    # Run analysis
    results = analyze_metrics_price_relationship(sample_df, random_metrics)
    
    # Generate consolidated markdown report in English
    report_file = generate_markdown_report(results, metrics_name="Random Metric Test")
    
    print(f"\nAnalysis complete! Report generated at: {report_file}") 
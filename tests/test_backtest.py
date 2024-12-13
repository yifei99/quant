import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import unittest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

from backtest.backtest_engine import BacktestEngine
from backtest.strategy import FactorThresholdsStrategy
from backtest.performance import PerformanceEvaluator
from factors.factor_definitions import USDTIssuanceFactor
from factors.factor_engine import FactorEngine
from factors.optimizer import StrategyOptimizer

class TestBacktestIntegration(unittest.TestCase):
    def setUp(self):
        print("\n--- Setting up test environment ---")
        self.engine = BacktestEngine(initial_capital=100000.0, commission=0.001)
        self.evaluator = PerformanceEvaluator()

    def create_data(self, start_date_str, increments):
        """
        Create a 10-day test dataset.
        increments: a list of length 10, representing daily increments in USDT issuance 
        relative to a base of 1,000,000,000 USDT.
        """
        print(f"Creating data starting from {start_date_str} with increments: {increments}")
        start_date = datetime.strptime(start_date_str, "%Y-%m-%d")
        dates = [start_date + timedelta(days=i) for i in range(10)]
        timestamps = [int(d.timestamp()) for d in dates]

        # Base USDT and increments
        usdt_values = [1000000000 + inc for inc in increments]
        stablecoins_data = pd.DataFrame({
            'Timestamp': timestamps,
            'Date': dates,
            'USDT': usdt_values
        })

        # Simulate BTC prices for 10 days
        btc_prices = [30000 + i*500 for i in range(10)]  # ascending prices
        timestamps_ms = [int(d.timestamp() * 1000) for d in dates]
        btc_data = pd.DataFrame({
            'timestamp_start': timestamps_ms,
            'open': [p - 200 for p in btc_prices],
            'high': [p + 200 for p in btc_prices],
            'low': [p - 300 for p in btc_prices],
            'close': btc_prices,
            'volume': [1000 + i*100 for i in range(10)]
        })

        # Prepare USDT issuance data
        stablecoins_data['Date'] = pd.to_datetime(stablecoins_data['Date'])
        stablecoins_data = stablecoins_data.sort_values('Date')
        stablecoins_data['USDT_issuance'] = stablecoins_data['USDT'].diff().fillna(0)

        # Merge stablecoin issuance with BTC data
        data = btc_data.copy()
        data['Date'] = pd.to_datetime(data['timestamp_start'], unit='ms')
        data = pd.merge(data, stablecoins_data[['Date', 'USDT_issuance']], on='Date', how='left')
        data['USDT_issuance'] = data['USDT_issuance'].fillna(0)
        print("Data created successfully.\n")
        return data

    def test_incremental_issuance(self):
        print("\n=== Test: Incremental USDT Issuance ===")
        increments = [i * 10000000 for i in range(10)]  # 0,10e6,20e6,...90e6
        data = self.create_data("2021-03-31", increments)
        self.run_test_scenario(data, scenario_name="Incremental Issuance")

    def test_stable_issuance(self):
        print("\n=== Test: Stable USDT Issuance ===")
        increments = [i * 10000 for i in range(10)]  # small increments: 0,10k,20k,...
        data = self.create_data("2021-03-31", increments)
        self.run_test_scenario(data, scenario_name="Stable Issuance")

    def test_volatile_issuance(self):
        print("\n=== Test: Volatile USDT Issuance ===")
        increments = [
            0, 
            -50000000,   # large drop
            100000000,   # large increase
            -20000000,
            50000000,
            0,
            -10000000,
            20000000,
            -5000000,
            70000000
        ]
        data = self.create_data("2021-03-31", increments)
        self.run_test_scenario(data, scenario_name="Volatile Issuance")

    def run_test_scenario(self, data, scenario_name):
        print(f"--- Running test scenario: {scenario_name} ---")

        # Initialize factor engine and factor
        print("Registering USDT factor and initializing factor engine...")
        factor_engine = FactorEngine()
        usdt_factor = USDTIssuanceFactor(name='usdt_issuance', threshold=10000000)
        factor_engine.register_factor(usdt_factor)

        # Initialize strategy
        print("Initializing strategy with the USDT factor...")
        strategy = FactorThresholdsStrategy(factors=[usdt_factor])

        # Run backtest
        print("Running backtest...")
        portfolio = self.engine.run_backtest(data, strategy, factor_engine=factor_engine)
        print("Backtest completed.")
        
        self.assertIn('Date', portfolio.columns, "Portfolio should have a 'Date' column after backtest.")
        self.assertIn('total', portfolio.columns, "Portfolio should have a 'total' column.")
        self.assertGreaterEqual(len(portfolio), 10, "Portfolio should cover at least 10 days of data.")

        # Evaluate performance
        print("Evaluating performance metrics...")
        metrics = self.evaluator.calculate_performance_metrics(portfolio)
        self.assertIn('Annualized Return', metrics, "Metrics dictionary should include 'Annualized Return'.")
        self.assertIn('Sharpe Ratio', metrics, "Metrics dictionary should include 'Sharpe Ratio'.")
        self.assertIn('Max Drawdown', metrics, "Metrics dictionary should include 'Max Drawdown'.")

        # Print out metrics for verbosity
        print(f"Performance Metrics for {scenario_name}:")
        for k, v in metrics.items():
            print(f"  {k}: {v}")
        
        # Test optimization
        print("Testing optimizer with a range of thresholds...")
        optimizer = StrategyOptimizer(engine=self.engine, evaluator=self.evaluator)
        threshold_values = [-15000000, -10000000, -5000000, 0, 5000000, 10000000]
        optimization_results = optimizer.optimize_threshold(
            data=data,
            threshold_values=threshold_values,
            factor_class=USDTIssuanceFactor,
            strategy_class=FactorThresholdsStrategy
        )
        self.assertTrue(len(optimization_results) > 0, "Optimization should yield at least one result.")

        # Print optimization results
        print("Optimization Results:")
        for th, sharpe in optimization_results.items():
            print(f"  Threshold {th}: Sharpe Ratio = {sharpe}")

        optimal_threshold, optimal_sharpe = optimizer.find_optimal_threshold(optimization_results)
        self.assertIn(optimal_threshold, optimization_results, "Optimal threshold should be present in optimization results.")
        self.assertIsInstance(optimal_sharpe, float, "Optimal sharpe should be a float value.")

        # Print the optimal result
        print(f"Optimal Threshold for {scenario_name}: {optimal_threshold}, Sharpe: {optimal_sharpe}")
        print(f"--- Scenario {scenario_name} completed successfully ---\n")


if __name__ == "__main__":
    # Run tests with higher verbosity
    unittest.main(verbosity=2)

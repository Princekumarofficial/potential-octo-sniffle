import unittest
import pandas as pd
from .engine.execution import execution_returns
from .engine.fees import intraday_charges
from .engine.ga_optimizer import evaluate

class ExecutionTests(unittest.TestCase):
    def test_round_trip_and_terminal_fee(self):
        ix = pd.to_datetime(["2024-01-01 10:00", "2024-01-01 10:01", "2024-01-01 15:15"])
        close = pd.Series([100., 110., 110.], index=ix)
        net = execution_returns(close, pd.Series(1., index=ix), capital=10000, allocation=.12)
        fees = intraday_charges(1200, "BUY")["fee"] + intraday_charges(1320, "SELL")["fee"]
        self.assertAlmostEqual((1+net).prod()*10000, 10120-fees)
        self.assertAlmostEqual(net.attrs["execution"]["fees"], fees)
        self.assertEqual(net.attrs["execution"]["fills"], 2)

    def test_no_overnight_gap_and_short_fees(self):
        ix = pd.to_datetime(["2024-01-01 10:00", "2024-01-01 15:15", "2024-01-02 10:00", "2024-01-02 15:15"])
        close = pd.Series([100., 100., 200., 200.], index=ix)
        net = execution_returns(close, pd.Series(-1., index=ix), capital=10000, allocation=.12)
        self.assertAlmostEqual((1+net).prod()*10000, 10000-net.attrs["execution"]["fees"])
        self.assertEqual(net.attrs["execution"]["fills"], 4)

    def test_optimizer_report_uses_fees(self):
        ix = pd.to_datetime(["2024-01-01 10:00", "2024-01-01 15:15"])
        frame = pd.DataFrame({"Close": [100., 101.]}, index=ix)
        def strategy(df, params, cost, finalize):
            signal = pd.Series(1., index=df.index)
            return finalize(df.Close.pct_change(), signal, cost), signal
        report = evaluate(frame, strategy, {}, .0005, report=True)
        self.assertGreater(report["fees"], 0)
        self.assertAlmostEqual(report["cumulative_return"], report["final_equity"]/report["initial_capital"]-1)

    def test_cash_has_no_charges(self):
        ix = pd.date_range("2024-01-01 10:00", periods=3, freq="min")
        net = execution_returns(pd.Series([100., 110., 120.], index=ix), pd.Series(0., index=ix))
        self.assertEqual(net.sum(), 0)
        self.assertEqual(net.attrs["execution"]["fees"], 0)

    def test_batch_months_compound_after_fees(self):
        from unittest.mock import patch
        from .batch import evaluate_stock
        ix = pd.to_datetime(["2024-01-01 10:00", "2024-02-01 10:00", "2024-02-01 15:15", "2024-03-01 10:00", "2024-03-01 15:15"])
        frame = pd.DataFrame({"Close": [100., 100., 110., 110., 120.]}, index=ix)
        stats = {"sharpe": 1., "sortino": 1., "dd": -.1}
        candidate = (1., {"tree": ["<", "rsi_14", 30]}, {"train": stats})
        def rule(df, params, cost, finalize):
            signal = pd.Series(1., index=df.index)
            return finalize(df.Close.pct_change(), signal, cost), signal
        with patch("full_predictor.GA_STRATEGY.batch.load_config", return_value={}), patch("full_predictor.GA_STRATEGY.batch.genetic_optimize", return_value={}), patch("full_predictor.GA_STRATEGY.batch.optimize_rules", return_value={"GP": candidate}), patch("full_predictor.GA_STRATEGY.batch.run_rule", side_effect=rule):
            monthly, summaries = evaluate_stock("TEST", frame, pd.period_range("2024-02", "2024-03", freq="M"), 1, 0, {})
        rows = [r for r in monthly if r["strategy"] == "GP"]
        summary = next(r for r in summaries if r["strategy"] == "GP")
        self.assertAlmostEqual(summary["cumulative_return"], (1+rows[0]["net_return"])*(1+rows[1]["net_return"])-1)
        self.assertAlmostEqual(summary["fees"], sum(r["fees"] for r in rows))
        self.assertEqual(rows[1]["initial_capital"], rows[0]["final_equity"])

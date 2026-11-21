import unittest
from unittest.mock import patch
from pathlib import Path

import numpy as np
import pandas as pd

from .portfolio import rank_months, simulate, prepare
from .engine.fees import intraday_charges, affordable_shares


class PortfolioTests(unittest.TestCase):
    def test_leverage_margin_and_realized_profit(self):
        dates = pd.to_datetime(["2024-04-01 09:16", "2024-04-01 15:15"]).as_unit("ns")
        market = {"A": (dates.asi8, np.array([100., 110.]), np.array([100., 110.]))}
        events = {dates[0].value: [("A", 1, 100.)], dates[1].value: [("A", 0, 110.)]}
        curve, fills = simulate(market, events, {dates[0].to_period("M"): {"A"}},
                                capital=10000, allocation=.12, leverage=5, allocation_mode="margin")
        self.assertEqual(fills.iloc[0].shares, 60)
        self.assertEqual(fills.iloc[0].margin, 1200)
        self.assertEqual(fills.iloc[0].financed, 4800)
        self.assertAlmostEqual(curve.equity.iloc[-1], 10600 - fills.fee.sum())
        self.assertEqual(curve.financed.iloc[-1], 0)
        _, capped = simulate(market, events, {dates[0].to_period("M"): {"A"}},
                             capital=10000, allocation=.12, leverage=5,
                             allocation_mode="margin", leverage_by_stock={"A": 2})
        self.assertEqual(capped.iloc[0].shares, 24)

    def test_targets_execute_at_next_open_and_exit_on_deselection(self):
        dates = pd.to_datetime(["2024-04-30 09:15", "2024-04-30 09:16", "2024-05-01 09:15"])
        frame = pd.DataFrame({"Open": [100., 101., 102.], "Close": [100., 101., 102.]}, index=dates)
        frame["High"] = frame.Close
        frame["Low"] = frame.Close
        rows = pd.DataFrame([
            dict(stock="A", month=pd.Period(month), parameters='{"rsi_period": 14}',
                 selected_strategy="RSI_Mean_Reversion")
            for month in ("2024-04", "2024-05")
        ])
        def fake_strategy(bars, params, cost, finalize):
            return None, pd.Series(1., index=bars.index)
        with patch("full_predictor.GA_STRATEGY.portfolio.load_data", return_value=frame), patch(
                "full_predictor.GA_STRATEGY.portfolio.core_strategies.run_rsi", side_effect=fake_strategy):
            _, events, _ = prepare(rows, Path("."), {pd.Period("2024-04"): {"A"}, pd.Period("2024-05"): set()})
        self.assertEqual(events[dates[0].value][0][1], 0)
        self.assertEqual(events[dates[1].value], [("A", 1, 101., False, False)])
        self.assertEqual(events[dates[2].value], [("A", 0, 102., False, False)])

    def test_fyers_charges_and_cash(self):
        buy = intraday_charges(40000, "BUY")
        sell = intraday_charges(42000, "SELL")
        self.assertEqual(buy["brokerage"], 12)
        self.assertEqual(sell["stt"], 10.5)
        self.assertEqual(buy["stamp"], 1.2)
        self.assertEqual(intraday_charges(100000, "BUY")["brokerage"], 20)
        self.assertAlmostEqual(buy["fee"] + sell["fee"], 43.8)
        self.assertEqual(affordable_shares(100, 1000, 1000), 9)

    def test_intraday_exit_and_component_reconciliation(self):
        dates = pd.to_datetime(["2024-04-01 09:16", "2024-04-01 15:15"]).as_unit("ns")
        market = {"A": (dates.asi8, np.array([100.,110.]), np.array([100.,110.]))}
        events = {dates[0].value: [("A",1,100.)], dates[1].value: [("A",0,110.)]}
        curve, fills = simulate(market, events, {dates[0].to_period("M"): {"A"}}, capital=10000)
        fees = intraday_charges(400,"BUY")["fee"] + intraday_charges(440,"SELL")["fee"]
        self.assertAlmostEqual(curve.equity.iloc[-1], 10040-fees)
        self.assertEqual(curve.positions.iloc[-1], 0)
        self.assertEqual(fills.iloc[-1].note, "SCHEDULED_EXIT")
        events[dates[1].value] = [("A",0,110.,False,True)]
        curve, fills = simulate(market, events, {dates[0].to_period("M"): {"A"}}, capital=10000)
        self.assertEqual(curve.positions.iloc[-1], 1)
        self.assertEqual(fills.iloc[-1].note, "ASSUMED_DELIVERY_CONVERSION")
        self.assertEqual(len(curve.attrs["overnight_positions"]), 1)

    def test_rank_uses_only_previous_three_months(self):
        months = pd.period_range("2024-01", periods=5, freq="M")
        frame = pd.DataFrame([
            dict(stock=stock, month=month, net_return=ret)
            for stock, ret in (("A", .1), ("B", .02))
            for month in months
        ])
        eligible, _ = rank_months(frame, 1)
        self.assertEqual(eligible[months[3]], {"A"})
        frame.loc[frame.month >= months[3], "net_return"] = 100
        updated, _ = rank_months(frame, 1)
        self.assertEqual(updated[months[3]], eligible[months[3]])
        incomplete = frame.loc[~(frame.stock.eq("A") & frame.month.eq(months[1]))]
        updated, _ = rank_months(incomplete, 1)
        self.assertEqual(updated[months[3]], {"B"})

    def test_cash_and_four_percent_allocation(self):
        dates = pd.to_datetime(["2024-04-01 09:15", "2024-04-01 09:16", "2024-04-02 09:15"]).as_unit("ns")
        market = {"A": (dates.asi8, np.array([100., 120., 110.]), np.array([101., 121., 110.]))}
        events = {dates[0].value: [("A", 1, 100.)], dates[2].value: [("A", 0, 110.)]}
        curve, fills = simulate(market, events, {pd.Period("2024-04"): {"A"}}, capital=10000, cost=.0005)
        self.assertEqual(fills.iloc[0].shares, 4)
        self.assertAlmostEqual(curve.equity.iloc[-1], 10000 + 40 - .2 - .22)
        self.assertTrue((curve.cash >= 0).all())
        self.assertAlmostEqual(curve.equity.iloc[0], 10000 - .2 + 4 * 21)

    def test_simultaneous_entries_cannot_overspend(self):
        date = pd.Timestamp("2024-04-01 09:15")
        market = {stock: (np.array([date.value]), np.array([100.]), np.array([100.]))
                  for stock in ("A", "B")}
        events = {date.value: [(stock, 1, 100.) for stock in market]}
        curve, fills = simulate(market, events, {date.to_period("M"): set(market)},
                                capital=1000, allocation=.75, cost=.0005)
        self.assertGreaterEqual(curve.cash.iloc[-1], 0)
        self.assertTrue((fills.shares * fills.price <= fills.allocation_budget).all())


if __name__ == "__main__":
    unittest.main()

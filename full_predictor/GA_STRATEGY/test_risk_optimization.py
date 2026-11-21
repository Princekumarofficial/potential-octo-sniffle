import unittest
from unittest.mock import patch
import pandas as pd
from .engine.risk import risk_candidates, apply_risk_management
from .engine.gp_optimizer import optimize_rules, run_rule
from .engine.ga_optimizer import genetic_optimize, _noop_finalize

class RiskSearchTests(unittest.TestCase):
    def frame(self):
        ix = pd.to_datetime(["2024-01-02 10:00", "2024-01-02 10:01", "2024-01-02 15:15"])
        return pd.DataFrame({"Close": [100., 101., 101.], "rsi_14": 20., "rsi_30": 20., "bb_position_20": .1, "bb_position_60": .1}, index=ix)

    def test_grid(self):
        grid = risk_candidates()
        self.assertEqual(len(grid), 9)
        for params in grid:
            self.assertAlmostEqual(params["take_profit"] / params["stop_loss"], params["risk_reward_ratio"])

    def test_session_resets_entry_price(self):
        ix = pd.to_datetime(["2024-01-01 10:00", "2024-01-01 15:15", "2024-01-02 10:00", "2024-01-02 10:01", "2024-01-02 15:15"])
        close = pd.Series([100., 100., 90., 90., 90.], index=ix)
        actual = apply_risk_management(close, pd.Series(1., index=ix), stop_loss=.005, take_profit=.01)
        self.assertEqual(actual.tolist(), [1., 0., 1., 1., 0.])

    def test_gp_saved_risk_changes_exit(self):
        frame = self.frame()
        tree = ("<", "rsi_14", 30)
        _, tight = run_rule(frame, {"tree": tree, "stop_loss": .005, "take_profit": .005}, 0, _noop_finalize)
        _, wide = run_rule(frame, {"tree": tree, "stop_loss": .005, "take_profit": .03}, 0, _noop_finalize)
        self.assertEqual(tight.tolist(), [1., 0., 0.])
        self.assertEqual(wide.tolist(), [1., 1., 0.])
        _, legacy = run_rule(frame, tree, 0, _noop_finalize)
        self.assertEqual(legacy.tolist(), wide.tolist())

    def test_gp_search_saves_and_reuses_training_choice(self):
        frame, heldout = self.frame(), self.frame()
        def evaluate(df, fn, params, cost, **kwargs):
            if df is heldout:
                self.assertEqual(params["stop_loss"], .01)
                self.assertEqual(params["risk_reward_ratio"], 3.)
            return {"sharpe": params["stop_loss"] + params["risk_reward_ratio"], "sortino": 1., "dd": -.1}
        with patch("full_predictor.GA_STRATEGY.engine.gp_optimizer.evaluate", side_effect=evaluate):
            result = optimize_rules({"train": frame, "test": heldout}, population_size=4, generations=1, include_tree=True)
        params = result["GP"][1]
        self.assertEqual(params["take_profit"], .03)
        self.assertIn("tree", params)

    def test_ga_joint_search(self):
        frame = self.frame()
        config = {"RSI": {"function": "run_rsi", "params": ["rsi_period"], "grid": {"rsi_period": [10, 20]}}}
        def evaluate(df, fn, params, cost, **kwargs):
            return {"sharpe": params["rsi_period"] + params["stop_loss"] + params["risk_reward_ratio"], "sortino": 1., "dd": -.1}
        with patch("full_predictor.GA_STRATEGY.engine.ga_optimizer.load_config", return_value=config), patch("full_predictor.GA_STRATEGY.engine.ga_optimizer.evaluate", side_effect=evaluate):
            result = genetic_optimize(frame, frame)
        params = result["RSI"][1]
        self.assertEqual(params["rsi_period"], 20)
        self.assertEqual(params["stop_loss"], .01)
        self.assertEqual(params["take_profit"], .03)

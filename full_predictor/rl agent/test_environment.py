import sys
import unittest
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import pandas as pd
from environment import TradingEnv
from GA_STRATEGY.engine.fees import intraday_charges


class Tests(unittest.TestCase):
    def env(self):
        times = pd.to_datetime(['2024-01-01 10:00', '2024-01-01 10:01', '2024-01-01 15:15'])
        return TradingEnv(times, np.array([[[100., 100.]], [[110., 111.]], [[120., 120.]]]), np.zeros((3, 1, 5)), 10000, .1)

    def test_next_open_and_close(self):
        env = self.env()
        env.step([1])
        self.assertEqual(env.fills[0]['price'], 110.)
        self.assertEqual(env.quantity[0], 9)
        _, _, done, _ = env.step([0])
        fees = intraday_charges(990., 'BUY')['fee'] + intraday_charges(1080., 'SELL')['fee']
        self.assertAlmostEqual(env.equity, 10090-fees)
        self.assertTrue(done)
        self.assertEqual(env.quantity[0], 0)

    def test_cash_and_sell_without_position(self):
        env = self.env()
        env.step([2])
        env.step([0])
        self.assertEqual(env.equity, 10000)
        self.assertEqual(env.fills, [])


class USMarketTests(unittest.TestCase):
    def test_us_afternoon_and_fees(self):
        from GA_STRATEGY.engine.fees import us_intraday_charges
        times = pd.to_datetime(['2024-07-01 15:30', '2024-07-01 15:31', '2024-07-01 15:55', '2024-07-01 16:00'])
        prices = np.array([[[100., 100.]], [[100., 100.]], [[110., 110.]], [[110., 110.]]])
        env = TradingEnv(times, prices, np.zeros((4, 1, 5)), 10000, .1, market='us')
        env.step([1])
        self.assertEqual(env.quantity[0], 10)
        self.assertEqual(env.fills[0]['fee'], us_intraday_charges(1000, 'BUY', 10)['fee'])
        env.step([0])
        self.assertEqual(env.quantity[0], 0)
        self.assertAlmostEqual(env.equity, 10100 - sum(f['fee'] for f in env.fills))
        self.assertEqual(env.fills[-1]['fee'], us_intraday_charges(1100, 'SELL', 10)['fee'])
        env.step([1])
        self.assertEqual(len(env.fills), 2)

    def test_us_entry_cutoff(self):
        times = pd.to_datetime(['2024-07-01 15:44', '2024-07-01 15:45', '2024-07-01 15:55'])
        env = TradingEnv(times, np.full((3, 1, 2), 100.), np.zeros((3, 1, 5)), market='us')
        env.step([1])
        self.assertEqual(len(env.fills), 0)

    def test_timezone_dst(self):
        from markets import local_index
        utc = pd.to_datetime(['2024-01-02 14:30', '2024-07-01 13:30'])
        local = local_index(utc, 'us', 'UTC')
        self.assertEqual(list(local.hour), [9, 9])
        self.assertEqual(list(local.minute), [30, 30])
        aware = utc.tz_localize('UTC')
        self.assertTrue(local.equals(local_index(aware, 'us')))


class AllStockTests(unittest.TestCase):
    def make_env(self, **kwargs):
        times = pd.to_datetime(['2024-01-02 10:00','2024-01-02 10:01','2024-01-02 10:02','2024-01-02 15:55'])
        prices = np.full((4, 2, 2), 100.)
        prices[2, :, :] = 90.
        return TradingEnv(times, prices, np.zeros((4,2,5)), 10000., .1, market='us', **kwargs)

    def test_position_cap_and_priority(self):
        env = self.make_env(max_positions=1)
        env.step([1,1], [0.2,0.8])
        self.assertEqual(env.quantity.tolist(), [0,10])

    def test_missing_stock_never_fills(self):
        available = np.ones((4,2),bool)
        available[1,0] = False
        env = self.make_env(availability=available)
        env.step([1,1])
        self.assertEqual(env.quantity[0],0)

    def test_drawdown_reduces_reward(self):
        plain = self.make_env()
        penalized = self.make_env(drawdown_penalty=1.)
        plain.step([1,0]); penalized.step([1,0])
        _, first, _, _ = plain.step([0,0])
        _, second, _, _ = penalized.step([0,0])
        self.assertLess(second, first)
        self.assertAlmostEqual(first-second, .01)


if __name__ == '__main__':
    unittest.main()

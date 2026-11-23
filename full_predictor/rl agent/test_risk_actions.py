import sys
import unittest
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import pandas as pd
from environment import TradingEnv
from train import Policy
from risk_actions import ACTION_COUNT, ACTION_RISK

class RiskTests(unittest.TestCase):
    def env(self, third=100., third_open=100.):
        times = pd.to_datetime(['2024-01-02 10:00','2024-01-02 10:01','2024-01-02 10:02','2024-01-02 15:55'])
        prices = np.array([[[100.,100.]],[[100.,100.]],[[third_open,third]],[[100.,100.]]])
        return TradingEnv(times, prices, np.zeros((4,1,9)), 10000., .1, market='us', risk_actions=True)

    def test_take_profit(self):
        env = self.env(third=101.)
        env.step([1])
        self.assertEqual(env.sl[0],.005)
        env.step([0])
        self.assertEqual(env.fills[-1]['reason'],'take_profit')
        self.assertEqual(env.quantity[0],0)

    def test_stop_gap_uses_open(self):
        env = self.env(third=98.,third_open=98.)
        env.step([1]);env.step([0])
        self.assertEqual(env.fills[-1]['reason'],'stop_loss')
        self.assertEqual(env.fills[-1]['price'],98.)

    def test_agent_can_exit_early(self):
        env = self.env()
        env.step([10]);env.step([2])
        self.assertEqual(env.fills[-1]['reason'],'agent')
        self.assertEqual(env.fills[-1]['take_profit'],.03)
        self.assertEqual(env.quantity[0],0)

    def test_risk_policy_gradients(self):
        import torch
        env = self.env()
        model = Policy(env.observation().shape[-1], ACTION_COUNT)
        dist,value = model(env.observation(),env.mask())
        self.assertEqual(dist.probs.shape,(1,11))
        action = dist.sample()
        (-dist.log_prob(action).sum()+value.square()).backward()
        self.assertTrue(all(torch.isfinite(p.grad).all() for p in model.parameters()))
        env.step([1])
        self.assertEqual(np.flatnonzero(env.mask()[0]).tolist(),[0,2])

if __name__ == '__main__': unittest.main()

import sys
import unittest
from pathlib import Path
import numpy as np
import torch
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from train import Policy
from test_environment import AllStockTests
from train_all import evaluate, stats


class BatchedTests(unittest.TestCase):
    def test_batch_joint_loss_has_finite_gradients(self):
        torch.manual_seed(42)
        model = Policy(13)
        obs = torch.zeros(8, 95, 13)
        hidden = model.body(obs)
        distribution = torch.distributions.Categorical(logits=model.actor(hidden))
        actions = distribution.sample()
        value = model.critic(hidden).squeeze(-1).mean(-1)
        advantage = torch.ones(8)*.01-value
        loss = -(distribution.log_prob(actions).sum(-1)*advantage.detach()).mean()+.5*advantage.square().mean()
        loss.backward()
        self.assertTrue(all(torch.isfinite(p.grad).all() for p in model.parameters()))

    def test_evaluation_finishes_and_liquidates(self):
        env = AllStockTests().make_env(max_positions=1, drawdown_penalty=1.)
        model = Policy(env.observation().shape[-1])
        report = evaluate(model, env)
        self.assertTrue(np.isfinite(report['final_equity']))
        self.assertEqual(env.quantity.sum(), 0)
        self.assertEqual(len(env.curve), len(env.times))


if __name__ == '__main__': unittest.main()

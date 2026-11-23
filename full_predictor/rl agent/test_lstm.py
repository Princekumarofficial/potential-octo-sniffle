import unittest
import numpy as np
import pandas as pd
import torch
from train import Policy
from lstm_policy import LSTMPolicy,SequenceEnv
from environment import TradingEnv
from analyze import HistoryView,explain

class LSTMTests(unittest.TestCase):
    def test_shapes_masks_and_history_gradient(self):
        torch.manual_seed(42)
        model=LSTMPolicy(15)
        x=torch.randn(2,3,4,15,requires_grad=True)
        mask=torch.ones(2,3,3,dtype=torch.bool);mask[:,:,2]=False
        dist,value=model(x,mask)
        self.assertEqual(dist.probs.shape,(2,3,3))
        self.assertEqual(value.shape,(2,))
        self.assertEqual(dist.probs[:,:,2].sum().item(),0)
        dist.probs[0,0,1].backward()
        self.assertGreater(x.grad[0,0,0].abs().sum().item(),0)
        self.assertEqual(x.grad[0,1].abs().sum().item(),0)

    def test_causal_padding_reset_and_missing_mask(self):
        times=pd.to_datetime(['2026-01-02 10:00','2026-01-02 10:03','2026-01-03 10:00','2026-01-03 15:55'])
        features=np.arange(4,dtype=np.float32).reshape(4,1,1)
        available=np.array([[True],[False],[True],[True]])
        base=TradingEnv(times,np.full((4,1,2),100.),features,market='us',availability=available)
        env=SequenceEnv(base,3)
        obs=env.reset()
        self.assertTrue((obs[:,:2]==0).all())
        obs,_,_,_=env.step([0])
        self.assertEqual(obs[0,-1,0],1)
        self.assertEqual(obs[0,-1,-2],0)
        self.assertAlmostEqual(obs[0,-1,-1],np.log1p(3),places=6)
        obs,_,_,_=env.step([0])
        self.assertTrue((obs[:,:2]==0).all())
        self.assertEqual(obs[0,-1,0],2)

    def test_history_explanation_keeps_previous_bars(self):
        model=LSTMPolicy(3)
        sequence=np.ones((4,3),dtype=np.float32)
        view=HistoryView(model,sequence)
        base,gradient,after,_=explain(view,sequence[-1],np.ones(3,bool),0)
        self.assertTrue(np.isfinite(gradient).all())
        with torch.no_grad(): expected=model(sequence[None],np.ones((1,3),bool))[0].probs[0,0].item()
        self.assertAlmostEqual(base,expected,places=6)

if __name__=='__main__': unittest.main()

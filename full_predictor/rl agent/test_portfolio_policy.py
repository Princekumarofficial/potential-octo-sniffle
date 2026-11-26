import unittest
import numpy as np
import torch
from portfolio_policy import PortfolioPolicy, greedy_actions
from risk_actions import BUY_ACTIONS
from analyze import ContextView, probabilities, explain

class JointTests(unittest.TestCase):
    def test_global_input_and_gradient(self):
        torch.manual_seed(3)
        model=PortfolioPolicy(16,95)
        x=torch.randn(2,95,16,requires_grad=True)
        dist,value=model(x,torch.ones(2,95,11,dtype=torch.bool))
        self.assertEqual(dist.probs.shape,(2,95,11))
        self.assertEqual(value.shape,(2,))
        gradient=torch.autograd.grad(dist.probs[0,0,0],x,retain_graph=True)[0]
        self.assertGreater(gradient[0,1].abs().sum().item(),0)
        loss=-(dist.log_prob(dist.sample()).sum(-1)*(.1-value).detach()).mean()+value.square().mean()
        loss.backward()
        self.assertTrue(all(torch.isfinite(p.grad).all() for p in model.parameters()))

    def test_buy_not_split_against_hold(self):
        p=torch.zeros(11)
        p[0]=.4
        p[BUY_ACTIONS]=.6/9
        self.assertEqual(p.argmax().item(),0)
        self.assertIn(greedy_actions(p,True).item(),BUY_ACTIONS)

    def test_mask_and_context_explanation(self):
        model=PortfolioPolicy(3,2)
        obs=np.zeros((2,3),dtype=np.float32)
        mask=np.ones((2,11),dtype=bool)
        mask[0,BUY_ACTIONS]=False
        dist,_=model(obs,mask)
        self.assertEqual(dist.probs[0,BUY_ACTIONS].sum().item(),0)
        view=ContextView(model,obs,mask,1)
        np.testing.assert_allclose(probabilities(view,obs[1:2],mask[1:2]).detach().numpy(),dist.probs[1:].detach().numpy(),atol=1e-7)
        _,grad,_,_=explain(view,obs[1],mask[1],0)
        self.assertTrue(np.isfinite(grad).all())

if __name__=='__main__': unittest.main()

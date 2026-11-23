import unittest
import numpy as np
import torch
from train import Policy
from analyze import explain, probabilities

class ExplanationTests(unittest.TestCase):
    def test_ablation_and_mask(self):
        model = Policy(3)
        obs=np.array([.1, .2, .3],dtype=np.float32)
        mask=np.array([True,True,False])
        base, grad, after, actions=explain(model,obs,mask,1)
        for i in range(3):
            x=obs.copy(); x[i]=0
            expected=probabilities(model,x[None],mask[None])[0,1].item()
            self.assertAlmostEqual(after[i],expected,places=6)
        self.assertTrue(np.isfinite(grad).all())
        self.assertFalse((actions==2).any())
        self.assertTrue(all(p.grad is None for p in model.parameters()))

    def test_zero_input_has_zero_ablation_effect(self):
        model=Policy(4,11)
        base,_,after,_=explain(model,np.zeros(4,dtype=np.float32),np.ones(11,bool),0)
        np.testing.assert_allclose(after,base,atol=1e-7)

if __name__=='__main__': unittest.main()

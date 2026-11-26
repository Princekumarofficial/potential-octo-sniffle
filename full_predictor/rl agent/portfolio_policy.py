"""One global portfolio network with conditional trade and bracket outputs."""
import torch
from torch import nn
from risk_actions import BUY_ACTIONS, ACTION_COUNT


class PortfolioPolicy(nn.Module):
    def __init__(self, inputs, stocks):
        super().__init__()
        self.inputs, self.stocks = inputs, stocks
        self.body = nn.Sequential(nn.Linear(stocks*inputs,256),nn.Tanh(),nn.Linear(256,256),nn.Tanh())
        self.signal_head = nn.Linear(256,stocks*3)
        self.risk_head = nn.Linear(256,stocks*9)
        self.critic = nn.Linear(256,1)

    def forward(self, obs, mask):
        device = next(self.parameters()).device
        obs = torch.as_tensor(obs,dtype=torch.float32,device=device)
        mask = torch.as_tensor(mask,dtype=torch.bool,device=device)
        if obs.shape[-2:] != (self.stocks,self.inputs):
            raise ValueError("Expected complete stock universe in saved ordering")
        hidden = self.body(obs.flatten(-2))
        signals = self.signal_head(hidden).reshape(*obs.shape[:-2],self.stocks,3)
        signal_mask = torch.stack((mask[...,0],mask[...,BUY_ACTIONS].any(-1),mask[...,2]),-1)
        signal_logp = torch.log_softmax(signals.masked_fill(~signal_mask,-1e9),-1)
        risk_logp = torch.log_softmax(self.risk_head(hidden).reshape(*obs.shape[:-2],self.stocks,9),-1)
        joint = torch.empty(*obs.shape[:-2],self.stocks,ACTION_COUNT,device=device)
        joint[...,0],joint[...,2] = signal_logp[...,0],signal_logp[...,2]
        joint[...,BUY_ACTIONS] = signal_logp[...,1,None]+risk_logp
        return torch.distributions.Categorical(logits=joint),self.critic(hidden).squeeze(-1)


def greedy_actions(probabilities, hierarchical=False):
    if not hierarchical:
        return probabilities.argmax(-1)
    signals = torch.stack((probabilities[...,0], probabilities[...,BUY_ACTIONS].sum(-1), probabilities[...,2]),-1)
    signal = signals.argmax(-1)
    risk = probabilities[...,BUY_ACTIONS].argmax(-1)
    buy_ids = torch.tensor(BUY_ACTIONS,device=probabilities.device)[risk]
    return torch.where(signal==1,buy_ids,signal)

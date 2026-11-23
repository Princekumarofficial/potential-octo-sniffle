"""Causal rolling-window classic policy; shared weights, three actions per stock."""
from collections import deque
import numpy as np
import torch
from torch import nn


class LSTMPolicy(nn.Module):
    def __init__(self, inputs, hidden=64):
        super().__init__()
        self.lstm = nn.LSTM(inputs, hidden, batch_first=True)
        self.actor = nn.Linear(hidden, 3)
        self.critic = nn.Linear(hidden, 1)

    def forward(self, obs, mask):
        device = next(self.parameters()).device
        x = torch.as_tensor(obs,dtype=torch.float32,device=device)
        mask = torch.as_tensor(mask,dtype=torch.bool,device=device)
        # [stocks, lookback, features] or [batch, stocks, lookback, features].
        if x.ndim not in (3,4): raise ValueError("LSTM expects stock sequences")
        shape = x.shape[:-2]
        _, (hidden, _) = self.lstm(x.reshape(-1,x.shape[-2],x.shape[-1]))
        encoded = hidden[-1].reshape(*shape,-1)
        logits = self.actor(encoded).masked_fill(~mask,-1e9)
        return torch.distributions.Categorical(logits=logits), self.critic(encoded).squeeze(-1).mean(-1)


class SequenceEnv:
    """Keep actual past account observations, never recompute them from today's state."""
    def __init__(self, env, lookback=30):
        if lookback < 2: raise ValueError("lookback must be at least two")
        self.env, self.lookback = env, lookback
        self.history = deque(maxlen=lookback)
        self.last_time = None

    def __getattr__(self, name): return getattr(self.env,name)

    def append(self, obs):
        stamp = self.env.times[self.env.i]
        new_day = self.last_time is None or stamp.date()!=self.last_time.date()
        if new_day: self.history.clear()
        gap = 0. if new_day else (stamp-self.last_time).total_seconds()/60
        row = np.column_stack((obs,self.env.available[self.env.i].astype(float),np.full(self.env.n,np.log1p(gap))))
        self.history.append(row.astype(np.float32))
        self.last_time = stamp
        result = np.zeros((self.env.n,self.lookback,row.shape[-1]),dtype=np.float32)
        result[:,-len(self.history):,:] = np.stack(self.history,axis=1)
        return result

    def reset(self):
        self.history.clear();self.last_time=None
        return self.append(self.env.reset())

    def step(self, actions, priorities=None):
        obs,reward,done,info = self.env.step(actions,priorities)
        return self.append(obs),reward,done,info

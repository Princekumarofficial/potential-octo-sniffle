"""Train a shared actor-critic policy and select checkpoints on validation only."""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import pandas as pd
import torch
from torch import nn
from data import prepare
from environment import TradingEnv
from markets import MARKETS
from risk_actions import ACTION_COUNT, ACTION_RISK


class Policy(nn.Module):
    def __init__(self, inputs, actions=3):
        super().__init__()
        self.body = nn.Sequential(nn.Linear(inputs, 64), nn.Tanh(), nn.Linear(64, 64), nn.Tanh())
        self.actor, self.critic = nn.Linear(64, actions), nn.Linear(64, 1)

    def forward(self, obs, mask):
        hidden = self.body(torch.as_tensor(obs, dtype=torch.float32))
        logits = self.actor(hidden).masked_fill(~torch.as_tensor(mask), -1e9)
        return torch.distributions.Categorical(logits=logits), self.critic(hidden).mean()


def evaluate(model, env):
    obs, done = env.reset(), False
    with torch.no_grad():
        while not done:
            dist, _ = model(obs, env.mask())
            obs, _, done, _ = env.step(dist.logits.argmax(-1).numpy())
    return env.equity / env.capital - 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--market', choices=tuple(MARKETS), default='india')
    parser.add_argument('--source-timezone', help='Timezone for naive input timestamps, e.g. UTC; default exchange local')
    parser.add_argument('--data-dir', type=Path, required=True)
    parser.add_argument('--results', type=Path, required=True)
    parser.add_argument('--stocks', nargs='+', required=True)
    parser.add_argument('--train-end', required=True, help='Exclusive YYYY-MM-DD boundary')
    parser.add_argument('--valid-end', required=True, help='Exclusive YYYY-MM-DD boundary')
    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--capital', type=float, default=1_000_000.)
    parser.add_argument('--allocation', type=float, default=.04)
    parser.add_argument('--output', type=Path, default=Path('reports/rl_agent'))
    args = parser.parse_args()
    if args.epochs < 1 or pd.Timestamp(args.train_end) >= pd.Timestamp(args.valid_end):
        parser.error('Positive epochs and ordered split dates required')
    if args.output.exists() and any(args.output.iterdir()):
        parser.error('Use an empty output directory to preserve earlier experiments')
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    times, prices, features, names = prepare(args.data_dir, args.results, args.stocks, market=args.market, source_timezone=args.source_timezone)
    a, b = pd.Timestamp(args.train_end), pd.Timestamp(args.valid_end)
    selections = {'train': times < a, 'valid': (times >= a) & (times < b), 'test': times >= b}
    envs = {name: TradingEnv(times[mask], prices[mask], features[mask], args.capital, args.allocation, market=args.market, risk_actions=True)
            for name, mask in selections.items()}
    args.output.mkdir(parents=True, exist_ok=True)
    model = Policy(features.shape[-1] + 7, ACTION_COUNT)
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)
    best, history = -np.inf, []
    # Online one-step advantage actor critic. Rewards are portfolio log returns.
    for epoch in range(args.epochs):
        env = envs['train']
        obs, done = env.reset(), False
        while not done:
            dist, value = model(obs, env.mask())
            action = dist.sample()
            next_obs, reward, done, _ = env.step(action.numpy())
            with torch.no_grad():
                _, next_value = model(next_obs, env.mask())
                target = 100 * reward + (0 if done else .99 * next_value)
            advantage = target - value
            loss = -dist.log_prob(action).sum() * advantage.detach() + .5 * advantage.square() - .001 * dist.entropy().mean()
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.)
            optimizer.step()
            obs = next_obs
        score = evaluate(model, envs['valid'])
        history.append(dict(epoch=epoch+1, validation_return=score))
        print(f'Epoch {epoch+1}: validation return {score:.4%}', flush=True)
        if score > best:
            best = score
            torch.save({'state_dict': model.state_dict(), 'inputs': features.shape[-1]+7, 'actions': ACTION_COUNT, 'action_risk': ACTION_RISK,
                        'stocks': args.stocks, 'strategy_features': names, 'market': args.market, 'currency': MARKETS[args.market]['currency']}, args.output / 'policy.pt')
    checkpoint = torch.load(args.output / 'policy.pt', weights_only=True)
    model.load_state_dict(checkpoint['state_dict'])
    report = {}
    for name in ('valid', 'test'):
        env = envs[name]
        result = evaluate(model, env)
        equity = pd.Series(env.curve, index=env.times)
        equity.to_csv(args.output / f'{name}_equity.csv', header=['equity'])
        fills = pd.DataFrame(env.fills)
        if len(fills):
            fills['stock'] = fills.stock_index.map(dict(enumerate(args.stocks)))
        fills.to_csv(args.output / f'{name}_trades.csv', index=False)
        report[name] = dict(market=args.market, currency=MARKETS[args.market]["currency"], cumulative_return=result, final_equity=env.equity,
                            max_drawdown=float((equity/equity.cummax().clip(lower=args.capital)-1).min()),
                            fees=sum(f['fee'] for f in env.fills), fills=len(env.fills))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 1, figsize=(12, 7))
    for ax, name in zip(axes, ('valid', 'test')):
        ax.plot(envs[name].times, envs[name].curve)
        ax.axhline(args.capital, color='gray', linestyle='--')
        ax.set_title(name.title() + ' portfolio equity')
        ax.set_ylabel(MARKETS[args.market]['currency'])
    fig.tight_layout()
    fig.savefig(args.output / 'equity.png', dpi=150)
    plt.close(fig)
    (args.output / 'summary.json').write_text(json.dumps(report, indent=2))
    (args.output / 'config.json').write_text(json.dumps(vars(args), default=str, indent=2))
    pd.DataFrame(history).to_csv(args.output / 'training.csv', index=False)


if __name__ == '__main__':
    main()

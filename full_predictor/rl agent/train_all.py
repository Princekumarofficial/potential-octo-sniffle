"""All-stock batched actor-critic experiment with explicit availability masks."""
import argparse
import json
import time
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from train import Policy
from data import prepare
from environment import TradingEnv


def stats(env):
    equity = np.asarray(env.curve)
    return dict(cumulative_return=float(equity[-1]/env.capital-1),
                max_drawdown=float((equity/np.maximum.accumulate(equity)-1).min()),
                final_equity=float(equity[-1]), fees=sum(f['fee'] for f in env.fills), fills=len(env.fills))


def evaluate(policy, env):
    obs, done = env.reset(), False
    with torch.no_grad():
        while not done:
            dist, _ = policy(obs, env.mask())
            obs, _, done, _ = env.step(dist.logits.argmax(-1).numpy(), dist.probs[:, 1].numpy())
    return stats(env)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data-dir', type=Path, required=True)
    p.add_argument('--results', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--device', choices=['auto','cpu','cuda'], default='auto')
    p.add_argument('--epochs', type=int, default=5)
    p.add_argument('--start', default='2025-12-01')
    p.add_argument('--train-end', default='2026-06-01')
    p.add_argument('--valid-end', default='2026-09-01')
    p.add_argument('--capital', type=float, default=100000.)
    p.add_argument('--allocation', type=float, default=.04)
    p.add_argument('--max-positions', type=int, default=10)
    p.add_argument('--drawdown-penalty', type=float, default=1.)
    p.add_argument('--rollout', type=int, default=256)
    p.add_argument('--seed', type=int, default=42)
    args = p.parse_args()
    if args.epochs < 1 or args.rollout < 2 or not args.start < args.train_end < args.valid_end:
        p.error('Invalid epochs, rollout or dates')
    if args.device == 'cuda' and not torch.cuda.is_available():
        p.error('CUDA unavailable in this Python environment')
    device = 'cuda' if args.device != 'cpu' and torch.cuda.is_available() else 'cpu'
    torch.set_num_threads(1)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if args.output.exists() and any(args.output.iterdir()):
        p.error('Output must be empty')
    args.output.mkdir(parents=True, exist_ok=True)
    rows = pd.read_csv(args.results)
    stocks = sorted(set(rows.stock) & {f.stem for f in args.data_dir.glob('*.parquet')})
    print(f'Preparing {len(stocks)} stocks; training device={device}', flush=True)
    times, prices, features, names, available = prepare(args.data_dir, args.results, stocks,
        market='us', alignment='union', start=args.start)
    np.savez_compressed(args.output/'prepared.npz', times=times.asi8, prices=prices, features=features, available=available)
    config = {**vars(args), 'stocks': stocks, 'device_used': device, 'strategy_features': names}
    (args.output/'config.json').write_text(json.dumps(config, default=str, indent=2))
    pd.DataFrame({'stock': stocks, 'available_bars': available.sum(0)}).to_csv(args.output/'coverage.csv', index=False)
    masks = {'train': times < args.train_end, 'valid': (times >= args.train_end) & (times < args.valid_end), 'test': times >= args.valid_end}
    envs = {name: TradingEnv(times[m], prices[m], features[m], args.capital, args.allocation,
        market='us', availability=available[m], max_positions=args.max_positions, drawdown_penalty=args.drawdown_penalty)
        for name,m in masks.items()}
    print('Split bars: '+str({k:len(v.times) for k,v in envs.items()}), flush=True)
    inputs = features.shape[-1]+4
    learner, actor = Policy(inputs).to(device), Policy(inputs)
    actor.load_state_dict(learner.state_dict())
    optimizer = torch.optim.Adam(learner.parameters(), lr=3e-4)
    best, history = -np.inf, []
    started = time.monotonic()
    for epoch in range(args.epochs):
        env = envs['train']
        obs, done = env.reset(), False
        while not done:
            observations, masks_batch, actions, rewards, values = [], [], [], [], []
            for _ in range(args.rollout):
                mask = env.mask()
                with torch.no_grad():
                    dist, value = actor(obs, mask)
                    action = dist.sample()
                observations.append(obs)
                masks_batch.append(mask)
                actions.append(action.numpy())
                values.append(value.item())
                obs, reward, done, _ = env.step(action.numpy(), dist.probs[:, 1].numpy())
                rewards.append(100*reward)
                if done: break
            with torch.no_grad():
                bootstrap = 0. if done else actor(obs, env.mask())[1].item()
            returns = []
            for reward in reversed(rewards):
                bootstrap = reward + .99*bootstrap
                returns.append(bootstrap)
            target = torch.tensor(list(reversed(returns)), dtype=torch.float32, device=device)
            x = torch.tensor(np.asarray(observations), device=device)
            h = learner.body(x)
            logits = learner.actor(h).masked_fill(~torch.tensor(np.asarray(masks_batch), device=device), -1e9)
            distribution = torch.distributions.Categorical(logits=logits)
            value = learner.critic(h).squeeze(-1).mean(-1)
            advantage = target-value
            loss = -(distribution.log_prob(torch.tensor(np.asarray(actions), device=device)).sum(-1)*advantage.detach()).mean() + .5*advantage.square().mean() - .001*distribution.entropy().mean()
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(learner.parameters(), 1.)
            optimizer.step()
            actor.load_state_dict({k:v.detach().cpu() for k,v in learner.state_dict().items()})
            if env.i % (args.rollout*40) == 0:
                print(f'Epoch {epoch+1} step {env.i}/{len(env.times)} elapsed {time.monotonic()-started:.0f}s', flush=True)
        validation = evaluate(actor, envs['valid'])
        score = validation['cumulative_return'] + args.drawdown_penalty*validation['max_drawdown']
        history.append(dict(epoch=epoch+1, score=score, **validation))
        pd.DataFrame(history).to_csv(args.output/'training.csv', index=False)
        print(f'Epoch {epoch+1}: {validation}, objective={score:.6f}', flush=True)
        if score > best:
            best = score
            torch.save(dict(state_dict=actor.state_dict(), inputs=inputs, stocks=stocks, market='us', currency='USD'), args.output/'policy.pt')
    actor.load_state_dict(torch.load(args.output/'policy.pt', weights_only=True)['state_dict'])
    report = {'device': device, 'stocks': len(stocks), 'cash_baseline_return': 0., 'validation_selection_score': best}
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2,1,figsize=(12,7))
    for ax,name in zip(axes,['valid','test']):
        env = envs[name]
        report[name] = evaluate(actor, env)
        pd.Series(env.curve,index=env.times,name='equity').to_csv(args.output/f'{name}_equity.csv')
        trades = pd.DataFrame(env.fills, columns=['timestamp','stock_index','side','shares','price','fee'])
        trades['stock'] = trades.stock_index.map(dict(enumerate(stocks)))
        trades.to_csv(args.output/f'{name}_trades.csv',index=False)
        if len(trades):
            trades.groupby('stock').agg(fills=('side','size'),fees=('fee','sum')).to_csv(args.output/f'{name}_stock_activity.csv')
        ax.plot(env.times,env.curve)
        ax.axhline(args.capital,color='gray',linestyle='--')
        ax.set_title(name.title()+' equity'); ax.set_ylabel('USD')
    fig.tight_layout(); fig.savefig(args.output/'equity.png',dpi=150); plt.close(fig)
    (args.output/'summary.json').write_text(json.dumps(report,indent=2))
    print(json.dumps(report,indent=2),flush=True)


if __name__ == '__main__': main()

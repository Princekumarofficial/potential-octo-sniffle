"""Replay all-stock checkpoints and audit decisions without updating weights."""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from train import Policy
from lstm_policy import LSTMPolicy, SequenceEnv
from portfolio_policy import PortfolioPolicy, greedy_actions
from environment import TradingEnv
from risk_actions import ACTION_RISK, BUY_ACTIONS


class ContextView:
    """Vary one stock while retaining all other contemporaneous inputs."""
    def __init__(self, model, context, masks, stock):
        self.model, self.context, self.masks, self.stock = model, context, masks, stock
    def probs(self, obs, mask):
        x = torch.as_tensor(obs,dtype=torch.float32)
        full = torch.tensor(self.context,dtype=torch.float32).unsqueeze(0).repeat(len(x),1,1)
        full[:,self.stock,:] = x
        masks = torch.tensor(self.masks).unsqueeze(0).repeat(len(x),1,1)
        masks[:,self.stock,:] = torch.as_tensor(mask)
        return self.model(full,masks)[0].probs[:,self.stock,:]


class HistoryView:
    """Explain latest inputs while preserving preceding observations."""
    def __init__(self, model, sequence): self.model,self.sequence=model,sequence
    def probs(self, obs, mask):
        x=torch.as_tensor(obs,dtype=torch.float32)
        sequences=torch.tensor(self.sequence).unsqueeze(0).repeat(len(x),1,1)
        sequences[:,-1,:]=x
        return self.model(sequences,mask)[0].probs


def probabilities(model, obs, mask):
    if isinstance(model, (PortfolioPolicy,LSTMPolicy)):
        return model(obs, mask)[0].probs
    if isinstance(model, (ContextView,HistoryView)):
        return model.probs(obs, mask)
    x = torch.as_tensor(obs, dtype=torch.float32)
    return torch.softmax(model.actor(model.body(x)).masked_fill(~torch.as_tensor(mask), -1e9), -1)


def explain(model, observation, mask, action):
    """Local sensitivity and one-at-a-time zero-reference interventions."""
    x = torch.tensor(observation[None], dtype=torch.float32, requires_grad=True)
    prob = probabilities(model, x, mask[None])[0, action]
    gradient = torch.autograd.grad(prob, x)[0][0].detach().numpy()
    variants = np.repeat(observation[None], len(observation), axis=0)
    variants[np.arange(len(observation)), np.arange(len(observation))] = 0.
    with torch.no_grad():
        after = probabilities(model, variants, np.repeat(mask[None],len(observation),axis=0))
    return float(prob.detach()), gradient, after[:, action].numpy(), greedy_actions(after,isinstance(model,ContextView)).numpy()


def label(action, risk):
    if action == 0: return 'HOLD'
    if action == 2: return 'SELL'
    if not risk: return 'BUY'
    sl,tp = ACTION_RISK[action]
    return f'BUY SL={sl:.2%} TP={tp:.2%}'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--split', choices=['valid','test'], default='test')
    parser.add_argument('--output', type=Path)
    parser.add_argument('--max-explanations', type=int, default=1000)
    parser.add_argument('--hold-stride', type=int, default=1000)
    args = parser.parse_args()
    if args.max_explanations < 1 or args.hold_stride < 1: parser.error('Positive explanation limits required')
    out = args.output or args.run / ('analysis_'+args.split)
    if out.exists() and any(out.iterdir()): parser.error('Use an empty analysis output directory')
    config = json.loads((args.run/'config.json').read_text())
    checkpoint = torch.load(args.run/'policy.pt', map_location='cpu', weights_only=True)
    actions_count = checkpoint.get('actions',3)
    risk = actions_count == 11
    if actions_count not in (3,11): raise ValueError('Unsupported checkpoint action schema')
    if risk and checkpoint.get('action_risk') != ACTION_RISK:
        raise ValueError('Checkpoint risk mapping differs from installed environment')
    stocks = checkpoint['stocks']
    if stocks != config['stocks']: raise ValueError('Stock ordering mismatch')
    cache = args.run/'prepared.npz'
    if cache.exists():
        with np.load(cache) as archive:
            data = {key: archive[key] for key in archive.files}
    else:
        from data import prepare
        def locate(value):
            path = Path(value)
            for base in [Path.cwd(), Path(__file__).resolve().parents[1], Path(__file__).resolve().parents[2]]:
                candidate = base/path
                if candidate.exists(): return candidate
            raise FileNotFoundError(f"Cannot locate recorded input {value}")
        index, prices, features, _, available = prepare(locate(config['data_dir']), locate(config['results']), stocks,
            market=checkpoint.get('market','us'), alignment='union', start=config.get('start'))
        data = dict(times=index.as_unit("ns").asi8, prices=prices, features=features, available=available)
    times = pd.to_datetime(data['times'])
    select = (times >= config['train_end']) & (times < config['valid_end']) if args.split=='valid' else times >= config['valid_end']
    env = TradingEnv(times[select], data['prices'][select], data['features'][select], config['capital'], config['allocation'],
        market=checkpoint.get('market','us'), availability=data['available'][select], max_positions=config['max_positions'],
        drawdown_penalty=config['drawdown_penalty'], risk_actions=risk)
    names = ['open_relative','high_relative','low_relative','close_relative',*config['strategy_features'],
             'position_open','invested_fraction','cash_fraction','time_of_day']
    if risk: names += ['unrealized_return','stop_loss','take_profit']
    temporal = checkpoint.get('architecture') == 'classic_lstm_v1'
    if temporal:
        env = SequenceEnv(env,checkpoint['lookback'])
        names += ['available','log_elapsed_minutes']
    if len(names) != checkpoint['inputs']: raise ValueError('Feature schema mismatch')
    joint = checkpoint.get('architecture') == 'portfolio_joint_v1'
    model = LSTMPolicy(checkpoint['inputs']) if temporal else (PortfolioPolicy(checkpoint['inputs'],len(stocks)) if joint else Policy(checkpoint['inputs'],actions_count))
    model.load_state_dict(checkpoint['state_dict']); model.eval(); torch.set_num_threads(1)
    records, effects, fills = [], [], []
    obs, done = env.reset(), False
    counts = np.zeros(actions_count,dtype=np.int64)
    explanation_count = 0
    while not done:
        index = env.i
        stamp = env.times[index]
        mask = env.mask()
        with torch.no_grad(): probs = probabilities(model, obs, mask).numpy()
        actions = greedy_actions(torch.tensor(probs),joint).numpy()
        counts += np.bincount(actions,minlength=actions_count)
        buys = probs[:, BUY_ACTIONS if risk else [1]].sum(-1)
        count = len(env.fills)
        next_obs, _, done, _ = env.step(actions, buys)
        new_fills = env.fills[count:]
        by_stock = {}
        for fill in new_fills:
            j = fill['stock_index']
            by_stock.setdefault(j,[]).append(fill)
            fills.append(dict(**fill, stock=stocks[j], decision_time=str(stamp), policy_action=label(int(actions[j]),risk)))
        # Record every trade/BUY/SELL request; sample HOLD for inactive policies.
        chosen = set(by_stock) | set(np.flatnonzero(actions != 0))
        if index % args.hold_stride == 0: chosen.update(np.flatnonzero(env.available[index]))
        for j in sorted(chosen):
            action = int(actions[j]); row_id=len(records)
            rank = int(np.sum(buys > buys[j])+1)
            latest = obs[j,-1] if temporal else obs[j]
            record = dict(id=row_id, decision_time=str(stamp), stock=stocks[j], action=label(action,risk),
                selected_probability=float(probs[j,action]), hold_probability=float(probs[j,0]),
                buy_probability=float(buys[j]), sell_probability=float(probs[j,2]), buy_priority_rank=rank,
                entropy=float(-np.sum(probs[j]*np.log(probs[j]+1e-30))),
                mask=mask[j].tolist(), action_probabilities=probs[j].tolist(),
                fills=len(by_stock.get(j,[])), execution_reasons=[f.get('reason','unknown') for f in by_stock.get(j,[])],
                observation=dict(zip(names,map(float,latest))))
            if explanation_count < args.max_explanations:
                view = HistoryView(model,obs[j]) if temporal else (ContextView(model,obs,mask,j) if joint else model)
                base, gradient, after, changed = explain(view,latest,mask[j],action)
                for k,name in enumerate(names):
                    effects.append(dict(decision_id=row_id,stock=stocks[j],feature=name,value=float(latest[k]),
                        probability_gradient=float(gradient[k]),original_probability=base,
                        zero_reference_probability=float(after[k]),probability_drop=float(base-after[k]),
                        zero_reference_action=label(int(changed[k]),risk)))
                # Keep account state and prices fixed, remove all strategy votes together.
                variant = latest.copy(); variant[4:4+len(config['strategy_features'])]=0
                with torch.no_grad(): changed_probs=probabilities(view,variant[None],mask[j:j+1])[0].numpy()
                record['without_strategy_signals_probability'] = float(changed_probs[action])
                record['without_strategy_signals_action'] = label(int(greedy_actions(torch.tensor(changed_probs),joint)),risk)
                explanation_count += 1
            records.append(record)
        obs = next_obs
    out.mkdir(parents=True,exist_ok=True)
    pd.DataFrame(records).to_json(out/'decisions.jsonl',orient='records',lines=True)
    pd.DataFrame(fills).to_csv(out/'fills.csv',index=False)
    effect_frame = pd.DataFrame(effects)
    effect_frame.to_csv(out/'feature_effects.csv',index=False)
    importance = effect_frame.assign(abs_drop=effect_frame.probability_drop.abs()).groupby('feature').agg(
        mean_absolute_probability_change=('abs_drop','mean'),mean_signed_drop=('probability_drop','mean'),count=('abs_drop','size'))
    importance.sort_values('mean_absolute_probability_change',ascending=False).to_csv(out/'feature_summary.csv')
    original = json.loads((args.run/'summary.json').read_text())[args.split]
    matched = bool(np.isclose(env.equity,original['final_equity'],rtol=0,atol=.01) and len(env.fills)==original['fills'])
    summary = dict(split=args.split, final_equity=env.equity, fills=len(env.fills), replay_matches_original=matched,
        decisions_recorded=len(records),decisions_explained=explanation_count,
        action_counts={label(i,risk):int(v) for i,v in enumerate(counts)})
    (out/'summary.json').write_text(json.dumps(summary,indent=2))
    (out/'README.md').write_text('# Post hoc decision audit\n\nRead summary.json first: replay_matches_original must be true before treating this as an explanation of the original run. A mismatch means the current simulator or inputs differ; do not interpret it as an exact historical replay.\n\nEach decisions.jsonl record links the observation time, stock, action probabilities, action mask, entry priority and resulting fills. The observation precedes execution. Risk and session exits are environmental rules, not necessarily agent SELL decisions. A BUY request can fail due to funding, capacity, availability or session constraints; zero fills alone does not identify which constraint blocked it.\n\nfeature_effects.csv measures changes in the chosen action probability when one feature is set to zero, keeping the action mask fixed. Positive probability_drop means the observed value supported that action relative to zero. Gradients measure local sensitivity and depend on feature units. Removing all strategy signals is a separate intervention in the decision records. These are model-behavior diagnostics, not causal explanations of market returns, SHAP values or proofs of why a profitable trade occurred. Setting account or price inputs to zero may produce unrealistic states.\n\nFeature summary covers the first capped explanations in chronological order, including sampled HOLD decisions. It is not an unbiased global importance estimate. The complete replay continues after that cap. Action counts include all stock-timestamps, including unavailable stocks forced to HOLD. Probability expresses model preference, not calibrated likelihood of profit. Entry priority uses total BUY probability, while the chosen action is the largest individual category.\n\nOld three-action checkpoints are supported. New checkpoints must match the installed risk-action mapping. Analysis never updates model weights or searches parameters. Do not use test explanations to tune a model and then call that same period untouched test data.\n')
    print(json.dumps(summary,indent=2),flush=True)


if __name__ == '__main__': main()

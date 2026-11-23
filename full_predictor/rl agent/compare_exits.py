"""Frozen-policy counterfactual: suppress requested SELL, retain protective exits."""
import argparse
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import pandas as pd
import torch
from portfolio_policy import PortfolioPolicy, greedy_actions
from environment import TradingEnv
from risk_actions import BUY_ACTIONS, ACTION_RISK
from train_all import stats


def replay(model, env, suppress):
    obs, done = env.reset(), False
    suppressed = 0
    with torch.no_grad():
        while not done:
            dist, _ = model(obs, env.mask())
            actions = greedy_actions(dist.probs, True).numpy()
            if suppress:
                suppressed += int((actions == 2).sum())
                actions[actions == 2] = 0
            obs, _, done, _ = env.step(actions, dist.probs[:, BUY_ACTIONS].sum(-1).numpy())
    result = stats(env)
    result['suppressed_sell_requests'] = suppressed
    result['gross_pnl_before_deducted_fees'] = result['final_equity']-env.capital+result['fees']
    result['net_pnl'] = result['final_equity']-env.capital
    result['exit_reasons'] = pd.Series([f['reason'] for f in env.fills if f['side']=='SELL'],dtype=str).value_counts().to_dict()
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run',type=Path,required=True)
    p.add_argument('--output',type=Path)
    args=p.parse_args()
    out=args.output or args.run/'exit_comparison'
    if out.exists() and any(out.iterdir()): p.error('Use an empty output directory')
    config=json.loads((args.run/'config.json').read_text())
    checkpoint=torch.load(args.run/'policy.pt',map_location='cpu',weights_only=True)
    if checkpoint.get('architecture')!='portfolio_joint_v1' or checkpoint.get('action_risk')!=ACTION_RISK:
        raise ValueError('Requires compatible joint TP/SL checkpoint')
    model=PortfolioPolicy(checkpoint['inputs'],len(checkpoint['stocks']))
    model.load_state_dict(checkpoint['state_dict']); model.eval(); torch.set_num_threads(1)
    data=np.load(args.run/'prepared.npz'); times=pd.to_datetime(data['times'])
    original=json.loads((args.run/'summary.json').read_text())
    out.mkdir(parents=True,exist_ok=True)
    report={}
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(2,1,figsize=(12,8))
    for ax,split in zip(axes,['valid','test']):
        selected=(times>=config['train_end']) & (times<config['valid_end']) if split=='valid' else times>=config['valid_end']
        report[split]={}
        for mode,suppress in [('original',False),('no_agent_sell',True)]:
            env=TradingEnv(times[selected],data['prices'][selected],data['features'][selected],config['capital'],config['allocation'],
                market='us',availability=data['available'][selected],max_positions=config['max_positions'],drawdown_penalty=config['drawdown_penalty'],risk_actions=True)
            result=replay(model,env,suppress)
            if not suppress:
                result['matches_saved_run']=bool(abs(result['final_equity']-original[split]['final_equity'])<.01 and result['fills']==original[split]['fills'])
                if not result['matches_saved_run']: raise ValueError('Baseline replay mismatch; comparison aborted')
            report[split][mode]=result
            trades=pd.DataFrame(env.fills)
            if len(trades): trades['stock']=trades.stock_index.map(dict(enumerate(checkpoint['stocks'])))
            trades.to_csv(out/f'{split}_{mode}_trades.csv',index=False)
            pd.Series(env.curve,index=env.times,name='equity').to_csv(out/f'{split}_{mode}_equity.csv')
            ax.plot(env.times,env.curve,label=mode)
            print(split,mode,json.dumps(result),flush=True)
        ax.set_title(split); ax.set_ylabel('USD'); ax.legend()
    fig.tight_layout();fig.savefig(out/'comparison.png',dpi=150);plt.close(fig)
    (out/'summary.json').write_text(json.dumps(report,indent=2))
    (out/'README.md').write_text('Frozen-checkpoint diagnostic, not retraining. Agent SELL requests become HOLD after action selection. TP/SL and session liquidation remain active. Later observations, cash, entry choices and priorities are recomputed on the counterfactual account. Fees remain enabled. Baseline final equity and fill count must match the saved run. Test outcomes have been inspected and are development evidence, not fresh out-of-sample confirmation. Gross P&L adds back deducted fees for realized trades; it is not a separate fee-free simulation.\n')


if __name__=='__main__': main()

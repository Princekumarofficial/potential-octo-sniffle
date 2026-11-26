"""Audit the original policy versus signal ablations and random daily selectors."""
import sys,json
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import pandas as pd
import torch
from train import Policy
from environment import TradingEnv
from train_all import stats

root=Path(__file__).resolve().parents[1]
run=root/'reports/rl_us_all_drawdown'
cache_run=root/'reports/rl_us_all_run2'
out=run/'behavior_audit'
out.mkdir(exist_ok=True)
config=json.loads((run/'config.json').read_text())
cache_config=json.loads((cache_run/'config.json').read_text())
assert config['stocks']==cache_config['stocks'] and config['start']==cache_config['start']
with np.load(cache_run/'prepared.npz') as f: data={k:f[k] for k in f.files}
times=pd.to_datetime(data['times']); selected=times>=config['valid_end']
checkpoint=torch.load(run/'policy.pt',weights_only=True,map_location='cpu')
model=Policy(checkpoint['inputs']);model.load_state_dict(checkpoint['state_dict']);model.eval();torch.set_num_threads(1)
report={}; original_buys=None
for mode in ['original','zero_signals','zero_prices',*[f'random_{i}' for i in range(5)]]:
 env=TradingEnv(times[selected],data['prices'][selected],data['features'][selected],config['capital'],config['allocation'],market='us',availability=data['available'][selected],max_positions=config['max_positions'],drawdown_penalty=config['drawdown_penalty'])
 obs,done=env.reset(),False
 rng=np.random.default_rng(int(mode.split('_')[1]) if mode.startswith('random_') else 0)
 day=None; priority=None; snapshots=[]
 while not done:
  modified=obs.copy()
  if mode=='zero_signals': modified[:,4:9]=0
  if mode=='zero_prices': modified[:,:4]=0
  if mode.startswith('random_'):
   if day!=env.times[env.i].date():
    day=env.times[env.i].date();priority=rng.random(env.n)
   actions=np.where(env.quantity==0,1,0)
  else:
   with torch.no_grad():
    dist,_=model(modified,env.mask())
    actions=dist.probs.argmax(-1).numpy();priority=dist.probs[:,1].numpy()
   if mode=='original' and env.times[env.i].hour==9 and env.times[env.i].minute==30:
    for j in np.flatnonzero(env.available[env.i]):
     snapshots.append(dict(time=str(env.times[env.i]),stock=config['stocks'][j],buy_probability=float(priority[j]),hold_probability=float(dist.probs[j,0]),**{name:float(obs[j,k]) for k,name in enumerate(['open','high','low','close',*config['strategy_features']])}))
  obs,_,done,_=env.step(actions,priority)
 result=stats(env)
 buys={(f['timestamp'],f['stock_index']) for f in env.fills if f['side']=='BUY'}
 if mode=='original':
  saved=json.loads((run/'summary.json').read_text())['test']
  result['matches_original']=abs(result['final_equity']-saved['final_equity'])<.01 and result['fills']==saved['fills']
  print('Saved baseline:',saved,'current replay:',result,flush=True)
  if not result['matches_original']: print('WARNING: current replay differs; ablations describe current replay only',flush=True)
  original_buys=buys
  pd.DataFrame(snapshots).to_csv(out/'opening_decisions.csv',index=False)
 result['same_entries_as_original']=len(buys & original_buys)
 result['total_entries']=len(buys)
 report[mode]=result
 pd.Series(env.curve,index=env.times,name='equity').to_csv(out/f'{mode}_equity.csv')
 print(mode,json.dumps(result),flush=True)
(out/'summary.json').write_text(json.dumps(report,indent=2))

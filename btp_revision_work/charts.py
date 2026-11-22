from pathlib import Path
import numpy as np,pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import StrMethodFormatter
w=Path('btp_revision_work')
plt.rcParams.update({'font.size':11,'axes.spines.top':False,'axes.spines.right':False,'axes.labelcolor':'#263f4c','axes.edgecolor':'#9aaeb8','axes.titlelocation':'left','font.family':'DejaVu Sans'})
def finish(fig,name):fig.tight_layout();fig.savefig(w/(name+'.png'),dpi=175,bbox_inches='tight',facecolor='white');plt.close(fig)
green='#146c65';navy='#254868';gold='#b56b21'
r=pd.read_csv('reports/real_trading_2026/simulation_2026_trades.csv',parse_dates=['date'])
start=pd.Timestamp('2026-07-24'); b=np.r_[10000,r.benchmark]; exacte=10000*np.exp(np.r_[0,np.cumsum(b[1:]/b[:-1]-1)])
fig,ax=plt.subplots(figsize=(8.8,3.15));dates=[start,*r.date];ax.plot(dates,np.r_[10000,r.capital],color=green,lw=2.1,label='LSTM trading strategy');ax.plot(dates,exacte,color=gold,lw=2,label='Passive long EUR (exact compounding)');ax.axhline(10000,color='gray',ls=':',lw=1);ax.set_ylabel('Account value (USD)');ax.legend(frameon=False,ncol=2,fontsize=9);ax.grid(axis='y',alpha=.18);fig.autofmt_xdate(rotation=0);finish(fig,'eur-benchmark')
g=pd.read_csv('full_predictor/eda/eda_outputs/missing_run_summary.csv').nlargest(7,'missing_minutes').sort_values('missing_minutes')
fig,ax=plt.subplots(figsize=(8.8,3.1));ax.barh(g.stock,g.missing_minutes,color='#adc5ca',label='All minutes in gaps');ax.barh(g.stock,g.isolated_minutes,color=green,label='Minutes in one-bar gaps');ax.legend(frameon=False,fontsize=9);ax.set_xlabel('Missing minutes in audited runs');finish(fig,'gap-audit')
m=pd.read_csv('reports/indicator_analysis/model_summary.csv')
fig,ax=plt.subplots(figsize=(8.8,3));x=np.arange(3)
for i,(label,key,color) in enumerate([('All 61','all_indicators',navy),('Compact 32','compact',green),('Recent returns','recent_returns',gold)]):
 a=m[m.model.eq(key)].sort_values('horizon_bars');ax.bar(x+(i-1)*.23,100*a.mean_skill,.21,label=label,color=color)
ax.set_xticks(x,['5 minutes','15 minutes','60 minutes']);ax.axhline(0,color='black',lw=.8);ax.set_ylabel('MSE skill vs persistence (%)');ax.legend(frameon=False,ncol=3,fontsize=9);finish(fig,'skill')
fig,ax=plt.subplots(figsize=(8.8,2.6));x=np.arange(3);ax.bar(x-.18,[55.51,50.95,53.30],.36,color=green,label='LSTM');ax.bar(x+.18,[55.07,50.07,50.07],.36,color='#adbac1',label='Always up');ax.set_ylim(0,66);ax.set_xticks(x,['USD/INR','EUR/USD','EUR/USD fine-tuned']);ax.set_ylabel('Correct direction (%)');ax.legend(ncol=2,frameon=False,loc='upper center',fontsize=9);finish(fig,'accuracy')

fig,axs=plt.subplots(1,2,figsize=(9,2.7))
for ax,stock in zip(axs,['LTIM','RELIANCE']):
 for rule,col,lab in [('lstm_threshold',navy,'Threshold'),('rl_policy',green,'Q-learning')]:
  a=pd.read_csv('full_predictor/reports/'+stock+'_'+rule+'_trades.csv',parse_dates=['exit_time'])
  ax.plot(a.exit_time,a.net_pnl_per_10000.cumsum(),label=lab,color=col,lw=1.8)
 ax.set_title(stock);ax.axhline(0,c='gray',lw=.6);ax.tick_params(axis='x',labelsize=8);ax.locator_params(axis='y',nbins=4);ax.grid(axis='y',alpha=.18)
 import matplotlib.dates as md
 ax.xaxis.set_major_locator(md.AutoDateLocator(minticks=2,maxticks=4));ax.xaxis.set_major_formatter(md.DateFormatter('%b %y'))
axs[0].set_ylabel('Cumulative fixed-size P&L');axs[1].legend(frameon=False,fontsize=8);finish(fig,'paired-trades')

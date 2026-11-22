# RL agent over GA and GP strategy signals

This isolated research module learns a buy, sell-to-close or hold action for each stock. It imports existing data loaders, strategy functions and FYERS fee helpers without modifying GA_STRATEGY. It implements its own step-based environment because the existing portfolio simulator is a batch replay function, not an interactive RL environment.

## Inputs and decisions

Supply processed equity Parquet files and fresh batch `monthly_results.csv` containing each strategy and GP. Every saved strategy must have `fit_end` before its evaluation month. Parameters remain frozen within that month. The loader rejects duplicate strategy rows and intersects stock timestamps where all strategy signals are available. Check the resulting date range: the intersection can substantially reduce coverage. Historical strategy searches and feature engineering must themselves be causal; timestamp checks alone cannot establish that.

Each stock observation contains Open, High, Low and Close relative to the previous close, four configured strategy signals plus GP, position status, invested fraction, cash fraction and time of day. No full-history normalization is fitted. A shared neural network produces a categorical action per stock. It has no stock-ID embedding or cross-stock attention; shared cash and a portfolio reward couple decisions.

- HOLD preserves the position, or remains in cash.
- BUY opens one whole-share long position if flat.
- SELL closes a position; it does not short.

Actions observed at close t execute at the next common bar's open. Sells execute first; buys use the specified stock order when cash is insufficient. There is no repeated buying while already invested. Default initial capital is INR 1,000,000 and each entry gets at most 4% of current account equity. Entry fees must fit available cash. There is no leverage.

For the default `--market india`, session rules prohibit entries at/after 15:00 and force closing at/after 15:15 or the final observed daily bar. Cross-day actions expire. The final episode bar liquidates holdings. India uses shared FYERS intraday charges; US uses the existing US fee helper, with no income tax. It assumes successful fills: circuit limits, trapped exits, delivery conversion, slippage, market impact and broker margin behavior are not implemented. The last-observed-bar liquidation is an idealization using the known dataset boundary. This is not execution-equivalent to the full portfolio simulator. Choose the market explicitly and supply the correct source timezone.

## Learning and leakage control

The learner is a small PyTorch advantage actor-critic with two 64-unit hidden layers and separate actor/value heads. It learns online from one-step advantages, discount 0.99, entropy regularization and gradient clipping. Reward is the log change in shared portfolio equity after charges, scaled by 100 for learning. Invalid position actions are masked. This is a baseline implementation, not PPO or a claim of profitable learning.

Training, validation and test are chronological, each starting a fresh account. Only training bars update weights. Validation return selects the checkpoint. The held-out test is evaluated once after selection. Do not repeatedly adjust the model based on that test. Zero return from staying in cash is a meaningful baseline; compare against GA Selected, random policies and buy-and-hold under consistent execution before drawing conclusions. Repeated seeds and walk-forward RL retraining are future experiments; this version uses fixed chronological splits.

## Run

From `full_predictor`, using the project Python environment:

```powershell
python "rl agent/train.py" --data-dir datasets/processed_datasets/NIFTY50 --results reports/risk_blend/monthly_results.csv --stocks ABB ADANIENT ADANIPORTS --train-end 2023-01-01 --valid-end 2024-01-01 --epochs 5 --output reports/rl_trial
python "rl agent/test_environment.py"
```

Choose dates covered by your report. The output directory must be empty. Required packages are numpy, pandas, pyarrow, torch and matplotlib; the module does not require Gym or Stable Baselines. CPU is used by default. Long minute-level datasets take substantial time with per-step neural-network updates; begin with a short dataset and a few stocks.

Outputs: `policy.pt`, run configuration, epoch validation history, held-out summary, validation/test equity and trade CSVs, and `equity.png`. The checkpoint stores network weights and feature/stock metadata. `Policy` and `evaluate` in train.py can reload it with the same observation layout. No live trading or broker connection is included.

All files for this addition live here. Existing source files and previous results remain untouched.


## US stocks

Use `--market us`. This explicitly selects USD reporting, America/New_York
clock conversion, regular-session bars from 09:30 through 16:00, entry cutoff
15:45 and square-off at 15:55 (or the last observed daily bar if earlier).
Dates used for splits are New York local dates. DST is handled during timezone
conversion. The environment uses the existing `us_intraday_charges` helper:
its zero-commission scenario with encoded SEC and FINRA sell charges. These
are the repository's rates, not a freshly verified tariff or a historical
rate schedule. No FYERS charges are applied in US mode.

From `full_predictor`, replace the example data/report paths and dates with
your actual US batch outputs:

```powershell
python "rl agent/train.py" --market us --data-dir datasets/processed_datasets/us_stocks --results reports/batch_us_stocks_5x/monthly_results.csv --stocks AAPL MSFT NVDA --train-end 2026-06-01 --valid-end 2026-09-01 --capital 100000 --allocation 0.04 --epochs 5 --output reports/rl_us_trial
```

Naive timestamps default to exchange-local time. If your naive timestamps
represent UTC, add `--source-timezone UTC`. This setting also applies to
naive `fit_end` timestamps in the batch report. Timezone-aware timestamps
are converted directly, preserving their recorded instants. Data and report
must use compatible time conventions. The saved monthly labels must refer
to exchange-local evaluation months. Extended-hours bars are filtered out.

Parquet files still need precomputed features and quality columns expected
by `GA_STRATEGY.main.load_data`; raw OHLC CSVs alone are insufficient. The
report must contain saved, prior-fitted parameters for every configured
strategy and GP. Training metadata and checkpoint include the explicit
market; charts and summaries use USD. Capital is in USD, without FX conversion.

No exchange holiday or early-close calendar is installed. Early closes use
the dataset's final observed bar, so missing data can resemble an early
close. Signal generation reuses existing GA/GP functions, including their
session-time inference; the RL fill/fee engine itself uses the explicit
market. Truncated sessions can still affect upstream risk signals. Review
those signals before interpreting results. US leverage, short selling,
settlement restrictions and broker-specific account rules are not modeled.


## All-stock selective experiment

`train_all.py` discovers every stock present in both the report and Parquet
folder. It aligns the union of timestamps and blocks trades on missing bars.
Missing inputs are zero and masked actions prevent unavailable entries;
forward prices are valuation marks only, never synthetic fills. Each stock
liquidates on its final observed session bar under the existing idealized
end-of-data assumption. Coverage is saved in `coverage.csv`.

The policy can consider the entire universe but holds at most 10 stocks by
default, with 4% entry allocation per stock. When multiple BUY requests
compete for slots, higher current policy BUY probabilities get priority.
Existing positions remain until SELL or session liquidation; this is not a
monthly trailing-performance top-ten filter. No minimum number of positions
is required. Price appreciation can move invested fractions above entry caps.

The new objective rewards log equity growth minus a penalty for increases
in drawdown from the running peak. Default penalty is 1. Validation chooses
the checkpoint maximizing cumulative return minus absolute maximum drawdown
with the same coefficient. This encourages, but does not enforce, low
drawdown. Staying entirely in cash may beat an unprofitable learned policy;
the report includes that baseline.

The batched actor-critic collects 256 steps per rollout and uses bootstrapped
multi-step returns for a single gradient update. All stock actions contribute
to the joint log probability. Inference stays on CPU; `--device auto` uses
CUDA for batched learning when available, otherwise CPU. `--device cuda`
fails explicitly when the installed PyTorch lacks CUDA support. CPU-only
PyTorch cannot use an installed NVIDIA GPU. The previous one-step trainer
remains available. No packages are installed automatically.

```powershell
python "rl agent/train_all.py" --data-dir datasets/processed_datasets/us_stocks --results reports/batch_us_stocks_5x/monthly_results.csv --output reports/rl_us_all_drawdown --epochs 5 --device auto
```

Defaults: USD 100,000, 4% allocation, 10 positions, data from December 2025,
training before June 2026, validation June-August, test September onward.
All boundaries and risk settings are CLI options. Each split starts fresh.
This script is US-only. It stores the aligned arrays, coverage, configuration,
best model, epoch metrics, held-out equity/trades/stock activity and a plot.
Stock activity is not a ranking of profitability. Report provenance and
historical survivorship remain properties of the supplied universe.

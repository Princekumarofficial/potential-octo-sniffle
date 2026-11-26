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


## Agent-selected TP and SL

Both trainers now learn an 11-action categorical policy: HOLD, SELL, and nine
BUY variants. Each BUY variant selects SL from 0.5%, 0.75%, 1% and reward/risk
from 1, 2, 3; TP equals SL times that ratio. Edit `risk_actions.py` to change
the values while preserving the action mapping. These are RL decisions per
entry, distinct from the saved GA/GP risk settings used to generate inputs.

Brackets are fixed from the actual entry fill. Observations now also contain
unrealized entry-relative return and active SL/TP fractions (16 total inputs
with the default signals). HOLD keeps the bracket. SELL remains available
and exits at the next available eligible open, without waiting for TP/SL.
The agent cannot widen or move a bracket on an open trade in this version.

Existing brackets are checked at observed opens, then at observed closes.
A gap crossing exits at the open, and a close crossing exits at the close,
with ordinary sell charges. There is no high/low touch simulation and no
claim of an exact stop-price fill. Missing bars defer risk checks. A bracket
exit at the open takes priority over an agent SELL. Session liquidation is
still enforced. A BUY request does not reopen an open-triggered stopped
position in the same bar.

Trade CSVs preserve entry price, SL, TP and the exit reason: `agent`,
`stop_loss`, `take_profit`, or `session_end`. All-stock entry prioritization
uses the sum of probabilities of all BUY variants. Position masks prohibit
BUY variants while invested. Both actor losses train the selected joint
BUY-and-bracket action, so risk choices receive the portfolio reward.

Retraining is required: previous 3-action/13-input checkpoints cannot be
loaded into this new 11-action/16-input model. Saved checkpoints include the
action count and mapping. Use the same training command with a new output
folder. No new full training run was performed as part of this change.


## Post hoc explanations

Run from `full_predictor` after an all-stock training run:

```powershell
python "rl agent/analyze.py" --run reports/rl_us_all_run2 --split test --max-explanations 1000
```

This is a read-only model replay, not further training. It supports old
three-action and new TP/SL checkpoints. `prepared.npz` is preferred; if
missing, inputs are reconstructed from the recorded configuration. The tool
checks replay equity and fill count against the original summary. A mismatch
is flagged and explanations must not be treated as an exact original replay.

Outputs go into `analysis_test` (or `analysis_valid`):
- `decisions.jsonl`: observation values, action probabilities, masks, BUY
  priority, strategy-signal removal response, and execution reasons.
- `fills.csv`: fill timestamp, preceding policy decision and actual exit reason.
- `feature_effects.csv`: local probability gradients and single-feature
  zero-reference perturbations.
- `feature_summary.csv`: average absolute probability changes for explained
  decisions; the sample is chronological and capped, not globally representative.
- `summary.json` and an interpretation README.

Every trade and non-HOLD request is recorded; HOLD observations are sampled
at `--hold-stride` intervals (default 1000). The explanation cap limits costly
perturbations but does not truncate replay. Use `--output` for another analysis
folder; existing nonempty folders are protected. These diagnostics explain
model sensitivity, not economic causation or calibrated profit probabilities.
Zero-reference account states can be unrealistic. A forced stop or session
exit must not be interpreted as an agent-chosen SELL. Analysis currently targets
`train_all.py` artifacts; the single-stock trainer does not save the same schema.


## Current global portfolio architecture

`train_all.py` now uses `PortfolioPolicy` (`portfolio_policy.py`), architecture
ID `portfolio_joint_v1`. This supersedes the shared independent-stock network
for new all-stock runs. One full [N,16] observation is flattened into one
N*16 input. Two 256-unit Tanh layers produce N*3 trade logits, N*9 conditional
risk logits, and one portfolio value. Each stock's outputs can depend on
every stock's inputs. Stock order and universe size are fixed by the saved
checkpoint. The original `train.py` remains a legacy trainer.

Trade choices are HOLD, BUY, SELL. Only BUY uses the conditional nine-way
SL/TP choice. Training uses a factored joint categorical distribution:
P(BUY, bracket)=P(BUY)*P(bracket|BUY). Deterministic evaluation first selects
the trade using total BUY probability, then the bracket. Thus nine BUY
variants no longer split BUY's vote against one HOLD category. This fixes a
structural evaluation bias; it does not guarantee profitable trading or
eliminate legitimate cash-only behavior.

Training logs now include training fills and return separately from validation.
New checkpoints require retraining; existing independent-stock checkpoints
remain supported by the analyzer. Post-hoc per-stock perturbations retain
all other stock observations, so the global context is preserved. The local
feature summary still explains the selected stock's features only, not all
cross-stock interactions. Reward, allocation, session and drawdown constraints
are unchanged. Earlier performance figures do not apply to this architecture.

Use the same all-stock command with a fresh output folder, for example
`--output reports/rl_us_joint_tp_sl_run1`. No full retraining was performed
as part of the architecture change.


## Select either model from the command line

Both models are retained in `train_all.py`; no source edits are needed:

| Flag | Network | Observation/action schema | RL brackets |
| --- | --- | --- | --- |
| `--model classic` | Shared per-stock 64/64 network; average stock critic | 13 inputs per stock, HOLD/BUY/SELL | Disabled |
| `--model joint` | Global flattened stock input; 256/256 network and portfolio critic | 16 inputs per stock, trade + conditional bracket | Enabled |

`joint` remains the default to preserve existing commands. Both operate on
all available stocks and use the same portfolio accounting, rollout schedule,
fee model, allocation, position cap and drawdown objective. Classic is the
architecture used by the earlier approximately 4% experiment; retraining it
does not guarantee that return. Upstream GA/GP signals still retain their
saved risk settings even when classic's own RL brackets are disabled.

From `full_predictor`:

```powershell
python "rl agent/train_all.py" --model classic --data-dir datasets/processed_datasets/us_stocks --results reports/batch_us_stocks_5x/monthly_results.csv --output reports/rl_us_classic_run1 --epochs 5 --device cuda
python "rl agent/train_all.py" --model joint --data-dir datasets/processed_datasets/us_stocks --results reports/batch_us_stocks_5x/monthly_results.csv --output reports/rl_us_joint_run2 --epochs 5 --device cuda
```

These commands inherit USD 100,000 capital, 4% allocation, 10 positions,
drawdown penalty 1, December 2025 start, June 2026 train end, and September
2026 validation end. Other existing flags override these defaults.

```powershell
python "rl agent/analyze.py" --run reports/rl_us_classic_run1 --split test
python "rl agent/analyze.py" --run reports/rl_us_joint_run2 --split test
```

Configuration, checkpoint and summary record the selected model. Analysis
automatically detects the architecture and action schema from the checkpoint;
no model flag is needed. Old checkpoints without a model field remain
supported. Output directories must be new/empty to preserve previous runs.


## Classic model with temporal LSTM inputs

Add `--model classic --use-lstm`; optionally set `--lookback 30`. This flag
is rejected for joint mode. Without it, existing classic and joint modes
are unchanged. The saved architecture is `classic_lstm_v1`.

```powershell
python "rl agent/train_all.py" --model classic --use-lstm --lookback 30 --data-dir datasets/processed_datasets/us_stocks --results reports/batch_us_stocks_5x/monthly_results.csv --output reports/rl_us_classic_lstm_run1 --epochs 5 --device cuda --capital 100000 --allocation 0.04 --max-positions 10 --drawdown-penalty 1 --start 2025-12-01 --train-end 2026-06-01 --valid-end 2026-09-01
python "rl agent/analyze.py" --run reports/rl_us_classic_lstm_run1 --split test --max-explanations 1000
```

One shared single-layer 64-hidden-unit LSTM processes each stock's sequence,
followed by three-action actor and scalar critic heads. The stock critic
outputs are averaged into a portfolio value. There is no joint cross-stock
layer in this classic mode. Actions remain HOLD, BUY and SELL, with no
RL-generated TP/SL. Fees, allocation and the drawdown objective are unchanged.

The LSTM receives the classic 13 observation features plus availability and
log(1 + elapsed minutes since previous aligned timestamp), for 15 features
per timestep. Rollout observations have shape [batch, stocks, lookback, 15].
Past account observations are stored as they actually occurred, never
reconstructed using the current account state. Missing-stock bars retain
the existing zero-feature/valuation behavior with explicit availability.
History uses aligned timestamps, not a fixed number of actual bars per stock.

Windows end at the decision timestamp. They are left-zero-padded at session
start and reset at every new calendar session and at each split/episode.
There is no carry of recurrent hidden state beyond the explicit window;
each forward pass starts from zero state. Windows persist across rollout
boundaries within the same session. All lookback steps participate in the
training gradient. The current implementation still collects actions on CPU
and moves rollout batches to CUDA for learning. Longer windows increase CPU
inference time, memory use and training cost; GPU does not accelerate every
stage. No performance improvement is assumed without fresh evaluation.

Checkpoints save the architecture, input count and lookback; analysis loads
these automatically. LSTM post-hoc explanations vary only the latest input
values while holding the preceding sequence fixed. They do not attribute
importance to every lag or remove strategy signals throughout the full
history. Old checkpoints remain usable in their original model modes.

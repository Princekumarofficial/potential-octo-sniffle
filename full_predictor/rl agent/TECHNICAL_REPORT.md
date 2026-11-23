# Reinforcement Learning Trading Agent Technical Report

> Architecture update: new all-stock runs now use `portfolio_joint_v1` in `portfolio_policy.py`: flattened N�16 inputs, two 256-unit Tanh layers, N�3 trade outputs, N�9 conditional risk outputs, and one portfolio critic. Deterministic decisions choose HOLD/BUY/SELL first, then a BUY bracket. The shared 64-unit architecture described below is the preceding implementation; its parameter count and independent-stock critic description do not describe the current all-stock model. See the current global portfolio architecture section in README.md.

**Temporal option:** `--model classic --use-lstm --lookback 30` enables a shared per-stock 64-unit LSTM with three actions and 15 features per timestamp (classic features plus availability and log elapsed minutes). Causal windows reset per session and split. Analysis detects the saved LSTM architecture; see README.md for full semantics.

**Command selection:** `train_all.py --model classic` selects the earlier shared three-action model; `--model joint` selects the global TP/SL model (default). Checkpoints and summaries record the choice, and analysis detects it automatically.

**Implementation snapshot: 22 November 2026**  
**Scope: `train_all.py` with agent-selected TP and SL**

## 1. Executive overview

The system is a two-stage historical trading framework. GA parameter search and genetic programming first generate trading signals using previously fitted parameters. A shared multi-stock actor–critic model then uses those signals, price features and account information to request HOLD, SELL, or BUY with a selected take-profit and stop-loss bracket.

The agent learns from portfolio returns after modeled transaction charges, with a penalty for worsening drawdown. It can consider the complete stock universe while limiting simultaneous positions. Its outputs are evaluated chronologically and can be inspected through a separate post-hoc replay tool.

This is a research simulator, not a live broker integration. The current model is not DQN, PPO, a transformer, or a recurrent network. Older three-action experiment results must not be attributed to the current eleven-action TP/SL model.

## 2. System workflow

```text
Historical OHLCV
    -> Feature engineering and processed Parquet files
    -> GA parameter search and GP rule evolution
    -> Saved monthly strategy parameters
    -> Replay of strategy signals
    -> OHLC features + strategy signals + account observations
    -> Shared actor-critic network
    -> HOLD / SELL / BUY with TP and SL
    -> Shared-capital execution environment
    -> Fills, charges, equity and drawdown
    -> Portfolio reward and network updates
    -> Validation checkpoint selection
    -> Held-out testing and post-hoc analysis
```

The RL agent does not change the underlying GA/GP rules. It learns another decision layer above their signals. Upstream strategies can be retrained monthly while the trained RL policy remains fixed during its validation and test periods.

## 3. Implementation components

All RL additions are isolated in this folder and reuse functions from GA_STRATEGY.

| File | Responsibility |
| --- | --- |
| [data.py](data.py) | Load prices and regenerate saved strategy signals |
| [markets.py](markets.py) | Timezone conversion and session cutoffs |
| [environment.py](environment.py) | Positions, cash, fills, TP/SL, fees and reward |
| [risk_actions.py](risk_actions.py) | TP/SL combinations and action mapping |
| [train.py](train.py) | Shared network definition and original single-step trainer |
| [train_all.py](train_all.py) | All-stock batched training |
| [analyze.py](analyze.py) | Post-hoc replay and feature-sensitivity analysis |

Existing GA_STRATEGY modules provide the data loader, strategy functions and transaction-charge helpers. The all-stock trainer is the subject of the equations and defaults below; the original single-step trainer uses a different update schedule.

## 4. Data inputs and preparation

### 4.1 Processed market data

Each stock has a Parquet file with a timestamp index, OHLCV, precomputed technical indicators, and quality/segment fields expected by the existing loader. Volume may exist in the dataset, but is not directly included in the RL observation.

The inherited loader sorts timestamps, validates the index and retains observed valid bars. It masks relevant indicators after imputed bars contaminate a segment. These safeguards do not independently establish the correctness of corporate-action treatment, historical universe membership or all feature calculations.

### 4.2 Monthly strategy reports

The batch `monthly_results.csv` supplies stock, month, strategy, saved parameters and `fit_end`. GP rows include a saved rule tree. The loader requires:

$$
\text{fit\_end} < \text{evaluation month start}
$$

Parameters are replayed for their evaluation month. Missing parameters represent no valid fitted candidate and produce a cash signal; malformed JSON raises an error. The loader clears legacy feature/timing caches around strategy calls to avoid cross-frame cache collisions.

### 4.3 Universe and alignment

Stocks are discovered from the intersection of report symbols and available Parquet filenames. The supplied US experiment contains 95 stocks, but this count is data-dependent.

The all-stock trainer aligns the union of timestamps. Missing observations are zero-filled for neural inputs and accompanied by an availability mask. Forward-carried prices are valuation marks only, not synthetic execution prices. The environment blocks fills on unavailable bars. The resulting coverage is exported per stock.

### 4.4 Market and timezone

The all-stock entry point currently selects US mode explicitly. Exchange-local time is America/New_York. Timezone-aware data is converted using timezone rules, including daylight saving time; naive input is assumed exchange-local in this entry point. Unlike the original trainer, `train_all.py` does not expose a source-timezone option. Naive UTC data must therefore be converted appropriately before using this entry point.

The configured US regular-session filter is 09:30 through 16:00. No exchange holiday or early-close calendar is installed; final observed daily bars provide the early-end fallback.

## 5. Observation space

Each stock receives 16 features.

| Group | Features | Count |
| --- | --- | ---: |
| Prices | Open, High, Low, Close relative to previous close | 4 |
| Strategy signals | VWAP, RSI, Bollinger, Donchian, GP | 5 |
| Account | Position open, invested fraction, cash fraction, time of day | 4 |
| Active trade | Unrealized return, SL fraction, TP fraction | 3 |
| Total | | 16 |

For price component P:

$$
x^{price}_{t,i}=\frac{P_{t,i}}{C_{t-1,i}}-1
$$

Other features include:

$$
\text{InvestedFraction}_{t,i}=\frac{q_{t,i}C_{t,i}}{E_t},\qquad
\text{CashFraction}_t=\frac{\text{Cash}_t}{E_t}
$$

$$
\text{UnrealizedReturn}_{t,i}=\frac{C_{t,i}}{P_{entry,i}}-1
$$

Time of day is minutes since midnight divided by 1,440. When flat, unrealized return and active SL/TP are zero. Inputs do not use a scaler fitted on the complete dataset.

For N stocks, the observation shape is `[N, 16]`; for 95 stocks it is `[95, 16]`. There is no explicit historical sequence, recurrent state, stock-ID embedding or cross-stock attention. Technical signals carry some historical information, but the neural model is memoryless.

## 6. Neural architecture

```text
16 per-stock inputs
    -> Linear(16, 64) -> Tanh
    -> Linear(64, 64) -> Tanh
        -> Actor: Linear(64, 11) -> masked categorical distribution
        -> Critic: Linear(64, 1) -> mean over stocks
```

The current default network has 6,028 trainable parameters. All stocks share the same weights. The critic averages per-stock outputs:

$$
V_\phi(S_t)=\frac{1}{N}\sum_{i=1}^{N}v_\phi(o_{t,i})
$$

This is a simplified portfolio value estimate, not an explicit model of correlations or interactions among holdings. Unavailable stock rows are also included in this average.

## 7. Actions and risk brackets

The policy has eleven categorical actions:

| ID | Decision | SL | TP |
| ---: | --- | ---: | ---: |
| 0 | HOLD | — | — |
| 1 | BUY | 0.50% | 0.50% |
| 2 | SELL | — | — |
| 3 | BUY | 0.50% | 1.00% |
| 4 | BUY | 0.50% | 1.50% |
| 5 | BUY | 0.75% | 0.75% |
| 6 | BUY | 0.75% | 1.50% |
| 7 | BUY | 0.75% | 2.25% |
| 8 | BUY | 1.00% | 1.00% |
| 9 | BUY | 1.00% | 2.00% |
| 10 | BUY | 1.00% | 3.00% |

The nine BUY variants combine three SL values with reward-to-risk ratios 1, 2 and 3:

$$
TP=SL\times\text{reward-to-risk ratio}
$$

TP/SL selection is part of the joint categorical action. There is no continuous TP or SL output head. These brackets are separate from the saved GA/GP risk settings used to produce input signals.

When flat, HOLD and BUY variants are available. When invested, HOLD and SELL are available. When the stock is unavailable, only HOLD is available. Funding, session and position-cap restrictions are enforced separately during execution.

SELL requests close an existing long position; they do not open a short. The agent can SELL before TP/SL is reached. Brackets remain fixed after entry: the agent cannot trail or widen them in this version.

## 8. Policy-based stock selection

During training, actions are sampled from the categorical distribution. During validation and test, the largest individual action probability determines the requested action:

$$
a_{t,i}=\arg\max_a\pi_\theta(a\mid o_{t,i})
$$

Competing BUY requests are prioritized by aggregate BUY probability:

$$
B_{t,i}=\sum_{a\in\mathcal{B}}\pi_\theta(a\mid o_{t,i})
$$

Requests are processed in descending B, subject to cash and position slots. Stable ordering resolves equal priorities. A large aggregate BUY probability does not guarantee a BUY request if HOLD is the largest individual category.

This selection is not the trailing-three-month stock ranking used by the separate GA portfolio simulator. Existing positions are not automatically replaced whenever another stock gets a higher priority.

## 9. Trading environment and execution

| Setting | All-stock default |
| --- | ---: |
| Initial capital | USD 100,000 |
| Entry allocation | 4% of current equity |
| Maximum simultaneous positions | 10 |
| Direction | Long-only |
| Leverage | None |
| Shares | Whole shares |
| New-entry cutoff | Before 15:45 New York time |
| Scheduled liquidation | At or after 15:55 |

Ten positions at 4% imply approximately 40% entry allocation capacity. This is not a continuously rebalanced exposure ceiling: price movements change invested fractions.

### 9.1 Sizing

After eligible sales, the environment marks portfolio equity at current opens and calculates a shared per-entry budget:

$$
\text{Budget}_t=\alpha E_t^{open}
$$

$$
q_i=\left\lfloor\frac{\min(\text{Budget}_t,\text{Cash}_t)}{P_i^{open}}\right\rfloor
$$

Quantity is reduced until entry value plus fees fits available cash. It is held constant until exit. The budget is computed once before processing that timestamp's BUY requests.

### 9.2 Event order

At each next aligned timestamp:

1. Existing brackets are checked at observed opens.
2. Agent SELL requests execute where permitted.
3. BUY requests are processed by policy priority.
4. Brackets are checked at observed closes, including newly opened positions.
5. Session-end or final-observed-bar liquidation occurs.
6. Equity is marked and reward calculated.

An observation at close t produces discretionary orders at the next aligned timestamp's open. Missing observations may prevent execution. Requests from a preceding calendar day expire.

### 9.3 Bracket fills

For long positions:

$$
P_{SL}=P_{entry}(1-SL),\qquad P_{TP}=P_{entry}(1+TP)
$$

An open crossing exits at the observed open; a close crossing exits at the observed close. A bracket exit at the open takes priority over an agent SELL and prevents reopening that stock from a BUY request in the same step.

The simulator does not use intrabar High/Low to simulate bracket touches. SL and TP are not guarantees of exact realized exit prices. Missing bars defer checks. Each stock's final observed session bar is known from the dataset and used for liquidation, an idealization that uses end-of-data knowledge.

## 10. Accounting and charges

Portfolio equity is:

$$
E_t=\text{Cash}_t+\sum_iq_{t,i}C_{t,i}
$$

BUY and SELL cash updates are:

$$
\text{Cash}_{after\ buy}=\text{Cash}_{before}-qP-\text{BUY fee}
$$

$$
\text{Cash}_{after\ sell}=\text{Cash}_{before}+qP-\text{SELL fee}
$$

US mode uses the repository's `us_intraday_charges` helper: its encoded zero-commission scenario with SEC and FINRA charges on sales. This is an implemented fee assumption, not a freshly verified broker tariff or historical date-specific schedule. Income tax is not modeled.

The RL simulator does not currently model bid/ask spread, slippage, market impact, partial fills, circuit-blocked exits, borrowing, short selling or broker-specific settlement restrictions. It is not execution-equivalent to the full GA_STRATEGY portfolio simulator.

## 11. Reward

Let M be the running peak and D the nonnegative drawdown magnitude:

$$
M_t=\max(M_{t-1},E_t),\qquad D_t=1-\frac{E_t}{M_t}
$$

The transition reward is:

$$
r_t=\log\left(\frac{E_{t+1}}{E_t}\right)-\lambda\max(0,D_{t+1}-D_t)
$$

The default penalty coefficient is 1. Equity growth earns reward and worsening drawdown incurs a penalty. Recovery does not refund earlier penalties. Repeated deterioration after recovery can therefore generate multiple penalties; this is not equivalent to penalizing maximum drawdown only.

Learning scales the reward by 100:

$$
\tilde r_t=100r_t
$$

Reported financial returns are not multiplied by 100. Remaining in cash normally earns zero, making cash-only behavior a possible learned outcome.

## 12. Q calculation and critic interpretation

There is no Q-table or separate Q-network. The model is an actor–critic rather than DQN or Q-learning.

The theoretical action-value and state-value functions are:

$$
Q^\pi(S_t,A_t)=\mathbb{E}_\pi\left[\sum_{k=0}^{\infty}\gamma^k\tilde r_{t+k}\mid S_t,A_t\right]
$$

$$
V^\pi(S_t)=\mathbb{E}_\pi\left[\sum_{k=0}^{\infty}\gamma^k\tilde r_{t+k}\mid S_t\right]
$$

Here A is the joint vector of requested actions across stocks. The critic approximates V. A rollout return provides a sampled estimate of return following the selected joint action, not Q estimates for every possible alternative.

For a rollout ending at T:

$$
G_T=\begin{cases}0,&\text{terminal episode}\\V_\phi(S_T),&\text{otherwise}\end{cases}
$$

Working backward:

$$
G_t=\tilde r_t+\gamma G_{t+1},\qquad \gamma=0.99
$$

Advantage is:

$$
\hat A_t=G_t-V_\phi(S_t)
$$

G is a bootstrapped multi-step return target. Action probabilities are neither Q-values nor calibrated probabilities of profit. A discount of 0.99 applies per observed step; after 100 steps its weight is approximately 0.366, emphasizing a relatively short horizon on minute data.

## 13. Loss function

For T rollout steps and N stocks:

$$
L_{actor}=-\frac{1}{T}\sum_t\left[\sum_i\log\pi_\theta(a_{t,i}\mid o_{t,i})\right]\operatorname{stopgrad}(\hat A_t)
$$

$$
L_{critic}=\frac{1}{T}\sum_t(G_t-V_\phi(S_t))^2
$$

$$
H=\frac{1}{TN}\sum_{t,i}H(\pi_\theta(\cdot\mid o_{t,i}))
$$

The implemented total loss is:

$$
L=L_{actor}+0.5L_{critic}-0.001H
$$

All stocks share the portfolio advantage, so the agent does not receive a separate reward identifying each stock's contribution. The actor loss sums stock log probabilities, while entropy is averaged over stocks; changing universe size changes their relative scaling.

Policy probabilities also determine execution priority, an extra allocation heuristic. The loss uses requested actions, including requests that might not fill, rather than only executed trades. The implementation has no PPO clipping, replay buffer, target Q-network or generalized advantage estimation.

## 14. Training configuration

| Setting | Default |
| --- | ---: |
| Optimizer | Adam |
| Learning rate | 0.0003 |
| Discount | 0.99 per step |
| Rollout length | 256 |
| Updates | One per rollout |
| Gradient norm limit | 1.0 |
| Critic coefficient | 0.5 |
| Entropy coefficient | 0.001 |
| Epochs | 5 |
| Seed | 42 |

Every epoch replays the chronological training split from fresh capital. A CPU actor collects transitions. A learner performs batched updates on CPU or CUDA, after which its weights are copied to the actor. GPU accelerates learning updates, not price preparation, strategy replay, simulation or per-step inference. PyTorch CPU thread count is set to one by the all-stock trainer.

The default training window starts December 1, 2025 and ends before June 1, 2026. Validation covers June through August 2026. Test begins September 1, 2026. Boundaries are configurable. Each split starts with fresh cash and no positions.

Only training changes weights. Validation selects a checkpoint using:

$$
S_{validation}=R_{validation}-\lambda|MDD_{validation}|
$$

Since stored maximum drawdown is negative, code adds the weighted negative drawdown to return. This checkpoint score differs from the per-step reward. The best learned checkpoint is saved; the code does not automatically substitute cash if all learned scores are negative. RL monthly retraining is not implemented in this entry point.

## 15. Metrics

| Metric | Definition |
| --- | --- |
| Cumulative return | Final equity / initial equity minus one |
| Final equity | Final cash plus marked holdings |
| Maximum drawdown | Minimum equity / running peak minus one |
| Fees | Sum of order charges |
| Fills | Number of BUY and SELL executions |
| Validation score | Return minus weighted absolute maximum drawdown |

A round trip normally creates two fills. Maximum drawdown uses observed portfolio marks; it does not capture unobserved intrabar losses.

CAGR can be calculated from exported equity:

$$
CAGR=\left(\frac{E_T}{E_0}\right)^{1/Y}-1
$$

Y is actual elapsed calendar time in years. CAGR, Sharpe, Sortino and Calmar are not automatically reported by the current all-stock RL trainer. Annualizing a short test period does not establish an observed full-year return.

## 16. Saved outputs

| File | Contents |
| --- | --- |
| `prepared.npz` | Prices, features, timestamps and availability |
| `config.json` | Settings, stock ordering and feature metadata |
| `coverage.csv` | Available bars per stock |
| `policy.pt` | Selected weights, input/action dimensions and risk mapping |
| `training.csv` | Per-epoch validation statistics |
| `summary.json` | Final validation/test results |
| `valid_equity.csv`, `test_equity.csv` | Equity series |
| `valid_trades.csv`, `test_trades.csv` | Orders, fees, entry prices, brackets and reasons |
| Stock activity CSVs | Fill and fee totals by stock |
| `equity.png` | Validation/test portfolio plots |

Execution reasons distinguish `agent`, `stop_loss`, `take_profit` and `session_end`. Stock activity is not a stock profitability ranking. Previous checkpoints with three actions and thirteen inputs are incompatible with the new training architecture and require retraining.

## 17. Post-hoc analysis

The analyzer replays a checkpoint without updating weights. It records observations, masks, action probabilities, requested actions, aggregate BUY priority, resulting fills and exit reasons. It supports old three-action and current TP/SL checkpoints when the saved schema is compatible.

Local sensitivity for chosen action a is:

$$
\frac{\partial\pi_\theta(a\mid o)}{\partial o_j}
$$

A one-feature intervention replaces input j with zero, keeping other inputs and the action mask fixed:

$$
\Delta_j=\pi_\theta(a\mid o)-\pi_\theta(a\mid o_{j\leftarrow0})
$$

Positive delta means the observed value supports the selected action relative to zero. A separate intervention zeros all strategy inputs together and reports the resulting probability and preferred action.

Outputs include `decisions.jsonl`, `fills.csv`, `feature_effects.csv`, `feature_summary.csv`, `summary.json` and interpretation notes. Every trade and non-HOLD request is recorded; HOLD decisions are sampled at configured intervals. Costly explanations are chronologically capped, while replay continues to completion.

The analyzer compares final equity and fill count with the original summary. This is a consistency check, not a full fill-by-fill identity check. It prefers saved prepared arrays and can reconstruct inputs from recorded paths if those arrays are missing.

These diagnostics explain model sensitivity, not economic causation, SHAP attribution or calibrated confidence of profit. Gradients depend on units; zero-reference states can be unrealistic; capped chronological samples are not unbiased global importance estimates. A forced exit must not be described as a policy-chosen SELL. Reusing test explanations to tune the system makes that period development data rather than an untouched test.

## 18. Limitations and evidence status

1. The network has no temporal neural memory or explicit cross-stock attention.
2. Current drawdown and running peak affect reward but are not explicit observation features; the observation is therefore incomplete for the reward-generating account state.
3. The same portfolio advantage is applied to all requested stock actions, complicating credit assignment.
4. Cash-only behavior is a valid outcome of the objective and deterministic action selection.
5. Execution lacks spread, slippage, partial fills and a full exchange calendar.
6. Brackets use open/close checks rather than intrabar touch simulation.
7. Final-observed-bar liquidation assumes knowledge of the available data boundary.
8. Historical constituents, corporate actions and upstream feature causality need separate data audits.
9. Existing GA/GP strategy helpers retain their own session inference; explicit RL market settings do not independently repair upstream assumptions.
10. Earlier approximately 4.02% held-out return and 1.38% drawdown belonged to the three-action experiment, before RL-selected TP/SL. They are not results of the current eleven-action model.

The latest reported automated suite completed 22 tests covering selected accounting, risk-action, masking, learning-gradient and explanation behaviors. Unit checks do not establish market profitability or comprehensive realism. A controlled fresh experiment with the current schema, multiple seeds and consistent execution assumptions is needed for new performance claims.

## 19. Run commands

From the `full_predictor` directory, using the project environment:

```powershell
python "rl agent/train_all.py" --data-dir datasets/processed_datasets/us_stocks --results reports/batch_us_stocks_5x/monthly_results.csv --output reports/rl_us_tp_sl_run1 --epochs 5 --device cuda --capital 100000 --allocation 0.04 --max-positions 10 --drawdown-penalty 1 --start 2025-12-01 --train-end 2026-06-01 --valid-end 2026-09-01
```

After training:

```powershell
python "rl agent/analyze.py" --run reports/rl_us_tp_sl_run1 --split test --max-explanations 1000
```

Use a new output directory for each run. `--device cuda` requires CUDA-enabled PyTorch; `--device auto` falls back to CPU when necessary. Preserve configuration, prepared-data provenance and checkpoint metadata when comparing experiments.

## 20. Suggested methodology description

The implementation is a shared multi-stock actor–critic trading agent that combines price features and GA/GP strategy signals, selects discrete entry risk brackets, and learns from transaction-cost-adjusted portfolio returns with a drawdown-deterioration penalty. Chronological validation selects a checkpoint, and separate replay diagnostics examine its requested actions, execution outcomes and sensitivity to inputs.

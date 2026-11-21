# Batch Walk-Forward Optimization Runner

The `batch.py` runner executes **rolling walk-forward strategy optimization** across multiple Nifty 50 stocks concurrently, evaluating parameter grids and genetic programming rules on out-of-sample monthly test periods.

---

## 1. Quick Start

Run from the `full_predictor` directory:

```powershell
# Full 5-year run across all stocks with 8 CPU worker processes
python -m GA_STRATEGY.batch --workers 8 --years 5 --objective composite --output reports/composite_long_5years

# Test a single calendar year (e.g. 2022) on selected stocks
python -m GA_STRATEGY.batch --workers 4 --year 2022 --stocks RELIANCE HINDALCO ADANIENT --output reports/test_2022
```

---

## 2. Walk-Forward Methodology

1. **Expanding / Rolling Fit Windows:**
   * Every month, the engine uses the immediately preceding **6 calendar months** (`train-months = 6`) to fit strategies.
   * The candidate with the highest training score is selected and frozen for evaluation during the subsequent **1 calendar month** test window.
   * The window rolls forward one month at a time across the entire multi-year evaluation horizon.

2. **Competing Strategies:**
   * **VWAP Mean Reversion** (48 parameter combinations)
   * **RSI Mean Reversion** (36 parameter combinations)
   * **Bollinger Bands Reversion** (12 parameter combinations)
   * **Donchian Channel Breakout** (4 parameter combinations)
   * **Genetic Programming (GP)** evolved Boolean rule trees
   * All candidates compete under the same unpenalized training objective score to select the month's **`Selected`** winner.

3. **Continuous Compounding:**
   * Positions and cash balances compound continuously across month boundaries.
   * Switching strategies at month boundaries correctly charges turnover fees based on the prior position and new target.

---

## 3. Training Objectives (`engine/objectives.py`)

Set via `--objective [composite | calmar | blend]`:

* **`composite` (Default):**
  $$\text{score} = 0.5 \cdot \text{Sharpe} + 0.8 \cdot \text{Sortino} + 2.0 \cdot \text{DD} - 5.0 \cdot \text{DD}^2 + 0.4 \cdot \min(\text{Calmar}, 2.0)$$
  Progressively penalizes deep drawdowns with a quadratic penalty while capping outsized Calmar outliers.
* **`calmar`:** $\text{CAGR} / |\text{Max Drawdown}|$
* **`blend`:** $0.7 \cdot \text{Sharpe} + 1.3 \cdot \text{Sortino} + 0.5 \cdot \text{Max Drawdown}$

---

## 4. Intraday Risk Management (`engine/risk.py`)

All evaluations (both training and test) run with trade-level risk controls:
* **Stop Loss:** `0.75%` (`DEFAULT_STOP_LOSS = 0.0075`) — immediately cuts adverse moves to Cash (`0.0`).
* **Take Profit:** `1.50%` (`DEFAULT_TAKE_PROFIT = 0.0150`) — locks in quick reversion gains.
* **Long-Only Mode (Default):** Short triggers (`-1.0`) are converted into cash exit signals. To allow shorting, pass `--allow-short`.

---

## 5. CLI Arguments

| Argument | Default | Description |
|---|---|---|
| `--workers` | `1` | Number of concurrent stock worker processes. Recommended: 6 to 8 on modern 8-core CPUs. |
| `--years` | `None` | Evaluate the latest $N \times 12$ calendar months (e.g. `--years 5` = 60 months). |
| `--year` | `None` | Evaluate a single calendar year (e.g. `--year 2023`). |
| `--start-month` | `None` | First evaluation month (e.g. `2020-01`). |
| `--end-month` | `None` | Last evaluation month (e.g. `2025-07`). |
| `--objective` | `composite` | Fitness objective: `composite`, `calmar`, or `blend`. |
| `--long-only` | `True` | Only execute Long positions (no shorting). |
| `--allow-short` | — | Flag to allow both Long and Short positions. |
| `--limit` | `5` | Maximum number of stocks to process (`0` processes all available Parquets). |
| `--stocks` | `None` | Explicit list of stock symbols (e.g. `--stocks RELIANCE TCS INFOSYS`). |
| `--output` | `reports/batch` | Output folder for results, summaries, and plots. |
| `--gp-population` | `40` | Population size for GP rule tree evolution. |
| `--gp-generations`| `12` | Number of generations for GP evolution. |

---

## 6. Generated Output Files

* **`monthly_results.csv`:** Full record of every evaluated stock-month, strategy parameters, training scores, out-of-sample returns, drawdowns, and transaction costs.
* **`stock_summary.csv`:** Aggregated multi-year CAGR, Calmar ratio, cumulative return, and max drawdown per stock across all strategy families.
* **`stock_rankings.csv`:** Performance ranking of the walk-forward **Monthly Selected Strategy** for each stock.
* **`plots/STOCK.png`:** Multi-panel visualization comparing cumulative return curves of each strategy family, Selected, and Buy & Hold.
* **`plots/STOCK_daily_returns.csv`:** Daily compounded return series used to produce the plots.
* **`errors.csv`:** Log of any stocks that encountered errors or insufficient data history.


### FYERS charges in training and batch evaluation

See the [execution engine guide](engine/README.md) for the full fee and
accounting model, monthly compounding, report fields, and execution assumptions.

GA grid candidates and GP rules now optimize returns after the shared
`engine/fees.py` FYERS charges, including brokerage, STT, exchange charges,
SEBI, IPFT, stamp duty and GST. This supersedes the earlier fixed 0.0005
cost description. Income tax is not deducted. Current tariff estimates are
applied throughout historical data; this is not a historical tariff schedule.

Set `STARTING_CAPITAL` (default 1,000,000 rupees) and `ALLOCATION` (default
0.12) in `engine/execution.py`. Each strategy has its own unleveraged scoring
account. Whole shares are sized from current equity at entry and held until
exit; fees use actual order turnover and the shared per-order rounding.
A reversal incurs separate closing and opening orders. Training/validation/
test evaluations each start a fresh account. In batch, each strategy and
Selected carry their account equity from one evaluation month into the next.
Monthly and full-history reports expose fees, fills, charge components,
initial/final equity, allocation and fee-model metadata. Charts and rankings
use those same after-charge returns.

Execution assumption: signals fill at the observed close; new positions are
allowed before 15:00 and positions exit at 15:15 or the last observed bar of
the day if earlier. The latter assumes a successful early close even with
missing bars. Each evaluation slice liquidates its last position. These are
idealized intraday fills, without slippage, circuit restrictions, leverage,
forced broker square-off fees or delivery conversion. The portfolio engine
continues to handle its own next-open fills and funding/blocked-exit rules.
Sharing charges does not make these execution models identical. Risk signals
are still generated by the existing strategy risk logic.

Existing CSVs are unchanged. Rerun GA/GP and batch into a new output folder,
then use the new monthly_results.csv for portfolio selection. Previous
reports used different fee, sizing and holding assumptions, so they are not
directly comparable to the new reports.


## Optimizing stop loss, take profit, and reward-to-risk

GA parameter search and GP now select risk settings using training data only.
The shared search space is `config/risk.json`:

```json
{
  "stop_loss": [0.005, 0.0075, 0.01],
  "risk_reward_ratio": [1.0, 2.0, 3.0]
}
```

Stop losses are fractions of entry price: 0.005 means 0.5%. The
`risk_reward_ratio` field is specifically **reward divided by risk**, TP / SL.
Take profit is derived as `stop_loss * risk_reward_ratio`. For example,
SL 0.75% and ratio 2 produce TP 1.5%. These are gross price-distance targets,
not guaranteed realized gains/losses or ratios after fees.

GA searches the Cartesian product of indicator parameters and the nine risk
pairs. GP evaluates each unique rule under all nine pairs, retains its best
training pair, and uses that score (with the existing complexity penalty)
to evolve the rule population. This is nested risk-grid search, not mutation
of separate risk genes. Both use the configured after-charge objective.
Validation and test results do not choose risk settings. The reported RSI
baseline also receives training-only risk tuning.

Saved parameters include `stop_loss`, `take_profit`, and `risk_reward_ratio`.
GP reports also preserve their rule and, where requested, tree. Batch passes
the complete parameter dictionary to replay; portfolio does the same.
Older GP tree-only inputs and saved reports without risk keys still use the
previous defaults (0.75% SL, 1.5% TP). New settings require retraining; they
are not retroactively inferred for old CSVs.

Stops and targets are checked against observed closing prices. A crossing
can therefore exit beyond its nominal threshold. A stop or target exit holds
the signal at cash until its raw target changes or a new session begins.
Risk tracking resets on daily liquidation so the following session starts
with a fresh entry reference. It uses the scoring engine's existing session
cutoffs; it does not recalculate entry references from portfolio next-open
fills. Portfolio execution and scoring can still differ.

Allocation, leverage, trailing stops, and maximum holding time are not added
to this search. Allocation and leverage retain their existing controls.
With the default grid, GA has nine times as many combinations, and each GP
rule needs nine training evaluations. Reduce the lists in `risk.json` to
limit runtime; one value in each list fixes the risk settings. SL and derived
TP must be positive and below 100%, with a finite positive ratio.

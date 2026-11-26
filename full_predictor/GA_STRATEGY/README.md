# GA_STRATEGY: Walk-Forward Algorithmic Trading & Portfolio Engine

A quantitative trading and portfolio backtesting framework designed for high-frequency Indian equity data (1-minute OHLCV). The engine performs **monthly walk-forward strategy optimization** (Grid Search and Genetic Programming), applies **intraday trade-level risk controls** (SL/TP and Long-Only enforcement), and simulates a **cash-constrained multi-stock portfolio** with realistic **NSE circuit-breaker filters**.

---

## Architecture Overview

```
                      1-Minute Parquet Data (NIFTY 50)
                                     │
                                     ▼
 ┌────────────────────────────────────────────────────────────────────────┐
 │ 1. BATCH WALK-FORWARD OPTIMIZER (batch.py)                             │
 │    • Rolling 6-month train / 1-month test walk-forward                 │
 │    • 4 Parametric Strategies (RSI, Bollinger, Donchian, VWAP)          │
 │    • Genetic Programming (GP) Boolean rule tree evolution              │
 │    • Multi-Objective Optimization: composite / calmar / blend          │
 │    • Centralized Risk Management: 0.75% SL, 1.50% TP, Long-Only        │
 │    • Parallel processing across CPU cores (--workers N)                │
 └───────────────────────────────────┬────────────────────────────────────┘
                                     │
                                     ▼ Outputs: monthly_results.csv
 ┌────────────────────────────────────────────────────────────────────────┐
 │ 2. PORTFOLIO REPLAY SIMULATOR (portfolio.py)                           │
 │    • ₹1,000,000 starting cash balance (no leverage / borrow)           │
 │    • Trailing 3-month cross-sectional momentum ranking (Top 10 / 15)   │
 │    • Position sizing: capped at 4% notional equity per trade           │
 │    • Realistic Exchange Circuit Handling:                              │
 │        - Upper Circuit (UC): Rejects BUY fills (no sellers available)  │
 │        - Lower Circuit (LC): Traps positions until circuit opens       │
 │    • Daily marks, fee attribution, and cash ledger tracking            │
 └───────────────────────────────────┬────────────────────────────────────┘
                                     │
                                     ▼
                  Reports, Trade Logs, Equity Curves & JSON Metrics
```

---

## 1. Batch Optimization Engine (`batch.py`)

The batch runner evaluates strategies across multiple stocks using expanding/rolling walk-forward windows. Every month, each stock refits all candidate strategies on the preceding 6 calendar months of training data and deploys the highest-scoring candidate out-of-sample for the test month.

### Key Features
* **Multi-Core Parallelism:** Evaluates multiple stocks concurrently using `ProcessPoolExecutor`.
* **Out-of-Sample Integrity:** Zero lookahead bias; parameters and GP rules are frozen at month boundaries.
* **Continuous Compounding:** Stitches positions across monthly boundaries, correctly charging transaction turnover on strategy transitions.

### Command Reference
```bash
# Full 5-year walk-forward optimization across all stocks using 8 CPU cores
python -m GA_STRATEGY.batch --workers 8 --years 5 --objective composite --output reports/composite_long_5years

# Run specific stocks
python -m GA_STRATEGY.batch --stocks RELIANCE HINDALCO ADANIENT --years 3 --output reports/sample_run

# Allow both Long and Short positions (default is Long-Only)
python -m GA_STRATEGY.batch --workers 8 --years 5 --allow-short --output reports/long_short_run
```

### Batch CLI Options
| Argument | Default | Description |
|---|---|---|
| `--workers` | `1` | Number of concurrent stock worker processes. |
| `--years` | `None` | Number of trailing years to evaluate (e.g. `--years 5` = 60 months). |
| `--year` | `None` | Single calendar year (e.g. `--year 2022`). |
| `--objective` | `composite` | Training fitness objective: `composite`, `calmar`, or `blend`. |
| `--long-only` | `True` | Enforce long-only execution (shorts convert to cash exits). |
| `--allow-short` | — | Flag to allow both long and short positions. |
| `--train-months` | `6` | Number of historical months used for fitting. |
| `--limit` | `5` | Limit number of stocks (`0` processes all available Parquets). |
| `--stocks` | `None` | Specific stock symbols to process. |
| `--output` | `reports/batch` | Output directory for CSV reports and plots. |

---

## 2. Training Objectives (`engine/objectives.py`)

The engine supports 3 distinct training objectives for selecting winning configurations and evaluating GP fitness:

### 1. `composite` (Recommended Default)
Designed to maximize risk-adjusted return while aggressively penalizing deep drawdowns and capping outsized return outliers:
$$\text{score} = 0.5 \cdot \text{Sharpe} + 0.8 \cdot \text{Sortino} + \underbrace{2.0 \cdot \text{DD} - 5.0 \cdot \text{DD}^2}_{\text{Progressive Quadratic Penalty}} + 0.4 \cdot \min(\text{Calmar}, 2.0)$$
* **Quadratic Drawdown Term:** Disproportionately crushes high-drawdown candidates (a 40% DD incurs an exponential penalty compared to a 10% DD).
* **Calmar Clamping:** Limits Calmar contribution to $+0.8$, preventing extreme bull runs (e.g. 10x CAGR) from masking severe drawdowns.

### 2. `calmar`
$$\text{score} = \frac{\text{CAGR}}{|\text{Max Drawdown}|}$$
Selects the candidate with the highest annual growth relative to its historical peak-to-trough drop.

### 3. `blend`
$$\text{score} = 0.7 \cdot \text{Sharpe} + 1.3 \cdot \text{Sortino} + 0.5 \cdot \text{Max Drawdown}$$
The classic linear risk-weighted score favoring Sortino over Sharpe.

---

## 3. Trade-Level Risk Management (`engine/risk.py`)

All strategies (Grid and GP) pass raw trading signals through the central risk engine before execution.

```python
# Central configuration in engine/risk.py
DEFAULT_STOP_LOSS   = 0.0075   # 0.75% Stop Loss
DEFAULT_TAKE_PROFIT = 0.0150   # 1.50% Take Profit
DEFAULT_LONG_ONLY   = True     # Long-Only mode enabled
```

### Risk Mechanics
1. **Stop Loss (0.75%):** As soon as unrealized position loss reaches $-0.75\%$, the position is immediately cut to **Cash (`0.0`)**.
2. **Take Profit (1.50%):** Once position profit reaches $+1.50\%$, profits are locked and the position exits to **Cash (`0.0`)**.
3. **Re-Entry Protection:** When a position is stopped out, the strategy remains in Cash until the underlying indicator setup resets, preventing rapid-fire re-entry into a crashing asset.
4. **Long-Only Filter:** Negative signals (`-1.0`) act as pure **exit signals to Cash**, never initiating short positions. This eliminates shorting into strong secular trends.

---

## 4. Cash Portfolio Simulation (`portfolio.py`)

`portfolio.py` takes the winning monthly strategies from `monthly_results.csv` and simulates a realistic, cash-constrained delivery portfolio.

### Portfolio Mechanics
* **Starting Capital:** Default ₹1,000,000 cash ledger.
* **Position Sizing:** Max **4% marked equity** per trade (e.g. ₹40,000 notional at entry), rounded down to whole shares.
* **Cross-Sectional Ranking:** Ranks all stocks by **trailing 3-month compounded return**. Only the **Top N** (e.g. Top 10) stocks are eligible to enter positions in any given month.
* **Execution Delay:** Signals generated at bar $t$ close are executed at the **next bar's Open** ($t+1$).
* **Friction:** Charges **0.05%** (`DEFAULT_COST = 0.0005`) on all buy and sell notionals.

### Realistic Circuit-Breaker Handling
In real Indian exchanges (NSE/BSE), stocks frequently hit daily circuit limits (5%, 10%, 20% bands). The portfolio simulator enforces strict market realism:

1. **Upper Circuit (UC) Rejection:**
   * At Upper Circuit, there are zero sellers in the order book.
   * **Rule:** If $\text{Day Return} \ge +4.8\%$ and price is pinned flat ($\text{High} == \text{Low}$), all **BUY orders are canceled**. No phantom paper fills.
2. **Lower Circuit (LC) Trap Modeling:**
   * At Lower Circuit, there are zero buyers in the order book.
   * **Rule:** If a position needs to exit (via Stop Loss or strategy signal) during Lower Circuit, **the sale cannot execute immediately**. The position remains open, takes daily mark-to-market losses, and **only executes when the circuit unlocks** and trading volume resumes.

### Portfolio Command Reference
```bash
python -m GA_STRATEGY.portfolio \
    --results reports/composite_long_5years/monthly_results.csv \
    --capital 1000000 \
    --allocation 0.04 \
    --top 10 \
    --output reports/top10_portfolio
```

---

## 5. Directory Structure & Key Files

```text
GA_STRATEGY/
├── batch.py                 # Walk-forward batch runner across all stocks
├── portfolio.py             # Top-N cash-ledger portfolio replay with circuit filters
├── main.py                  # Single-stock backtester and walk-forward CLI
├── config/
│   └── strategies.json      # Parameter grid definitions for GA strategies
├── engine/
│   ├── objectives.py        # Composite, Calmar, and Blend fitness functions
│   ├── risk.py              # Central SL/TP and Long-Only risk management
│   ├── backtester.py        # P&L accounting, metric calculations, and ledger
│   ├── ga_optimizer.py      # Grid search optimizer for parametric strategies
│   └── gp_optimizer.py      # Genetic programming rule tree evolution engine
└── strategies/
    └── core_strategies.py   # Vectorized RSI, Bollinger, Donchian & VWAP strategies
```

---

## 6. Generated Output Artifacts

### Batch Optimizer Outputs (`reports/batch/`)
* **`monthly_results.csv`:** Detailed per-month results for every strategy and stock, including parameter JSONs, training scores, out-of-sample returns, drawdowns, and transaction costs.
* **`stock_summary.csv`:** Full 5-year compounded CAGR, Calmar ratio, cumulative return, and max drawdown for every stock.
* **`stock_rankings.csv`:** Ranking of all stocks based on the walk-forward selected monthly strategy.
* **`plots/STOCK.png`:** Equity curves comparing each strategy family, the monthly-selected approach, and Buy & Hold.

### Portfolio Outputs (`reports/portfolio/`)
* **`portfolio_value.png`:** Total marked portfolio equity vs. cash balance over time.
* **`portfolio_value.csv`:** Daily time series of total equity, cash, invested capital, and open position counts.
* **`executed_trades.csv`:** Complete ledger of every executed fill (timestamp, stock, BUY/SELL, shares, price, fees, and execution notes).
* **`monthly_stock_ranks.csv`:** Trailing 3-month scores, eligibility flags, and monthly ranks for each stock.
* **`summary.json`:** High-level summary of total return, daily max drawdown, total fees paid, fill count, and capital utilization.

---

## 7. Recommended Workflow Summary

```powershell
# Step 1: Run full 5-year walk-forward optimization (Long-Only, Composite Objective)
python -m GA_STRATEGY.batch --workers 8 --years 5 --objective composite --output reports/composite_5years_run

# Step 2: Simulate Top-10 cash-allocated portfolio with circuit filters
python -m GA_STRATEGY.portfolio --results reports/composite_5years_run/monthly_results.csv --capital 1000000 --allocation 0.04 --top 10 --output reports/portfolio_top10
```


### FYERS charges in training and batch evaluation

See the [execution engine guide](engine/README.md) for calculations, settings,
worked examples, report fields, and differences from portfolio execution.

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

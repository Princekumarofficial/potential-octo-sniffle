# Execution and fee engine

This engine estimates how a single stock strategy's account changes when it buys or sells whole shares and pays transaction charges. GA and GP use these after-charge returns to choose strategies. Batch uses them to evaluate each following month and build cumulative-return charts.

It is a historical scoring simulator. It does not send orders to a broker.

## 1. Where it fits

```text
Parquet prices and precomputed indicators
    -> Strategy parameters or GP rule
    -> Signals: buy/long (+1), cash (0), short (-1)
    -> execution_returns(): sizing, fills, equity, FYERS charges
    -> After-charge returns
    -> Sharpe / Sortino / drawdown / Calmar
    -> Optimization score and evaluation reports
```

| File | Responsibility |
| --- | --- |
| `execution.py` | Simulates one stock's positions, account value, and charges |
| `fees.py` | Calculates charges for each BUY or SELL order |
| `backtester.py` | Calculates performance statistics and report summaries |
| `objectives.py` | Converts statistics into the objective being maximized |
| `ga_optimizer.py` | Searches configured parameter combinations using the shared evaluator |
| `gp_optimizer.py` | Evolves rules and scores them through the same evaluator |
| `../batch.py` | Retrains monthly and combines evaluation months |
| `../portfolio.py` | Separately simulates the multi-stock portfolio |

The GA module currently searches the configured parameter grid. GP evolves rule trees. The fee engine does not generate signals or change the search algorithm.

## 2. Default settings

Edit these constants in [execution.py](execution.py):

```python
STARTING_CAPITAL = 1_000_000.0
ALLOCATION = 0.12
```

This means a fresh account starts with ₹10 lakh and can allocate 12% of its current equity to one position. Allocation is a **position-value limit**, not a maximum permitted loss. There is no leverage in this engine.

Each stock and each strategy has a separate hypothetical account. Adding their account values together does not represent a shared-capital portfolio. Use the portfolio module for that.

There are currently no GA/batch command-line flags for these two settings. Change the constants before starting a new run, including worker processes. Changing portfolio capital or allocation flags does not change training settings.

Direct callers can override the defaults:

```python
from GA_STRATEGY.engine.execution import execution_returns

returns = execution_returns(
    frame["Close"], signal,
    capital=1_000_000,
    allocation=0.12,
)
details = returns.attrs["execution"]
```

The function expects a nonempty, chronologically ordered price Series with a DatetimeIndex and positive, finite prices. Signals are aligned to those prices; missing signals become zero. Signal values must be -1, 0, or 1. Supply timestamps in exchange-local time: the engine does not convert time zones or validate an exchange calendar.

## 3. When trades happen

1. A signal fills at the **same observed bar's closing price**.
2. New positions are allowed only before 15:00.
3. Existing positions can close when their target changes.
4. At or after 15:15, the target is forced to cash.
5. The last observed bar of every date is also forced to cash. The final bar of an evaluation slice therefore liquidates any remaining position.

If the 15:15 bar is missing, an existing position closes on the next observed bar at or after that time, or on the day's final observed bar if it is earlier. The engine assumes the fill succeeds.

An unchanged target keeps the existing share quantity; it does not resize every minute. A reversal closes the old position and opens a new one, charging both orders. At or after 15:00, a reversal only closes the old position because new entries are prohibited.

Strategy stop-loss and take-profit logic runs upstream when signals are generated. This engine does not independently inspect intrabar highs/lows to trigger stops.

## 4. Position size and account value

When flat, cash equals equity. At entry:

```text
position budget = current equity × allocation
shares = floor(position budget / entry price)
```

The quantity is reduced if the position value plus the entry charge would exceed available cash. If no whole share is affordable, no order is executed and no fee is charged. Entry charges can take total spending slightly above the allocation budget, but not above available cash.

For a long position:

```text
cash after BUY = cash before BUY - shares × price - BUY charge
equity = cash + shares × latest close
cash after SELL = cash before SELL + shares × price - SELL charge
```

For a short position, the share quantity is negative. Selling increases cash; buying back reduces cash. Equity still equals `cash + signed shares × close`, so short-sale proceeds are not treated as profit. This is simplified short accounting, without a broker margin model.

The position is marked to each observed close. The function raises an error if account equity becomes nonpositive.

## 5. Transaction charges

Every executed order calls [fees.py](fees.py). The following are the **rates encoded in this repository**, whose tariff label says checked 2026-11-16. This document does not independently reverify the tariff.

Let `V = whole shares × execution price`, the order turnover in rupees.

| Component | Intraday calculation in code |
| --- | --- |
| Brokerage | `min(₹20, V × 0.0003)` per order |
| STT | `V × 0.00025` on SELL only |
| Exchange | `V × 0.000030699` on both sides |
| SEBI | `V × 0.000001` on both sides |
| IPFT | `V × 0.000000001` on both sides |
| Stamp duty | `V × 0.00003` on BUY only |
| GST | 18% of brokerage + exchange + SEBI + IPFT + admin + DP |
| Admin / DP | Zero in this scoring engine |

GST is calculated from unrounded component amounts. Each component is then rounded to ₹0.01 using decimal half-up rounding; rounded components are summed into the order fee. Small components may round to zero.

The shared helper also supports delivery and administrative square-off charges, but this scoring engine does not enable those options. Its scheduled exits are assumed to be strategy-submitted orders.

These are transaction taxes and charges, **not income tax on profits**. Current encoded rates are applied to all historical dates, without historical tariff changes. Contract-note aggregation may produce different rounding. The helper records its source links: [FYERS charges](https://fyers.in/charges-list) and [FYERS pricing](https://fyers.in/pricing).

## 6. Worked example

Start with ₹10,00,000, allocation 12%, and buy at ₹100 before 15:00:

```text
Budget                          ₹1,20,000.00
Whole shares                    1,200
BUY charge                      ₹31.68
Cash after entry                 ₹8,79,968.32
```

If all 1,200 shares are sold at ₹105:

```text
Sale value                      ₹1,26,000.00
SELL charge                     ₹59.82
Total charges                   ₹91.50
Gross trading profit            ₹6,000.00
Net trading profit              ₹5,908.50
Final account equity            ₹10,05,908.50
Account return                  0.59085%
```

The stock rose 5%, but only 12% of account capital was invested. The account therefore earns less than 0.6% after fees. This explains why allocation changes the reported account return.

## 7. Returns and optimization

For each observed bar:

```text
bar return = current equity / previous equity - 1
first bar return = first equity / starting capital - 1
cumulative return = product(1 + bar returns) - 1
daily return = product(1 + that day's bar returns) - 1
```

The first return includes any initial entry fee. Final liquidation fees are included. Fees are already inside returns and must not be subtracted again.

The shared statistics use:

- **Sharpe:** mean daily return / sample daily standard deviation × √252; no risk-free-rate subtraction.
- **Sortino:** mean daily return / root mean square of negative daily returns × √252. Nonnegative days contribute zero to the denominator's average.
- **Drawdown:** equity / running peak - 1, with the initial capital included as a peak. Maximum drawdown is the most negative value.
- **CAGR:** compounded growth annualized using actual elapsed timestamp time and 365.25 days per year.
- **Calmar:** CAGR / absolute maximum drawdown.

Undefined or nonfinite objective inputs are skipped. For example, zero drawdown makes Calmar undefined in the current implementation.

The fee change leaves the objective formulas unchanged:

| Objective | Formula being maximized |
| --- | --- |
| `blend` (default) | `0.7 × Sharpe + 1.3 × Sortino + 0.5 × dd` |
| `calmar` | `Calmar` |
| `composite` | `0.5 × Sharpe + 0.8 × Sortino + 2 × dd - 5 × dd² + 0.4 × min(Calmar, 2)` |

Here `dd` is negative. GP also applies its existing tree-complexity penalty during evolution. Batch compares candidate winners using the shared unpenalized training objective.

The old `DEFAULT_COST` and `finalize_strategy` helpers remain for compatibility and signal-generation call signatures. Their provisional returns are discarded by the GA evaluator and batch scoring path; reported execution returns come from this engine. Passing `cost` to `evaluate()` no longer selects the scoring fee model.

## 8. Training versus monthly batch accounts

Every standalone training, validation, and test evaluation starts a fresh account using the default capital. Training candidates therefore compete under the same initial funding assumptions.

Batch follows a different account lifecycle for out-of-sample results:

1. Train candidates on the preceding training window using fresh scoring accounts.
2. Choose the winning strategy using training results only.
3. Evaluate the next month for every candidate winner and for `Selected`.
4. Carry each account's ending equity into its own next evaluation month.
5. Concatenate those after-charge returns for summaries and charts.

`Selected` follows the monthly winner but has its own capital history. Its returns need not exactly equal the same month's standalone winner because account sizes, whole-share rounding and capped brokerage can differ.

Positions do not carry overnight or across months in this engine. Equity carries forward. Training starts fresh even when the out-of-sample account has grown or shrunk.

## 9. Report fields

Execution metadata is attached to the return Series under `returns.attrs["execution"]` and incorporated into reports.

| Field | Meaning |
| --- | --- |
| `initial_capital`, `final_equity` | Account value in rupees at the start/end |
| `allocation` | Fraction allowed for the position's entry value |
| `fee_model` | Label identifying tariff and close-fill assumptions |
| `fees` | Total transaction charges in rupees |
| `charge_brokerage`, `charge_stt`, etc. | Rupee totals for each charge component |
| `fills` | Number of executed BUY/SELL orders; a round trip normally has two |
| `position_changes` | Same order count as `fills` in execution reports |
| `cost_drag` | Total fees / starting capital for the reporting period |
| `long_exposure`, `short_exposure` | Fraction of observed bars ending with that position direction; not capital utilization |
| `long_pnl`, `short_pnl` | Gross directional trading P&L divided by period starting capital |
| `directional_hit_rate` | Fraction of held-position bar moves with positive gross P&L, including zero moves as non-wins; not trade win rate |

Full-history batch fees and fills are sums across months. Full-history cost drag uses the original starting capital; it is not the sum of monthly cost-drag percentages. Directional P&L is converted back to rupees before aggregation, then normalized by original capital. Exposure is weighted by observed bar count. Full-history batch `directional_hit_rate` is currently unset.

Monthly returns use `net_return` and `max_drawdown`; summary reports use `cumulative_return` and `dd`. Returns are fractions: `0.10` means 10%; `-0.20` drawdown means a 20% decline from the peak.

Batch outputs remain `monthly_results.csv`, `stock_summary.csv`, `stock_rankings.csv`, and per-stock charts/curve CSVs. Curve CSVs contain cumulative daily returns. Buy-and-hold is a fully invested price benchmark without this strategy's sizing or charges.

## 10. Differences from the portfolio simulator

| Assumption | This scoring engine | Portfolio simulator |
| --- | --- | --- |
| Accounts | Independent account per stock/strategy | Shared capital across eligible stocks |
| Fill price | Signal bar's close | Its next-open execution rules |
| Leverage | None | Configurable leverage and funding rules |
| Blocked exits | Assumes successful fills | Circuit/blocked-exit handling |
| Overnight positions | Assumed liquidated daily | Can require delivery handling |
| Charges | Shared FYERS helper, intraday calls | Same helper with applicable execution options |

Sharing the charge calculator does not make the two simulations identical. Same-close fills can be optimistic because a signal derived from a closing price is assumed executable at that price. Early liquidation on a day's last observed bar also uses knowledge of where that day's data ends. Missing bars, slippage, bid/ask spread, liquidity, partial fills and exchange order restrictions are not modeled here.

## 11. Running and checking

From the `full_predictor` directory, using the project's Python environment:

```powershell
# Parameter-grid search with the default objective
python -m GA_STRATEGY.main

# GP rule search
python -m GA_STRATEGY.main --mode gp

# Five stocks, latest five years, monthly retraining
python -m GA_STRATEGY.batch --limit 5 --years 5 --workers 4 --output reports/batch_fyers

# Same evaluation using Calmar as the objective
python -m GA_STRATEGY.batch --limit 5 --years 5 --objective calmar --workers 4 --output reports/batch_fyers_calmar

# Focused accounting and portfolio regression checks
python -m unittest GA_STRATEGY.test_execution GA_STRATEGY.test_portfolio
```

Existing reports are not rewritten automatically. Generate new batch reports before using their monthly results for portfolio selection. Old fixed-percentage-cost results differ in sizing and execution assumptions as well as fees, so a performance change cannot be attributed solely to charges.


## Optimizing stop loss, take profit, and reward-to-risk

GA parameter search and GP now select risk settings using training data only.
The shared search space is `../config/risk.json`:

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

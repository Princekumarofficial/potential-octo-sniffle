# Finding trading rules with genetic programming

The optional `--objective calmar` switches fitness to calendar-time CAGR divided by absolute maximum drawdown, with the existing node-size penalty. The default `--objective blend` uses the original formula described below. The batch runner uses the same objective for monthly strategy selection.

## What is a trading rule?

A trading rule answers a question at the end of each price bar. A bar is one recorded minute of trading in this project. For example, the rule `rsi_14 < 30` asks whether a measure of recent price momentum is below 30. If the answer is true, the strategy aims to hold the stock. If it is false, the strategy aims to hold cash.

The [default grid search](README.md) starts with rules written by a person and tries different numbers inside them. Genetic programming, or GP, can also change which conditions appear and how they are joined. For example:

```text
(rsi_30 > 25 OR bb_position_60 < 1)
```

Here, either condition can tell the strategy to hold the stock. `AND` requires both conditions. A rule is called a **tree** because a condition with two parts can branch into smaller conditions, each of which can branch again.

## Run it

From `full_predictor`, with the project’s Python environment active:

```powershell
python -m GA_STRATEGY.main --mode gp --plot reports/gp_cumulative_returns.png
```

The default input is the RELIANCE Parquet. To build it from minute data, run:

```powershell
python -m helpers.feature_engineering --csv "datasets/NIFTY50/RELIANCE.csv"
```

Use `--data "C:\data\STOCK.parquet"` for a different stock. Use `--gp-seed`, `--gp-population`, and `--gp-generations` to specify the random seed, number of candidate rules per round, and number of rounds. The defaults are 42, 40, and 12. The same input and settings reproduce the same search. GP uses the project’s existing Python and pandas dependencies.

## What the search can use

| Input | Plain-language meaning | Thresholds the search can try |
|---|---|---|
| `rsi_14`, `rsi_30` | Compare recent upward and downward price moves over 14 or 30 bars. Higher RSI means upward moves dominated. | 25, 30, 40, 50, 60, 70, 75 |
| `bb_position_20`, `bb_position_60` | Locate the close between the lower and upper Bollinger bands, calculated over 20 or 60 bars. Zero is at the lower band. One is at the upper band. | 0, 0.25, 0.5, 0.75, 1 |

The [loader](main.py) uses valid observed prices and masks indicator values that may depend on imputed candles. Each rule goes flat if any indicator it references is missing. This includes warmup bars and masked bars. A return over a price gap is assigned to the later retained bar, using the position decided at the earlier retained bar.

## How rules improve over rounds

1. Generate random rules and include the simple `rsi_14 < 30` baseline in the first population.
2. Backtest every distinct rule on the training period and keep its score.
3. Keep the two best candidates. Pick parents by drawing three candidates and choosing the highest scorer. Make a child by replacing a branch of one parent with a branch from another. In 35% of child attempts, generate a new branch instead of borrowing it. Keep trees at depth four or less, counting the root at depth zero. Add one fresh random rule per round.
4. After the last round, freeze the highest-scoring training rule. Report it and the RSI baseline on training, validation, and test. The default split is six months, one month, and one month in chronological order.

This uses the same subtree crossover and mutation idea as the [reference engine](../../ga-strat-app/gp_strategy/gp_engine.py). Here, the inputs are saved minute indicators, and candidate returns use the same cost model as the fixed strategies.

## How a rule earns a score

The signal from one bar applies to the next retained close-to-close move. Each entry or exit costs 0.05% of equity in the model. Each split starts in cash. Returns compound, so gains and losses change the equity available for the next move.

```text
fitness = 0.7 × Sharpe + 1.3 × Sortino + 0.5 × signed max drawdown
          − 0.01 × number of nodes
```

Sharpe measures average daily return relative to everyday volatility. Sortino measures it relative to downside variation. Drawdown measures the worst fall from a previous equity peak and enters the formula as a negative number. Each comparison and each AND or OR counts as one node. The size penalty favors simpler rules when their performance is close. Candidates with undefined Sharpe or Sortino are skipped for selection.

## How to read the results

For a monthly experiment, add `--walk-forward 3 --train-months 6` to the GP command and omit `--plot`. The program learns a new rule from the prior six months before each of the last three calendar months. Each month’s displayed result comes from the rule chosen before that month began. See the [monthly mode guide](README.md#monthly-walk-forward-mode).

- **Training** is the practice period used to pick the rule. Its performance can benefit from trying many alternatives.
- **Validation and test** show how the selected rule performed later. The rule stays fixed in both periods.
- **Net return** tells you how much the strategy gained or lost after costs. **Exposure** tells you the fraction of bars the rule held the stock. The dashed line in the optional chart is buy and hold, which provides context when the stock itself rises or falls.
- The **RSI baseline** is a single long-or-cash condition evaluated under the same fees. The grid-search RSI strategy uses different position rules and is available through the default command.

See the [main README](README.md) for the detailed return formulas and [GP optimizer](engine/gp_optimizer.py) for the implementation.

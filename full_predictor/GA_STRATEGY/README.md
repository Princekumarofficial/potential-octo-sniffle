# GA_STRATEGY

Use the [batch runner](BATCH_README.md) to evaluate grid and GP strategies across multiple stocks and export monthly results, stock summaries, and rankings.

See the [beginner’s guide to genetic programming](GP_README.md) for the optional GP mode.

## Monthly walk-forward mode

From `full_predictor`, run:

```powershell
python -m GA_STRATEGY.main --walk-forward 3 --train-months 6
python -m GA_STRATEGY.main --mode gp --walk-forward 3 --train-months 6
```

These commands evaluate the three latest calendar months, including the latest month if it is partial. At each month boundary, the optimizer selects parameters or a GP rule using the preceding six calendar months. It freezes the selection for that evaluation month. Those bars become eligible for fitting in the following month. Each month starts flat, and the report shows monthly selections, monthly net returns, and returns compounded across evaluation months. If a strategy has no eligible candidate in a month, its return and full-period compound are shown as unavailable. The chart and validation/test length options apply to the fixed-split report.

`GA_STRATEGY` selects the parameters of four fixed trading rules. The current implementation exhaustively checks 100 configured combinations using a **training-only score**. The package name is historical. The search is deterministic and uses no population, mutation, or generations.

## Run the backtest

From `full_predictor`, using the project’s Python environment:

```powershell
python -m helpers.feature_engineering --csv "datasets/NIFTY50/RELIANCE.csv"
python -m GA_STRATEGY.main
```

The first command regenerates the stock’s Parquet, replacing the file if it already exists. The tester defaults to `datasets/processed_datasets/NIFTY50/RELIANCE.parquet`. Select a different file or split lengths with:

```powershell
python -m GA_STRATEGY.main --data "C:\data\STOCK.parquet" --train-months 6 --valid-months 1 --test-months 1
```

To save a PNG of daily compounded returns for the winning configurations and buy and hold, run:

```powershell
python -m GA_STRATEGY.main --plot reports/ga_cumulative_returns.png
```

Each panel compounds from its own starting equity and shares the same percentage scale.

## Data loaded

1. [Feature engineering](../helpers/feature_engineering.py) reads a per-stock CSV with `date, open, high, low, close, volume`. It sorts by timestamp and rejects invalid or duplicate timestamps. It aligns the data to the expected market-minute calendar in `eda/eda_outputs/expected_market_minutes.csv`, retaining scheduled minutes from the first through the last source date. Raw timestamps outside the calendar are excluded.
2. Each valid candle has finite OHLCV, positive prices, nonnegative volume, and consistent high and low bounds. For an eligible gap of one to three missing minutes, the builder fills each missing candle with the same OHLCV mean from equally many valid observed bars on each side. The surrounding bars must be contiguous and in the same session. Filled rows retain `is_missing_bar` and also have `is_imputed_bar` set.
3. The builder stores the datetime index, OHLCV, quality flags, a `segment_id`, and precomputed features. Indicator history resets at session boundaries, timestamp gaps, and remaining invalid or missing bars. A full rolling window is required for the rolling features. RSI smooths gains and losses using an exponential mean with alpha `1 / period`. Bollinger bands use the rolling close mean plus or minus the configured multiple of sample standard deviation. Donchian uses the highest high and lowest low over the preceding bars, excluding the current candle. Rolling VWAP uses the sum of close times volume divided by the sum of volume. Trend SMA uses the rolling mean close.
4. The [tester loader](main.py) sorts the Parquet and requires a unique datetime index without missing timestamps. It masks RSI, Bollinger, Donchian, VWAP, and SMA values from the first imputed bar through the end of that segment because the fill can use later prices. It retains observed, valid candles and drops imputed and other missing or invalid rows. An observed bar with masked features can still contribute price returns to an existing position.
5. The strategies use the saved `rsi_*`, `bb_upper_*`, `bb_lower_*`, `donchian_upper_*`, `donchian_lower_*`, `vwap_*`, and `sma_*` columns. VWAP also calculates its strategy-specific deviation statistic. If required indicator columns are missing, regenerate the file with the current feature builder. Other saved features, including GARCH and MACD, do not enter the four rules.

## Training, validation, and test

The split uses calendar-month offsets from the latest retained bar. By default, test covers the final month, validation covers the preceding month, and training covers the six months before validation. Each cutoff belongs to the later split. The cutoffs preserve the latest timestamp’s time of day, so a boundary can fall within a session. Every split must contain bars. A source with limited history can provide a shorter training period.

Each split starts with unit equity and a flat position. Precomputed features can use earlier bars in the same segment across a split boundary. The VWAP deviation statistic is calculated independently in each split. See [the entry point](main.py) and [the optimizer](engine/ga_optimizer.py).

### Parameters searched

| Strategy | Values tested | Combinations |
|---|---|---:|
| VWAP mean reversion | VWAP and deviation window `20, 40, 60, 120`, threshold `1.5, 2, 2.5, 3`, trend SMA `100, 200, 300` | 48 |
| RSI mean reversion | RSI period `10, 14, 20, 30`, oversold `25, 30, 35`, overbought `65, 70, 75` | 36 |
| Bollinger mean reversion | Period `20, 40, 60, 80`, standard-deviation multiplier `1.5, 2, 2.5` | 12 |
| Donchian breakout | Channel period `20, 50, 100, 150` | 4 |

Periods are in bars. The [strategy grid](config/strategies.json) defines the candidates. Added periods or Bollinger multipliers require corresponding feature columns in the Parquet.

For each strategy, the optimizer maximizes this score computed from **training returns after costs**:

```text
score = 0.7 × Sharpe + 1.3 × Sortino + 0.5 × max_drawdown
```

`max_drawdown` is a signed decimal, so `-0.10` subtracts `0.05` from the score. The ratios use compounded daily returns. Candidates with undefined Sharpe or Sortino are skipped. For ties, the first configuration in grid order wins. One winning configuration per strategy is evaluated unchanged on validation and test. The score favors risk-adjusted performance, with greater weight on Sortino than Sharpe. Cumulative return appears in the report.

### Signal rules

| Strategy | Go long | Go short |
|---|---|---|
| VWAP | Deviation statistic below the negative threshold and close above trend SMA | Deviation statistic above the positive threshold and close below trend SMA |
| RSI | RSI below oversold | RSI above overbought |
| Bollinger | Close below lower band | Close above upper band |
| Donchian | Close above prior channel high | Close below prior channel low |

VWAP’s deviation statistic is `(close − rolling VWAP) / (rolling sample std of close − VWAP differences + 1e-12)`. Its denominator uses a full window of differences within the same segment and split. It divides the current difference by its rolling standard deviation without subtracting the rolling mean of the differences.

Signals are target positions `+1` for long and `−1` for short. The strategy is flat until its first entry and then holds its last position until an opposite signal. Equality, missing indicators, and session changes generate no new signal on their own. There are no stop-loss or take-profit rules. See [strategy implementations](strategies/core_strategies.py).

## Execution and return calculations

For retained bar `t`, let `C[t]` be the close and `s[t]` the target position decided at that close. The position held over the price move ending at `t` is `s[t-1]`.

```text
market_return[t] = C[t] / C[t-1] - 1
gross_return[t]  = s[t-1] × market_return[t]
turnover[t]      = abs(s[t] - s[t-1])
net_return[t]    = gross_return[t] - 0.0005 × turnover[t]
equity[t]        = equity[t-1] × (1 + net_return[t])
```

- A signal can earn the next close-to-close move. Its execution cost is charged on the signal bar. A flat-to-position entry costs 5 basis points and a direct long-to-short or short-to-long reversal costs 10 basis points.
- At the first retained bar of each split, the previous position and market return are zero. A new entry still incurs its cost. The last bar is marked to market, with no forced liquidation or extra closing fee.
- Returns span adjacent retained closes, including overnight moves and gaps within a split. Positions can carry across sessions and gaps even though the indicators reset by segment. A cross-session move is attributed to the date of the later bar.
- The model uses close prices and a flat turnover fee. It has no separate spread, slippage, financing, borrow fees, short availability, liquidity limits, or margin model. Short exposure is modeled as the negative of the underlying simple return.

## Report metrics

| Metric | Calculation |
|---|---|
| Net compounded return | Product of `1 + net bar return` over the split, minus one |
| Buy and hold | Last retained close divided by first retained close, minus one, before costs |
| Daily returns | Product of `1 + bar return` within each date, minus one |
| Sharpe | Mean daily net return divided by sample daily standard deviation, multiplied by `sqrt(252)`. The risk-free rate is zero |
| Sortino | Mean daily net return divided by the root mean square of `min(daily net return, 0)` across all dates, multiplied by `sqrt(252)`. The target return is zero |
| Max drawdown | Lowest value of `equity / running peak equity − 1`, including starting equity in the peak |
| Long and short exposure | Fraction of retained bars whose incoming position was long or short, respectively |
| Position changes | Count of nonzero changes in target position, including the first entry. A reversal counts as one change and two units of turnover |
| Long and short P&L | Sum of opening bar equity times gross bar return for the corresponding incoming position, in units of starting capital |
| Cost drag | Sum of opening bar equity times bar execution cost, in units of starting capital |
| Directional hit rate | Fraction of positioned bars with a nonzero market move that earned positive gross return |
| Market daily volatility | Sample standard deviation of the underlying’s daily returns for the split |
| Worst session | Date with the lowest compounded daily net return, and that return |

Long P&L plus short P&L minus cost drag reconciles to net compounded return. Exposure and hit rate are bar-based. The report displays undefined ratios as `n/a`, such as Sharpe with zero daily volatility and Sortino with zero downside deviation. See [backtester and reporting code](engine/backtester.py).

To investigate a train-to-test drop, compare long and short P&L with the underlying direction and exposure in each split. Positive gross P&L alongside negative net return identifies a period where modeled costs absorbed the gains. Compare hit rate with P&L and the worst session to distinguish move frequency from move size. Training is the period used to choose the winner, so a strong training score can reflect parameter selection on that sample. The held-out figures describe the subsequent periods for that same configuration.

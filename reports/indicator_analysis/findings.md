# Findings from the initial indicator screen

The 32-feature compact subset outperformed all 61 features at each tested horizon, but neither beat price persistence on average. The recent-returns baseline performed best among the three main feature sets. This does not establish that every technical indicator is useless: the experiment uses a fixed linear ridge model, NIFTY 50 features only, and three chronological folds.

| Horizon | All 61: MSE skill | Compact 32: MSE skill | Recent returns: MSE skill |
| --- | ---: | ---: | ---: |
| 5 minutes | -0.406% | -0.067% | +0.006% |
| 15 minutes | -0.813% | -0.177% | -0.092% |
| 60 minutes | -1.594% | -0.185% | -0.061% |

Positive skill means lower prediction error than unchanged price. These are equally weighted means across folds, not trading returns. The tiny positive result for recent returns at five minutes was not positive in every fold.

## Candidates for reduction

- Keep either `bb_position_20` or `price_zscore_20`: they contain exactly the same information under an affine transformation.
- Keep one of `return_1` and `log_return_1` in an initial compact model. They have the same ranking and are nearly identical at small returns, although their transformations differ.
- Prefer relative moving-average distances to retaining every absolute moving-average level. The compact experiment excludes 18 absolute-level features, including raw MACD, ATR, and band levels. This is a modeling choice, not proof these levels cannot help another model.
- Several SMA/EMA distances have training rank correlations above 0.95. The exact proposed substitutions are recorded in `compact_removals.csv`. Correlation alone is not sufficient evidence to permanently discard a feature.
- RSI was also highly correlated with a retained trend feature in this dataset. Do not interpret this as a universal reason to remove RSI.

No generated feature was constant. Missing fractions ranged from zero to roughly 0.038%; these include both warm-up values and undefined ratios for flat ranges.

The largest absolute mean holdout rank association was about 0.023 at five minutes, 0.021 at fifteen minutes, and 0.042 at sixty minutes. These are weak univariate relationships, not demonstrated forecast accuracy.

Removing candle-shape features slightly improved compact-model mean skill at all three horizons. Removing oscillator features slightly worsened it at all three horizons. Other family effects varied with horizon; none established a strong forecasting advantage.

## Next decision

Preserve all original columns. Use `compact_features.json` as a candidate subset and recent returns as a baseline for the actual forecasting model. Choose the intended horizon before selecting a final set. Confirm any further selection with chronological validation and a new untouched test period; the folds in this analysis have already informed these findings. Nonlinear models and aligned VIX/NIFTY BANK inputs remain untested.

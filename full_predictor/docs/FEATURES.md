# Feature dataset

The feature section of `eda.ipynb` produces one Parquet file per stock in
`processed_datasets/NIFTY50/`. The current build covers **97 stocks, 87.4 million
minute rows, and 35 features**. It excludes IRFC, HDFCLIFE, and SBILIFE.

- **Indicators.** SMA, EMA, RSI, Bollinger Bands, trend and slope, simple and log
  returns, MACD, rolling minute volatility, and volume features. Intraday indicators
  restart at session boundaries and after missing or invalid bars.
- **Volatility.** The daily GARCH(1,1) forecast uses prior daily returns and is
  attached to each valid minute of the forecast session. The rolling volatility
  feature is calculated separately from minute returns.
- **Missing data.** All 102,724 missing minute bars remain NaN. Indicator warm-up
  and unavailable GARCH forecasts also remain NaN. Features are available at bar
  close for predicting later bars.
- **Validation.** All 97 files passed timestamp, feature range, gap, and Parquet
  round-trip checks. Nine GARCH fits failed and their unavailable forecasts remain
  NaN. See `feature_validation.csv`, `feature_nan_report.csv`, and
  `garch_fit_audit.csv` in `processed_datasets/` for details.

## Run or rebuild

Run the feature-engineering section of `eda.ipynb`, or call
`feature_engineering.transform_all(base_dir)`. Requires pandas, NumPy, arch and
pyarrow. Outputs are under `processed_datasets/NIFTY50`, one Parquet per stock.
IRFC, HDFCLIFE and SBILIFE are excluded. Raw CSVs are never modified.

The notebook loads existing results by default; set `REBUILD_FEATURE_DATA=True`
to regenerate them after changing the inputs, calendar, or implementation.
For parallel CPU processing from the project root, run
`python -m full_predictor.feature_engineering --workers 3`.

## Missing data and timing

The saved `eda_outputs/expected_market_minutes.csv` defines the inferred market
calendar. Each stock is reindexed to this grid between its first and last observed
dates. Missing OHLCV values remain NaN. No rows are dropped after feature creation,
no price interpolation is performed, and missing volume is not set to zero.
Off-grid/out-of-session rows are excluded and counted. Duplicate timestamps fail
explicitly. Invalid observed bars are preserved but their features are NaN.

`is_missing_bar`, `is_valid_bar`, and `segment_id` are metadata, not prices.
Intraday features restart after an invalid/missing bar, a nonconsecutive minute,
or a new session. Warm-up NaNs are intentional. Do not pass all columns blindly
to a model or globally drop NaNs and then construct row-based sequences: training
windows must respect segment boundaries and have observed targets.

All features use information available at the **close of the current bar**.
They are inputs for future targets, not predictions of the current bar. The
dataset has no shifted target, normalization, or train/test split yet.

## Definitions

| Feature | Definition |
| --- | --- |
| `returns`, `log_returns` | `close / previous_close - 1`, `log(close / previous_close)` within a continuous segment |
| `sma_{5,10,20,50,100}` | Trailing mean of closes; full window required |
| `ema_{5,10,20,50,100}` | Recursive EMA, alpha=2/(span+1), seeded from first segment close; span observations required |
| `rsi_14` | Gain/loss EWMs with alpha=1/14, recursive seed and 14 changes required; flat=50, positive gain with zero loss=100. This is a Wilder-style smoothing variant, not an initial-SMA seed. |
| `bb_middle_20`, `bb_upper_20`, `bb_lower_20` | SMA20 and plus/minus twice the sample standard deviation (ddof=1) |
| `bb_width_20`, `bb_position_20` | Width divided by middle; close's fractional position in band; zero width gives NaN position |
| `trend` | Sign of SMA20 minus SMA50: -1, 0, +1 |
| `trend_strength` | SMA20/SMA50 - 1 |
| `trend_slope_20` | Trailing 20-close least-squares slope, price units per minute bar |
| `trend_slope_pct_20` | Slope/SMA20 * 100 |
| `macd`, `macd_signal`, `macd_histogram` | EMA12-EMA26; EMA9 of MACD; MACD minus signal |
| `rolling_volatility_30` | Sample standard deviation of 30 consecutive minute log returns, decimal units, not annualized |
| `garch_daily_volatility` | Forecast daily log-return standard deviation, decimal units, available before the session; see below |
| `volume_sma_20`, `relative_volume_20` | SMA20(volume); volume/SMA20(volume) |
| `volume_zscore_20` | (volume-mean20)/sample_std20; zero standard deviation yields NaN |
| `volume_change` | volume/previous_volume-1; zero denominator yields NaN |
| `log_volume` | log(1+volume) |
| `dollar_volume` | close*volume; historical name, actually quote-currency turnover proxy (INR here) |
| `obv_segment` | Cumulative sign(close change)*volume, starting at zero in each continuous segment |
| `vwap_segment` | Cumulative typical-price*volume divided by cumulative volume, restarting at gaps/session changes; not an exchange trade-level VWAP |

## GARCH fitting

This is a **daily**, not minute-level, GARCH(1,1), zero-mean Gaussian model.
It uses the scheduled final minute's observed close. A missing final close stays
missing; it is not replaced by an earlier close. Daily returns span consecutive
scheduled sessions, including overnight changes, unlike the intraday returns.

Parameters are refitted at monthly boundaries using at most 252 prior daily
returns from the latest contiguous valid suffix (minimum 60 returns). Fits never
include the forecast day's close. Between fits, conditional variance updates
using the previous daily return. The forecast is attached to all valid minute
bars of the forecast session. Calibration/warm-up periods and unavailable forecasts
remain NaN. If historical continuity breaks, the state is cleared; refitting is
retried on the next monthly boundary. Nonconvergent fits are logged and forecasts
remain unavailable until a successful fit. No fallback volatility is relabelled GARCH.

Returns are multiplied by 100 for fitting and forecast sigma divided by 100 for
storage. See [arch model documentation](https://arch.readthedocs.io/en/stable/univariate/introduction.html).
This CPU implementation does not require or use CUDA.

## Validation artifacts

- `feature_validation.csv`: stock-level row counts, missing/invalid rows, complete
  feature rows, available GARCH rows, and serialization validation status.
- `feature_nan_report.csv`: per-feature NaN and finite counts, including NaNs on
  observed valid bars (warm-up, zero denominators, or unavailable GARCH).
- `garch_fit_audit.csv`: fit dates, final historical dates, sample counts,
  parameters, warnings and failures.
- `feature_manifest.json`: stock universe, source metadata, calendar/code hashes,
  feature policies, GARCH settings and dependency versions.

Checks enforce sorted unique timestamps, preserved missing raw rows, NaN features
on invalid bars, no infinities, RSI/trend bounds, Bollinger ordering, nonnegative
volatility, consistent returns, no cross-segment returns, and exact Parquet
round-trip values/null masks. Synthetic validation additionally checks linear
trend slope, flat-price behavior, gap warm-up, and prefix invariance: adding
future data does not change existing feature values or GARCH forecasts.

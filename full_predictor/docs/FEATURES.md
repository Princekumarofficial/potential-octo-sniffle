# Feature reference

The transformation writes one Parquet file per stock to `datasets/processed_datasets/NIFTY50/`.
For inputs, rebuild instructions, and validation reports, see [Transformation flow](DATA_TRANSFORMATION_FLOW.md).

## Missing bars and feature timing

The expected market-minute grid covers each stock's observed date range. Eligible gaps of one to three minutes use a per-column OHLCV average of equally many consecutive, valid observed bars before and after the gap, within the same session. Other gaps remain NaN. Invalid observed bars retain their source values and have NaN indicators.

| Metadata | Meaning |
| --- | --- |
| `is_missing_bar` | Timestamp was absent in the source, including rows filled later. |
| `is_imputed_bar` | The short-gap policy filled the bar. |
| `is_valid_bar` | The bar is usable for feature calculation, including eligible fills. |
| `segment_id` | Continuous run used for intraday feature calculations. |

Intraday indicators restart at session boundaries, timestamp gaps, and missing or invalid bars. The first 99 bars of a full segment are warm-up for SMA100. Feature NaNs can also arise from zero denominators and unavailable GARCH forecasts.

Two-sided fills use later observations and can influence subsequent indicators. For forecasting, the window datasets mask those fills and rebuild indicators using observed bars. See [Data loaders](DATA_LOADERS.md).

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
It uses the scheduled final minute's valid close. A missing final close stays
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

- `feature_validation.csv`: stock-level row counts, missing, imputed, remaining missing, and invalid rows, complete
  feature rows, available GARCH rows, and serialization validation status.
- `feature_nan_report.csv`: per-feature NaN and finite counts, including NaNs on
  valid bars (warm-up, zero denominators, or unavailable GARCH).
- `garch_fit_audit.csv`: fit dates, final historical dates, sample counts,
  parameters, warnings and failures.
- `feature_manifest.json`: stock universe, source metadata, calendar/code hashes,
  feature policies, GARCH settings and dependency versions.

Checks enforce sorted unique timestamps, preserved unfilled missing raw rows, NaN features
on invalid bars, no infinities, RSI/trend bounds, Bollinger ordering, nonnegative
volatility, consistent returns, no cross-segment returns, and exact Parquet
round-trip values/null masks. Synthetic validation additionally checks linear
trend slope, flat-price behavior, gap warm-up, and prefix invariance for indicators computed from observed bars and for daily GARCH forecasts.

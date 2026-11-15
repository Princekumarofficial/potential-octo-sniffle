"""Gap-aware OHLCV features used by eda.ipynb.

Short gaps use retrospective averaging. Intraday indicators reset at remaining gaps.
Daily GARCH forecasts use earlier daily closes only. See FEATURES.md.
"""
from pathlib import Path
import warnings

import numpy as np
import pandas as pd
from arch import arch_model

EXCLUDED_STOCKS = {"IRFC", "HDFCLIFE", "SBILIFE"}
RAW = ["open", "high", "low", "close", "volume"]
MISSING_POLICY = "Fill 1-3 absent bars with the mean of equally many observed bars on each side."
CALENDAR_PATH = Path("eda/eda_outputs/expected_market_minutes.csv")
PROCESSED_PATH = Path("datasets/processed_datasets")


def fill_short_gaps(frame):
    """Fill an n-minute gap with one OHLCV mean from n valid bars on each side."""
    missing = frame.is_missing_bar.to_numpy()
    observed_valid = frame.is_valid_bar.to_numpy(copy=True) & ~missing
    values = frame[RAW].to_numpy(dtype=float, copy=True)
    filled = np.zeros(len(frame), dtype=bool)
    boundaries = np.flatnonzero(np.diff(np.r_[False, missing, False]))
    for start, stop in boundaries.reshape(-1, 2):
        length = stop - start
        left, right = start - length, stop + length
        if length > 3 or left < 0 or right > len(frame):
            continue
        if not (observed_valid[left:start].all() and observed_valid[stop:right].all()):
            continue
        timestamps = frame.index[left:right]
        if timestamps[0].date() != timestamps[-1].date():
            continue
        if not (timestamps[1:] - timestamps[:-1] == pd.Timedelta(minutes=1)).all():
            continue
        values[start:stop] = np.concatenate((values[left:start], values[stop:right])).mean(axis=0)
        filled[start:stop] = True
    frame[RAW] = values
    frame["is_imputed_bar"] = filled
    frame["is_valid_bar"] = observed_valid | filled
    return frame


def align_stock(path, expected_index):
    raw = pd.read_csv(path, parse_dates=["date"]).set_index("date").sort_index()
    if raw.index.hasnans:
        raise ValueError(f"Invalid timestamps: {path}")
    if raw.index.has_duplicates:
        raise ValueError(f"Resolve duplicate timestamps before transformation: {path}")
    for col in RAW:
        raw[col] = pd.to_numeric(raw[col], errors="raise")
    # Retain complete first/last session grids, but do not invent pre-history.
    grid = expected_index[(expected_index.normalize() >= raw.index.min().normalize()) &
                          (expected_index.normalize() <= raw.index.max().normalize())]
    frame = raw[RAW].reindex(grid)
    frame.index.name = "date"
    frame["is_missing_bar"] = ~grid.isin(raw.index)
    prices = frame[RAW[:4]]
    frame["is_valid_bar"] = (
        np.isfinite(frame[RAW]).all(axis=1) & prices.gt(0).all(axis=1)
        & frame.volume.ge(0) & frame.high.ge(prices.max(axis=1))
        & frame.low.le(prices.min(axis=1)))
    audit = dict(raw_rows=len(raw), aligned_rows=len(frame),
                 excluded_timestamp_rows=int((~raw.index.isin(grid)).sum()),
                 missing_bars=int(frame.is_missing_bar.sum()),
                 invalid_observed_bars=int((~frame.is_valid_bar & ~frame.is_missing_bar).sum()))
    return frame, audit


def daily_garch(frame, min_obs=60, window=252):
    """Forecast each session's daily log-return sigma using only previous sessions.

    Monthly GARCH(1,1), zero-mean Gaussian fits on a contiguous historical suffix.
    Fit failures and missing endpoints remain NaN. Units: decimal daily return.
    """
    # Require the scheduled final bar; groupby.last would silently skip NaNs.
    closes = frame.close.where(frame.is_valid_bar).groupby(frame.index.normalize()).agg(lambda x: x.iloc[-1])
    returns = np.log(closes).diff() * 100
    forecast = pd.Series(np.nan, index=closes.index, dtype=float)
    fit_records = []
    params = None
    variance = None
    last_attempt = None
    for i, day in enumerate(closes.index):
        month = (day.year, day.month)
        if i == 0:
            continue
        previous_return = returns.iloc[i - 1]
        if not np.isfinite(previous_return):
            params = variance = None
        if month != last_attempt or (params is None and last_attempt is None):
            last_attempt = month
            history = returns.iloc[max(0, i-window):i]
            bad = np.flatnonzero(~np.isfinite(history.to_numpy()))
            if len(bad):
                history = history.iloc[bad[-1]+1:]
            if len(history) >= min_obs and history.std() > 1e-8:
                record = dict(forecast_start=str(day), fit_end=str(history.index[-1]), observations=len(history))
                try:
                    with warnings.catch_warnings(record=True) as caught:
                        fit = arch_model(history, mean="Zero", vol="GARCH", p=1, q=1,
                                         dist="normal", rescale=False).fit(disp="off", show_warning=False)
                    record["warnings"] = " | ".join(str(w.message) for w in caught)
                    if fit.convergence_flag != 0:
                        raise ValueError(f"GARCH convergence flag {fit.convergence_flag}")
                    params = fit.params
                    variance = float(fit.forecast(horizon=1, reindex=False).variance.iloc[-1, 0])
                    record["status"] = "ok"
                    record.update({k: float(v) for k, v in params.items()})
                except (ValueError, RuntimeError, FloatingPointError) as error:
                    params = variance = None
                    record.update(status="failed", error=str(error))
                fit_records.append(record)
                if variance is not None:
                    forecast.iloc[i] = np.sqrt(variance) / 100
                continue
        if params is not None and variance is not None and np.isfinite(previous_return):
            variance = float(params["omega"] + params["alpha[1]"] * previous_return**2 + params["beta[1]"] * variance)
            if variance > 0 and np.isfinite(variance):
                forecast.iloc[i] = np.sqrt(variance) / 100
            else:
                params = variance = None
    return pd.Series(forecast.reindex(frame.index.normalize()).to_numpy(), index=frame.index), fit_records


def rolling(series, segment, period, method="mean"):
    grouped = series.groupby(segment).rolling(period, min_periods=period)
    return getattr(grouped, method)().reset_index(level=0, drop=True).reindex(series.index)


def exponential_mean(series, segment, period, *, alpha=None):
    decay = {"span": period} if alpha is None else {"alpha": alpha}
    return (series.groupby(segment).ewm(adjust=False, min_periods=period, **decay).mean()
            .reset_index(level=0, drop=True).reindex(series.index))


def add_returns(features, close, segment):
    ratio = close.div(close.groupby(segment).shift())
    features["returns"] = ratio - 1
    features["log_returns"] = np.log(ratio)


def add_sma(features, close, segment, periods=(5, 10, 20, 50, 100, 200, 300)):
    for period in periods:
        features[f"sma_{period}"] = rolling(close, segment, period)


def add_ema(features, close, segment, periods=(5, 10, 20, 50, 100)):
    for period in periods:
        features[f"ema_{period}"] = exponential_mean(close, segment, period)


def add_rsi(features, close, segment, periods=(10, 14, 20, 30)):
    delta = close.groupby(segment).diff()
    for period in periods:
        gain = exponential_mean(delta.clip(lower=0), segment, period, alpha=1/period)
        loss = exponential_mean(-delta.clip(upper=0), segment, period, alpha=1/period)
        rsi = 100 - 100 / (1 + gain.div(loss.replace(0, np.nan)))
        features[f"rsi_{period}"] = rsi.mask((loss == 0) & (gain > 0), 100).mask(
            (loss == 0) & (gain == 0), 50
        )


def add_bollinger(features, close, segment, periods=(20, 40, 60, 80), deviations=(1.5, 2.0, 2.5)):
    for period in periods:
        middle = rolling(close, segment, period)
        std = rolling(close, segment, period, "std")
        features[f"bb_middle_{period}"] = middle
        for multiplier in deviations:
            suffix = str(period) if multiplier == 2 else f"{period}_{multiplier:g}sd"
            distance = multiplier * std
            lower = middle - distance
            width = 2 * distance
            features[f"bb_upper_{suffix}"] = middle + distance
            features[f"bb_lower_{suffix}"] = lower
            features[f"bb_width_{suffix}"] = width.div(middle)
            features[f"bb_position_{suffix}"] = (close - lower).div(width.replace(0, np.nan))


def add_donchian(features, high, low, segment, periods=(20, 50, 100, 150)):
    # Breakout thresholds use the preceding bars within the same segment.
    prior_high = high.groupby(segment).shift()
    prior_low = low.groupby(segment).shift()
    for period in periods:
        features[f"donchian_upper_{period}"] = rolling(prior_high, segment, period, "max")
        features[f"donchian_lower_{period}"] = rolling(prior_low, segment, period, "min")


def add_vwap(features, close, high, low, volume, segment, periods=(20, 40, 60, 120)):
    typical = (high + low + close) / 3
    cumulative_volume = volume.groupby(segment).cumsum().replace(0, np.nan)
    features["vwap_segment"] = (typical * volume).groupby(segment).cumsum().div(cumulative_volume)

    close_volume = close * volume
    for period in periods:
        window_volume = rolling(volume, segment, period, "sum").replace(0, np.nan)
        features[f"vwap_{period}"] = rolling(close_volume, segment, period, "sum").div(window_volume)


def add_trend(features, close, segment, fast=20, slow=50, slope_period=20):
    fast_mean = features[f"sma_{fast}"]
    slow_mean = features[f"sma_{slow}"]
    features["trend"] = np.sign(fast_mean - slow_mean)
    features["trend_strength"] = fast_mean.div(slow_mean) - 1

    age = close.groupby(segment).cumcount().astype(float)
    n = slope_period
    centered_sum = rolling(close * age, segment, n, "sum") - (
        age - (n - 1) / 2
    ) * rolling(close, segment, n, "sum")
    slope = centered_sum / (n * (n * n - 1) / 12)
    features[f"trend_slope_{n}"] = slope
    features[f"trend_slope_pct_{n}"] = slope.div(features[f"sma_{n}"]) * 100


def add_macd(features, close, segment, fast=12, slow=26, signal_period=9):
    macd = exponential_mean(close, segment, fast) - exponential_mean(close, segment, slow)
    signal = exponential_mean(macd, segment, signal_period)
    features["macd"] = macd
    features["macd_signal"] = signal
    features["macd_histogram"] = macd - signal


def add_rolling_volatility(features, log_returns, segment, periods=(30,)):
    for period in periods:
        features[f"rolling_volatility_{period}"] = rolling(log_returns, segment, period, "std")


def add_volume_features(features, close, volume, segment, periods=(20,)):
    for period in periods:
        mean = rolling(volume, segment, period)
        std = rolling(volume, segment, period, "std")
        features[f"volume_sma_{period}"] = mean
        features[f"relative_volume_{period}"] = volume.div(mean.replace(0, np.nan))
        features[f"volume_zscore_{period}"] = (volume - mean).div(std.replace(0, np.nan))
    features["volume_change"] = volume.div(volume.groupby(segment).shift().replace(0, np.nan)) - 1
    features["log_volume"] = np.log1p(volume)
    features["dollar_volume"] = close * volume


def add_obv(features, close, volume, segment):
    direction = np.sign(close.groupby(segment).diff()).fillna(0)
    features["obv_segment"] = (direction * volume).groupby(segment).cumsum()


def add_features(frame, include_garch=True):
    out = frame.copy()
    valid = out.is_valid_bar
    timestamps = out.index.to_series()
    boundary = (
        ~valid | ~valid.shift(fill_value=False)
        | timestamps.diff().ne(pd.Timedelta(minutes=1))
        | timestamps.dt.normalize().ne(timestamps.shift().dt.normalize())
    )
    segment = boundary.cumsum()
    out["segment_id"] = segment.to_numpy()
    close, high, low, volume = (out[column].where(valid) for column in ("close", "high", "low", "volume"))

    features = {}
    add_returns(features, close, segment)
    add_sma(features, close, segment)
    add_ema(features, close, segment)
    add_rsi(features, close, segment)
    add_bollinger(features, close, segment)
    add_donchian(features, high, low, segment)
    add_vwap(features, close, high, low, volume, segment)
    add_trend(features, close, segment)
    add_macd(features, close, segment)
    add_rolling_volatility(features, features["log_returns"], segment)
    add_volume_features(features, close, volume, segment)
    add_obv(features, close, volume, segment)

    records = []
    if include_garch:
        features["garch_daily_volatility"], records = daily_garch(out)
    feature_frame = pd.DataFrame(features, index=out.index).where(valid, axis=0)
    return pd.concat([out, feature_frame], axis=1), list(features), records


def validate_features(frame, features):
    if not frame.index.is_unique or not frame.index.is_monotonic_increasing:
        raise AssertionError("Feature timestamps must be unique and sorted")
    if np.isinf(frame[RAW+features].to_numpy()).any():
        raise AssertionError("Infinite raw or feature values")
    imputed = frame.get("is_imputed_bar", False)
    if not frame.loc[frame.is_missing_bar & ~np.asarray(imputed),RAW].isna().all().all():
        raise AssertionError("Missing raw bars were filled")
    if not frame.loc[~frame.is_valid_bar,features].isna().all().all():
        raise AssertionError("Features populated on missing/invalid bars")
    if not frame.rsi_14.dropna().between(0,100).all():
        raise AssertionError("RSI outside [0,100]")
    complete = frame[["bb_lower_20","bb_middle_20","bb_upper_20"]].dropna()
    if not ((complete.bb_lower_20 <= complete.bb_middle_20) & (complete.bb_middle_20 <= complete.bb_upper_20)).all():
        raise AssertionError("Bollinger ordering failure")
    if not frame.trend.dropna().isin([-1,0,1]).all():
        raise AssertionError("Invalid trend values")
    for col in ["rolling_volatility_30","garch_daily_volatility"]:
        if col in frame and not frame[col].dropna().ge(0).all():
            raise AssertionError(f"Negative volatility: {col}")
    if not np.allclose(frame.returns.dropna().to_numpy(), np.expm1(frame.log_returns.dropna().to_numpy()),rtol=1e-7,atol=1e-12):
        raise AssertionError("Return/log-return mismatch")
    # Initial segment rows must not contain a fabricated cross-gap return.
    starts = ~frame.segment_id.duplicated()
    if frame.loc[starts,["returns","log_returns"]].notna().any().any():
        raise AssertionError("Returns crossed a gap/session boundary")
    return pd.DataFrame({"feature":features,
        "nan_count":[int(frame[c].isna().sum()) for c in features],
        "finite_count":[int(frame[c].notna().sum()) for c in features],
        "nan_on_valid_bars":[int((frame[c].isna() & frame.is_valid_bar).sum()) for c in features]})


def _transform_one(args):
    path, grid_path, output = map(Path,args)
    grid = pd.DatetimeIndex(pd.read_csv(grid_path,parse_dates=["date"]).date)
    aligned,audit = align_stock(path,grid)
    fill_short_gaps(aligned)
    audit["imputed_bars"] = int(aligned.is_imputed_bar.sum())
    audit["remaining_missing_bars"] = audit["missing_bars"] - audit["imputed_bars"]
    transformed,features,records = add_features(aligned)
    report = validate_features(transformed,features)
    report.insert(0,"stock",path.stem)
    destination = output/f"{path.stem}.parquet"
    transformed.to_parquet(destination,engine="pyarrow",compression="zstd")
    restored = pd.read_parquet(destination)
    pd.testing.assert_frame_equal(transformed,restored,check_freq=False)
    summary = dict(stock=path.stem,**audit,feature_count=len(features),
        complete_feature_rows=int(transformed[features].notna().all(axis=1).sum()),
        garch_populated_rows=int(transformed.garch_daily_volatility.notna().sum()),
        output_bytes=destination.stat().st_size,validation="passed")
    return summary,report,[dict(stock=path.stem,**record) for record in records]


def write_feature_manifest(base_dir, summary):
    """Record calendar/code fingerprints and source file metadata for this run."""
    import hashlib
    import importlib.metadata
    import json
    base_dir = Path(base_dir)
    sources = [base_dir/"datasets/NIFTY50"/f"{stock}.csv" for stock in summary.stock]
    manifest = dict(
        stocks=summary.stock.tolist(), excluded_stocks=sorted(EXCLUDED_STOCKS),
        calendar_sha256=hashlib.sha256((base_dir / CALENDAR_PATH).read_bytes()).hexdigest(),
        feature_code_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        missing_policy=MISSING_POLICY,
        indicator_reset="session boundary, timestamp gap, missing or invalid bar",
        feature_availability="Retrospective. Filled bars and dependent features may use later observations.",
        garch=dict(frequency="daily", model="zero-mean Gaussian GARCH(1,1)",
                   refit="monthly", minimum_returns=60, maximum_returns=252,
                   units="decimal daily return standard deviation"),
        versions={name:importlib.metadata.version(name) for name in ["numpy","pandas","arch","pyarrow"]},
        source_files=[dict(path=str(p.resolve()),bytes=p.stat().st_size,mtime_ns=p.stat().st_mtime_ns) for p in sources])
    (base_dir / PROCESSED_PATH / "feature_manifest.json").write_text(json.dumps(manifest,indent=2),encoding="utf-8")


def transform_all(base_dir, stocks=None, workers=1):
    """Transform eligible stocks; workers>1 is intended for a guarded CLI script."""
    from concurrent.futures import ProcessPoolExecutor
    base_dir = Path(base_dir)
    output = base_dir / PROCESSED_PATH / "NIFTY50"
    output.mkdir(parents=True,exist_ok=True)
    paths = [p for p in sorted((base_dir/"datasets/NIFTY50").glob("*.csv"))
             if p.stem not in EXCLUDED_STOCKS and (stocks is None or p.stem in stocks)]
    if not paths:
        raise ValueError("No eligible input stocks")
    tasks = [(p, base_dir / CALENDAR_PATH, output) for p in paths]
    executor = ProcessPoolExecutor(max_workers=workers) if workers>1 else None
    results = executor.map(_transform_one,tasks) if executor else map(_transform_one,tasks)
    summaries,quality,fits = [],[],[]
    try:
        for number,(summary,report,records) in enumerate(results,1):
            summaries.append(summary);quality.append(report);fits.extend(records)
            print(f"[{number}/{len(paths)}] {summary['stock']}: {summary['aligned_rows']:,} rows, {summary['missing_bars']:,} missing bars; validated",flush=True)
    finally:
        if executor:
            executor.shutdown(wait=True,cancel_futures=True)
    summary = pd.DataFrame(summaries)
    summary.to_csv(output.parent/"feature_validation.csv",index=False)
    pd.concat(quality,ignore_index=True).to_csv(output.parent/"feature_nan_report.csv",index=False)
    pd.DataFrame(fits).to_csv(output.parent/"garch_fit_audit.csv",index=False)
    write_feature_manifest(base_dir,summary)
    return summary


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers",type=int,default=1)
    parser.add_argument("--csv", type=Path, help="Process one OHLCV CSV file.")
    args = parser.parse_args()
    base_dir = Path(__file__).resolve().parents[1]
    if args.csv is None:
        transform_all(base_dir, workers=args.workers)
    else:
        if not args.csv.is_file() or args.csv.suffix.lower() != ".csv":
            parser.error(f"Expected an existing CSV file: {args.csv}")
        output = base_dir / PROCESSED_PATH / "NIFTY50"
        output.mkdir(parents=True, exist_ok=True)
        summary, _, _ = _transform_one((args.csv, base_dir / CALENDAR_PATH, output))
        print(f"Wrote {output / (args.csv.stem + '.parquet')} with {summary['feature_count']} features.")

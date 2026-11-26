"""Causal observations from saved walk-forward strategies."""
import json
import numpy as np
import pandas as pd
from markets import MARKETS, local_index
from GA_STRATEGY.main import load_data
from GA_STRATEGY.engine.ga_optimizer import load_config, _noop_finalize
from GA_STRATEGY.engine.gp_optimizer import run_rule
from GA_STRATEGY.engine import gp_optimizer, execution
from GA_STRATEGY.strategies import core_strategies


def parse_parameters(value, context):
    """CSV nulls represent a strategy with no valid fitted candidate."""
    if pd.isna(value) or (isinstance(value, str) and not value.strip()):
        return None
    try:
        params = json.loads(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"Invalid parameters for {context}: expected JSON object or null") from error
    if params is not None and not isinstance(params, dict):
        raise ValueError(f"Invalid parameters for {context}: expected JSON object or null")
    return params


def prepare(data_dir, results, stocks, *, market="india", source_timezone=None, alignment="intersection", start=None):
    rows = pd.read_csv(results)
    config = load_config()
    names = [*config, "GP"]
    frames, features = [], []
    for stock in stocks:
        frame = load_data(data_dir / f"{stock}.parquet")
        frame.index = local_index(frame.index, market, source_timezone)
        settings = MARKETS[market]
        minutes = frame.index.hour * 60 + frame.index.minute
        frame = frame.loc[(minutes >= settings["start"]) & (minutes <= settings["end"])]
        signals = pd.DataFrame(np.nan, index=frame.index, columns=names)
        selected = rows[(rows.stock == stock) & rows.strategy.isin(names)]
        if selected.duplicated(["month", "strategy"]).any():
            raise ValueError(f"Duplicate stock-month-strategy rows: {stock}")
        for row in selected.itertuples():
            month = pd.Period(row.month, freq="M")
            fit_end = pd.Timestamp(row.fit_end)
            if not pd.isna(fit_end):
                fit_end = local_index(pd.DatetimeIndex([fit_end]), market, source_timezone)[0]
            if pd.isna(fit_end) or fit_end >= month.start_time:
                raise ValueError("Strategy inputs must be fitted before their evaluation month")
            bars = frame.loc[frame.index.to_period("M") == month]
            if bars.empty:
                continue
            params = parse_parameters(row.parameters, f"{stock} {row.month} {row.strategy}")
            signal = pd.Series(0., index=bars.index)
            if params is not None:
                # Legacy caches omit frame ownership/full timestamps. Different
                # stock-months can collide; scope their reuse to this call.
                gp_optimizer._GP_CACHE.clear()
                execution._INDEX_CACHE.clear()
                fn = run_rule if row.strategy == "GP" else getattr(core_strategies, config[row.strategy]["function"])
                _, signal = fn(bars, params, 0., _noop_finalize)
            signals.loc[bars.index, row.strategy] = signal
        # Features at t use only prices observed through t. No fitted scaler.
        previous = frame.Close.shift()
        price_features = frame[["Open", "High", "Low", "Close"]].div(previous, axis=0).sub(1)
        feat = pd.concat([price_features, signals], axis=1).replace([np.inf, -np.inf], np.nan).dropna()
        frames.append(frame.loc[feat.index, ["Open", "Close"]])
        print(f"Prepared {stock}: {len(feat):,} strategy bars", flush=True)
        features.append(feat)
    common = features[0].index
    for feat in features[1:]:
        common = common.union(feat.index) if alignment == "union" else common.intersection(feat.index)
    common = common.sort_values()
    if start is not None:
        common = common[common >= pd.Timestamp(start)]
    if len(common) < 3:
        raise ValueError("Insufficient bars with strategy reports")
    availability = np.stack([common.isin(f.index) for f in features], axis=1)
    # Forward marks are for valuation only. Missing bars never permit execution.
    prices = np.stack([f.reindex(common).ffill().fillna(1.).to_numpy() for f in frames], axis=1)
    values = np.stack([f.reindex(common).fillna(0.).to_numpy() for f in features], axis=1).astype(np.float32)
    if alignment == "union":
        return common, prices, values, names, availability
    return common, prices, values, names

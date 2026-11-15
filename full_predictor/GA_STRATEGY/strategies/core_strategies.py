import numpy as np
import pandas as pd
from ..engine.risk import DEFAULT_STOP_LOSS, DEFAULT_TAKE_PROFIT, DEFAULT_LONG_ONLY, apply_risk_management

def forward_fill(signal):
    return signal.replace(0.0, np.nan).ffill().fillna(0.0)

def _apply_risk(df, signal, params):
    sl = params.get("stop_loss", DEFAULT_STOP_LOSS) if params else DEFAULT_STOP_LOSS
    tp = params.get("take_profit", DEFAULT_TAKE_PROFIT) if params else DEFAULT_TAKE_PROFIT
    lo = params.get("long_only", DEFAULT_LONG_ONLY) if params else DEFAULT_LONG_ONLY
    return apply_risk_management(df["Close"], signal, stop_loss=sl, take_profit=tp, long_only=lo)

def run_rsi(df, params, cost, finalize_strategy_fn):
    per = int(params["rsi_period"])
    os = float(params["oversold_level"])
    ob = float(params["overbought_level"])

    rsi = df[f"rsi_{per}"]
    signal = pd.Series(0.0, index=df.index)
    signal[rsi < os] = 1.0
    signal[rsi > ob] = -1.0

    signal = _apply_risk(df, forward_fill(signal), params)
    ret = df["Close"].pct_change()
    return finalize_strategy_fn(ret, signal, cost), signal

def run_bollinger(df, params, cost, finalize_strategy_fn):
    period = int(params["bb_period"])
    dev = float(params["bb_std_dev"])

    suffix = str(period) if dev == 2 else f"{period}_{dev:g}sd"
    upper, lower = df[f"bb_upper_{suffix}"], df[f"bb_lower_{suffix}"]
    signal = pd.Series(0.0, index=df.index)
    signal[df["Close"] < lower] = 1.0
    signal[df["Close"] > upper] = -1.0

    signal = _apply_risk(df, forward_fill(signal), params)
    ret = df["Close"].pct_change()
    return finalize_strategy_fn(ret, signal, cost), signal

def run_donchian(df, params, cost, finalize_strategy_fn):
    period = int(params["channel_period"])
    upper, lower = df[f"donchian_upper_{period}"], df[f"donchian_lower_{period}"]
    signal = pd.Series(0.0, index=df.index)
    signal[df["Close"] > upper] = 1.0
    signal[df["Close"] < lower] = -1.0

    signal = _apply_risk(df, forward_fill(signal), params)
    ret = df["Close"].pct_change()
    return finalize_strategy_fn(ret, signal, cost), signal

def run_vwap(df, params, cost, finalize_strategy_fn):
    vw = int(params["vwap_window"])
    zt = float(params["z_threshold"])
    trend_ma = int(params["trend_ma"])

    vwap = df[f"vwap_{vw}"]
    diff = df["Close"] - vwap
    std = (diff.groupby(df["segment_id"]).rolling(vw).std()
           .reset_index(level=0, drop=True).reindex(df.index))
    z = diff / (std + 1e-12)

    trend = df[f"sma_{trend_ma}"]

    signal = pd.Series(0.0, index=df.index)
    signal[(z < -zt) & (df["Close"] > trend)] = 1.0
    signal[(z > zt) & (df["Close"] < trend)] = -1.0

    signal = _apply_risk(df, forward_fill(signal), params)
    ret = df["Close"].pct_change()
    return finalize_strategy_fn(ret, signal, cost), signal

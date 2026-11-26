"""Shared risk management parameters and position tracking (Stop Loss and Take Profit)."""
import json
from pathlib import Path
from itertools import product

import numpy as np
import pandas as pd

# Default risk limits for intraday/short-horizon trades
# 0.0075 = 0.75% stop loss, 0.015 = 1.50% take profit
DEFAULT_STOP_LOSS = 0.0075
DEFAULT_TAKE_PROFIT = 0.015
DEFAULT_LONG_ONLY = True


def risk_candidates():
    """Shared training-only search space; TP is derived, never independently tuned."""
    path = Path(__file__).resolve().parents[1] / "config" / "risk.json"
    grid = json.loads(path.read_text(encoding="utf-8"))
    stops, ratios = grid["stop_loss"], grid["risk_reward_ratio"]
    if not stops or not ratios or any(not np.isfinite(v) or v <= 0 for v in [*stops, *ratios]):
        raise ValueError("Risk grid requires finite positive stop losses and reward/risk ratios")
    if any(sl >= 1 or sl * rr >= 1 for sl, rr in product(stops, ratios)):
        raise ValueError("SL and derived TP must be below 100%")
    return [dict(stop_loss=sl, take_profit=round(sl * rr, 12), risk_reward_ratio=rr)
            for sl, rr in product(stops, ratios)]


def apply_risk_management(close, signal, stop_loss=DEFAULT_STOP_LOSS, take_profit=DEFAULT_TAKE_PROFIT, long_only=DEFAULT_LONG_ONLY):
    """
    Apply Stop Loss (SL) and Take Profit (TP) to an intended position signal.

    Parameters
    ----------
    close : pd.Series or np.ndarray
        Asset close prices.
    signal : pd.Series or np.ndarray
        Strategy intended position (1.0 for Long, -1.0 for Short, 0.0 for Cash).
    stop_loss : float or None
        Fractional stop loss threshold (e.g. 0.0075 for 0.75%). None or <= 0 disables SL.
    take_profit : float or None
        Fractional take profit threshold (e.g. 0.015 for 1.50%). None or <= 0 disables TP.
    long_only : bool
        If True, ignore short entries (-1.0 becomes exit to cash, never short).

    Returns
    -------
    pd.Series or np.ndarray
        Effective position series reflecting SL/TP exits.
    """
    if (stop_loss is None or stop_loss <= 0) and (take_profit is None or take_profit <= 0) and not long_only:
        return signal

    is_series = isinstance(signal, pd.Series)
    index = signal.index if is_series else None

    prices = close.to_numpy(dtype=float, copy=False) if hasattr(close, "to_numpy") else np.asarray(close, dtype=float)
    targets = signal.to_numpy(dtype=float, copy=False) if hasattr(signal, "to_numpy") else np.asarray(signal, dtype=float)
    n = len(prices)
    if n == 0:
        return signal

    if long_only:
        targets = np.where(targets < 0.0, 0.0, targets)

    sl = stop_loss if (stop_loss is not None and stop_loss > 0) else None
    tp = take_profit if (take_profit is not None and take_profit > 0) else None

    if sl is None and tp is None:
        return pd.Series(targets, index=index) if is_series else targets

    # Reset entry prices and stopped state at each executed session boundary.
    cutoff = None
    if is_series and isinstance(index, pd.DatetimeIndex):
        from .execution import _get_index_timing
        session_mask, _, minutes, cutoff, _ = _get_index_timing(index)
        targets = targets.copy()
        targets[session_mask] = 0.0

    changes = np.flatnonzero(np.r_[True, targets[1:] != targets[:-1], True])
    out = np.zeros(n, dtype=np.float64)

    for k in range(len(changes) - 1):
        start = changes[k]
        end = changes[k + 1]
        t = targets[start]
        if t == 0.0 or (cutoff is not None and minutes[start] >= cutoff):
            continue
        entry_price = prices[start]
        p_seg = prices[start:end]
        if t > 0.0:
            pnl = (p_seg - entry_price) / entry_price
        else:
            pnl = (entry_price - p_seg) / entry_price

        hit_sl = (pnl <= -sl) if sl is not None else None
        hit_tp = (pnl >= tp) if tp is not None else None

        first_stop = end
        if hit_sl is not None and hit_sl.any():
            first_stop = min(first_stop, start + int(np.argmax(hit_sl)))
        if hit_tp is not None and hit_tp.any():
            first_stop = min(first_stop, start + int(np.argmax(hit_tp)))

        out[start:first_stop] = t

    return pd.Series(out, index=index) if is_series else out

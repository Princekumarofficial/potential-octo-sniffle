"""Shared risk management parameters and position tracking (Stop Loss and Take Profit)."""
import numpy as np
import pandas as pd

# Default risk limits for intraday/short-horizon trades
# 0.0075 = 0.75% stop loss, 0.015 = 1.50% take profit
DEFAULT_STOP_LOSS = 0.0075
DEFAULT_TAKE_PROFIT = 0.015
DEFAULT_LONG_ONLY = True


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

    out = np.zeros(n, dtype=np.float64)
    pos = 0.0
    entry_price = 0.0
    stopped = False
    last_target = 0.0

    for i in range(n):
        t = targets[i]
        p = prices[i]

        # In long-only mode, a negative target acts as an exit signal to cash, not a short entry
        effective_target = 0.0 if (long_only and t < 0.0) else t

        if effective_target != last_target:
            stopped = False
            if effective_target == 0.0:
                pos = 0.0
                entry_price = 0.0
            else:
                pos = effective_target
                entry_price = p
        elif pos != 0.0 and not stopped:
            pnl = (p - entry_price) / entry_price if pos > 0.0 else (entry_price - p) / entry_price
            if stop_loss is not None and stop_loss > 0 and pnl <= -stop_loss:
                pos = 0.0
                stopped = True
            elif take_profit is not None and take_profit > 0 and pnl >= take_profit:
                pos = 0.0
                stopped = True

        last_target = effective_target
        out[i] = pos

    return pd.Series(out, index=index) if is_series else out

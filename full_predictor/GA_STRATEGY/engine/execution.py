"""Single-stock, unleveraged close-fill fee scenario for optimizer scoring.

Signals execute at the observed close. All positions close at 15:15 or the
last observed session bar, whichever comes first. No circuit/liquidity model.
Edit STARTING_CAPITAL and ALLOCATION to change the common scoring assumptions.
"""
import numpy as np
import pandas as pd
from .fees import COMPONENTS, intraday_charges

STARTING_CAPITAL = 1_000_000.0
ALLOCATION = 0.12
FEE_MODEL = "FYERS Standard NSE, current tariff 2026-11-16; close fills"


def execution_returns(close, signal, *, capital=None, allocation=None):
    capital = STARTING_CAPITAL if capital is None else capital
    allocation = ALLOCATION if allocation is None else allocation
    if not np.isfinite(capital) or capital <= 0 or not 0 < allocation <= 1:
        raise ValueError("Positive capital and allocation in (0, 1] required")
    prices = close.to_numpy(dtype=float)
    if not len(prices) or not np.isfinite(prices).all() or (prices <= 0).any():
        raise ValueError("Observed positive finite close prices required")
    target = signal.reindex(close.index).fillna(0).to_numpy(dtype=float).copy()
    if not np.isin(target, [-1, 0, 1]).all():
        raise ValueError("Expected long/cash/short signals")
    dates = close.index.normalize()
    last = np.r_[dates[:-1] != dates[1:], True]
    minutes = close.index.hour * 60 + close.index.minute
    target[(minutes >= 915) | last] = 0
    # After 15:00 only exits are allowed.
    changes = np.flatnonzero(np.r_[True, target[1:] != target[:-1]] | (minutes == 900))
    equity = np.empty(len(close))
    fees = np.zeros(len(close))
    exposure = np.zeros(len(close))
    gross_pnl = np.zeros(len(close))
    breakdown = dict.fromkeys(COMPONENTS, 0.0)
    cash, quantity, previous = float(capital), 0, 0
    fills = 0
    def charge(i, qty, side):
        nonlocal fills
        parts = intraday_charges(abs(qty) * prices[i], side)
        fees[i] += parts["fee"]
        for key in COMPONENTS:
            breakdown[key] += parts[key]
        fills += 1
        return parts["fee"]
    for i in changes:
        if i > previous:
            equity[previous:i] = cash + quantity * prices[previous:i]
            exposure[previous:i] = np.sign(quantity)
        desired = int(target[i])
        if desired != np.sign(quantity):
            if quantity:
                cash += quantity * prices[i] - charge(i, quantity, "SELL" if quantity > 0 else "BUY")
                quantity = 0
            if desired and minutes[i] < 900 and cash > 0:
                qty = int(cash * allocation / prices[i])
                # Reserve the entry charge even for fully allocated positions.
                while qty and qty * prices[i] + intraday_charges(qty * prices[i], "BUY" if desired > 0 else "SELL")["fee"] > cash:
                    qty -= 1
                if qty:
                    quantity = desired * qty
                    cash -= quantity * prices[i] + charge(i, qty, "BUY" if desired > 0 else "SELL")
        previous = i
    equity[previous:] = cash + quantity * prices[previous:]
    exposure[previous:] = np.sign(quantity)
    opening = np.r_[capital, equity[:-1]]
    if (equity <= 0).any():
        raise ValueError("Scoring account exhausted")
    gross_pnl[:] = equity - opening + fees
    held = np.r_[0, exposure[:-1]]
    result = pd.Series(equity / opening - 1, index=close.index)
    result.attrs["execution"] = dict(
        fees=float(fees.sum()), fills=fills, initial_capital=capital,
        final_equity=float(equity[-1]), allocation=allocation, fee_model=FEE_MODEL,
        cost_drag=float(fees.sum() / capital),
        position_changes=fills,
        long_pnl=float(gross_pnl[held > 0].sum() / capital),
        short_pnl=float(gross_pnl[held < 0].sum() / capital),
        directional_hit_rate=float((gross_pnl[held != 0] > 0).mean()) if (held != 0).any() else None,
        long_exposure=float((exposure > 0).mean()), short_exposure=float((exposure < 0).mean()),
        **{"charge_" + key: round(value, 2) for key, value in breakdown.items()})
    return result

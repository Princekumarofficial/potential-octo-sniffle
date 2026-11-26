"""Single-stock, unleveraged close-fill fee scenario for optimizer scoring.

Signals execute at the observed close. All positions close at 15:15 or the
last observed session bar, whichever comes first. No circuit/liquidity model.
Edit STARTING_CAPITAL and ALLOCATION to change the common scoring assumptions.
"""
import numpy as np
import pandas as pd
from .fees import COMPONENTS, intraday_charges, fast_intraday_fee, us_intraday_charges, fast_us_fee

STARTING_CAPITAL = 1_000_000.0
ALLOCATION = 0.12
FEE_MODEL = "FYERS Standard NSE, current tariff 2026-11-16; close fills"
FEE_MODEL_US = "US Regulated Equities ($0 Commission, SEC 31 + FINRA TAF on sell)"

_INDEX_CACHE = {}


def _get_index_timing(idx):
    if not len(idx):
        return np.array([], dtype=bool), np.array([], dtype=bool), np.array([], dtype=int), 900, False
    key = (len(idx), int(idx[0].value), int(idx[-1].value))
    val = _INDEX_CACHE.get(key)
    if val is None:
        if len(_INDEX_CACHE) > 32:
            _INDEX_CACHE.clear()
        dates = idx.normalize()
        last = np.r_[dates[:-1] != dates[1:], True]
        minutes = idx.hour * 60 + idx.minute
        if len(minutes) and minutes.max() > 930:
            # US market (09:30 - 16:00 EST): square-off at 15:55 (955), entry cutoff 15:45 (945)
            session_mask = (minutes >= 955) | last
            min_cutoff = (minutes == 945)
            cutoff_minute = 945
            is_us = True
        else:
            # Indian market (09:15 - 15:30 IST): square-off at 15:15 (915), entry cutoff 15:00 (900)
            session_mask = (minutes >= 915) | last
            min_cutoff = (minutes == 900)
            cutoff_minute = 900
            is_us = False
        val = (session_mask, min_cutoff, minutes, cutoff_minute, is_us)
        _INDEX_CACHE[key] = val
    return val


def execution_returns(close, signal, *, capital=None, allocation=None, leverage=1.0, report=True):
    capital = STARTING_CAPITAL if capital is None else capital
    allocation = ALLOCATION if allocation is None else allocation
    if not np.isfinite(capital) or capital <= 0 or not 0 < allocation <= 1:
        raise ValueError("Positive capital and allocation in (0, 1] required")
    if not 1.0 <= float(leverage) <= 5.0:
        raise ValueError("Leverage must be between 1.0 and 5.0")
    prices = close.to_numpy(dtype=float, copy=False) if hasattr(close, "to_numpy") else np.asarray(close, dtype=float)
    if not len(prices) or not np.isfinite(prices).all() or (prices <= 0).any():
        raise ValueError("Observed positive finite close prices required")
    
    if isinstance(signal, pd.Series) and signal.index.equals(close.index):
        target = signal.to_numpy(dtype=float, copy=True)
    elif hasattr(signal, "reindex"):
        target = signal.reindex(close.index).fillna(0).to_numpy(dtype=float)
    else:
        target = np.asarray(signal, dtype=float).copy()

    if not np.isin(target, [-1, 0, 1]).all():
        raise ValueError("Expected long/cash/short signals")

    session_mask, min_cutoff, minutes, cutoff_minute, is_us = _get_index_timing(close.index)
    active_fee_model = FEE_MODEL_US if is_us else FEE_MODEL
    target[session_mask] = 0
    # After cutoff only exits are allowed.
    changes = np.flatnonzero(np.r_[True, target[1:] != target[:-1]] | min_cutoff)
    equity = np.empty(len(close))
    cash, quantity, previous = float(capital), 0, 0

    if not report:
        for i in changes:
            if i > previous:
                if quantity == 0:
                    equity[previous:i] = cash
                else:
                    equity[previous:i] = cash + quantity * prices[previous:i]
            desired = int(target[i])
            cur_sign = 1 if quantity > 0 else (-1 if quantity < 0 else 0)
            if desired != cur_sign:
                p_i = prices[i]
                if quantity:
                    side = "SELL" if quantity > 0 else "BUY"
                    fee = fast_us_fee(abs(quantity) * p_i, side, quantity) if is_us else fast_intraday_fee(abs(quantity) * p_i, side)
                    cash += quantity * p_i - fee
                    quantity = 0
                if desired and minutes[i] < cutoff_minute and cash > 0:
                    qty = int(cash * allocation * leverage / p_i)
                    margin = (qty * p_i) / leverage
                    side = "BUY" if desired > 0 else "SELL"
                    if qty and (margin + 100.0 > cash):
                        while qty and ((qty * p_i) / leverage + (fast_us_fee(qty * p_i, side, qty) if is_us else fast_intraday_fee(qty * p_i, side)) > cash):
                            qty -= 1
                    if qty:
                        fee = fast_us_fee(qty * p_i, side, qty) if is_us else fast_intraday_fee(qty * p_i, side)
                        quantity = desired * qty
                        cash -= quantity * p_i + fee
            previous = i
        if quantity == 0:
            equity[previous:] = cash
        else:
            equity[previous:] = cash + quantity * prices[previous:]
        opening = np.r_[capital, equity[:-1]]
        if (equity <= 0).any():
            raise ValueError("Scoring account exhausted")
        return pd.Series(equity / opening - 1, index=close.index)

    fees = np.zeros(len(close))
    exposure = np.zeros(len(close))
    gross_pnl = np.zeros(len(close))
    breakdown = dict.fromkeys(COMPONENTS, 0.0)
    fills = 0

    def charge(i, qty, side):
        nonlocal fills
        if is_us:
            parts = us_intraday_charges(abs(qty) * prices[i], side, abs(qty))
        else:
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
        cur_sign = 1 if quantity > 0 else (-1 if quantity < 0 else 0)
        if desired != cur_sign:
            p_i = prices[i]
            if quantity:
                cash += quantity * p_i - charge(i, quantity, "SELL" if quantity > 0 else "BUY")
                quantity = 0
            if desired and minutes[i] < cutoff_minute and cash > 0:
                qty = int(cash * allocation * leverage / p_i)
                margin = (qty * p_i) / leverage
                side = "BUY" if desired > 0 else "SELL"
                if qty and (margin + 100.0 > cash):
                    while qty and ((qty * p_i) / leverage + (fast_us_fee(qty * p_i, side, qty) if is_us else fast_intraday_fee(qty * p_i, side)) > cash):
                        qty -= 1
                if qty:
                    quantity = desired * qty
                    cash -= quantity * p_i + charge(i, qty, side)
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
        final_equity=float(equity[-1]), allocation=allocation, leverage=float(leverage), fee_model=active_fee_model,
        cost_drag=float(fees.sum() / capital),
        position_changes=fills,
        long_pnl=float(gross_pnl[held > 0].sum() / capital),
        short_pnl=float(gross_pnl[held < 0].sum() / capital),
        directional_hit_rate=float((gross_pnl[held != 0] > 0).mean()) if (held != 0).any() else None,
        long_exposure=float((exposure > 0).mean()), short_exposure=float((exposure < 0).mean()),
        **{"charge_" + key: round(value, 2) for key, value in breakdown.items()})
    return result

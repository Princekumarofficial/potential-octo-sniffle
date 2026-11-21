"""Replay saved monthly strategies into a cash-funded top-ten stock portfolio."""
import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

from .main import DEFAULT_DATA_PATH, load_data
from .engine.ga_optimizer import load_config
from .engine.gp_optimizer import run_rule
from .engine.backtester import DEFAULT_COST, finalize_strategy
from .strategies import core_strategies
from .engine.fees import COMPONENTS, intraday_charges, affordable_shares

ENTRY_CUTOFF_MINUTE = 15 * 60
SQUARE_OFF_MINUTE = 15 * 60 + 15


def rank_months(selected, top_n=10):
    if selected.duplicated(["stock", "month"]).any():
        raise ValueError("Duplicate stock-month selections in the input.")
    returns = selected.pivot(index="month", columns="stock", values="net_return")
    calendar = pd.period_range(returns.index.min(), returns.index.max(), freq="M")
    returns = returns.reindex(calendar)
    records = []
    eligible = {}
    for i, month in enumerate(calendar):
        if i < 3:
            continue
        history = returns.iloc[i-3:i]
        scores = ((1 + history).prod() - 1).where(history.notna().all())
        scores = scores.dropna().sort_index().sort_values(ascending=False, kind="stable")
        eligible[month] = set(scores.head(top_n).index)
        for rank, (stock, score) in enumerate(scores.items(), 1):
            records.append(dict(month=str(month), stock=stock, rank=rank,
                                trailing_return=score, eligible=rank <= top_n,
                                history_start=str(calendar[i-3]), history_end=str(calendar[i-1])))
    return eligible, pd.DataFrame(records)


def prepare(selected, data_dir, eligible):
    config = load_config()
    market = {}
    events = defaultdict(list)
    candidates = []
    for stock, rows in selected.groupby("stock", sort=True):
        frame = load_data(data_dir / f"{stock}.parquet")
        frame_months = frame.index.to_period("M")
        parts = []
        for row in rows.sort_values("month").itertuples():
            bars = frame.loc[frame_months == row.month]
            if bars.empty:
                raise ValueError(f"No observed bars for {stock} {row.month}")
            params = json.loads(row.parameters)
            signal = pd.Series(0., index=bars.index)
            if params is not None:
                if row.selected_strategy == "GP":
                    _, signal = run_rule(bars, params["tree"], DEFAULT_COST, finalize_strategy)
                elif row.selected_strategy != "Cash":
                    params = {**params, "long_only": True}
                    fn = getattr(core_strategies, config[row.selected_strategy]["function"])
                    _, signal = fn(bars, params, DEFAULT_COST, finalize_strategy)
            if (signal < 0).any():
                raise ValueError("Portfolio replay expects long-only signals.")
            parts.append(pd.DataFrame({
                "Open": bars.Open, "High": bars.High, "Low": bars.Low, "Close": bars.Close,
                "Volume": bars.Volume if "Volume" in bars else 0, "target": signal
            }))
            candidates.append(dict(stock=stock, month=str(row.month),
                                   target_changes=int(signal.diff().fillna(signal).ne(0).sum()),
                                   eligible=stock in eligible.get(row.month, set())))
        bars = pd.concat(parts).sort_index()
        dates = bars.index.as_unit("ns")

        # Detect circuit limits from daily change against previous day close and flat bar range
        day_close = bars["Close"].groupby(bars.index.date).last()
        prev_close_map = day_close.shift(1).to_dict()
        prev_closes = np.array([prev_close_map.get(d, np.nan) for d in bars.index.date])
        day_ret = (bars["Open"].to_numpy(float) - prev_closes) / prev_closes
        is_flat = (bars["High"] == bars["Low"]).to_numpy()
        roll_range = ((bars["High"].rolling(3).max() - bars["Low"].rolling(3).min()) / bars["Close"]).fillna(0).to_numpy()

        is_uc = (day_ret >= 0.048) & (is_flat | (roll_range < 0.001))
        is_lc = (day_ret <= -0.048) & (is_flat | (roll_range < 0.001))

        # Targets become executable only at the next observed bar's open.
        # Reset signal execution at each session. Never carry yesterday's target.
        target = bars.target.groupby(bars.index.normalize()).shift(fill_value=0).to_numpy()
        allowed_months = [month for month, stocks in eligible.items() if stock in stocks]
        allowed = dates.to_period("M").isin(allowed_months)
        minutes = dates.hour * 60 + dates.minute
        # User-configured session exits, before broker administrative square-off.
        target = np.where(allowed & (minutes < SQUARE_OFF_MINUTE), target, 0)
        changes = np.flatnonzero(np.diff(np.r_[0, target]) != 0)
        opens = bars.Open.to_numpy(float)
        closes = bars.Close.to_numpy(float)
        timestamps = dates.asi8
        market[stock] = (timestamps, opens, closes, is_uc, is_lc)
        # Add scheduled exits even if the target was already flat.
        scheduled = np.flatnonzero(minutes >= SQUARE_OFF_MINUTE)
        # Pending lower-circuit exits may retry at observed unlocked bars.
        unlocks = np.flatnonzero(np.r_[False, is_lc[:-1] & ~is_lc[1:]])
        session_starts = np.flatnonzero(~dates.normalize().duplicated())
        changes = np.unique(np.r_[changes, scheduled, unlocks, session_starts])
        for index in changes:
            events[timestamps[index]].append((stock, int(target[index]), opens[index], bool(is_uc[index]), bool(is_lc[index])))
        print(f"Prepared {stock}: {len(bars):,} bars (UC bars: {is_uc.sum():,}, LC bars: {is_lc.sum():,})", flush=True)
    return market, events, pd.DataFrame(candidates)


def simulate(market, events, eligible, capital=1_000_000., allocation=.04, cost=None, *, leverage=1., allocation_mode="notional", leverage_by_stock=None):
    if capital <= 0 or not 0 < allocation <= 1 or (cost is not None and cost < 0):
        raise ValueError("Invalid capital, allocation, or transaction cost.")
    if not 1 <= leverage <= 5 or allocation_mode not in ("notional", "margin"):
        raise ValueError("Leverage must be between 1 and 5 and allocation mode notional or margin.")
    leverage_by_stock = leverage_by_stock or {}
    if any(not 1 <= float(limit) <= 5 for limit in leverage_by_stock.values()):
        raise ValueError("Stock leverage limits must be between 1 and 5.")
    cash = capital
    holdings = {}
    financing = {}
    margins = {}
    margin_events = []
    fills = []
    snapshots = []
    first_month = min(eligible)
    # Include daily marks even when no orders execute.
    days = set()
    for item in market.values():
        times = item[0]
        dates = pd.to_datetime(times).normalize().unique()
        days.update(dates[dates >= first_month.start_time].as_unit("ns").asi8)
    days = sorted(days)
    marks = {day + pd.Timedelta(hours=23, minutes=59).value for day in days}
    timeline = sorted(set(events) | marks)
    if leverage > 1 or any(float(limit) > 1 for limit in leverage_by_stock.values()):
        timeline = sorted(set(timeline).union(*(item[0] for item in market.values())))
    trapped_exits = set()
    entries = {}
    converted = set()
    overnight_records = []

    def value(timestamp, at_close=False):
        total = cash
        for stock, quantity in holdings.items():
            times, opens, closes = market[stock][:3]
            index = np.searchsorted(times, timestamp, side="right") - 1
            # Only the open is known at an execution timestamp.
            price = opens[index] if times[index] == timestamp and not at_close else closes[index]
            total += quantity * price
        return total - sum(financing.values())

    def charges(notional, side):
        if cost is not None:
            return {**dict.fromkeys(COMPONENTS, 0.), "fee": notional * cost}
        return intraday_charges(notional, side)

    def sell(stock, price, timestamp, note):
        nonlocal cash
        quantity = holdings.pop(stock)
        detail = (intraday_charges(quantity * price, "SELL", delivery=True)
                  if stock in converted and cost is None else charges(quantity * price, "SELL"))
        converted.discard(stock)
        entries.pop(stock, None)
        cash += quantity * price - detail["fee"] - financing.pop(stock, 0.)
        margins.pop(stock, None)
        trapped_exits.discard(stock)
        fills.append(dict(timestamp=pd.Timestamp(timestamp), stock=stock, side="SELL",
                          shares=quantity, price=price, cash=cash, allocation_budget=None,
                          equity_before=None, note=note, **detail))

    for timestamp in timeline:
        stamp = pd.Timestamp(timestamp)
        if stamp.to_period("M") < first_month:
            continue
        # Losses consume free margin. Mark deficits and request liquidation.
        if holdings and value(timestamp) < sum(margins.values()):
            margin_events.append(dict(timestamp=stamp, equity=value(timestamp),
                                      required_margin=sum(margins.values()), reason="MARGIN_SHORTFALL"))
            trapped_exits.update(holdings)
        orders = sorted(events.get(timestamp, []))
        scheduled_stocks = {order[0] for order in orders}
        for stock in sorted(trapped_exits - scheduled_stocks):
            item = market[stock]
            index = np.searchsorted(item[0], timestamp)
            if index < len(item[0]) and item[0][index] == timestamp:
                flags = (bool(item[3][index]), bool(item[4][index])) if len(item) > 3 else (False, False)
                orders.append((stock, 0, item[1][index], *flags))
        normalized = []
        exited = set()
        for order in orders:
            stock, target, price = order[:3]
            is_uc, is_lc = order[3:] if len(order) == 5 else (False, False)
            normalized.append((stock, target, price, is_uc, is_lc))
            if stock in trapped_exits and not is_lc and stock in holdings:
                sell(stock, price, timestamp, "PENDING_EXIT_FILLED")
                exited.add(stock)
            if (target == 0 or stamp.hour * 60 + stamp.minute >= SQUARE_OFF_MINUTE) and stock in holdings:
                if is_lc:
                    trapped_exits.add(stock)
                else:
                    note = "SCHEDULED_EXIT" if stamp.hour * 60 + stamp.minute >= SQUARE_OFF_MINUTE else "NORMAL_SELL"
                    sell(stock, price, timestamp, note)

        equity = value(timestamp)
        budget = equity * allocation
        for stock, target, price, is_uc, is_lc in normalized:
            if target != 1 or stock in holdings or stock in exited or stamp.hour * 60 + stamp.minute >= ENTRY_CUTOFF_MINUTE:
                continue
            if is_uc or is_lc:
                continue
            stock_leverage = float(leverage_by_stock.get(stock, leverage))
            notional_budget = budget * stock_leverage if allocation_mode == "margin" else budget
            available_margin = max(0., min(cash, value(timestamp) - sum(margins.values())))
            low, high = 0, max(0, int(min(notional_budget, available_margin * stock_leverage) / price))
            while low < high:
                trial = (low + high + 1) // 2
                trial_fee = charges(trial * price, "BUY")["fee"]
                if trial * price / stock_leverage + trial_fee <= available_margin:
                    low = trial
                else:
                    high = trial - 1
            quantity = low
            if quantity <= 0:
                continue
            detail = charges(quantity * price, "BUY")
            margin = quantity * price / stock_leverage
            financing[stock] = quantity * price - margin
            margins[stock] = margin
            cash -= margin + detail["fee"]
            holdings[stock] = quantity
            entries[stock] = (quantity * price, detail)
            fills.append(dict(timestamp=stamp, stock=stock, side="BUY",
                              shares=quantity, price=price, cash=cash,
                              allocation_budget=notional_budget, equity_before=equity, note="BUY",
                              leverage=stock_leverage, margin=margin, financed=financing[stock], **detail))
        if timestamp in marks:
            if holdings and cost is None:
                for stock, quantity in holdings.items():
                    overnight_records.append(dict(date=stamp.normalize(), stock=stock, shares=quantity,
                                                  reason="BLOCKED_OR_MISSING_EXIT", assumed_treatment="DELIVERY"))
                    trapped_exits.add(stock)
                    if stock not in converted:
                        # Delivery requires full funding. A debit is disclosed if cash is insufficient.
                        cash -= financing.pop(stock, 0.)
                        notional, original = entries[stock]
                        margins[stock] = notional
                        delivery = intraday_charges(notional, "BUY", delivery=True)
                        adjustment = {key: round(delivery[key] - original[key], 2) for key in COMPONENTS}
                        adjustment["fee"] = round(sum(adjustment.values()), 2)
                        cash -= adjustment["fee"]
                        converted.add(stock)
                        fills.append(dict(timestamp=stamp, stock=stock, side="FEE", shares=0, price=0.,
                                          cash=cash, allocation_budget=None, equity_before=None,
                                          note="ASSUMED_DELIVERY_CONVERSION", **adjustment))
            equity = value(timestamp, at_close=True)
            if cash < 0:
                margin_events.append(dict(timestamp=stamp, equity=equity,
                                          required_margin=sum(margins.values()), reason="DELIVERY_CASH_DEBIT"))
            snapshots.append(dict(date=pd.Timestamp(timestamp).normalize(), equity=equity,
                                  cash=cash, invested=equity-cash, positions=len(holdings),
                                  blocked_margin=sum(margins.values()), financed=sum(financing.values()),
                                  gross_exposure=equity-cash+sum(financing.values()),
                                  available_margin=max(0., min(cash, equity-sum(margins.values())))))
    curve = pd.DataFrame(snapshots)
    curve.attrs["overnight_positions"] = overnight_records
    curve.attrs["margin_events"] = margin_events
    return curve, pd.DataFrame(fills, columns=[
        "timestamp", "stock", "side", "shares", "price", "fee", "cash", "allocation_budget", "equity_before", "note", "leverage", "margin", "financed", *COMPONENTS])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, required=True, help="Batch monthly_results.csv.")
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_PATH.parent)
    parser.add_argument("--output", type=Path, default=Path("reports/portfolio"))
    parser.add_argument("--capital", type=float, default=1_000_000)
    parser.add_argument("--allocation", type=float, default=.04)
    parser.add_argument("--top", type=int, default=10)
    parser.add_argument("--leverage", type=float, default=1., help="Assumed intraday leverage, 1 to 5.")
    parser.add_argument("--allocation-mode", choices=("notional", "margin"), default="notional")
    parser.add_argument("--leverage-file", type=Path, help="JSON mapping stock symbols to leverage limits.")
    args = parser.parse_args()
    if not 1 <= args.leverage <= 5:
        parser.error("--leverage must be between 1 and 5.")
    limits = json.loads(args.leverage_file.read_text()) if args.leverage_file else {}
    if not isinstance(limits, dict) or any(
            not isinstance(limit, (int, float)) or not 1 <= limit <= 5 for limit in limits.values()):
        parser.error("--leverage-file must map stock symbols to numbers between 1 and 5.")

    if args.top < 1 or args.capital <= 0 or not 0 < args.allocation <= 1:
        parser.error("Top and capital must be positive and allocation in (0, 1].")
    selected = pd.read_csv(args.results)
    selected = selected.loc[selected.strategy.eq("Selected")].copy()
    if selected.empty:
        parser.error("No Selected rows in the batch results.")
    if "short_exposure" in selected and selected.short_exposure.gt(0).any():
        parser.error("Use long-only batch results for this cash-funded portfolio.")
    selected["month"] = pd.PeriodIndex(selected.month, freq="M")
    eligible, ranks = rank_months(selected, args.top)
    if not eligible:
        parser.error("At least four months are needed for ranking and execution.")
    market, events, candidates = prepare(selected, args.data_dir, eligible)
    curve, trades = simulate(market, events, eligible, args.capital, args.allocation,
                             leverage=args.leverage, allocation_mode=args.allocation_mode, leverage_by_stock=limits)
    if curve.empty:
        raise ValueError("No portfolio evaluation bars.")
    args.output.mkdir(parents=True, exist_ok=True)
    ranks.to_csv(args.output / "monthly_stock_ranks.csv", index=False)
    candidates.to_csv(args.output / "candidate_activity.csv", index=False)
    trades.insert(0, "order_id", range(1, len(trades) + 1))
    trades.to_csv(args.output / "executed_trades.csv", index=False)
    trades.groupby(pd.to_datetime(trades.timestamp).dt.date)[[*COMPONENTS, "fee"]].sum().to_csv(
        args.output / "daily_charges.csv", index_label="date")
    curve.to_csv(args.output / "portfolio_value.csv", index=False)
    pd.DataFrame(curve.attrs["margin_events"], columns=["timestamp", "equity", "required_margin", "reason"]).to_csv(
        args.output / "margin_events.csv", index=False)
    pd.DataFrame(curve.attrs["overnight_positions"], columns=["date", "stock", "shares", "reason", "assumed_treatment"]).to_csv(args.output / "overnight_positions.csv", index=False)
    growth = curve.equity.iloc[-1] / args.capital
    peak = curve.equity.cummax().clip(lower=args.capital)
    summary = dict(initial_capital=args.capital, final_equity=float(curve.equity.iloc[-1]),
                   cumulative_return=float(growth-1), daily_max_drawdown=float((curve.equity/peak-1).min()),
                   fees=float(trades.fee.sum()), fills=int(trades.side.isin(["BUY", "SELL"]).sum()), overnight_stock_days=len(curve.attrs["overnight_positions"]), allocation=args.allocation,
                   leverage=args.leverage, allocation_mode=args.allocation_mode, leverage_by_stock=limits,
                   margin_events=len(curve.attrs["margin_events"]), ranking_months=3, top=args.top, start=str(curve.date.iloc[0]),
                   end=str(curve.date.iloc[-1]),
                   fee_model="FYERS Standard NSE intraday, current tariff checked 2026-11-16",
                   rounding="Per-component per-order paise estimates",
                   charge_breakdown={key: float(trades[key].sum()) for key in COMPONENTS})
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    from matplotlib.figure import Figure
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    figure = Figure(figsize=(12, 5), layout="constrained")
    FigureCanvasAgg(figure)
    ax = figure.subplots()
    ax.plot(curve.date, curve.equity, label="Portfolio value")
    ax.plot(curve.date, curve.cash, label="Cash", alpha=.6)
    ax.axhline(args.capital, color="gray", linestyle="--", label="Starting capital")
    ax.set(title="Monthly top-stock portfolio", ylabel="Value (INR)")
    ax.legend()
    ax.grid(alpha=.2)
    figure.savefig(args.output / "portfolio_value.png", dpi=160)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

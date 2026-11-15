"""Monthly stock evaluation using the existing grid and GP optimizers."""
import argparse
from concurrent.futures import ProcessPoolExecutor
import contextlib
import io
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .engine.objectives import DEFAULT_OBJECTIVE, OBJECTIVES, training_score
from .main import DEFAULT_DATA_PATH, load_data
from .engine.ga_optimizer import genetic_optimize, load_config
from .engine.gp_optimizer import optimize_rules, run_rule
from .engine.backtester import DEFAULT_COST, finalize_strategy, StrategyProfitCalculator, to_daily_returns, sharpe, sortino, max_drawdown
from .strategies import core_strategies


def metrics(close, signal):
    returns = finalize_strategy(close.pct_change(), signal, DEFAULT_COST)
    report = StrategyProfitCalculator(close, signal, returns).summary()
    elapsed = (close.index[-1] - close.index[0]).total_seconds() / (365.25 * 86400)
    growth = 1 + report["cumulative_return"]
    cagr = float(np.expm1(np.log(growth) / elapsed)) if elapsed > 0 and growth > 0 else None
    report["cagr"] = cagr
    report["calmar"] = cagr / abs(report["dd"]) if cagr is not None and report["dd"] < 0 else None
    return report


def evaluation_months(index, train_months, end_month=None, recent_months=None, start_month=None):
    """Reserve the first calendar months for fitting, then test every next month."""
    last = index[-1].to_period("M")
    if end_month is not None:
        last = min(last, pd.Period(end_month, freq="M"))
    first = index[0].to_period("M") + train_months
    if recent_months is not None:
        first = max(first, last - recent_months + 1)
    if start_month is not None:
        first = max(first, pd.Period(start_month, freq='M'))
    months = pd.period_range(start=first, end=last, freq="M")
    if months.empty:
        raise ValueError("Not enough history for training followed by an evaluation month.")
    return months


def evaluate_stock(stock, frame, months, train_months, valid_months, gp_options, plot_dir=None, objective=DEFAULT_OBJECTIVE):
    config = load_config()
    names = [*config, "GP", "Selected"]
    histories = {name: [] for name in names}
    monthly = []
    for number, month in enumerate(months):
        start, end = month.start_time, (month + 1).start_time
        fit_start = (month - train_months).start_time
        train = frame.loc[(frame.index >= fit_start) & (frame.index < start)]
        test = frame.loc[(frame.index >= start) & (frame.index < end)]
        if train.empty or test.empty or frame.index[0].to_period("M") > fit_start.to_period("M"):
            raise ValueError(f"Insufficient data for {stock} in {month}")
        with contextlib.redirect_stdout(io.StringIO()):
            grid = genetic_optimize(train, train, objective=objective)
            gp = optimize_rules({"train": train}, include_tree=True, objective=objective, **gp_options)
        candidates = {**grid, "GP": gp["GP"]}
        # Compare all winners using the same unpenalized training objective.
        scores = {}
        for name, (_, params, reports) in candidates.items():
            if params is not None:
                stats = reports["train"]
                value = training_score(stats, objective)
                if value is not None:
                    scores[name] = value
        winner = max(scores, key=scores.get) if scores else "Cash"
        signals = {}
        for name, (_, params, _) in candidates.items():
            signal = pd.Series(0.0, index=test.index)
            if params is not None:
                fn = run_rule if name == "GP" else getattr(core_strategies, config[name]["function"])
                arguments = params["tree"] if name == "GP" else params
                _, signal = fn(test, arguments, DEFAULT_COST, finalize_strategy)
            signals[name] = signal
        signals["Selected"] = signals.get(winner, pd.Series(0.0, index=test.index))
        for name, signal in signals.items():
            previous = histories[name]
            # Include the preceding close and target so switches incur true turnover.
            context_close = test["Close"]
            context_signal = signal
            if previous:
                prior_close, prior_signal = previous[-1]
                context_close = pd.concat([prior_close.iloc[-1:], context_close])
                context_signal = pd.concat([prior_signal.iloc[-1:], context_signal])
            returns = finalize_strategy(context_close.pct_change(), context_signal, DEFAULT_COST).loc[test.index]
            daily = to_daily_returns(returns)
            incoming = context_signal.shift(fill_value=0).loc[test.index]
            turnover = context_signal.diff().fillna(context_signal).abs().loc[test.index]
            equity = (1 + returns).cumprod()
            opening = equity.shift(fill_value=1.0)
            period_report = {
                "net_return": float(equity.iloc[-1] - 1),
                "max_drawdown": max_drawdown(returns),
                "sharpe": sharpe(daily), "sortino": sortino(daily),
                "cost_drag": float((opening * turnover * DEFAULT_COST).sum()),
                "long_exposure": float(incoming.gt(0).mean()),
                "short_exposure": float(incoming.lt(0).mean()),
                "position_changes": int(turnover.gt(0).sum()),
                "benchmark_return": float((1 + context_close.pct_change().fillna(0).loc[test.index]).prod() - 1),
            }
            selection_name = winner if name == "Selected" else name
            params = candidates.get(selection_name, (None, None, None))[1]
            monthly.append(dict(stock=stock, month=str(month), objective=objective,
                period="validation" if number < valid_months else "test",
                strategy=name, selected_strategy=winner, selected=name == winner,
                fit_start=train.index[0], fit_end=train.index[-1],
                evaluation_start=test.index[0], evaluation_end=test.index[-1],
                training_score=scores.get(selection_name),
                parameters=json.dumps(params), **period_report))
            histories[name].append((test["Close"], signal))
        print(f"{stock} {month}: selected {winner}", flush=True)
    summaries = []
    curves = {}
    for name, periods in histories.items():
        close = pd.concat([pair[0] for pair in periods])
        signal = pd.concat([pair[1] for pair in periods])
        summaries.append(dict(stock=stock, strategy=name, objective=objective, **metrics(close, signal)))
        returns = finalize_strategy(close.pct_change(), signal, DEFAULT_COST)
        curves[name] = (1 + to_daily_returns(returns)).cumprod() - 1
    if plot_dir is not None:
        curves["Buy and hold"] = (1 + to_daily_returns(close.pct_change().fillna(0))).cumprod() - 1
        save_stock_plot(stock, pd.DataFrame(curves), months[valid_months].start_time, plot_dir)
    return monthly, summaries


def save_stock_plot(stock, curves, test_start, destination):
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure
    from matplotlib.ticker import PercentFormatter

    figure = Figure(figsize=(12, 6), layout="constrained")
    FigureCanvasAgg(figure)
    ax = figure.subplots()
    for name in curves:
        label = "Monthly selected" if name == "Selected" else name.replace("_", " ")
        ax.plot(curves.index, curves[name], label=label,
                linewidth=2.8 if name == "Selected" else 1.6,
                linestyle="--" if name == "Buy and hold" else "-")
    ax.axvline(test_start, color="0.4", linestyle=":", label="Test begins")
    ax.axhline(0, color="0.65", linewidth=0.7)
    ax.yaxis.set_major_formatter(PercentFormatter(1))
    ax.set_title(f"{stock} cumulative returns — monthly walk-forward evaluation")
    ax.set_ylabel("Compounded return after strategy costs")
    ax.grid(axis="y", alpha=0.2)
    ax.legend(fontsize=8, loc="best")
    ax.tick_params(axis="x", rotation=25)
    destination.mkdir(parents=True, exist_ok=True)
    figure.savefig(destination / f"{stock}.png", dpi=160)
    curves.to_csv(destination / f"{stock}_daily_returns.csv", index_label="date")


def process_stock(task):
    """Load and evaluate one stock inside its own worker process."""
    path, args = task
    if hasattr(args, "long_only"):
        from .engine import risk
        risk.DEFAULT_LONG_ONLY = args.long_only
    try:
        frame = load_data(path)
        if frame.empty:
            raise ValueError("No observed bars")
        recent = None if args.test_months is None else args.valid_months + args.test_months
        start_month, end_month = args.start_month, args.end_month
        if args.year is not None:
            start_month, end_month = f"{args.year}-01", f"{args.year}-12"
        if args.years is not None:
            recent = args.years * 12
        months = evaluation_months(frame.index, args.train_months, end_month, recent, start_month)
        if args.valid_months >= len(months):
            raise ValueError("The requested validation labels leave no test months.")
        print(f"{path.stem}: {len(months)} evaluation months, {months[0]} through {months[-1]}", flush=True)
        monthly, summary = evaluate_stock(
            path.stem, frame, months, args.train_months, args.valid_months,
            dict(seed=args.seed, population_size=args.gp_population, generations=args.gp_generations),
            plot_dir=args.output / "plots", objective=args.objective,
        )
        return monthly, summary, None
    except (ValueError, KeyError, OSError) as error:
        print(f"{path.stem}: {error}", flush=True)
        return [], [], dict(stock=path.stem, error=str(error))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_PATH.parent)
    parser.add_argument("--stocks", nargs="+")
    parser.add_argument("--limit", type=int, default=5, help="Use 0 for all stocks.")
    parser.add_argument("--train-months", type=int, default=6)
    parser.add_argument("--valid-months", type=int, default=0,
                        help="Label the first evaluation months validation. Default labels all months test.")
    parser.add_argument("--test-months", type=int,
                        help="Optionally restrict evaluation to recent validation plus test months.")
    parser.add_argument("--end-month", help="Common last evaluation month, YYYY-MM.")
    parser.add_argument("--gp-population", type=int, default=40)
    parser.add_argument("--gp-generations", type=int, default=12)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path, default=Path("reports/batch"))
    parser.add_argument("--workers", type=int, default=1, help="Number of stocks processed concurrently.")
    parser.add_argument("--objective", choices=OBJECTIVES, default=DEFAULT_OBJECTIVE)
    parser.add_argument("--long-only", dest="long_only", action="store_true", default=True,
                        help="Only take long positions (no shorting). Default is True.")
    parser.add_argument("--allow-short", dest="long_only", action="store_false",
                        help="Allow both long and short positions.")
    parser.add_argument("--start-month", help="First evaluation month, YYYY-MM.")
    dates = parser.add_mutually_exclusive_group()
    dates.add_argument("--years", type=int, help="Evaluate the latest N times 12 calendar months.")
    dates.add_argument("--year", type=int, help="Evaluate one calendar year.")
    args = parser.parse_args()
    if args.years is not None and args.years < 1:
        parser.error("--years must be positive.")
    if (args.year is not None or args.years is not None) and (
            args.start_month is not None or args.test_months is not None
            or (args.year is not None and args.end_month is not None)):
        parser.error("Use either a year option or explicit month ranges.")

    if (min(args.train_months, args.gp_generations, args.workers) < 1 or args.valid_months < 0
            or (args.test_months is not None and args.test_months < 1)
            or args.limit < 0 or args.gp_population < 4):
        parser.error("Workers, training/test months and generations must be positive, population at least 4, and limits nonnegative.")
    paths = sorted(args.data_dir.glob("*.parquet"))
    if args.stocks:
        wanted = set(symbol.upper() for symbol in args.stocks)
        paths = [path for path in paths if path.stem.upper() in wanted]
        missing = wanted - {path.stem.upper() for path in paths}
        if missing:
            parser.error(f"Missing stocks: {sorted(missing)}")
    elif args.limit:
        paths = paths[:args.limit]
    if not paths:
        parser.error("No Parquet files found.")
    errors = []
    rows, summaries = [], []
    tasks = [(path, args) for path in paths]
    executor = ProcessPoolExecutor(max_workers=min(args.workers, len(paths))) if args.workers > 1 else None
    try:
        results = executor.map(process_stock, tasks) if executor else map(process_stock, tasks)
        for monthly, summary, error in results:
            rows.extend(monthly)
            summaries.extend(summary)
            if error is not None:
                errors.append(error)
    finally:
        if executor is not None:
            executor.shutdown(wait=True, cancel_futures=True)
    args.output.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(errors, columns=["stock", "error"]).to_csv(args.output / "errors.csv", index=False)
    if not summaries:
        raise RuntimeError("No stocks completed. See errors.csv.")
    pd.DataFrame(rows).to_csv(args.output / "monthly_results.csv", index=False)
    totals = pd.DataFrame(summaries)
    totals["realized_best"] = totals.groupby("stock")["cumulative_return"].transform("max").eq(totals.cumulative_return)
    totals.to_csv(args.output / "stock_summary.csv", index=False)
    ranking = totals.loc[totals.strategy.eq("Selected")].sort_values("cagr", ascending=False).copy()
    selections = pd.DataFrame(rows)
    selections = selections.loc[selections.strategy.eq("Selected")]
    schedule = selections.groupby("stock").apply(
        lambda group: json.dumps(dict(zip(group.month, group.selected_strategy))),
        include_groups=False,
    )
    ranking["selection_by_month"] = ranking.stock.map(schedule)
    ranking["strategy"] = "Monthly selected strategy"
    ranking.insert(0, "rank", range(1, len(ranking) + 1))
    ranking.to_csv(args.output / "stock_rankings.csv", index=False)
    print(ranking[["rank", "stock", "cumulative_return", "cagr", "dd"]].to_string(index=False))


if __name__ == "__main__":
    main()

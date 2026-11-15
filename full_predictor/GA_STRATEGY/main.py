import argparse
from pathlib import Path

import pandas as pd
from .engine.objectives import DEFAULT_OBJECTIVE, OBJECTIVES
from .engine.ga_optimizer import genetic_optimize


OHLCV = ["open", "high", "low", "close", "volume"]
DEFAULT_DATA_PATH = (
    Path(__file__).resolve().parents[1] / "datasets" / "processed_datasets"
    / "NIFTY50" / "HINDALCO.parquet"
)


def load_data(path: str | Path) -> pd.DataFrame:
    df = pd.read_parquet(path).sort_index()
    if not isinstance(df.index, pd.DatetimeIndex) or df.index.hasnans or df.index.has_duplicates:
        raise ValueError("Expected unique, nonmissing timestamps in the Parquet index.")
    # A feature can inherit a retrospective fill anywhere earlier in its segment.
    imputed = df.is_valid_bar & df.is_missing_bar
    contaminated = imputed.groupby(df.segment_id).cummax()
    indicator_columns = df.columns[df.columns.str.startswith(
        ("rsi_", "bb_", "donchian_", "vwap_", "sma_")
    )]
    df.loc[contaminated, indicator_columns] = float("nan")
    observed = df.loc[df.is_valid_bar & ~df.is_missing_bar]
    return observed.rename(columns={column: column.capitalize() for column in OHLCV})


def run_ga_strategy(
    data_path: str | Path,
    train_months: int = 6,
    valid_months: int = 1,
    test_months: int = 1,
    *,
    include_curves: bool = False,
    gp_options: dict | None = None,
    objective: str = DEFAULT_OBJECTIVE,
):
    """
    Run GA parameter optimisation for existing strategies.

    Parameters
    ----------
    data_path : str
        Path to a stock Parquet file produced by feature_engineering.py.
    train_months : int
        Length of training window.
    valid_months : int
        Length of validation window.
    test_months : int
        Length of final held-out test window.
    include_curves : bool
        Include daily compounded return curves in each split report.
    gp_options : dict or None
        Run GP with these search options when supplied.

    Returns
    -------
    dict
        Strategy names mapped to (training score, parameters, metrics).
    """
    if any(type(months) is not int or months < 1 for months in (train_months, valid_months, test_months)):
        raise ValueError("Train, validation, and test months must be positive integers.")
    df = load_data(data_path)

    test_start = df.index.max() - pd.DateOffset(months=test_months)
    train_end = test_start - pd.DateOffset(months=valid_months)
    train_start = train_end - pd.DateOffset(months=train_months)

    df_train = df[(df.index >= train_start) & (df.index < train_end)]
    df_valid = df[(df.index >= train_end) & (df.index < test_start)]
    df_test = df[df.index >= test_start]
    if df_train.empty or df_valid.empty or df_test.empty:
        raise ValueError("The dataset must contain bars in all three windows.")

    try:
        if gp_options is None:
            results = genetic_optimize(df_train, df_valid, df_test, include_curves=include_curves, objective=objective)
        else:
            from .engine.gp_optimizer import optimize_rules

            results = optimize_rules(
                {"train": df_train, "valid": df_valid, "test": df_test},
                include_curves=include_curves, objective=objective, **gp_options,
            )
    except KeyError as error:
        raise ValueError(
            f"Missing precomputed column {error.args[0]!r}. Regenerate {data_path} "
            "with helpers.feature_engineering before running the tester."
        ) from error
    return results


def print_period(label, metrics):
    sharpe = metrics["sharpe"]
    sortino = metrics["sortino"]
    hit_rate = metrics["directional_hit_rate"]
    volatility = metrics["market_daily_volatility"]
    print(f"  {label} | {metrics['start'][:10]} to {metrics['end'][:10]} ({metrics['sessions']} sessions)")
    print(f"    Net {metrics['cumulative_return']:+.2%} | Buy & hold {metrics['buy_and_hold_return']:+.2%}"
          f" | Drawdown {metrics['dd']:.2%}")
    print(f"    Gross P&L contribution: long {metrics['long_pnl']:+.2%}, short {metrics['short_pnl']:+.2%}"
          f" | Cost drag {metrics['cost_drag']:.2%}")
    print(f"    Exposure: long {metrics['long_exposure']:.1%}, short {metrics['short_exposure']:.1%}"
          f" | Position changes {metrics['position_changes']}")
    ratios = [f"{ratio:.2f}" if ratio is not None else "n/a" for ratio in (sharpe, sortino)]
    rates = [f"{rate:.2%}" if rate is not None else "n/a" for rate in (hit_rate, volatility)]
    print(f"    Sharpe {ratios[0]} | Sortino {ratios[1]} | Directional hit rate {rates[0]}"
          f" | Market daily volatility {rates[1]}")
    print(f"    Worst session {metrics['worst_day']} ({metrics['worst_day_return']:+.2%})")


def run_walk_forward(data_path, lookback_months, evaluation_months, *, gp_options=None, objective=DEFAULT_OBJECTIVE):
    """Select on past bars and evaluate on each of the latest calendar months."""
    if min(lookback_months, evaluation_months) < 1:
        raise ValueError("Lookback and evaluation months must be positive.")
    frame = load_data(data_path)
    if frame.empty:
        raise ValueError("The dataset has no valid observed bars.")
    months = pd.period_range(end=frame.index[-1].to_period("M"), periods=evaluation_months, freq="M")
    folds = []
    for month in months:
        start = month.start_time
        fit_start = (month - lookback_months).start_time
        test_end = (month + 1).start_time
        train = frame.loc[(frame.index >= fit_start) & (frame.index < start)]
        test = frame.loc[(frame.index >= start) & (frame.index < test_end)]
        if frame.index[0].to_period("M") > fit_start.to_period("M") or train.empty or test.empty:
            raise ValueError(f"Insufficient lookback or evaluation data for {month}.")
        folds.append((month, train, test))

    rows = []
    for month, train, test in folds:
        print(f"\n{month} | Fitting on {train.index[0]} through {train.index[-1]}")
        if gp_options is None:
            results = genetic_optimize(train, test, objective=objective)
            evaluation_key = "valid"
        else:
            from .engine.gp_optimizer import optimize_rules

            results = optimize_rules({"train": train, "test": test}, objective=objective, **gp_options)
            evaluation_key = "test"
        for name, (score, params, reports) in results.items():
            report = reports[evaluation_key]
            rows.append(dict(
                month=str(month), strategy=name, fit_end=train.index[-1],
                evaluation_start=test.index[0], evaluation_end=test.index[-1],
                training_score=score, selection=params,
                net_return=report["cumulative_return"] if report else None,
                buy_and_hold=float(test["Close"].iloc[-1] / test["Close"].iloc[0] - 1),
            ))
    return pd.DataFrame(rows)


def save_return_plot(results, destination, stock):
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure
    from matplotlib.ticker import PercentFormatter

    figure = Figure(figsize=(14, 5), layout="constrained")
    FigureCanvasAgg(figure)
    axes = figure.subplots(1, 3, sharey=True)
    figure.suptitle(f"{stock} cumulative returns by period", fontsize=16)
    for ax, split, title in zip(axes, ("train", "valid", "test"), ("Training", "Validation", "Test")):
        benchmark = None
        for name, (_, _, reports) in results.items():
            if reports[split] is None:
                continue
            curves = reports[split]["daily_curves"]
            ax.plot(curves.index, curves["strategy"], label=name.split("_")[0], linewidth=1.8)
            benchmark = curves["benchmark"]
        if benchmark is not None:
            ax.plot(benchmark.index, benchmark, "--", color="0.3", label="Buy and hold")
            ax.legend(fontsize=8, loc="best")
        ax.axhline(0, color="0.65", linewidth=0.8)
        ax.set_title(title)
        ax.yaxis.set_major_formatter(PercentFormatter(1))
        ax.grid(axis="y", alpha=0.2)
        ax.tick_params(axis="x", labelrotation=30, labelsize=8)
    axes[0].set_ylabel("Compounded return")
    destination.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(destination, dpi=160)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Backtest strategies on processed stock bars.")
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA_PATH, help="Path to a stock Parquet file.")
    parser.add_argument("--train-months", type=int, default=6)
    parser.add_argument("--valid-months", type=int, default=1)
    parser.add_argument("--test-months", type=int, default=1)
    parser.add_argument("--plot", type=Path, help="Save cumulative return curves to a PNG file.")
    parser.add_argument("--mode", choices=("grid", "gp"), default="grid")
    parser.add_argument("--gp-seed", type=int, default=42)
    parser.add_argument("--gp-population", type=int, default=40)
    parser.add_argument("--gp-generations", type=int, default=12)
    parser.add_argument("--walk-forward", type=int, metavar="MONTHS",
                        help="Refit and evaluate over the latest calendar months.")
    parser.add_argument("--objective", choices=OBJECTIVES, default=DEFAULT_OBJECTIVE)
    parser.add_argument("--long-only", dest="long_only", action="store_true", default=True,
                        help="Only take long positions (no shorting). Default is True.")
    parser.add_argument("--allow-short", dest="long_only", action="store_false",
                        help="Allow both long and short positions.")
    args = parser.parse_args()
    if hasattr(args, "long_only"):
        from .engine import risk
        risk.DEFAULT_LONG_ONLY = args.long_only
    gp_options = None
    if args.mode == "gp":
        if args.gp_population < 4 or args.gp_generations < 1:
            parser.error("GP needs at least four candidates and one generation.")
        gp_options = dict(seed=args.gp_seed, population_size=args.gp_population,
                          generations=args.gp_generations)
    if args.walk_forward is not None:
        if args.plot is not None:
            parser.error("--plot applies to the fixed-split report.")
        try:
            monthly = run_walk_forward(args.data, args.train_months, args.walk_forward, gp_options=gp_options, objective=args.objective)
        except (KeyError, ValueError) as error:
            parser.error(str(error))
        returns = monthly.pivot(index="month", columns="strategy", values="net_return")
        returns["Buy and hold"] = monthly.groupby("month")["buy_and_hold"].first()
        returns.loc["Compounded"] = (1 + returns).prod(skipna=False) - 1
        print("\nMonthly out-of-sample returns")
        print(returns.to_string(float_format=lambda value: f"{value:+.2%}", na_rep="n/a"))
        print("\nMonthly selections")
        print(monthly[["month", "strategy", "selection"]].to_string(index=False))
        raise SystemExit(0)
    res = run_ga_strategy(
        args.data, args.train_months, args.valid_months, args.test_months,
        include_curves=args.plot is not None,
        gp_options=gp_options, objective=args.objective,
    )

    print("\nFINAL RESULTS")
    for strat, (score, params, metrics) in res.items():
        print(f"\nStrategy: {strat}")
        if params is None:
            print("  No candidate had a defined training score.")
            continue
        if args.mode == "gp":
            print(f"  Long if {params['rule']}")
        else:
            print(f"  Best Params: {params}")
        print(f"  Training Score: {score:.3f}" if score is not None else "  Training Score: n/a")
        for split, report in metrics.items():
            print_period(split.upper(), report)

    if args.plot is not None:
        save_return_plot(res, args.plot, args.data.stem)
        print(f"\nSaved return plot to {args.plot.resolve()}")

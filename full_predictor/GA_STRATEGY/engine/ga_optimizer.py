import json
from itertools import product
from pathlib import Path

from .backtester import (
    calmar_ratio,
    DEFAULT_COST,
    StrategyProfitCalculator,
    finalize_strategy,
    max_drawdown,
    sharpe,
    sortino,
    to_daily_returns,
)
from ..strategies import core_strategies
from .execution import execution_returns
from .objectives import DEFAULT_OBJECTIVE, training_score


CONFIG_PATH = Path(__file__).resolve().parents[1] / "config" / "strategies.json"


def load_config(path=CONFIG_PATH):
    with open(path, encoding="utf-8") as config:
        return json.load(config)


def evaluate(df, fn, params, cost, *, report=False, include_curves=False):
    _, signal = fn(df, params, 0.0, finalize_strategy)
    returns = execution_returns(df["Close"], signal)
    if report:
        metrics = StrategyProfitCalculator(df["Close"], signal, returns).summary(include_curves=include_curves)
        metrics["calmar"] = calmar_ratio(returns, metrics["dd"])
        return metrics
    daily = to_daily_returns(returns)
    drawdown = max_drawdown(returns)
    return {
        "sharpe": sharpe(daily),
        "sortino": sortino(daily),
        "dd": drawdown,
        "calmar": calmar_ratio(returns, drawdown),
    }


def genetic_optimize(df_train, df_valid, df_test=None, *, include_curves=False, objective=DEFAULT_OBJECTIVE):
    """Search the configured grids on training data and report held-out metrics."""
    results = {}
    for name, strategy in load_config().items():
        print(f"\nOptimizing {name}...")
        fn = getattr(core_strategies, strategy["function"])
        names = strategy["params"]
        grid = strategy["grid"]
        best_score, best_params = None, None
        checked = 0

        for values in product(*(grid[param] for param in names)):
            checked += 1
            params = dict(zip(names, values))
            train = evaluate(df_train, fn, params, DEFAULT_COST)
            score = training_score(train, objective)
            if score is None:
                continue
            if best_score is None or score > best_score:
                best_score, best_params = score, params

        print(f"  Checked {checked} parameter combinations.")
        splits = {"train": df_train, "valid": df_valid}
        if df_test is not None:
            splits["test"] = df_test
        reports = {
            split: evaluate(frame, fn, best_params, DEFAULT_COST, report=True, include_curves=include_curves)
            if best_params is not None else None
            for split, frame in splits.items()
        }
        results[name] = (best_score, best_params, reports)

    return results

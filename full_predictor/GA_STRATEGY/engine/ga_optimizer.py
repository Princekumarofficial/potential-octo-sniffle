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
from .risk import risk_candidates
from .objectives import DEFAULT_OBJECTIVE, training_score


CONFIG_PATH = Path(__file__).resolve().parents[1] / "config" / "strategies.json"


def load_config(path=CONFIG_PATH):
    with open(path, encoding="utf-8") as config:
        return json.load(config)


def _noop_finalize(ret, signal, cost):
    return None


def evaluate(df, fn, params, cost, *, report=False, include_curves=False, leverage=1.0):
    _, signal = fn(df, params, 0.0, _noop_finalize)
    returns = execution_returns(df["Close"], signal, leverage=leverage, report=report)
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


def genetic_optimize(df_train, df_valid, df_test=None, *, include_curves=False, objective=DEFAULT_OBJECTIVE, leverage=1.0):
    """Search the configured grids on training data and report held-out metrics."""
    results = {}
    for name, strategy in load_config().items():
        print(f"\nOptimizing {name}...")
        fn = getattr(core_strategies, strategy["function"])
        names = strategy["params"]
        grid = strategy["grid"]
        risk_grid = risk_candidates()
        best_score, best_params = None, None
        checked = 0

        for values, risk_params in product(product(*(grid[param] for param in names)), risk_grid):
            checked += 1
            params = {**dict(zip(names, values)), **risk_params}
            train = evaluate(df_train, fn, params, DEFAULT_COST, leverage=leverage)
            score = training_score(train, objective)
            if score is None:
                continue
            if best_score is None or score > best_score:
                best_score, best_params = score, params

        print(f"  Checked {checked} parameter combinations.")
        reports = {}
        if best_params is not None:
            train_rep = evaluate(df_train, fn, best_params, DEFAULT_COST, report=True, include_curves=include_curves, leverage=leverage)
            valid_rep = train_rep if df_valid is df_train else evaluate(df_valid, fn, best_params, DEFAULT_COST, report=True, include_curves=include_curves, leverage=leverage)
            reports["train"] = train_rep
            reports["valid"] = valid_rep
            if df_test is not None:
                reports["test"] = evaluate(df_test, fn, best_params, DEFAULT_COST, report=True, include_curves=include_curves, leverage=leverage)
        else:
            reports["train"] = None
            reports["valid"] = None
            if df_test is not None:
                reports["test"] = None
        results[name] = (best_score, best_params, reports)

    return results

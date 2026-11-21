"""Evolve Boolean trading rules over precomputed indicators."""

import math
import random

import numpy as np
import pandas as pd

from .backtester import DEFAULT_COST
from .ga_optimizer import evaluate
from .objectives import DEFAULT_OBJECTIVE, training_score
from .risk import DEFAULT_STOP_LOSS, DEFAULT_TAKE_PROFIT, apply_risk_management, risk_candidates


# Comparisons are leaves. Boolean operators only accept Boolean subtrees.
FEATURE_THRESHOLDS = {
    "rsi_14": (25, 30, 40, 50, 60, 70, 75),
    "rsi_30": (25, 30, 40, 50, 60, 70, 75),
    "bb_position_20": (0, 0.25, 0.5, 0.75, 1),
    "bb_position_60": (0, 0.25, 0.5, 0.75, 1),
}
BASELINE = ("<", "rsi_14", 30)
MAX_DEPTH = 4


def random_tree(rng, depth):
    if depth == 0 or rng.random() < 0.45:
        feature = rng.choice(tuple(FEATURE_THRESHOLDS))
        return rng.choice(("<", ">")), feature, rng.choice(FEATURE_THRESHOLDS[feature])
    return rng.choice(("AND", "OR")), random_tree(rng, depth - 1), random_tree(rng, depth - 1)


def paths(tree, path=()):
    yield path
    if tree[0] in ("AND", "OR"):
        yield from paths(tree[1], (*path, 1))
        yield from paths(tree[2], (*path, 2))


def subtree(tree, path):
    for branch in path:
        tree = tree[branch]
    return tree


def replace(tree, path, branch):
    if not path:
        return branch
    children = list(tree)
    children[path[0]] = replace(children[path[0]], path[1:], branch)
    return tuple(children)


def describe(tree):
    op, left, right = tree
    if op in ("AND", "OR"):
        return f"({describe(left)} {op} {describe(right)})"
    return f"{left} {op} {right:g}"


def _rule_values_np(feats, tree):
    op, left, right = tree
    if op == "AND":
        return _rule_values_np(feats, left) & _rule_values_np(feats, right)
    if op == "OR":
        return _rule_values_np(feats, left) | _rule_values_np(feats, right)
    arr = feats[left]
    return (arr < right) if op == "<" else (arr > right)


_GP_CACHE = {}


def _get_frame_arrays(frame):
    key = (id(frame), len(frame))
    cached = _GP_CACHE.get(key)
    if cached is not None:
        return cached[0], cached[1], cached[2]
    if len(_GP_CACHE) > 32:
        _GP_CACHE.clear()
    feats = {col: frame[col].to_numpy(dtype=float) for col in FEATURE_THRESHOLDS if col in frame.columns}
    valids = {col: ~np.isnan(feats[col]) for col in feats}
    close = frame["Close"].to_numpy(dtype=float)
    _GP_CACHE[key] = (feats, valids, close)
    return feats, valids, close


def rule_values(frame, tree):
    feats, _, _ = _get_frame_arrays(frame)
    res = _rule_values_np(feats, tree)
    return pd.Series(res, index=frame.index)


def run_rule(frame, tree, cost, finalize):
    # Older saved reports pass a bare tree; new reports also carry risk settings.
    params = tree if isinstance(tree, dict) else {}
    tree = params.get("tree", tree)
    feats, valids, close = _get_frame_arrays(frame)
    used_features = {subtree(tree, path)[1] for path in paths(tree)
                     if subtree(tree, path)[0] in ("<", ">")}
    avail = np.ones(len(close), dtype=bool)
    for f in used_features:
        if f in valids:
            avail &= valids[f]
    raw_signal = np.where(_rule_values_np(feats, tree) & avail, 1.0, 0.0)
    signal = apply_risk_management(
        frame["Close"], pd.Series(raw_signal, index=frame.index),
        stop_loss=params.get("stop_loss", DEFAULT_STOP_LOSS),
        take_profit=params.get("take_profit", DEFAULT_TAKE_PROFIT),
        long_only=True,
    )
    if finalize is not None and getattr(finalize, "__name__", "") == "_noop_finalize":
        return None, signal
    return finalize(frame["Close"].pct_change(), signal, cost), signal


def optimize_rules(splits, *, seed=42, population_size=40, generations=12, include_curves=False, include_tree=False, objective=DEFAULT_OBJECTIVE, leverage=1.0):
    if population_size < 4 or generations < 1:
        raise ValueError("GP needs at least four candidates and one generation.")
    train = splits["train"]
    missing = set(FEATURE_THRESHOLDS) - set(train.columns)
    if missing:
        raise KeyError(", ".join(sorted(missing)))
    rng = random.Random(seed)
    population = [BASELINE, *(random_tree(rng, 3) for _ in range(population_size - 1))]
    fitness = {}
    best_risk = {}
    risk_grid = risk_candidates()

    def score(tree):
        if tree not in fitness:
            best_value = -math.inf
            chosen = risk_grid[0]
            for risk_params in risk_grid:
                stats = evaluate(train, run_rule, {"tree": tree, **risk_params}, DEFAULT_COST, leverage=leverage)
                value = training_score(stats, objective)
                if value is not None and value > best_value:
                    best_value, chosen = value, risk_params
            best_risk[tree] = chosen
            size = sum(1 for _ in paths(tree))
            fitness[tree] = best_value - 0.01 * size
        return fitness[tree]

    for generation in range(generations):
        ranked = sorted(population, key=score, reverse=True)
        print(f"GP generation {generation + 1}/{generations} | best {score(ranked[0]):.3f}"
              f" | {len(fitness)} unique rules evaluated")
        if generation + 1 == generations:
            break
        population = ranked[:2]
        while len(population) < population_size - 1:
            parent, donor = (max(rng.sample(ranked, 3), key=score) for _ in range(2))
            target = rng.choice(tuple(paths(parent)))
            branch = subtree(donor, rng.choice(tuple(paths(donor))))
            if rng.random() < 0.35:
                branch = random_tree(rng, min(2, MAX_DEPTH - len(target)))
            child = replace(parent, target, branch)
            population.append(child if max(map(len, paths(child))) <= MAX_DEPTH else parent)
        population.append(random_tree(rng, 3))

    winner = ranked[0]
    results = {}
    for name, tree in (("GP", winner), ("RSI baseline", BASELINE)):
        valid_score = score(tree)
        if name == "GP" and not math.isfinite(valid_score):
            results[name] = (None, None, dict.fromkeys(splits))
            continue
        reports = {
            period: evaluate(frame, run_rule, {"tree": tree, **best_risk[tree]}, DEFAULT_COST, report=True, include_curves=include_curves, leverage=leverage)
            for period, frame in splits.items()
        }
        results[name] = (valid_score if math.isfinite(valid_score) else None,
                         {"rule": describe(tree), **best_risk[tree]}, reports)
        if include_tree:
            results[name][1]["tree"] = tree
    return results

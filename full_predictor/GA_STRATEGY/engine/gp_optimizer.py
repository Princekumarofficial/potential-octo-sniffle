"""Evolve Boolean trading rules over precomputed indicators."""

import math
import random

import pandas as pd

from .backtester import DEFAULT_COST
from .ga_optimizer import evaluate
from .objectives import DEFAULT_OBJECTIVE, training_score
from .risk import DEFAULT_STOP_LOSS, DEFAULT_TAKE_PROFIT, apply_risk_management


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


def rule_values(frame, tree):
    op, left, right = tree
    if op == "AND":
        return rule_values(frame, left) & rule_values(frame, right)
    if op == "OR":
        return rule_values(frame, left) | rule_values(frame, right)
    values = frame[left]
    return values.lt(right) if op == "<" else values.gt(right)


def run_rule(frame, tree, cost, finalize):
    used_features = {subtree(tree, path)[1] for path in paths(tree)
                     if subtree(tree, path)[0] in ("<", ">")}
    available = frame[list(used_features)].notna().all(axis=1)
    raw_signal = (rule_values(frame, tree) & available).astype(float)
    signal = apply_risk_management(frame["Close"], raw_signal)
    return finalize(frame["Close"].pct_change(), signal, cost), signal


def optimize_rules(splits, *, seed=42, population_size=40, generations=12, include_curves=False, include_tree=False, objective=DEFAULT_OBJECTIVE):
    if population_size < 4 or generations < 1:
        raise ValueError("GP needs at least four candidates and one generation.")
    train = splits["train"]
    missing = set(FEATURE_THRESHOLDS) - set(train.columns)
    if missing:
        raise KeyError(", ".join(sorted(missing)))
    rng = random.Random(seed)
    population = [BASELINE, *(random_tree(rng, 3) for _ in range(population_size - 1))]
    fitness = {}

    def score(tree):
        if tree not in fitness:
            stats = evaluate(train, run_rule, tree, DEFAULT_COST)
            value = training_score(stats, objective)
            if value is None:
                fitness[tree] = -math.inf
            else:
                size = sum(1 for _ in paths(tree))
                fitness[tree] = value - 0.01 * size
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
            period: evaluate(frame, run_rule, tree, DEFAULT_COST, report=True, include_curves=include_curves)
            for period, frame in splits.items()
        }
        results[name] = (valid_score if math.isfinite(valid_score) else None,
                         {"rule": describe(tree)}, reports)
        if include_tree:
            results[name][1]["tree"] = tree
    return results

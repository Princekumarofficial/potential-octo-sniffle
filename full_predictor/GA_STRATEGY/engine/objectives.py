"""Shared training objectives for parameter search and genetic programming."""
import math

DEFAULT_OBJECTIVE = "blend"
OBJECTIVES = ("blend", "calmar", "composite")


def training_score(metrics, objective=DEFAULT_OBJECTIVE):
    if objective not in OBJECTIVES:
        raise ValueError(f"Unknown objective: {objective}")
    if objective == "calmar":
        score = metrics["calmar"]
    elif objective == "composite":
        sh, so = metrics["sharpe"], metrics["sortino"]
        cal = metrics["calmar"]
        if sh is None or so is None or cal is None:
            score = None
        else:
            dd = metrics["dd"]
            score = (0.5 * sh + 0.8 * so
                     + 2.0 * dd - 5.0 * dd * dd
                     + 0.4 * min(cal, 2.0))
    else:
        sh, so = metrics["sharpe"], metrics["sortino"]
        score = None if sh is None or so is None else 0.7 * sh + 1.3 * so + 0.5 * metrics["dd"]
    return score if score is not None and math.isfinite(score) else None

from __future__ import annotations

from statistics import NormalDist
import operator

import numpy as np


def series(y) -> np.ndarray:
    values = np.ascontiguousarray(y, dtype=np.float64)
    if values.ndim != 1:
        raise ValueError("y must be a one-dimensional array")
    if len(values) < 3:
        raise ValueError("y must contain at least three observations")
    if not np.all(np.isfinite(values)):
        raise ValueError("y must contain only finite values")
    if not values.flags.writeable:
        values = values.copy()
    return values


def reject_exog(X, X_future=None) -> None:
    if X is not None or X_future is not None:
        raise NotImplementedError("exogenous regressors are not covered by this port")


def integer(value, name: str, *, minimum: int = 0) -> int:
    if isinstance(value, (bool, np.bool_)):
        raise TypeError(f"{name} must be an integer")
    try:
        result = operator.index(value)
    except TypeError:
        raise TypeError(f"{name} must be an integer") from None
    if result < minimum:
        qualifier = "positive" if minimum == 1 else "non-negative"
        raise ValueError(f"{name} must be {qualifier}")
    return result


def horizon(h) -> int:
    return integer(h, "h", minimum=1)


def intervals(result, mean, sigma, level, scale=1.0):
    if level is None:
        return result
    try:
        levels = sorted(float(item) for item in level)
    except (TypeError, ValueError):
        raise ValueError("level must contain numeric percentages") from None
    if not levels or not all(np.isfinite(item) and 0.0 < item < 100.0 for item in levels):
        raise ValueError("level percentages must be finite and between 0 and 100")
    if len(set(levels)) != len(levels):
        raise ValueError("level percentages must be unique")
    for item in reversed(levels):
        z = NormalDist().inv_cdf(0.5 + float(item) / 200.0)
        result[f"lo-{item:g}"] = mean - z * sigma * scale
    for item in levels:
        z = NormalDist().inv_cdf(0.5 + float(item) / 200.0)
        result[f"hi-{item:g}"] = mean + z * sigma * scale
    return result

"""Benchmark the covered public APIs against StatsForecast 2.1.1."""

from __future__ import annotations

import math
import os
import platform
import sys
import time
import warnings

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "python"))

import mojostatsforecast as mojo_sf  # noqa: E402
from statsforecast import models as upstream  # noqa: E402

warnings.filterwarnings(
    "ignore", message="invalid value encountered in subtract", category=RuntimeWarning
)


def timeit(function, repeat=3):
    best = math.inf
    for _ in range(repeat):
        start = time.perf_counter()
        function()
        best = min(best, time.perf_counter() - start)
    return best


def random_walk(n, seed=0):
    rng = np.random.default_rng(seed)
    return np.ascontiguousarray(100 + np.cumsum(rng.normal(scale=0.2, size=n)))


def seasonal(n, period=12, seed=1):
    rng = np.random.default_rng(seed)
    pattern = 3 * np.sin(2 * np.pi * np.arange(period) / period)
    return np.ascontiguousarray(
        20 + 0.002 * np.arange(n) + np.resize(pattern, n)
        + rng.normal(scale=0.2, size=n)
    )


CASES = [
    (
        "AutoETS ANN forecast (50k)",
        lambda y=random_walk(50_000): mojo_sf.AutoETS(model="ANN").forecast(y, 24),
        lambda y=random_walk(50_000): upstream.AutoETS(model="ANN").forecast(y, 24),
    ),
    (
        "AutoETS AAA forecast (20k, m=12)",
        lambda y=seasonal(20_000): mojo_sf.AutoETS(12, "AAA").forecast(y, 24),
        lambda y=seasonal(20_000): upstream.AutoETS(12, "AAA").forecast(y, 24),
    ),
    (
        "ARIMA(2,0,1) forecast (5k)",
        lambda y=random_walk(5_000): mojo_sf.ARIMA((2, 0, 1)).forecast(y, 24),
        lambda y=random_walk(5_000): upstream.ARIMA((2, 0, 1)).forecast(y, 24),
    ),
    (
        "Theta forecast (5k)",
        lambda y=random_walk(5_000): mojo_sf.Theta().forecast(y, 24),
        lambda y=random_walk(5_000): upstream.Theta().forecast(y, 24),
    ),
]


def main():
    print(f"Machine: {platform.processor() or platform.machine()} | {platform.platform()}")
    print("| case | mojo-statsforecast | statsforecast | result |")
    print("| --- | ---: | ---: | ---: |")
    for name, ours, theirs in CASES:
        ours()
        theirs()
        mojo_time = timeit(ours)
        upstream_time = timeit(theirs)
        ratio = upstream_time / mojo_time
        result = f"{ratio:.2f}x faster" if ratio >= 1 else f"{1 / ratio:.2f}x slower"
        print(
            f"| {name} | {mojo_time * 1e3:.2f} ms | "
            f"{upstream_time * 1e3:.2f} ms | {result} |"
        )


if __name__ == "__main__":
    main()

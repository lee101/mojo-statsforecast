# mojo-statsforecast

ETS, ARIMA, and Theta forecasting implemented with native
[Mojo](https://www.modular.com/mojo) recursions and a Python API shaped like
[StatsForecast](https://github.com/Nixtla/statsforecast).

```python
import numpy as np
from mojostatsforecast.models import AutoETS

y = np.array([10, 12, 14, 12, 11, 13, 15, 13] * 4, dtype=float)
model = AutoETS(season_length=4, model="AAA").fit(y)
forecast = model.predict(h=4, level=[80, 95])
print(forecast["mean"])
```

The model classes provide the upstream `fit`, `predict`, `predict_in_sample`,
`forecast`, and `forward` entry points and return dictionaries with `mean`,
`fitted`, `lo-*`, and `hi-*` arrays.

## Coverage

| family | covered |
| --- | --- |
| ETS | `AutoETS`; additive-error `ANN`, `AAN`, `AAA`, optional damped additive trend, and `Z` selection by AICc |
| ARIMA | `ARIMA(p,d,q)` for non-seasonal `p` and `q`, `d=0..2`, CSS coefficient estimation, fixed coefficients, and `AutoARIMA` AICc search |
| Theta | `Theta`, `OptimizedTheta`, `AutoTheta`, plus compatible dynamic-class entry points; additive or multiplicative classical seasonal adjustment |
| common API | point forecasts, fitted values, normal intervals, and the five core model methods |

This is intentionally not the whole StatsForecast project. Multiplicative ETS
errors/trends, seasonal ARIMA terms, exogenous regressors, Box-Cox transforms,
ARIMA drift and ML estimation, conformal intervals, non-normal distributions,
missing values, and simulation
are not covered. Unsupported structural features raise `NotImplementedError`;
they do not silently fall back to Python. `DynamicTheta` currently uses the
classical Theta recursion and `DynamicOptimizedTheta` uses the optimized
recursion rather than time-varying theta states.

The parity suite uses the real conda-forge `statsforecast==2.1.1`. It checks
point forecasts and ARIMA coefficients numerically, while separate NumPy
references check the native ETS and ARMA filtering recurrences, including the
ANN SIMD tail.

## Install

```bash
pixi install
pixi run build
pixi run test
```

`pixi install` supplies Python 3.13, the pinned Mojo nightly, NumPy, SciPy,
pytest, and StatsForecast 2.1.1. The build creates
`dist/libmojo-statsforecast.so`. Set `PYTHONPATH=python` when using the package
outside a Pixi task, or install it with `pip install -e .` after building.

## Performance

Measured with `pixi run bench` on an x86-64 machine running Linux
6.8.0-136-generic and glibc 2.39. Both libraries are warmed before timing;
each cell is the best of three complete public `forecast` calls, including
parameter estimation.

| case | mojo-statsforecast | statsforecast | result |
| --- | ---: | ---: | ---: |
| AutoETS ANN forecast (50k) | 8.67 ms | 112.93 ms | 13.02x faster |
| AutoETS AAA forecast (20k, m=12) | 284.15 ms | 6458.56 ms | 22.73x faster |
| ARIMA(2,0,1) forecast (5k) | 37.05 ms | 313.82 ms | 8.47x faster |
| Theta forecast (5k) | 1.33 ms | 935.01 ms | 701.55x faster |

These results are for the covered algorithms and sizes, not a claim about the
many upstream models this repository does not implement. The benchmark source,
data generation, warm-up, and timing loop are in `bench/bench.py`.

There is no GPU path. The covered single-series kernels are low-arithmetic-
intensity recurrences with loop-carried state, so neither GPU offload nor CPU
thread launch has enough independent work to justify its overhead.

## How it works

All time-series state transitions live in one Mojo compilation unit. ETS uses
innovations state-space updates for level, additive trend, and additive
seasonality. ARIMA performs conditional residual recursion and future ARMA
recursion over differenced observations. Theta reuses the native SES filter
and combines its terminal level with the regression-line drift.

Python owns model selection and numerical optimization. Each objective
evaluation is one ctypes call into Mojo over the complete series, so the
recurrence loop stays native without reproducing a mature optimizer.

Arrays cross the C ABI as integer addresses because exported Mojo functions
cannot be parametric over pointer origins. Inputs are C-contiguous `float64`;
state, fitted, residual, and forecast buffers are NumPy-owned row-major arrays.
Mojo allocates nothing and never retains a pointer after a call.

## Development

```bash
pixi run build
pixi run test
pixi run bench
```

The benchmark task holds `/tmp/mojo-bench.lock` so simultaneous factory jobs
cannot contaminate results.

## License

MIT

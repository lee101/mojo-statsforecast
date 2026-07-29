from __future__ import annotations

import numpy as np
from scipy.optimize import minimize

from ._common import horizon, integer, intervals, reject_exog, series
from ._lib import addr, lib


def _difference(y: np.ndarray, d: int) -> tuple[np.ndarray, list[np.ndarray]]:
    levels = []
    current = y
    for _ in range(d):
        levels.append(current.copy())
        current = np.ascontiguousarray(np.diff(current))
    return current, levels


def _integrate(forecast: np.ndarray, levels: list[np.ndarray]) -> np.ndarray:
    result = forecast.copy()
    for previous in reversed(levels):
        result = np.cumsum(result) + previous[-1]
    return result


def _fit_arima(y: np.ndarray, p: int, q: int, include_mean: bool, fixed=None):
    count = int(include_mean) + p + q
    coefficients = np.zeros(count, dtype=np.float64)
    offset = int(include_mean)
    if include_mean:
        coefficients[0] = np.mean(y)
    if p:
        rows = len(y) - p
        design = np.column_stack(
            [y[p - j - 1 : len(y) - j - 1] for j in range(p)]
        )
        target = y[p:]
        if include_mean:
            design = np.column_stack([np.ones(rows), design])
        estimate = np.linalg.lstsq(design, target, rcond=None)[0]
        if include_mean:
            ar = estimate[1:]
            coefficients[1 : 1 + p] = ar
            denominator = 1.0 - np.sum(ar)
            coefficients[0] = estimate[0] / (
                denominator if abs(denominator) > 1e-6 else 1e-6
            )
        else:
            coefficients[:p] = estimate
    coefficients[offset + p :] = 0.05
    residuals = np.empty_like(y)
    fitted = np.empty_like(y)
    native = lib().msf_arima_residuals
    fixed = fixed or {}
    keys = (
        (["intercept"] if include_mean else [])
        + [f"ar{i + 1}" for i in range(p)]
        + [f"ma{i + 1}" for i in range(q)]
    )
    free = [i for i, key in enumerate(keys) if key not in fixed]
    for i, key in enumerate(keys):
        if key in fixed:
            coefficients[i] = float(fixed[key])

    def objective(free_values):
        coefficients[free] = free_values
        return native(
            addr(y), addr(coefficients), addr(residuals), addr(fitted),
            len(y), p, q, include_mean,
        )

    if free:
        bounds = [
            (None, None) if include_mean and index == 0 else (-1.99, 1.99)
            for index in free
        ]
        optimum = minimize(
            objective, coefficients[free].copy(), method="L-BFGS-B", bounds=bounds,
            options={"maxiter": 400, "ftol": 1e-12},
        )
        if not optimum.success or not np.all(np.isfinite(optimum.x)):
            raise RuntimeError(f"ARIMA optimization failed: {optimum.message}")
        coefficients[free] = optimum.x
    sse = native(
        addr(y), addr(coefficients), addr(residuals), addr(fitted),
        len(y), p, q, include_mean,
    )
    start = max(p, q)
    n_eff = len(y) - start
    k = len(free)
    sigma2 = sse / max(n_eff - k, 1)
    aic = n_eff * np.log(max(sse / n_eff, np.finfo(float).tiny)) + 2 * k
    aicc = aic + 2 * k * (k + 1) / max(n_eff - k - 1, 1)
    return {
        "coef_array": coefficients.copy(),
        "coef": dict(zip(keys, coefficients)),
        "residuals": residuals.copy(),
        "fitted_diff": fitted.copy(),
        "sigma2": sigma2,
        "aicc": aicc,
        "p": p,
        "q": q,
    }


class ARIMA:
    uses_exog = False

    def __init__(
        self, order=(0, 0, 0), season_length: int = 1,
        seasonal_order=(0, 0, 0), include_mean: bool = True,
        include_drift: bool = False, include_constant=None, blambda=None,
        biasadj: bool = False, method: str = "CSS-ML", fixed=None,
        distribution: str = "normal", alias: str = "ARIMA",
        prediction_intervals=None,
    ):
        if len(order) != 3:
            raise ValueError("order must be a non-negative (p, d, q) tuple")
        order = tuple(integer(item, "order component") for item in order)
        if tuple(seasonal_order) != (0, 0, 0):
            raise NotImplementedError("seasonal ARIMA terms are not covered")
        if season_length != 1:
            raise NotImplementedError("season_length is only used with seasonal ARIMA")
        if blambda is not None:
            raise NotImplementedError("Box-Cox ARIMA is not covered")
        if include_drift or (include_constant is True and order[1] > 0):
            raise NotImplementedError("ARIMA drift terms are not covered")
        if distribution != "normal":
            raise NotImplementedError("only normal forecast errors are covered")
        if prediction_intervals is not None:
            raise NotImplementedError("conformal prediction intervals are not covered")
        if biasadj:
            raise NotImplementedError("ARIMA bias adjustment is not covered")
        if method not in ("CSS", "CSS-ML"):
            raise NotImplementedError("only CSS coefficient estimation is covered")
        if include_constant is not None:
            include_mean = bool(include_constant)
        self.order = order
        self.season_length = int(season_length)
        self.seasonal_order = tuple(seasonal_order)
        self.include_mean = include_mean
        self.include_drift = include_drift
        self.include_constant = include_constant
        self.blambda = blambda
        self.biasadj = biasadj
        self.method = method
        self.fixed = fixed
        self.distribution = distribution
        self.alias = alias
        self.prediction_intervals = prediction_intervals

    def __repr__(self):
        return self.alias

    def fit(self, y, X=None):
        reject_exog(X)
        values = series(y)
        p, d, q = self.order
        if d > 2:
            raise NotImplementedError("d > 2 is not covered")
        differenced, levels = _difference(values, d)
        if len(differenced) <= max(p, q):
            raise ValueError("the differenced series must be longer than every AR/MA lag")
        include_mean = bool(self.include_mean and d == 0)
        self.model_ = _fit_arima(differenced, p, q, include_mean, self.fixed)
        self.model_.update(
            {"order": self.order, "levels": levels, "include_mean": include_mean}
        )
        self._y = values
        self._diff_y = differenced
        fitted = values.copy()
        start = max(p, q) + d
        if d == 0:
            fitted[:] = self.model_["fitted_diff"]
        else:
            errors = self.model_["residuals"]
            fitted[start:] = values[start:] - errors[max(p, q):]
        self.model_["fitted"] = fitted
        return self

    def predict(self, h: int, X=None, level=None):
        reject_exog(X)
        if not hasattr(self, "model_"):
            raise Exception("You have to use the `fit` method first")
        p, d, q = self.order
        h = horizon(h)
        diff_forecast = np.empty(h, dtype=np.float64)
        lib().msf_arima_forecast(
            addr(self._diff_y), addr(self.model_["residuals"]),
            addr(self.model_["coef_array"]), addr(diff_forecast),
            len(self._diff_y), p, q, h, self.model_["include_mean"],
        )
        mean = _integrate(diff_forecast, self.model_["levels"])
        scale = np.sqrt(np.arange(1, h + 1)) if d else 1.0
        return intervals(
            {"mean": mean}, mean, np.sqrt(self.model_["sigma2"]), level, scale
        )

    def predict_in_sample(self, level=None):
        if not hasattr(self, "model_"):
            raise Exception("You have to use the `fit` method first")
        fitted = self.model_["fitted"].copy()
        return intervals(
            {"fitted": fitted}, fitted, np.sqrt(self.model_["sigma2"]), level
        )

    def forecast(
        self, y, h: int, X=None, X_future=None, level=None, fitted: bool = False
    ):
        reject_exog(X, X_future)
        self.fit(y)
        result = self.predict(h, level=level)
        if fitted:
            result["fitted"] = self.model_["fitted"].copy()
        return result

    def forward(
        self, y, h: int, X=None, X_future=None, level=None, fitted: bool = False
    ):
        reject_exog(X, X_future)
        return self.forecast(y, h, level=level, fitted=fitted)


class AutoARIMA:
    uses_exog = False

    def __init__(
        self, d=None, D=None, max_p: int = 5, max_q: int = 5,
        max_P: int = 2, max_Q: int = 2, max_order: int = 5,
        max_d: int = 2, max_D: int = 1, start_p: int = 2,
        start_q: int = 2, start_P: int = 1, start_Q: int = 1,
        stationary: bool = False, seasonal: bool = True, ic: str = "aicc",
        stepwise: bool = True, nmodels: int = 94, trace: bool = False,
        approximation=False, method=None, truncate=None, test: str = "kpss",
        test_kwargs=None, seasonal_test: str = "seas",
        seasonal_test_kwargs=None, allowdrift: bool = True,
        allowmean: bool = True, blambda=None, biasadj: bool = False,
        season_length: int = 1, distribution: str = "normal",
        alias: str = "AutoARIMA", prediction_intervals=None,
    ):
        if season_length != 1:
            raise NotImplementedError("automatic seasonal ARIMA is not covered")
        if D not in (None, 0):
            raise NotImplementedError("seasonal differencing is not covered")
        if blambda is not None:
            raise NotImplementedError("Box-Cox ARIMA is not covered")
        if ic != "aicc":
            raise NotImplementedError("AutoARIMA currently selects by AICc")
        if distribution != "normal":
            raise NotImplementedError("only normal forecast errors are covered")
        if prediction_intervals is not None:
            raise NotImplementedError("conformal prediction intervals are not covered")
        if biasadj:
            raise NotImplementedError("ARIMA bias adjustment is not covered")
        self.d = None if d is None else integer(d, "d")
        self.max_p = integer(max_p, "max_p")
        self.max_q = integer(max_q, "max_q")
        self.max_order = integer(max_order, "max_order")
        self.max_d = integer(max_d, "max_d")
        self.stationary = stationary
        self.allowmean = allowmean
        self.alias = alias
        self.prediction_intervals = prediction_intervals
        self.distribution = distribution

    def __repr__(self):
        return self.alias

    def fit(self, y, X=None):
        reject_exog(X)
        values = series(y)
        if self.d is None:
            ratio = np.std(np.diff(values)) / max(np.std(values), 1e-12)
            differences = 1 if not self.stationary and ratio < 0.8 else 0
        else:
            differences = self.d
        if differences > 2:
            raise NotImplementedError("d > 2 is not covered")
        candidates = []
        for p in range(self.max_p + 1):
            for q in range(self.max_q + 1):
                if p + q > self.max_order:
                    continue
                candidates.append(
                    ARIMA(
                        (p, differences, q), include_mean=self.allowmean,
                        alias=self.alias,
                        prediction_intervals=self.prediction_intervals,
                        distribution=self.distribution,
                    ).fit(values)
                )
        self._model = min(candidates, key=lambda item: item.model_["aicc"])
        self.model_ = self._model.model_
        self._y = values
        return self

    def predict(self, h: int, X=None, level=None):
        return self._model.predict(h, X=X, level=level)

    def predict_in_sample(self, level=None):
        return self._model.predict_in_sample(level=level)

    def forecast(
        self, y, h: int, X=None, X_future=None, level=None, fitted: bool = False
    ):
        self.fit(y, X=X)
        result = self._model.predict(h, X=X_future, level=level)
        if fitted:
            result["fitted"] = self.model_["fitted"].copy()
        return result

    def forward(
        self, y, h: int, X=None, X_future=None, level=None, fitted: bool = False
    ):
        return self.forecast(y, h, X, X_future, level, fitted)

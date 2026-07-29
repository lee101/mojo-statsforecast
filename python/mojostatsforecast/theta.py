from __future__ import annotations

import numpy as np
from scipy.optimize import minimize_scalar

from ._common import horizon, integer, intervals, reject_exog, series
from ._lib import addr, lib


def _seasonal_factors(y: np.ndarray, m: int, kind: str) -> np.ndarray:
    if m == 1:
        return np.ones(1)
    cycles = len(y) // m
    trimmed = y[: cycles * m].reshape(cycles, m)
    if kind == "multiplicative":
        bases = np.mean(trimmed, axis=1)
        if np.any(bases == 0):
            raise ValueError("multiplicative seasonality requires non-zero cycle means")
        factors = np.mean(trimmed / bases[:, None], axis=0)
        factors = factors / np.mean(factors)
        if not np.all(np.isfinite(factors)) or np.any(factors == 0):
            raise ValueError("multiplicative seasonality requires non-zero factors")
        return factors
    factors = np.mean(trimmed - np.mean(trimmed, axis=1)[:, None], axis=0)
    return factors - np.mean(factors)


class Theta:
    def __init__(
        self, season_length: int = 1,
        decomposition_type: str = "multiplicative", alias: str = "Theta",
        prediction_intervals=None, distribution: str = "normal",
    ):
        if decomposition_type not in ("multiplicative", "additive"):
            raise ValueError("decomposition_type must be multiplicative or additive")
        if distribution != "normal":
            raise NotImplementedError("only normal forecast errors are covered")
        if prediction_intervals is not None:
            raise NotImplementedError("conformal prediction intervals are not covered")
        self.season_length = integer(season_length, "season_length", minimum=1)
        self.decomposition_type = decomposition_type
        self.alias = alias
        self.prediction_intervals = prediction_intervals
        self.distribution = distribution

    def __repr__(self):
        return self.alias

    def fit(self, y, X=None):
        reject_exog(X)
        values = series(y)
        m = self.season_length
        if m > 1 and len(values) < 2 * m:
            raise ValueError("Theta seasonality requires at least two complete seasons")
        if getattr(self, "_optimized", False) and len(values) < 5:
            raise ValueError("optimized Theta requires at least five observations")
        factors = _seasonal_factors(values, m, self.decomposition_type)
        index = np.arange(len(values)) % m
        if m == 1:
            adjusted = values.copy()
        elif self.decomposition_type == "multiplicative":
            adjusted = np.ascontiguousarray(values / factors[index])
        else:
            adjusted = np.ascontiguousarray(values - factors[index])
        time = np.arange(len(values), dtype=np.float64)
        slope, intercept = np.polyfit(time, adjusted, 1)
        initial = np.zeros(3, dtype=np.float64)
        initial[0] = adjusted[0]
        state = np.empty_like(initial)
        fitted = np.empty_like(adjusted)

        def objective(alpha):
            return lib().msf_ets_filter(
                addr(adjusted), addr(fitted), addr(initial), addr(state),
                len(adjusted), 1, alpha, 0.0, 0.0, 1.0, 0, 0,
            )

        weight = 0.5
        if getattr(self, "_optimized", False):
            alpha = 0.99
            objective(alpha)
            levels = fitted + alpha * (adjusted - fitted)

            def optimized_objective(theta):
                weight_value = 1.0 - 1.0 / theta
                errors = []
                for horizon in (1, 2, 3):
                    origins = np.arange(3, len(adjusted) - horizon)
                    drift = weight_value * slope * (
                        horizon - 1.0 + 1.0 / alpha
                        - (1.0 - alpha) ** (origins + 1) / alpha
                    )
                    errors.append(adjusted[origins + horizon] - levels[origins] - drift)
                return float(np.mean(np.concatenate(errors) ** 2))

            optimum = minimize_scalar(
                optimized_objective, bounds=(1.0, 1000.0), method="bounded",
                options={"xatol": 1e-8},
            )
            if not optimum.success or not np.isfinite(optimum.x):
                raise RuntimeError(f"Theta optimization failed: {optimum.message}")
            weight = 1.0 - 1.0 / float(optimum.x)
            sse = objective(alpha)
        else:
            optimum = minimize_scalar(
                objective, bounds=(1e-5, 0.99999), method="bounded",
                options={"xatol": 1e-10},
            )
            if not optimum.success or not np.isfinite(optimum.x):
                raise RuntimeError(f"Theta optimization failed: {optimum.message}")
            alpha = float(optimum.x)
            sse = objective(alpha)
        ses_level = float(state[0])
        drift_shape = slope * (
            time - 1.0 + 1.0 / alpha
            - (1.0 - alpha) ** (time + 1) / alpha
        )
        fitted_theta = fitted + weight * drift_shape
        if m > 1:
            if self.decomposition_type == "multiplicative":
                fitted_theta *= factors[index]
            else:
                fitted_theta += factors[index]
        residuals = values - fitted_theta
        self.model_ = {
            "alpha": alpha,
            "slope": float(slope),
            "intercept": float(intercept),
            "level": ses_level,
            "weight": weight,
            "seasonal": factors,
            "fitted": fitted_theta,
            "residuals": residuals,
            "sigma2": float(
                np.sum(residuals[2:] ** 2) / max(len(values) - 4, 1)
            ),
            "sse": sse,
        }
        self._y = values
        return self

    def predict(self, h: int, X=None, level=None):
        reject_exog(X)
        if not hasattr(self, "model_"):
            raise Exception("You have to use the `fit` method first")
        h = horizon(h)
        steps = np.arange(1, h + 1, dtype=np.float64)
        alpha = self.model_["alpha"]
        drift = self.model_["weight"] * self.model_["slope"] * (
            steps - 1.0 + 1.0 / alpha
            - (1.0 - alpha) ** len(self._y) / alpha
        )
        mean = self.model_["level"] + drift
        m = self.season_length
        if m > 1:
            factors = self.model_["seasonal"][
                (len(self._y) + np.arange(h)) % m
            ]
            if self.decomposition_type == "multiplicative":
                mean = mean * factors
            else:
                mean = mean + factors
        mean = np.ascontiguousarray(mean)
        return intervals(
            {"mean": mean}, mean, np.sqrt(self.model_["sigma2"]), level,
            np.sqrt(steps),
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
        return self.forecast(y, h, X, X_future, level, fitted)


class OptimizedTheta(Theta):
    def __init__(
        self, season_length=1, decomposition_type="multiplicative",
        alias="OptimizedTheta", prediction_intervals=None,
    ):
        super().__init__(
            season_length, decomposition_type, alias, prediction_intervals
        )
        self._optimized = True


class DynamicTheta(Theta):
    def __init__(
        self, season_length=1, decomposition_type="multiplicative",
        alias="DynamicTheta", prediction_intervals=None,
    ):
        super().__init__(
            season_length, decomposition_type, alias, prediction_intervals
        )


class DynamicOptimizedTheta(Theta):
    def __init__(
        self, season_length=1, decomposition_type="multiplicative",
        alias="DynamicOptimizedTheta", prediction_intervals=None,
    ):
        super().__init__(
            season_length, decomposition_type, alias, prediction_intervals
        )
        self._optimized = True


class AutoTheta(Theta):
    def __init__(
        self, season_length: int = 1,
        decomposition_type: str = "multiplicative", model: str | None = None,
        alias: str = "AutoTheta", prediction_intervals=None,
        distribution: str = "normal",
    ):
        if model not in (None, "STM", "OTM", "DSTM", "DOTM"):
            raise ValueError("model must be one of None, STM, OTM, DSTM, DOTM")
        super().__init__(
            season_length, decomposition_type, alias, prediction_intervals,
            distribution,
        )
        self.model = model
        self._optimized = model in (None, "OTM", "DOTM")

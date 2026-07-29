from __future__ import annotations

import numpy as np
from scipy.optimize import minimize

from ._common import horizon, integer, intervals, reject_exog, series
from ._lib import addr, lib


def _initial_state(y: np.ndarray, m: int, trend: bool, seasonal: bool) -> np.ndarray:
    state = np.zeros(m + 2, dtype=np.float64)
    if seasonal:
        first = y[:m]
        state[0] = float(np.mean(first))
        state[2:] = first - state[0]
        if len(y) >= 2 * m:
            state[1] = float(np.mean(y[m : 2 * m] - first) / m)
    else:
        state[0] = float(y[0])
        if trend:
            steps = min(4, len(y) - 1)
            state[1] = float((y[steps] - y[0]) / steps)
    return state


def _fit_ets(y: np.ndarray, m: int, code: str, damped: bool, phi_value):
    trend = code[1] == "A"
    seasonal = code[2] == "A"
    initial = _initial_state(y, m, trend, seasonal)
    fitted = np.empty_like(y)
    state = np.empty_like(initial)
    native = lib().msf_ets_filter
    native_sse_grad_ann = (
        lib().msf_ets_sse_grad_ann if not trend and not seasonal else None
    )
    y_address = addr(y)
    fitted_address = addr(fitted)
    initial_address = addr(initial)
    state_address = addr(state)

    names = ["alpha"]
    x0 = [0.2]
    bounds = [(1e-5, 0.99999)]
    if trend:
        names.append("beta")
        x0.append(0.05)
        bounds.append((1e-6, 0.99999))
    if seasonal:
        names.append("gamma")
        x0.append(0.05)
        bounds.append((1e-6, 0.99999))
    if damped and phi_value is None:
        names.append("phi")
        x0.append(0.98)
        bounds.append((0.8, 0.98))
    names.append("level")
    level_index = len(x0)
    x0.append(initial[0])
    span = max(float(np.ptp(y)), 1.0)
    bounds.append((float(np.min(y) - span), float(np.max(y) + span)))
    if trend:
        names.append("initial_trend")
        trend_index = len(x0)
        x0.append(initial[1])
        bounds.append((-span, span))

    def unpack(x):
        values = dict(zip(names, x))
        return (
            values["alpha"],
            values.get("beta", 0.0),
            values.get("gamma", 0.0),
            values.get("phi", phi_value if damped else 1.0),
        )

    def objective(x):
        alpha, beta, gamma, phi = unpack(x)
        if beta > alpha or (seasonal and gamma > 1.0 - alpha):
            return 1e100
        initial[0] = x[level_index]
        if trend:
            initial[1] = x[trend_index]
        return native(
            y_address, fitted_address, initial_address, state_address, len(y), m,
            alpha, beta, gamma, phi, trend, seasonal,
        )

    if native_sse_grad_ann is not None:
        gradient = np.empty(2, dtype=np.float64)
        gradient_address = addr(gradient)

        def objective_with_gradient(x):
            sse = native_sse_grad_ann(
                y_address, len(y), x[level_index], x[0], gradient_address
            )
            return sse, gradient.copy()

        optimum = minimize(
            objective_with_gradient, np.asarray(x0), method="L-BFGS-B",
            jac=True, bounds=bounds, options={"maxiter": 400, "ftol": 1e-12},
        )
    else:
        optimum = minimize(
            objective, np.asarray(x0), method="Nelder-Mead", bounds=bounds,
            options={"maxiter": 3000, "xatol": 1e-8, "fatol": 1e-8},
        )
    if not optimum.success or not np.all(np.isfinite(optimum.x)):
        raise RuntimeError(f"ETS optimization failed: {optimum.message}")
    alpha, beta, gamma, phi = unpack(optimum.x)
    initial[0] = optimum.x[level_index]
    if trend:
        initial[1] = optimum.x[trend_index]
    sse = native(
        y_address, fitted_address, initial_address, state_address, len(y), m,
        alpha, beta, gamma, phi, trend, seasonal,
    )
    k = len(optimum.x) + (m - 1 if seasonal else 0)
    n = len(y)
    aic = n * np.log(max(sse / n, np.finfo(float).tiny)) + 2 * k
    aicc = aic + 2 * k * (k + 1) / max(n - k - 1, 1)
    return {
        "model": code,
        "params": {"alpha": alpha, "beta": beta, "gamma": gamma, "phi": phi},
        "state": state,
        "initial_state": initial,
        "fitted": fitted,
        "residuals": y - fitted,
        "sigma2": sse / max(n - k, 1),
        "sse": sse,
        "aicc": aicc,
        "n_params": k,
    }


class AutoETS:
    def __init__(
        self, season_length: int = 1, model: str = "ZZZ",
        damped: bool | None = None, phi: float | None = None,
        alias: str = "AutoETS", prediction_intervals=None,
        distribution: str = "normal",
    ):
        model = model.upper()
        if len(model) != 3 or any(
            c not in allowed for c, allowed in zip(model, ("AZ", "NAZ", "NAZ"))
        ):
            raise ValueError(
                "covered ETS components are A/Z error, N/A/Z trend and season"
            )
        if phi is not None and not 0.8 <= phi <= 0.98:
            raise ValueError("phi must be between 0.8 and 0.98")
        if distribution != "normal":
            raise NotImplementedError("only normal forecast errors are covered")
        if prediction_intervals is not None:
            raise NotImplementedError("conformal prediction intervals are not covered")
        self.season_length = integer(season_length, "season_length", minimum=1)
        self.model = model
        self.damped = damped
        self.phi = phi
        self.alias = alias
        self.prediction_intervals = prediction_intervals
        self.distribution = distribution

    def __repr__(self):
        return self.alias

    def fit(self, y: np.ndarray, X=None):
        reject_exog(X)
        values = series(y)
        if self.model[2] == "A" and len(values) < 2 * self.season_length:
            raise ValueError("seasonal ETS requires at least two complete seasons")
        trends = ["N", "A"] if self.model[1] == "Z" else [self.model[1]]
        seasons = ["N"]
        if self.season_length > 1 and self.model[2] == "Z":
            seasons.append("A")
        elif self.model[2] != "Z":
            seasons = [self.model[2]]
        candidates = []
        for trend in trends:
            for seasonality in seasons:
                if seasonality == "A" and len(values) < 2 * self.season_length:
                    continue
                code = "A" + trend + seasonality
                damped = bool(self.damped) and trend == "A"
                candidates.append(
                    _fit_ets(
                        values, self.season_length, code, damped, self.phi
                    )
                )
        self.model_ = min(candidates, key=lambda item: item["aicc"])
        self._y = values
        return self

    def predict(self, h: int, X=None, level=None):
        reject_exog(X)
        if not hasattr(self, "model_"):
            raise Exception("You have to use the `fit` method first")
        h = horizon(h)
        mean = np.empty(h, dtype=np.float64)
        code = self.model_["model"]
        lib().msf_ets_forecast(
            addr(self.model_["state"]), addr(mean), self.season_length, h,
            len(self._y) % self.season_length, self.model_["params"]["phi"],
            code[1] == "A", code[2] == "A",
        )
        return intervals(
            {"mean": mean}, mean, np.sqrt(self.model_["sigma2"]), level,
            np.sqrt(np.arange(1, h + 1)),
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
        if not hasattr(self, "model_"):
            raise Exception("You have to use the `fit` method first")
        clone = AutoETS(
            self.season_length, self.model_["model"], self.damped, self.phi,
            self.alias, self.prediction_intervals, self.distribution,
        )
        return clone.forecast(y, h, level=level, fitted=fitted)

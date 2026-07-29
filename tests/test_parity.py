import inspect

import numpy as np
import pytest

from mojostatsforecast.models import (
    ARIMA,
    AutoARIMA,
    AutoETS,
    AutoTheta,
    DynamicOptimizedTheta,
    DynamicTheta,
    OptimizedTheta,
    Theta,
)

upstream = pytest.importorskip("statsforecast.models")


@pytest.fixture(scope="module")
def trend_series():
    rng = np.random.default_rng(4)
    y = np.empty(300)
    y[0] = 2.0
    for i in range(1, len(y)):
        y[i] = 2.0 + 0.7 * (y[i - 1] - 2.0) + rng.normal(scale=0.2)
    return y


@pytest.fixture(scope="module")
def seasonal_series():
    rng = np.random.default_rng(3)
    n = 83
    return (
        5
        + 0.2 * np.arange(n)
        + np.resize(np.array([-1.0, 0.5, 2.0, -0.5]), n)
        + rng.normal(0, 0.05, n)
    )


@pytest.mark.parametrize("model", ["ANN", "AAN"])
def test_ets_point_forecast_parity(model):
    y = np.arange(1.0, 61.0) + 0.1 * np.sin(np.arange(60))
    ours = AutoETS(model=model).forecast(y, 8)["mean"]
    theirs = upstream.AutoETS(model=model).forecast(y, 8)["mean"]
    assert np.allclose(ours, theirs, atol=0.02)


def test_seasonal_ets_point_forecast_parity(seasonal_series):
    ours = AutoETS(4, "AAA").forecast(seasonal_series, 8)["mean"]
    theirs = upstream.AutoETS(4, "AAA").forecast(seasonal_series, 8)["mean"]
    assert np.allclose(ours, theirs, atol=0.2)


@pytest.mark.parametrize("order", [(1, 0, 0), (2, 0, 0), (1, 0, 1), (0, 1, 1)])
def test_arima_point_forecast_parity(trend_series, order):
    ours = ARIMA(order).forecast(trend_series, 8)["mean"]
    theirs = upstream.ARIMA(order, method="CSS-ML").forecast(
        trend_series, 8
    )["mean"]
    assert np.allclose(ours, theirs, atol=0.01)


def test_arima_coefficients_match_upstream(trend_series):
    ours = ARIMA((1, 0, 1)).fit(trend_series).model_["coef"]
    theirs = upstream.ARIMA((1, 0, 1), method="CSS-ML").fit(
        trend_series
    ).model_["coef"]
    assert ours["ar1"] == pytest.approx(theirs["ar1"], abs=0.005)
    assert ours["ma1"] == pytest.approx(theirs["ma1"], abs=0.005)
    assert ours["intercept"] == pytest.approx(theirs["intercept"], abs=0.005)


def test_classic_theta_point_forecast_parity():
    y = np.arange(1.0, 61.0) + 0.1 * np.sin(np.arange(60))
    ours = Theta().forecast(y, 8)["mean"]
    theirs = upstream.Theta().forecast(y, 8)["mean"]
    assert np.allclose(ours, theirs, atol=0.01)


def test_auto_theta_point_forecast_parity():
    y = np.arange(1.0, 61.0) + 0.1 * np.sin(np.arange(60))
    ours = AutoTheta().forecast(y, 8)["mean"]
    theirs = upstream.AutoTheta().forecast(y, 8)["mean"]
    assert np.allclose(ours, theirs, atol=0.03)


def test_public_signatures_cover_upstream_core_arguments():
    for ours, theirs in [
        (AutoETS, upstream.AutoETS),
        (ARIMA, upstream.ARIMA),
        (AutoARIMA, upstream.AutoARIMA),
        (AutoTheta, upstream.AutoTheta),
    ]:
        ours_parameters = set(inspect.signature(ours).parameters)
        theirs_parameters = set(inspect.signature(theirs).parameters)
        assert theirs_parameters <= ours_parameters


@pytest.mark.parametrize(
    "model",
    [
        AutoETS(model="ANN"),
        ARIMA((1, 0, 0)),
        Theta(),
        OptimizedTheta(),
        DynamicTheta(),
        DynamicOptimizedTheta(),
        AutoTheta(),
    ],
)
def test_fit_predict_and_forecast_contract(model, trend_series):
    fitted_model = model.fit(trend_series)
    prediction = fitted_model.predict(6, level=[80, 95])
    in_sample = fitted_model.predict_in_sample()
    direct = model.forecast(trend_series, 6, fitted=True)
    forwarded = model.forward(trend_series, 6)
    assert set(prediction) == {"mean", "lo-95", "lo-80", "hi-80", "hi-95"}
    assert prediction["mean"].shape == (6,)
    assert in_sample["fitted"].shape == trend_series.shape
    assert np.all(prediction["lo-95"] <= prediction["lo-80"])
    assert np.all(prediction["hi-95"] >= prediction["hi-80"])
    assert direct["fitted"].shape == trend_series.shape
    assert forwarded["mean"].shape == (6,)


def test_fixed_arima_coefficients_are_honored(trend_series):
    model = ARIMA(
        (1, 0, 1), fixed={"ar1": 0.5, "ma1": -0.2, "intercept": 2.0}
    ).fit(trend_series)
    assert model.model_["coef"] == {
        "intercept": 2.0,
        "ar1": 0.5,
        "ma1": -0.2,
    }


def test_auto_arima_returns_finite_forecast(trend_series):
    model = AutoARIMA(max_p=2, max_q=2, max_order=3).fit(trend_series)
    assert np.all(np.isfinite(model.predict(10)["mean"]))
    assert sum(model.model_["order"][::2]) <= 3


def test_covered_ets_selection_damping_and_theta_seasonality(seasonal_series):
    selected = AutoETS(4, "AZZ", damped=True).fit(seasonal_series)
    assert selected.model_["model"] in {"ANN", "AAN", "AAA"}
    assert np.all(np.isfinite(selected.predict(4)["mean"]))
    for decomposition in ("additive", "multiplicative"):
        forecast = Theta(4, decomposition).forecast(seasonal_series, 4)["mean"]
        assert np.all(np.isfinite(forecast))


def test_second_difference_arima_is_covered():
    y = np.arange(1.0, 31.0) ** 2
    forecast = ARIMA((1, 2, 1)).forecast(y, 4)["mean"]
    assert forecast.shape == (4,)
    assert np.all(np.isfinite(forecast))


def test_unsupported_features_fail_explicitly(trend_series):
    with pytest.raises(NotImplementedError):
        ARIMA((1, 0, 0), season_length=12)
    with pytest.raises(NotImplementedError):
        AutoETS().fit(trend_series, X=np.ones((len(trend_series), 1)))


@pytest.mark.parametrize("model", [AutoETS(model="ANN"), ARIMA(), Theta()])
def test_invalid_horizons_and_levels_never_reach_native_code(model, trend_series):
    fitted = model.fit(trend_series)
    for invalid in (0, -1, 1.5, True):
        with pytest.raises((TypeError, ValueError)):
            fitted.predict(invalid)
    for invalid in ([0], [100], [80, 80], ["bad"]):
        with pytest.raises(ValueError):
            fitted.predict(2, level=invalid)


def test_arima_rejects_unsafe_lag_and_silent_order_narrowing():
    with pytest.raises(TypeError):
        ARIMA((1.5, 0, 0))
    with pytest.raises(ValueError):
        ARIMA((3, 2, 0)).fit([1.0, 2.0, 3.0])

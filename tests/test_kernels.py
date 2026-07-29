import numpy as np
import pytest

from mojostatsforecast._lib import addr, lib


def reference_ets(y, initial, alpha, beta, gamma, phi, trend, seasonal, m):
    state = initial.copy()
    fitted = np.empty_like(y)
    for t, value in enumerate(y):
        base = state[0] + (phi * state[1] if trend else 0.0)
        fitted[t] = base + (state[2 + t % m] if seasonal else 0.0)
        error = value - fitted[t]
        state[0] = base + alpha * error
        if trend:
            state[1] = phi * state[1] + beta * error
        if seasonal:
            state[2 + t % m] += gamma * error
    return fitted, state


@pytest.mark.parametrize(
    "trend,seasonal,m",
    [(False, False, 1), (True, False, 1), (True, True, 4)],
)
def test_ets_filter_matches_numpy(trend, seasonal, m):
    y = np.ascontiguousarray(
        10 + 0.2 * np.arange(40) + np.resize([-1.0, 0.5, 1.0, -0.5], 40)
    )
    initial = np.zeros(m + 2)
    initial[:2] = [9.5, 0.15]
    if seasonal:
        initial[2:] = [-1.0, 0.5, 1.0, -0.5]
    expected_fit, expected_state = reference_ets(
        y, initial, 0.3, 0.05, 0.1, 0.95, trend, seasonal, m
    )
    fitted = np.empty_like(y)
    state = np.empty_like(initial)
    sse = lib().msf_ets_filter(
        addr(y), addr(fitted), addr(initial), addr(state), len(y), m,
        0.3, 0.05, 0.1, 0.95, trend, seasonal,
    )
    assert np.allclose(fitted, expected_fit)
    assert np.allclose(state, expected_state)
    assert sse == pytest.approx(np.sum((y - expected_fit) ** 2))


def test_ann_simd_tail_matches_numpy():
    rng = np.random.default_rng(8)
    y = np.ascontiguousarray(rng.normal(size=43))
    initial = np.array([0.25, 0.0, 0.0])
    expected_fit, expected_state = reference_ets(
        y, initial, 0.37, 0.0, 0.0, 1.0, False, False, 1
    )
    fitted = np.empty_like(y)
    state = np.empty_like(initial)
    sse = lib().msf_ets_filter(
        addr(y), addr(fitted), addr(initial), addr(state), len(y), 1,
        0.37, 0.0, 0.0, 1.0, 0, 0,
    )
    objective_sse = lib().msf_ets_sse_ann(addr(y), len(y), initial[0], 0.37)
    expected_sse = np.sum((y - expected_fit) ** 2)
    assert np.allclose(fitted, expected_fit)
    assert np.allclose(state, expected_state)
    assert sse == pytest.approx(expected_sse)
    assert objective_sse == pytest.approx(expected_sse)


def test_ann_sse_gradient_matches_finite_difference():
    rng = np.random.default_rng(12)
    y = np.ascontiguousarray(rng.normal(size=43))
    gradient = np.empty(2)
    level = 0.25
    alpha = 0.37
    sse = lib().msf_ets_sse_grad_ann(
        addr(y), len(y), level, alpha, addr(gradient)
    )
    epsilon = 1e-6
    alpha_low = lib().msf_ets_sse_ann(addr(y), len(y), level, alpha - epsilon)
    alpha_high = lib().msf_ets_sse_ann(addr(y), len(y), level, alpha + epsilon)
    level_low = lib().msf_ets_sse_ann(addr(y), len(y), level - epsilon, alpha)
    level_high = lib().msf_ets_sse_ann(addr(y), len(y), level + epsilon, alpha)
    assert sse == pytest.approx(
        lib().msf_ets_sse_ann(addr(y), len(y), level, alpha)
    )
    assert gradient[0] == pytest.approx(
        (alpha_high - alpha_low) / (2 * epsilon), rel=1e-6
    )
    assert gradient[1] == pytest.approx(
        (level_high - level_low) / (2 * epsilon), rel=1e-6
    )


def test_addr_rejects_buffers_that_violate_the_native_contract():
    with pytest.raises(TypeError):
        addr(np.arange(3, dtype=np.float32))
    with pytest.raises(ValueError):
        addr(np.arange(6, dtype=np.float64)[::2])
    with pytest.raises(ValueError):
        addr(np.empty(0, dtype=np.float64))
    readonly = np.arange(3, dtype=np.float64)
    readonly.flags.writeable = False
    with pytest.raises(ValueError):
        addr(readonly)


def reference_arima(y, coef, p, q, include_mean):
    residuals = np.zeros_like(y)
    fitted = y.copy()
    mean = coef[0] if include_mean else 0.0
    offset = int(include_mean)
    for t in range(max(p, q), len(y)):
        fitted[t] = mean
        for j in range(p):
            fitted[t] += coef[offset + j] * (y[t - j - 1] - mean)
        for j in range(q):
            fitted[t] += coef[offset + p + j] * residuals[t - j - 1]
        residuals[t] = y[t] - fitted[t]
    return fitted, residuals


@pytest.mark.parametrize("p,q", [(1, 0), (0, 1), (2, 1), (1, 2)])
def test_arima_recursion_matches_numpy(p, q):
    rng = np.random.default_rng(5)
    y = np.ascontiguousarray(rng.normal(size=100))
    coef = np.ascontiguousarray([0.2] + [0.15] * p + [-0.1] * q)
    expected_fit, expected_residuals = reference_arima(y, coef, p, q, True)
    fitted = np.empty_like(y)
    residuals = np.empty_like(y)
    sse = lib().msf_arima_residuals(
        addr(y), addr(coef), addr(residuals), addr(fitted), len(y), p, q, 1
    )
    assert np.allclose(fitted, expected_fit)
    assert np.allclose(residuals, expected_residuals)
    assert sse == pytest.approx(np.sum(expected_residuals[max(p, q):] ** 2))

"""Native forecasting recursions exposed through one C ABI compilation unit."""

from std.sys.info import simd_width_of

comptime Ptr = UnsafePointer[Float64, AnyOrigin[mut=True]]


def ptr(address: Int) -> Ptr:
    return Ptr(unsafe_from_address=address)


def ets_filter(
    y: Ptr, fitted: Ptr, initial: Ptr, state: Ptr, n: Int, m: Int,
    alpha: Float64, beta: Float64, gamma: Float64, phi: Float64,
    trend: Bool, seasonal: Bool,
) -> Float64:
    state[0] = initial[0]
    state[1] = initial[1]
    for j in range(m):
        state[2 + j] = initial[2 + j]
    if not trend and not seasonal:
        comptime W = simd_width_of[DType.float64]()
        var level = state[0]
        var sse = 0.0
        var vector_end = n - n % W
        for t in range(0, vector_end, W):
            var values = y.load[width=W](t)
            var predictions = values
            comptime for lane in range(W):
                predictions[lane] = level
                level += alpha * (values[lane] - level)
            fitted.store(t, predictions)
            var errors = values - predictions
            sse += (errors * errors).reduce_add()
        for t in range(vector_end, n):
            fitted[t] = level
            var error = y[t] - level
            sse += error * error
            level += alpha * error
        state[0] = level
        return sse
    var sse = 0.0
    for t in range(n):
        var slot = t % m
        var old_level = state[0]
        var old_trend = state[1]
        var base = old_level
        if trend:
            base += phi * old_trend
        var pred = base
        if seasonal:
            pred += state[2 + slot]
        fitted[t] = pred
        var error = y[t] - pred
        sse += error * error
        state[0] = base + alpha * error
        if trend:
            state[1] = phi * old_trend + beta * error
        if seasonal:
            state[2 + slot] += gamma * error
    return sse


def ets_sse_ann(y: Ptr, n: Int, initial_level: Float64, alpha: Float64) -> Float64:
    comptime W = simd_width_of[DType.float64]()
    var level = initial_level
    var sse = 0.0
    var vector_end = n - n % W
    for t in range(0, vector_end, W):
        var values = y.load[width=W](t)
        var predictions = values
        comptime for lane in range(W):
            predictions[lane] = level
            level += alpha * (values[lane] - level)
        var errors = values - predictions
        sse += (errors * errors).reduce_add()
    for t in range(vector_end, n):
        var error = y[t] - level
        sse += error * error
        level += alpha * error
    return sse


def ets_sse_grad_ann(
    y: Ptr, gradient: Ptr, n: Int, initial_level: Float64, alpha: Float64,
) -> Float64:
    comptime W = simd_width_of[DType.float64]()
    var level = initial_level
    var level_alpha = 0.0
    var level_initial = 1.0
    var gradient_alpha = 0.0
    var gradient_initial = 0.0
    var sse = 0.0
    var decay = 1.0 - alpha
    var vector_end = n - n % W
    for t in range(0, vector_end, W):
        var values = y.load[width=W](t)
        var errors = values
        var alpha_derivatives = values
        var initial_derivatives = values
        comptime for lane in range(W):
            errors[lane] = values[lane] - level
            alpha_derivatives[lane] = level_alpha
            initial_derivatives[lane] = level_initial
            level_alpha = decay * level_alpha + errors[lane]
            level_initial *= decay
            level += alpha * errors[lane]
        sse += (errors * errors).reduce_add()
        gradient_alpha -= 2.0 * (errors * alpha_derivatives).reduce_add()
        gradient_initial -= 2.0 * (errors * initial_derivatives).reduce_add()
    for t in range(vector_end, n):
        var error = y[t] - level
        sse += error * error
        gradient_alpha -= 2.0 * error * level_alpha
        gradient_initial -= 2.0 * error * level_initial
        level_alpha = decay * level_alpha + error
        level_initial *= decay
        level += alpha * error
    gradient[0] = gradient_alpha
    gradient[1] = gradient_initial
    return sse


def ets_forecast(
    state: Ptr, destination: Ptr, m: Int, h: Int, phase: Int, phi: Float64,
    trend: Bool, seasonal: Bool,
):
    var phi_sum = 0.0
    var phi_power = 1.0
    for step in range(h):
        if trend:
            phi_power *= phi
            phi_sum += phi_power
        var value = state[0]
        if trend:
            value += phi_sum * state[1]
        if seasonal:
            value += state[2 + ((phase + step) % m)]
        destination[step] = value


def arima_residuals(
    y: Ptr, coefficients: Ptr, residuals: Ptr, fitted: Ptr,
    n: Int, p: Int, q: Int, include_mean: Bool,
) -> Float64:
    var start = p if p > q else q
    var mean = coefficients[0] if include_mean else 0.0
    var offset = 1 if include_mean else 0
    for t in range(n):
        residuals[t] = 0.0
        fitted[t] = y[t]
    var sse = 0.0
    for t in range(start, n):
        var pred = mean
        for j in range(p):
            pred += coefficients[offset + j] * (y[t - j - 1] - mean)
        for j in range(q):
            pred += coefficients[offset + p + j] * residuals[t - j - 1]
        fitted[t] = pred
        var error = y[t] - pred
        residuals[t] = error
        sse += error * error
    return sse


def arima_forecast(
    history: Ptr, residuals: Ptr, coefficients: Ptr, destination: Ptr,
    n: Int, p: Int, q: Int, h: Int, include_mean: Bool,
):
    var mean = coefficients[0] if include_mean else 0.0
    var offset = 1 if include_mean else 0
    for step in range(h):
        var pred = mean
        for j in range(p):
            var index = n + step - j - 1
            var lagged = history[index] if index < n else destination[index - n]
            pred += coefficients[offset + j] * (lagged - mean)
        for j in range(q):
            var index = n + step - j - 1
            if index < n:
                pred += coefficients[offset + p + j] * residuals[index]
        destination[step] = pred


@export("msf_ets_filter")
def msf_ets_filter(
    y: Int, fitted: Int, initial: Int, state: Int, n: Int, m: Int,
    alpha: Float64, beta: Float64, gamma: Float64, phi: Float64,
    trend: Int, seasonal: Int,
) abi("C") -> Float64:
    return ets_filter(
        ptr(y), ptr(fitted), ptr(initial), ptr(state), n, m,
        alpha, beta, gamma, phi, trend != 0, seasonal != 0,
    )


@export("msf_ets_sse_ann")
def msf_ets_sse_ann(
    y: Int, n: Int, initial_level: Float64, alpha: Float64,
) abi("C") -> Float64:
    return ets_sse_ann(ptr(y), n, initial_level, alpha)


@export("msf_ets_sse_grad_ann")
def msf_ets_sse_grad_ann(
    y: Int, n: Int, initial_level: Float64, alpha: Float64, gradient: Int,
) abi("C") -> Float64:
    return ets_sse_grad_ann(ptr(y), ptr(gradient), n, initial_level, alpha)


@export("msf_ets_forecast")
def msf_ets_forecast(
    state: Int, destination: Int, m: Int, h: Int, phase: Int, phi: Float64,
    trend: Int, seasonal: Int,
) abi("C"):
    ets_forecast(
        ptr(state), ptr(destination), m, h, phase, phi,
        trend != 0, seasonal != 0
    )


@export("msf_arima_residuals")
def msf_arima_residuals(
    y: Int, coefficients: Int, residuals: Int, fitted: Int,
    n: Int, p: Int, q: Int, include_mean: Int,
) abi("C") -> Float64:
    return arima_residuals(
        ptr(y), ptr(coefficients), ptr(residuals), ptr(fitted),
        n, p, q, include_mean != 0,
    )


@export("msf_arima_forecast")
def msf_arima_forecast(
    history: Int, residuals: Int, coefficients: Int, destination: Int,
    n: Int, p: Int, q: Int, h: Int, include_mean: Int,
) abi("C"):
    arima_forecast(
        ptr(history), ptr(residuals), ptr(coefficients), ptr(destination),
        n, p, q, h, include_mean != 0,
    )

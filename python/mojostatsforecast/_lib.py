"""ctypes bridge to the Mojo forecasting kernels."""

from __future__ import annotations

import ctypes
import os
import subprocess

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
LIB = os.environ.get("MOJOSTATSFORECAST_LIB") or os.path.join(
    ROOT, "dist", "libmojo-statsforecast.so"
)
I = ctypes.c_int64
F = ctypes.c_double

_SIGNATURES = {
    "msf_ets_filter": ([I, I, I, I, I, I, F, F, F, F, I, I], F),
    "msf_ets_sse_ann": ([I, I, F, F], F),
    "msf_ets_sse_grad_ann": ([I, I, F, F, I], F),
    "msf_ets_forecast": ([I, I, I, I, I, F, I, I], None),
    "msf_arima_residuals": ([I, I, I, I, I, I, I, I], F),
    "msf_arima_forecast": ([I, I, I, I, I, I, I, I, I], None),
}

_library: ctypes.CDLL | None = None


def build() -> str:
    source = os.path.join(ROOT, "src", "kernels.mojo")
    stale = not os.path.exists(LIB) or os.path.getmtime(source) > os.path.getmtime(LIB)
    if not os.environ.get("MOJOSTATSFORECAST_LIB") and stale:
        subprocess.run(
            ["bash", os.path.join(ROOT, "build", "build.sh")],
            cwd=ROOT,
            check=True,
        )
    if not os.path.exists(LIB):
        raise FileNotFoundError(
            f"shared library not found at {LIB}; run `pixi run build`"
        )
    return LIB


def lib() -> ctypes.CDLL:
    global _library
    if _library is None:
        _library = ctypes.CDLL(build())
        for name, (argtypes, restype) in _SIGNATURES.items():
            fn = getattr(_library, name)
            fn.argtypes = argtypes
            fn.restype = restype
    return _library


def addr(values: np.ndarray) -> int:
    if not isinstance(values, np.ndarray):
        raise TypeError("native buffers must be NumPy arrays")
    if values.dtype != np.float64:
        raise TypeError("native buffers must have dtype float64")
    if values.ndim != 1 or not values.flags.c_contiguous:
        raise ValueError("native buffers must be one-dimensional and C-contiguous")
    if values.size == 0:
        raise ValueError("native buffers must not be empty")
    if not values.flags.writeable:
        raise ValueError("native buffers must be writeable")
    address = int(values.ctypes.data)
    if address == 0:
        raise ValueError("native buffer has a null address")
    if address % values.dtype.alignment:
        raise ValueError("native buffer is not correctly aligned")
    return address

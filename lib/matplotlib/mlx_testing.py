"""Testing helpers backed by MLX."""
from __future__ import annotations

import math

import mlx.core as mx

def _array(value):
    return value if isinstance(value, mx.array) else mx.array(value)


def _stream_for(*arrays):
    return mx.cpu


def assert_allclose(actual, desired, rtol=1e-7, atol=0, **kwargs):
    actual = _array(actual)
    desired = _array(desired)
    stream = _stream_for(actual, desired)
    if not bool(mx.allclose(actual, desired, rtol=rtol, atol=atol,
                            equal_nan=True, stream=stream).item()):
        raise AssertionError(f"arrays are not close: {actual} != {desired}")


def assert_array_equal(actual, desired, **kwargs):
    actual = _array(actual)
    desired = _array(desired)
    stream = _stream_for(actual, desired)
    if not bool(mx.array_equal(actual, desired, stream=stream).item()):
        raise AssertionError(f"arrays are not equal: {actual} != {desired}")


def assert_array_less(actual, desired, **kwargs):
    actual = _array(actual)
    desired = _array(desired)
    stream = _stream_for(actual, desired)
    if not bool(mx.all(actual < desired, stream=stream).item()):
        raise AssertionError(f"array values are not all less: {actual} >= {desired}")


def assert_array_almost_equal(actual, desired, decimal=6, **kwargs):
    assert_allclose(actual, desired, rtol=10 ** -decimal, atol=10 ** -decimal)


def assert_array_almost_equal_nulp(actual, desired, *args, **kwargs):
    assert_array_almost_equal(actual, desired, **kwargs)


def assert_approx_equal(actual, desired, significant=7, **kwargs):
    actual = float(_array(actual))
    desired = float(_array(desired))
    if actual == desired:
        return
    if not math.isfinite(actual) or not math.isfinite(desired):
        if math.isnan(actual) and math.isnan(desired):
            return
        raise AssertionError(f"items are not equal: {actual} != {desired}")
    scale = 0.5 * (abs(desired) + abs(actual))
    scale = 10 ** math.floor(math.log10(scale)) if scale else 1.0
    if abs(desired / scale - actual / scale) >= 10 ** -(significant - 1):
        raise AssertionError(
            f"items are not equal to {significant} significant digits: "
            f"{actual} != {desired}"
        )


assert_almost_equal = assert_array_almost_equal

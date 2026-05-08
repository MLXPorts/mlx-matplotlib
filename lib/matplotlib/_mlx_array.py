"""MLX-backed array helpers.

This module provides a focused array API implemented on top of MLX.
It is intentionally incomplete but covers the subset used by this codebase.
"""
from __future__ import annotations

import math
import itertools
import operator
import builtins as _builtins
import gc
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any, Iterable, Iterator, List, Sequence, Tuple

import mlx.core as mx
from matplotlib import _mlx_overrides as _mx_overrides

_ORIGINAL_MX_ARRAY = mx.array
_ORIGINAL_MX_EVAL = mx.eval
_ORIGINAL_MX_STREAM = mx.stream
_ORIGINAL_MX_ABS = mx.abs
_ORIGINAL_MX_SQUARE = mx.square
_ORIGINAL_MX_ADD = mx.add
_ORIGINAL_MX_SUBTRACT = mx.subtract
_ORIGINAL_MX_MULTIPLY = mx.multiply
_ORIGINAL_MX_DIVIDE = mx.divide
_ORIGINAL_MX_POWER = mx.power
_ORIGINAL_MX_REMAINDER = mx.remainder
_ORIGINAL_MX_DIVMOD = mx.divmod
_ORIGINAL_MX_MATMUL = mx.matmul
_ORIGINAL_MX_ARCTAN2 = mx.arctan2
_ORIGINAL_MX_LOG = mx.log
_ORIGINAL_MX_LOG2 = mx.log2
_ORIGINAL_MX_LOG10 = mx.log10
_ORIGINAL_MX_SIN = mx.sin
_ORIGINAL_MX_COS = mx.cos
_ORIGINAL_MX_ARCSIN = mx.arcsin
_ORIGINAL_MX_ARCCOS = mx.arccos
_ORIGINAL_MX_ARCTAN = mx.arctan
_ORIGINAL_MX_FLOOR = mx.floor
_ORIGINAL_MX_CEIL = mx.ceil
_ORIGINAL_MX_SQRT = mx.sqrt
_ORIGINAL_MX_DEGREES = mx.degrees
_ORIGINAL_MX_RADIANS = mx.radians
_ORIGINAL_MX_SORT = mx.sort
_ORIGINAL_MX_ARGSORT = mx.argsort
_ORIGINAL_MX_WHERE = mx.where
_ORIGINAL_MX_EQUAL = mx.equal
_ORIGINAL_MX_NOT_EQUAL = mx.not_equal
_ORIGINAL_MX_LESS = mx.less
_ORIGINAL_MX_LESS_EQUAL = mx.less_equal
_ORIGINAL_MX_GREATER = mx.greater
_ORIGINAL_MX_GREATER_EQUAL = mx.greater_equal
_ORIGINAL_MX_LOGICAL_NOT = mx.logical_not
_ORIGINAL_MX_LOGICAL_AND = mx.logical_and
_ORIGINAL_MX_LOGICAL_OR = mx.logical_or
_ORIGINAL_MX_CONCATENATE = mx.concatenate
_ORIGINAL_MX_STACK = mx.stack
_ORIGINAL_MX_ISFINITE = mx.isfinite
_ORIGINAL_MX_ISNAN = mx.isnan
_ORIGINAL_MX_ISINF = mx.isinf
_ORIGINAL_MX_ISCLOSE = mx.isclose
_ORIGINAL_MX_ALLCLOSE = mx.allclose
_ORIGINAL_MX_ROUND = mx.round
_ORIGINAL_MX_SUM = mx.sum
_ORIGINAL_MX_MEAN = mx.mean
_ORIGINAL_MX_MIN = mx.min
_ORIGINAL_MX_MAX = mx.max
_ORIGINAL_MX_MINIMUM = mx.minimum
_ORIGINAL_MX_MAXIMUM = mx.maximum
_ORIGINAL_MX_CUMSUM = mx.cumsum
_ORIGINAL_MX_TAKE = mx.take
_ORIGINAL_MX_SLICE = mx.slice
_ORIGINAL_MX_ROLL = mx.roll
_ORIGINAL_MX_CLIP = mx.clip
_ORIGINAL_MX_BROADCAST_ARRAYS = mx.broadcast_arrays
_ORIGINAL_MX_BROADCAST_TO = mx.broadcast_to
_ORIGINAL_MX_PAD = mx.pad
_ORIGINAL_MX_EYE = mx.eye
_ORIGINAL_MX_IDENTITY = mx.identity
_ORIGINAL_MX_ZEROS = mx.zeros
_ORIGINAL_MX_ONES = mx.ones
_ORIGINAL_MX_FULL = mx.full
_ORIGINAL_MX_ARANGE = mx.arange
_ORIGINAL_MX_LINSPACE = mx.linspace
_ORIGINAL_MX_RESHAPE = mx.reshape
_ORIGINAL_MX_FLATTEN = getattr(mx, "flatten", None)
_ORIGINAL_MX_SQUEEZE = mx.squeeze
_ORIGINAL_MX_EXPAND_DIMS = mx.expand_dims
_ORIGINAL_MX_TRANSPOSE = mx.transpose
_ORIGINAL_MX_SWAPAXES = mx.swapaxes
_ORIGINAL_MX_MOVEAXIS = mx.moveaxis
_ORIGINAL_MX_REPEAT = mx.repeat
_ORIGINAL_MX_TILE = mx.tile
_ORIGINAL_MX_CONV1D = mx.conv1d
_ORIGINAL_MX_DEFAULT_STREAM = mx.default_stream
_ORIGINAL_MX_SET_DEFAULT_STREAM = mx.set_default_stream
_ORIGINAL_MX_NEW_STREAM = mx.new_stream
_ORIGINAL_MX_DEFAULT_DEVICE = mx.default_device
_ORIGINAL_MX_SET_DEFAULT_DEVICE = mx.set_default_device
_PRECISE_STREAM_STACK: list[Any] = []


def _active_stream() -> Any | None:
    if _PRECISE_STREAM_STACK:
        return _PRECISE_STREAM_STACK[-1]
    return _ORIGINAL_MX_DEFAULT_DEVICE()


def _precise_dtype(dtype: Any | None) -> Any | None:
    return _unwrap_dtype(dtype)


def _is_structured_dtype(dtype: Any | None) -> bool:
    return (isinstance(dtype, (list, tuple))
            and _builtins.all(
                isinstance(field, (list, tuple))
                and len(field) >= 2
                and isinstance(field[0], str)
                for field in dtype))


class _StructuredDType:
    def __init__(self, fields: Sequence[Sequence[Any]]):
        self.names = tuple(field[0] for field in fields)
        self.fields = {
            field[0]: (field[1], index)
            for index, field in enumerate(fields)
        }


class _StructuredArray:
    def __init__(self, values: Iterable[Any], dtype: Sequence[Sequence[Any]]):
        self._rows = [tuple(row) for row in values]
        self.dtype = _StructuredDType(dtype)

    @property
    def shape(self) -> Tuple[int]:
        return (len(self._rows),)

    @property
    def ndim(self) -> int:
        return 1

    @property
    def size(self) -> int:
        return len(self._rows)

    def __len__(self) -> int:
        return len(self._rows)

    def __iter__(self):
        return iter(self._rows)

    def __getitem__(self, key: Any):
        if isinstance(key, str):
            field_dtype, index = self.dtype.fields[key]
            return _construct_mx_array(
                [row[index] for row in self._rows],
                dtype=field_dtype)
        if isinstance(key, _ORIGINAL_MX_ARRAY):
            key = key.tolist()
        if isinstance(key, (list, tuple)):
            return _StructuredArray(
                [self._rows[int(index)] for index in key],
                [(name, self.dtype.fields[name][0])
                 for name in self.dtype.names])
        if isinstance(key, slice):
            return _StructuredArray(
                self._rows[key],
                [(name, self.dtype.fields[name][0])
                 for name in self.dtype.names])
        return self._rows[key]

    def tolist(self) -> list[tuple[Any, ...]]:
        return list(self._rows)


def _contains_python_float(value: Any) -> bool:
    if isinstance(value, _ORIGINAL_MX_ARRAY):
        return False
    if isinstance(value, float):
        return True
    if isinstance(value, (list, tuple)):
        return _builtins.any(_contains_python_float(item) for item in value)
    return False


def _construct_mx_array(obj: Any, dtype: Any | None = None, *,
                        stream: Any | None = None, **kwargs: Any) -> mx.array:
    if _is_structured_dtype(dtype):
        return _StructuredArray(obj, dtype)
    target_dtype = _precise_dtype(dtype)
    if kwargs:
        bad = next(iter(kwargs))
        raise TypeError(f"array() got an unexpected keyword argument {bad!r}")
    if target_dtype is None and _contains_python_float(obj):
        target_dtype = mx.float64
    return _mx_overrides.MlxPreciseArray(
        obj, dtype=target_dtype, stream=stream if stream is not None else _active_stream())


def _eval_mx_arrays(*arrays: Any) -> None:
    precise = [a for a in arrays
               if isinstance(a, _ORIGINAL_MX_ARRAY) and a.dtype == mx.float64]
    regular = [a for a in arrays
               if not (isinstance(a, _ORIGINAL_MX_ARRAY) and a.dtype == mx.float64)]
    for array in precise:
        _mx_overrides.eval_precise_array(array)
    if regular:
        _ORIGINAL_MX_EVAL(*regular)


class _PreciseStreamContext:
    def __init__(self, stream_or_device: Any):
        self._stream_or_device = stream_or_device
        self._context = _ORIGINAL_MX_STREAM(stream_or_device)

    def __enter__(self):
        result = self._context.__enter__()
        _PRECISE_STREAM_STACK.append(self._stream_or_device)
        return result

    def __exit__(self, exc_type, exc, tb):
        try:
            return self._context.__exit__(exc_type, exc, tb)
        finally:
            _PRECISE_STREAM_STACK.pop()


def _precise_stream(stream_or_device: Any):
    return _PreciseStreamContext(stream_or_device)


def _has_precise_array(value: Any) -> bool:
    return isinstance(value, _ORIGINAL_MX_ARRAY)


def _has_precise_float64(value: Any) -> bool:
    return (isinstance(value, _ORIGINAL_MX_ARRAY)
            and getattr(value, "dtype", None) == mx.float64)


def _needs_precise_scalar_route(value: Any) -> bool:
    return _has_precise_array(value) or _contains_python_float(value)


def _shadow_stream(stream: Any | None) -> Any | None:
    return stream if stream is not None else _active_stream()


def _array_stream(value: Any) -> Any | None:
    return getattr(value, "_mlx_stream", None) or _active_stream()


def _abs_shadow(value: Any, *args: Any,
                stream: Any | None = None, **kwargs: Any) -> Any:
    if args or kwargs or not _has_precise_array(value):
        if stream is None:
            return _ORIGINAL_MX_ABS(value, *args, **kwargs)
        return _ORIGINAL_MX_ABS(value, *args, stream=stream, **kwargs)
    return _mx_overrides.abs_precise(value, stream=_shadow_stream(stream))


def _square_shadow(value: Any, *args: Any,
                   stream: Any | None = None, **kwargs: Any) -> Any:
    if args or kwargs or not _has_precise_array(value):
        if stream is None:
            return _ORIGINAL_MX_SQUARE(value, *args, **kwargs)
        return _ORIGINAL_MX_SQUARE(value, *args, stream=stream, **kwargs)
    return _multiply_shadow(value, value, stream=stream)


def _shadow_binary(original: Any, precise: Any, left: Any, right: Any,
                   *args: Any, stream: Any | None = None, **kwargs: Any) -> Any:
    if args or kwargs or not (
            _needs_precise_scalar_route(left)
            or _needs_precise_scalar_route(right)):
        if stream is None:
            return original(left, right, *args, **kwargs)
        return original(left, right, *args, stream=stream, **kwargs)
    return precise(left, right, stream=_shadow_stream(stream))


def _add_shadow(left: Any, right: Any, *args: Any,
                stream: Any | None = None, **kwargs: Any) -> Any:
    return _shadow_binary(_ORIGINAL_MX_ADD, _mx_overrides.add_precise,
                          left, right, *args, stream=stream, **kwargs)


def _subtract_shadow(left: Any, right: Any, *args: Any,
                     stream: Any | None = None, **kwargs: Any) -> Any:
    return _shadow_binary(_ORIGINAL_MX_SUBTRACT, _mx_overrides.subtract_precise,
                          left, right, *args, stream=stream, **kwargs)


def _multiply_shadow(left: Any, right: Any, *args: Any,
                     stream: Any | None = None, **kwargs: Any) -> Any:
    return _shadow_binary(_ORIGINAL_MX_MULTIPLY, _mx_overrides.multiply_precise,
                          left, right, *args, stream=stream, **kwargs)


def _divide_shadow(left: Any, right: Any, *args: Any,
                   stream: Any | None = None, **kwargs: Any) -> Any:
    return _shadow_binary(_ORIGINAL_MX_DIVIDE, _mx_overrides.divide_precise,
                          left, right, *args, stream=stream, **kwargs)


def _power_shadow(left: Any, right: Any, *args: Any,
                  stream: Any | None = None, **kwargs: Any) -> Any:
    return _shadow_binary(_ORIGINAL_MX_POWER, _mx_overrides.power_precise,
                          left, right, *args, stream=stream, **kwargs)


def _remainder_shadow(left: Any, right: Any, *args: Any,
                      stream: Any | None = None, **kwargs: Any) -> Any:
    return _shadow_binary(_ORIGINAL_MX_REMAINDER, _mx_overrides.remainder_precise,
                          left, right, *args, stream=stream, **kwargs)


def _matmul_shadow(left: Any, right: Any, *args: Any,
                   stream: Any | None = None, **kwargs: Any) -> Any:
    return _shadow_binary(_ORIGINAL_MX_MATMUL, _mx_overrides.matmul_precise,
                          left, right, *args, stream=stream, **kwargs)


def _arctan2_shadow(left: Any, right: Any, *args: Any,
                    stream: Any | None = None, **kwargs: Any) -> Any:
    return _shadow_binary(_ORIGINAL_MX_ARCTAN2,
                          _mx_overrides.arctan2_precise,
                          left, right, *args, stream=stream, **kwargs)


def _where_shadow(condition: Any, x: Any, y: Any, *args: Any,
                  stream: Any | None = None, **kwargs: Any) -> Any:
    if args or kwargs or not (
            _needs_precise_scalar_route(x) or _needs_precise_scalar_route(y)):
        if stream is None:
            return _ORIGINAL_MX_WHERE(condition, x, y, *args, **kwargs)
        return _ORIGINAL_MX_WHERE(
            condition, x, y, *args, stream=stream, **kwargs)
    return _mx_overrides.where_precise(
        condition, x, y, stream=_shadow_stream(stream))


def _equal_shadow(left: Any, right: Any, *args: Any,
                  stream: Any | None = None, **kwargs: Any) -> Any:
    return _shadow_binary(_ORIGINAL_MX_EQUAL, _mx_overrides.equal_precise,
                          left, right, *args, stream=stream, **kwargs)


def _not_equal_shadow(left: Any, right: Any, *args: Any,
                      stream: Any | None = None, **kwargs: Any) -> Any:
    return _shadow_binary(_ORIGINAL_MX_NOT_EQUAL,
                          _mx_overrides.not_equal_precise,
                          left, right, *args, stream=stream, **kwargs)


def _less_shadow(left: Any, right: Any, *args: Any,
                 stream: Any | None = None, **kwargs: Any) -> Any:
    return _shadow_binary(_ORIGINAL_MX_LESS, _mx_overrides.less_precise,
                          left, right, *args, stream=stream, **kwargs)


def _less_equal_shadow(left: Any, right: Any, *args: Any,
                       stream: Any | None = None, **kwargs: Any) -> Any:
    return _shadow_binary(_ORIGINAL_MX_LESS_EQUAL,
                          _mx_overrides.less_equal_precise,
                          left, right, *args, stream=stream, **kwargs)


def _greater_shadow(left: Any, right: Any, *args: Any,
                    stream: Any | None = None, **kwargs: Any) -> Any:
    if args or kwargs or not (_has_precise_array(left) or _has_precise_array(right)):
        if stream is None:
            return _ORIGINAL_MX_GREATER(left, right, *args, **kwargs)
        return _ORIGINAL_MX_GREATER(
            left, right, *args, stream=stream, **kwargs)
    return _mx_overrides.less_precise(
        right, left, stream=_shadow_stream(stream))


def _greater_equal_shadow(left: Any, right: Any, *args: Any,
                          stream: Any | None = None, **kwargs: Any) -> Any:
    if args or kwargs or not (_has_precise_array(left) or _has_precise_array(right)):
        if stream is None:
            return _ORIGINAL_MX_GREATER_EQUAL(left, right, *args, **kwargs)
        return _ORIGINAL_MX_GREATER_EQUAL(
            left, right, *args, stream=stream, **kwargs)
    return _mx_overrides.less_equal_precise(
        right, left, stream=_shadow_stream(stream))


def _logical_not_shadow(value: Any, *args: Any,
                        stream: Any | None = None, **kwargs: Any) -> Any:
    if args or kwargs or not _needs_precise_scalar_route(value):
        if stream is None:
            return _ORIGINAL_MX_LOGICAL_NOT(value, *args, **kwargs)
        return _ORIGINAL_MX_LOGICAL_NOT(value, *args, stream=stream, **kwargs)
    return _mx_overrides.logical_not_precise(
        value, stream=_shadow_stream(stream))


def _logical_binary_shadow(original: Any, precise: Any,
                           left: Any, right: Any, *args: Any,
                           stream: Any | None = None,
                           **kwargs: Any) -> Any:
    if args or kwargs or not (
            _needs_precise_scalar_route(left)
            or _needs_precise_scalar_route(right)):
        if stream is None:
            return original(left, right, *args, **kwargs)
        return original(left, right, *args, stream=stream, **kwargs)
    return precise(left, right, stream=_shadow_stream(stream))


def _logical_and_shadow(left: Any, right: Any, *args: Any,
                        stream: Any | None = None, **kwargs: Any) -> Any:
    return _logical_binary_shadow(
        _ORIGINAL_MX_LOGICAL_AND,
        _mx_overrides.logical_and_precise,
        left, right, *args, stream=stream, **kwargs)


def _logical_or_shadow(left: Any, right: Any, *args: Any,
                       stream: Any | None = None, **kwargs: Any) -> Any:
    return _logical_binary_shadow(
        _ORIGINAL_MX_LOGICAL_OR,
        _mx_overrides.logical_or_precise,
        left, right, *args, stream=stream, **kwargs)


def _minimum_shadow(left: Any, right: Any, *args: Any,
                    stream: Any | None = None, **kwargs: Any) -> Any:
    if args or kwargs or not (_has_precise_float64(left) or _has_precise_float64(right)):
        if stream is None:
            return _ORIGINAL_MX_MINIMUM(left, right, *args, **kwargs)
        return _ORIGINAL_MX_MINIMUM(left, right, *args, stream=stream, **kwargs)
    actual_stream = _shadow_stream(stream)
    return _where_shadow(
        _less_equal_shadow(left, right, stream=actual_stream),
        left, right, stream=actual_stream)


def _maximum_shadow(left: Any, right: Any, *args: Any,
                    stream: Any | None = None, **kwargs: Any) -> Any:
    if args or kwargs or not (_has_precise_float64(left) or _has_precise_float64(right)):
        if stream is None:
            return _ORIGINAL_MX_MAXIMUM(left, right, *args, **kwargs)
        return _ORIGINAL_MX_MAXIMUM(left, right, *args, stream=stream, **kwargs)
    actual_stream = _shadow_stream(stream)
    return _where_shadow(
        _greater_equal_shadow(left, right, stream=actual_stream),
        left, right, stream=actual_stream)


def _concatenate_shadow(arrays: Any, axis: int = 0, *args: Any,
                        stream: Any | None = None, **kwargs: Any) -> Any:
    out = kwargs.pop("out", None)
    if args or kwargs:
        if stream is None:
            return _ORIGINAL_MX_CONCATENATE(arrays, axis=axis, *args, **kwargs)
        return _ORIGINAL_MX_CONCATENATE(
            arrays, axis=axis, *args, stream=stream, **kwargs)
    result = _mx_overrides.concatenate_precise(
        arrays, axis=axis, stream=_shadow_stream(stream))
    if out is not None:
        out[:] = result
        return out
    return result


def _stack_shadow(arrays: Any, axis: int = 0, *args: Any,
                  stream: Any | None = None, **kwargs: Any) -> Any:
    if args or kwargs:
        if stream is None:
            return _ORIGINAL_MX_STACK(arrays, axis=axis, *args, **kwargs)
        return _ORIGINAL_MX_STACK(
            arrays, axis=axis, *args, stream=stream, **kwargs)
    return _mx_overrides.stack_precise(
        arrays, axis=axis, stream=_shadow_stream(stream))


def _isfinite_shadow(value: Any, *args: Any,
                     stream: Any | None = None, **kwargs: Any) -> Any:
    if not _has_precise_array(value):
        value = mx.array(value)
    if args or kwargs:
        if stream is None:
            return _ORIGINAL_MX_ISFINITE(value, *args, **kwargs)
        return _ORIGINAL_MX_ISFINITE(value, *args, stream=stream, **kwargs)
    return _mx_overrides.isfinite_precise(value, stream=_shadow_stream(stream))


def _isnan_shadow(value: Any, *args: Any,
                  stream: Any | None = None, **kwargs: Any) -> Any:
    if not _has_precise_array(value):
        value = mx.array(value)
    if args or kwargs or not _has_precise_array(value):
        if stream is None:
            return _ORIGINAL_MX_ISNAN(value, *args, **kwargs)
        return _ORIGINAL_MX_ISNAN(value, *args, stream=stream, **kwargs)
    actual_stream = _shadow_stream(stream)
    return _not_equal_shadow(value, value, stream=actual_stream)


def _isinf_shadow(value: Any, *args: Any,
                  stream: Any | None = None, **kwargs: Any) -> Any:
    if not _has_precise_array(value):
        value = mx.array(value)
    if args or kwargs or not _has_precise_array(value):
        if stream is None:
            return _ORIGINAL_MX_ISINF(value, *args, **kwargs)
        return _ORIGINAL_MX_ISINF(value, *args, stream=stream, **kwargs)
    actual_stream = _shadow_stream(stream)
    return mx.logical_and(
        mx.logical_not(_isfinite_shadow(value, stream=actual_stream)),
        mx.logical_not(_isnan_shadow(value, stream=actual_stream)),
        stream=actual_stream)


def _isclose_shadow(left: Any, right: Any, *args: Any,
                    rtol: float = 1e-5, atol: Any = 1e-8,
                    equal_nan: bool = False,
                    stream: Any | None = None, **kwargs: Any) -> Any:
    if not _has_precise_array(left):
        left = mx.array(left)
    if not _has_precise_array(right):
        right = mx.array(right)
    if not _has_precise_array(atol):
        atol = mx.array(atol)
    if (args or kwargs or equal_nan
            or not (_has_precise_float64(left)
                    or _has_precise_float64(right)
                    or _has_precise_float64(atol))):
        if stream is None:
            return _ORIGINAL_MX_ISCLOSE(
                left, right, *args, rtol=rtol, atol=atol,
                equal_nan=equal_nan, **kwargs)
        return _ORIGINAL_MX_ISCLOSE(
            left, right, *args, rtol=rtol, atol=atol,
            equal_nan=equal_nan, stream=stream, **kwargs)
    actual_stream = _shadow_stream(stream)
    diff = _abs_shadow(_subtract_shadow(left, right, stream=actual_stream),
                       stream=actual_stream)
    tol = _add_shadow(
        atol,
        _multiply_shadow(rtol, _abs_shadow(right, stream=actual_stream),
                         stream=actual_stream),
        stream=actual_stream)
    return _less_equal_shadow(diff, tol, stream=actual_stream)


def _allclose_shadow(left: Any, right: Any, *args: Any,
                     rtol: float = 1e-5, atol: float = 1e-8,
                     equal_nan: bool = False,
                     stream: Any | None = None, **kwargs: Any) -> Any:
    if not _has_precise_array(left):
        left = mx.array(left)
    if not _has_precise_array(right):
        right = mx.array(right)
    if args or kwargs or not (_has_precise_array(left) or _has_precise_array(right)):
        if stream is None:
            return _ORIGINAL_MX_ALLCLOSE(
                left, right, rtol=rtol, atol=atol, equal_nan=equal_nan, **kwargs)
        return _ORIGINAL_MX_ALLCLOSE(
            left, right, rtol=rtol, atol=atol, equal_nan=equal_nan,
            stream=stream, **kwargs)
    return _mx_overrides.allclose_precise(
        left, right, rtol=rtol, atol=atol, equal_nan=equal_nan,
        stream=_shadow_stream(stream))


def _round_shadow(value: Any, *args: Any, decimals: int = 0,
                  stream: Any | None = None, **kwargs: Any) -> Any:
    if args or kwargs or not _has_precise_float64(value):
        if stream is None:
            return _ORIGINAL_MX_ROUND(
                value, *args, decimals=decimals, **kwargs)
        return _ORIGINAL_MX_ROUND(
            value, *args, decimals=decimals, stream=stream, **kwargs)
    return _mx_overrides.round_precise(
        value, decimals=decimals, stream=_shadow_stream(stream))


def _sum_shadow(value: Any, axis: Any | None = None, *args: Any,
                keepdims: bool = False, stream: Any | None = None,
                **kwargs: Any) -> Any:
    if not _has_precise_array(value):
        value = mx.array(value)
    if args or kwargs or not _has_precise_array(value) or getattr(
            value, "dtype", None) != mx.float64:
        if stream is None:
            return _ORIGINAL_MX_SUM(
                value, *args, axis=axis, keepdims=keepdims, **kwargs)
        return _ORIGINAL_MX_SUM(
            value, *args, axis=axis, keepdims=keepdims, stream=stream, **kwargs)
    return _mx_overrides.sum_float64(
        value, axis=axis, keepdims=keepdims, stream=_shadow_stream(stream))


def _mean_shadow(value: Any, axis: Any | None = None, *args: Any,
                 keepdims: bool = False, stream: Any | None = None,
                 **kwargs: Any) -> Any:
    if not _has_precise_array(value):
        value = mx.array(value)
    if args or kwargs or not _has_precise_array(value) or getattr(
            value, "dtype", None) != mx.float64:
        if stream is None:
            return _ORIGINAL_MX_MEAN(
                value, *args, axis=axis, keepdims=keepdims, **kwargs)
        return _ORIGINAL_MX_MEAN(
            value, *args, axis=axis, keepdims=keepdims, stream=stream, **kwargs)
    return _mx_overrides.mean_float64(
        value, axis=axis, keepdims=keepdims, stream=_shadow_stream(stream))


def _min_shadow(value: Any, axis: Any | None = None, *args: Any,
                stream: Any | None = None, **kwargs: Any) -> Any:
    if args or kwargs or not _has_precise_array(value):
        if stream is None:
            return _ORIGINAL_MX_MIN(value, axis=axis, *args, **kwargs)
        return _ORIGINAL_MX_MIN(value, axis=axis, *args, stream=stream, **kwargs)
    return _mx_overrides.reduce_minmax_precise(
        value, axis=axis, is_max=False, stream=_shadow_stream(stream))


def _max_shadow(value: Any, axis: Any | None = None, *args: Any,
                stream: Any | None = None, **kwargs: Any) -> Any:
    if args or kwargs or not _has_precise_array(value):
        if stream is None:
            return _ORIGINAL_MX_MAX(value, axis=axis, *args, **kwargs)
        return _ORIGINAL_MX_MAX(value, axis=axis, *args, stream=stream, **kwargs)
    return _mx_overrides.reduce_minmax_precise(
        value, axis=axis, is_max=True, stream=_shadow_stream(stream))


def _cumsum_shadow(value: Any, axis: Any | None = None, *args: Any,
                   reverse: bool = False, inclusive: bool = True,
                   stream: Any | None = None, **kwargs: Any) -> Any:
    if (args or kwargs or not _has_precise_array(value)
            or getattr(value, "dtype", None) != mx.float64):
        if stream is None:
            return _ORIGINAL_MX_CUMSUM(
                value, axis=axis, reverse=reverse, inclusive=inclusive, **kwargs)
        return _ORIGINAL_MX_CUMSUM(
            value, axis=axis, reverse=reverse, inclusive=inclusive,
            stream=stream, **kwargs)
    return _mx_overrides.cumsum_precise(
        value, axis, reverse=reverse, inclusive=inclusive,
        stream=_shadow_stream(stream))


def _clip_shadow(value: Any, a_min: Any, a_max: Any, *args: Any,
                 stream: Any | None = None, **kwargs: Any) -> Any:
    if args or kwargs:
        if stream is None:
            return _ORIGINAL_MX_CLIP(value, a_min, a_max, *args, **kwargs)
        return _ORIGINAL_MX_CLIP(
            value, a_min, a_max, *args, stream=stream, **kwargs)
    actual_stream = _shadow_stream(stream)
    if not _has_precise_array(value):
        value = mx.array(value)
    if a_min is not None:
        value = _maximum_shadow(value, a_min, stream=actual_stream)
    if a_max is not None:
        value = _minimum_shadow(value, a_max, stream=actual_stream)
    return value


def _broadcast_to_shadow(value: Any, shape: Any, *args: Any,
                         stream: Any | None = None, **kwargs: Any) -> Any:
    if args or kwargs:
        if stream is None:
            return _ORIGINAL_MX_BROADCAST_TO(value, shape, *args, **kwargs)
        return _ORIGINAL_MX_BROADCAST_TO(
            value, shape, *args, stream=stream, **kwargs)
    if not _has_precise_array(value):
        value = mx.array(value)
    target_shape = _shape_tuple(shape)
    if tuple(value.shape) == target_shape:
        return value
    if getattr(value, "dtype", None) == mx.float64:
        actual_stream = _shadow_stream(stream)
        return _add_shadow(
            mx.zeros(target_shape, dtype=value.dtype, stream=actual_stream),
            value, stream=actual_stream)
    if stream is None:
        return _ORIGINAL_MX_BROADCAST_TO(value, target_shape)
    return _ORIGINAL_MX_BROADCAST_TO(value, target_shape, stream=stream)


def _broadcast_arrays_shadow(*arrays: Any, stream: Any | None = None,
                             **kwargs: Any) -> Tuple[mx.array, ...]:
    if kwargs:
        if stream is None:
            return _ORIGINAL_MX_BROADCAST_ARRAYS(*arrays, **kwargs)
        return _ORIGINAL_MX_BROADCAST_ARRAYS(*arrays, stream=stream, **kwargs)
    values = tuple(
        value if _has_precise_array(value) else mx.array(value)
        for value in arrays)
    shape = mx.broadcast_shapes(*(value.shape for value in values))
    return tuple(mx.broadcast_to(value, shape, stream=stream)
                 for value in values)


def _normalize_pad_width(pad_width: Any, ndim: int) -> Tuple[Tuple[int, int], ...]:
    if isinstance(pad_width, int):
        return ((pad_width, pad_width),) * ndim
    if isinstance(pad_width, _ORIGINAL_MX_ARRAY):
        pad_width = pad_width.tolist()
    if len(pad_width) == 2 and all(isinstance(item, int) for item in pad_width):
        before, after = pad_width
        return ((int(before), int(after)),) * ndim
    pads = tuple((int(before), int(after)) for before, after in pad_width)
    if len(pads) != ndim:
        raise ValueError("pad_width must match array rank")
    return pads


def _pad_shadow(value: Any, pad_width: Any, mode: str = "constant",
                *args: Any, constant_values: Any = 0,
                stream: Any | None = None, **kwargs: Any) -> Any:
    if args or kwargs or mode != "constant":
        if stream is None:
            return _ORIGINAL_MX_PAD(value, pad_width, mode, *args, **kwargs)
        return _ORIGINAL_MX_PAD(
            value, pad_width, mode, *args, stream=stream, **kwargs)
    if not _has_precise_array(value):
        value = mx.array(value)
    if getattr(value, "dtype", None) != mx.float64:
        if stream is None:
            return _ORIGINAL_MX_PAD(
                value, pad_width, mode, constant_values=constant_values)
        return _ORIGINAL_MX_PAD(
            value, pad_width, mode, constant_values=constant_values,
            stream=stream)

    result = value
    actual_stream = _shadow_stream(stream)
    pads = _normalize_pad_width(pad_width, result.ndim)
    for axis, (before, after) in enumerate(pads):
        if before < 0 or after < 0:
            raise ValueError("pad_width cannot contain negative values")
        if before == 0 and after == 0:
            continue
        pieces = []
        if before:
            shape = list(result.shape)
            shape[axis] = before
            pieces.append(mx.full(
                tuple(shape), constant_values, dtype=result.dtype,
                stream=actual_stream))
        pieces.append(result)
        if after:
            shape = list(result.shape)
            shape[axis] = after
            pieces.append(mx.full(
                tuple(shape), constant_values, dtype=result.dtype,
                stream=actual_stream))
        result = mx.concatenate(pieces, axis=axis, stream=actual_stream)
    return result


def _eye_shadow(n: int, m: int | None = None, k: int = 0,
                dtype: Any | None = None, *,
                stream: Any | None = None) -> Any:
    target_dtype = _unwrap_dtype(dtype) or mx.float32
    if target_dtype != mx.float64:
        if stream is None:
            return _wrap_factory_array(
                _ORIGINAL_MX_EYE(n, m, k=k, dtype=target_dtype), stream)
        return _wrap_factory_array(
            _ORIGINAL_MX_EYE(n, m, k=k, dtype=target_dtype, stream=stream),
            stream)
    if m is None:
        m = n
    data = [
        [1.0 if col - row == k else 0.0 for col in range(m)]
        for row in range(n)
    ]
    return _construct_mx_array(
        data, dtype=target_dtype, stream=_shadow_stream(stream))


def _identity_shadow(n: int, dtype: Any | None = None, *,
                     stream: Any | None = None) -> Any:
    return _eye_shadow(n, n, dtype=dtype, stream=stream)


def _wrap_factory_array(value: Any, stream: Any | None = None) -> Any:
    return _mx_overrides.MlxPreciseArray(
        value, stream=stream if stream is not None else _active_stream())


def _zeros_shadow(shape: Any, dtype: Any | None = None, *,
                  stream: Any | None = None) -> Any:
    target_dtype = _unwrap_dtype(dtype) or mx.float32
    if target_dtype == mx.float64:
        return _mx_overrides.full_float64(
            shape, 0.0, stream=_shadow_stream(stream))
    if stream is None:
        return _wrap_factory_array(
            _ORIGINAL_MX_ZEROS(shape, dtype=target_dtype), stream)
    return _wrap_factory_array(
        _ORIGINAL_MX_ZEROS(shape, dtype=target_dtype, stream=stream), stream)


def _ones_shadow(shape: Any, dtype: Any | None = None, *,
                 stream: Any | None = None) -> Any:
    target_dtype = _unwrap_dtype(dtype) or mx.float32
    if target_dtype == mx.float64:
        return _mx_overrides.full_float64(
            shape, 1.0, stream=_shadow_stream(stream))
    if stream is None:
        return _wrap_factory_array(
            _ORIGINAL_MX_ONES(shape, dtype=target_dtype), stream)
    return _wrap_factory_array(
        _ORIGINAL_MX_ONES(shape, dtype=target_dtype, stream=stream), stream)


def _full_shadow(shape: Any, fill_value: Any, dtype: Any | None = None, *,
                 stream: Any | None = None) -> Any:
    target_dtype = _unwrap_dtype(dtype)
    if target_dtype == mx.float64:
        return _mx_overrides.full_float64(
            shape, float(fill_value), stream=_shadow_stream(stream))
    if stream is None:
        if target_dtype is None:
            return _wrap_factory_array(_ORIGINAL_MX_FULL(shape, fill_value),
                                       stream)
        return _wrap_factory_array(
            _ORIGINAL_MX_FULL(shape, fill_value, dtype=target_dtype), stream)
    if target_dtype is None:
        return _wrap_factory_array(
            _ORIGINAL_MX_FULL(shape, fill_value, stream=stream), stream)
    return _wrap_factory_array(
        _ORIGINAL_MX_FULL(shape, fill_value, dtype=target_dtype, stream=stream),
        stream)


class _DateTimeArray(list):
    def __init__(self, values: Iterable[Any], unit: str = "D",
                 shape: Tuple[int, ...] | None = None):
        super().__init__(values)
        self._unit = unit
        self._shape = shape if shape is not None else (len(self),)

    @property
    def dtype(self) -> str:
        return f"datetime64[{self._unit}]"

    @property
    def shape(self) -> Tuple[int]:
        return self._shape

    @property
    def ndim(self) -> int:
        return len(self._shape)

    @property
    def size(self) -> int:
        return len(self)

    def astype(self, dtype: Any, *args: Any, **kwargs: Any):
        if _dtype_kind(dtype) == "M":
            return _DateTimeArray(self, _datetime_unit(dtype, None, None))
        if _dtype_kind(dtype) == "O":
            return list(self)
        raise TypeError("datetime data is not MLX tensor data")

    def tolist(self) -> list[Any]:
        return list(self)

    def __getitem__(self, key: Any):
        if isinstance(key, _ORIGINAL_MX_ARRAY):
            key = key.tolist()
        if isinstance(key, tuple):
            if len(key) == 1:
                return self[key[0]]
            if len(key) == 2 and key[1] is None:
                rows = self[key[0]]
                values = rows.tolist() if isinstance(rows, _DateTimeArray) else [rows]
                return _DateTimeArray(values, self._unit, (len(values), 1))
            if len(key) == 2 and self.ndim == 2:
                row_key, col_key = key
                if isinstance(col_key, _ORIGINAL_MX_ARRAY):
                    col_key = col_key.item()
                if isinstance(col_key, slice):
                    cols = range(*col_key.indices(self.shape[1]))
                    if list(cols) != [0]:
                        raise IndexError("datetime array column index out of range")
                    rows = self[row_key]
                    values = rows.tolist() if isinstance(rows, _DateTimeArray) else [rows]
                    return _DateTimeArray(values, self._unit, (len(values), 1))
                if int(col_key) != 0:
                    raise IndexError("datetime array column index out of range")
                return self[row_key]
            if not any(part is None or isinstance(part, slice) for part in key):
                return _DateTimeArray(
                    [list.__getitem__(self, int(i)) for i in key],
                    self._unit)
            raise IndexError("too many indices for datetime array")
        if isinstance(key, (list, tuple)):
            return _DateTimeArray(
                [list.__getitem__(self, int(i)) for i in key],
                self._unit)
        value = list.__getitem__(self, key)
        if isinstance(key, slice):
            return _DateTimeArray(value, self._unit)
        return value


def _datetime_unit(dtype: Any, start: Any, stop: Any) -> str:
    if isinstance(dtype, str) and dtype.startswith("datetime64"):
        if "[" in dtype and dtype.endswith("]"):
            return dtype[dtype.index("[") + 1:-1] or "D"
        return "D" if all(
            isinstance(value, str) and "T" not in value and len(value) <= 10
            for value in (start, stop) if value is not None) else "s"
    return "D"


def _datetime_step(value: Any, unit: str) -> timedelta:
    if isinstance(value, timedelta):
        return value
    if isinstance(value, _ORIGINAL_MX_ARRAY):
        value = value.item()
    scale = {
        "W": lambda v: timedelta(weeks=v),
        "D": lambda v: timedelta(days=v),
        "h": lambda v: timedelta(hours=v),
        "m": lambda v: timedelta(minutes=v),
        "s": lambda v: timedelta(seconds=v),
        "ms": lambda v: timedelta(milliseconds=v),
        "us": lambda v: timedelta(microseconds=v),
        "ns": lambda v: timedelta(microseconds=v / 1000),
    }.get(unit, lambda v: timedelta(days=v))
    return scale(value)


def _is_datetime_arange(start: Any, stop: Any, dtype: Any) -> bool:
    return (_dtype_kind(dtype) == "M"
            or isinstance(start, datetime)
            or isinstance(stop, datetime))


def _datetime_arange(start: Any, stop: Any, step: Any,
                     dtype: Any) -> _DateTimeArray:
    unit = _datetime_unit(dtype, start, stop)
    current = datetime64(start)
    end = datetime64(stop)
    delta = _datetime_step(step, unit)
    values = []
    if delta.total_seconds() == 0:
        raise ValueError("datetime arange step must be non-zero")
    if delta.total_seconds() > 0:
        while current < end:
            values.append(current)
            current = current + delta
    else:
        while current > end:
            values.append(current)
            current = current + delta
    return _DateTimeArray(values, unit)


def _arange_shadow(start: Any, stop: Any | None = None, step: Any = 1,
                   dtype: Any | None = None, *,
                   stream: Any | None = None) -> Any:
    start = _to_scalar(start)
    if stop is None:
        start, stop = 0, _to_scalar(start)
    else:
        stop = _to_scalar(stop)
    step = _to_scalar(step)
    if _is_datetime_arange(start, stop, dtype):
        return _datetime_arange(start, stop, step, dtype)
    target_dtype = _unwrap_dtype(dtype)
    if target_dtype == mx.float64:
        return _mx_overrides.arange_float64(
            float(start), float(stop), float(step),
            stream=_shadow_stream(stream))
    if stream is None:
        if target_dtype is None:
            return _wrap_factory_array(
                _ORIGINAL_MX_ARANGE(start, stop, step), stream)
        return _wrap_factory_array(
            _ORIGINAL_MX_ARANGE(start, stop, step, dtype=target_dtype), stream)
    if target_dtype is None:
        return _wrap_factory_array(
            _ORIGINAL_MX_ARANGE(start, stop, step, stream=stream), stream)
    return _wrap_factory_array(
        _ORIGINAL_MX_ARANGE(
            start, stop, step, dtype=target_dtype, stream=stream),
        stream)


def _linspace_shadow(start: Any, stop: Any, num: int | None = 50,
                     dtype: Any | None = None, *,
                     stream: Any | None = None) -> Any:
    target_dtype = _unwrap_dtype(dtype) or mx.float32
    if target_dtype != mx.float64:
        if stream is None:
            return _wrap_factory_array(
                _ORIGINAL_MX_LINSPACE(
                    start, stop, num, dtype=target_dtype), stream)
        return _wrap_factory_array(
            _ORIGINAL_MX_LINSPACE(
                start, stop, num, dtype=target_dtype, stream=stream), stream)

    count = 50 if num is None else int(num)
    if count < 0:
        raise ValueError("Number of samples must be non-negative")
    actual_stream = _shadow_stream(stream)
    if count == 0:
        return _mx_overrides.full_float64((0,), 0.0, stream=actual_stream)
    start_value = float(_to_scalar(start))
    stop_value = float(_to_scalar(stop))
    if count == 1:
        return _mx_overrides.full_float64(
            (1,), start_value, stream=actual_stream)

    step = (stop_value - start_value) / (count - 1)
    indices = _mx_overrides.arange_float64(
        0.0, float(count), 1.0, stream=actual_stream)
    return _add_shadow(
        _mx_overrides.float64_scalar(start_value, stream=actual_stream),
        _multiply_shadow(
            _mx_overrides.float64_scalar(step, stream=actual_stream),
            indices, stream=actual_stream),
        stream=actual_stream)


def _repeat_shadow(value: Any, repeats: Any, axis: int | None = None,
                   *, stream: Any | None = None) -> Any:
    try:
        repeat_count = operator.index(repeats)
    except TypeError:
        if isinstance(repeats, _ORIGINAL_MX_ARRAY) and repeats.size == 1:
            repeat_count = int(repeats.item())
        else:
            if stream is None:
                return _wrap_factory_array(
                    _ORIGINAL_MX_REPEAT(value, repeats, axis=axis), stream)
            return _wrap_factory_array(
                _ORIGINAL_MX_REPEAT(value, repeats, axis=axis, stream=stream),
                stream)
    if repeat_count < 0:
        raise ValueError("repeats must be non-negative")

    actual_stream = _shadow_stream(stream)
    if not _has_precise_array(value):
        value = mx.array(value, stream=actual_stream)
    if axis is None:
        value = mx.reshape(value, (value.size,), stream=actual_stream)
        axis = 0
    else:
        axis = int(axis)
        if axis < 0:
            axis += value.ndim
        if axis < 0 or axis >= value.ndim:
            raise ValueError("axis out of bounds")

    expanded = mx.expand_dims(value, axis + 1, stream=actual_stream)
    broadcast_shape = (
        tuple(value.shape[:axis + 1])
        + (repeat_count,)
        + tuple(value.shape[axis + 1:]))
    repeated = mx.broadcast_to(expanded, broadcast_shape, stream=actual_stream)
    result_shape = tuple(value.shape)
    result_shape = (
        result_shape[:axis]
        + (result_shape[axis] * repeat_count,)
        + result_shape[axis + 1:])
    return mx.reshape(repeated, result_shape, stream=actual_stream)


def _tile_reps(reps: Any) -> Tuple[int, ...]:
    try:
        return (operator.index(reps),)
    except TypeError:
        pass
    if isinstance(reps, _ORIGINAL_MX_ARRAY):
        if reps.ndim == 0 or reps.size == 1:
            return (int(reps.item()),)
        reps = reps.tolist()
    return tuple(operator.index(rep) for rep in reps)


def _tile_shadow(value: Any, reps: Any, *args: Any,
                 stream: Any | None = None, **kwargs: Any) -> Any:
    if args or kwargs or not (
            _needs_precise_scalar_route(value)
            or _contains_python_float(value)):
        if stream is None:
            return _wrap_factory_array(
                _ORIGINAL_MX_TILE(value, reps, *args, **kwargs), stream)
        return _wrap_factory_array(
            _ORIGINAL_MX_TILE(value, reps, *args, stream=stream, **kwargs),
            stream)

    actual_stream = _shadow_stream(stream)
    if not _has_precise_array(value):
        value = mx.array(value, stream=actual_stream)

    reps_tuple = _tile_reps(reps)
    if _builtins.any(rep < 0 for rep in reps_tuple):
        raise ValueError("reps must be non-negative")

    ndim = _builtins.max(value.ndim, len(reps_tuple))
    if value.ndim < ndim:
        value = mx.reshape(
            value, (1,) * (ndim - value.ndim) + tuple(value.shape),
            stream=actual_stream)
    if len(reps_tuple) < ndim:
        reps_tuple = (1,) * (ndim - len(reps_tuple)) + reps_tuple

    reshape_shape = tuple(
        dim for pair in zip((1,) * ndim, value.shape) for dim in pair)
    broadcast_shape = tuple(
        dim for pair in zip(reps_tuple, value.shape) for dim in pair)
    result_shape = tuple(
        dim * rep for dim, rep in zip(value.shape, reps_tuple))

    tiled = mx.reshape(value, reshape_shape, stream=actual_stream)
    tiled = mx.broadcast_to(tiled, broadcast_shape, stream=actual_stream)
    return mx.reshape(tiled, result_shape, stream=actual_stream)


def _reshape_shadow(value: Any, shape: Any, *args: Any,
                    stream: Any | None = None, **kwargs: Any) -> Any:
    if not _has_precise_array(value):
        value = mx.array(value)
    if (args or kwargs or not _has_precise_array(value)
            or getattr(value, "dtype", None) != mx.float64):
        if stream is None:
            return _wrap_factory_array(
                _ORIGINAL_MX_RESHAPE(value, shape, *args, **kwargs), stream)
        return _wrap_factory_array(
            _ORIGINAL_MX_RESHAPE(
                value, shape, *args, stream=stream, **kwargs),
            stream)
    return _mx_overrides.reshape_precise(
        value, shape, stream=_shadow_stream(stream))


def _flattened_shape(shape: Sequence[int], start_axis: int, end_axis: int) -> Tuple[int, ...]:
    ndim = len(shape)
    if ndim == 0:
        return ()
    if start_axis < 0:
        start_axis += ndim
    if end_axis < 0:
        end_axis += ndim
    if start_axis < 0 or end_axis < 0 or start_axis >= ndim or end_axis >= ndim:
        raise ValueError("flatten axis out of bounds")
    if start_axis > end_axis:
        raise ValueError("flatten start_axis must be less than or equal to end_axis")
    flattened = math.prod(shape[start_axis:end_axis + 1])
    return (*shape[:start_axis], flattened, *shape[end_axis + 1:])


def _flatten_shadow(value: Any, start_axis: int = 0, end_axis: int = -1,
                    *args: Any, stream: Any | None = None,
                    **kwargs: Any) -> Any:
    if not _has_precise_array(value):
        value = mx.array(value)
    if (args or kwargs or not _has_precise_array(value)
            or getattr(value, "dtype", None) != mx.float64):
        if _ORIGINAL_MX_FLATTEN is None:
            shape = _flattened_shape(tuple(_to_mx(value).shape), start_axis, end_axis)
            return _reshape_shadow(value, shape, stream=stream)
        if stream is None:
            return _ORIGINAL_MX_FLATTEN(
                value, start_axis=start_axis, end_axis=end_axis, **kwargs)
        return _ORIGINAL_MX_FLATTEN(
            value, start_axis=start_axis, end_axis=end_axis, stream=stream, **kwargs)
    shape = _flattened_shape(tuple(value.shape), start_axis, end_axis)
    return _mx_overrides.reshape_precise(
        value, shape, stream=_shadow_stream(stream))


def _take_shadow(value: Any, indices: Any, axis: Any | None = None,
                 *args: Any, stream: Any | None = None, **kwargs: Any) -> Any:
    if not _has_precise_array(value):
        value = mx.array(value)
    if not _has_precise_array(indices):
        indices = mx.array(indices)
    if args or kwargs or not _has_precise_array(value) or getattr(
            value, "dtype", None) != mx.float64:
        if axis is None:
            if stream is None:
                return _ORIGINAL_MX_TAKE(value, indices)
            return _ORIGINAL_MX_TAKE(value, indices, stream=stream)
        if stream is None:
            return _ORIGINAL_MX_TAKE(value, indices, axis=axis)
        return _ORIGINAL_MX_TAKE(value, indices, axis=axis, stream=stream)
    return _mx_overrides.take_precise(
        value, indices, axis, stream=_shadow_stream(stream))


def _slice_shadow(value: Any, start_indices: Any, axes: Any,
                  slice_size: Any, *args: Any,
                  stream: Any | None = None, **kwargs: Any) -> Any:
    if not _has_precise_array(value):
        value = mx.array(value)
    if (args or kwargs or not _has_precise_array(value)
            or getattr(value, "dtype", None) != mx.float64):
        if stream is None:
            return _ORIGINAL_MX_SLICE(
                value, start_indices, axes, slice_size, *args, **kwargs)
        return _ORIGINAL_MX_SLICE(
            value, start_indices, axes, slice_size, *args,
            stream=stream, **kwargs)

    actual_stream = _shadow_stream(stream)
    if isinstance(start_indices, _ORIGINAL_MX_ARRAY):
        start_values = start_indices.tolist()
    else:
        start_values = start_indices
    if isinstance(axes, int):
        axes_values = (axes,)
    else:
        axes_values = tuple(axes)
    if isinstance(slice_size, int):
        size_values = (slice_size,)
    else:
        size_values = tuple(slice_size)
    if len(axes_values) != len(start_values) or len(axes_values) != len(size_values):
        raise ValueError("slice axes, starts, and sizes must have the same length")

    starts = [0] * value.ndim
    stops = list(value.shape)
    strides = [1] * value.ndim
    for raw_axis, raw_start, raw_size in zip(
            axes_values, start_values, size_values):
        axis = int(raw_axis)
        if axis < 0:
            axis += value.ndim
        if axis < 0 or axis >= value.ndim:
            raise ValueError("slice axis out of bounds")
        start = int(raw_start)
        starts[axis] = start
        stops[axis] = start + int(raw_size)
    return _mx_overrides.slice_precise(
        value, tuple(starts), tuple(stops), tuple(strides),
        stream=actual_stream)


def _roll_one_axis(value: Any, shift: int, axis: int,
                   stream: Any | None = None) -> Any:
    dim = value.shape[axis]
    if dim == 0:
        return value
    shift %= dim
    if shift == 0:
        return value
    starts = [0] * value.ndim
    stops = list(value.shape)
    strides = [1] * value.ndim
    starts[axis] = dim - shift
    tail = _mx_overrides.slice_precise(
        value, tuple(starts), tuple(stops), tuple(strides), stream=stream)
    starts[axis] = 0
    stops[axis] = dim - shift
    head = _mx_overrides.slice_precise(
        value, tuple(starts), tuple(stops), tuple(strides), stream=stream)
    return mx.concatenate((tail, head), axis=axis, stream=stream)


def _roll_shadow(value: Any, shift: Any, axis: Any | None = None,
                 *args: Any, stream: Any | None = None,
                 **kwargs: Any) -> Any:
    if not _has_precise_array(value):
        value = mx.array(value)
    if (args or kwargs or getattr(value, "dtype", None) != mx.float64):
        if stream is None:
            return _ORIGINAL_MX_ROLL(value, shift, axis=axis, *args, **kwargs)
        return _ORIGINAL_MX_ROLL(
            value, shift, axis=axis, *args, stream=stream, **kwargs)
    actual_stream = _shadow_stream(stream)
    original_shape = tuple(value.shape)
    if axis is None:
        value = mx.reshape(value, (value.size,), stream=actual_stream)
        axes = (0,)
    else:
        axes = (axis,) if isinstance(axis, int) else tuple(axis)
    shifts = (shift,) if isinstance(shift, int) else tuple(shift)
    if len(shifts) == 1 and len(axes) > 1:
        shifts = shifts * len(axes)
    if len(shifts) != len(axes):
        raise ValueError("shift and axis must have matching lengths")
    for raw_shift, raw_axis in zip(shifts, axes):
        axis_index = int(raw_axis)
        if axis_index < 0:
            axis_index += value.ndim
        if axis_index < 0 or axis_index >= value.ndim:
            raise ValueError("roll axis out of bounds")
        value = _roll_one_axis(
            value, int(raw_shift), axis_index, stream=actual_stream)
    if axis is None:
        value = mx.reshape(value, original_shape, stream=actual_stream)
    return value


def _log_shadow(value: Any, *args: Any,
                stream: Any | None = None, **kwargs: Any) -> Any:
    if args or kwargs:
        if stream is None:
            return _ORIGINAL_MX_LOG(value, *args, **kwargs)
        return _ORIGINAL_MX_LOG(value, *args, stream=stream, **kwargs)
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return math.log(value)
    if not _has_precise_array(value) and isinstance(value, mx.array):
        if stream is None:
            return _ORIGINAL_MX_LOG(value)
        return _ORIGINAL_MX_LOG(value, stream=stream)
    if _has_precise_array(value) and getattr(value, "dtype", None) != mx.float64:
        if stream is None:
            return _ORIGINAL_MX_LOG(value)
        return _ORIGINAL_MX_LOG(value, stream=stream)
    actual_stream = _shadow_stream(stream)
    return _wrap_factory_array(
        _mx_overrides.log_float64(value, stream=actual_stream), actual_stream)


def _log2_shadow(value: Any, *args: Any,
                 stream: Any | None = None, **kwargs: Any) -> Any:
    if args or kwargs:
        if stream is None:
            return _ORIGINAL_MX_LOG2(value, *args, **kwargs)
        return _ORIGINAL_MX_LOG2(value, *args, stream=stream, **kwargs)
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return math.log2(value)
    if not _has_precise_array(value) and isinstance(value, mx.array):
        if stream is None:
            return _ORIGINAL_MX_LOG2(value)
        return _ORIGINAL_MX_LOG2(value, stream=stream)
    if _has_precise_array(value) and getattr(value, "dtype", None) != mx.float64:
        if stream is None:
            return _ORIGINAL_MX_LOG2(value)
        return _ORIGINAL_MX_LOG2(value, stream=stream)
    actual_stream = _shadow_stream(stream)
    return _wrap_factory_array(
        _mx_overrides.log2_float64(value, stream=actual_stream), actual_stream)


def _log10_shadow(value: Any, *args: Any,
                  stream: Any | None = None, **kwargs: Any) -> Any:
    if args or kwargs:
        if stream is None:
            return _ORIGINAL_MX_LOG10(value, *args, **kwargs)
        return _ORIGINAL_MX_LOG10(value, *args, stream=stream, **kwargs)
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return math.log10(value)
    if not _has_precise_array(value) and isinstance(value, mx.array):
        if stream is None:
            return _ORIGINAL_MX_LOG10(value)
        return _ORIGINAL_MX_LOG10(value, stream=stream)
    if _has_precise_array(value) and getattr(value, "dtype", None) != mx.float64:
        if stream is None:
            return _ORIGINAL_MX_LOG10(value)
        return _ORIGINAL_MX_LOG10(value, stream=stream)
    actual_stream = _shadow_stream(stream)
    return _wrap_factory_array(
        _mx_overrides.log10_float64(value, stream=actual_stream), actual_stream)


def _sin_shadow(value: Any, *args: Any,
                stream: Any | None = None, **kwargs: Any) -> Any:
    if args or kwargs:
        if stream is None:
            return _ORIGINAL_MX_SIN(value, *args, **kwargs)
        return _ORIGINAL_MX_SIN(value, *args, stream=stream, **kwargs)
    if not _has_precise_array(value):
        value = mx.array(value)
    if getattr(value, "dtype", None) != mx.float64:
        if stream is None:
            return _ORIGINAL_MX_SIN(value)
        return _ORIGINAL_MX_SIN(value, stream=stream)
    actual_stream = _shadow_stream(stream)
    return _wrap_factory_array(
        _mx_overrides.sin_float64(value, stream=actual_stream), actual_stream)


def _cos_shadow(value: Any, *args: Any,
                stream: Any | None = None, **kwargs: Any) -> Any:
    if args or kwargs:
        if stream is None:
            return _ORIGINAL_MX_COS(value, *args, **kwargs)
        return _ORIGINAL_MX_COS(value, *args, stream=stream, **kwargs)
    if not _has_precise_array(value):
        value = mx.array(value)
    if getattr(value, "dtype", None) != mx.float64:
        if stream is None:
            return _ORIGINAL_MX_COS(value)
        return _ORIGINAL_MX_COS(value, stream=stream)
    actual_stream = _shadow_stream(stream)
    return _wrap_factory_array(
        _mx_overrides.cos_float64(value, stream=actual_stream), actual_stream)


def _unary_float64_shadow(original: Any, precise: Any, value: Any,
                          *args: Any, stream: Any | None = None,
                          **kwargs: Any) -> Any:
    if args or kwargs:
        if stream is None:
            return original(value, *args, **kwargs)
        return original(value, *args, stream=stream, **kwargs)
    if not _has_precise_array(value):
        value = mx.array(value)
    if getattr(value, "dtype", None) != mx.float64:
        if stream is None:
            return original(value)
        return original(value, stream=stream)
    actual_stream = _shadow_stream(stream)
    return _wrap_factory_array(
        precise(value, stream=actual_stream), actual_stream)


def _arcsin_shadow(value: Any, *args: Any,
                   stream: Any | None = None, **kwargs: Any) -> Any:
    return _unary_float64_shadow(
        _ORIGINAL_MX_ARCSIN, _mx_overrides.arcsin_float64,
        value, *args, stream=stream, **kwargs)


def _arccos_shadow(value: Any, *args: Any,
                   stream: Any | None = None, **kwargs: Any) -> Any:
    return _unary_float64_shadow(
        _ORIGINAL_MX_ARCCOS, _mx_overrides.arccos_float64,
        value, *args, stream=stream, **kwargs)


def _arctan_shadow(value: Any, *args: Any,
                   stream: Any | None = None, **kwargs: Any) -> Any:
    return _unary_float64_shadow(
        _ORIGINAL_MX_ARCTAN, _mx_overrides.arctan_float64,
        value, *args, stream=stream, **kwargs)


def _floor_shadow(value: Any, *args: Any,
                  stream: Any | None = None, **kwargs: Any) -> Any:
    if args or kwargs:
        if stream is None:
            return _ORIGINAL_MX_FLOOR(value, *args, **kwargs)
        return _ORIGINAL_MX_FLOOR(value, *args, stream=stream, **kwargs)
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return math.floor(value)
    if not _has_precise_array(value) and isinstance(value, mx.array):
        if stream is None:
            return _ORIGINAL_MX_FLOOR(value)
        return _ORIGINAL_MX_FLOOR(value, stream=stream)
    if _has_precise_array(value) and getattr(value, "dtype", None) != mx.float64:
        if stream is None:
            return _ORIGINAL_MX_FLOOR(value)
        return _ORIGINAL_MX_FLOOR(value, stream=stream)
    actual_stream = _shadow_stream(stream)
    return _wrap_factory_array(
        _mx_overrides.floor_float64(value, stream=actual_stream), actual_stream)


def _ceil_shadow(value: Any, *args: Any,
                 stream: Any | None = None, **kwargs: Any) -> Any:
    if args or kwargs:
        if stream is None:
            return _ORIGINAL_MX_CEIL(value, *args, **kwargs)
        return _ORIGINAL_MX_CEIL(value, *args, stream=stream, **kwargs)
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return math.ceil(value)
    if not _has_precise_array(value) and isinstance(value, mx.array):
        if stream is None:
            return _ORIGINAL_MX_CEIL(value)
        return _ORIGINAL_MX_CEIL(value, stream=stream)
    if _has_precise_array(value) and getattr(value, "dtype", None) != mx.float64:
        if stream is None:
            return _ORIGINAL_MX_CEIL(value)
        return _ORIGINAL_MX_CEIL(value, stream=stream)
    actual_stream = _shadow_stream(stream)
    return _wrap_factory_array(
        _mx_overrides.ceil_float64(value, stream=actual_stream), actual_stream)


def _divmod_shadow(left: Any, right: Any, *args: Any,
                   stream: Any | None = None, **kwargs: Any) -> Any:
    if args or kwargs or not (
            _needs_precise_scalar_route(left)
            or _needs_precise_scalar_route(right)):
        if stream is None:
            return _ORIGINAL_MX_DIVMOD(left, right, *args, **kwargs)
        return _ORIGINAL_MX_DIVMOD(left, right, *args, stream=stream, **kwargs)
    actual_stream = _shadow_stream(stream)
    quotient = _floor_shadow(
        _divide_shadow(left, right, stream=actual_stream),
        stream=actual_stream)
    remainder = _subtract_shadow(
        left,
        _multiply_shadow(quotient, right, stream=actual_stream),
        stream=actual_stream)
    return quotient, remainder


def _sqrt_shadow(value: Any, *args: Any,
                 stream: Any | None = None, **kwargs: Any) -> Any:
    if args or kwargs:
        if stream is None:
            return _ORIGINAL_MX_SQRT(value, *args, **kwargs)
        return _ORIGINAL_MX_SQRT(value, *args, stream=stream, **kwargs)
    if not _has_precise_array(value):
        value = mx.array(value)
    if _has_precise_array(value) and getattr(value, "dtype", None) != mx.float64:
        if stream is None:
            return _ORIGINAL_MX_SQRT(value)
        return _ORIGINAL_MX_SQRT(value, stream=stream)
    actual_stream = _shadow_stream(stream)
    return _wrap_factory_array(
        _mx_overrides.sqrt_float64(value, stream=actual_stream), actual_stream)


def _degrees_shadow(value: Any, *args: Any,
                    stream: Any | None = None, **kwargs: Any) -> Any:
    if args or kwargs:
        if stream is None:
            return _ORIGINAL_MX_DEGREES(value, *args, **kwargs)
        return _ORIGINAL_MX_DEGREES(value, *args, stream=stream, **kwargs)
    if not _has_precise_array(value):
        value = mx.array(value)
    if _has_precise_array(value) and getattr(value, "dtype", None) != mx.float64:
        if stream is None:
            return _ORIGINAL_MX_DEGREES(value)
        return _ORIGINAL_MX_DEGREES(value, stream=stream)
    actual_stream = _shadow_stream(stream)
    return _wrap_factory_array(
        _mx_overrides.degrees_float64(value, stream=actual_stream),
        actual_stream)


def _radians_shadow(value: Any, *args: Any,
                    stream: Any | None = None, **kwargs: Any) -> Any:
    if args or kwargs:
        if stream is None:
            return _ORIGINAL_MX_RADIANS(value, *args, **kwargs)
        return _ORIGINAL_MX_RADIANS(value, *args, stream=stream, **kwargs)
    if not _has_precise_array(value):
        value = mx.array(value)
    if _has_precise_array(value) and getattr(value, "dtype", None) != mx.float64:
        if stream is None:
            return _ORIGINAL_MX_RADIANS(value)
        return _ORIGINAL_MX_RADIANS(value, stream=stream)
    actual_stream = _shadow_stream(stream)
    return _wrap_factory_array(
        _mx_overrides.radians_float64(value, stream=actual_stream),
        actual_stream)


def _sort_shadow(value: Any, axis: Any | None = -1, *args: Any,
                 stream: Any | None = None, **kwargs: Any) -> Any:
    if args or kwargs or not _has_precise_array(value) or getattr(
            value, "dtype", None) != mx.float64:
        if stream is None:
            return _ORIGINAL_MX_SORT(value, axis=axis, *args, **kwargs)
        return _ORIGINAL_MX_SORT(value, axis=axis, *args, stream=stream, **kwargs)
    actual_stream = _shadow_stream(stream)
    return _wrap_factory_array(
        _mx_overrides.sort_float64(value, axis=axis, stream=actual_stream),
        actual_stream)


def _argsort_shadow(value: Any, axis: Any | None = -1, *args: Any,
                    stream: Any | None = None, **kwargs: Any) -> Any:
    if not _has_precise_array(value):
        value = mx.array(value)
    if args or kwargs or not _has_precise_array(value) or getattr(
            value, "dtype", None) != mx.float64:
        if stream is None:
            return _ORIGINAL_MX_ARGSORT(value, axis=axis, *args, **kwargs)
        return _ORIGINAL_MX_ARGSORT(
            value, axis=axis, *args, stream=stream, **kwargs)
    actual_stream = _shadow_stream(stream)
    return _wrap_factory_array(
        _mx_overrides.argsort_float64(value, axis=axis, stream=actual_stream),
        actual_stream)


def _squeezed_shape(shape: Sequence[int], axis: Any | None = None) -> Tuple[int, ...]:
    new_shape = list(shape)
    if axis is None:
        return tuple(dim for dim in new_shape if dim != 1)
    axes = axis if isinstance(axis, (list, tuple)) else (axis,)
    normalized = []
    ndim = len(new_shape)
    for ax in axes:
        index = int(ax)
        if index < 0:
            index += ndim
        if index < 0 or index >= ndim:
            raise ValueError("axis out of bounds")
        normalized.append(index)
    for index in sorted(set(normalized), reverse=True):
        if new_shape[index] != 1:
            raise ValueError("cannot select an axis to squeeze out which has size not equal to one")
        del new_shape[index]
    return tuple(new_shape)


def _squeeze_shadow(value: Any, axis: Any | None = None, *args: Any,
                    stream: Any | None = None, **kwargs: Any) -> Any:
    if not _has_precise_array(value):
        value = mx.array(value)
    if (args or kwargs or not _has_precise_array(value)
            or getattr(value, "dtype", None) != mx.float64):
        if stream is None:
            return _ORIGINAL_MX_SQUEEZE(value, axis=axis, *args, **kwargs)
        return _ORIGINAL_MX_SQUEEZE(
            value, axis=axis, *args, stream=stream, **kwargs)
    return _mx_overrides.reshape_precise(
        value, _squeezed_shape(value.shape, axis), stream=_shadow_stream(stream))


def _expand_dims_shadow(value: Any, axis: Any, *args: Any,
                        stream: Any | None = None, **kwargs: Any) -> Any:
    if not _has_precise_array(value):
        value = mx.array(value)
    if (args or kwargs or not _has_precise_array(value)
            or getattr(value, "dtype", None) != mx.float64):
        if stream is None:
            return _ORIGINAL_MX_EXPAND_DIMS(value, axis=axis, *args, **kwargs)
        return _ORIGINAL_MX_EXPAND_DIMS(
            value, axis=axis, *args, stream=stream, **kwargs)
    axes = [axis] if isinstance(axis, int) else list(axis)
    ndim = value.ndim
    shape = list(value.shape)
    for ax in sorted((a + ndim + 1 if a < 0 else a) for a in axes):
        shape.insert(ax, 1)
        ndim += 1
    return _mx_overrides.reshape_precise(
        value, tuple(shape), stream=_shadow_stream(stream))


def _transpose_shadow(value: Any, axes: Any | None = None, *args: Any,
                      stream: Any | None = None, **kwargs: Any) -> Any:
    if not _has_precise_array(value):
        value = mx.array(value)
    if args or kwargs or not _has_precise_array(value):
        if stream is None:
            return _ORIGINAL_MX_TRANSPOSE(value, axes=axes, *args, **kwargs)
        return _ORIGINAL_MX_TRANSPOSE(
            value, axes=axes, *args, stream=stream, **kwargs)
    return _mx_overrides.transpose_precise(
        value, axes, stream=_shadow_stream(stream))


def _swapaxes_shadow(value: Any, axis1: int, axis2: int, *args: Any,
                     stream: Any | None = None, **kwargs: Any) -> Any:
    if not _has_precise_array(value):
        value = mx.array(value)
    if (args or kwargs or getattr(value, "dtype", None) != mx.float64):
        if stream is None:
            return _ORIGINAL_MX_SWAPAXES(value, axis1, axis2, *args, **kwargs)
        return _ORIGINAL_MX_SWAPAXES(
            value, axis1, axis2, *args, stream=stream, **kwargs)
    ndim = value.ndim
    left = axis1 + ndim if axis1 < 0 else axis1
    right = axis2 + ndim if axis2 < 0 else axis2
    if left < 0 or right < 0 or left >= ndim or right >= ndim:
        raise ValueError("swapaxes axis out of bounds")
    axes = list(range(ndim))
    axes[left], axes[right] = axes[right], axes[left]
    return _transpose_shadow(value, axes, stream=stream)


def _moveaxis_shadow(value: Any, source: Any, destination: Any, *args: Any,
                     stream: Any | None = None, **kwargs: Any) -> Any:
    if not _has_precise_array(value):
        value = mx.array(value)
    if (args or kwargs or getattr(value, "dtype", None) != mx.float64):
        if stream is None:
            return _ORIGINAL_MX_MOVEAXIS(
                value, source, destination, *args, **kwargs)
        return _ORIGINAL_MX_MOVEAXIS(
            value, source, destination, *args, stream=stream, **kwargs)
    sources = (source,) if isinstance(source, int) else tuple(source)
    destinations = ((destination,) if isinstance(destination, int)
                    else tuple(destination))
    if len(sources) != len(destinations):
        raise ValueError("source and destination arguments must have the same length")
    ndim = value.ndim

    def normalize(axis):
        axis = axis + ndim if axis < 0 else axis
        if axis < 0 or axis >= ndim:
            raise ValueError("moveaxis axis out of bounds")
        return axis

    normalized_sources = [normalize(int(axis)) for axis in sources]
    normalized_destinations = [normalize(int(axis)) for axis in destinations]
    order = [axis for axis in range(ndim) if axis not in normalized_sources]
    for destination_axis, source_axis in sorted(
            zip(normalized_destinations, normalized_sources)):
        order.insert(destination_axis, source_axis)
    return _transpose_shadow(value, order, stream=stream)


def _conv1d_shadow(input: Any, weight: Any, stride: int = 1,
                   padding: int = 0, dilation: int = 1, groups: int = 1,
                   *args: Any, stream: Any | None = None,
                   **kwargs: Any) -> Any:
    if not _has_precise_array(input):
        input = mx.array(input)
    if not _has_precise_array(weight):
        weight = mx.array(weight)
    wants_float64 = (
        getattr(input, "dtype", None) == mx.float64
        or getattr(weight, "dtype", None) == mx.float64)
    if args or kwargs or not wants_float64:
        if stream is None:
            return _wrap_factory_array(
                _ORIGINAL_MX_CONV1D(
                    input, weight, stride, padding, dilation, groups,
                    *args, **kwargs),
                stream)
        return _wrap_factory_array(
            _ORIGINAL_MX_CONV1D(
                input, weight, stride, padding, dilation, groups,
                *args, stream=stream, **kwargs),
            stream)
    actual_stream = _shadow_stream(stream)
    return _mx_overrides.conv1d_precise(
        input, weight, stride=stride, padding=padding, dilation=dilation,
        groups=groups, stream=actual_stream)


mx.eval = _eval_mx_arrays
mx.stream = _precise_stream
mx.abs = _abs_shadow
mx.square = _square_shadow
mx.add = _add_shadow
mx.subtract = _subtract_shadow
mx.multiply = _multiply_shadow
mx.divide = _divide_shadow
mx.power = _power_shadow
mx.remainder = _remainder_shadow
mx.divmod = _divmod_shadow
mx.matmul = _matmul_shadow
mx.arctan2 = _arctan2_shadow
mx.log = _log_shadow
mx.log2 = _log2_shadow
mx.log10 = _log10_shadow
mx.sin = _sin_shadow
mx.cos = _cos_shadow
mx.arcsin = _arcsin_shadow
mx.arccos = _arccos_shadow
mx.arctan = _arctan_shadow
mx.floor = _floor_shadow
mx.ceil = _ceil_shadow
mx.sqrt = _sqrt_shadow
mx.degrees = _degrees_shadow
mx.radians = _radians_shadow
mx.sort = _sort_shadow
mx.argsort = _argsort_shadow
mx.where = _where_shadow
mx.equal = _equal_shadow
mx.not_equal = _not_equal_shadow
mx.less = _less_shadow
mx.less_equal = _less_equal_shadow
mx.greater = _greater_shadow
mx.greater_equal = _greater_equal_shadow
mx.logical_not = _logical_not_shadow
mx.logical_and = _logical_and_shadow
mx.logical_or = _logical_or_shadow
mx.concatenate = _concatenate_shadow
mx.stack = _stack_shadow
mx.isfinite = _isfinite_shadow
mx.isnan = _isnan_shadow
mx.isinf = _isinf_shadow
mx.isclose = _isclose_shadow
mx.allclose = _allclose_shadow
mx.round = _round_shadow
mx.sum = _sum_shadow
mx.mean = _mean_shadow
mx.min = _min_shadow
mx.max = _max_shadow
mx.minimum = _minimum_shadow
mx.maximum = _maximum_shadow
mx.cumsum = _cumsum_shadow
mx.take = _take_shadow
mx.slice = _slice_shadow
mx.roll = _roll_shadow
mx.clip = _clip_shadow
mx.broadcast_arrays = _broadcast_arrays_shadow
mx.broadcast_to = _broadcast_to_shadow
mx.pad = _pad_shadow
mx.eye = _eye_shadow
mx.identity = _identity_shadow
mx.zeros = _zeros_shadow
mx.ones = _ones_shadow
mx.full = _full_shadow
mx.arange = _arange_shadow
mx.linspace = _linspace_shadow
mx.reshape = _reshape_shadow
if _ORIGINAL_MX_FLATTEN is not None:
    mx.flatten = _flatten_shadow
mx.squeeze = _squeeze_shadow
mx.expand_dims = _expand_dims_shadow
mx.transpose = _transpose_shadow
mx.swapaxes = _swapaxes_shadow
mx.moveaxis = _moveaxis_shadow
mx.repeat = _repeat_shadow
mx.tile = _tile_shadow
mx.conv1d = _conv1d_shadow
mx.deg2rad = mx.radians
mx.rad2deg = mx.degrees
if not hasattr(mx.array, "copy"):
    mx.array.copy = lambda self: mx.array(self)
if not hasattr(mx.array, "ravel"):
    mx.array.ravel = lambda self, *, stream=None: mx.reshape(
        self, (self.size,), stream=stream)
if not hasattr(mx.array, "searchsorted"):
    mx.array.searchsorted = lambda self, v, side="left", sorter=None: searchsorted(
        self, v, side=side, sorter=sorter)
if not hasattr(mx.array, "nonzero"):
    mx.array.nonzero = lambda self: nonzero(self)
if not hasattr(mx.array, "__index__"):
    mx.array.__index__ = lambda self: int(self.item())
if not hasattr(mx.array, "__int__"):
    mx.array.__int__ = lambda self: int(self.item())
if not hasattr(mx.array, "__float__"):
    mx.array.__float__ = lambda self: float(self.item())
if not hasattr(mx.array, "_mlx_array_orig_item"):
    mx.array._mlx_array_orig_item = mx.array.item

    def _array_item(self, *args):
        if not args and getattr(self, "dtype", None) == mx.float64:
            try:
                return _mx_overrides.item_precise(
                    self, stream=_array_stream(self))
            except RuntimeError as exc:
                if "without a primitive" not in str(exc):
                    raise
        return mx.array._mlx_array_orig_item(self, *args)

    mx.array.item = _array_item

if not hasattr(mx.array, "_mlx_array_orig_tolist"):
    mx.array._mlx_array_orig_tolist = mx.array.tolist

    def _array_tolist(self):
        if getattr(self, "dtype", None) != mx.float64:
            return mx.array._mlx_array_orig_tolist(self)
        return _mx_overrides.tolist_precise(self, stream=_array_stream(self))

    mx.array.tolist = _array_tolist
if not hasattr(mx.array, "_mlx_array_orig_squeeze"):
    mx.array._mlx_array_orig_squeeze = mx.array.squeeze

    def _array_squeeze(self, axis=None, *args, **kwargs):
        if not args and not kwargs:
            return _squeeze_shadow(self, axis=axis)
        return mx.array._mlx_array_orig_squeeze(self, axis=axis, *args, **kwargs)

    mx.array.squeeze = _array_squeeze
if not hasattr(mx.array, "_mlx_array_orig_reshape"):
    mx.array._mlx_array_orig_reshape = mx.array.reshape

    def _array_reshape(self, *shape, **kwargs):
        stream = kwargs.pop("stream", None)
        if kwargs:
            return mx.array._mlx_array_orig_reshape(self, *shape, **kwargs)
        if len(shape) == 1:
            new_shape = shape[0]
        else:
            new_shape = shape
        return _reshape_shadow(self, new_shape, stream=stream)

    mx.array.reshape = _array_reshape
if hasattr(mx.array, "flatten") and not hasattr(mx.array, "_mlx_array_orig_flatten"):
    mx.array._mlx_array_orig_flatten = mx.array.flatten

    def _array_flatten(self, start_axis=0, end_axis=-1, *args, **kwargs):
        stream = kwargs.pop("stream", None)
        if kwargs or args:
            return mx.array._mlx_array_orig_flatten(self, *args, **kwargs)
        if getattr(self, "dtype", None) == mx.float64:
            shape = _flattened_shape(tuple(self.shape), start_axis, end_axis)
            return _mx_overrides.reshape_precise(
                self, shape, stream=_shadow_stream(stream))
        if stream is None:
            return mx.array._mlx_array_orig_flatten(
                self, start_axis=start_axis, end_axis=end_axis)
        return mx.array._mlx_array_orig_flatten(
            self, start_axis=start_axis, end_axis=end_axis, stream=stream)

    mx.array.flatten = _array_flatten
if hasattr(mx.array, "ravel") and not hasattr(mx.array, "_mlx_array_orig_ravel"):
    mx.array._mlx_array_orig_ravel = mx.array.ravel

    def _array_ravel(self, *args, **kwargs):
        stream = kwargs.pop("stream", None)
        if kwargs or args:
            return mx.array._mlx_array_orig_ravel(self, *args, **kwargs)
        if getattr(self, "dtype", None) == mx.float64:
            return _mx_overrides.reshape_precise(
                self, (self.size,), stream=_shadow_stream(stream))
        if stream is None:
            return mx.array._mlx_array_orig_ravel(self)
        return mx.array._mlx_array_orig_ravel(self, stream=stream)

    mx.array.ravel = _array_ravel
if not hasattr(mx.array, "_mlx_array_orig_cumsum"):
    mx.array._mlx_array_orig_cumsum = mx.array.cumsum

    def _array_cumsum(self, axis=None, *args, **kwargs):
        return _cumsum_shadow(self, axis=axis, *args, **kwargs)

    mx.array.cumsum = _array_cumsum
if not hasattr(mx.array, "__round__"):
    mx.array.__round__ = lambda self, ndigits=None: round(
        float(self.item()), ndigits) if ndigits is not None else round(
            float(self.item()))
if not hasattr(mx.array, "__deepcopy__"):
    def _array_deepcopy(self, memo):
        return _mx_overrides.MlxPreciseArray(self, stream=_active_stream())

    mx.array.__deepcopy__ = _array_deepcopy
if not hasattr(mx.array, "__divmod__"):
    mx.array.__divmod__ = lambda self, other: _divmod_shadow(self, other)
if not hasattr(mx.array, "__rdivmod__"):
    mx.array.__rdivmod__ = lambda self, other: _divmod_shadow(other, self)
if not hasattr(mx.array, "__rand__"):
    mx.array.__rand__ = lambda self, other: mx.logical_and(_to_mx(other), self)
if not hasattr(mx.array, "__ror__"):
    mx.array.__ror__ = lambda self, other: mx.logical_or(_to_mx(other), self)
if not hasattr(mx.array, "__rxor__"):
    mx.array.__rxor__ = lambda self, other: mx.not_equal(_to_mx(other), self)
if not hasattr(mx.array, "_mlx_array_orig_eq"):
    mx.array._mlx_array_orig_eq = mx.array.__eq__
    mx.array.__eq__ = lambda self, other: _equal_shadow(self, other)
if not hasattr(mx.array, "_mlx_array_orig_ne"):
    mx.array._mlx_array_orig_ne = mx.array.__ne__
    mx.array.__ne__ = lambda self, other: _not_equal_shadow(self, other)
if not hasattr(mx.array, "_mlx_array_orig_lt"):
    mx.array._mlx_array_orig_lt = mx.array.__lt__
    mx.array.__lt__ = lambda self, other: _less_shadow(self, other)
if not hasattr(mx.array, "_mlx_array_orig_le"):
    mx.array._mlx_array_orig_le = mx.array.__le__
    mx.array.__le__ = lambda self, other: _less_equal_shadow(self, other)
if not hasattr(mx.array, "_mlx_array_orig_gt"):
    mx.array._mlx_array_orig_gt = mx.array.__gt__
    mx.array.__gt__ = lambda self, other: _greater_shadow(self, other)
if not hasattr(mx.array, "_mlx_array_orig_ge"):
    mx.array._mlx_array_orig_ge = mx.array.__ge__
    mx.array.__ge__ = lambda self, other: _greater_equal_shadow(self, other)
if not hasattr(mx.array, "_mlx_array_orig_abs"):
    mx.array._mlx_array_orig_abs = mx.array.__abs__
    mx.array.__abs__ = lambda self: _abs_shadow(self)
if not hasattr(mx.array, "data"):
    mx.array.data = property(lambda self: self)
if not hasattr(mx.array, "take"):
    mx.array.take = lambda self, indices, axis=None, mode=None: take(
        self, indices, axis=axis, mode=mode)
if not hasattr(mx.array, "_mlx_array_orig_mul"):
    mx.array._mlx_array_orig_mul = mx.array.__mul__

    def _array_mul(self, other):
        if isinstance(other, (list, tuple)):
            other = mx.array(other, dtype=self.dtype)
        return _multiply_shadow(self, other)

    mx.array.__mul__ = _array_mul
if not hasattr(mx.array, "_mlx_array_orig_rmul"):
    mx.array._mlx_array_orig_rmul = mx.array.__rmul__

    def _array_rmul(self, other):
        if isinstance(other, (list, tuple)):
            other = mx.array(other, dtype=self.dtype)
        return _multiply_shadow(other, self)

    mx.array.__rmul__ = _array_rmul
if not hasattr(mx.array, "_mlx_array_orig_truediv"):
    mx.array._mlx_array_orig_truediv = mx.array.__truediv__

    def _array_truediv(self, other):
        if isinstance(other, (list, tuple)):
            other = mx.array(other, dtype=self.dtype)
        return _divide_shadow(self, other)

    mx.array.__truediv__ = _array_truediv
if not hasattr(mx.array, "_mlx_array_orig_rtruediv"):
    mx.array._mlx_array_orig_rtruediv = mx.array.__rtruediv__

    def _array_rtruediv(self, other):
        if isinstance(other, (list, tuple)):
            other = mx.array(other, dtype=self.dtype)
        return _divide_shadow(other, self)

    mx.array.__rtruediv__ = _array_rtruediv
if not hasattr(mx.array, "_mlx_array_orig_astype"):
    mx.array._mlx_array_orig_astype = mx.array.astype

    def _array_astype(self, dtype, *args, **kwargs):
        stream = kwargs.pop("stream", None)
        if not args and not kwargs:
            return _mx_overrides.astype_precise(
                self, _unwrap_dtype(dtype), stream=_shadow_stream(stream))
        if stream is not None:
            kwargs["stream"] = stream
        return mx.array._mlx_array_orig_astype(
            self, _unwrap_dtype(dtype), *args, **kwargs)

    mx.array.astype = _array_astype
if not hasattr(mx.array, "_mlx_array_orig_min"):
    mx.array._mlx_array_orig_min = mx.array.min

    def _array_min(self, axis=None, *args, **kwargs):
        if not args and not kwargs:
            return _mx_overrides.reduce_minmax_precise(
                self, axis=axis, is_max=False, stream=_active_stream())
        return mx.array._mlx_array_orig_min(self, axis=axis, *args, **kwargs)

    mx.array.min = _array_min
if not hasattr(mx.array, "_mlx_array_orig_max"):
    mx.array._mlx_array_orig_max = mx.array.max

    def _array_max(self, axis=None, *args, **kwargs):
        if not args and not kwargs:
            return _mx_overrides.reduce_minmax_precise(
                self, axis=axis, is_max=True, stream=_active_stream())
        return mx.array._mlx_array_orig_max(self, axis=axis, *args, **kwargs)

    mx.array.max = _array_max
if not hasattr(mx.array, "_mlx_array_orig_setitem"):
    mx.array._mlx_array_orig_setitem = mx.array.__setitem__

    def _normalize_basic_setitem_key(array, key):
        def scalar_index(value):
            if isinstance(value, _ORIGINAL_MX_ARRAY) and value.size == 1:
                if value.dtype == mx.bool_ or str(value.dtype).endswith("bool"):
                    return value
                return int(value.item())
            return value

        if isinstance(key, tuple):
            parts = tuple(scalar_index(part) for part in key)
        else:
            parts = (scalar_index(key),)

        if _builtins.any(part is None for part in parts):
            return None

        ellipsis_at = next(
            (index for index, part in enumerate(parts) if part is Ellipsis),
            None)
        specified = _builtins.sum(1 for part in parts if part is not Ellipsis)
        if ellipsis_at is not None:
            fill = array.ndim - (specified - 1)
            if fill < 0:
                raise IndexError(
                    f"too many indices for array with {array.ndim} dimensions")
            parts = (parts[:ellipsis_at] + (slice(None),) * fill
                     + parts[ellipsis_at + 1:])
        else:
            fill = array.ndim - specified
            if fill < 0:
                raise IndexError(
                    f"too many indices for array with {array.ndim} dimensions")
            parts = parts + (slice(None),) * fill

        starts = []
        stops = []
        strides = []
        exposed_shape = []
        full_shape = []
        for axis, part in enumerate(parts):
            dim = int(array.shape[axis])
            if isinstance(part, slice):
                start, stop, step = part.indices(dim)
                length = len(range(start, stop, step))
                starts.append(start)
                stops.append(stop)
                strides.append(step)
                exposed_shape.append(length)
                full_shape.append(length)
                continue
            try:
                index = operator.index(part)
            except TypeError:
                return None
            if index < 0:
                index += dim
            if index < 0 or index >= dim:
                raise IndexError("index out of bounds")
            starts.append(index)
            stops.append(index + 1)
            strides.append(1)
            full_shape.append(1)

        return (
            tuple(starts),
            tuple(stops),
            tuple(strides),
            tuple(exposed_shape),
            tuple(full_shape),
        )

    def _setitem_update_value(value, dtype, exposed_shape, full_shape, stream):
        if isinstance(value, (list, tuple)):
            value = mx.array(value, dtype=dtype, stream=stream)
        elif not isinstance(value, _ORIGINAL_MX_ARRAY):
            value = mx.array(value, dtype=dtype, stream=stream)
        elif value.dtype != dtype:
            value = value.astype(dtype, stream=stream)

        value_shape = tuple(value.shape)
        if exposed_shape:
            if value_shape != exposed_shape:
                value = mx.broadcast_to(value, exposed_shape, stream=stream)
        elif value_shape != ():
            if value.size != 1:
                raise ValueError("cannot assign non-scalar value to scalar slice")
            value = mx.reshape(value, (), stream=stream)

        if tuple(value.shape) != full_shape:
            value = mx.reshape(value, full_shape, stream=stream)
        return value

    def _array_setitem(self, key, value):
        if isinstance(self, _mx_overrides.MlxPreciseArray):
            normalized = _normalize_basic_setitem_key(self, key)
            if normalized is not None:
                starts, stops, strides, exposed_shape, full_shape = normalized
                stream = _array_stream(self)
                update = _setitem_update_value(
                    value, self.dtype, exposed_shape, full_shape, stream)
                _mx_overrides.setitem_precise(
                    self, update, starts, stops, strides, stream=stream)
                return None
        if isinstance(value, (list, tuple)):
            value = mx.array(value, dtype=self.dtype)
        return _ORIGINAL_MX_ARRAY._mlx_array_orig_setitem(self, key, value)

    mx.array.__setitem__ = _array_setitem
if not hasattr(mx.array, "_mlx_array_orig_getitem"):
    mx.array._mlx_array_orig_getitem = mx.array.__getitem__

    def _array_getitem(self, key):
        stream = _array_stream(self)
        dtype = getattr(self, "dtype", None)
        field_names = getattr(dtype, "names", None)
        if isinstance(key, str) and field_names and key in field_names:
            return self[(slice(None), field_names.index(key))]
        if (isinstance(self, _mx_overrides.MlxPreciseArray)
                and getattr(self, "dtype", None) == mx.float64):
            try:
                _mx_overrides.eval_precise_array(self)
            except RuntimeError as exc:
                if "without a primitive" not in str(exc):
                    raise

        def scalar_index(value):
            if isinstance(value, _ORIGINAL_MX_ARRAY) and value.size == 1:
                if value.dtype == mx.bool_ or str(value.dtype).endswith("bool"):
                    return value
                return int(value.item())
            return value

        def normalize(part):
            if isinstance(part, slice):
                return slice(scalar_index(part.start),
                             scalar_index(part.stop),
                             scalar_index(part.step))
            return scalar_index(part)

        if isinstance(key, tuple):
            key = tuple(normalize(part) for part in key)
            ellipsis_at = next(
                (index for index, part in enumerate(key) if part is Ellipsis),
                None)
            if ellipsis_at is not None:
                consumed = _builtins.sum(
                    1 for part in key if part is not Ellipsis and part is not None)
                fill = self.ndim - consumed
                key = (key[:ellipsis_at] + (slice(None),) * fill
                       + key[ellipsis_at + 1:])
        else:
            key = normalize(key)

        def apply_mx_getitem(value, item):
            parts = item if isinstance(item, tuple) else (item,)
            result = value
            axis = 0
            for part in parts:
                if part is None:
                    result = mx.expand_dims(result, axis=axis, stream=stream)
                    axis += 1
                    continue
                if part is Ellipsis:
                    axis = result.ndim - (len(parts) - parts.index(part) - 1)
                    continue
                if axis >= result.ndim:
                    return NotImplemented
                dim = result.shape[axis]
                if isinstance(part, slice):
                    start, stop, step = part.indices(dim)
                    if step < 0:
                        indices = mx.arange(
                            start, stop, step, dtype=mx.int32, stream=stream)
                        result = mx.take(result, indices, axis=axis, stream=stream)
                    elif step == 1:
                        starts = [0] * result.ndim
                        stops = list(result.shape)
                        strides = [1] * result.ndim
                        starts[axis] = start
                        stops[axis] = _builtins.max(start, stop)
                        result = _mx_overrides.slice_precise(
                            result, starts, stops, strides, stream=stream)
                    else:
                        starts = [0] * result.ndim
                        stops = list(result.shape)
                        strides = [1] * result.ndim
                        starts[axis] = start
                        stops[axis] = stop
                        strides[axis] = step
                        result = _mx_overrides.slice_precise(
                            result, starts, stops, strides, stream=stream)
                    axis += 1
                    continue
                if isinstance(part, int):
                    index = part + dim if part < 0 else part
                    if index < 0 or index >= dim:
                        raise IndexError("index out of bounds")
                    starts = [0] * result.ndim
                    stops = list(result.shape)
                    strides = [1] * result.ndim
                    starts[axis] = index
                    stops[axis] = index + 1
                    result = _mx_overrides.slice_precise(
                        result, starts, stops, strides, stream=stream)
                    shape = list(result.shape)
                    del shape[axis]
                    result = mx.reshape(result, tuple(shape), stream=stream)
                    continue
                if isinstance(part, (list, tuple)):
                    if _builtins.all(isinstance(index, int) for index in part):
                        pieces = []
                        for raw_index in part:
                            index = raw_index + dim if raw_index < 0 else raw_index
                            if index < 0 or index >= dim:
                                raise IndexError("index out of bounds")
                            starts = [0] * result.ndim
                            stops = list(result.shape)
                            strides = [1] * result.ndim
                            starts[axis] = index
                            stops[axis] = index + 1
                            pieces.append(_mx_overrides.slice_precise(
                                result, starts, stops, strides, stream=stream))
                        result = mx.concatenate(pieces, axis=axis, stream=stream)
                    else:
                        indices = mx.array(part, dtype=mx.int32, stream=stream)
                        result = mx.take(result, indices, axis=axis, stream=stream)
                    axis += 1
                    continue
                if isinstance(part, _ORIGINAL_MX_ARRAY):
                    if part.dtype == mx.bool_ or str(part.dtype).endswith("bool"):
                        indices = nonzero(part)[0]
                    else:
                        indices = part.astype(mx.int32)
                    result = take(result, indices, axis=axis)
                    axis += 1
                    continue
                return NotImplemented
            return result

        handled = apply_mx_getitem(self, key)
        if handled is not NotImplemented:
            return handled
        try:
            return _ORIGINAL_MX_ARRAY._mlx_array_orig_getitem(self, key)
        except (NotImplementedError, ValueError):
            raise

    mx.array.__getitem__ = _array_getitem
if not hasattr(mx.array, "_mlx_array_orig_iter"):
    mx.array._mlx_array_orig_iter = mx.array.__iter__

    def _array_iter(self):
        if self.ndim == 0:
            raise TypeError("iteration over a 0-d array")
        for index in range(self.shape[0]):
            yield self[index]

    mx.array.__iter__ = _array_iter
if not hasattr(mx.array, "_mlx_array_orig_matmul"):
    mx.array._mlx_array_orig_matmul = mx.array.__matmul__

    def _array_matmul(self, other):
        if isinstance(other, (list, tuple)):
            other = mx.array([_to_scalar(item) for item in other],
                             dtype=self.dtype)
        return _matmul_shadow(self, other)

    mx.array.__matmul__ = _array_matmul
if not hasattr(mx.array, "_mlx_array_orig_rmatmul"):
    mx.array._mlx_array_orig_rmatmul = getattr(mx.array, "__rmatmul__", None)

    def _array_rmatmul(self, other):
        if isinstance(other, (list, tuple)):
            other = mx.array([_to_scalar(item) for item in other],
                             dtype=self.dtype)
        return _matmul_shadow(other, self)

    mx.array.__rmatmul__ = _array_rmatmul
if not hasattr(mx.array, "_mlx_array_orig_sub"):
    mx.array._mlx_array_orig_sub = mx.array.__sub__

    def _array_sub(self, other):
        if isinstance(other, (list, tuple)):
            other = mx.array([_to_scalar(item) for item in other],
                             dtype=self.dtype)
        return _subtract_shadow(self, other)

    mx.array.__sub__ = _array_sub
if not hasattr(mx.array, "_mlx_array_orig_rsub"):
    mx.array._mlx_array_orig_rsub = getattr(mx.array, "__rsub__", None)

    def _array_rsub(self, other):
        if isinstance(other, (list, tuple)):
            other = mx.array([_to_scalar(item) for item in other],
                             dtype=self.dtype)
        return _subtract_shadow(other, self)

    mx.array.__rsub__ = _array_rsub
if not hasattr(mx.array, "_mlx_array_orig_add"):
    mx.array._mlx_array_orig_add = mx.array.__add__

    def _array_add(self, other):
        if isinstance(other, (list, tuple)):
            other = mx.array([_to_scalar(item) for item in other],
                             dtype=self.dtype)
        return _add_shadow(self, other)

    mx.array.__add__ = _array_add
if not hasattr(mx.array, "_mlx_array_orig_radd"):
    mx.array._mlx_array_orig_radd = getattr(mx.array, "__radd__", None)

    def _array_radd(self, other):
        if isinstance(other, (list, tuple)):
            other = mx.array([_to_scalar(item) for item in other],
                             dtype=self.dtype)
        return _add_shadow(self, other)

    mx.array.__radd__ = _array_radd
if not hasattr(mx.array, "add"):
    mx.array.add = lambda self, other, *, stream=None: _add_shadow(
        self, other, stream=stream)
if not hasattr(mx.array, "subtract"):
    mx.array.subtract = lambda self, other, *, stream=None: _subtract_shadow(
        self, other, stream=stream)
if not hasattr(mx.array, "multiply"):
    mx.array.multiply = lambda self, other, *, stream=None: _multiply_shadow(
        self, other, stream=stream)
if not hasattr(mx.array, "divide"):
    mx.array.divide = lambda self, other, *, stream=None: _divide_shadow(
        self, other, stream=stream)
if not hasattr(mx.array, "matmul"):
    mx.array.matmul = lambda self, other, *, stream=None: _matmul_shadow(
        self, other, stream=stream)
if not hasattr(mx.array, "flat"):
    def _array_flat(self):
        def flatten(value):
            if isinstance(value, list):
                for item in value:
                    yield from flatten(item)
            else:
                yield value
        return list(flatten(self.tolist()))

    mx.array.flat = property(_array_flat)
if not hasattr(mx.array, "flags"):
    class _ArrayFlags:
        @property
        def writeable(self):
            return True

        @writeable.setter
        def writeable(self, value):
            pass

    mx.array.flags = property(lambda self: _ArrayFlags())

mx.array = _mx_overrides.MlxPreciseArray
mx.array.__index__ = lambda self: int(self.item())
mx.array.__int__ = lambda self: int(self.item())
mx.array.__float__ = lambda self: float(self.item())
mx.array.__format__ = lambda self, spec: format(self.item(), spec)
mx.array.__deepcopy__ = lambda self, memo: _mx_overrides.MlxPreciseArray(
    self, stream=_active_stream())
mx.array.__divmod__ = lambda self, other: _divmod_shadow(self, other)
mx.array.__rdivmod__ = lambda self, other: _divmod_shadow(other, self)

@dataclass(frozen=True)
class DType:
    """A small callable dtype wrapper around an MLX dtype.

    Dtypes like ``mx.uint8`` are both valid ``dtype=`` values and callable
    scalar/array constructors. MLX exposes dtype objects but they are not
    callable, so we wrap them here.
    """

    mx_dtype: Any
    name: str
    kind: str
    itemsize: int
    char: str

    def __call__(self, x: Any) -> Any:
        if self.mx_dtype is _builtins.object:
            return x
        arr = _construct_mx_array(x, dtype=self.mx_dtype)
        return arr.item() if arr.size == 1 else arr

    @property
    def type(self) -> "DType":
        return self

    def __eq__(self, other: Any) -> bool:
        return _unwrap_dtype(other) == self.mx_dtype

    def __hash__(self) -> int:
        return hash((self.name, self.mx_dtype))

    def __repr__(self) -> str:  # pragma: no cover
        return f"mx.{self.name}"


_STRING_TO_MX_DTYPE = {
    "?": mx.bool_,
    "bool": mx.bool_,
    "bool_": mx.bool_,
    "b": mx.int8,
    "i1": mx.int8,
    "int8": mx.int8,
    "h": mx.int16,
    "i2": mx.int16,
    "int16": mx.int16,
    "i": mx.int32,
    "i4": mx.int32,
    "int32": mx.int32,
    "l": mx.int64,
    "q": mx.int64,
    "i8": mx.int64,
    "int": mx.int64,
    "int64": mx.int64,
    "B": mx.uint8,
    "u1": mx.uint8,
    "uint8": mx.uint8,
    "H": mx.uint16,
    "u2": mx.uint16,
    "uint16": mx.uint16,
    "I": mx.uint32,
    "u4": mx.uint32,
    "uint32": mx.uint32,
    "L": mx.uint64,
    "Q": mx.uint64,
    "u8": mx.uint64,
    "uint64": mx.uint64,
    "e": mx.float16,
    "f2": mx.float16,
    "float16": mx.float16,
    "f": mx.float32,
    "f4": mx.float32,
    "float32": mx.float32,
    "d": mx.float64,
    "f8": mx.float64,
    "float": mx.float64,
    "float64": mx.float64,
    "double": mx.float64,
    "g": mx.float64,
    "longdouble": mx.float64,
    "float128": mx.float64,
}


def _unwrap_dtype(dtype: Any | None) -> Any | None:
    if isinstance(dtype, DType):
        return dtype.mx_dtype
    if dtype is _builtins.bool:
        return mx.bool_
    if dtype is _builtins.int:
        return mx.int64
    if dtype is _builtins.float:
        return mx.float64
    if isinstance(dtype, str):
        if dtype.startswith(("S", "U")) or dtype in {"O", "object", "str", "bytes"}:
            return _builtins.object
        if dtype[:1] in {"<", ">", "=", "|", "!"}:
            dtype = dtype[1:]
        return _STRING_TO_MX_DTYPE.get(dtype, dtype)
    return dtype


# Public dtypes (MLXArrayBackend-like: usable as dtype= and callable constructors).
bool_ = DType(mx.bool_, "bool_", "b", 1, "?")
float16 = DType(mx.float16, "float16", "f", 2, "e")
float32 = DType(mx.float32, "float32", "f", 4, "f")
float64 = DType(mx.float64, "float64", "f", 8, "d")
bfloat16 = DType(mx.bfloat16, "bfloat16", "f", 2, "E")
int8 = DType(mx.int8, "int8", "i", 1, "b")
int16 = DType(mx.int16, "int16", "i", 2, "h")
int32 = DType(mx.int32, "int32", "i", 4, "i")
int64 = DType(mx.int64, "int64", "i", 8, "q")
uint8 = DType(mx.uint8, "uint8", "u", 1, "B")
uint16 = DType(mx.uint16, "uint16", "u", 2, "H")
uint32 = DType(mx.uint32, "uint32", "u", 4, "I")
uint64 = DType(mx.uint64, "uint64", "u", 8, "Q")
longdouble = float64
float128 = float64
floating = float
integer = int
number = (int, float)
_MLX_FLOATING_CATEGORY = getattr(mx, "floating", None)
_MLX_INTEGER_CATEGORY = getattr(mx, "integer", None)
_object_dtype = DType(_builtins.object, "object", "O", 0, "O")
_DTYPE_BY_MX = {
    dt.mx_dtype: dt for dt in (
        bool_, float16, float32, float64, bfloat16,
        int8, int16, int32, int64, uint8, uint16, uint32, uint64,
    )
}
_DTYPE_BY_NAME = {
    **{dt.name: dt for dt in _DTYPE_BY_MX.values()},
    **{dt.char: dt for dt in _DTYPE_BY_MX.values()},
    "bool": bool_,
    "float": float64,
    "double": float64,
    "int": int64,
    "longdouble": longdouble,
    "float128": float128,
    "O": _object_dtype,
    "object": _object_dtype,
}
_DTYPE_BY_NAME.update({
    "u1": uint8, "u2": uint16, "u4": uint32, "u8": uint64,
    "i1": int8, "i2": int16, "i4": int32, "i8": int64,
    "f2": float16, "f4": float32, "f8": float64,
})

if not hasattr(type(mx.float32), "kind"):
    type(mx.float32).kind = property(lambda self: _DTYPE_BY_MX.get(self, _object_dtype).kind)
if not hasattr(type(mx.float32), "char"):
    type(mx.float32).char = property(lambda self: _DTYPE_BY_MX.get(self, _object_dtype).char)
if not hasattr(type(mx.float32), "itemsize"):
    type(mx.float32).itemsize = property(lambda self: _DTYPE_BY_MX.get(self, _object_dtype).itemsize)
if not hasattr(type(mx.float32), "isnative"):
    type(mx.float32).isnative = property(lambda self: True)
if not hasattr(type(mx.float32), "_mlx_array_orig_eq"):
    type(mx.float32)._mlx_array_orig_eq = type(mx.float32).__eq__

    def _mx_dtype_eq(self, other):
        other = _unwrap_dtype(other)
        if isinstance(other, type(mx.float32)):
            return repr(self) == repr(other)
        return False

    type(mx.float32).__eq__ = _mx_dtype_eq

if not hasattr(type(mx.float32), "_mlx_array_call"):
    def _mx_dtype_call(self, value):
        arr = _construct_mx_array(value, dtype=self)
        return arr.item() if arr.size == 1 else arr

    type(mx.float32)._mlx_array_call = _mx_dtype_call
    type(mx.float32).__call__ = _mx_dtype_call


def dtype(value: Any, *args: Any, copy: bool | None = None, **kwargs: Any) -> DType:
    if isinstance(value, DType):
        return value
    if value in _DTYPE_BY_MX:
        return _DTYPE_BY_MX[value]
    if value is _builtins.bool:
        return bool_
    if value is _builtins.int:
        return int64
    if value is _builtins.float:
        return float64
    if value is _builtins.object:
        return _object_dtype
    if isinstance(value, str):
        if value.startswith(("S", "U")):
            return _object_dtype
        if value in _DTYPE_BY_NAME:
            return _DTYPE_BY_NAME[value]
    raise AttributeError("dtype")

# MLXArrayBackend-like scalars/constants
pi = math.pi
e = math.e
inf = float("inf")
nan = float("nan")
newaxis = None

ndarray = mx.array


class flatiter:
    """Placeholder for MLXArrayBackend's ndarray.flat iterator type."""


@dataclass(frozen=True)
class _FInfo:
    _info: Any
    tiny: float

    @property
    def dtype(self) -> Any:
        return self._info.dtype

    @property
    def eps(self) -> float:
        return self._info.eps

    @property
    def max(self) -> float:
        return self._info.max

    @property
    def min(self) -> float:
        return self._info.min


def finfo(dtype: Any) -> _FInfo:
    mx_dtype = _unwrap_dtype(dtype)
    tiny = {
        mx.float16: 6.103515625e-05,
        mx.float32: 1.1754943508222875e-38,
        mx.float64: 2.2250738585072014e-308,
    }.get(mx_dtype, 0.0)
    return _FInfo(mx.finfo(mx_dtype), tiny)


_py_min = _builtins.min
_py_max = _builtins.max


def _copy_nested(value: Any) -> Any:
    if isinstance(value, _ORIGINAL_MX_ARRAY):
        return value.tolist()
    if isinstance(value, tuple):
        return [_copy_nested(v) for v in value]
    if isinstance(value, list):
        return [_copy_nested(v) for v in value]
    if isinstance(value, range):
        return list(value)
    return value


def _infer_shape(value: Any) -> Tuple[int, ...]:
    if not isinstance(value, list):
        return ()
    if not value:
        return (0,)
    return (len(value),) + _infer_shape(value[0])


def _reshape_flat(flat: list[Any], shape: Tuple[int, ...]) -> Any:
    if not shape:
        return flat[0]
    step = math.prod(shape[1:]) if len(shape) > 1 else 1
    return [_reshape_flat(flat[i * step:(i + 1) * step], shape[1:])
            for i in range(shape[0])]


def _squeeze_nested(value: Any) -> Any:
    if isinstance(value, list) and len(value) == 1:
        return _squeeze_nested(value[0])
    if isinstance(value, list):
        return [_squeeze_nested(v) for v in value]
    return value


def _contains_object_data(value: Any) -> bool:
    if isinstance(value, (str, bytes, datetime, timedelta)) or value is None:
        return True
    if isinstance(value, (list, tuple)):
        return _builtins.any(_contains_object_data(v) for v in value)
    return False


def _image_to_nested(image: Any) -> list[Any]:
    width, height = image.size
    data_getter = getattr(image, "get_flattened_data", None)
    pixels = list(data_getter() if data_getter is not None else image.getdata())
    bands = len(image.getbands())
    rows = []
    for row in range(height):
        start = row * width
        stop = start + width
        row_pixels = pixels[start:stop]
        if bands == 1:
            rows.append(row_pixels)
        else:
            rows.append([list(pixel) for pixel in row_pixels])
    return rows


def _shape_tuple(shape: Any) -> Tuple[int, ...]:
    if isinstance(shape, mx.array):
        shape = shape.tolist()
    if isinstance(shape, int):
        return (shape,)
    if shape is None:
        return ()
    return tuple(int(v) for v in shape)


def _python_getitem(data: Any, key: Any) -> Any:
    if not isinstance(key, tuple):
        if isinstance(key, mx.array):
            key = key.tolist()
        if isinstance(key, (list, tuple)):
            return [data[int(item)] for item in key]
        return data[key]
    if not key:
        return data
    first, *rest = key
    if isinstance(first, mx.array):
        first = first.tolist()
    if first is None:
        return [_python_getitem(data, tuple(rest))]
    if (isinstance(first, tuple)
            and _builtins.all(isinstance(item, int) for item in first)):
        first = list(first)
    if isinstance(first, list):
        selected = [data[int(item)] for item in first]
    else:
        selected = data[first]
    if rest and isinstance(first, (slice, list)):
        return [_python_getitem(item, tuple(rest)) for item in selected]
    if rest:
        return _python_getitem(selected, tuple(rest))
    return selected


def _to_mx(x: Any, dtype: Any | None = None) -> mx.array:
    if isinstance(x, range):
        x = list(x)
    if isinstance(x, MaskedArray):
        x = x.data
    if (hasattr(x, "getdata") and hasattr(x, "getbands")
            and hasattr(x, "size")):
        if dtype is None:
            dtype = mx.uint8
        x = _image_to_nested(x)
    elif hasattr(x, "__array__") and not isinstance(x, mx.array):
        try:
            x = x.__array__()
        except TypeError:
            x = x.__array__(dtype=dtype)
    if isinstance(x, _ORIGINAL_MX_ARRAY) and dtype is None:
        return x
    dtype = _unwrap_dtype(dtype)
    if isinstance(x, (list, tuple)):
        x = _copy_nested(x)
    if dtype is _builtins.object or _contains_object_data(x):
        raise TypeError("Python object data is not MLX tensor data")
    if dtype is None:
        try:
            return _construct_mx_array(x)
        except (TypeError, ValueError):
            raise TypeError("Python object data is not MLX tensor data")
    return _construct_mx_array(x, dtype=dtype)


def _to_scalar(x: Any) -> Any:
    if isinstance(x, _ORIGINAL_MX_ARRAY) and x.size == 1:
        return x.item()
    return x


def array(obj: Any, dtype: Any | None = None, *,
          stream: Any | None = None) -> mx.array:
    return _construct_mx_array(obj, dtype=dtype, stream=stream)


def atleast_1d(*arys: Any) -> Tuple[mx.array, ...] | mx.array:
    res = []
    for a in arys:
        arr = _to_mx(a)
        if arr.ndim == 0:
            arr = mx.reshape(arr, (1,))
        res.append(arr)
    return tuple(res) if len(res) > 1 else res[0]


def atleast_2d(*arys: Any) -> Tuple[mx.array, ...] | mx.array:
    res = []
    for a in arys:
        arr = _to_mx(a)
        if arr.ndim == 0:
            arr = mx.reshape(arr, (1, 1))
        elif arr.ndim == 1:
            arr = mx.reshape(arr, (1, arr.shape[0]))
        res.append(arr)
    return tuple(res) if len(res) > 1 else res[0]


def atleast_3d(*arys: Any) -> Tuple[mx.array, ...] | mx.array:
    res = []
    for a in arys:
        arr = _to_mx(a)
        if arr.ndim == 0:
            arr = mx.reshape(arr, (1, 1, 1))
        elif arr.ndim == 1:
            arr = mx.reshape(arr, (1, arr.shape[0], 1))
        elif arr.ndim == 2:
            arr = mx.reshape(arr, (1, *arr.shape))
        res.append(arr)
    return tuple(res) if len(res) > 1 else res[0]


def zeros(shape: Any, dtype: Any | None = None) -> mx.array:
    if _unwrap_dtype(dtype) is _builtins.object:
        raise TypeError("Python object data is not MLX tensor data")
    return mx.zeros(shape, dtype=_unwrap_dtype(dtype))


def ones(shape: Any, dtype: Any | None = None) -> mx.array:
    if _unwrap_dtype(dtype) is _builtins.object:
        raise TypeError("Python object data is not MLX tensor data")
    return mx.ones(shape, dtype=_unwrap_dtype(dtype))


def full(shape: Any, fill_value: Any, dtype: Any | None = None) -> mx.array:
    if _unwrap_dtype(dtype) is _builtins.object or _contains_object_data(fill_value):
        raise TypeError("Python object data is not MLX tensor data")
    return mx.full(shape, fill_value, dtype=_unwrap_dtype(dtype))


def empty(shape: Any, dtype: Any | None = None) -> mx.array:
    if dtype is _builtins.object or dtype == "object":
        raise TypeError("Python object data is not MLX tensor data")
    # MLX does not expose uninitialized arrays; use zeros as a safe fallback.
    return mx.zeros(shape, dtype=_unwrap_dtype(dtype))


def zeros_like(a: Any, dtype: Any | None = None) -> mx.array:
    arr = _to_mx(a)
    return mx.zeros(arr.shape, dtype=_unwrap_dtype(dtype) or arr.dtype)


def ones_like(a: Any, dtype: Any | None = None) -> mx.array:
    arr = _to_mx(a)
    return mx.ones(arr.shape, dtype=_unwrap_dtype(dtype) or arr.dtype)


def full_like(a: Any, fill_value: Any, dtype: Any | None = None) -> mx.array:
    arr = _to_mx(a)
    if _contains_object_data(fill_value):
        raise TypeError("Python object data is not MLX tensor data")
    return mx.full(arr.shape, fill_value, dtype=_unwrap_dtype(dtype) or arr.dtype)


def empty_like(a: Any, dtype: Any | None = None) -> mx.array:
    arr = _to_mx(a)
    return mx.zeros(arr.shape, dtype=_unwrap_dtype(dtype) or arr.dtype)


def arange(*args: Any, **kwargs: Any) -> mx.array:
    args = tuple(_to_scalar(arg) if isinstance(arg, mx.array) else arg for arg in args)
    if "dtype" in kwargs:
        kwargs["dtype"] = _unwrap_dtype(kwargs["dtype"])
    return mx.arange(*args, **kwargs)


def linspace(*args: Any, **kwargs: Any) -> mx.array:
    if "dtype" in kwargs:
        kwargs["dtype"] = _unwrap_dtype(kwargs["dtype"])
    return mx.linspace(*args, **kwargs)


def logspace(start: float, stop: float, num: int = 50, *,
             base: float = 10.0, dtype: Any | None = None,
             stream: Any | None = None) -> mx.array:
    actual_stream = _shadow_stream(stream)
    target_dtype = _unwrap_dtype(dtype) or mx.float64
    exponents = mx.linspace(
        start, stop, num, dtype=target_dtype, stream=actual_stream)
    base_value = mx.array(base, dtype=target_dtype, stream=actual_stream)
    return mx.power(base_value, exponents, stream=actual_stream)


def geomspace(start: float, stop: float, num: int = 50, *,
              dtype: Any | None = None,
              stream: Any | None = None) -> mx.array:
    actual_stream = _shadow_stream(stream)
    target_dtype = _unwrap_dtype(dtype) or mx.float64
    start_value = mx.array(start, dtype=target_dtype, stream=actual_stream)
    stop_value = mx.array(stop, dtype=target_dtype, stream=actual_stream)
    if num == 0:
        return mx.zeros((0,), dtype=target_dtype, stream=actual_stream)
    if num == 1:
        return mx.reshape(start_value, (1,), stream=actual_stream)
    t = mx.linspace(0.0, 1.0, num, dtype=target_dtype, stream=actual_stream)
    return start_value * mx.power(stop_value / start_value, t,
                                  stream=actual_stream)


def reshape(a: Any, newshape: Any, *, stream: Any | None = None) -> mx.array:
    arr = _to_mx(a)
    return mx.reshape(arr, newshape, stream=stream)


def ravel(a: Any, *, stream: Any | None = None) -> mx.array:
    arr = _to_mx(a)
    return mx.reshape(arr, (arr.size,), stream=stream)


def squeeze(a: Any, axis: Any | None = None) -> mx.array:
    arr = _to_mx(a)
    return mx.squeeze(arr, axis=axis)


def expand_dims(a: Any, axis: int) -> mx.array:
    return mx.expand_dims(_to_mx(a), axis)


def transpose(a: Any, axes: Any | None = None) -> mx.array:
    return mx.transpose(_to_mx(a), axes=axes)


def swapaxes(a: Any, axis1: int, axis2: int) -> mx.array:
    return mx.swapaxes(_to_mx(a), axis1, axis2)


def moveaxis(a: Any, source: Any, destination: Any) -> mx.array:
    return mx.moveaxis(_to_mx(a), source, destination)


def stack(arrays: Sequence[Any], axis: int = 0) -> mx.array:
    return mx.stack([_to_mx(a) for a in arrays], axis=axis)


def concatenate(arrays: Sequence[Any], axis: int = 0) -> mx.array:
    converted = [_to_mx(a) for a in arrays]
    try:
        return mx.concatenate(converted, axis=axis)
    except (TypeError, ValueError):
        lists = [a.tolist() if hasattr(a, "tolist") else a for a in converted]
        if axis in (0, None):
            data = []
            for item in lists:
                if isinstance(item, list):
                    data.extend(item)
                else:
                    data.append(item)
            return _to_mx(data)
        if axis == 1:
            rows = []
            for row_parts in zip(*lists):
                row = []
                for part in row_parts:
                    row.extend(part if isinstance(part, list) else [part])
                rows.append(row)
            return _to_mx(rows)
        raise


def column_stack(tup: Sequence[Any]) -> mx.array:
    arrays: List[mx.array] = []
    for a in tup:
        arr = _to_mx(a)
        if arr.ndim == 1:
            arr = mx.reshape(arr, (arr.shape[0], 1))
        elif arr.ndim != 2:
            raise ValueError("column_stack expects 1D or 2D arrays")
        arrays.append(arr)
    return mx.concatenate(arrays, axis=1)


def row_stack(tup: Sequence[Any]) -> mx.array:
    arrays = [atleast_2d(a) for a in tup]
    return concatenate(arrays, axis=0)


def dstack(tup: Sequence[Any]) -> mx.array:
    arrays = [atleast_3d(a) for a in tup]
    return concatenate(arrays, axis=2)


def hstack(tup: Sequence[Any]) -> mx.array:
    arrays = [atleast_1d(a) for a in tup]
    return concatenate(arrays, axis=1 if arrays[0].ndim > 1 else 0)


def vstack(tup: Sequence[Any]) -> mx.array:
    arrays = [atleast_2d(a) for a in tup]
    return concatenate(arrays, axis=0)


def tile(a: Any, reps: Any) -> mx.array:
    return mx.tile(_to_mx(a), reps)


def repeat(a: Any, repeats: Any, axis: int | None = None) -> mx.array:
    return mx.repeat(_to_mx(a), repeats, axis=axis)


def append(arr: Any, values: Any, axis: int | None = None) -> mx.array:
    arr_mx = _to_mx(arr)
    val_mx = _to_mx(values)
    if axis is None:
        return concatenate([ravel(arr_mx), ravel(val_mx)], axis=0)
    return concatenate([arr_mx, val_mx], axis=axis)


def insert(arr: Any, obj: int, values: Any, axis: int | None = None) -> mx.array:
    arr_mx = _to_mx(arr)
    val_mx = _to_mx(values)
    if axis is None:
        arr_mx = ravel(arr_mx)
        val_mx = ravel(val_mx)
        axis = 0
    before = take(arr_mx, arange(0, obj), axis=axis)
    after = take(arr_mx, arange(obj, arr_mx.shape[axis]), axis=axis)
    return concatenate([before, val_mx, after], axis=axis)


def delete(arr: Any, obj: int, axis: int | None = None) -> mx.array:
    arr_mx = _to_mx(arr)
    if axis is None:
        arr_mx = ravel(arr_mx)
        axis = 0
    before = take(arr_mx, arange(0, obj), axis=axis)
    after = take(arr_mx, arange(obj + 1, arr_mx.shape[axis]), axis=axis)
    return concatenate([before, after], axis=axis)


def take(a: Any, indices: Any, axis: int | None = None,
         mode: str | None = None) -> mx.array:
    arr = _to_mx(a)
    idx = _to_mx(indices)
    if mode == "clip":
        dim = arr.shape[axis or 0]
        idx = mx.clip(idx, 0, _builtins.max(dim - 1, 0)).astype(mx.int64)
    return _wrap_factory_array(_take_shadow(arr, idx, axis=axis))


def put(a: Any, indices: Any, values: Any) -> mx.array:
    arr = _to_mx(a)
    idx = _to_mx(indices)
    vals = _to_mx(values)
    arr_list = arr.tolist()
    flat = list(_flatten(arr_list))
    idx_list = _to_mx(idx).tolist()
    if not isinstance(idx_list, list):
        idx_list = [idx_list]
    val_list = _to_mx(vals).tolist()
    if not isinstance(val_list, list):
        val_list = [val_list]
    for i, v in zip(idx_list, itertools.cycle(val_list)):
        flat[i] = v
    return mx.array(flat).reshape(arr.shape)


def where(condition: Any, x: Any, y: Any) -> mx.array:
    return mx.where(_to_mx(condition), _to_mx(x), _to_mx(y))


def divide(a: Any, b: Any, dtype: Any | None = None, **kwargs: Any) -> mx.array:
    result = mx.divide(_to_mx(a), _to_mx(b))
    return result.astype(dtype) if dtype is not None else result


true_divide = divide


def mod(a: Any, b: Any, *, stream: Any | None = None) -> mx.array:
    return mx.remainder(_to_mx(a), _to_mx(b), stream=stream)


remainder = mod


def copysign(x1: Any, x2: Any) -> mx.array:
    return _to_scalar(mx.sign(_to_mx(x2)) * mx.abs(_to_mx(x1)))


def clip(a: Any, a_min: Any, a_max: Any) -> mx.array:
    return mx.clip(_to_mx(a), a_min, a_max)


def diff(a: Any, n: int = 1, axis: int = -1) -> mx.array:
    arr = _to_mx(a)
    for _ in range(n):
        slice1 = [slice(None)] * arr.ndim
        slice2 = [slice(None)] * arr.ndim
        slice1[axis] = slice(1, None)
        slice2[axis] = slice(None, -1)
        arr = arr[tuple(slice1)] - arr[tuple(slice2)]
    return arr


def unique(a: Any) -> mx.array:
    values = sorted(set(_flatten(_to_mx(a).tolist())))
    if values and _contains_object_data(values):
        return values
    return mx.array(values)


def intersect1d(ar1: Any, ar2: Any, assume_unique: bool = False,
                return_indices: bool = False):
    left = list(_flatten(_to_mx(ar1).tolist()))
    right = list(_flatten(_to_mx(ar2).tolist()))
    values = sorted(set(left).intersection(right))
    out = values if _contains_object_data(values) else mx.array(values)
    if not return_indices:
        return out
    left_idx = mx.array([left.index(v) for v in values])
    right_idx = mx.array([right.index(v) for v in values])
    return out, left_idx, right_idx


def isin(element: Any, test_elements: Any, assume_unique: bool = False,
         invert: bool = False) -> mx.array:
    tests = set(_flatten(_to_mx(test_elements).tolist()))

    def contains(value: Any) -> Any:
        if isinstance(value, list):
            return [contains(v) for v in value]
        present = value in tests
        return not present if invert else present

    return mx.array(contains(_to_mx(element).tolist()), dtype=mx.bool_)


def in1d(ar1: Any, ar2: Any, assume_unique: bool = False,
         invert: bool = False) -> mx.array:
    return ravel(isin(ar1, ar2, assume_unique=assume_unique, invert=invert))


def sort(a: Any, axis: int | None = -1, *,
         stream: Any | None = None) -> mx.array:
    return mx.sort(_to_mx(a), axis=axis, stream=stream)


def argsort(a: Any, axis: int | None = -1, *,
            stream: Any | None = None) -> mx.array:
    return mx.argsort(_to_mx(a), axis=axis, stream=stream)


def argmax(a: Any, axis: int | None = None) -> Any:
    return _to_scalar(mx.argmax(_to_mx(a), axis=axis))


def argmin(a: Any, axis: int | None = None) -> Any:
    return _to_scalar(mx.argmin(_to_mx(a), axis=axis))


def sum(a: Any, axis: Any | None = None, keepdims: bool = False, *,
        stream: Any | None = None) -> Any:
    return _to_scalar(mx.sum(_to_mx(a), axis=axis, keepdims=keepdims,
                             stream=stream))


def mean(a: Any, axis: Any | None = None, keepdims: bool = False, *,
         stream: Any | None = None) -> Any:
    return _to_scalar(mx.mean(_to_mx(a), axis=axis, keepdims=keepdims,
                              stream=stream))


def std(a: Any, axis: int | None = None) -> Any:
    return _to_scalar(mx.std(_to_mx(a), axis=axis))


def var(a: Any, axis: int | None = None) -> Any:
    return _to_scalar(mx.var(_to_mx(a), axis=axis))


def min(a: Any, axis: int | None = None, initial: Any | None = None) -> Any:
    arr = _to_mx(a)
    if arr.size == 0 and initial is not None:
        return initial
    result = mx.min(arr, axis=axis)
    if initial is not None:
        result = mx.minimum(result, mx.array(initial, dtype=arr.dtype))
    return _to_scalar(result)


def max(a: Any, axis: int | None = None, initial: Any | None = None) -> Any:
    arr = _to_mx(a)
    if arr.size == 0 and initial is not None:
        return initial
    result = mx.max(arr, axis=axis)
    if initial is not None:
        result = mx.maximum(result, mx.array(initial, dtype=arr.dtype))
    return _to_scalar(result)


def prod(a: Any, axis: int | None = None) -> Any:
    return mx.prod(_to_mx(a), axis=axis)


def roots(p: Any) -> mx.array:
    coeffs = [float(v) for v in _flatten(_to_mx(p).tolist())]
    while coeffs and abs(coeffs[0]) == 0:
        coeffs.pop(0)
    degree = len(coeffs) - 1
    if degree <= 0:
        return mx.array([])
    if degree == 1:
        a, b = coeffs
        return mx.array([-b / a])
    if degree == 2:
        a, b, c = coeffs
        disc = b * b - 4 * a * c
        if disc < 0:
            return mx.array([])
        root = math.sqrt(disc)
        return mx.array([(-b + root) / (2 * a), (-b - root) / (2 * a)])
    raise NotImplementedError("roots currently supports linear/quadratic polynomials")


def cumsum(a: Any, axis: int | None = None, *, reverse: bool = False,
           inclusive: bool = True, stream: Any | None = None) -> mx.array:
    return mx.cumsum(_to_mx(a), axis=axis, reverse=reverse,
                     inclusive=inclusive, stream=stream)


def cumprod(a: Any, axis: int | None = None) -> mx.array:
    return mx.cumprod(_to_mx(a), axis=axis)


def ptp(a: Any, axis: int | None = None) -> Any:
    arr = _to_mx(a)
    return _to_scalar(mx.max(arr, axis=axis) - mx.min(arr, axis=axis))


def all(a: Any, axis: int | None = None) -> Any:
    return _to_scalar(mx.all(_to_mx(a), axis=axis))


def any(a: Any, axis: int | None = None) -> Any:
    return _to_scalar(mx.any(_to_mx(a), axis=axis))


def _python_isfinite(value: Any) -> Any:
    if isinstance(value, list):
        return [_python_isfinite(item) for item in value]
    if value is None:
        return False
    try:
        return math.isfinite(value)
    except (TypeError, ValueError):
        return True


def isfinite(a: Any) -> mx.array:
    arr = _to_mx(a)
    return mx.isfinite(arr)


def isinf(a: Any) -> mx.array:
    return mx.isinf(_to_mx(a))


def isnan(a: Any) -> mx.array:
    return mx.isnan(_to_mx(a))


def isclose(a: Any, b: Any, rtol: float = 1e-5, atol: float = 1e-8) -> mx.array:
    return mx.isclose(_to_mx(a), _to_mx(b), rtol=rtol, atol=atol)


def allclose(a: Any, b: Any, rtol: float = 1e-5, atol: float = 1e-8) -> bool:
    return bool(mx.all(isclose(a, b, rtol=rtol, atol=atol)).item())


def array_equal(a: Any, b: Any) -> bool:
    return bool(mx.all(mx.equal(_to_mx(a), _to_mx(b))).item())


def equal(a: Any, b: Any) -> mx.array:
    return mx.equal(_to_mx(a), _to_mx(b))


def not_equal(a: Any, b: Any) -> mx.array:
    return mx.not_equal(_to_mx(a), _to_mx(b))


def less(a: Any, b: Any) -> mx.array:
    return mx.less(_to_mx(a), _to_mx(b))


def less_equal(a: Any, b: Any) -> mx.array:
    return mx.less_equal(_to_mx(a), _to_mx(b))


def greater(a: Any, b: Any) -> mx.array:
    return mx.greater(_to_mx(a), _to_mx(b))


def greater_equal(a: Any, b: Any) -> mx.array:
    return mx.greater_equal(_to_mx(a), _to_mx(b))


def logical_and(a: Any, b: Any) -> mx.array:
    return mx.logical_and(_to_mx(a), _to_mx(b))


def logical_or(a: Any, b: Any) -> mx.array:
    return mx.logical_or(_to_mx(a), _to_mx(b))


def logical_not(a: Any) -> mx.array:
    return mx.logical_not(_to_mx(a))


def logical_xor(a: Any, b: Any) -> mx.array:
    return mx.not_equal(_to_mx(a), _to_mx(b))


def _logical_reduce(func, arrays: Any, axis: int | None = 0):
    seq = list(arrays)
    if not seq:
        return True
    result = _to_mx(seq[0])
    for item in seq[1:]:
        result = func(result, item)
    return result


logical_and.reduce = lambda arrays, axis=0, dtype=None, out=None: _logical_reduce(
    logical_and, arrays, axis=axis)
logical_or.reduce = lambda arrays, axis=0, dtype=None, out=None: _logical_reduce(
    logical_or, arrays, axis=axis)


def sign(a: Any) -> mx.array:
    return mx.sign(_to_mx(a))


def abs(a: Any) -> mx.array:
    return mx.abs(_to_mx(a))


def sqrt(a: Any, *, stream: Any | None = None) -> mx.array:
    return mx.sqrt(_to_mx(a), stream=stream)


def exp(a: Any) -> mx.array:
    return mx.exp(_to_mx(a))


def log(a: Any) -> mx.array:
    return mx.log(_to_mx(a))


def log2(a: Any) -> mx.array:
    return mx.log2(_to_mx(a))


def log10(a: Any) -> mx.array:
    return mx.log10(_to_mx(a))


def power(a: Any, b: Any) -> mx.array:
    return mx.power(_to_mx(a), _to_mx(b))


def square(a: Any) -> mx.array:
    return mx.square(_to_mx(a))


def floor(a: Any) -> mx.array:
    return mx.floor(_to_mx(a))


def ceil(a: Any) -> mx.array:
    return mx.ceil(_to_mx(a))


def round(a: Any, decimals: int = 0) -> mx.array:
    return mx.round(_to_mx(a), decimals=decimals)


def tan(a: Any) -> mx.array:
    return mx.tan(_to_mx(a))


def sin(a: Any) -> mx.array:
    return mx.sin(_to_mx(a))


def cos(a: Any) -> mx.array:
    return mx.cos(_to_mx(a))


def arcsin(a: Any) -> mx.array:
    return mx.arcsin(_to_mx(a))


def arccos(a: Any) -> mx.array:
    return mx.arccos(_to_mx(a))


def arctan(a: Any) -> mx.array:
    return mx.arctan(_to_mx(a))


def arctan2(y: Any, x: Any) -> mx.array:
    return mx.arctan2(_to_mx(y), _to_mx(x))


def degrees(a: Any, *, stream: Any | None = None) -> mx.array:
    return mx.degrees(_to_mx(a), stream=stream)


def radians(a: Any, *, stream: Any | None = None) -> mx.array:
    return mx.radians(_to_mx(a), stream=stream)


def deg2rad(a: Any) -> mx.array:
    return radians(a)


def rad2deg(a: Any) -> mx.array:
    return degrees(a)


def hypot(x: Any, y: Any) -> mx.array:
    x_mx = _to_mx(x)
    y_mx = _to_mx(y)
    return mx.sqrt(x_mx * x_mx + y_mx * y_mx)


def matmul(a: Any, b: Any, *, stream: Any | None = None) -> mx.array:
    return mx.matmul(_to_mx(a), _to_mx(b), stream=stream)


def dot(a: Any, b: Any) -> mx.array:
    return mx.matmul(_to_mx(a), _to_mx(b))


def outer(a: Any, b: Any) -> mx.array:
    a = _to_mx(a).reshape((-1, 1))
    b = _to_mx(b).reshape((1, -1))
    return mx.matmul(a, b)


class _Power:
    def __call__(self, a: Any, b: Any, **kwargs: Any) -> mx.array:
        return mx.power(_to_mx(a), _to_mx(b))

    def outer(self, a: Any, b: Any) -> mx.array:
        a = _to_mx(a)
        b = _to_mx(b)
        if a.ndim == 0 or b.ndim == 0:
            return mx.power(a, b)
        return mx.power(a.reshape((-1, 1)), b.reshape((1, -1)))


power = _Power()


def tensordot(a: Any, b: Any, axes: int | Tuple[Any, Any] = 2) -> mx.array:
    return mx.tensordot(_to_mx(a), _to_mx(b), axes=axes)


def cross(a: Any, b: Any) -> mx.array:
    return mx.linalg.cross(_to_mx(a), _to_mx(b))


def diag(v: Any, k: int = 0) -> mx.array:
    arr = _to_mx(v)
    if arr.ndim == 1:
        n = arr.shape[0] + abs(k)
        out = mx.zeros((n, n), dtype=arr.dtype)
        idx = mx.arange(arr.shape[0])
        if k >= 0:
            out[idx, idx + k] = arr
        else:
            out[idx - k, idx] = arr
        return out
    if arr.ndim == 2:
        idx = mx.arange(_py_min(arr.shape))
        if k >= 0:
            return arr[idx, idx + k]
        return arr[idx - k, idx]
    raise ValueError("diag expects 1D or 2D array")


def eye(n: int, m: int | None = None, k: int = 0, dtype: Any | None = None) -> mx.array:
    return mx.eye(n, m, k=k, dtype=_unwrap_dtype(dtype) or float32.mx_dtype)


def identity(n: int, dtype: Any | None = None) -> mx.array:
    return mx.identity(n, dtype=_unwrap_dtype(dtype) or float32.mx_dtype)


def meshgrid(*arrays: Any, **kwargs: Any) -> List[mx.array]:
    arrays = [_to_mx(a) for a in arrays]
    return mx.meshgrid(*arrays, **kwargs)


def broadcast_to(a: Any, shape: Any) -> mx.array:
    return mx.broadcast_to(_to_mx(a), shape)


def broadcast_arrays(*args: Any, **kwargs: Any) -> Tuple[mx.array, ...]:
    return mx.broadcast_arrays(*[_to_mx(a) for a in args])


def shape(a: Any) -> Tuple[int, ...]:
    if isinstance(a, (list, tuple)):
        return _infer_shape(_copy_nested(a))
    arr = _to_mx(a)
    return tuple(arr.shape)


def size(a: Any) -> int:
    return int(_to_mx(a).size)


def ndim(a: Any) -> int:
    return int(_to_mx(a).ndim)


def copy(a: Any) -> mx.array:
    return _to_mx(a)


def copyto(dst: Any, src: Any) -> mx.array:
    return _to_mx(src)


def isscalar(obj: Any) -> bool:
    return not isinstance(obj, (list, tuple, dict, mx.array))


def iterable(obj: Any) -> bool:
    if isinstance(obj, mx.array):
        return obj.ndim > 0
    try:
        iter(obj)
        return True
    except (TypeError, IndexError):
        return False


def vectorize(pyfunc, otypes: Any | None = None, **_kwargs: Any):
    def wrapped(*args, **kwargs):
        vectors = [list(_flatten(_to_mx(a).tolist())) if isinstance(a, mx.array) else list(_flatten(a)) for a in args]
        result = [pyfunc(*vals, **kwargs) for vals in zip(*vectors)]
        if otypes == "O" or otypes == ["O"] or otypes == ("O",):
            return result
        try:
            return mx.array(result)
        except (TypeError, ValueError):
            return result
    return wrapped


def fromiter(iterable_obj: Iterable[Any], dtype: Any | None = None, count: int | None = None) -> mx.array:
    items = list(iterable_obj) if count is None else list(itertools.islice(iterable_obj, count))
    return mx.array(items, dtype=_unwrap_dtype(dtype))


def frombuffer(buffer_obj: bytes, dtype: Any | None = None) -> mx.array:
    # Interpret buffer as uint8 unless dtype provided
    data = list(buffer_obj)
    return mx.array(data, dtype=_unwrap_dtype(dtype) or uint8.mx_dtype)


def fromfile(*args: Any, **kwargs: Any) -> mx.array:
    raise NotImplementedError("fromfile is not supported without MLXArrayBackend")


def genfromtxt(*args: Any, **kwargs: Any) -> mx.array:
    raise NotImplementedError("genfromtxt is not supported without MLXArrayBackend")


def loadtxt(*args: Any, **kwargs: Any) -> mx.array:
    raise NotImplementedError("loadtxt is not supported without MLXArrayBackend")


def histogram(a: Any, bins: int = 10, range: Tuple[float, float] | None = None):
    arr = _to_mx(a).tolist()
    if range is None:
        min_v = _py_min(arr)
        max_v = _py_max(arr)
    else:
        min_v, max_v = range
    bin_edges = [min_v + (max_v - min_v) * i / bins for i in range(bins + 1)]
    counts = [0] * bins
    for v in arr:
        if v < min_v or v > max_v:
            continue
        idx = _py_min(int((v - min_v) / (max_v - min_v) * bins), bins - 1)
        counts[idx] += 1
    return mx.array(counts), mx.array(bin_edges)


def histogram2d(x: Any, y: Any, bins: int = 10, range: Tuple[Tuple[float, float], Tuple[float, float]] | None = None):
    x_list = _to_mx(x).tolist()
    y_list = _to_mx(y).tolist()
    if range is None:
        x_min, x_max = _py_min(x_list), _py_max(x_list)
        y_min, y_max = _py_min(y_list), _py_max(y_list)
    else:
        (x_min, x_max), (y_min, y_max) = range
    x_edges = [x_min + (x_max - x_min) * i / bins for i in range(bins + 1)]
    y_edges = [y_min + (y_max - y_min) * i / bins for i in range(bins + 1)]
    counts = [[0 for _ in range(bins)] for _ in range(bins)]
    for xv, yv in zip(x_list, y_list):
        if xv < x_min or xv > x_max or yv < y_min or yv > y_max:
            continue
        xi = _py_min(int((xv - x_min) / (x_max - x_min) * bins), bins - 1)
        yi = _py_min(int((yv - y_min) / (y_max - y_min) * bins), bins - 1)
        counts[xi][yi] += 1
    return mx.array(counts), mx.array(x_edges), mx.array(y_edges)


def bincount(x: Any, weights: Any | None = None,
             minlength: int | None = None):
    actual_stream = _active_stream()
    x_arr = mx.reshape(_to_mx(x).astype(mx.int32), (-1,),
                       stream=actual_stream)
    if x_arr.size == 0:
        size = 0 if minlength is None else minlength
    elif minlength is None:
        size = int(mx.max(x_arr, stream=actual_stream).item()) + 1
    else:
        size = minlength
    if size == 0:
        dtype = mx.int64 if weights is None else _to_mx(weights).dtype
        return mx.zeros((0,), dtype=dtype, stream=actual_stream)
    if weights is None:
        return _mx_overrides.bincount_int32(
            x_arr, minlength=size, stream=actual_stream)

    bins = mx.arange(size, dtype=mx.int32, stream=actual_stream)
    matches = (mx.expand_dims(bins, 1, stream=actual_stream)
               == mx.expand_dims(x_arr, 0, stream=actual_stream))
    weight_arr = mx.reshape(_to_mx(weights), (-1,), stream=actual_stream)
    return mx.sum(matches.astype(weight_arr.dtype)
                  * mx.expand_dims(weight_arr, 0, stream=actual_stream),
                  axis=1, stream=actual_stream)


def convolve(a: Any, v: Any, mode: str = "full") -> mx.array:
    a_list = list(_flatten(_to_mx(a).tolist()))
    v_list = list(_flatten(_to_mx(v).tolist()))
    out = []
    n = len(a_list)
    m = len(v_list)
    for i in range(n + m - 1):
        s = 0
        for j in range(m):
            if 0 <= i - j < n:
                s += a_list[i - j] * v_list[j]
        out.append(s)
    if mode == "valid":
        target = _builtins.max(n, m) - _builtins.min(n, m) + 1
        start = _builtins.min(n, m) - 1
        end = start + target
        out = out[start:end]
    elif mode == "same":
        target = _builtins.max(n, m)
        start = (len(out) - target) // 2
        end = start + target
        out = out[start:end]
    return mx.array(out)


def correlate(a: Any, v: Any, mode: str = "valid") -> mx.array:
    return convolve(a, list(reversed(list(_flatten(_to_mx(v).tolist())))), mode=mode)


def interp(x: Any, xp: Any, fp: Any):
    x_arr = _to_mx(x)
    xp_arr = mx.reshape(_to_mx(xp), (-1,))
    fp_arr = mx.reshape(_to_mx(fp), (-1,))
    if xp_arr.size != fp_arr.size:
        raise ValueError("xp and fp must have the same length")
    if xp_arr.size == 0:
        raise ValueError("xp must be non-empty")
    if xp_arr.size == 1:
        return mx.reshape(mx.full(x_arr.size, fp_arr[0], dtype=fp_arr.dtype),
                          x_arr.shape)

    x_flat = mx.reshape(x_arr, (-1,))
    coord_dtype = mx.float64 if (
        getattr(x_flat, "dtype", None) == mx.float64
        or getattr(xp_arr, "dtype", None) == mx.float64
        or getattr(fp_arr, "dtype", None) == mx.float64
    ) else mx.float32
    x_calc = x_flat.astype(coord_dtype)
    xp_calc = xp_arr.astype(coord_dtype)
    actual_stream = _active_stream()
    x_expanded = mx.expand_dims(x_calc, 1, stream=actual_stream)
    xp_expanded = mx.expand_dims(xp_calc, 0, stream=actual_stream)
    hi = mx.sum(x_expanded > xp_expanded, axis=1, stream=actual_stream)
    hi = mx.clip(hi, 1, xp_arr.size - 1, stream=actual_stream).astype(mx.int32)
    lo = hi - 1

    x0 = mx.take(xp_calc, lo, stream=actual_stream)
    x1 = mx.take(xp_calc, hi, stream=actual_stream)
    y0 = mx.take(fp_arr, lo, stream=actual_stream)
    y1 = mx.take(fp_arr, hi, stream=actual_stream)
    t = (x_calc - x0) / (x1 - x0)
    out = y0 + t * (y1 - y0)
    out = mx.where(x_calc <= xp_calc[0], fp_arr[0], out, stream=actual_stream)
    out = mx.where(x_calc >= xp_calc[-1], fp_arr[-1], out, stream=actual_stream)
    return mx.reshape(out, x_arr.shape, stream=actual_stream)


def searchsorted(a: Any, v: Any, side: str = "left", sorter: Any | None = None):
    if side not in {"left", "right"}:
        raise ValueError("side must be 'left' or 'right'")
    actual_stream = _active_stream()
    arr = mx.reshape(_to_mx(a), (-1,), stream=actual_stream)
    if sorter is not None:
        arr = mx.take(arr, _to_mx(sorter).astype(mx.int32),
                      stream=actual_stream)
    values = _to_mx(v)
    values_shape = tuple(values.shape)
    values_flat = mx.reshape(values, (-1,), stream=actual_stream)
    if arr.size == 0:
        return mx.zeros(values_shape, dtype=mx.int32, stream=actual_stream)
    edges = mx.expand_dims(arr, 1, stream=actual_stream)
    probes = mx.expand_dims(values_flat, 0, stream=actual_stream)
    if side == "left":
        before = edges < probes
    else:
        before = edges <= probes
    indices = mx.sum(before.astype(mx.int32), axis=0, stream=actual_stream)
    return mx.reshape(indices, values_shape, stream=actual_stream)


mx.searchsorted = searchsorted


def nan_to_num(x: Any, nan: float = 0.0, posinf: float | None = None, neginf: float | None = None):
    x_mx = _to_mx(x)
    posinf = inf if posinf is None else posinf
    neginf = -inf if neginf is None else neginf
    return mx.nan_to_num(x_mx, nan=nan, posinf=posinf, neginf=neginf)


def nanmin(x: Any):
    x_mx = _to_mx(x)
    return _to_scalar(mx.min(mx.where(mx.isnan(x_mx), inf, x_mx)))


def nanmax(x: Any):
    x_mx = _to_mx(x)
    return _to_scalar(mx.max(mx.where(mx.isnan(x_mx), -inf, x_mx)))


def nanmean(x: Any):
    x_mx = _to_mx(x)
    mask = mx.isnan(x_mx)
    values = mx.where(mask, 0, x_mx)
    count = mx.sum(mx.where(mask, 0, 1))
    return _to_scalar(mx.sum(values) / count)


def nanstd(x: Any):
    x_mx = _to_mx(x)
    return _to_scalar(mx.std(mx.where(mx.isnan(x_mx), 0, x_mx)))


def nanvar(x: Any):
    x_mx = _to_mx(x)
    return _to_scalar(mx.var(mx.where(mx.isnan(x_mx), 0, x_mx)))


def percentile(a: Any, q: Any):
    arr = sorted(_flatten(_to_mx(a).tolist()))
    if not isinstance(q, list):
        q_list = [q]
    else:
        q_list = q
    out = []
    for qv in q_list:
        idx = int(round((qv / 100.0) * (len(arr) - 1)))
        out.append(arr[idx])
    return mx.array(out)


def median(a: Any):
    return percentile(a, 50)[0]


def average(a: Any, weights: Any | None = None):
    arr = _to_mx(a)
    if weights is None:
        return mean(arr)
    w = _to_mx(weights)
    return _to_scalar(mx.sum(arr * w) / mx.sum(w))


def isreal(x: Any) -> mx.array:
    return mx.isfinite(_to_mx(x))


def iscomplexobj(x: Any) -> bool:
    arr = _to_mx(x)
    return arr.dtype in (mx.complex64, mx.complex128) if hasattr(mx, "complex64") else False


def real(x: Any) -> mx.array:
    return _to_mx(x).real


def imag(x: Any) -> mx.array:
    return _to_mx(x).imag


def conj(x: Any) -> mx.array:
    return mx.conj(_to_mx(x))


def conjugate(x: Any) -> mx.array:
    return conj(x)


def angle(x: Any) -> mx.array:
    z = _to_mx(x)
    return mx.arctan2(z.imag, z.real)




def roll(a: Any, shift: int, axis: int | None = None) -> mx.array:
    return mx.roll(_to_mx(a), shift, axis=axis)


def flip(a: Any, axis: int | None = None) -> mx.array:
    arr = _to_mx(a)
    if axis is None:
        axis = 0
    idx = [slice(None)] * arr.ndim
    idx[axis] = slice(None, None, -1)
    return arr[tuple(idx)]


def flipud(a: Any) -> mx.array:
    return flip(a, axis=0)


def fliplr(a: Any) -> mx.array:
    return flip(a, axis=1)


def mgrid_getitem(key):
    slices = key if isinstance(key, tuple) else (key,)
    grids = []
    for s in slices:
        arr = _slice_to_array(s)
        grids.append(arr)
    meshes = mx.meshgrid(*grids, indexing="ij")
    return mx.stack(meshes)


def ogrid_getitem(key):
    slices = key if isinstance(key, tuple) else (key,)
    arrays = []
    for i, s in enumerate(slices):
        arr = _slice_to_array(s)
        shape = [1] * len(slices)
        shape[i] = arr.shape[0]
        arrays.append(mx.reshape(arr, shape))
    return tuple(arrays)


class _MGrid:
    def __getitem__(self, key):
        return mgrid_getitem(key)


class _OGrid:
    def __getitem__(self, key):
        return ogrid_getitem(key)


mgrid = _MGrid()
ogrid = _OGrid()


class _IndexExp:
    def __getitem__(self, key):
        return key


index_exp = _IndexExp()


class _SliceObj:
    def __getitem__(self, key):
        return key


s_ = _SliceObj()


def _r_concat(seq):
    parts = []
    for item in seq:
        if isinstance(item, slice):
            parts.append(_slice_to_array(item))
        else:
            parts.append(_to_mx(item))
    return concatenate(parts, axis=0)


def _c_concat(seq):
    parts = []
    for item in seq:
        if isinstance(item, slice):
            parts.append(_slice_to_array(item))
        else:
            parts.append(_to_mx(item))
    return concatenate(parts, axis=1)


class _RClass:
    def __getitem__(self, key):
        key = key if isinstance(key, tuple) else (key,)
        return _r_concat(key)

    def __call__(self, seq):
        return _r_concat(seq)


class _CClass:
    def __getitem__(self, key):
        key = key if isinstance(key, tuple) else (key,)
        return _c_concat(key)

    def __call__(self, seq):
        return _c_concat(seq)


r_ = _RClass()
c_ = _CClass()


def ix_(*args: Any):
    arrays = [ravel(_to_mx(a)) for a in args]
    shape = [len(a) for a in arrays]
    grids = []
    for i, a in enumerate(arrays):
        reshape_shape = [1] * len(arrays)
        reshape_shape[i] = shape[i]
        grids.append(mx.reshape(a, reshape_shape))
    return tuple(grids)


def indices(dimensions: Sequence[int]):
    ranges = [arange(dim) for dim in dimensions]
    meshes = mx.meshgrid(*ranges, indexing="ij")
    return mx.stack(meshes)


def nonzero(a: Any):
    arr = _to_mx(a)
    coords = [idx for idx, v in _iter_indices(arr.tolist()) if v]
    if not coords:
        return tuple(mx.array([], dtype=mx.int32) for _ in range(arr.ndim))
    axes = list(zip(*coords))
    return tuple(mx.array(axis) for axis in axes)


def count_nonzero(a: Any, axis: int | None = None) -> Any:
    return sum(_to_mx(a) != 0, axis=axis)


mx.nonzero = nonzero
mx.count_nonzero = count_nonzero


def argwhere(a: Any):
    arr = _to_mx(a)
    coords = []
    for idx, v in _iter_indices(arr.tolist()):
        if v:
            coords.append(idx)
    return mx.array(coords)


def unravel_index(indices: Any, shape: Sequence[int]):
    if isinstance(indices, (list, tuple, mx.array)) and not isscalar(indices):
        values = indices.tolist() if isinstance(indices, mx.array) else list(indices)
        coords = [unravel_index(idx, shape) for idx in values]
        return tuple(mx.array(axis) for axis in zip(*coords))
    idx = int(_to_mx(indices).item())
    coords = []
    for dim in reversed(shape):
        coords.append(idx % dim)
        idx //= dim
    return tuple(reversed(coords))


def ravel_multi_index(multi_index: Sequence[Any], dims: Sequence[int]):
    def ravel_one(indices: Sequence[Any]) -> int:
        idx = 0
        for i, dim in zip(indices, dims):
            idx = idx * dim + int(i)
        return idx

    if _builtins.any(isinstance(i, (list, tuple, mx.array)) and not isscalar(i)
                     for i in multi_index):
        columns = []
        for i in multi_index:
            if isinstance(i, mx.array):
                columns.append(i.tolist())
            elif isinstance(i, (list, tuple)):
                columns.append(list(i))
            else:
                columns.append([i])
        return [ravel_one(values) for values in zip(*columns)]

    idx = 0
    for i, dim in zip(multi_index, dims):
        idx = idx * dim + int(i)
    return idx


def ndindex(*shape: int):
    return itertools.product(*[range(s) for s in shape])


def ndenumerate(a: Any):
    for idx, v in _iter_indices(_to_mx(a).tolist()):
        yield idx, v


def broadcast_shapes(*shapes: Any) -> Tuple[int, ...]:
    return mx.broadcast_shapes(*shapes)


def array_split(ary: Any, indices_or_sections: int, axis: int = 0):
    arr = _to_mx(ary)
    size = arr.shape[axis]
    if isinstance(indices_or_sections, int):
        step = math.ceil(size / indices_or_sections)
        indices = list(range(step, size, step))
    else:
        indices = list(indices_or_sections)
    parts = []
    start = 0
    for end in indices + [size]:
        slc = [slice(None)] * arr.ndim
        slc[axis] = slice(start, end)
        parts.append(arr[tuple(slc)])
        start = end
    return parts


def split(ary: Any, indices_or_sections: int, axis: int = 0):
    return array_split(ary, indices_or_sections, axis=axis)


def linspace_indices(start: float, stop: float, num: int):
    return linspace(start, stop, num)


def spacing(x: Any):
    x_val = float(_to_scalar(_to_mx(x)))
    return math.ulp(x_val)


def nextafter(x1: Any, x2: Any):
    x1_val = float(_to_scalar(_to_mx(x1)))
    x2_val = float(_to_scalar(_to_mx(x2)))
    return math.nextafter(x1_val, x2_val)


def promote_types(t1: Any, t2: Any):
    return t1 if t1 == t2 else float64


def can_cast(from_: Any, to: Any, casting: str | None = None):
    return True


def _dtype_kind(value: Any) -> str | None:
    if isinstance(value, DType):
        return value.kind
    if value in _DTYPE_BY_MX:
        return _DTYPE_BY_MX[value].kind
    if value is _builtins.bool:
        return "b"
    if value is _builtins.int:
        return "i"
    if value is _builtins.float:
        return "f"
    if value is datetime64:
        return "M"
    if value is timedelta64:
        return "m"
    if isinstance(value, str):
        if value.startswith("datetime64"):
            return "M"
        if value.startswith("timedelta64"):
            return "m"
        dt = _DTYPE_BY_NAME.get(value)
        return dt.kind if dt is not None else None
    return None


def issubdtype(arg1: Any, arg2: Any):
    kind1 = _dtype_kind(arg1)
    kind2 = _dtype_kind(arg2)
    if arg2 is integer or arg2 is _MLX_INTEGER_CATEGORY:
        return kind1 in {"i", "u"}
    if arg2 is floating or arg2 is _MLX_FLOATING_CATEGORY:
        return kind1 == "f"
    if kind2 is None:
        return arg1 == arg2
    return kind1 == kind2


def min_scalar_type(arg: Any):
    return type(arg)


def require(a: Any, **kwargs: Any):
    return _to_mx(a)


def may_share_memory(a: Any, b: Any, max_work: Any | None = None) -> bool:
    return a is b


def broadcast_arrays(*args: Any, **kwargs: Any):
    return mx.broadcast_arrays(*[_to_mx(a) for a in args])


def corrcoef(m: Any, y: Any | None = None, rowvar: bool = True):
    arr = _to_mx(m)
    if y is not None:
        arr = concatenate([arr, _to_mx(y)], axis=0)
    mean_arr = mean(arr, axis=1 if rowvar else 0)
    arr_centered = arr - mean_arr
    cov = mx.matmul(arr_centered, transpose(arr_centered)) / (arr.shape[1] - 1)
    return cov


def cov(m: Any, y: Any | None = None, rowvar: bool = True, bias: bool = False, ddof: int | None = None):
    arr = _to_mx(m)
    if y is not None:
        arr = concatenate([arr, _to_mx(y)], axis=0)
    mean_arr = mean(arr, axis=1 if rowvar else 0)
    arr_centered = arr - mean_arr
    denom = arr.shape[1] - (0 if bias else 1)
    return mx.matmul(arr_centered, transpose(arr_centered)) / denom


def apply_along_axis(func1d, axis: int, arr: Any, *args, **kwargs):
    arr_mx = _to_mx(arr)
    if arr_mx.size == 0:
        return arr_mx
    if axis < 0:
        axis += arr_mx.ndim
    if arr_mx.ndim == 1:
        return _to_mx(func1d(arr_mx, *args, **kwargs))
    data = arr_mx.tolist()
    if arr_mx.ndim == 2 and axis == 1:
        return _to_mx([
            _to_mx(func1d(_to_mx(row), *args, **kwargs)).tolist()
            for row in data
        ])
    if arr_mx.ndim == 2 and axis == 0:
        columns = list(zip(*data))
        processed = [
            _to_mx(func1d(_to_mx(list(column)), *args, **kwargs)).tolist()
            for column in columns
        ]
        return _to_mx([list(row) for row in zip(*processed)])
    results = []
    for idx in range(arr_mx.shape[axis]):
        slc = [slice(None)] * arr_mx.ndim
        slc[axis] = idx
        results.append(func1d(arr_mx[tuple(slc)], *args, **kwargs))
    return stack(results, axis=axis)


def gradient(f: Any, *varargs: Any, **kwargs: Any):
    f_mx = _to_mx(f)
    return diff(f_mx)


def pad(array: Any, pad_width: Any, mode: str = "constant", constant_values: Any = 0):
    arr = _to_mx(array)
    if isinstance(pad_width, int):
        pad_width = [(pad_width, pad_width)] * arr.ndim
    pad_width = list(pad_width)
    new_shape = [arr.shape[i] + pad_width[i][0] + pad_width[i][1] for i in range(arr.ndim)]
    out = mx.full(new_shape, constant_values, dtype=arr.dtype)
    slices = tuple(slice(pad_width[i][0], pad_width[i][0] + arr.shape[i]) for i in range(arr.ndim))
    out[slices] = arr
    return out


def triu(m: Any, k: int = 0):
    arr = _to_mx(m)
    mask = mx.triu(mx.ones(arr.shape), k)
    return arr * mask


def tril(m: Any, k: int = 0):
    arr = _to_mx(m)
    mask = mx.tril(mx.ones(arr.shape), k)
    return arr * mask


def sinc(x: Any):
    x_mx = _to_mx(x)
    return mx.where(x_mx == 0, 1, mx.sin(pi * x_mx) / (pi * x_mx))


def hanning(M: int):
    n = arange(M)
    return 0.5 - 0.5 * cos(2 * pi * n / (M - 1))


def blackman(M: int):
    n = arange(M)
    return 0.42 - 0.5 * cos(2 * pi * n / (M - 1)) + 0.08 * cos(4 * pi * n / (M - 1))


def unwrap(p: Any, discont: float = pi):
    p_mx = _to_mx(p)
    if p_mx.size <= 1:
        return p_mx
    dp = diff(p_mx)
    dp = where(abs(dp) > discont, dp - 2 * pi * mx.sign(dp), dp)
    return p_mx[:1] + cumsum(dp)


def bytes_(s: Any):
    return bytes(s, "utf-8") if isinstance(s, str) else bytes(s)


def str_(s: Any):
    return str(s)


def int_(x: Any):
    return int(x)


def float_(x: Any):
    return float(x)


def object_(x: Any):
    return x


def minnan(x: Any):
    return nanmin(x)


def maxnan(x: Any):
    return nanmax(x)


@dataclass(frozen=True)
class _DateTime64String:
    value: str


def datetime64(value: Any, *args: Any, **kwargs: Any):
    if isinstance(value, datetime):
        return value
    if isinstance(value, str):
        if value == "NaT":
            return None
        try:
            return datetime.fromisoformat(value)
        except ValueError:
            try:
                return datetime.fromisoformat(value + "T00:00:00")
            except ValueError:
                return _DateTime64String(value)
    return value


def timedelta64(value: Any, *args: Any, **kwargs: Any):
    if isinstance(value, timedelta):
        return value
    if isinstance(value, (int, float)):
        unit = args[0] if args else kwargs.get("unit", "s")
        if unit == "ns":
            return timedelta(microseconds=value / 1000)
        if unit == "us":
            return timedelta(microseconds=value)
        if unit == "ms":
            return timedelta(milliseconds=value)
        if unit == "m":
            return timedelta(minutes=value)
        if unit == "h":
            return timedelta(hours=value)
        if unit == "D":
            return timedelta(days=value)
        return timedelta(seconds=value)
    return value


def _slice_to_array(s: slice) -> mx.array:
    start = 0 if s.start is None else s.start
    stop = s.stop
    step = 1 if s.step is None else s.step
    if isinstance(step, complex):
        num = int(abs(step))
        return linspace(start, stop, num)
    return arange(start, stop, step)


def _flatten(items: Any) -> Iterator[Any]:
    if isinstance(items, (list, tuple)):
        for item in items:
            yield from _flatten(item)
    else:
        yield items


def _iter_indices(arr: Any, prefix: Tuple[int, ...] = ()):  # arr is nested list
    if not isinstance(arr, list):
        yield prefix, arr
    else:
        for i, v in enumerate(arr):
            yield from _iter_indices(v, prefix + (i,))


class _ErrState:
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        return False


def errstate(**kwargs):
    return _ErrState()


def seterr(**kwargs):
    return {}


def _random_shape(size: Any | None = None, args: Tuple[Any, ...] = ()) -> Tuple[int, ...]:
    if args:
        if len(args) == 1 and isinstance(args[0], (tuple, list, mx.array)):
            return _shape_tuple(args[0])
        return tuple(int(v) for v in args)
    return _shape_tuple(size)


class _Random:
    def seed(self, seed_val: int | None = None):
        mx.random.seed(seed_val or 0)

    def rand(self, *shape: int):
        return _wrap_factory_array(mx.random.uniform(shape=_random_shape(args=shape)))

    def randn(self, *shape: int):
        return _wrap_factory_array(mx.random.normal(shape=_random_shape(args=shape)))

    def randint(self, low: int, high: int | None = None, size: Any | None = None):
        if high is None:
            low, high = 0, low
        return _wrap_factory_array(
            mx.random.randint(low, high, shape=_random_shape(size)))

    def multivariate_normal(self, mean: Any, cov: Any, size: Any | None = None):
        # MLX currently only supports float32 for multivariate normals.
        if size is None:
            shape = ()
        elif isinstance(size, int):
            shape = (size,)
        else:
            shape = tuple(size)

        mean_arr = array(mean, dtype=float32)
        cov_arr = array(cov, dtype=float32)
        return mx.random.multivariate_normal(mean_arr, cov_arr, shape=shape, dtype=float32.mx_dtype)

    def random(self, size: Any | None = None):
        return _wrap_factory_array(mx.random.uniform(shape=_random_shape(size)))

    def random_sample(self, size: Any | None = None):
        return self.random(size)

    def sample(self, size: Any | None = None):
        return self.random(size)

    def uniform(self, low: float = 0.0, high: float = 1.0, size: Any | None = None):
        return _wrap_factory_array(
            mx.random.uniform(low=low, high=high, shape=_random_shape(size)))

    def normal(self, loc: float = 0.0, scale: float = 1.0, size: Any | None = None):
        return _wrap_factory_array(
            mx.random.normal(shape=_random_shape(size)) * scale + loc)

    def standard_normal(self, size: Any | None = None):
        return self.normal(size=size)

    def lognormal(self, mean: float = 0.0, sigma: float = 1.0, size: Any | None = None):
        return mx.exp(self.normal(mean, sigma, size=size))

    def permutation(self, x: Any):
        arr = _to_mx(x)
        idx = mx.random.permutation(arr.shape[0])
        return arr[idx]

    def default_rng(self, seed: int | None = None):
        if seed is not None:
            self.seed(seed)
        return _Generator(self)


class _Generator:
    def __init__(self, random_module: _Random):
        self._random = random_module

    def random(self, size: Any | None = None):
        return self._random.random(size)

    def uniform(self, low: float = 0.0, high: float = 1.0, size: Any | None = None):
        return self._random.uniform(low=low, high=high, size=size)

    def normal(self, loc: float = 0.0, scale: float = 1.0, size: Any | None = None):
        return self._random.normal(loc=loc, scale=scale, size=size)

    def lognormal(self, mean: float = 0.0, sigma: float = 1.0, size: Any | None = None):
        return self._random.lognormal(mean=mean, sigma=sigma, size=size)

    def integers(self, low: int, high: int | None = None, size: Any | None = None,
                 dtype: Any | None = None, endpoint: bool = False):
        if endpoint and high is not None:
            high += 1
        values = self._random.randint(low, high=high, size=size)
        return values.astype(dtype) if dtype is not None else values


random = _Random()
default_rng = random.default_rng


def _testing_plain(value: Any) -> Any:
    if isinstance(value, mx.array):
        return value.tolist()
    if isinstance(value, (list, tuple)):
        return [_testing_plain(item) for item in value]
    return value


def _testing_equal(a: Any, b: Any) -> bool:
    a = _testing_plain(a)
    b = _testing_plain(b)
    if isinstance(a, list) or isinstance(b, list):
        return (isinstance(a, list) and isinstance(b, list)
                and len(a) == len(b)
                and _builtins.all(_testing_equal(x, y) for x, y in zip(a, b)))
    try:
        if math.isnan(a) and math.isnan(b):
            return True
    except TypeError:
        pass
    return a == b


class _Testing:
    def assert_allclose(self, a: Any, b: Any, rtol: float = 1e-5, atol: float = 1e-8, err_msg: str | None = None):
        if not allclose(a, b, rtol=rtol, atol=atol):
            raise AssertionError(err_msg or "Arrays are not equal within tolerance")

    def assert_array_equal(self, a: Any, b: Any, err_msg: str | None = None):
        if not array_equal(a, b) and not _testing_equal(a, b):
            raise AssertionError(err_msg or "Arrays are not equal")

    def assert_array_almost_equal(self, a: Any, b: Any, decimal: int = 6):
        rtol = 10 ** (-decimal)
        self.assert_allclose(a, b, rtol=rtol, atol=rtol)

    def assert_array_less(self, a: Any, b: Any, err_msg: str | None = None):
        if not bool(mx.all(less(a, b)).item()):
            raise AssertionError(err_msg or "Arrays are not ordered")

    def assert_equal(self, a: Any, b: Any, err_msg: str | None = None):
        if isinstance(a, (list, tuple, mx.array)) or isinstance(b, (list, tuple, mx.array)):
            return self.assert_array_equal(a, b, err_msg=err_msg)
        if a != b:
            raise AssertionError(err_msg or f"{a!r} != {b!r}")

    def assert_almost_equal(self, a: Any, b: Any, decimal: int = 6):
        return self.assert_array_almost_equal(a, b, decimal=decimal)

    def assert_array_max_ulp(self, a: Any, b: Any, maxulp: int = 1, dtype: Any | None = None):
        return self.assert_allclose(a, b, rtol=1e-6, atol=1e-12)

    def assert_raises(self, exc_type: type[BaseException], func, *args: Any, **kwargs: Any):
        try:
            func(*args, **kwargs)
        except exc_type:
            return
        raise AssertionError(f"{exc_type.__name__} was not raised")

    def break_cycles(self):
        gc.collect()


class VisibleDeprecationWarning(Warning):
    pass


testing = _Testing()


class _MaskedConstant:
    def __repr__(self) -> str:  # pragma: no cover - representation only
        return "masked"


masked = _MaskedConstant()


@dataclass
class MaskedArray:
    data: mx.array
    mask: mx.array | None

    @property
    def shape(self):
        return self.data.shape

    @property
    def ndim(self):
        return self.data.ndim

    @property
    def size(self):
        return self.data.size

    @property
    def dtype(self):
        return self.data.dtype

    def filled(self, fill_value: Any = 0):
        if self.mask is None:
            return self.data
        return mx.where(self.mask, _to_mx(fill_value), self.data)

    def min(self, *args: Any, **kwargs: Any):
        return min(self.filled(), *args, **kwargs)

    def max(self, *args: Any, **kwargs: Any):
        return max(self.filled(), *args, **kwargs)

    def astype(self, dtype: Any, *args: Any, **kwargs: Any):
        return MaskedArray(self.data.astype(dtype), self.mask)

    def __len__(self):
        return len(self.data)

    def __iter__(self):
        data = self.data.tolist()
        if not isinstance(data, list):
            data = [data]
        return iter(data)

    def __getitem__(self, key: Any):
        data = self.data[key]
        if self.mask is None:
            mask = None
        elif getattr(self.mask, "ndim", 0) == 0:
            mask = self.mask
        else:
            mask = self.mask[key]
        return MaskedArray(data, mask)

    def __array__(self):  # pragma: no cover - compatibility shim
        return self.data


class _MA:
    masked = masked
    MaskedArray = MaskedArray

    def array(self, data: Any, mask: Any | None = None, dtype: Any | None = None):
        return MaskedArray(data=_to_mx(data, dtype=dtype), mask=_to_mx(mask) if mask is not None else None)

    def masked_array(self, data: Any, mask: Any | None = None, dtype: Any | None = None):
        return self.array(data, mask=mask, dtype=dtype)

    def isMA(self, data: Any) -> bool:
        return isinstance(data, MaskedArray)

    def isMaskedArray(self, data: Any) -> bool:
        return isinstance(data, MaskedArray)

    def is_masked(self, data: Any) -> bool:
        return isinstance(data, MaskedArray)

    def getdata(self, data: Any):
        return data.data if isinstance(data, MaskedArray) else data

    def getmask(self, data: Any):
        if isinstance(data, MaskedArray) and data.mask is not None:
            return data.mask
        return None

    def getmaskarray(self, data: Any):
        if isinstance(data, MaskedArray) and data.mask is not None:
            return data.mask
        arr = _to_mx(data)
        return mx.zeros(arr.shape, dtype=bool_.mx_dtype)

    def filled(self, data: Any, fill_value: Any = 0):
        if isinstance(data, MaskedArray):
            return data.filled(fill_value)
        return _to_mx(data)

    def masked_where(self, condition: Any, data: Any):
        return MaskedArray(data=_to_mx(data), mask=_to_mx(condition))

    def masked_invalid(self, data: Any, copy: bool | None = None):
        arr = _to_mx(data)
        mask = mx.logical_or(mx.isnan(arr), mx.isinf(arr))
        return MaskedArray(data=arr, mask=mask)

    def masked_equal(self, data: Any, value: Any):
        arr = _to_mx(data)
        return MaskedArray(data=arr, mask=mx.equal(arr, value))

    def masked_less(self, data: Any, value: Any):
        arr = _to_mx(data)
        return MaskedArray(data=arr, mask=mx.less(arr, value))

    def masked_greater(self, data: Any, value: Any):
        arr = _to_mx(data)
        return MaskedArray(data=arr, mask=mx.greater(arr, value))

    def masked_all(self, shape: Any, dtype: Any | None = None):
        arr = mx.zeros(shape, dtype=_unwrap_dtype(dtype) or float32.mx_dtype)
        mask = mx.ones(shape, dtype=bool_.mx_dtype)
        return MaskedArray(data=arr, mask=mask)

    def count(self, data: Any):
        if not isinstance(data, MaskedArray) or data.mask is None:
            return int(_to_mx(data).size)
        return int(mx.sum(mx.logical_not(data.mask)).item())

    def argsort(self, data: Any, axis: int | None = -1):
        arr = self.filled(data, fill_value=inf)
        return mx.argsort(_to_mx(arr), axis=axis)

    def sqrt(self, data: Any):
        arr = self.filled(data, fill_value=nan)
        return MaskedArray(data=mx.sqrt(_to_mx(arr)), mask=self.getmaskarray(data))

    def ptp(self, data: Any, axis: int | None = None):
        arr = self.filled(data, fill_value=nan)
        return _to_scalar(mx.max(arr, axis=axis) - mx.min(arr, axis=axis))

    def ravel(self, data: Any):
        if isinstance(data, MaskedArray):
            return MaskedArray(data=ravel(data.data), mask=ravel(data.mask) if data.mask is not None else None)
        return ravel(data)

    def concatenate(self, arrays: Sequence[Any], axis: int = 0):
        datas = [self.getdata(a) for a in arrays]
        masks = [self.getmaskarray(a) for a in arrays]
        return MaskedArray(data=concatenate(datas, axis=axis), mask=concatenate(masks, axis=axis))

    def stack(self, arrays: Sequence[Any], axis: int = 0):
        datas = [self.getdata(a) for a in arrays]
        masks = [self.getmaskarray(a) for a in arrays]
        return MaskedArray(data=stack(datas, axis=axis), mask=stack(masks, axis=axis))

    def column_stack(self, arrays: Sequence[Any]):
        datas = [self.getdata(a) for a in arrays]
        masks = [self.getmaskarray(a) for a in arrays]
        return MaskedArray(data=column_stack(datas), mask=column_stack(masks))

    def hstack(self, arrays: Sequence[Any]):
        datas = [self.getdata(a) for a in arrays]
        masks = [self.getmaskarray(a) for a in arrays]
        return MaskedArray(data=hstack(datas), mask=hstack(masks))


ma = _MA()
mx.ma = ma


def _install_mx_module_surface() -> None:
    mx._DateTime64String = _DateTime64String
    mx.datetime64 = datetime64
    mx.timedelta64 = timedelta64
    mx.shape = shape
    mx.ndim = ndim
    mx.s_ = s_
    mx.index_exp = index_exp
    mx.mgrid = mgrid
    mx.ogrid = ogrid
    mx.indices = indices
    mx.atleast_1d = atleast_1d
    mx.atleast_2d = atleast_2d
    mx.atleast_3d = atleast_3d
    mx.column_stack = column_stack
    mx.dstack = dstack
    mx.hstack = hstack
    mx.vstack = vstack
    mx.sinc = sinc
    mx.interp = interp
    mx.bincount = bincount
    mx.logspace = logspace
    mx.geomspace = geomspace
    mx.zeros_like = zeros_like
    mx.ones_like = ones_like
    mx.empty_like = empty_like
    mx.full_like = full_like
    mx.unique = unique
    mx.errstate = errstate
    mx.seterr = seterr
    mx.dtype = dtype
    mx.issubdtype = issubdtype
    mx.testing = testing
    if not hasattr(mx.random, "rand"):
        mx.random.rand = random.rand
    if not hasattr(mx.random, "randn"):
        mx.random.randn = random.randn
    if not hasattr(mx.random, "random"):
        mx.random.random = random.random
    if not hasattr(mx.random, "random_sample"):
        mx.random.random_sample = random.random_sample
    if not hasattr(mx.random, "sample"):
        mx.random.sample = random.sample
    if not hasattr(mx.random, "standard_normal"):
        mx.random.standard_normal = random.standard_normal


_install_mx_module_surface()


class _Linalg:
    def __getattr__(self, name: str):
        return getattr(mx.linalg, name)


linalg = _Linalg()


class _FFT:
    def __getattr__(self, name: str):
        return getattr(mx.fft, name)


fft = _FFT()


def __getattr__(name: str):
    if hasattr(mx, name):
        return getattr(mx, name)
    raise AttributeError(name)


__all__ = [name for name in globals().keys() if not name.startswith("_")]

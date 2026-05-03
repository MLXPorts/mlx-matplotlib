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
import weakref
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any, Iterable, Iterator, List, Sequence, Tuple

import mlx.core as mx

# Matplotlib's array-facing internals routinely request float64 arrays.
# MLX's GPU backend rejects float64 constructors, so keep this compatibility
# layer on CPU unless callers explicitly move arrays elsewhere.
mx.set_default_device(mx.cpu)
if not hasattr(mx.array, "copy"):
    mx.array.copy = lambda self: mx.array(self)
if not hasattr(mx.array, "ravel"):
    mx.array.ravel = lambda self: mx.reshape(self, (self.size,))
if not hasattr(mx.array, "searchsorted"):
    mx.array.searchsorted = lambda self, v, side="left", sorter=None: searchsorted(
        self, v, side=side)
if not hasattr(mx.array, "nonzero"):
    mx.array.nonzero = lambda self: nonzero(self)
if not hasattr(mx.array, "__index__"):
    mx.array.__index__ = lambda self: int(self.item())
if not hasattr(mx.array, "__int__"):
    mx.array.__int__ = lambda self: int(self.item())
if not hasattr(mx.array, "__float__"):
    mx.array.__float__ = lambda self: float(self.item())
if not hasattr(mx.array, "__round__"):
    mx.array.__round__ = lambda self, ndigits=None: round(
        float(self.item()), ndigits) if ndigits is not None else round(
            float(self.item()))
if not hasattr(mx.array, "_mlx_array_orig_rpow"):
    mx.array._mlx_array_orig_rpow = mx.array.__rpow__

    def _array_rpow(self, other):
        if isinstance(other, (int, float)) and float(other) == 10.0:
            def pow10(value):
                if isinstance(value, list):
                    return [pow10(item) for item in value]
                return 10.0 ** float(value)
            return mx.array(pow10(self.tolist()), dtype=mx.float64)
        return mx.array._mlx_array_orig_rpow(self, other)

    mx.array.__rpow__ = _array_rpow
if not hasattr(mx.array, "__divmod__"):
    def _array_divmod(self, other):
        left = _to_scalar(self)
        right = _to_scalar(other)
        if not isinstance(left, mx.array) and not isinstance(right, mx.array):
            return _builtins.divmod(left, right)
        left_arr = _to_mx(left)
        right_arr = _to_mx(right)
        quotient = mx.floor(left_arr / right_arr)
        return quotient, left_arr - quotient * right_arr

    mx.array.__divmod__ = _array_divmod
if not hasattr(mx.array, "__rdivmod__"):
    def _array_rdivmod(self, other):
        return _array_divmod(_to_mx(other), self)

    mx.array.__rdivmod__ = _array_rdivmod
if not hasattr(mx.array, "__rand__"):
    mx.array.__rand__ = lambda self, other: mx.logical_and(_to_mx(other), self)
if not hasattr(mx.array, "__ror__"):
    mx.array.__ror__ = lambda self, other: mx.logical_or(_to_mx(other), self)
if not hasattr(mx.array, "__rxor__"):
    mx.array.__rxor__ = lambda self, other: mx.not_equal(_to_mx(other), self)
if not hasattr(mx.array, "_mlx_array_orig_eq"):
    mx.array._mlx_array_orig_eq = mx.array.__eq__

    def _array_eq(self, other):
        if isinstance(other, (int, float, bool)) and self.size == 1:
            left = self.item()
            if isinstance(left, float) or isinstance(other, float):
                return mx.array(math.isclose(
                    float(left), float(other), rel_tol=1e-12, abs_tol=1e-12),
                    dtype=mx.bool_)
            return mx.array(left == other, dtype=mx.bool_)
        if isinstance(other, mx.array) and self.size == 1 and other.size == 1:
            left = self.item()
            right = other.item()
            if isinstance(left, float) or isinstance(right, float):
                return mx.array(math.isclose(
                    float(left), float(right), rel_tol=1e-12, abs_tol=1e-12),
                    dtype=mx.bool_)
            return mx.array(left == right, dtype=mx.bool_)
        if isinstance(other, (list, tuple)):
            other = mx.array(other, dtype=self.dtype)
        return mx.array._mlx_array_orig_eq(self, other)

    mx.array.__eq__ = _array_eq
if not hasattr(mx.array, "_mlx_array_orig_ne"):
    mx.array._mlx_array_orig_ne = mx.array.__ne__

    def _array_ne(self, other):
        if isinstance(other, (int, float, bool)) and self.size == 1:
            left = self.item()
            if isinstance(left, float) or isinstance(other, float):
                return mx.array(not math.isclose(
                    float(left), float(other), rel_tol=1e-12, abs_tol=1e-12),
                    dtype=mx.bool_)
            return mx.array(left != other, dtype=mx.bool_)
        if isinstance(other, mx.array) and self.size == 1 and other.size == 1:
            left = self.item()
            right = other.item()
            if isinstance(left, float) or isinstance(right, float):
                return mx.array(not math.isclose(
                    float(left), float(right), rel_tol=1e-12, abs_tol=1e-12),
                    dtype=mx.bool_)
            return mx.array(left != right, dtype=mx.bool_)
        if isinstance(other, (list, tuple)):
            other = mx.array(other, dtype=self.dtype)
        return mx.array._mlx_array_orig_ne(self, other)

    mx.array.__ne__ = _array_ne
if not hasattr(mx.array, "_mlx_array_orig_iter"):
    mx.array._mlx_array_orig_iter = mx.array.__iter__

    def _array_iter(self):
        if self.ndim == 0:
            raise TypeError("iteration over a 0-d array")

        def generate():
            for item in mx.array._mlx_array_orig_iter(self):
                if self.ndim == 1 and isinstance(item, mx.array) and item.size == 1:
                    yield item.item()
                else:
                    yield item

        return generate()

    mx.array.__iter__ = _array_iter
if not hasattr(mx.array, "data"):
    mx.array.data = property(lambda self: self)
if not hasattr(mx.array, "take"):
    mx.array.take = lambda self, indices, axis=None, mode=None: take(
        self, indices, axis=axis, mode=mode)
if not hasattr(mx.array, "repeat"):
    mx.array.repeat = lambda self, repeats, axis=None: repeat(
        self, repeats, axis=axis)
if not hasattr(mx.array, "_mlx_array_orig_mul"):
    mx.array._mlx_array_orig_mul = mx.array.__mul__

    def _array_mul(self, other):
        if isinstance(other, (list, tuple)):
            other = mx.array(other, dtype=self.dtype)
        elif isinstance(other, _PythonArray):
            other = mx.array(other.tolist(), dtype=self.dtype).reshape(other.shape)
        return mx.array._mlx_array_orig_mul(self, other)

    mx.array.__mul__ = _array_mul
if not hasattr(mx.array, "_mlx_array_orig_rmul"):
    mx.array._mlx_array_orig_rmul = mx.array.__rmul__

    def _array_rmul(self, other):
        if isinstance(other, (list, tuple)):
            other = mx.array(other, dtype=self.dtype)
        return mx.array._mlx_array_orig_rmul(self, other)

    mx.array.__rmul__ = _array_rmul
if not hasattr(mx.array, "_mlx_array_orig_truediv"):
    mx.array._mlx_array_orig_truediv = mx.array.__truediv__

    def _array_truediv(self, other):
        if isinstance(other, (list, tuple)):
            other = mx.array(other, dtype=self.dtype)
        return mx.array._mlx_array_orig_truediv(self, other)

    mx.array.__truediv__ = _array_truediv
if not hasattr(mx.array, "_mlx_array_orig_rtruediv"):
    mx.array._mlx_array_orig_rtruediv = mx.array.__rtruediv__

    def _array_rtruediv(self, other):
        if isinstance(other, (list, tuple)):
            other = mx.array(other, dtype=self.dtype)
        return mx.array._mlx_array_orig_rtruediv(self, other)

    mx.array.__rtruediv__ = _array_rtruediv
if not hasattr(mx.array, "_mlx_array_orig_astype"):
    mx.array._mlx_array_orig_astype = mx.array.astype

    def _array_astype(self, dtype, *args, **kwargs):
        return mx.array._mlx_array_orig_astype(
            self, _unwrap_dtype(dtype), *args, **kwargs)

    mx.array.astype = _array_astype
if not hasattr(mx.array, "_mlx_array_orig_setitem"):
    mx.array._mlx_array_orig_setitem = mx.array.__setitem__

    def _array_setitem(self, key, value):
        if isinstance(value, (list, tuple)):
            value = mx.array(value, dtype=self.dtype)
        elif isinstance(value, _PythonArray):
            value = mx.array(value.tolist(), dtype=self.dtype).reshape(value.shape)
        try:
            return mx.array._mlx_array_orig_setitem(self, key, value)
        except ValueError:
            is_bool_key = (
                isinstance(key, mx.array)
                and (key.dtype == mx.bool_ or "bool" in str(key.dtype)))
            if not is_bool_key:
                raise
            data_flat = list(_flatten(self.tolist()))
            mask_flat = list(_flatten(key.tolist()))
            if hasattr(value, "tolist"):
                value = value.tolist()
            if (hasattr(value, "shape")
                    and tuple(value.shape) == tuple(self.shape)):
                value_flat = list(_flatten(value.tolist()))
            else:
                value_flat = list(_flatten(value))
            if not value_flat:
                return None
            value_iter = itertools.cycle(value_flat) if len(value_flat) == 1 else iter(value_flat)
            for idx, keep in enumerate(mask_flat):
                if keep:
                    data_flat[idx] = next(value_iter)
            replacement = mx.array(_reshape_flat(data_flat, tuple(self.shape)),
                                   dtype=self.dtype)
            full_key = ((slice(None),) * self.ndim
                        if self.ndim else slice(None))
            return mx.array._mlx_array_orig_setitem(self, full_key, replacement)

    mx.array.__setitem__ = _array_setitem
if not hasattr(mx.array, "_mlx_array_orig_getitem"):
    mx.array._mlx_array_orig_getitem = mx.array.__getitem__

    def _array_getitem(self, key):
        def scalar_index(value):
            if isinstance(value, mx.array) and value.size == 1:
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
            if Ellipsis in key:
                ellipsis_at = key.index(Ellipsis)
                fill = self.ndim - (len(key) - 1)
                key = (key[:ellipsis_at] + (slice(None),) * fill
                       + key[ellipsis_at + 1:])
        else:
            key = normalize(key)
            if isinstance(key, range):
                key = list(key)

        def negative_slice(part):
            return (isinstance(part, slice) and
                    _builtins.any(
                        isinstance(value, int) and value < 0
                        for value in (part.start, part.stop, part.step)
                        if value is not None))

        if ((isinstance(key, tuple) and _builtins.any(
                negative_slice(part) for part in key))
                or negative_slice(key)):
            return mx.array(_python_getitem(self.tolist(), key), dtype=self.dtype)
        if isinstance(key, tuple) and _builtins.any(
                isinstance(part, (list, tuple)) for part in key):
            return mx.array(_python_getitem(self.tolist(), key), dtype=self.dtype)
        if isinstance(key, mx.array) and (key.dtype == mx.bool_ or
                                          str(key.dtype).endswith("bool_")):
            values = self.tolist()
            mask = key.tolist()
            if not isinstance(values, list):
                values = [values]
            if not isinstance(mask, list):
                mask = [mask]
            return mx.array([value for value, keep in zip(values, mask) if keep],
                            dtype=self.dtype)
        try:
            return mx.array._mlx_array_orig_getitem(self, key)
        except (NotImplementedError, ValueError):
            return mx.array(_python_getitem(self.tolist(), key), dtype=self.dtype)

    mx.array.__getitem__ = _array_getitem
if not hasattr(mx.array, "_mlx_array_orig_matmul"):
    mx.array._mlx_array_orig_matmul = mx.array.__matmul__

    def _array_matmul(self, other):
        if isinstance(other, (list, tuple)):
            other = mx.array([_to_scalar(item) for item in other],
                             dtype=self.dtype)
        return mx.array._mlx_array_orig_matmul(self, other)

    mx.array.__matmul__ = _array_matmul
if not hasattr(mx.array, "_mlx_array_orig_sub"):
    mx.array._mlx_array_orig_sub = mx.array.__sub__

    def _array_sub(self, other):
        if isinstance(other, (list, tuple)):
            other = mx.array([_to_scalar(item) for item in other],
                             dtype=self.dtype)
        elif isinstance(other, _PythonArray):
            other = mx.array(other.tolist(), dtype=self.dtype).reshape(other.shape)
        return mx.array._mlx_array_orig_sub(self, other)

    mx.array.__sub__ = _array_sub
if not hasattr(mx.array, "_mlx_array_orig_add"):
    mx.array._mlx_array_orig_add = mx.array.__add__

    def _array_add(self, other):
        if isinstance(other, (list, tuple)):
            other = mx.array([_to_scalar(item) for item in other],
                             dtype=self.dtype)
        elif isinstance(other, _PythonArray):
            other = mx.array(other.tolist(), dtype=self.dtype).reshape(other.shape)
        return mx.array._mlx_array_orig_add(self, other)

    mx.array.__add__ = _array_add
if not hasattr(mx.array, "flat"):
    def _array_flat(self):
        def flatten(value):
            if isinstance(value, list):
                for item in value:
                    yield from flatten(item)
            else:
                yield value
        return flatiter(list(flatten(self.tolist())))

    mx.array.flat = property(_array_flat)
if not hasattr(mx.array, "flags"):
    _array_writeable = {}

    def _clear_array_writeable(key):
        _array_writeable.pop(key, None)

    class _ArrayFlags:
        def __init__(self, array):
            self._key = id(array)
            weakref.finalize(array, _clear_array_writeable, self._key)

        @property
        def writeable(self):
            return _array_writeable.get(self._key, True)

        @writeable.setter
        def writeable(self, value):
            _array_writeable[self._key] = bool(value)

    mx.array.flags = property(lambda self: _ArrayFlags(self))

@dataclass(frozen=True)
class DType:
    """A small callable dtype wrapper around an MLX dtype.

    Dtypes like ``mlxarr.uint8`` are both valid ``dtype=`` values and callable
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
        arr = mx.array(x, dtype=self.mx_dtype)
        return arr.item() if arr.size == 1 else arr

    @property
    def type(self) -> "DType":
        return self

    @property
    def isnative(self) -> bool:
        return True

    def __eq__(self, other: Any) -> bool:
        return _unwrap_dtype(other) == self.mx_dtype

    def __hash__(self) -> int:
        return hash((self.name, self.mx_dtype))

    def __repr__(self) -> str:  # pragma: no cover
        return f"mlxarr.{self.name}"


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
double = float64
intp = int64
floating = float
integer = int
number = (int, float)
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
    "intp": intp,
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
    def __init__(self, values: Any):
        self._values = list(values)
        self._index = 0

    def __iter__(self):
        return self

    def __next__(self):
        if self._index >= len(self._values):
            raise StopIteration
        value = self._values[self._index]
        self._index += 1
        return value

    def __getitem__(self, key: Any):
        if isinstance(key, int):
            self._index = key
        return self._values[key]

    def tolist(self):
        return list(self._values)


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


class _PythonArray:
    def __init__(self, data: Any, dtype: Any | None = None,
                 shape: Tuple[int, ...] | None = None):
        if isinstance(data, _PythonArray):
            self._data = _copy_nested(data._data)
            self.dtype = data.dtype if dtype is None else dtype
        else:
            self._data = _copy_nested(data)
            self.dtype = dtype or _object_dtype
        self.shape = tuple(shape) if shape is not None else _infer_shape(self._data)
        self.ndim = len(self.shape)
        self.size = math.prod(self.shape) if self.shape else 1

    def __iter__(self):
        return iter(self._data if isinstance(self._data, list) else [self._data])

    def __len__(self):
        return self.shape[0] if self.shape else 1

    def __getitem__(self, key: Any) -> Any:
        if isinstance(key, mx.array):
            key = key.tolist()
        if isinstance(key, range):
            key = list(key)
        if isinstance(key, list) and key and _builtins.all(isinstance(v, bool) for v in key):
            return _PythonArray([v for v, keep in zip(self.tolist(), key) if keep],
                                dtype=self.dtype)
        if isinstance(key, list):
            values = self._data if isinstance(self._data, list) else [self._data]
            return _PythonArray([values[int(v)] for v in key], dtype=self.dtype)
        if isinstance(key, tuple):
            value = _python_getitem(self._data, key)
            if (self.size == 0 and self.ndim == 1 and len(key) == 2
                    and isinstance(key[0], slice) and key[1] is None):
                return _PythonArray(value, dtype=self.dtype,
                                    shape=(self.shape[0], 1))
            return _PythonArray(value, dtype=self.dtype) if isinstance(value, list) else value
        value = self._data[key] if isinstance(self._data, list) else self._data
        return _PythonArray(value, dtype=self.dtype) if isinstance(value, list) else value

    def __setitem__(self, key: Any, value: Any) -> None:
        if not isinstance(key, tuple):
            self._data[key] = value
            return
        target = self._data
        for part in key[:-1]:
            target = target[part]
        target[key[-1]] = value

    @property
    def flat(self):
        return iter(list(_flatten(self._data)))

    def item(self):
        if self.size != 1:
            raise ValueError("can only convert an array of size 1 to a Python scalar")
        return next(_flatten(self._data))

    def squeeze(self):
        return _PythonArray(_squeeze_nested(self._data), dtype=self.dtype)

    def ravel(self):
        return _PythonArray(list(_flatten(self._data)), dtype=self.dtype)

    def reshape(self, shape: Any):
        flat = list(_flatten(self._data))
        shape_tuple = _shape_tuple(shape)
        if -1 in shape_tuple:
            known = math.prod(v for v in shape_tuple if v != -1)
            inferred = len(flat) // known if known else 0
            shape_tuple = tuple(inferred if v == -1 else v for v in shape_tuple)
        return _PythonArray(_reshape_flat(flat, shape_tuple),
                            dtype=self.dtype)

    def astype(self, dtype: Any, *args: Any, **kwargs: Any):
        mx_dtype = _unwrap_dtype(dtype)
        if mx_dtype is _builtins.object:
            return _PythonArray(self, dtype=_object_dtype)
        data = self.tolist()
        if mx_dtype in {mx.float16, mx.float32, mx.float64, mx.bfloat16}:
            data = _coerce_nested(data, float)
        elif mx_dtype in {mx.int8, mx.int16, mx.int32, mx.int64,
                          mx.uint8, mx.uint16, mx.uint32, mx.uint64}:
            data = _coerce_nested(data, int)
        return mx.array(data, dtype=mx_dtype)

    def _elementwise(self, other: Any, op, out_dtype: Any | None = None):
        left = list(_flatten(self._data))
        if isinstance(other, _PythonArray):
            right = list(_flatten(other.tolist()))
        elif isinstance(other, mx.array):
            right = list(_flatten(other.tolist()))
        elif isinstance(other, (list, tuple)):
            right = list(_flatten(other))
        else:
            right = [other] * len(left)
        if len(right) == 1 and len(left) != 1:
            right = right * len(left)
        values = [op(a, b) for a, b in zip(left, right)]
        data = _reshape_flat(values, self.shape) if values else []
        if out_dtype is None:
            return _PythonArray(data, dtype=self.dtype, shape=self.shape)
        return mx.array(data, dtype=out_dtype).reshape(self.shape)

    def __eq__(self, other: Any):
        return self._elementwise(other, operator.eq, bool_.mx_dtype)

    def __ne__(self, other: Any):
        return self._elementwise(other, operator.ne, bool_.mx_dtype)

    def __lt__(self, other: Any):
        return self._elementwise(other, operator.lt, bool_.mx_dtype)

    def __le__(self, other: Any):
        return self._elementwise(other, operator.le, bool_.mx_dtype)

    def __gt__(self, other: Any):
        return self._elementwise(other, operator.gt, bool_.mx_dtype)

    def __ge__(self, other: Any):
        return self._elementwise(other, operator.ge, bool_.mx_dtype)

    def __neg__(self):
        return self._elementwise(0, lambda value, _: -value)

    def _numeric(self):
        return self.astype(float)

    def __add__(self, other: Any):
        if self.size == 0 and getattr(other, "size", None) == 0:
            return other
        return self._numeric() + other

    def __radd__(self, other: Any):
        if self.size == 0 and getattr(other, "size", None) == 0:
            return other
        return other + self._numeric()

    def __sub__(self, other: Any):
        if self.size == 0 and getattr(other, "size", None) == 0:
            return -other
        return self._numeric() - other

    def __rsub__(self, other: Any):
        if self.size == 0 and getattr(other, "size", None) == 0:
            return other
        return other - self._numeric()

    def tolist(self):
        return _copy_nested(self._data)


class _ObjectNDArray(_PythonArray):
    def __init__(self, shape: Any):
        if isinstance(shape, int):
            shape = (shape,)
        self._storage_shape = tuple(shape)
        super().__init__(_reshape_flat([None] * math.prod(self._storage_shape),
                                       self._storage_shape),
                         dtype=_object_dtype)

    def _offset(self, key: Any) -> int:
        if not isinstance(key, tuple):
            key = (key,)
        offset = 0
        stride = self.size
        for idx, dim in zip(key, self.shape):
            stride //= dim
            offset += idx * stride
        return offset

    def __getitem__(self, key: Any) -> Any:
        return list(_flatten(self._data))[self._offset(key)]

    def __setitem__(self, key: Any, value: Any) -> None:
        flat = list(_flatten(self._data))
        flat[self._offset(key)] = value
        self._data = _reshape_flat(flat, self.shape)


def _copy_nested(value: Any) -> Any:
    if isinstance(value, _PythonArray):
        return value.tolist()
    if isinstance(value, mx.array):
        return value.tolist()
    if isinstance(value, tuple):
        return [_copy_nested(v) for v in value]
    if isinstance(value, list):
        return [_copy_nested(v) for v in value]
    if isinstance(value, range):
        return list(value)
    return value


def _coerce_nested(value: Any, func) -> Any:
    if isinstance(value, _PythonArray):
        return _coerce_nested(value.tolist(), func)
    if isinstance(value, mx.array):
        return _coerce_nested(value.tolist(), func)
    if isinstance(value, tuple):
        return [_coerce_nested(v, func) for v in value]
    if isinstance(value, list):
        return [_coerce_nested(v, func) for v in value]
    return func(value)


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
    if isinstance(value, _PythonArray):
        return True
    if isinstance(value, (str, bytes, datetime, timedelta)) or value is None:
        return True
    if isinstance(value, (list, tuple)):
        return _builtins.any(_contains_object_data(v) for v in value)
    return False


def _contains_float_data(value: Any) -> bool:
    if isinstance(value, _PythonArray):
        value = value.tolist()
    if isinstance(value, mx.array):
        try:
            return dtype(value.dtype) == float64
        except AttributeError:
            return False
    if isinstance(value, float):
        return True
    if isinstance(value, (list, tuple)):
        return _builtins.any(_contains_float_data(v) for v in value)
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
        if isinstance(key, range):
            key = list(key)
        if isinstance(key, (list, tuple)):
            return [data[int(item)] for item in key]
        return data[key]
    if not key:
        return data
    first, *rest = key
    if isinstance(first, mx.array):
        first = first.tolist()
    if isinstance(first, range):
        first = list(first)
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
    if isinstance(x, flatiter):
        x = x.tolist()
    if isinstance(x, range):
        x = list(x)
    if isinstance(x, MaskedArray):
        x = x.data
    if isinstance(x, _PythonArray):
        if dtype is None or _unwrap_dtype(dtype) is _builtins.object:
            return x
        return x.astype(dtype)
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
    if isinstance(x, mx.array) and dtype is None:
        return x
    dtype = _unwrap_dtype(dtype)
    if isinstance(x, (list, tuple)):
        x = _copy_nested(x)
    if dtype is _builtins.object or _contains_object_data(x):
        return _PythonArray(x, dtype=_object_dtype)
    if dtype is None:
        try:
            if _contains_float_data(x):
                return mx.array(x, dtype=mx.float64)
            return mx.array(x)
        except (TypeError, ValueError):
            return _PythonArray(x, dtype=_object_dtype)
    return mx.array(x, dtype=dtype)


def _to_scalar(x: Any) -> Any:
    if isinstance(x, mx.array) and x.size == 1:
        return x.item()
    return x


def array(obj: Any, dtype: Any | None = None, copy: bool | None = True,
          order: Any | None = None, subok: bool = False, ndmin: int = 0,
          like: Any | None = None) -> mx.array:
    arr = _to_mx(obj, dtype=dtype)
    while getattr(arr, "ndim", 0) < ndmin:
        arr = arr.reshape((1,) + tuple(arr.shape))
    return arr


def asarray(obj: Any, dtype: Any | None = None, order: Any | None = None,
            copy: bool | None = None, like: Any | None = None) -> mx.array:
    return _to_mx(obj, dtype=dtype)


def asanyarray(obj: Any, dtype: Any | None = None, order: Any | None = None,
               like: Any | None = None) -> mx.array:
    return _to_mx(obj, dtype=dtype)


def atleast_1d(*arys: Any) -> Tuple[mx.array, ...] | mx.array:
    res = []
    for a in arys:
        arr = _to_mx(a)
        if arr.ndim == 0:
            arr = arr.reshape((1,)) if isinstance(arr, _PythonArray) else mx.reshape(arr, (1,))
        res.append(arr)
    return tuple(res) if len(res) > 1 else res[0]


def atleast_2d(*arys: Any) -> Tuple[mx.array, ...] | mx.array:
    res = []
    for a in arys:
        arr = _to_mx(a)
        if arr.ndim == 0:
            arr = arr.reshape((1, 1)) if isinstance(arr, _PythonArray) else mx.reshape(arr, (1, 1))
        elif arr.ndim == 1:
            shape = (1, arr.shape[0])
            arr = arr.reshape(shape) if isinstance(arr, _PythonArray) else mx.reshape(arr, shape)
        res.append(arr)
    return tuple(res) if len(res) > 1 else res[0]


def atleast_3d(*arys: Any) -> Tuple[mx.array, ...] | mx.array:
    res = []
    for a in arys:
        arr = _to_mx(a)
        if arr.ndim == 0:
            arr = arr.reshape((1, 1, 1)) if isinstance(arr, _PythonArray) else mx.reshape(arr, (1, 1, 1))
        elif arr.ndim == 1:
            shape = (1, arr.shape[0], 1)
            arr = arr.reshape(shape) if isinstance(arr, _PythonArray) else mx.reshape(arr, shape)
        elif arr.ndim == 2:
            shape = (1, *arr.shape)
            arr = arr.reshape(shape) if isinstance(arr, _PythonArray) else mx.reshape(arr, shape)
        res.append(arr)
    return tuple(res) if len(res) > 1 else res[0]


def zeros(shape: Any, dtype: Any | None = None) -> mx.array:
    if _unwrap_dtype(dtype) is _builtins.object:
        shape_tuple = _shape_tuple(shape)
        return _PythonArray(_reshape_flat([0] * math.prod(shape_tuple), shape_tuple),
                            dtype=_object_dtype)
    return mx.zeros(shape, dtype=_unwrap_dtype(dtype))


def ones(shape: Any, dtype: Any | None = None) -> mx.array:
    if _unwrap_dtype(dtype) is _builtins.object:
        shape_tuple = _shape_tuple(shape)
        return _PythonArray(_reshape_flat([1] * math.prod(shape_tuple), shape_tuple),
                            dtype=_object_dtype)
    return mx.ones(shape, dtype=_unwrap_dtype(dtype))


def full(shape: Any, fill_value: Any, dtype: Any | None = None) -> mx.array:
    if _unwrap_dtype(dtype) is _builtins.object or _contains_object_data(fill_value):
        shape_tuple = _shape_tuple(shape)
        return _PythonArray(_reshape_flat([fill_value] * math.prod(shape_tuple),
                                          shape_tuple),
                            dtype=_object_dtype)
    return mx.full(shape, fill_value, dtype=_unwrap_dtype(dtype))


def empty(shape: Any, dtype: Any | None = None) -> mx.array:
    if dtype is _builtins.object or dtype == "object":
        return _ObjectNDArray(shape)
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
    if isinstance(arr, _PythonArray) or _contains_object_data(fill_value):
        return full(arr.shape, fill_value, dtype=dtype or _object_dtype)
    return mx.full(arr.shape, fill_value, dtype=_unwrap_dtype(dtype) or arr.dtype)


def empty_like(a: Any, dtype: Any | None = None) -> mx.array:
    arr = _to_mx(a)
    return mx.zeros(arr.shape, dtype=_unwrap_dtype(dtype) or arr.dtype)


def arange(*args: Any, **kwargs: Any) -> mx.array:
    args = tuple(_to_scalar(arg) if isinstance(arg, mx.array) else arg for arg in args)
    if "dtype" in kwargs:
        kwargs["dtype"] = _unwrap_dtype(kwargs["dtype"])
    return mx.arange(*args, **kwargs)


def linspace(start: Any, stop: Any, num: int = 50, endpoint: bool = True,
             retstep: bool = False, dtype: Any | None = None,
             axis: int = 0) -> mx.array:
    start = _to_scalar(start)
    stop = _to_scalar(stop)
    num = int(num)
    if num <= 0:
        result = mx.array([], dtype=_unwrap_dtype(dtype) or mx.float32)
        return (result, nan) if retstep else result
    div = num - 1 if endpoint and num > 1 else num
    step = (stop - start) / div if div else nan
    effective_stop = stop if endpoint else stop - step
    kwargs = {}
    if dtype is not None:
        kwargs["dtype"] = _unwrap_dtype(dtype)
    else:
        kwargs["dtype"] = mx.float64
    result = mx.linspace(start, effective_stop, num, **kwargs)
    return (result, step) if retstep else result


def logspace(start: float, stop: float, num: int = 50, base: float = 10.0) -> mx.array:
    return mx.power(base, mx.linspace(start, stop, num))


def geomspace(start: float, stop: float, num: int = 50) -> mx.array:
    return mx.exp(mx.linspace(math.log(start), math.log(stop), num))


def reshape(a: Any, newshape: Any) -> mx.array:
    arr = _to_mx(a)
    if isinstance(arr, _PythonArray):
        return arr.reshape(_shape_tuple(newshape))
    return mx.reshape(arr, _shape_tuple(newshape))


def ravel(a: Any) -> mx.array:
    arr = _to_mx(a)
    if isinstance(arr, _PythonArray):
        return arr.ravel()
    return mx.reshape(arr, (arr.size,))


def squeeze(a: Any, axis: Any | None = None) -> mx.array:
    arr = _to_mx(a)
    if isinstance(arr, _PythonArray):
        return arr.squeeze()
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


def concatenate(arrays: Sequence[Any], axis: int = 0, out: Any | None = None) -> mx.array:
    converted = [_to_mx(a) for a in arrays]
    try:
        result = mx.concatenate(converted, axis=axis)
    except (TypeError, ValueError):
        lists = [a.tolist() if hasattr(a, "tolist") else a for a in converted]
        if axis in (0, None):
            data = []
            for item in lists:
                if isinstance(item, list):
                    data.extend(item)
                else:
                    data.append(item)
            result = _to_mx(data)
        elif axis == 1:
            rows = []
            for row_parts in zip(*lists):
                row = []
                for part in row_parts:
                    row.extend(part if isinstance(part, list) else [part])
                rows.append(row)
            result = _to_mx(rows)
        else:
            raise
    if out is not None:
        out[:] = result.astype(out.dtype)
        return out
    return result


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
    arr = _to_mx(a)
    reps_tuple = _shape_tuple(reps)
    if isinstance(arr, _PythonArray):
        data = arr.tolist()
        if len(reps_tuple) == 1:
            repeat_count = reps_tuple[0]
            if isinstance(data, list):
                return _PythonArray(data * repeat_count, dtype=arr.dtype)
            return _PythonArray([data] * repeat_count, dtype=arr.dtype)
        raise NotImplementedError("object tiling supports one-dimensional reps")
    return mx.tile(arr, reps_tuple)


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
    return mx.take(arr, idx, axis=axis)


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


def where(condition: Any, x: Any = None, y: Any = None) -> mx.array:
    if x is None and y is None:
        return nonzero(condition)
    if x is None or y is None:
        raise ValueError("where requires both x and y when selecting values")
    cond = _to_mx(condition)
    x_arr = _to_mx(x)
    y_arr = _to_mx(y)
    try:
        return mx.where(cond, x_arr, y_arr)
    except (TypeError, ValueError):
        def choose(c, xv, yv):
            if isinstance(c, list):
                return [
                    choose(ci,
                           xv[i] if isinstance(xv, list) else xv,
                           yv[i] if isinstance(yv, list) else yv)
                    for i, ci in enumerate(c)
                ]
            return xv if c else yv

        x_list = x_arr.tolist() if hasattr(x_arr, "tolist") else x_arr
        y_list = y_arr.tolist() if hasattr(y_arr, "tolist") else y_arr
        return _to_mx(choose(cond.tolist(), x_list, y_list))


def extract(condition: Any, arr: Any):
    arr_mx = _to_mx(arr)
    cond_flat = list(_flatten(_to_mx(condition).tolist()))
    arr_flat = list(_flatten(arr_mx.tolist()))
    values = [value for keep, value in zip(cond_flat, arr_flat) if keep]
    if isinstance(arr_mx, _PythonArray):
        return _PythonArray(values, dtype=arr_mx.dtype)
    return mx.array(values, dtype=arr_mx.dtype)


def divide(a: Any, b: Any, dtype: Any | None = None, **kwargs: Any) -> mx.array:
    result = mx.divide(_to_mx(a), _to_mx(b))
    return result.astype(dtype) if dtype is not None else result


true_divide = divide


def mod(a: Any, b: Any) -> mx.array:
    return _to_mx(a) % _to_mx(b)


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
        return _PythonArray(values, dtype=_object_dtype)
    return mx.array(values)


def intersect1d(ar1: Any, ar2: Any, assume_unique: bool = False,
                return_indices: bool = False):
    left = list(_flatten(_to_mx(ar1).tolist()))
    right = list(_flatten(_to_mx(ar2).tolist()))
    values = sorted(set(left).intersection(right))
    out = (_PythonArray(values, dtype=_object_dtype)
           if _contains_object_data(values) else mx.array(values))
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


def sort(a: Any, axis: int | None = -1) -> mx.array:
    return mx.sort(_to_mx(a), axis=axis)


def argsort(a: Any, axis: int | None = -1) -> mx.array:
    return mx.argsort(_to_mx(a), axis=axis)


def argmax(a: Any, axis: int | None = None) -> Any:
    return _to_scalar(mx.argmax(_to_mx(a), axis=axis))


def argmin(a: Any, axis: int | None = None) -> Any:
    return _to_scalar(mx.argmin(_to_mx(a), axis=axis))


def sum(a: Any, axis: int | None = None) -> Any:
    return _to_scalar(mx.sum(_to_mx(a), axis=axis))


def mean(a: Any, axis: int | None = None) -> Any:
    return _to_scalar(mx.mean(_to_mx(a), axis=axis))


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


def cumsum(a: Any, axis: int | None = None) -> mx.array:
    return mx.cumsum(_to_mx(a), axis=axis)


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
    if isinstance(arr, _PythonArray):
        result = _python_isfinite(arr.tolist())
        return mx.array(result, dtype=mx.bool_) if isinstance(result, list) else result
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
    if isinstance(a, _PythonArray) or isinstance(b, _PythonArray):
        return _testing_equal(a, b)
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


def sqrt(a: Any) -> mx.array:
    return mx.sqrt(_to_mx(a))


def exp(a: Any) -> mx.array:
    return mx.exp(_to_mx(a))


def log(a: Any) -> mx.array:
    return mx.log(_to_mx(a))


def log2(a: Any) -> mx.array:
    return mx.log2(_to_mx(a))


def log10(a: Any) -> mx.array:
    return mx.log10(_to_mx(a))


def _power_impl(a: Any, b: Any) -> mx.array:
    if isinstance(a, (int, float)) and float(a) == 10.0:
        b_arr = _to_mx(b)
        values = _coerce_nested(b_arr.tolist(), lambda value: 10.0 ** float(value))
        return mx.array(values, dtype=mx.float64)
    return mx.power(_to_mx(a), _to_mx(b))


def power(a: Any, b: Any) -> mx.array:
    return _power_impl(a, b)


def square(a: Any) -> mx.array:
    return mx.square(_to_mx(a))


def floor(a: Any) -> mx.array:
    return mx.floor(_to_mx(a))


def ceil(a: Any) -> mx.array:
    return mx.ceil(_to_mx(a))


def round(a: Any, decimals: int = 0) -> mx.array:
    return mx.round(_to_mx(a), decimals=decimals)


around = round


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


def degrees(a: Any) -> mx.array:
    return mx.degrees(_to_mx(a))


def radians(a: Any) -> mx.array:
    return mx.radians(_to_mx(a))


def deg2rad(a: Any) -> mx.array:
    return radians(a)


def rad2deg(a: Any) -> mx.array:
    return degrees(a)


def hypot(x: Any, y: Any) -> mx.array:
    x_mx = _to_mx(x)
    y_mx = _to_mx(y)
    return mx.sqrt(x_mx * x_mx + y_mx * y_mx)


def matmul(a: Any, b: Any) -> mx.array:
    return mx.matmul(_to_mx(a), _to_mx(b))


def dot(a: Any, b: Any) -> mx.array:
    return mx.matmul(_to_mx(a), _to_mx(b))


def outer(a: Any, b: Any) -> mx.array:
    a = _to_mx(a).reshape((-1, 1))
    b = _to_mx(b).reshape((1, -1))
    return mx.matmul(a, b)


class _Power:
    def __call__(self, a: Any, b: Any, **kwargs: Any) -> mx.array:
        return _power_impl(a, b)

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
    if m is None:
        m = n
    out = mx.zeros((n, m), dtype=_unwrap_dtype(dtype) or float32.mx_dtype)
    idx = mx.arange(_py_min(n, m))
    if k >= 0:
        out[idx, idx + k] = 1
    else:
        out[idx - k, idx] = 1
    return out


def identity(n: int, dtype: Any | None = None) -> mx.array:
    return eye(n, n, dtype=dtype)


def meshgrid(*arrays: Any, **kwargs: Any) -> List[mx.array]:
    arrays = [_to_mx(a) for a in arrays]
    return mx.meshgrid(*arrays, **kwargs)


def broadcast_to(a: Any, shape: Any) -> mx.array:
    arr = _to_mx(a)
    shape_tuple = _shape_tuple(shape)
    try:
        return mx.broadcast_to(arr, shape_tuple)
    except (TypeError, ValueError):
        if getattr(arr, "size", None) == 0 and math.prod(shape_tuple) == 0:
            return _PythonArray(_reshape_flat([], shape_tuple),
                                dtype=getattr(arr, "dtype", _object_dtype),
                                shape=shape_tuple)
        if (getattr(arr, "ndim", None) == 1 and len(shape_tuple) == 2
                and arr.shape[0] == shape_tuple[1]):
            if isinstance(arr, _PythonArray):
                return _PythonArray([arr.tolist()] * shape_tuple[0],
                                    dtype=arr.dtype, shape=shape_tuple)
            return mx.stack([arr] * shape_tuple[0], axis=0)
        if getattr(arr, "size", None) == 1:
            return full(shape_tuple, _to_scalar(arr),
                        dtype=getattr(arr, "dtype", None))
        raise


def broadcast_arrays(*args: Any, **kwargs: Any) -> Tuple[mx.array, ...]:
    return mx.broadcast_arrays(*[_to_mx(a) for a in args])


def shape(a: Any) -> Tuple[int, ...]:
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
    return not isinstance(obj, (list, tuple, dict, mx.array, _PythonArray))


def iterable(obj: Any) -> bool:
    if isinstance(obj, (mx.array, _PythonArray)):
        return obj.ndim > 0
    try:
        iter(obj)
        return True
    except (TypeError, IndexError):
        return False


class _ObjectArray(list):
    def tolist(self):
        return list(self)


def vectorize(pyfunc, otypes: Any | None = None, **_kwargs: Any):
    def wrapped(*args, **kwargs):
        vectors = [list(_flatten(_to_mx(a).tolist())) if isinstance(a, mx.array) else list(_flatten(a)) for a in args]
        result = [pyfunc(*vals, **kwargs) for vals in zip(*vectors)]
        if otypes == "O" or otypes == ["O"] or otypes == ("O",):
            return _ObjectArray(result)
        try:
            return mx.array(result)
        except (TypeError, ValueError):
            return _ObjectArray(result)
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


def histogram(a: Any, bins: Any = 10,
              range: Tuple[float, float] | None = None,
              weights: Any | None = None,
              density: bool | None = None):
    arr = [float(v) for v in _flatten(_to_mx(a).tolist())]
    weight_values = ([1.0] * len(arr) if weights is None
                     else [float(v) for v in _flatten(_to_mx(weights).tolist())])
    if isinstance(bins, str):
        bins = 10
    if isinstance(bins, mx.array):
        bins = bins.tolist()
    if isinstance(bins, (list, tuple)):
        bin_edges = [float(v) for v in bins]
        bin_count = _py_max(len(bin_edges) - 1, 0)
    else:
        bin_count = int(bins)
        if range is None:
            min_v = _py_min(arr) if arr else 0.0
            max_v = _py_max(arr) if arr else 1.0
        else:
            min_v, max_v = range
        if min_v == max_v:
            min_v -= 0.5
            max_v += 0.5
        bin_edges = [min_v + (max_v - min_v) * i / bin_count
                     for i in _builtins.range(bin_count + 1)]
    counts = [0.0] * bin_count
    if bin_count == 0:
        return mx.array(counts), mx.array(bin_edges)
    min_v = bin_edges[0]
    max_v = bin_edges[-1]
    width = (max_v - min_v) / bin_count if bin_count else 1.0
    for v, weight in zip(arr, weight_values):
        if math.isnan(v):
            continue
        if v < min_v or v > max_v:
            continue
        if v == max_v:
            idx = bin_count - 1
        elif isinstance(bins, (list, tuple)):
            idx = next((i for i in _builtins.range(bin_count)
                        if bin_edges[i] <= v < bin_edges[i + 1]), None)
            if idx is None:
                continue
        else:
            idx = _py_min(int((v - min_v) / width), bin_count - 1)
        counts[idx] += weight
    if density:
        total = _builtins.sum(counts)
        if total:
            counts = [count / (total * (bin_edges[i + 1] - bin_edges[i]))
                      for i, count in enumerate(counts)]
    return mx.array(counts), mx.array(bin_edges)


def histogram_bin_edges(a: Any, bins: Any = 10,
                        range: Tuple[float, float] | None = None,
                        weights: Any | None = None):
    return histogram(a, bins=bins, range=range, weights=weights)[1]


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


def bincount(x: Any, minlength: int | None = None):
    xs = [int(v) for v in _flatten(_to_mx(x).tolist())]
    size = _py_max(xs) + 1 if xs else 0
    if minlength is not None:
        size = _py_max(size, minlength)
    counts = [0] * size
    for v in xs:
        counts[v] += 1
    return mx.array(counts)


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
    x_list = _to_mx(x).tolist()
    xp_list = _to_mx(xp).tolist()
    fp_list = _to_mx(fp).tolist()
    out = []
    for xv in x_list:
        if xv <= xp_list[0]:
            out.append(fp_list[0])
            continue
        if xv >= xp_list[-1]:
            out.append(fp_list[-1])
            continue
        for i in range(1, len(xp_list)):
            if xv <= xp_list[i]:
                x0, x1 = xp_list[i - 1], xp_list[i]
                y0, y1 = fp_list[i - 1], fp_list[i]
                t = (xv - x0) / (x1 - x0)
                out.append(y0 + t * (y1 - y0))
                break
    return mx.array(out)


def searchsorted(a: Any, v: Any, side: str = "left"):
    arr = sorted(_flatten(_to_mx(a).tolist()))
    scalar = isscalar(v)
    values = _to_mx(v).tolist()
    if not isinstance(values, list):
        values = [values]
    result = []
    for val in values:
        if side == "left":
            idx = next((i for i, x in enumerate(arr) if x >= val), len(arr))
        else:
            idx = next((i for i, x in enumerate(arr) if x > val), len(arr))
        result.append(idx)
    if scalar:
        return result[0]
    return mx.array(result)


def digitize(x: Any, bins: Any, right: bool = False):
    side = "left" if right else "right"
    return searchsorted(bins, x, side=side)


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
    if isinstance(q, mx.array):
        q = q.tolist()
    if isinstance(q, (list, tuple)):
        q_list = q
    else:
        q_list = [q]
    out = []
    for qv in q_list:
        qv = _to_scalar(qv)
        idx = int(round((qv / 100.0) * (len(arr) - 1)))
        out.append(arr[idx])
    return mx.array(out)


def median(a: Any, axis: int | None = None, overwrite_input: bool | None = None):
    arr = _to_mx(a)
    if axis is None:
        return percentile(arr, 50)[0]
    data = arr.tolist()
    if axis < 0:
        axis += arr.ndim
    if axis == 0:
        return mx.array([percentile(list(col), 50)[0] for col in zip(*data)])
    if axis == 1:
        return mx.array([percentile(row, 50)[0] for row in data])
    raise NotImplementedError("median currently supports axis 0, axis 1, or None")


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
        return tuple(mx.array([]) for _ in range(arr.ndim))
    axes = list(zip(*coords))
    return tuple(mx.array(axis) for axis in axes)


def count_nonzero(a: Any, axis: int | None = None) -> Any:
    return sum(_to_mx(a) != 0, axis=axis)


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
    if arg2 is integer:
        return kind1 in {"i", "u"}
    if arg2 is floating:
        return kind1 == "f"
    if kind2 is None:
        return arg1 == arg2
    return kind1 == kind2


def min_scalar_type(arg: Any):
    if isinstance(arg, MaskedArray):
        return dtype(arg.dtype)
    if isinstance(arg, _PythonArray):
        return dtype(arg.dtype)
    if isinstance(arg, mx.array):
        return dtype(arg.dtype)
    if isinstance(arg, (list, tuple, range)):
        values = list(_flatten(_copy_nested(arg)))
        if not values:
            return float64
        if _builtins.any(isinstance(value, float) for value in values):
            return float64
        if _builtins.any(isinstance(value, int) and not isinstance(value, bool)
                         for value in values):
            return int64
        if _builtins.all(isinstance(value, bool) for value in values):
            return bool_
        return _object_dtype
    try:
        return dtype(type(arg))
    except AttributeError:
        return _object_dtype


def require(a: Any, **kwargs: Any):
    return _to_mx(a)


def may_share_memory(a: Any, b: Any, max_work: Any | None = None) -> bool:
    return a is b


def broadcast_arrays(*args: Any, **kwargs: Any):
    converted = [None if arg is None else _to_mx(arg) for arg in args]
    try:
        if _builtins.all(arg is not None and not isinstance(arg, _PythonArray)
                         for arg in converted):
            return mx.broadcast_arrays(*converted)
    except (TypeError, ValueError, RuntimeError):
        pass

    target_shape: Tuple[int, ...] = ()
    for arg in converted:
        if arg is None:
            continue
        shape = tuple(getattr(arg, "shape", ()))
        if math.prod(shape or (1,)) > math.prod(target_shape or (1,)):
            target_shape = shape

    target_size = math.prod(target_shape) if target_shape else 1
    result = []
    for arg in converted:
        if arg is None:
            data = _reshape_flat([None] * target_size, target_shape)
            result.append(_PythonArray(data, dtype=_object_dtype))
        elif isinstance(arg, _PythonArray):
            if arg.shape == target_shape:
                result.append(arg)
            elif arg.size == 1:
                data = _reshape_flat([arg.item()] * target_size, target_shape)
                result.append(_PythonArray(data, dtype=arg.dtype))
            else:
                raise ValueError("shape mismatch: objects cannot be broadcast")
        else:
            result.append(mx.broadcast_to(arg, target_shape))
    return tuple(result)


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
    if isinstance(items, _PythonArray):
        yield from _flatten(items.tolist())
        return
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
        return mx.random.uniform(shape=_random_shape(args=shape))

    def randn(self, *shape: int):
        return mx.random.normal(shape=_random_shape(args=shape))

    def randint(self, low: int, high: int | None = None, size: Any | None = None):
        if high is None:
            low, high = 0, low
        return mx.random.randint(low, high, shape=_random_shape(size))

    def multivariate_normal(self, mean: Any, cov: Any, size: Any | None = None):
        # MLX currently only supports float32 for multivariate normals.
        if size is None:
            shape = ()
        elif isinstance(size, int):
            shape = (size,)
        else:
            shape = tuple(size)

        mean_arr = asarray(mean, dtype=float32)
        cov_arr = asarray(cov, dtype=float32)
        return mx.random.multivariate_normal(mean_arr, cov_arr, shape=shape, dtype=float32.mx_dtype)

    def random(self, size: Any | None = None):
        return mx.random.uniform(shape=_random_shape(size))

    random_sample = random
    sample = random
    ranf = random

    def uniform(self, low: float = 0.0, high: float = 1.0, size: Any | None = None):
        return mx.random.uniform(low=low, high=high, shape=_random_shape(size))

    def normal(self, loc: float = 0.0, scale: float = 1.0, size: Any | None = None):
        return mx.random.normal(shape=_random_shape(size)) * scale + loc

    def lognormal(self, mean: float = 0.0, sigma: float = 1.0, size: Any | None = None):
        return mx.exp(self.normal(mean, sigma, size=size))

    def permutation(self, x: Any):
        arr = _to_mx(x)
        idx = mx.random.permutation(arr.shape[0])
        return arr[idx]

    def shuffle(self, x: Any) -> None:
        shuffled = self.permutation(x)
        if isinstance(x, list):
            x[:] = shuffled.tolist()
            return None
        if isinstance(x, mx.array):
            full_key = (slice(None),) * x.ndim if x.ndim else slice(None)
            x[full_key] = shuffled
        return None

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
        return _testing_plain(value.tolist())
    if isinstance(value, _PythonArray):
        return _testing_plain(value.tolist())
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
        if isinstance(a, (list, tuple, mx.array, _PythonArray)) or isinstance(b, (list, tuple, mx.array, _PythonArray)):
            return self.assert_array_equal(a, b, err_msg=err_msg)
        if a != b:
            raise AssertionError(err_msg or f"{a!r} != {b!r}")

    def assert_almost_equal(self, a: Any, b: Any, decimal: int = 6):
        return self.assert_array_almost_equal(a, b, decimal=decimal)

    def assert_array_max_ulp(self, a: Any, b: Any, maxulp: int = 1, dtype: Any | None = None):
        return self.assert_allclose(a, b, rtol=1e-6, atol=1e-12)

    def assert_array_almost_equal_nulp(self, a: Any, b: Any, nulp: int = 1):
        return self.assert_array_max_ulp(a, b, maxulp=nulp)

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
    __array_priority__ = 1000

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

    def fill(self, value: Any) -> None:
        full_key = (slice(None),) * self.data.ndim if self.data.ndim else slice(None)
        self.data[full_key] = full(self.data.shape, value, dtype=self.data.dtype)

    def compressed(self):
        if self.mask is None:
            return ravel(self.data)
        return ravel(self.data[~self.mask])

    def ravel(self):
        mask = ravel(self.mask) if self.mask is not None else None
        return MaskedArray(ravel(self.data), mask)

    def reshape(self, *shape: Any):
        newshape = shape[0] if len(shape) == 1 else shape
        mask = reshape(self.mask, newshape) if self.mask is not None else None
        return MaskedArray(reshape(self.data, newshape), mask)

    def tolist(self):
        return self.data.tolist()

    def min(self, *args: Any, **kwargs: Any):
        return mx.min(_to_mx(self.filled()), *args, **kwargs)

    def max(self, *args: Any, **kwargs: Any):
        return mx.max(_to_mx(self.filled()), *args, **kwargs)

    def astype(self, dtype: Any, *args: Any, **kwargs: Any):
        return MaskedArray(self.data.astype(dtype), self.mask)

    def any(self, axis: int | None = None):
        return any(self.filled(False), axis=axis)

    def all(self, axis: int | None = None):
        return all(self.filled(True), axis=axis)

    def _combined_mask(self, other: Any):
        other_mask = other.mask if isinstance(other, MaskedArray) else None
        if self.mask is None:
            return other_mask
        if other_mask is None:
            return self.mask
        return logical_or(self.mask, other_mask)

    def _binary(self, other: Any, op):
        other_data = other.data if isinstance(other, MaskedArray) else other
        return MaskedArray(op(self.data, other_data), self._combined_mask(other))

    def _rbinary(self, other: Any, op):
        other_data = other.data if isinstance(other, MaskedArray) else other
        return MaskedArray(op(other_data, self.data), self._combined_mask(other))

    def __mul__(self, other: Any):
        return self._binary(other, operator.mul)

    def __rmul__(self, other: Any):
        return self._rbinary(other, operator.mul)

    def __truediv__(self, other: Any):
        return self._binary(other, operator.truediv)

    def __rtruediv__(self, other: Any):
        return self._rbinary(other, operator.truediv)

    def __add__(self, other: Any):
        return self._binary(other, operator.add)

    def __radd__(self, other: Any):
        return self._rbinary(other, operator.add)

    def __sub__(self, other: Any):
        return self._binary(other, operator.sub)

    def __rsub__(self, other: Any):
        return self._rbinary(other, operator.sub)

    def __neg__(self):
        return MaskedArray(-self.data, self.mask)

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

    def __setitem__(self, key: Any, value: Any) -> None:
        if isinstance(value, MaskedArray):
            self.data[key] = value.data
            if self.mask is not None and value.mask is not None:
                self.mask[key] = value.mask
            return
        self.data[key] = value
        if self.mask is not None:
            self.mask[key] = False

    def __array__(self, dtype: Any | None = None, copy: bool | None = None):
        if dtype is None:
            return self.data
        try:
            return self.data.astype(dtype)
        except (TypeError, ValueError):
            return self.data


class _MA:
    masked = masked
    MaskedArray = MaskedArray

    @staticmethod
    def _dtype_like(value: Any) -> bool:
        if value in {_builtins.bool, _builtins.int, _builtins.float,
                     _builtins.object}:
            return True
        if isinstance(value, (DType, str, type)):
            return True
        try:
            return value in _DTYPE_BY_MX
        except TypeError:
            return False

    def array(self, data: Any, dtype: Any | None = None,
              copy: bool | None = None, order: Any | None = None,
              mask: Any | None = None, **kwargs: Any):
        if mask is None and dtype is not None and not self._dtype_like(dtype):
            mask = dtype
            dtype = None
        arr = _to_mx(data, dtype=dtype)
        if mask is None or mask is False:
            mask_arr = mx.zeros(arr.shape, dtype=bool_.mx_dtype)
        elif mask is True:
            mask_arr = mx.ones(arr.shape, dtype=bool_.mx_dtype)
        else:
            mask_arr = _to_mx(mask)
        return MaskedArray(data=arr, mask=mask_arr)

    def masked_array(self, data: Any, mask: Any | None = None,
                     dtype: Any | None = None, copy: bool | None = None,
                     **kwargs: Any):
        return self.array(data, dtype=dtype, copy=copy, mask=mask, **kwargs)

    def asarray(self, data: Any, dtype: Any | None = None):
        if isinstance(data, MaskedArray):
            return data.astype(dtype) if dtype is not None else data
        return self.array(data, dtype=dtype)

    def asanyarray(self, data: Any, dtype: Any | None = None):
        return self.asarray(data, dtype=dtype)

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

    def mask_or(self, m1: Any, m2: Any, copy: bool | None = None,
                shrink: bool | None = None):
        if (m1 is None or m1 is False) and (m2 is None or m2 is False):
            return None
        if m1 is None or m1 is False:
            return _to_mx(m2)
        if m2 is None or m2 is False:
            return _to_mx(m1)
        return logical_or(m1, m2)

    def filled(self, data: Any, fill_value: Any = 0):
        if isinstance(data, MaskedArray):
            return data.filled(fill_value)
        return _to_mx(data)

    def min(self, data: Any, *args: Any, **kwargs: Any):
        return data.min(*args, **kwargs) if isinstance(data, MaskedArray) else min(
            data, *args, **kwargs)

    def max(self, data: Any, *args: Any, **kwargs: Any):
        return data.max(*args, **kwargs) if isinstance(data, MaskedArray) else max(
            data, *args, **kwargs)

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

    def empty(self, shape: Any, dtype: Any | None = None):
        arr = globals()["empty"](shape, dtype=dtype)
        return MaskedArray(data=arr, mask=mx.zeros(_shape_tuple(shape),
                                                   dtype=bool_.mx_dtype))

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

    def arctan2(self, y: Any, x: Any):
        mask = self.mask_or(self.getmask(y), self.getmask(x))
        result = mx.arctan2(_to_mx(self.getdata(y)), _to_mx(self.getdata(x)))
        return MaskedArray(data=result, mask=mask) if mask is not None else result

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


class _Linalg:
    def det(self, a: Any):
        arr = _to_mx(a)

        def matrix_det(values):
            rows = [list(row) for row in values]
            n = len(rows)
            if _builtins.any(len(row) != n for row in rows):
                raise ValueError("last two dimensions must be square")
            if n == 0:
                return 1.0
            if n == 1:
                return rows[0][0]
            if n == 2:
                return rows[0][0] * rows[1][1] - rows[0][1] * rows[1][0]
            total = 0.0
            for col, value in enumerate(rows[0]):
                minor = [row[:col] + row[col + 1:] for row in rows[1:]]
                total += ((-1) ** col) * value * matrix_det(minor)
            return total

        def det_nested(values, ndim):
            if ndim == 2:
                return matrix_det(values)
            return [det_nested(item, ndim - 1) for item in values]

        result = det_nested(arr.tolist(), arr.ndim)
        return result if arr.ndim == 2 else mx.array(result)

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

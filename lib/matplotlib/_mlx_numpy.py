"""MLX-backed NumPy compatibility shim.

This module provides a minimal NumPy-like API implemented on top of MLX.
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

# Matplotlib's NumPy-facing internals routinely request float64 arrays.
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
if not hasattr(mx.array, "_mlx_numpy_orig_astype"):
    mx.array._mlx_numpy_orig_astype = mx.array.astype

    def _array_astype(self, dtype, *args, **kwargs):
        return mx.array._mlx_numpy_orig_astype(
            self, _unwrap_dtype(dtype), *args, **kwargs)

    mx.array.astype = _array_astype
if not hasattr(mx.array, "_mlx_numpy_orig_setitem"):
    mx.array._mlx_numpy_orig_setitem = mx.array.__setitem__

    def _array_setitem(self, key, value):
        if isinstance(value, (list, tuple)):
            value = mx.array(value, dtype=self.dtype)
        return mx.array._mlx_numpy_orig_setitem(self, key, value)

    mx.array.__setitem__ = _array_setitem
if not hasattr(mx.array, "_mlx_numpy_orig_getitem"):
    mx.array._mlx_numpy_orig_getitem = mx.array.__getitem__

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
            return mx.array._mlx_numpy_orig_getitem(self, key)
        except (NotImplementedError, ValueError):
            return mx.array(_python_getitem(self.tolist(), key), dtype=self.dtype)

    mx.array.__getitem__ = _array_getitem
if not hasattr(mx.array, "_mlx_numpy_orig_matmul"):
    mx.array._mlx_numpy_orig_matmul = mx.array.__matmul__

    def _array_matmul(self, other):
        if isinstance(other, (list, tuple)):
            other = mx.array([_to_scalar(item) for item in other],
                             dtype=self.dtype)
        return mx.array._mlx_numpy_orig_matmul(self, other)

    mx.array.__matmul__ = _array_matmul
if not hasattr(mx.array, "_mlx_numpy_orig_sub"):
    mx.array._mlx_numpy_orig_sub = mx.array.__sub__

    def _array_sub(self, other):
        if isinstance(other, (list, tuple)):
            other = mx.array([_to_scalar(item) for item in other],
                             dtype=self.dtype)
        return mx.array._mlx_numpy_orig_sub(self, other)

    mx.array.__sub__ = _array_sub
if not hasattr(mx.array, "_mlx_numpy_orig_add"):
    mx.array._mlx_numpy_orig_add = mx.array.__add__

    def _array_add(self, other):
        if isinstance(other, (list, tuple)):
            other = mx.array([_to_scalar(item) for item in other],
                             dtype=self.dtype)
        return mx.array._mlx_numpy_orig_add(self, other)

    mx.array.__add__ = _array_add
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

@dataclass(frozen=True)
class DType:
    """A small callable dtype wrapper (NumPy-style) around an MLX dtype.

    In NumPy, dtypes like ``np.uint8`` are both valid ``dtype=`` values and
    callable scalar/array constructors (e.g. ``np.uint8(0)``). MLX exposes dtype
    objects but they are not callable, so we wrap them here.
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

    def __eq__(self, other: Any) -> bool:
        return _unwrap_dtype(other) == self.mx_dtype

    def __hash__(self) -> int:
        return hash((self.name, self.mx_dtype))

    def __repr__(self) -> str:  # pragma: no cover
        return f"np.{self.name}"


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


# Public dtypes (NumPy-like: usable as dtype= and callable constructors).
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
if not hasattr(type(mx.float32), "_mlx_numpy_orig_eq"):
    type(mx.float32)._mlx_numpy_orig_eq = type(mx.float32).__eq__

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

# NumPy-like scalars/constants
pi = math.pi
e = math.e
inf = float("inf")
nan = float("nan")
newaxis = None

ndarray = mx.array


class flatiter:
    """Placeholder for NumPy's ndarray.flat iterator type."""


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
    def __init__(self, data: Any, dtype: Any | None = None):
        if isinstance(data, _PythonArray):
            self._data = _copy_nested(data._data)
            self.dtype = data.dtype if dtype is None else dtype
        else:
            self._data = _copy_nested(data)
            self.dtype = dtype or _object_dtype
        self.shape = _infer_shape(self._data)
        self.ndim = len(self.shape)
        self.size = math.prod(self.shape) if self.shape else 1

    def __iter__(self):
        return iter(self._data if isinstance(self._data, list) else [self._data])

    def __len__(self):
        return self.shape[0] if self.shape else 1

    def __getitem__(self, key: Any) -> Any:
        if isinstance(key, mx.array):
            key = key.tolist()
        if isinstance(key, list) and key and _builtins.all(isinstance(v, bool) for v in key):
            return _PythonArray([v for v, keep in zip(self.tolist(), key) if keep],
                                dtype=self.dtype)
        if isinstance(key, tuple):
            value = self._data
            for part in key:
                value = value[part]
            return value
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
        return _PythonArray(_reshape_flat(list(_flatten(self._data)), tuple(shape)),
                            dtype=self.dtype)

    def astype(self, dtype: Any, *args: Any, **kwargs: Any):
        mx_dtype = _unwrap_dtype(dtype)
        if mx_dtype is _builtins.object:
            return _PythonArray(self, dtype=_object_dtype)
        return mx.array(self.tolist(), dtype=mx_dtype)

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
    if isinstance(value, mx.array):
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
    if isinstance(value, _PythonArray):
        return True
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
    result = mx.min(arr, axis=axis)
    if initial is not None:
        result = mx.minimum(result, mx.array(initial, dtype=arr.dtype))
    return _to_scalar(result)


def max(a: Any, axis: int | None = None, initial: Any | None = None) -> Any:
    arr = _to_mx(a)
    if arr.size == 0 and initial is not None:
        return initial
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
    raise NotImplementedError("fromfile is not supported without NumPy")


def genfromtxt(*args: Any, **kwargs: Any) -> mx.array:
    raise NotImplementedError("genfromtxt is not supported without NumPy")


def loadtxt(*args: Any, **kwargs: Any) -> mx.array:
    raise NotImplementedError("loadtxt is not supported without NumPy")


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


def issubdtype(arg1: Any, arg2: Any):
    return True


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


def datetime64(value: Any, *args: Any, **kwargs: Any):
    if isinstance(value, datetime):
        return value
    if isinstance(value, str):
        try:
            return datetime.fromisoformat(value)
        except ValueError:
            return datetime.fromisoformat(value + "T00:00:00")
    return value


def timedelta64(value: Any, *args: Any, **kwargs: Any):
    if isinstance(value, timedelta):
        return value
    if isinstance(value, (int, float)):
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


class _Testing:
    def assert_allclose(self, a: Any, b: Any, rtol: float = 1e-5, atol: float = 1e-8, err_msg: str | None = None):
        if not allclose(a, b, rtol=rtol, atol=atol):
            raise AssertionError(err_msg or "Arrays are not equal within tolerance")

    def assert_array_equal(self, a: Any, b: Any, err_msg: str | None = None):
        if not array_equal(a, b):
            raise AssertionError(err_msg or "Arrays are not equal")

    def assert_array_almost_equal(self, a: Any, b: Any, decimal: int = 6):
        rtol = 10 ** (-decimal)
        self.assert_allclose(a, b, rtol=rtol, atol=rtol)

    def assert_array_less(self, a: Any, b: Any, err_msg: str | None = None):
        if not bool(mx.all(less(a, b)).item()):
            raise AssertionError(err_msg or "Arrays are not ordered")

    def assert_equal(self, a: Any, b: Any, err_msg: str | None = None):
        if isinstance(a, (list, tuple, _PythonArray)) or isinstance(b, (list, tuple, _PythonArray)):
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

    def array(self, data: Any, mask: Any | None = None, dtype: Any | None = None, copy: bool | None = None):
        return MaskedArray(data=_to_mx(data, dtype=dtype), mask=_to_mx(mask) if mask is not None else None)

    def masked_array(self, data: Any, mask: Any | None = None, dtype: Any | None = None, copy: bool | None = None):
        return self.array(data, mask=mask, dtype=dtype, copy=copy)

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

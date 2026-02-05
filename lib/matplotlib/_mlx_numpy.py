"""MLX-backed NumPy compatibility shim.

This module provides a minimal NumPy-like API implemented on top of MLX.
It is intentionally incomplete but covers the subset used by this codebase.
"""
from __future__ import annotations

import math
import itertools
import operator
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any, Iterable, Iterator, List, Sequence, Tuple

import mlx.core as mx

# Public dtypes
bool_ = mx.bool_
float16 = mx.float16
float32 = mx.float32
float64 = mx.float64
bfloat16 = mx.bfloat16
int8 = mx.int8
int16 = mx.int16
int32 = mx.int32
int64 = mx.int64
uint8 = mx.uint8
uint16 = mx.uint16
uint32 = mx.uint32
uint64 = mx.uint64

# NumPy-like scalars/constants
pi = math.pi
e = math.e
inf = float("inf")
nan = float("nan")
newaxis = None

ndarray = mx.array


def _to_mx(x: Any, dtype: Any | None = None) -> mx.array:
    if isinstance(x, mx.array) and dtype is None:
        return x
    if dtype is None:
        return mx.array(x)
    return mx.array(x, dtype=dtype)


def _to_scalar(x: Any) -> Any:
    if isinstance(x, mx.array) and x.size == 1:
        return x.item()
    return x


def array(obj: Any, dtype: Any | None = None) -> mx.array:
    return _to_mx(obj, dtype=dtype)


def asarray(obj: Any, dtype: Any | None = None) -> mx.array:
    return _to_mx(obj, dtype=dtype)


def asanyarray(obj: Any, dtype: Any | None = None) -> mx.array:
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
    return mx.zeros(shape, dtype=dtype)


def ones(shape: Any, dtype: Any | None = None) -> mx.array:
    return mx.ones(shape, dtype=dtype)


def full(shape: Any, fill_value: Any, dtype: Any | None = None) -> mx.array:
    return mx.full(shape, fill_value, dtype=dtype)


def empty(shape: Any, dtype: Any | None = None) -> mx.array:
    # MLX does not expose uninitialized arrays; use zeros as a safe fallback.
    return mx.zeros(shape, dtype=dtype)


def zeros_like(a: Any, dtype: Any | None = None) -> mx.array:
    arr = _to_mx(a)
    return mx.zeros(arr.shape, dtype=dtype or arr.dtype)


def ones_like(a: Any, dtype: Any | None = None) -> mx.array:
    arr = _to_mx(a)
    return mx.ones(arr.shape, dtype=dtype or arr.dtype)


def full_like(a: Any, fill_value: Any, dtype: Any | None = None) -> mx.array:
    arr = _to_mx(a)
    return mx.full(arr.shape, fill_value, dtype=dtype or arr.dtype)


def empty_like(a: Any, dtype: Any | None = None) -> mx.array:
    arr = _to_mx(a)
    return mx.zeros(arr.shape, dtype=dtype or arr.dtype)


def arange(*args: Any, **kwargs: Any) -> mx.array:
    return mx.arange(*args, **kwargs)


def linspace(*args: Any, **kwargs: Any) -> mx.array:
    return mx.linspace(*args, **kwargs)


def logspace(start: float, stop: float, num: int = 50, base: float = 10.0) -> mx.array:
    return mx.power(base, mx.linspace(start, stop, num))


def geomspace(start: float, stop: float, num: int = 50) -> mx.array:
    return mx.exp(mx.linspace(math.log(start), math.log(stop), num))


def reshape(a: Any, newshape: Any) -> mx.array:
    return mx.reshape(_to_mx(a), newshape)


def ravel(a: Any) -> mx.array:
    arr = _to_mx(a)
    return mx.reshape(arr, (arr.size,))


def squeeze(a: Any, axis: Any | None = None) -> mx.array:
    return mx.squeeze(_to_mx(a), axis=axis)


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
    return mx.concatenate([_to_mx(a) for a in arrays], axis=axis)


def column_stack(tup: Sequence[Any]) -> mx.array:
    arrays = [atleast_2d(a) for a in tup]
    arrays = [a.T if a.ndim == 1 else a for a in arrays]
    return concatenate(arrays, axis=1)


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


def take(a: Any, indices: Any, axis: int | None = None) -> mx.array:
    return mx.take(_to_mx(a), indices, axis=axis)


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
    return mx.array(values)


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


def min(a: Any, axis: int | None = None) -> Any:
    return _to_scalar(mx.min(_to_mx(a), axis=axis))


def max(a: Any, axis: int | None = None) -> Any:
    return _to_scalar(mx.max(_to_mx(a), axis=axis))


def prod(a: Any, axis: int | None = None) -> Any:
    return _to_scalar(mx.prod(_to_mx(a), axis=axis))


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


def isfinite(a: Any) -> mx.array:
    return mx.isfinite(_to_mx(a))


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
        idx = mx.arange(min(arr.shape))
        if k >= 0:
            return arr[idx, idx + k]
        return arr[idx - k, idx]
    raise ValueError("diag expects 1D or 2D array")


def eye(n: int, m: int | None = None, k: int = 0, dtype: Any | None = None) -> mx.array:
    if m is None:
        m = n
    out = mx.zeros((n, m), dtype=dtype or float32)
    idx = mx.arange(min(n, m))
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
    return mx.broadcast_to(_to_mx(a), shape)


def broadcast_arrays(*args: Any) -> Tuple[mx.array, ...]:
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
    return not isinstance(obj, (list, tuple, dict, mx.array))


def iterable(obj: Any) -> bool:
    try:
        iter(obj)
        return True
    except TypeError:
        return False


def vectorize(pyfunc):
    def wrapped(*args, **kwargs):
        vectors = [list(_flatten(_to_mx(a).tolist())) if isinstance(a, mx.array) else list(_flatten(a)) for a in args]
        result = [pyfunc(*vals, **kwargs) for vals in zip(*vectors)]
        return mx.array(result)
    return wrapped


def fromiter(iterable_obj: Iterable[Any], dtype: Any | None = None, count: int | None = None) -> mx.array:
    items = list(iterable_obj) if count is None else list(itertools.islice(iterable_obj, count))
    return mx.array(items, dtype=dtype)


def frombuffer(buffer_obj: bytes, dtype: Any | None = None) -> mx.array:
    # Interpret buffer as uint8 unless dtype provided
    data = list(buffer_obj)
    return mx.array(data, dtype=dtype or uint8)


def fromfile(*args: Any, **kwargs: Any) -> mx.array:
    raise NotImplementedError("fromfile is not supported without NumPy")


def genfromtxt(*args: Any, **kwargs: Any) -> mx.array:
    raise NotImplementedError("genfromtxt is not supported without NumPy")


def loadtxt(*args: Any, **kwargs: Any) -> mx.array:
    raise NotImplementedError("loadtxt is not supported without NumPy")


def histogram(a: Any, bins: int = 10, range: Tuple[float, float] | None = None):
    arr = _to_mx(a).tolist()
    if range is None:
        min_v = min(arr)
        max_v = max(arr)
    else:
        min_v, max_v = range
    bin_edges = [min_v + (max_v - min_v) * i / bins for i in range(bins + 1)]
    counts = [0] * bins
    for v in arr:
        if v < min_v or v > max_v:
            continue
        idx = min(int((v - min_v) / (max_v - min_v) * bins), bins - 1)
        counts[idx] += 1
    return mx.array(counts), mx.array(bin_edges)


def histogram2d(x: Any, y: Any, bins: int = 10, range: Tuple[Tuple[float, float], Tuple[float, float]] | None = None):
    x_list = _to_mx(x).tolist()
    y_list = _to_mx(y).tolist()
    if range is None:
        x_min, x_max = min(x_list), max(x_list)
        y_min, y_max = min(y_list), max(y_list)
    else:
        (x_min, x_max), (y_min, y_max) = range
    x_edges = [x_min + (x_max - x_min) * i / bins for i in range(bins + 1)]
    y_edges = [y_min + (y_max - y_min) * i / bins for i in range(bins + 1)]
    counts = [[0 for _ in range(bins)] for _ in range(bins)]
    for xv, yv in zip(x_list, y_list):
        if xv < x_min or xv > x_max or yv < y_min or yv > y_max:
            continue
        xi = min(int((xv - x_min) / (x_max - x_min) * bins), bins - 1)
        yi = min(int((yv - y_min) / (y_max - y_min) * bins), bins - 1)
        counts[xi][yi] += 1
    return mx.array(counts), mx.array(x_edges), mx.array(y_edges)


def bincount(x: Any, minlength: int | None = None):
    xs = [int(v) for v in _flatten(_to_mx(x).tolist())]
    size = max(xs) + 1 if xs else 0
    if minlength is not None:
        size = max(size, minlength)
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
        start = m - 1
        end = n
        out = out[start:end]
    elif mode == "same":
        start = (m - 1) // 2
        end = start + n
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


def r_(seq):
    parts = []
    for item in seq:
        if isinstance(item, slice):
            parts.append(_slice_to_array(item))
        else:
            parts.append(_to_mx(item))
    return concatenate(parts, axis=0)


def c_(seq):
    parts = []
    for item in seq:
        if isinstance(item, slice):
            parts.append(_slice_to_array(item))
        else:
            parts.append(_to_mx(item))
    return concatenate(parts, axis=1)


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


def argwhere(a: Any):
    arr = _to_mx(a)
    coords = []
    for idx, v in _iter_indices(arr.tolist()):
        if v:
            coords.append(idx)
    return mx.array(coords)


def unravel_index(indices: Any, shape: Sequence[int]):
    idx = int(_to_mx(indices).item())
    coords = []
    for dim in reversed(shape):
        coords.append(idx % dim)
        idx //= dim
    return tuple(reversed(coords))


def ravel_multi_index(multi_index: Sequence[Any], dims: Sequence[int]):
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


def can_cast(from_: Any, to: Any):
    return True


def issubdtype(arg1: Any, arg2: Any):
    return True


def min_scalar_type(arg: Any):
    return type(arg)


def require(a: Any, **kwargs: Any):
    return _to_mx(a)


def broadcast_arrays(*args: Any):
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


class _Random:
    def seed(self, seed_val: int | None = None):
        mx.random.seed(seed_val or 0)

    def rand(self, *shape: int):
        return mx.random.uniform(shape)

    def randn(self, *shape: int):
        return mx.random.normal(shape)

    def randint(self, low: int, high: int | None = None, size: Any | None = None):
        if high is None:
            low, high = 0, low
        if size is None:
            size = ()
        return mx.random.randint(low, high, size)

    def random(self, size: Any | None = None):
        if size is None:
            size = ()
        return mx.random.uniform(size)

    def uniform(self, low: float = 0.0, high: float = 1.0, size: Any | None = None):
        if size is None:
            size = ()
        return mx.random.uniform(size) * (high - low) + low

    def normal(self, loc: float = 0.0, scale: float = 1.0, size: Any | None = None):
        if size is None:
            size = ()
        return mx.random.normal(size) * scale + loc

    def permutation(self, x: Any):
        arr = _to_mx(x)
        idx = mx.random.permutation(arr.shape[0])
        return arr[idx]


random = _Random()


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

    def filled(self, fill_value: Any = 0):
        if self.mask is None:
            return self.data
        return mx.where(self.mask, _to_mx(fill_value), self.data)

    def __array__(self):  # pragma: no cover - compatibility shim
        return self.data


class _MA:
    masked = masked

    def array(self, data: Any, mask: Any | None = None, dtype: Any | None = None, copy: bool | None = None):
        return MaskedArray(data=_to_mx(data, dtype=dtype), mask=_to_mx(mask) if mask is not None else None)

    def masked_array(self, data: Any, mask: Any | None = None, dtype: Any | None = None, copy: bool | None = None):
        return self.array(data, mask=mask, dtype=dtype, copy=copy)

    def asarray(self, data: Any):
        return self.array(data) if not isinstance(data, MaskedArray) else data

    def asanyarray(self, data: Any):
        return self.asarray(data)

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
        return mx.zeros(arr.shape, dtype=bool_)

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
        arr = mx.zeros(shape, dtype=dtype or float32)
        mask = mx.ones(shape, dtype=bool_)
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

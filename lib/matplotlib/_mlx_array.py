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
import sys
import weakref
from dataclasses import dataclass
from decimal import Decimal
from datetime import datetime, timedelta
from typing import Any, Iterable, Iterator, List, Sequence, Tuple

import mlx.core as mx
try:
    from matplotlib import _mlx_overrides
except ImportError:
    _mlx_overrides = None

_PANDAS_BOOL_INDEXER_PATCHED = False


def _patch_pandas_bool_indexer() -> None:
    global _PANDAS_BOOL_INDEXER_PATCHED
    if _PANDAS_BOOL_INDEXER_PATCHED:
        return
    pd_common = sys.modules.get("pandas.core.common")
    pd_frame = sys.modules.get("pandas.core.frame")
    if pd_common is None or pd_frame is None:
        return
    original_is_bool_indexer = pd_common.is_bool_indexer
    original_check_bool_indexer = pd_frame.check_bool_indexer

    def is_bool_indexer(key: Any) -> bool:
        if (isinstance(key, mx.array)
                and key.dtype == mx.bool_
                and key.ndim == 1):
            return True
        return original_is_bool_indexer(key)

    def check_bool_indexer(index: Any, key: Any) -> Any:
        if (isinstance(key, mx.array)
                and key.dtype == mx.bool_
                and key.ndim == 1):
            return key
        return original_check_bool_indexer(index, key)

    pd_common.is_bool_indexer = is_bool_indexer
    pd_frame.check_bool_indexer = check_bool_indexer
    _PANDAS_BOOL_INDEXER_PATCHED = True


def _coerce_float64_value(target: Any, value: Any,
                          stream: Any | None = None) -> Any:
    if _mlx_overrides is None:
        return value
    if isinstance(value, float):
        target_dtype = getattr(target, "dtype", None)
        if target_dtype in {
                mx.bool_, mx.int8, mx.int16, mx.int32, mx.int64,
                mx.uint8, mx.uint16, mx.uint32, mx.uint64}:
            return _mlx_overrides.float64_scalar(value, stream=stream)
    return _mlx_overrides.coerce_float64_value(target, value, stream=stream)


# Matplotlib's array-facing internals routinely request float64 arrays.
# MLX's GPU backend rejects float64 constructors, so keep this compatibility
# layer on CPU unless callers explicitly move arrays elsewhere.
mx.set_default_device(mx.cpu)
if not hasattr(mx.array, "copy"):
    mx.array.copy = lambda self, order="C": mx.contiguous(self)
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
if not hasattr(mx.array, "__array_interface__"):
    _MX_DTYPE_TO_TYPESTR = {
        mx.float32: "<f4", mx.float64: "<f8", mx.float16: "<f2",
        mx.bfloat16: "<V2",  # bfloat16 is not standard IEEE float16
        mx.int8: "<i1", mx.int16: "<i2", mx.int32: "<i4", mx.int64: "<i8",
        mx.uint8: "|u1", mx.uint16: "<u2", mx.uint32: "<u4", mx.uint64: "<u8",
        mx.bool_: "|u1",
    }
    _MX_DTYPE_TO_ARRAY_FMT = {
        mx.float32: "f", mx.float64: "d",
        mx.int8: "b", mx.int16: "h", mx.int32: "i", mx.int64: "q",
        mx.uint8: "B", mx.uint16: "H", mx.uint32: "I", mx.uint64: "Q",
        mx.bool_: "B",
    }

    def _mx_array_interface(self):
        typestr = _MX_DTYPE_TO_TYPESTR.get(self.dtype, "<f4")
        fmt = _MX_DTYPE_TO_ARRAY_FMT.get(self.dtype)
        flat = list(_flatten(self.tolist()))
        from array import array as _stdlib_array
        if fmt is not None:
            data = _stdlib_array(fmt, flat).tobytes()
        else:
            # float16/bfloat16: convert via float32
            data = _stdlib_array("f", [float(v) for v in flat]).tobytes()
            typestr = "<f4"
        return {
            "data": data,
            "shape": self.shape,
            "typestr": typestr,
            "version": 3,
        }

    mx.array.__array_interface__ = property(_mx_array_interface)
if not hasattr(mx.array, "_mlx_array_orig_rpow"):
    mx.array._mlx_array_orig_rpow = mx.array.__rpow__

    def _array_rpow(self, other):
        other = _coerce_float64_value(self, other)
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
        if other is None:
            return mx.zeros(self.shape, dtype=mx.bool_)
        if (isinstance(other, (int, float, bool))
                and self.ndim == 0
                and self.size == 1):
            left = self.item()
            if isinstance(left, float) or isinstance(other, float):
                return mx.array(math.isclose(
                    float(left), float(other), rel_tol=1e-12, abs_tol=1e-12),
                    dtype=mx.bool_)
            return mx.array(left == other, dtype=mx.bool_)
        if (isinstance(other, mx.array)
                and self.ndim == 0
                and other.ndim == 0
                and self.size == 1
                and other.size == 1):
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
        if other is None:
            return mx.ones(self.shape, dtype=mx.bool_)
        if (isinstance(other, (int, float, bool))
                and self.ndim == 0
                and self.size == 1):
            left = self.item()
            if isinstance(left, float) or isinstance(other, float):
                return mx.array(not math.isclose(
                    float(left), float(other), rel_tol=1e-12, abs_tol=1e-12),
                    dtype=mx.bool_)
            return mx.array(left != other, dtype=mx.bool_)
        if (isinstance(other, mx.array)
                and self.ndim == 0
                and other.ndim == 0
                and self.size == 1
                and other.size == 1):
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
if not hasattr(mx.array, "argsort"):
    mx.array.argsort = lambda self, axis=-1: argsort(self, axis=axis)
if not hasattr(mx.array, "_mlx_array_orig_mul"):
    mx.array._mlx_array_orig_mul = mx.array.__mul__

    def _array_mul(self, other):
        other = _coerce_float64_value(self, other)
        if isinstance(other, timedelta):
            return _ObjectArray(
                _coerce_nested(self.tolist(), lambda value: value * other),
                dtype=_object_dtype)
        if "MaskedArray" in globals() and isinstance(other, MaskedArray):
            return other.__rmul__(self)
        if isinstance(other, (list, tuple)):
            other = mx.array(other, dtype=self.dtype)
        elif isinstance(other, _ObjectArray):
            other = mx.array(other.tolist(), dtype=self.dtype).reshape(other.shape)
        return mx.array._mlx_array_orig_mul(self, other)

    mx.array.__mul__ = _array_mul
    mx.array.__imul__ = _array_mul
if not hasattr(mx.array, "_mlx_array_orig_rmul"):
    mx.array._mlx_array_orig_rmul = mx.array.__rmul__

    def _array_rmul(self, other):
        other = _coerce_float64_value(self, other)
        if isinstance(other, timedelta):
            return _ObjectArray(
                _coerce_nested(self.tolist(), lambda value: other * value),
                dtype=_object_dtype)
        if isinstance(other, (list, tuple)):
            other = mx.array(other, dtype=self.dtype)
        return mx.array._mlx_array_orig_rmul(self, other)

    mx.array.__rmul__ = _array_rmul
if not hasattr(mx.array, "_mlx_array_orig_truediv"):
    mx.array._mlx_array_orig_truediv = mx.array.__truediv__

    def _array_truediv(self, other):
        other = _coerce_float64_value(self, other)
        if isinstance(other, (list, tuple)):
            other = mx.array(other, dtype=self.dtype)
        return mx.array._mlx_array_orig_truediv(self, other)

    mx.array.__truediv__ = _array_truediv
    mx.array.__itruediv__ = _array_truediv
if not hasattr(mx.array, "_mlx_array_orig_rtruediv"):
    mx.array._mlx_array_orig_rtruediv = mx.array.__rtruediv__

    def _array_rtruediv(self, other):
        other = _coerce_float64_value(self, other)
        if isinstance(other, (list, tuple)):
            other = mx.array(other, dtype=self.dtype)
        return mx.array._mlx_array_orig_rtruediv(self, other)

    mx.array.__rtruediv__ = _array_rtruediv
if not hasattr(mx.array, "_mlx_array_orig_floordiv"):
    mx.array._mlx_array_orig_floordiv = mx.array.__floordiv__

    def _array_floordiv(self, other):
        other = _coerce_float64_value(self, other)
        return mx.array._mlx_array_orig_floordiv(self, other)

    mx.array.__floordiv__ = _array_floordiv
if not hasattr(mx.array, "_mlx_array_orig_rfloordiv"):
    mx.array._mlx_array_orig_rfloordiv = mx.array.__rfloordiv__

    def _array_rfloordiv(self, other):
        other = _coerce_float64_value(self, other)
        return mx.array._mlx_array_orig_rfloordiv(self, other)

    mx.array.__rfloordiv__ = _array_rfloordiv
if not hasattr(mx.array, "_mlx_array_orig_mod"):
    mx.array._mlx_array_orig_mod = mx.array.__mod__

    def _array_mod(self, other):
        other = _coerce_float64_value(self, other)
        return mx.array._mlx_array_orig_mod(self, other)

    mx.array.__mod__ = _array_mod
if not hasattr(mx.array, "_mlx_array_orig_rmod"):
    mx.array._mlx_array_orig_rmod = mx.array.__rmod__

    def _array_rmod(self, other):
        other = _coerce_float64_value(self, other)
        return mx.array._mlx_array_orig_rmod(self, other)

    mx.array.__rmod__ = _array_rmod
if not hasattr(mx.array, "_mlx_array_orig_astype"):
    mx.array._mlx_array_orig_astype = mx.array.astype

    def _array_astype(self, dtype, *args, **kwargs):
        dtype = _unwrap_dtype(dtype)
        if dtype is _builtins.object:
            def to_nested(array):
                if array.ndim == 0:
                    return array.item()
                return [to_nested(array[idx]) for idx in range(array.shape[0])]
            return _ObjectArray(to_nested(self), dtype=_object_dtype)
        return mx.array._mlx_array_orig_astype(self, dtype, *args, **kwargs)

    mx.array.astype = _array_astype
if not hasattr(mx.array, "_mlx_array_orig_max_method"):
    mx.array._mlx_array_orig_max_method = mx.array.max

    def _array_max_method(self, axis=None, out=None, keepdims=False,
                          initial=None, where=True):
        if initial is not None and self.size == 0:
            return mx.array(initial, dtype=self.dtype)
        result = mx.array._mlx_array_orig_max_method(
            self, axis=axis, keepdims=keepdims)
        if initial is not None:
            result = mx.maximum(result, mx.array(initial, dtype=result.dtype))
        return result

    mx.array.max = _array_max_method
if not hasattr(mx.array, "_mlx_array_orig_min_method"):
    mx.array._mlx_array_orig_min_method = mx.array.min

    def _array_min_method(self, axis=None, out=None, keepdims=False,
                          initial=None, where=True):
        if initial is not None and self.size == 0:
            return mx.array(initial, dtype=self.dtype)
        result = mx.array._mlx_array_orig_min_method(
            self, axis=axis, keepdims=keepdims)
        if initial is not None:
            result = mx.minimum(result, mx.array(initial, dtype=result.dtype))
        return result

    mx.array.min = _array_min_method
if not hasattr(mx.array, "_mlx_array_orig_setitem"):
    mx.array._mlx_array_orig_setitem = mx.array.__setitem__

    def _array_setitem(self, key, value):
        def scalar_index(part):
            if isinstance(part, MaskedArray):
                part = part.data
            if (isinstance(part, mx.array)
                    and part.ndim == 0
                    and part.size == 1
                    and part.dtype != mx.bool_
                    and "bool" not in str(part.dtype)):
                return int(part.item())
            return part

        if isinstance(key, tuple):
            key = tuple(scalar_index(part) for part in key)
        else:
            key = scalar_index(key)
        if _mlx_overrides is not None:
            value = _coerce_float64_value(self, value)
        if value is None and self.dtype in {
                mx.float16, mx.float32, mx.float64, mx.bfloat16,
                mx.complex64, mx.int8, mx.int16, mx.int32, mx.int64,
                mx.uint8, mx.uint16, mx.uint32, mx.uint64}:
            value = 0
        def setitem_value(value):
            if isinstance(value, MaskedArray):
                if (value.mask is not None
                        and self.dtype in {mx.float16, mx.float32, mx.float64,
                                           mx.bfloat16, mx.complex64}):
                    return value.filled(nan)
                return value.data
            if isinstance(value, _ObjectArray):
                return _copy_nested(value._data)
            return value
        value = setitem_value(value)
        if isinstance(value, (list, tuple)):
            values = [setitem_value(v) for v in value]
            if self.dtype == mx.float64:
                exact = _exact_float64_array_from_python(values)
                value = exact if exact is not None else mx.array(
                    values, dtype=self.dtype)
            else:
                value = mx.array(values, dtype=self.dtype)
        def is_empty_integer_index(part):
            if isinstance(part, mx.array):
                return (part.size == 0
                        and part.dtype != mx.bool_
                        and "bool" not in str(part.dtype))
            if isinstance(part, memoryview):
                return len(part) == 0
            if isinstance(part, list):
                return len(part) == 0
            return False

        if (is_empty_integer_index(key)
                or (isinstance(key, tuple)
                    and _builtins.any(is_empty_integer_index(part) for part in key))):
            return None

        def is_bool_array(part):
            return (isinstance(part, mx.array)
                    and (part.dtype == mx.bool_ or "bool" in str(part.dtype)))

        if is_bool_array(key):
            replacement_value = _to_mx(value, dtype=self.dtype)
            if tuple(key.shape) == tuple(self.shape):
                selection = key
            elif key.ndim == 1 and self.ndim > 1 and key.shape[0] == self.shape[0]:
                selection = key
                while selection.ndim < self.ndim:
                    selection = mx.expand_dims(selection, -1)
            elif key.size == self.size:
                selection = mx.reshape(key, self.shape)
            else:
                selection = None
            if selection is not None:
                replacement = mx.where(selection, replacement_value, self)
                full_key = ((slice(None),) * self.ndim
                            if self.ndim else slice(None))
                return mx.array._mlx_array_orig_setitem(
                    self, full_key, replacement)

        try:
            return mx.array._mlx_array_orig_setitem(self, key, value)
        except (ValueError, SystemError):
            if (isinstance(key, tuple)
                    and len(key) == 2
                    and is_bool_array(key[0])
                    and isinstance(key[1], int)):
                mask = key[0]
                while mask.ndim < self.ndim:
                    mask = mx.expand_dims(mask, -1)
                axis_mask = mx.equal(
                    mx.arange(self.shape[1], dtype=mx.int32), key[1])
                axis_shape = (1, self.shape[1]) + (1,) * (self.ndim - 2)
                axis_mask = mx.reshape(axis_mask, axis_shape)
                selection = mx.logical_and(mask, axis_mask)
                replacement_value = _to_mx(value, dtype=self.dtype)
                replacement = mx.where(selection, replacement_value, self)
                full_key = ((slice(None),) * self.ndim
                            if self.ndim else slice(None))
                return mx.array._mlx_array_orig_setitem(
                    self, full_key, replacement)
            is_bool_key = (
                isinstance(key, mx.array)
                and (key.dtype == mx.bool_ or "bool" in str(key.dtype)))
            if not is_bool_key:
                raise
            if tuple(key.shape) == tuple(self.shape):
                replacement_value = _to_mx(value, dtype=self.dtype)
                replacement = mx.where(key, replacement_value, self)
                full_key = ((slice(None),) * self.ndim
                            if self.ndim else slice(None))
                return mx.array._mlx_array_orig_setitem(
                    self, full_key, replacement)
            if key.size == self.size:
                mask = mx.reshape(key, self.shape)
                replacement_value = _to_mx(value, dtype=self.dtype)
                replacement = mx.where(mask, replacement_value, self)
                full_key = ((slice(None),) * self.ndim
                            if self.ndim else slice(None))
                return mx.array._mlx_array_orig_setitem(
                    self, full_key, replacement)
            raise

    mx.array.__setitem__ = _array_setitem
if not hasattr(mx.array, "_mlx_array_orig_view"):
    mx.array._mlx_array_orig_view = mx.array.view

    def _array_view(self, dtype, *args, **kwargs):
        return mx.array._mlx_array_orig_view(
            self, _unwrap_dtype(dtype), *args, **kwargs)

    mx.array.view = _array_view
if not hasattr(mx.array, "_mlx_array_orig_getitem"):
    mx.array._mlx_array_orig_getitem = mx.array.__getitem__

    def _array_getitem(self, key):
        def is_bool_array(value):
            return (isinstance(value, mx.array)
                    and (value.dtype == mx.bool_ or "bool" in str(value.dtype)))

        def scalar_index(value):
            if (isinstance(value, mx.array)
                    and value.ndim == 0
                    and value.size == 1):
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
                specified = _builtins.sum(
                    1 for part in key
                    if part is not Ellipsis and part is not None)
                fill = self.ndim - specified
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

        def bool_indices(mask):
            indices = mx.arange(mask.shape[0], dtype=mx.int32)
            sentinel = mx.full(mask.shape, mask.shape[0], dtype=mx.int32)
            selected = mx.sort(mx.where(mask, indices, sentinel))
            count = int(mx.sum(mask.astype(mx.int32)).item())
            return selected[:count]

        def take_axis(result, part, axis):
            if isinstance(part, mx.array):
                if part.dtype == mx.bool_ or "bool" in str(part.dtype):
                    if part.ndim != 1:
                        raise NotImplementedError(
                            "MLX boolean tuple indexing currently supports 1-D masks")
                    return mx.take(result, bool_indices(part), axis=axis)
                return mx.take(result, part, axis=axis)
            if isinstance(part, (list, tuple)):
                return mx.take(result, mx.array(part, dtype=mx.int32), axis=axis)
            axis_key = [slice(None)] * result.ndim
            axis_key[axis] = part
            return mx.array._mlx_array_orig_getitem(result, tuple(axis_key))

        def tuple_getitem(parts):
            result = self
            axis = 0
            for part in parts:
                if part is Ellipsis:
                    continue
                if part is None:
                    result = mx.expand_dims(result, axis)
                    axis += 1
                    continue
                result = take_axis(result, part, axis)
                if not isinstance(part, int):
                    axis += 1
            return result

        if isinstance(key, slice) and negative_slice(key):
            start, stop, step = key.indices(self.shape[0])
            if step < 0:
                indices = mx.arange(start, stop, step)
                return mx.take(self, indices, axis=0)
            return mx.array._mlx_array_orig_getitem(self, slice(start, stop, step))
        if isinstance(key, tuple) and _builtins.any(
                negative_slice(part) for part in key):
            try:
                result = self
                axis = 0
                for part in key:
                    if part is None:
                        result = mx.expand_dims(result, axis)
                        axis += 1
                        continue
                    if isinstance(part, slice) and negative_slice(part):
                        start, stop, step = part.indices(result.shape[axis])
                        if step < 0:
                            indices = mx.arange(start, stop, step).astype(mx.int32)
                            result = mx.take(result, indices, axis=axis)
                        else:
                            axis_key = [slice(None)] * result.ndim
                            axis_key[axis] = slice(start, stop, step)
                            result = mx.array._mlx_array_orig_getitem(
                                result, tuple(axis_key))
                        axis += 1
                    else:
                        axis_key = [slice(None)] * result.ndim
                        axis_key[axis] = part
                        result = mx.array._mlx_array_orig_getitem(
                            result, tuple(axis_key))
                        if not isinstance(part, int):
                            axis += 1
                return result
            except (NotImplementedError, ValueError, TypeError):
                raise
        if ((isinstance(key, tuple) and _builtins.any(
                negative_slice(part) for part in key))
                or negative_slice(key)):
            raise NotImplementedError(
                "MLX negative slicing fallback requires a native implementation")
        if isinstance(key, tuple) and _builtins.any(
                isinstance(part, (list, tuple)) for part in key):
            try:
                return tuple_getitem(key)
            except (NotImplementedError, ValueError, TypeError):
                raise
        if isinstance(key, tuple) and _builtins.any(is_bool_array(part) for part in key):
            return tuple_getitem(key)
        if isinstance(key, mx.array) and (key.dtype == mx.bool_ or
                                          str(key.dtype).endswith("bool_")):
            if key.ndim == 1 and self.ndim >= 1 and key.shape[0] == self.shape[0]:
                indices = mx.arange(key.shape[0], dtype=mx.int32)
                sentinel = mx.full(key.shape, key.shape[0], dtype=mx.int32)
                selected = mx.sort(mx.where(key, indices, sentinel))
                count = int(mx.sum(key.astype(mx.int32)).item())
                return mx.take(self, selected[:count], axis=0)
            flat_self = mx.reshape(self, (self.size,))
            flat_key = mx.reshape(key, (key.size,))
            indices = mx.arange(flat_key.shape[0], dtype=mx.int32)
            sentinel = mx.full(flat_key.shape, flat_key.shape[0], dtype=mx.int32)
            selected = mx.sort(mx.where(flat_key, indices, sentinel))
            count = int(mx.sum(flat_key.astype(mx.int32)).item())
            return mx.take(flat_self, selected[:count], axis=0)
        if isinstance(key, tuple) and _builtins.any(
                isinstance(part, mx.array) for part in key):
            try:
                return mx.array._mlx_array_orig_getitem(self, key)
            except (NotImplementedError, ValueError, TypeError):
                return tuple_getitem(key)
        if isinstance(key, mx.array):
            if key.size == 0 and key.dtype != mx.bool_ and "bool" not in str(key.dtype):
                return mx.array([], dtype=self.dtype).reshape(
                    (0,) + tuple(self.shape[1:]))
            return mx.take(self, key, axis=0)
        try:
            return mx.array._mlx_array_orig_getitem(self, key)
        except (NotImplementedError, ValueError):
            raise

    mx.array.__getitem__ = _array_getitem
if not hasattr(mx.array, "_mlx_array_orig_matmul"):
    mx.array._mlx_array_orig_matmul = mx.array.__matmul__

    def _array_matmul(self, other):
        if isinstance(other, (list, tuple)):
            other = mx.array([_to_scalar(item) for item in other],
                             dtype=self.dtype)
        return mx.array._mlx_array_orig_matmul(self, other)

    mx.array.__matmul__ = _array_matmul

def _float64_comparison_override(name: str, left: Any, right: Any):
    if (_mlx_overrides is None or not isinstance(left, mx.array)
            or left.dtype != mx.float64):
        return None
    if not isinstance(right, (int, float, mx.array)):
        return None
    try:
        return getattr(_mlx_overrides, name)(left, right)
    except Exception:
        return None


if not hasattr(mx.array, "_mlx_array_orig_lt"):
    mx.array._mlx_array_orig_lt = mx.array.__lt__

    def _array_lt(self, other):
        other = _coerce_float64_value(self, other)
        result = _float64_comparison_override("less_float64", self, other)
        if result is not None:
            return result
        return mx.array._mlx_array_orig_lt(self, other)

    mx.array.__lt__ = _array_lt
if not hasattr(mx.array, "_mlx_array_orig_gt"):
    mx.array._mlx_array_orig_gt = mx.array.__gt__

    def _array_gt(self, other):
        other = _coerce_float64_value(self, other)
        result = _float64_comparison_override("greater_float64", self, other)
        if result is not None:
            return result
        return mx.array._mlx_array_orig_gt(self, other)

    mx.array.__gt__ = _array_gt
if not hasattr(mx.array, "_mlx_array_orig_ge"):
    mx.array._mlx_array_orig_ge = mx.array.__ge__

    def _array_ge(self, other):
        other = _coerce_float64_value(self, other)
        override = _float64_comparison_override(
            "greater_equal_float64", self, other)
        if override is not None:
            return override
        result = mx.array._mlx_array_orig_ge(self, other)
        if isinstance(other, (int, float, mx.array)):
            try:
                result = mx.logical_or(result, mx.equal(self - other, 0))
            except Exception:
                pass
        return result

    mx.array.__ge__ = _array_ge
if not hasattr(mx.array, "_mlx_array_orig_le"):
    mx.array._mlx_array_orig_le = mx.array.__le__

    def _array_le(self, other):
        other = _coerce_float64_value(self, other)
        override = _float64_comparison_override("less_equal_float64", self, other)
        if override is not None:
            return override
        result = mx.array._mlx_array_orig_le(self, other)
        if isinstance(other, (int, float, mx.array)):
            try:
                result = mx.logical_or(result, mx.equal(self - other, 0))
            except Exception:
                pass
        return result

    mx.array.__le__ = _array_le
if not hasattr(mx.array, "_mlx_array_orig_sub"):
    mx.array._mlx_array_orig_sub = mx.array.__sub__

    def _array_sub(self, other):
        other = _coerce_float64_value(self, other)
        if isinstance(other, range):
            other = list(other)
        if "MaskedArray" in globals() and isinstance(other, MaskedArray):
            return other.__rsub__(self)
        if isinstance(other, (list, tuple)):
            other = mx.array([_to_scalar(item) for item in other],
                             dtype=self.dtype)
        elif isinstance(other, _ObjectArray):
            other = mx.array(other.tolist(), dtype=self.dtype).reshape(other.shape)
        return mx.array._mlx_array_orig_sub(self, other)

    mx.array.__sub__ = _array_sub
    mx.array.__isub__ = _array_sub
if not hasattr(mx.array, "_mlx_array_orig_add"):
    mx.array._mlx_array_orig_add = mx.array.__add__

    def _array_add(self, other):
        other = _coerce_float64_value(self, other)
        if isinstance(other, range):
            other = list(other)
        if "MaskedArray" in globals() and isinstance(other, MaskedArray):
            return other.__radd__(self)
        if isinstance(other, (list, tuple)):
            other = mx.array([_to_scalar(item) for item in other],
                             dtype=self.dtype)
        elif isinstance(other, _ObjectArray):
            other = mx.array(other.tolist(), dtype=self.dtype).reshape(other.shape)
        return mx.array._mlx_array_orig_add(self, other)

    mx.array.__add__ = _array_add
    mx.array.__iadd__ = _array_add
if not hasattr(mx.array, "_mlx_array_orig_radd"):
    mx.array._mlx_array_orig_radd = mx.array.__radd__

    def _array_radd(self, other):
        other = _coerce_float64_value(self, other)
        if isinstance(other, range):
            other = list(other)
        return mx.array._mlx_array_orig_radd(self, other)

    mx.array.__radd__ = _array_radd
if not hasattr(mx.array, "_mlx_array_orig_rsub"):
    mx.array._mlx_array_orig_rsub = mx.array.__rsub__

    def _array_rsub(self, other):
        other = _coerce_float64_value(self, other)
        if isinstance(other, range):
            other = list(other)
        if isinstance(other, (list, tuple)):
            other = mx.array([_to_scalar(item) for item in other],
                             dtype=self.dtype)
        elif isinstance(other, _ObjectArray):
            other = mx.array(other.tolist(), dtype=self.dtype).reshape(other.shape)
        elif (hasattr(other, "__array__")
              and not isinstance(other, (mx.array, _ObjectArray))):
            try:
                other = _to_mx(other.__array__())
            except TypeError:
                try:
                    other = _to_mx(other.__array__(dtype=float))
                except Exception:
                    pass
        return mx.array._mlx_array_orig_rsub(self, other)

    mx.array.__rsub__ = _array_rsub
if not hasattr(mx.array, "_mlx_array_orig_pow"):
    mx.array._mlx_array_orig_pow = mx.array.__pow__

    def _array_pow(self, other):
        other = _coerce_float64_value(self, other)
        return mx.array._mlx_array_orig_pow(self, other)

    mx.array.__pow__ = _array_pow
if not hasattr(mx.array, "flat"):
    def _array_flat(self):
        return flatiter(mx.reshape(self, (self.size,)))

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
if not hasattr(mx.array, "_mlx_array_orig_shape_property"):
    mx.array._mlx_array_orig_shape_property = mx.array.shape
    _array_shape_overrides = {}

    def _clear_array_shape_override(key):
        _array_shape_overrides.pop(key, None)

    def _array_shape_get(self):
        return _array_shape_overrides.get(
            id(self), mx.array._mlx_array_orig_shape_property.fget(self))

    def _array_shape_set(self, value):
        shape_tuple = _shape_tuple(value)
        if math.prod(shape_tuple) != self.size:
            raise ValueError(
                f"cannot reshape array of size {self.size} into shape {shape_tuple}")
        key = id(self)
        _array_shape_overrides[key] = shape_tuple
        weakref.finalize(self, _clear_array_shape_override, key)

    mx.array.shape = property(_array_shape_get, _array_shape_set)
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
    "F": mx.complex64,
    "c8": mx.complex64,
    "complex": mx.complex64,
    "complex64": mx.complex64,
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
    if dtype is _builtins.object:
        return _builtins.object
    if dtype is _builtins.str:
        return _builtins.object
    if isinstance(dtype, str):
        if dtype.startswith(("S", "U")) or dtype in {"O", "object", "str", "bytes"}:
            return _builtins.object
        return _STRING_TO_MX_DTYPE.get(dtype, dtype)
    name = getattr(dtype, "__name__", None)
    if name in _STRING_TO_MX_DTYPE:
        return _STRING_TO_MX_DTYPE[name]
    return dtype


def _dtype_from_array_like(value: Any) -> Any | None:
    dtype = getattr(value, "dtype", None)
    if dtype is None:
        return None
    kind = getattr(dtype, "kind", None)
    if kind in {"O", "S", "U"}:
        return _builtins.object
    name = getattr(dtype, "name", None)
    if name in _STRING_TO_MX_DTYPE:
        return _STRING_TO_MX_DTYPE[name]
    char = getattr(dtype, "char", None)
    if char in _STRING_TO_MX_DTYPE:
        return _STRING_TO_MX_DTYPE[char]
    return None


def _coerce_float64_fill_value(dtype: Any | None, value: Any,
                               stream: Any | None = None) -> Any:
    if _mlx_overrides is None or _unwrap_dtype(dtype) != mx.float64:
        return value
    return _mlx_overrides.float64_scalar(value, stream=stream)


# Public dtypes (MLXArrayBackend-like: usable as dtype= and callable constructors).
bool_ = DType(mx.bool_, "bool_", "b", 1, "?")
float16 = DType(mx.float16, "float16", "f", 2, "e")
float32 = DType(mx.float32, "float32", "f", 4, "f")
float64 = DType(mx.float64, "float64", "f", 8, "d")
complex64 = DType(mx.complex64, "complex64", "c", 8, "F")
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
complexfloating = complex
integer = int
number = (int, float, complex)
_object_dtype = DType(_builtins.object, "object", "O", 0, "O")
_DTYPE_BY_MX = {
    dt.mx_dtype: dt for dt in (
        bool_, float16, float32, float64, complex64, bfloat16,
        int8, int16, int32, int64, uint8, uint16, uint32, uint64,
    )
}
_DTYPE_BY_NAME = {
    **{dt.name: dt for dt in _DTYPE_BY_MX.values()},
    **{dt.char: dt for dt in _DTYPE_BY_MX.values()},
    "bool": bool_,
    "float": float64,
    "double": float64,
    "complex": complex64,
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
    "F": complex64, "c8": complex64,
})

if not hasattr(type(mx.float32), "kind"):
    type(mx.float32).kind = property(lambda self: _DTYPE_BY_MX.get(self, _object_dtype).kind)
if not hasattr(type(mx.float32), "char"):
    type(mx.float32).char = property(lambda self: _DTYPE_BY_MX.get(self, _object_dtype).char)
if not hasattr(type(mx.float32), "itemsize"):
    type(mx.float32).itemsize = property(lambda self: _DTYPE_BY_MX.get(self, _object_dtype).itemsize)
if not hasattr(type(mx.float32), "isnative"):
    type(mx.float32).isnative = property(lambda self: True)
if not hasattr(type(mx.float32), "newbyteorder"):
    type(mx.float32).newbyteorder = lambda self, order=None: self
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
    def resolution(self) -> float:
        return {
            mx.float16: 1e-3,
            mx.float32: 1e-6,
            mx.float64: 1e-15,
        }.get(self.dtype, self.eps)

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


class _ObjectArray:
    def __init__(self, data: Any, dtype: Any | None = None,
                 shape: Tuple[int, ...] | None = None):
        dtype = _unwrap_dtype(dtype)
        if dtype not in {None, _builtins.object} and dtype != _object_dtype:
            raise TypeError("_ObjectArray is only for Python object metadata")
        if isinstance(data, _ObjectArray):
            self._data = _copy_nested(data._data)
        else:
            self._data = _copy_nested(data)
        self.dtype = _object_dtype
        self.shape = tuple(shape) if shape is not None else _infer_shape(self._data)
        self.ndim = len(self.shape)
        self.size = math.prod(self.shape) if self.shape else 1

    def __iter__(self):
        values = self._data if isinstance(self._data, list) else [self._data]
        return (_ObjectArray(value, dtype=self.dtype)
                if isinstance(value, list) else value for value in values)

    def __len__(self):
        return self.shape[0] if self.shape else 1

    def copy(self, order: str = "C"):
        return _ObjectArray(self, dtype=self.dtype, shape=self.shape)

    def __getitem__(self, key: Any) -> Any:
        if isinstance(key, str):
            values = self._data if isinstance(self._data, list) else [self._data]
            if values and _builtins.all(isinstance(value, dict) for value in values):
                return _ObjectArray([value[key] for value in values])
        if isinstance(key, mx.array):
            key = key.tolist()
        if isinstance(key, _ObjectArray):
            key = _copy_nested(key._data)
        if isinstance(key, range):
            key = list(key)
        if isinstance(key, list) and _builtins.all(
                isinstance(v, bool) for v in _flatten(key)):
            values = list(_flatten(self._data))
            mask = list(_flatten(key))
            return _ObjectArray([v for v, keep in zip(values, mask) if keep],
                                dtype=self.dtype)
        if isinstance(key, list):
            values = self._data if isinstance(self._data, list) else [self._data]
            return _ObjectArray([values[int(v)] for v in key], dtype=self.dtype)
        if isinstance(key, tuple):
            value = _python_getitem(self._data, key)
            if (self.size == 0 and self.ndim == 1 and len(key) == 2
                    and isinstance(key[0], slice) and key[1] is None):
                return _ObjectArray(value, dtype=self.dtype,
                                    shape=(self.shape[0], 1))
            return _ObjectArray(value, dtype=self.dtype) if isinstance(value, list) else value
        if key is None:
            return _ObjectArray([_copy_nested(self._data)], dtype=self.dtype,
                                shape=(1,) + self.shape)
        value = self._data[key] if isinstance(self._data, list) else self._data
        return _ObjectArray(value, dtype=self.dtype) if isinstance(value, list) else value

    def __setitem__(self, key: Any, value: Any) -> None:
        if isinstance(key, mx.array):
            key = key.tolist()
        if isinstance(key, list):
            def set_mask(data, mask):
                if isinstance(mask, list):
                    if not isinstance(data, list):
                        return value if _builtins.any(_flatten(mask)) else data
                    for idx, keep in enumerate(mask):
                        if isinstance(keep, list):
                            data[idx] = set_mask(data[idx], keep)
                        elif keep:
                            data[idx] = value
                    return data
                return value if mask else data

            if _builtins.all(isinstance(item, bool) for item in _flatten(key)):
                self._data = set_mask(self._data, key)
                return
        if not isinstance(key, tuple):
            self._data[key] = value
            return
        target = self._data
        for part in key[:-1]:
            target = target[part]
        target[key[-1]] = value

    @property
    def flat(self):
        return flatiter(list(_flatten(self._data)))

    def item(self):
        if self.size != 1:
            raise ValueError("can only convert an array of size 1 to a Python scalar")
        return next(_flatten(self._data))

    def squeeze(self):
        return _ObjectArray(_squeeze_nested(self._data), dtype=self.dtype)

    def ravel(self):
        return _ObjectArray(list(_flatten(self._data)), dtype=self.dtype)

    def flatten(self):
        return self.ravel()

    @property
    def T(self):
        if self.ndim < 2:
            return _ObjectArray(self._data, dtype=self.dtype, shape=self.shape)
        if self.ndim == 2:
            flat = list(_flatten(self._data))
            rows, cols = self.shape[0], self.shape[1]
            transposed = [[flat[r * cols + c] for r in range(rows)]
                          for c in range(cols)]
            return _ObjectArray(transposed, dtype=self.dtype,
                                shape=(cols, rows))
        # For ndim > 2: delegate to numeric conversion and transpose
        try:
            return _ObjectArray(
                mx.transpose(self.astype(mx.float64)).tolist(),
                dtype=self.dtype)
        except Exception:
            # Fallback: reverse-shape view (correct shape, may have wrong data order)
            return _ObjectArray(
                _reshape_flat(list(_flatten(self._data)),
                              tuple(reversed(self.shape))),
                dtype=self.dtype)

    def reshape(self, *shape: Any):
        if len(shape) == 1:
            shape = shape[0]
        flat = list(_flatten(self._data))
        shape_tuple = _shape_tuple(shape)
        if -1 in shape_tuple:
            known = math.prod(v for v in shape_tuple if v != -1)
            inferred = len(flat) // known if known else 0
            shape_tuple = tuple(inferred if v == -1 else v for v in shape_tuple)
        return _ObjectArray(_reshape_flat(flat, shape_tuple),
                            dtype=self.dtype, shape=shape_tuple)

    def astype(self, dtype: Any, *args: Any, **kwargs: Any):
        mx_dtype = _unwrap_dtype(dtype)
        if mx_dtype is _builtins.object:
            return _ObjectArray(self, dtype=_object_dtype)
        return _to_mx(self._data, dtype=mx_dtype)

    def tolist(self):
        return _copy_nested(self._data)


class _ObjectNDArray(_ObjectArray):
    def __new__(cls, shape: Any):
        return _builtins.object.__new__(cls)

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
        if isinstance(key, slice) and len(self.shape) == 1:
            return self._data[key]
        return list(_flatten(self._data))[self._offset(key)]

    def __setitem__(self, key: Any, value: Any) -> None:
        if isinstance(key, slice) and len(self.shape) == 1:
            data = list(self._data)
            data[key] = list(value)
            self._data = data
            return
        flat = list(_flatten(self._data))
        flat[self._offset(key)] = value
        self._data = _reshape_flat(flat, self.shape)


class _StructuredNDArray:
    def __init__(self, shape: Any, dtype_spec: Sequence[Any]):
        self.shape = _shape_tuple(shape)
        self.ndim = len(self.shape)
        self.size = math.prod(self.shape) if self.shape else 1
        self.dtype = dtype_spec
        self._fields = []
        self._data = {}
        for field in dtype_spec:
            name, dtype_name = field[:2]
            subshape = _shape_tuple(field[2]) if len(field) > 2 else ()
            if not subshape and isinstance(dtype_name, str) and dtype_name[:1].isdigit():
                count_len = 0
                while count_len < len(dtype_name) and dtype_name[count_len].isdigit():
                    count_len += 1
                subshape = (int(dtype_name[:count_len]),)
                dtype_name = dtype_name[count_len:]
            self._fields.append((name, dtype_name, subshape))
            width = math.prod(subshape) if subshape else 1
            self._data[name] = [[0] * width for _ in range(self.size)]

    def __len__(self):
        return self.shape[0] if self.shape else 1

    def __getitem__(self, key: Any):
        if isinstance(key, str):
            values = self._data[key]
            field = next(field for field in self._fields if field[0] == key)
            subshape = field[2]
            if subshape:
                return _ObjectArray(
                    _reshape_flat(list(_flatten(values)), self.shape + subshape),
                    dtype=_object_dtype)
            return _ObjectArray([value[0] for value in values],
                                dtype=_object_dtype, shape=self.shape)
        raise TypeError("structured array indices must be field names")

    def __setitem__(self, key: Any, value: Any) -> None:
        if not isinstance(key, str):
            raise TypeError("structured array indices must be field names")
        field = next(field for field in self._fields if field[0] == key)
        subshape = field[2]
        width = math.prod(subshape) if subshape else 1
        if hasattr(value, "tolist"):
            value = value.tolist()
        if not isinstance(value, list):
            rows = [[value] * width for _ in range(self.size)]
        else:
            flat = list(_flatten(value))
            if len(flat) == width:
                rows = [flat[:] for _ in range(self.size)]
            else:
                rows = [flat[i * width:(i + 1) * width]
                        for i in range(self.size)]
        self._data[key] = rows

    @staticmethod
    def _pack_value(dtype_name: Any, value: Any) -> bytes:
        import struct
        if hasattr(value, "item"):
            value = value.item()
        if dtype_name in {"u1", "uint8", "|u1"}:
            return struct.pack("B", int(value) & 0xff)
        if dtype_name == ">u4":
            return struct.pack(">I", int(value) & 0xffffffff)
        if dtype_name in {"<u4", "u4", "uint32"}:
            return struct.pack("<I", int(value) & 0xffffffff)
        raise TypeError(f"unsupported structured dtype field {dtype_name!r}")

    def tobytes(self) -> bytes:
        chunks = []
        for row in range(self.size):
            for name, dtype_name, _subshape in self._fields:
                for value in self._data[name][row]:
                    chunks.append(self._pack_value(dtype_name, value))
        return b"".join(chunks)


def _copy_nested(value: Any) -> Any:
    if isinstance(value, _ObjectArray):
        return _copy_nested(value._data)
    if isinstance(value, mx.array):
        return value
    if isinstance(value, tuple):
        return [_copy_nested(v) for v in value]
    if isinstance(value, list):
        return [_copy_nested(v) for v in value]
    if isinstance(value, range):
        return list(value)
    return value


def _coerce_nested(value: Any, func) -> Any:
    if isinstance(value, _ObjectArray):
        return _coerce_nested(value._data, func)
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
    first_shape = _infer_shape(value[0])
    if first_shape and _builtins.all(
            isinstance(item, list) and _infer_shape(item) == first_shape
            for item in value):
        return (len(value),) + first_shape
    if first_shape:
        return (len(value),)
    return (len(value),)


def _reshape_flat(flat: list[Any], shape: Tuple[int, ...]) -> Any:
    if not shape:
        return flat[0]
    step = math.prod(shape[1:]) if len(shape) > 1 else 1
    return [_reshape_flat(flat[i * step:(i + 1) * step], shape[1:])
            for i in range(shape[0])]


def _filled_object_array(shape: Any, value: Any, dtype: Any | None = None) -> "_ObjectArray":
    dtype = _unwrap_dtype(dtype)
    if dtype not in {None, _builtins.object} and dtype != _object_dtype:
        raise TypeError("_filled_object_array is only for Python object metadata")
    shape_tuple = _shape_tuple(shape)
    size = math.prod(shape_tuple) if shape_tuple else 1
    data = _reshape_flat([value] * size, shape_tuple) if size else []
    return _ObjectArray(data, dtype=_object_dtype, shape=shape_tuple)


def _squeeze_nested(value: Any) -> Any:
    if isinstance(value, list) and len(value) == 1:
        return _squeeze_nested(value[0])
    if isinstance(value, list):
        return [_squeeze_nested(v) for v in value]
    return value


def _contains_object_data(value: Any) -> bool:
    if isinstance(value, _ObjectArray):
        return True
    if isinstance(value, (str, bytes, datetime, timedelta)) or value is None:
        return True
    if isinstance(value, (list, tuple)):
        return _builtins.any(_contains_object_data(v) for v in value)
    return False


def _contains_decimal_data(value: Any) -> bool:
    if isinstance(value, Decimal):
        return True
    if isinstance(value, _ObjectArray):
        value = value._data
    if isinstance(value, (list, tuple)):
        return _builtins.any(_contains_decimal_data(v) for v in value)
    return False


def _contains_temporal_data(value: Any) -> bool:
    if isinstance(value, (datetime, timedelta)):
        return True
    if isinstance(value, _ObjectArray):
        value = value._data
    if isinstance(value, (list, tuple)):
        return _builtins.any(_contains_temporal_data(v) for v in value)
    return False


def _replace_masked_none(data: Any, mask: Any) -> Any:
    if isinstance(mask, mx.array):
        mask = mask.tolist()
    if isinstance(data, _ObjectArray):
        data = data.tolist()
    if isinstance(mask, (list, tuple)) and isinstance(data, (list, tuple)):
        return [_replace_masked_none(d, m) for d, m in zip(data, mask)]
    if mask and data is None:
        return 0
    return data


def _contains_float_data(value: Any) -> bool:
    if isinstance(value, _ObjectArray):
        value = value._data
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
        if isinstance(key, memoryview):
            key = key.tolist()
        if isinstance(key, range):
            key = list(key)
        if isinstance(key, (list, tuple)):
            if key and isinstance(key[0], (list, tuple)):
                return [_python_getitem(data, item) for item in key]
            if _builtins.all(isinstance(item, bool) for item in key):
                return [value for value, keep in zip(data, key) if keep]
            return [data[int(item)] for item in key]
        return data[key]
    if not key:
        return data
    first, *rest = key
    if isinstance(first, mx.array):
        first = first.tolist()
    if isinstance(first, memoryview):
        first = first.tolist()
    if isinstance(first, range):
        first = list(first)
    if first is None:
        return [_python_getitem(data, tuple(rest))]
    if (isinstance(first, tuple)
            and _builtins.all(isinstance(item, int) for item in first)):
        first = list(first)
    if isinstance(first, list):
        if first and isinstance(first[0], (list, tuple)):
            selected = [_python_getitem(data, item) for item in first]
        elif _builtins.all(isinstance(item, bool) for item in first):
            selected = [item for item, keep in zip(data, first) if keep]
        else:
            selected = [data[int(item)] for item in first]
    else:
        selected = data[first]
    if rest and isinstance(first, (slice, list)):
        return [_python_getitem(item, tuple(rest)) for item in selected]
    if rest:
        return _python_getitem(selected, tuple(rest))
    return selected


def _copy_mx_array(arr: mx.array, stream: Any | None = None) -> mx.array:
    return mx.contiguous(arr, stream=stream)


def _exact_float64_array_from_python(value: Any,
                                     stream: Any | None = None) -> mx.array | None:
    if _mlx_overrides is None:
        return None
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return _mlx_overrides.float64_scalar(float(value), stream=stream)
    if isinstance(value, (list, tuple)):
        if not value:
            return mx.array([], dtype=mx.float64)
        children = [
            _exact_float64_array_from_python(item, stream=stream)
            for item in value
        ]
        if _builtins.any(child is None for child in children):
            return None
        if stream is not None:
            return mx.stack(children, axis=0, stream=stream)
        return mx.stack(children, axis=0)
    return None


def _to_mx(x: Any, dtype: Any | None = None,
           stream: Any | None = None) -> mx.array:
    _patch_pandas_bool_indexer()
    if isinstance(x, flatiter):
        x = x.tolist()
    if isinstance(x, memoryview):
        if dtype is None and x.format in {"B", "b"}:
            dtype = mx.uint8
        x = x.tolist()
    if isinstance(x, range):
        x = list(x)
    if isinstance(x, MaskedArray):
        x = x.data
    if isinstance(x, _ObjectArray):
        if _unwrap_dtype(dtype) is _builtins.object:
            return x
        return _to_mx(x._data, dtype=dtype or x.dtype, stream=stream)
    if (hasattr(x, "getdata") and hasattr(x, "getbands")
            and hasattr(x, "size")):
        if dtype is None:
            dtype = mx.uint8
        x = _image_to_nested(x)
    elif hasattr(x, "to_list"):
        try:
            x = x.to_list()
        except Exception:
            pass
    elif hasattr(x, "__array__") and not isinstance(x, mx.array):
        try:
            x = x.__array__()
        except TypeError:
            x = x.__array__(dtype=dtype)
        if dtype is None:
            dtype = _dtype_from_array_like(x)
    if isinstance(x, mx.array):
        dtype = _unwrap_dtype(dtype)
        if dtype is None or x.dtype == dtype:
            return x
        return x.astype(dtype, stream=stream)
    if (isinstance(dtype, (list, tuple))
            and _builtins.all(isinstance(field, (list, tuple)) and field
                              for field in dtype)):
        names = [field[0] for field in dtype]
        rows = [dict(zip(names, row)) for row in x]
        return _ObjectArray(rows, dtype=_object_dtype)
    string_dtype = dtype is _builtins.str or dtype == "str"
    dtype = _unwrap_dtype(dtype)
    if isinstance(x, (list, tuple)):
        x = tuple(
            item.data if isinstance(item, MaskedArray)
            else _copy_nested(item._data) if isinstance(item, _ObjectArray)
            else item
            for item in x)
        if x and _builtins.any(isinstance(item, mx.array) for item in x):
            first_array = next(item for item in x if isinstance(item, mx.array))
            row_dtype = dtype if dtype is not None else first_array.dtype
            arrays = [
                item.astype(row_dtype, stream=stream)
                if isinstance(item, mx.array) and item.dtype != row_dtype
                else item if isinstance(item, mx.array)
                else _to_mx(item, dtype=row_dtype, stream=stream)
                for item in x]
            if stream is not None:
                return mx.stack(arrays, axis=0, stream=stream)
            return mx.stack(arrays, axis=0)
        x = _copy_nested(x)
    if string_dtype:
        return _ObjectArray(_coerce_nested(x, str), dtype=_object_dtype)
    if _contains_decimal_data(x):
        if dtype in {mx.float16, mx.float32, mx.float64, mx.bfloat16}:
            x = _coerce_nested(x, float)
        elif dtype in {mx.int8, mx.int16, mx.int32, mx.int64,
                       mx.uint8, mx.uint16, mx.uint32, mx.uint64}:
            x = _coerce_nested(x, int)
    if (_contains_object_data(x) and not _contains_temporal_data(x)
            and dtype in {mx.float16, mx.float32, mx.float64, mx.bfloat16,
                          mx.complex64, mx.int8, mx.int16, mx.int32, mx.int64,
                          mx.uint8, mx.uint16, mx.uint32, mx.uint64}):
        converter = complex if dtype == mx.complex64 else (
            float if dtype in {mx.float16, mx.float32, mx.float64, mx.bfloat16}
            else int)
        x = _coerce_nested(x, converter)
    if dtype is _builtins.object or _contains_object_data(x):
        return _ObjectArray(x, dtype=_object_dtype)
    if dtype is None:
        try:
            if _contains_float_data(x):
                arr = _exact_float64_array_from_python(x, stream=stream)
                if arr is None:
                    arr = mx.array(x, dtype=mx.float64)
                return _copy_mx_array(arr, stream=stream) if stream is not None else arr
            arr = mx.array(x)
            return _copy_mx_array(arr, stream=stream) if stream is not None else arr
        except (TypeError, ValueError):
            return _ObjectArray(x, dtype=_object_dtype)
    if dtype == mx.float64:
        arr = _exact_float64_array_from_python(x, stream=stream)
        if arr is not None:
            return arr
    arr = mx.array(x, dtype=dtype)
    return _copy_mx_array(arr, stream=stream) if stream is not None else arr


def _to_scalar(x: Any) -> Any:
    if isinstance(x, mx.array) and x.size == 1:
        return x.item()
    return x


def array(obj: Any, dtype: Any | None = None, copy: bool | None = True,
          order: Any | None = None, subok: bool = False, ndmin: int = 0,
          like: Any | None = None, stream: Any | None = None) -> mx.array:
    if isinstance(obj, MaskedArray) and subok:
        data = _copy_mx_array(obj.data, stream=stream) if copy else obj.data
        mask = (_copy_mx_array(obj.mask, stream=stream)
                if copy and obj.mask is not None else obj.mask)
        arr = MaskedArray(data, mask)
        if dtype is not None:
            arr = arr.astype(dtype)
        while getattr(arr, "ndim", 0) < ndmin:
            arr = arr.reshape((1,) + tuple(arr.shape))
        return arr
    arr = _to_mx(obj, dtype=dtype, stream=stream)
    if copy and isinstance(arr, mx.array):
        arr = _copy_mx_array(arr, stream=stream)
    elif copy and isinstance(arr, _ObjectArray):
        arr = _ObjectArray(arr, dtype=arr.dtype, shape=arr.shape)
    while getattr(arr, "ndim", 0) < ndmin:
        arr = arr.reshape((1,) + tuple(arr.shape))
    return arr


def asarray(obj: Any, dtype: Any | None = None, order: Any | None = None,
            copy: bool | None = None, like: Any | None = None,
            stream: Any | None = None) -> mx.array:
    arr = _to_mx(obj, dtype=dtype, stream=stream)
    if copy and isinstance(arr, mx.array):
        return _copy_mx_array(arr, stream=stream)
    return arr


def ascontiguousarray(obj: Any, dtype: Any | None = None,
                      stream: Any | None = None) -> mx.array:
    return mx.contiguous(_to_mx(obj, dtype=dtype, stream=stream),
                         stream=stream)


def asanyarray(obj: Any, dtype: Any | None = None, order: Any | None = None,
               like: Any | None = None, stream: Any | None = None) -> mx.array:
    if isinstance(obj, MaskedArray):
        return obj.astype(dtype) if dtype is not None else obj
    return _to_mx(obj, dtype=dtype, stream=stream)


def atleast_1d(*arys: Any) -> Tuple[mx.array, ...] | mx.array:
    res = []
    for a in arys:
        arr = a if isinstance(a, MaskedArray) else _to_mx(a)
        if arr.ndim == 0:
            arr = arr.reshape((1,)) if isinstance(arr, (_ObjectArray, MaskedArray)) else mx.reshape(arr, (1,))
        res.append(arr)
    return tuple(res) if len(res) > 1 else res[0]


def atleast_2d(*arys: Any) -> Tuple[mx.array, ...] | mx.array:
    res = []
    for a in arys:
        arr = _to_mx(a)
        if arr.ndim == 0:
            arr = arr.reshape((1, 1)) if isinstance(arr, _ObjectArray) else mx.reshape(arr, (1, 1))
        elif arr.ndim == 1:
            shape = (1, arr.shape[0])
            arr = arr.reshape(shape) if isinstance(arr, _ObjectArray) else mx.reshape(arr, shape)
        res.append(arr)
    return tuple(res) if len(res) > 1 else res[0]


def atleast_3d(*arys: Any) -> Tuple[mx.array, ...] | mx.array:
    res = []
    for a in arys:
        arr = _to_mx(a)
        if arr.ndim == 0:
            arr = arr.reshape((1, 1, 1)) if isinstance(arr, _ObjectArray) else mx.reshape(arr, (1, 1, 1))
        elif arr.ndim == 1:
            shape = (1, arr.shape[0], 1)
            arr = arr.reshape(shape) if isinstance(arr, _ObjectArray) else mx.reshape(arr, shape)
        elif arr.ndim == 2:
            shape = (1, *arr.shape)
            arr = arr.reshape(shape) if isinstance(arr, _ObjectArray) else mx.reshape(arr, shape)
        res.append(arr)
    return tuple(res) if len(res) > 1 else res[0]


def zeros(shape: Any, dtype: Any | None = None,
          stream: Any | None = None) -> mx.array:
    dtype = _unwrap_dtype(dtype)
    if dtype is _builtins.object:
        return _filled_object_array(shape, 0, dtype or _object_dtype)
    return mx.zeros(shape, dtype=_unwrap_dtype(dtype), stream=stream)


def ones(shape: Any, dtype: Any | None = None,
         stream: Any | None = None) -> mx.array:
    dtype = _unwrap_dtype(dtype)
    if dtype is _builtins.object:
        return _filled_object_array(shape, 1, dtype or _object_dtype)
    return mx.ones(shape, dtype=dtype, stream=stream)


def full(shape: Any, fill_value: Any, dtype: Any | None = None,
         stream: Any | None = None) -> mx.array:
    if _unwrap_dtype(dtype) is _builtins.object or _contains_object_data(fill_value):
        return _filled_object_array(shape, fill_value, _object_dtype)
    dtype = _unwrap_dtype(dtype)
    fill_value = _coerce_float64_fill_value(dtype, fill_value, stream=stream)
    return mx.full(shape, fill_value, dtype=dtype, stream=stream)


def empty(shape: Any, dtype: Any | None = None,
          stream: Any | None = None) -> mx.array:
    if dtype is _builtins.object or dtype == "object":
        return _ObjectNDArray(shape)
    unwrapped = _unwrap_dtype(dtype)
    if isinstance(dtype, (list, tuple)) and _builtins.all(
            isinstance(f, tuple) for f in dtype):
        return _StructuredNDArray(shape, dtype)
    # MLX does not expose uninitialized arrays; use zeros as a safe fallback.
    return mx.zeros(shape, dtype=unwrapped, stream=stream)


def zeros_like(a: Any, dtype: Any | None = None,
               stream: Any | None = None) -> mx.array:
    arr = _to_mx(a)
    dtype = _unwrap_dtype(dtype)
    if isinstance(arr, _ObjectArray):
        return zeros(arr.shape, dtype=dtype or arr.dtype)
    if dtype is None or dtype == arr.dtype:
        return mx.zeros_like(arr, stream=stream)
    return mx.zeros(arr.shape, dtype=dtype, stream=stream)


def ones_like(a: Any, dtype: Any | None = None,
              stream: Any | None = None) -> mx.array:
    arr = _to_mx(a)
    dtype = _unwrap_dtype(dtype)
    if isinstance(arr, _ObjectArray):
        return ones(arr.shape, dtype=dtype or arr.dtype)
    if dtype is None or dtype == arr.dtype:
        return mx.ones_like(arr, stream=stream)
    return mx.ones(arr.shape, dtype=dtype, stream=stream)


def full_like(a: Any, fill_value: Any, dtype: Any | None = None,
              stream: Any | None = None) -> mx.array:
    arr = _to_mx(a)
    if isinstance(arr, _ObjectArray) or _contains_object_data(fill_value):
        return full(arr.shape, fill_value, dtype=dtype or _object_dtype)
    dtype = _unwrap_dtype(dtype) or arr.dtype
    fill_value = _coerce_float64_fill_value(dtype, fill_value, stream=stream)
    return mx.full(arr.shape, fill_value, dtype=dtype, stream=stream)


def empty_like(a: Any, dtype: Any | None = None,
               stream: Any | None = None) -> mx.array:
    arr = _to_mx(a)
    dtype = _unwrap_dtype(dtype)
    if isinstance(arr, _ObjectArray):
        return zeros(arr.shape, dtype=dtype or arr.dtype)
    if dtype is None or dtype == arr.dtype:
        return mx.zeros_like(arr, stream=stream)
    return mx.zeros(arr.shape, dtype=dtype, stream=stream)


def arange(*args: Any, **kwargs: Any) -> mx.array:
    args = tuple(_to_scalar(arg) if isinstance(arg, mx.array) else arg for arg in args)
    dtype_arg = kwargs.get("dtype")
    dtype_text = str(dtype_arg) if dtype_arg is not None else ""
    is_datetime_range = (
        dtype_text.startswith("datetime64")
        or (len(args) >= 2
            and isinstance(datetime64(args[0]), datetime)
            and isinstance(datetime64(args[1]), datetime)))
    if is_datetime_range:
        unit = "D"
        if "[" in dtype_text and dtype_text.endswith("]"):
            unit = dtype_text[dtype_text.index("[") + 1:-1]
        start = datetime64(args[0])
        stop = datetime64(args[1])
        step_value = args[2] if len(args) > 2 else 1
        step = step_value if isinstance(step_value, timedelta) else timedelta64(
            step_value, unit)
        values = []
        current = start
        while current < stop:
            values.append(current)
            current = current + step
        return _ObjectArray(values, dtype=_object_dtype)
    if "dtype" in kwargs:
        kwargs["dtype"] = _unwrap_dtype(kwargs["dtype"])
    elif _builtins.any(isinstance(arg, float) for arg in args if arg is not None):
        kwargs["dtype"] = mx.float64
    if (_mlx_overrides is not None and kwargs.get("dtype") == mx.float64
            and not _in_forked_child()):
        if len(args) == 1:
            start, stop, step = 0.0, args[0], 1.0
        elif len(args) == 2:
            start, stop = args
            step = 1.0
        else:
            start, stop, step = args[:3]
        return _mlx_overrides.arange_float64(
            float(start), float(stop), float(step), stream=kwargs.get("stream"))
    if _in_forked_child():
        if len(args) == 1:
            start, stop, step = 0, args[0], 1
        elif len(args) == 2:
            start, stop = args
            step = 1
        else:
            start, stop, step = args[:3]
        values = []
        current = start
        if step == 0:
            raise ValueError("arange: step must not be zero")
        while (current < stop) if step > 0 else (current > stop):
            values.append(current)
            current += step
        return _ObjectArray(values, dtype=kwargs.get("dtype"))
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
    if _in_forked_child():
        values = [start + i * step for i in range(num)]
        if endpoint and values:
            values[-1] = stop
        result = _ObjectArray(values, dtype=kwargs["dtype"])
        return (result, step) if retstep else result
    result = mx.linspace(start, effective_stop, num, **kwargs)
    return (result, step) if retstep else result


def logspace(start: float, stop: float, num: int = 50, base: float = 10.0) -> mx.array:
    return mx.power(base, mx.linspace(start, stop, num))


def geomspace(start: float, stop: float, num: int = 50) -> mx.array:
    return mx.exp(mx.linspace(math.log(start), math.log(stop), num))


def reshape(a: Any, newshape: Any) -> mx.array:
    arr = _to_mx(a)
    if isinstance(arr, _ObjectArray):
        return arr.reshape(_shape_tuple(newshape))
    return mx.reshape(arr, _shape_tuple(newshape))


def resize(a: Any, new_shape: Any) -> mx.array:
    arr = _to_mx(a)
    shape_tuple = _shape_tuple(new_shape)
    total = math.prod(shape_tuple) if shape_tuple else 1
    if total == 0:
        return mx.array([], dtype=getattr(arr, "dtype", mx.float64)).reshape(shape_tuple)
    flat = list(_flatten(arr.tolist()))
    if not flat:
        values = []
    else:
        values = [flat[idx % len(flat)] for idx in _builtins.range(total)]
    data = _reshape_flat(values, shape_tuple)
    if isinstance(arr, _ObjectArray):
        return _ObjectArray(data, dtype=arr.dtype, shape=shape_tuple)
    return mx.array(data, dtype=arr.dtype)


def ravel(a: Any) -> mx.array:
    if isinstance(a, MaskedArray):
        return a.ravel()
    arr = _to_mx(a)
    if isinstance(arr, _ObjectArray):
        return arr.ravel()
    return mx.reshape(arr, (arr.size,))


def squeeze(a: Any, axis: Any | None = None) -> mx.array:
    arr = _to_mx(a)
    if isinstance(arr, _ObjectArray):
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
    converted = [_to_mx(a) for a in arrays]
    if _builtins.any(isinstance(a, _ObjectArray) for a in converted):
        lists = [a.tolist() if hasattr(a, "tolist") else a for a in converted]
        shape_tuple = _infer_shape(lists[0])
        ndim = len(shape_tuple)
        if axis < 0:
            axis += ndim + 1

        def stack_nested(items, depth):
            if depth == axis:
                return [_copy_nested(item) for item in items]
            return [stack_nested([item[idx] for item in items], depth + 1)
                    for idx in _builtins.range(len(items[0]))]

        return _ObjectArray(stack_nested(lists, 0), dtype=_object_dtype)
    return mx.stack(converted, axis=axis)


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
            arr = arr.reshape((arr.shape[0], 1)) if isinstance(arr, _ObjectArray) else mx.reshape(arr, (arr.shape[0], 1))
        elif arr.ndim != 2:
            raise ValueError("column_stack expects 1D or 2D arrays")
        arrays.append(arr)
    if arrays and _builtins.all(isinstance(arr, _ObjectArray) for arr in arrays):
        if _builtins.any(arr.shape[0] == 0 for arr in arrays):
            return _ObjectArray([], dtype=arrays[0].dtype,
                                shape=(0, len(arrays)))
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
    arr = _to_mx(a)
    reps_tuple = _shape_tuple(reps)
    if isinstance(arr, _ObjectArray):
        data = arr.tolist()
        if len(reps_tuple) == 1:
            repeat_count = reps_tuple[0]
            if isinstance(data, list):
                return _ObjectArray(data * repeat_count, dtype=arr.dtype)
            return _ObjectArray([data] * repeat_count, dtype=arr.dtype)
        raise NotImplementedError("object tiling supports one-dimensional reps")
    return mx.tile(arr, reps_tuple)


def repeat(a: Any, repeats: Any, axis: int | None = None) -> mx.array:
    arr = _to_mx(a)
    if isinstance(repeats, mx.array):
        repeats = repeats.tolist()
    if isinstance(arr, _ObjectArray) or isinstance(repeats, (list, tuple)):
        if axis is None:
            values = list(_flatten(arr.tolist()))
            if isinstance(repeats, (list, tuple)):
                counts = [int(count) for count in repeats]
            else:
                counts = [int(repeats)] * len(values)
            out = []
            for value, count in zip(values, counts):
                out.extend(_copy_nested(value) for _ in range(count))
            if isinstance(arr, _ObjectArray):
                return _ObjectArray(out, dtype=arr.dtype)
            return mx.array(out, dtype=arr.dtype)
        axis = int(axis)
        if axis < 0:
            axis += arr.ndim

        def repeat_axis(data, depth):
            if depth == axis:
                items = data if isinstance(data, list) else [data]
                counts = (repeats if isinstance(repeats, (list, tuple))
                          else [repeats] * len(items))
                out = []
                for item, count in zip(items, counts):
                    count = int(count)
                    out.extend(_copy_nested(item) for _ in range(count))
                return out
            return [repeat_axis(item, depth + 1) for item in data]

        out = repeat_axis(arr.tolist(), 0)
        if isinstance(arr, _ObjectArray):
            return _ObjectArray(out, dtype=arr.dtype)
        return mx.array(out, dtype=arr.dtype)
    return mx.repeat(arr, repeats, axis=axis)


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
    if isinstance(arr_mx, _ObjectArray):
        return _ObjectArray(values, dtype=arr_mx.dtype)
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


def clip(a: Any, a_min: Any, a_max: Any, out: Any | None = None) -> mx.array:
    result = mx.clip(_to_mx(a), a_min, a_max)
    if out is not None:
        full_key = (slice(None),) * out.ndim if getattr(out, "ndim", 0) else slice(None)
        out[full_key] = result
        return out
    return result


def diff(a: Any, n: int = 1, axis: int = -1) -> mx.array:
    arr = _to_mx(a)
    for _ in range(n):
        norm_axis = axis + arr.ndim if axis < 0 else axis
        stop = arr.shape[norm_axis] - 1
        lead = tuple(
            slice(1, None) if dim == norm_axis else slice(None)
            for dim in range(arr.ndim))
        trail = tuple(
            slice(0, stop) if dim == norm_axis else slice(None)
            for dim in range(arr.ndim))
        arr = arr[lead] - arr[trail]
    return arr


def unique(a: Any) -> mx.array:
    values = sorted(set(_flatten(_to_mx(a).tolist())))
    if values and _contains_object_data(values):
        return _ObjectArray(values, dtype=_object_dtype)
    return mx.array(values)


def intersect1d(ar1: Any, ar2: Any, assume_unique: bool = False,
                return_indices: bool = False):
    left = list(_flatten(_to_mx(ar1).tolist()))
    right = list(_flatten(_to_mx(ar2).tolist()))
    values = sorted(set(left).intersection(right))
    out = (_ObjectArray(values, dtype=_object_dtype)
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
    if isinstance(arr, _ObjectArray):
        return arr.min(axis=axis, initial=initial)
    if arr.size == 0 and initial is not None:
        return initial
    result = mx.min(arr, axis=axis)
    if initial is not None:
        result = mx.minimum(result, mx.array(initial, dtype=arr.dtype))
    return _to_scalar(result)


def max(a: Any, axis: int | None = None, initial: Any | None = None) -> Any:
    arr = _to_mx(a)
    if isinstance(arr, _ObjectArray):
        return arr.max(axis=axis, initial=initial)
    if arr.size == 0 and initial is not None:
        return initial
    result = mx.max(arr, axis=axis)
    if initial is not None:
        result = mx.maximum(result, mx.array(initial, dtype=arr.dtype))
    return _to_scalar(result)


def prod(a: Any, axis: int | None = None) -> Any:
    return mx.prod(_to_mx(a), axis=axis)


def roots(p: Any) -> mx.array:
    arr = _to_mx(p)
    if isinstance(arr, mx.array):
        coeffs = [float(v) for v in mx.reshape(arr, (arr.size,))]
    else:
        coeffs = [float(v) for v in _flatten(arr._data)]
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


def cumsum(a: Any, axis: int | None = None,
           dtype: Any | None = None) -> mx.array:
    arr = _to_mx(a)
    if dtype is not None:
        arr = arr.astype(dtype)
    if isinstance(arr, _ObjectArray):
        if axis is not None:
            raise NotImplementedError("PythonArray cumsum currently supports axis=None")
        total = 0
        values = []
        for value in _flatten(arr._data):
            total += value
            values.append(total)
        return _ObjectArray(values, dtype=arr.dtype)
    return mx.cumsum(arr, axis=axis)


def cumprod(a: Any, axis: int | None = None,
            dtype: Any | None = None) -> mx.array:
    arr = _to_mx(a)
    if dtype is not None:
        arr = arr.astype(dtype)
    if isinstance(arr, _ObjectArray):
        if axis is not None:
            raise NotImplementedError("PythonArray cumprod currently supports axis=None")
        total = 1
        values = []
        for value in _flatten(arr._data):
            total *= value
            values.append(total)
        return _ObjectArray(values, dtype=arr.dtype)
    return mx.cumprod(arr, axis=axis)


def ptp(a: Any, axis: int | None = None) -> Any:
    arr = _to_mx(a)
    return _to_scalar(mx.max(arr, axis=axis) - mx.min(arr, axis=axis))


def all(a: Any, axis: int | None = None) -> Any:
    arr = _to_mx(a)
    if isinstance(arr, _ObjectArray):
        if axis is None:
            return _builtins.all(bool(value) for value in _flatten(arr.tolist()))
    return _to_scalar(mx.all(arr, axis=axis))


def any(a: Any, axis: int | None = None) -> Any:
    arr = _to_mx(a)
    if isinstance(arr, _ObjectArray):
        if axis is None:
            return _builtins.any(bool(value) for value in _flatten(arr.tolist()))
    return _to_scalar(mx.any(arr, axis=axis))


def _python_isfinite(value: Any) -> Any:
    if isinstance(value, list):
        return [_python_isfinite(item) for item in value]
    if value is None:
        return False
    try:
        return math.isfinite(value)
    except (TypeError, ValueError):
        return True


def _to_bool_result(value: Any) -> mx.array:
    if _in_forked_child():
        return _ObjectArray(value, dtype=bool_.mx_dtype)
    try:
        return mx.array(value, dtype=mx.bool_)
    except ValueError:
        return _ObjectArray(value, dtype=bool_.mx_dtype)


def isfinite(a: Any) -> mx.array:
    arr = _to_mx(a)
    if isinstance(arr, _ObjectArray):
        result = _python_isfinite(arr.tolist())
        return _to_bool_result(result)
    return mx.isfinite(arr)


def isinf(a: Any) -> mx.array:
    arr = _to_mx(a)
    if isinstance(arr, _ObjectArray):
        def isinf_one(value):
            try:
                return math.isinf(_to_scalar(value))
            except TypeError:
                return False
        return _to_bool_result(_coerce_nested(arr.tolist(), isinf_one))
    return mx.isinf(arr)


def isnan(a: Any) -> mx.array:
    arr = _to_mx(a)
    if isinstance(arr, _ObjectArray):
        def isnan_one(value):
            try:
                return math.isnan(_to_scalar(value))
            except TypeError:
                return False
        return _to_bool_result(_coerce_nested(arr.tolist(), isnan_one))
    return mx.isnan(arr)


def isclose(a: Any, b: Any, rtol: float = 1e-5, atol: float = 1e-8) -> mx.array:
    a_mx = _to_mx(a)
    b_mx = _to_mx(b)
    if isinstance(a_mx, _ObjectArray):
        a_mx = a_mx.astype(mx.float64)
    if isinstance(b_mx, _ObjectArray):
        b_mx = b_mx.astype(mx.float64)
    return mx.isclose(a_mx, b_mx, rtol=rtol, atol=atol)


def allclose(a: Any, b: Any, rtol: float = 1e-5, atol: float = 1e-8) -> bool:
    return bool(mx.all(isclose(a, b, rtol=rtol, atol=atol)).item())


def array_equal(a: Any, b: Any) -> bool:
    if isinstance(a, MaskedArray) or isinstance(b, MaskedArray):
        a_mask = ma.getmaskarray(a)
        b_mask = ma.getmaskarray(b)
        if not array_equal(a_mask, b_mask):
            return False
        mask_values = list(_flatten(a_mask.tolist()))
        a_data = ma.getdata(a)
        b_data = ma.getdata(b)
        if not _builtins.any(mask_values):
            return array_equal(a_data, b_data)
        keep = logical_not(a_mask)
        return array_equal(_to_mx(a_data)[keep], _to_mx(b_data)[keep])
    a_mx = _to_mx(a)
    b_mx = _to_mx(b)
    if tuple(getattr(a_mx, "shape", ())) != tuple(getattr(b_mx, "shape", ())):
        return False
    if isinstance(a_mx, _ObjectArray) or isinstance(b_mx, _ObjectArray):
        return _testing_equal(a, b)
    return bool(mx.all(mx.equal(a_mx, b_mx)).item())


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
    arr = _to_mx(a)
    if isinstance(arr, _ObjectArray):
        return _ObjectArray(
            _coerce_nested(arr.tolist(), lambda value: not bool(value)),
            dtype=bool_.mx_dtype)
    return mx.logical_not(arr)


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
    arr = _to_mx(a)
    if isinstance(arr, _ObjectArray) or _dtype_kind(arr.dtype) in {"i", "u"}:
        arr = arr.astype(mx.float64)
    return mx.sqrt(arr)


def exp(a: Any) -> mx.array:
    return mx.exp(_to_mx(a))


_FLOAT64_UNARY_INPUT_DTYPES = {
    mx.bool_, mx.int8, mx.int16, mx.int32, mx.int64,
    mx.uint8, mx.uint16, mx.uint32, mx.uint64, mx.float64,
}


def _call_mx_unary(op, arr: Any, stream: Any | None = None) -> mx.array:
    return op(arr) if stream is None else op(arr, stream=stream)


def _call_float64_unary(name: str, op, arr: Any,
                        stream: Any | None = None) -> mx.array:
    if (_mlx_overrides is not None and isinstance(arr, mx.array)
            and arr.dtype in _FLOAT64_UNARY_INPUT_DTYPES):
        return getattr(_mlx_overrides, f"{name}_float64")(arr, stream=stream)
    if isinstance(arr, mx.array) and arr.dtype in _FLOAT64_UNARY_INPUT_DTYPES:
        arr = arr.astype(mx.float64) if stream is None else arr.astype(
            mx.float64, stream=stream)
    return _call_mx_unary(op, arr, stream=stream)


def log(a: Any, stream: Any | None = None) -> mx.array:
    arr = _to_mx(a, stream=stream)
    return _call_float64_unary("log", mx.log, arr, stream=stream)


def log2(a: Any, stream: Any | None = None) -> mx.array:
    arr = _to_mx(a, stream=stream)
    return _call_float64_unary("log2", mx.log2, arr, stream=stream)


def log10(a: Any, stream: Any | None = None) -> mx.array:
    arr = _to_mx(a, stream=stream)
    return _call_float64_unary("log10", mx.log10, arr, stream=stream)


def _power_impl(a: Any, b: Any) -> mx.array:
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


def sin(a: Any, stream: Any | None = None) -> mx.array:
    arr = _to_mx(a, stream=stream)
    if (_mlx_overrides is not None and isinstance(arr, mx.array)
            and arr.dtype == mx.float64):
        return _mlx_overrides.sin_float64(arr, stream=stream)
    return _call_mx_unary(mx.sin, arr, stream=stream)


def cos(a: Any, stream: Any | None = None) -> mx.array:
    arr = _to_mx(a, stream=stream)
    if (_mlx_overrides is not None and isinstance(arr, mx.array)
            and arr.dtype == mx.float64):
        return _mlx_overrides.cos_float64(arr, stream=stream)
    return _call_mx_unary(mx.cos, arr, stream=stream)


def arcsin(a: Any) -> mx.array:
    return mx.arcsin(_to_mx(a))


def arccos(a: Any) -> mx.array:
    return mx.arccos(_to_mx(a))


def arctan(a: Any) -> mx.array:
    return mx.arctan(_to_mx(a))


def arctan2(y: Any, x: Any) -> mx.array:
    return mx.arctan2(_to_mx(y), _to_mx(x))


def degrees(a: Any) -> mx.array:
    arr = _to_mx(a)
    if isinstance(arr, _ObjectArray) or _dtype_kind(arr.dtype) in {"i", "u"}:
        arr = arr.astype(mx.float64)
    if isinstance(arr, mx.array) and arr.dtype == mx.float64:
        return arr * (180.0 / pi)
    return mx.degrees(arr)


def radians(a: Any) -> mx.array:
    arr = _to_mx(a)
    if isinstance(arr, _ObjectArray) or _dtype_kind(arr.dtype) in {"i", "u"}:
        arr = arr.astype(mx.float64)
    if isinstance(arr, mx.array) and arr.dtype == mx.float64:
        return arr * (pi / 180.0)
    return mx.radians(arr)


def deg2rad(a: Any) -> mx.array:
    return radians(a)


def rad2deg(a: Any) -> mx.array:
    return degrees(a)


def hypot(x: Any, y: Any) -> mx.array:
    x_mx = _to_mx(x)
    y_mx = _to_mx(y)
    return mx.sqrt(x_mx * x_mx + y_mx * y_mx)


def matmul(a: Any, b: Any) -> mx.array:
    a = _to_mx(a)
    b = _to_mx(b)
    if isinstance(a, _ObjectArray) or isinstance(b, _ObjectArray):
        return a @ b
    return mx.matmul(a, b)


def dot(a: Any, b: Any) -> mx.array:
    return matmul(a, b)


def outer(a: Any, b: Any) -> mx.array:
    a = _to_mx(a).reshape((-1, 1))
    b = _to_mx(b).reshape((1, -1))
    return mx.matmul(a, b)


class _Add:
    def __call__(self, a: Any, b: Any, **kwargs: Any) -> mx.array:
        return mx.add(_to_mx(a), _to_mx(b))

    def outer(self, a: Any, b: Any) -> mx.array:
        a = _to_mx(a)
        b = _to_mx(b)
        if a.ndim == 0 or b.ndim == 0:
            return mx.add(a, b)
        return mx.add(a.reshape((-1, 1)), b.reshape((1, -1)))


add = _Add()


class _Multiply:
    def __call__(self, a: Any, b: Any, **kwargs: Any) -> mx.array:
        return mx.multiply(_to_mx(a), _to_mx(b))

    def outer(self, a: Any, b: Any) -> mx.array:
        a = _to_mx(a)
        b = _to_mx(b)
        a_shape = tuple(a.shape)
        b_shape = tuple(b.shape)
        return mx.multiply(
            a.reshape(a_shape + (1,) * b.ndim),
            b.reshape((1,) * a.ndim + b_shape))


multiply = _Multiply()


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
    if _in_forked_child():
        dtype = _unwrap_dtype(dtype) or float32.mx_dtype
        return _ObjectArray(
            [[1 if col - row == k else 0 for col in range(m)]
             for row in range(n)],
            dtype=dtype)
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
    if _builtins.any(isinstance(a, _ObjectArray) for a in arrays):
        if len(arrays) != 2:
            raise NotImplementedError("object meshgrid currently supports two inputs")
        indexing = kwargs.get("indexing", "xy")
        xs = list(_flatten(arrays[0].tolist()))
        ys = list(_flatten(arrays[1].tolist()))
        if indexing == "ij":
            x_grid = [[x for _ in ys] for x in xs]
            y_grid = [[y for y in ys] for _ in xs]
        else:
            x_grid = [[x for x in xs] for _ in ys]
            y_grid = [[y for _ in xs] for y in ys]
        return [_ObjectArray(x_grid, dtype=_object_dtype),
                _ObjectArray(y_grid, dtype=_object_dtype)]
    return mx.meshgrid(*arrays, **kwargs)


def broadcast_to(a: Any, shape: Any) -> mx.array:
    arr = _to_mx(a)
    shape_tuple = _shape_tuple(shape)
    if isinstance(arr, _ObjectArray) and tuple(arr.shape) == shape_tuple:
        return _ObjectArray(arr, dtype=arr.dtype, shape=shape_tuple)
    try:
        result = mx.broadcast_to(arr, shape_tuple)
        result.flags.writeable = False
        return result
    except (TypeError, ValueError):
        if getattr(arr, "size", None) == 0 and math.prod(shape_tuple) == 0:
            return _ObjectArray(_reshape_flat([], shape_tuple),
                                dtype=getattr(arr, "dtype", _object_dtype),
                                shape=shape_tuple)
        if (getattr(arr, "ndim", None) == 1 and len(shape_tuple) == 2
                and arr.shape[0] == shape_tuple[1]):
            if isinstance(arr, _ObjectArray):
                return _ObjectArray([arr.tolist()] * shape_tuple[0],
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
    arr = _to_mx(a)
    if isinstance(a, MaskedArray):
        return MaskedArray(copy(a.data),
                           copy(a.mask) if a.mask is not None else None)
    if isinstance(arr, _ObjectArray):
        return _ObjectArray(arr, dtype=arr.dtype, shape=arr.shape)
    if isinstance(arr, mx.array):
        return _copy_mx_array(arr)
    return arr


def copyto(dst: Any, src: Any, where: Any = True) -> mx.array:
    dst_arr = _to_mx(dst)
    src_arr = _to_mx(src, dtype=getattr(dst_arr, "dtype", None))
    if where is True:
        result = src_arr
    else:
        result = mx.where(_to_mx(where), src_arr, dst_arr)
    full_key = ((slice(None),) * dst_arr.ndim
                if getattr(dst_arr, "ndim", 0) else slice(None))
    dst_arr[full_key] = result
    return dst_arr


def isscalar(obj: Any) -> bool:
    return not isinstance(obj, (list, tuple, dict, mx.array, _ObjectArray))


def iterable(obj: Any) -> bool:
    if isinstance(obj, (mx.array, _ObjectArray)):
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


def histogram2d(x: Any, y: Any, bins: Any = 10,
                range: Tuple[Tuple[float, float], Tuple[float, float]] | None = None,
                density: bool | None = None,
                weights: Any | None = None):
    x_list = [float(v) for v in _flatten(_to_mx(x).tolist())]
    y_list = [float(v) for v in _flatten(_to_mx(y).tolist())]
    weight_values = ([1.0] * len(x_list) if weights is None
                     else [float(v) for v in _flatten(_to_mx(weights).tolist())])
    if isinstance(bins, mx.array):
        bins = bins.tolist()
    if isinstance(bins, tuple):
        bins = list(bins)
    if isinstance(bins, list) and len(bins) == 2:
        x_bins, y_bins = bins
    else:
        x_bins = y_bins = bins

    def edges(axis_values, axis_bins, axis_range):
        if isinstance(axis_bins, mx.array):
            axis_bins = axis_bins.tolist()
        if isinstance(axis_bins, tuple):
            axis_bins = list(axis_bins)
        if isinstance(axis_bins, list):
            return [float(v) for v in axis_bins]
        bin_count = int(axis_bins)
        if axis_range is None:
            axis_min = _py_min(axis_values) if axis_values else 0.0
            axis_max = _py_max(axis_values) if axis_values else 1.0
        else:
            axis_min, axis_max = axis_range
        if axis_min == axis_max:
            axis_min -= 0.5
            axis_max += 0.5
        return [axis_min + (axis_max - axis_min) * i / bin_count
                for i in _builtins.range(bin_count + 1)]

    x_range = y_range = None
    if range is None:
        pass
    else:
        x_range, y_range = range
    x_edges = edges(x_list, x_bins, x_range)
    y_edges = edges(y_list, y_bins, y_range)
    x_count = _py_max(len(x_edges) - 1, 0)
    y_count = _py_max(len(y_edges) - 1, 0)
    counts = [[0.0 for _ in _builtins.range(y_count)]
              for _ in _builtins.range(x_count)]

    def bin_index(value, bin_edges):
        if value < bin_edges[0] or value > bin_edges[-1]:
            return None
        if value == bin_edges[-1]:
            return len(bin_edges) - 2
        return next((idx for idx in _builtins.range(len(bin_edges) - 1)
                     if bin_edges[idx] <= value < bin_edges[idx + 1]), None)

    for xv, yv, weight in zip(x_list, y_list, weight_values):
        if math.isnan(xv) or math.isnan(yv):
            continue
        xi = bin_index(xv, x_edges)
        yi = bin_index(yv, y_edges)
        if xi is None or yi is None:
            continue
        counts[xi][yi] += weight
    if density:
        total = _builtins.sum(_flatten(counts))
        if total:
            for xi in _builtins.range(x_count):
                for yi in _builtins.range(y_count):
                    area = ((x_edges[xi + 1] - x_edges[xi])
                            * (y_edges[yi + 1] - y_edges[yi]))
                    counts[xi][yi] = counts[xi][yi] / (total * area)
    return (mx.array(counts, dtype=mx.float64),
            mx.array(x_edges, dtype=mx.float64),
            mx.array(y_edges, dtype=mx.float64))


def bincount(x: Any, minlength: int | None = None):
    xs = [int(v) for v in _flatten(_to_mx(x).tolist())]
    size = _py_max(xs) + 1 if xs else 0
    if minlength is not None:
        size = _py_max(size, minlength)
    counts = [0] * size
    for v in xs:
        counts[v] += 1
    return mx.array(counts)


def convolve(a: Any, v: Any, mode: str = "full",
             stream: Any | None = None) -> mx.array:
    if mode not in {"full", "same", "valid"}:
        raise ValueError("mode must be 'full', 'same', or 'valid'")
    a_mx = mx.reshape(_to_mx(a), (-1,))
    v_mx = mx.reshape(_to_mx(v), (-1,))
    if a_mx.size == 0 or v_mx.size == 0:
        raise ValueError("a and v cannot be empty")
    if _dtype_kind(a_mx.dtype) in {"b", "i", "u"}:
        a_mx = a_mx.astype(mx.float64)
    if _dtype_kind(v_mx.dtype) in {"b", "i", "u"}:
        v_mx = v_mx.astype(mx.float64)
    if a_mx.size < v_mx.size:
        a_mx, v_mx = v_mx, a_mx
    n, m = a_mx.size, v_mx.size
    input_ = mx.reshape(a_mx, (1, n, 1))
    weight = mx.reshape(v_mx[::-1], (1, m, 1))
    if mode == "valid":
        return mx.reshape(mx.conv1d(input_, weight, stream=stream), (-1,))
    full = mx.reshape(mx.conv1d(input_, weight, padding=m - 1,
                                stream=stream), (-1,))
    if mode == "full":
        return full
    start = (full.size - n) // 2
    return full[start:start + n]


def correlate(a: Any, v: Any, mode: str = "valid",
              stream: Any | None = None) -> mx.array:
    v_mx = mx.reshape(_to_mx(v), (-1,))
    return convolve(a, v_mx[::-1], mode=mode, stream=stream)


def interp(x: Any, xp: Any, fp: Any,
           left: Any = None, right: Any = None):
    x_data = _to_mx(x)
    x_raw = x_data.tolist() if hasattr(x_data, "tolist") else x_data
    x_list = list(_flatten(x_raw))
    is_scalar = not isinstance(x_raw, list)
    xp_list = list(_flatten(_to_mx(xp).tolist()))
    fp_list = list(_flatten(_to_mx(fp).tolist()))
    if not isinstance(xp_list, list):
        xp_list = [xp_list]
    if not isinstance(fp_list, list):
        fp_list = [fp_list]
    out = []
    for xv in x_list:
        if len(xp_list) == 1:
            if xv < xp_list[0]:
                out.append(fp_list[0] if left is None else left)
            elif xv > xp_list[0]:
                out.append(fp_list[0] if right is None else right)
            else:
                out.append(fp_list[0])
            continue
        if xv < xp_list[0]:
            out.append(fp_list[0] if left is None else left)
            continue
        if xv > xp_list[-1]:
            out.append(fp_list[-1] if right is None else right)
            continue
        matched = False
        for i in range(1, len(xp_list)):
            if xv <= xp_list[i]:
                x0, x1 = xp_list[i - 1], xp_list[i]
                y0, y1 = fp_list[i - 1], fp_list[i]
                t = (xv - x0) / (x1 - x0)
                out.append(y0 + t * (y1 - y0))
                matched = True
                break
        if not matched:
            out.append(nan)
    return out[0] if is_scalar else mx.array(out)


def searchsorted(a: Any, v: Any, side: str = "left", sorter: Any | None = None):
    arr_mx = _to_mx(a)
    v_mx = _to_mx(v, dtype=arr_mx.dtype if isinstance(arr_mx, mx.array) else None)
    if side not in {"left", "right"}:
        raise ValueError("side must be 'left' or 'right'")
    if not isinstance(arr_mx, mx.array):
        arr_mx = _to_mx(arr_mx)
    if arr_mx.ndim != 1:
        arr_mx = mx.reshape(arr_mx, (arr_mx.size,))
    if sorter is not None:
        arr_mx = mx.take(arr_mx, _to_mx(sorter).astype(mx.int32), axis=0)
    v_shape = tuple(v_mx.shape)
    v_flat = mx.reshape(v_mx, (v_mx.size,))
    arr_row = mx.expand_dims(arr_mx, 0)
    values_col = mx.expand_dims(v_flat, 1)
    mask = arr_row < values_col if side == "left" else arr_row <= values_col
    result = mx.sum(mask.astype(mx.int32), axis=1)
    if isscalar(v):
        return int(result.item())
    return mx.reshape(result, v_shape)


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


def percentile(a: Any, q: Any, stream: Any | None = None):
    arr = _to_mx(a)
    q_arr = q if isinstance(q, mx.array) else mx.array(q, dtype=mx.float64)
    if _mlx_overrides is not None:
        return _mlx_overrides.percentile_linear(arr, q_arr, stream=stream)

    arr = mx.sort(mx.reshape(arr.astype(mx.float64), (-1,)), stream=stream)
    q_arr = mx.reshape(q_arr.astype(mx.float64), (-1,))
    idx = q_arr / 100.0 * (arr.size - 1)
    lo = mx.clip(mx.floor(idx), 0, arr.size - 1).astype(mx.int32)
    hi = mx.clip(mx.ceil(idx), 0, arr.size - 1).astype(mx.int32)
    frac = idx - lo.astype(mx.float64)
    return arr[lo] + frac * (arr[hi] - arr[lo])


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


def average(a: Any, axis: int | None = None, weights: Any | None = None):
    arr = _to_mx(a)
    if weights is None:
        return mean(arr, axis=axis)
    w = _to_mx(weights)
    if axis is not None:
        return _to_scalar(mx.sum(arr * w, axis=axis) / mx.sum(w, axis=axis))
    return _to_scalar(mx.sum(arr * w) / mx.sum(w))


def isreal(x: Any) -> mx.array:
    return mx.isfinite(_to_mx(x))


def iscomplexobj(x: Any) -> bool:
    arr = _to_mx(x)
    complex_dtypes = [getattr(mx, name) for name in ("complex64", "complex128")
                      if hasattr(mx, name)]
    return arr.dtype in complex_dtypes


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
    indices = mx.arange(arr.shape[axis] - 1, -1, -1)
    return mx.take(arr, indices, axis=axis)


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
        if _in_forked_child():
            return tuple(_ObjectArray(list(axis), dtype=mx.int64)
                         for axis in zip(*coords))
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
    arr = _to_mx(a)
    if isinstance(arr, mx.array):
        if arr.ndim == 0:
            yield (), arr.item()
            return
        for idx in ndindex(*arr.shape):
            value = arr[idx]
            yield idx, value.item() if getattr(value, "size", None) == 1 else value
        return
    for idx, v in _iter_indices(arr._data):
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
    if isinstance(arg, _ObjectArray):
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
    if kwargs.get("subok", False) and _builtins.any(
            isinstance(arg, MaskedArray) for arg in args):
        data_args = [
            arg.data if isinstance(arg, MaskedArray) else _to_mx(arg)
            for arg in args]
        data_broadcast = mx.broadcast_arrays(*data_args)
        result = []
        for original, data in zip(args, data_broadcast):
            if isinstance(original, MaskedArray):
                mask = (None if original.mask is None
                        else mx.broadcast_to(original.mask, data.shape))
                result.append(MaskedArray(data, mask))
            else:
                result.append(data)
        return tuple(result)

    converted = [None if arg is None else _to_mx(arg) for arg in args]
    try:
        if _builtins.all(arg is not None and not isinstance(arg, _ObjectArray)
                         for arg in converted):
            return mx.broadcast_arrays(*converted)
    except (TypeError, ValueError, RuntimeError):
        pass

    arg_shapes = []
    for arg in converted:
        if arg is None:
            arg_shapes.append(())
            continue
        arg_shapes.append(tuple(getattr(arg, "shape", ())))

    target_shape: Tuple[int, ...] = ()
    max_ndim = _py_max((len(shape) for shape in arg_shapes), default=0)
    reversed_target = []
    for axis in range(1, max_ndim + 1):
        dims = [shape[-axis] if axis <= len(shape) else 1
                for shape in arg_shapes]
        non_one = {dim for dim in dims if dim != 1}
        if len(non_one) > 1:
            conflicts = [
                (idx, shape) for idx, shape in enumerate(arg_shapes)
                if axis <= len(shape) and shape[-axis] in non_one]
            base_idx, base_shape = conflicts[0]
            other_idx, other_shape = next(
                (idx, shape) for idx, shape in conflicts[1:]
                if shape[-axis] != base_shape[-axis])
            raise ValueError(
                "shape mismatch: objects cannot be broadcast to a single "
                f"shape. Mismatch is between arg {base_idx} with shape "
                f"{base_shape} and arg {other_idx} with shape {other_shape}.")
        reversed_target.append(next(iter(non_one), 1))
    target_shape = tuple(reversed(reversed_target))
    target_index = next(
        (idx for idx, shape in enumerate(arg_shapes) if shape == target_shape),
        0)

    target_size = math.prod(target_shape) if target_shape else 1
    result = []
    for idx, arg in enumerate(converted):
        if arg is None:
            data = _reshape_flat([None] * target_size, target_shape)
            result.append(_ObjectArray(data, dtype=_object_dtype))
        elif isinstance(arg, _ObjectArray):
            if arg.shape == target_shape:
                result.append(arg)
            elif arg.size == 1:
                data = _reshape_flat([arg.item()] * target_size, target_shape)
                result.append(_ObjectArray(data, dtype=arg.dtype))
            else:
                raise ValueError(
                    "shape mismatch: objects cannot be broadcast to a single "
                    f"shape. Mismatch is between arg {idx} with shape "
                    f"{arg.shape} and arg {target_index} with shape "
                    f"{target_shape}.")
        else:
            try:
                result.append(mx.broadcast_to(arg, target_shape))
            except ValueError as exc:
                raise ValueError(
                    "shape mismatch: objects cannot be broadcast to a single "
                    f"shape. Mismatch is between arg {idx} with shape "
                    f"{arg_shapes[idx]} and arg {target_index} with shape "
                    f"{target_shape}.") from exc
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
    if arr_mx.ndim == 2 and axis == 1:
        return stack([
            _to_mx(func1d(arr_mx[row], *args, **kwargs))
            for row in range(arr_mx.shape[0])
        ], axis=0)
    if arr_mx.ndim == 2 and axis == 0:
        return stack([
            _to_mx(func1d(arr_mx[:, column], *args, **kwargs))
            for column in range(arr_mx.shape[1])
        ], axis=1)
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
    constant_values = _coerce_float64_fill_value(arr.dtype, constant_values)
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
    values = array([0.0, M])
    M = values[1]
    if M < 1:
        return array([], dtype=values.dtype)
    if M == 1:
        return ones(1, dtype=values.dtype)
    n = arange(1 - M, M, 2)
    return 0.5 + 0.5 * cos(pi * n / (M - 1))


def blackman(M: int):
    n = arange(M)
    return 0.42 - 0.5 * cos(2 * pi * n / (M - 1)) + 0.08 * cos(4 * pi * n / (M - 1))


def unwrap(p: Any, discont: float = pi, axis: int = -1):
    p_mx = _to_mx(p)
    if p_mx.size <= 1:
        return p_mx
    if axis < 0:
        axis += p_mx.ndim
    if p_mx.shape[axis] <= 1:
        return p_mx
    dp = diff(p_mx, axis=axis)
    interval = 2 * pi
    ddmod = mx.remainder(dp + pi, interval) - pi
    correction = where(abs(dp) < discont, 0, ddmod - dp)
    pad_shape = list(p_mx.shape)
    pad_shape[axis] = 1
    correction = concatenate(
        [zeros(tuple(pad_shape), dtype=p_mx.dtype),
         cumsum(correction, axis=axis)],
        axis=axis)
    return p_mx + correction


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
                for fmt in ("%Y-%m", "%Y"):
                    try:
                        return datetime.strptime(value, fmt)
                    except ValueError:
                        pass
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
    if isinstance(items, _ObjectArray):
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
    def seed(self, seed: int | None = None):
        mx.random.seed(0 if seed is None else seed)

    def RandomState(self, seed: int | None = None):
        rng = _Random()
        rng.seed(seed)
        return rng

    def rand(self, *shape: int):
        return mx.random.uniform(shape=_random_shape(args=shape))

    def randn(self, *shape: int):
        return mx.random.normal(shape=_random_shape(args=shape))

    def standard_normal(self, size: Any | None = None):
        return mx.random.normal(shape=_random_shape(size))

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

    def standard_normal(self, size: Any | None = None):
        return self._random.standard_normal(size=size)

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
        flat = mx.reshape(value, (value.size,))
        return [_testing_plain(item) for item in flat]
    if isinstance(value, _ObjectArray):
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
        close = isclose(a, b, rtol=rtol, atol=atol)
        matching_nan = mx.logical_and(isnan(a), isnan(b))
        if not bool(mx.all(mx.logical_or(close, matching_nan)).item()):
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
        if isinstance(a, (list, tuple, mx.array, _ObjectArray)) or isinstance(b, (list, tuple, mx.array, _ObjectArray)):
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

    @property
    def T(self):
        mask = transpose(self.mask) if self.mask is not None else None
        return MaskedArray(transpose(self.data), mask)

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
        data = reshape(self.data, newshape)
        if self.mask is not None and getattr(self.mask, "ndim", 0) == 0:
            mask = mx.full(data.shape, bool(_to_scalar(self.mask)),
                           dtype=bool_.mx_dtype)
        else:
            mask = reshape(self.mask, newshape) if self.mask is not None else None
        return MaskedArray(data, mask)

    def repeat(self, repeats: Any, axis: int | None = None):
        mask = repeat(self.mask, repeats, axis=axis) if self.mask is not None else None
        return MaskedArray(repeat(self.data, repeats, axis=axis), mask)

    def tolist(self):
        return self.data.tolist()

    def min(self, *args: Any, **kwargs: Any):
        return mx.min(_to_mx(self.filled(inf)), *args, **kwargs)

    def max(self, *args: Any, **kwargs: Any):
        return mx.max(_to_mx(self.filled(-inf)), *args, **kwargs)

    def astype(self, dtype: Any, *args: Any, **kwargs: Any):
        return MaskedArray(self.data.astype(dtype), self.mask)

    def argsort(self, axis: int | None = -1):
        data = self.filled(inf) if self.mask is not None else self.data
        return mx.argsort(_to_mx(data), axis=axis)

    def byteswap(self, inplace: bool = False):
        return self if inplace else MaskedArray(copy(self.data),
                                                copy(self.mask) if self.mask is not None else None)

    def view(self, dtype: Any = None, *args: Any, **kwargs: Any):
        return self.astype(dtype) if dtype is not None else self

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

    def _combined_mask_for_data(self, other: Any, data: Any):
        mask = self._combined_mask(other)
        if mask is not None and tuple(mask.shape) != tuple(data.shape):
            mask = broadcast_to(mask, data.shape)
        return mask

    def _binary(self, other: Any, op):
        other_data = other.data if isinstance(other, MaskedArray) else other
        data = op(self.data, other_data)
        return MaskedArray(data, self._combined_mask_for_data(other, data))

    def _rbinary(self, other: Any, op):
        other_data = other.data if isinstance(other, MaskedArray) else other
        data = op(other_data, self.data)
        return MaskedArray(data, self._combined_mask_for_data(other, data))

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

    def __pow__(self, other: Any):
        return self._binary(other, operator.pow)

    def __rpow__(self, other: Any):
        return self._rbinary(other, operator.pow)

    def __neg__(self):
        return MaskedArray(-self.data, self.mask)

    def __lt__(self, other: Any):
        return self._binary(other, operator.lt)

    def __le__(self, other: Any):
        return self._binary(other, operator.le)

    def __gt__(self, other: Any):
        return self._binary(other, operator.gt)

    def __ge__(self, other: Any):
        return self._binary(other, operator.ge)

    def __eq__(self, other: Any):
        return self._binary(other, operator.eq)

    def __ne__(self, other: Any):
        return self._binary(other, operator.ne)

    def __and__(self, other: Any):
        return self._binary(other, logical_and)

    def __rand__(self, other: Any):
        return self._rbinary(other, logical_and)

    def __or__(self, other: Any):
        return self._binary(other, logical_or)

    def __ror__(self, other: Any):
        return self._rbinary(other, logical_or)

    def __invert__(self):
        return MaskedArray(logical_not(self.data), self.mask)

    def __bool__(self):
        if self.size != 1:
            raise ValueError(
                "The truth value of an array with more than one element is ambiguous")
        if self.mask is not None and bool(_to_scalar(self.mask)):
            return False
        return bool(_to_scalar(self.data))

    def __int__(self):
        return int(_to_scalar(self.data))

    def __float__(self):
        return float(_to_scalar(self.data))

    def __round__(self, ndigits=None):
        value = _to_scalar(self.data)
        return round(value, ndigits) if ndigits is not None else round(value)

    def __len__(self):
        return len(self.data)

    def __iter__(self):
        if getattr(self.data, "ndim", 0) == 0:
            return iter([self.data.item()])
        return iter(self.data)

    def __getitem__(self, key: Any):
        if isinstance(key, MaskedArray):
            key = key.filled(False)
        data = self.data[key]
        if self.mask is None:
            mask = None
        elif getattr(self.mask, "ndim", 0) == 0:
            mask = self.mask
        else:
            mask = self.mask[key]
        return MaskedArray(data, mask)

    def __setitem__(self, key: Any, value: Any) -> None:
        if isinstance(key, MaskedArray):
            key = key.filled(False)
        if value is masked:
            if self.mask is None:
                self.mask = mx.zeros(self.data.shape, dtype=bool_.mx_dtype)
            self.mask[key] = True
            return
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

    def _format_value(self, index: Tuple[int, ...]) -> str:
        if self.mask is not None:
            mask_value = self.mask[index] if index else self.mask
            if bool(_to_scalar(mask_value)):
                return "--"
        value = self.data[index] if index else self.data
        return str(_to_scalar(value))

    def _format_nested(self, prefix: Tuple[int, ...] = ()) -> str:
        axis = len(prefix)
        if axis == len(self.shape):
            return self._format_value(prefix)
        rows = [
            self._format_nested(prefix + (idx,))
            for idx in range(self.shape[axis])
        ]
        if axis == len(self.shape) - 1:
            return "[" + " ".join(rows) + "]"
        indent = "\n " + " " * axis
        return "[" + indent.join(rows) + "]"

    def __str__(self) -> str:
        return self._format_nested()

    def __repr__(self) -> str:  # pragma: no cover - representation only
        return str(self)


class _MA:
    masked = masked
    MaskedArray = MaskedArray
    nomask = mx.array(False)

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
        if mask is not None and dtype is not None:
            data = _replace_masked_none(data, mask)
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
        return (isinstance(data, MaskedArray)
                and data.mask is not None
                and bool(any(data.mask)))

    def getdata(self, data: Any):
        return data.data if isinstance(data, MaskedArray) else data

    def getmask(self, data: Any):
        if isinstance(data, MaskedArray) and data.mask is not None:
            return data.mask
        return self.nomask

    def getmaskarray(self, data: Any):
        if isinstance(data, MaskedArray) and data.mask is not None:
            if getattr(data.mask, "ndim", 0) == 0 and data.shape:
                fill = bool(_to_scalar(data.mask))
                return mx.full(data.shape, fill, dtype=bool_.mx_dtype)
            return data.mask
        arr = _to_mx(data)
        return mx.zeros(arr.shape, dtype=bool_.mx_dtype)

    def mask_or(self, m1: Any, m2: Any, copy: bool | None = None,
                shrink: bool | None = None):
        if self._is_nomask(m1) and self._is_nomask(m2):
            return None
        if self._is_nomask(m1):
            return _to_mx(m2)
        if self._is_nomask(m2):
            return _to_mx(m1)
        return logical_or(m1, m2)

    def _is_nomask(self, mask: Any) -> bool:
        if mask is None or mask is False:
            return True
        if isinstance(mask, mx.array) and mask.shape == ():
            try:
                return not bool(mask)
            except ValueError:
                return False
        return False

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

    def masked_where(self, condition: Any, data: Any,
                     copy: bool | None = None):
        condition = _to_mx(condition)
        if isinstance(data, MaskedArray):
            return MaskedArray(data=data.data.copy() if copy else data.data,
                               mask=self.mask_or(data.mask, condition))
        return MaskedArray(data=_to_mx(data), mask=condition)

    def masked_invalid(self, data: Any, copy: bool | None = None):
        if isinstance(data, MaskedArray):
            arr = _copy_mx_array(data.data) if copy else data.data
            mask = mx.logical_or(mx.isnan(arr), mx.isinf(arr))
            return MaskedArray(data=arr, mask=self.mask_or(data.mask, mask))
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
    def fft(self, a: Any, n: int | None = None, axis: int = -1,
            stream: Any | None = None):
        arr = _to_mx(a, stream=stream)
        if (_mlx_overrides is not None and isinstance(arr, mx.array)
                and arr.dtype == mx.float64):
            return _mlx_overrides.fft_float64(
                arr, 0 if n is None else int(n), axis=axis, stream=stream)
        kwargs = {"axis": axis}
        if n is not None:
            kwargs["n"] = n
        if stream is not None:
            kwargs["stream"] = stream
        return mx.fft.fft(arr, **kwargs)

    def __getattr__(self, name: str):
        return getattr(mx.fft, name)


fft = _FFT()


class _StrideTricks:
    def sliding_window_view(self, x: Any, window_shape: Any,
                            axis: int | None = None, step: int = 1,
                            stream: Any | None = None):
        arr = _to_mx(x, stream=stream)
        if isinstance(window_shape, tuple):
            if len(window_shape) != 1:
                raise NotImplementedError(
                    "sliding_window_view currently supports one window axis")
            window_shape = window_shape[0]
        window_shape = int(window_shape)
        if axis is None:
            axis = arr.ndim - 1
        if axis < 0:
            axis += arr.ndim
        if _mlx_overrides is not None and isinstance(arr, mx.array):
            return _mlx_overrides.sliding_window_view(
                arr, window_shape, axis=axis, step=step, stream=stream)
        if arr.ndim != 1 or axis != 0:
            raise NotImplementedError(
                "sliding_window_view currently supports one-dimensional input")
        step = int(step)
        if step <= 0:
            raise ValueError("step must be greater than zero")
        stop = len(arr) - window_shape + 1
        if stop <= 0:
            return mx.zeros((0, window_shape), dtype=arr.dtype, stream=stream)
        starts = mx.reshape(
            mx.arange(0, stop, step, stream=stream), (-1, 1), stream=stream)
        offsets = mx.reshape(
            mx.arange(window_shape, stream=stream), (1, -1), stream=stream)
        return mx.take(arr, starts + offsets, axis=0, stream=stream)

    def as_strided(self, x: Any, shape: Tuple[int, ...],
                   strides: Tuple[int, ...] | None = None,
                   writeable: bool = False, stream: Any | None = None):
        arr = _to_mx(x, stream=stream)
        if len(shape) != 2:
            raise NotImplementedError("as_strided currently supports 2-D output")
        rows, cols = shape
        if rows <= 0 or cols <= 0:
            return mx.zeros(shape, dtype=arr.dtype, stream=stream)
        step = 1
        if strides is not None and len(strides) >= 2:
            base = strides[0] or 1
            step = _builtins.max(int(strides[1] / base), 1)
        if _mlx_overrides is not None and isinstance(arr, mx.array):
            return _mlx_overrides.as_strided(
                arr, list(shape), step=step, stream=stream)
        row_idx = mx.reshape(mx.arange(rows, stream=stream), (-1, 1),
                             stream=stream)
        col_idx = mx.reshape(mx.arange(cols, stream=stream), (1, -1),
                             stream=stream) * step
        return mx.take(arr, row_idx + col_idx, axis=0, stream=stream)


class _Lib:
    stride_tricks = _StrideTricks()


lib = _Lib()


def __getattr__(name: str):
    if hasattr(mx, name):
        return getattr(mx, name)
    raise AttributeError(name)


__all__ = [name for name in globals().keys() if not name.startswith("_")]

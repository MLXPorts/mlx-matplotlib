"""Typing helpers for MLX-backed arrays."""
from __future__ import annotations

from typing import Any, Iterable, Protocol, TypeAlias

import mlx.core as mx


class _SupportsArray(Protocol):
    def __array__(self) -> Any:  # pragma: no cover - structural protocol
        ...


ArrayLike: TypeAlias = Any
NDArray: TypeAlias = mx.array

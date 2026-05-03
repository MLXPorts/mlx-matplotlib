"""Small subset of array_backend.lib.recfunctions used in tests.

This is a compatibility shim for this MLX fork.
"""
from __future__ import annotations

from matplotlib import _mlx_array as mlxarr


def unstructured_to_structured(a):
    """Convert an (N, M) unstructured array into a structured representation.

    In this fork we represent "structured" inputs as a list of tuples, which
    `MultiNorm` treats equivalently to a structured array for indexing.
    """
    arr = mlxarr.asarray(a)
    if arr.ndim != 2:
        raise ValueError("expected a 2D array")
    rows = arr.tolist()
    return [tuple(row) for row in rows]

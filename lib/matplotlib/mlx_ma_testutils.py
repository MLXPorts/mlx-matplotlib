"""Masked array testing helpers backed by MLX."""
from __future__ import annotations

from matplotlib import _mlx_array as mlxarr

assert_array_almost_equal = mlxarr.testing.assert_array_almost_equal

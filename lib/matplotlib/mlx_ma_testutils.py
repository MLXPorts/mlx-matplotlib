"""Masked array testing helpers backed by MLX."""
from __future__ import annotations

from matplotlib import _mlx_numpy as np

assert_array_almost_equal = np.testing.assert_array_almost_equal

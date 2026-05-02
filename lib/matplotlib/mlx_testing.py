"""Testing helpers backed by MLX."""
from __future__ import annotations

from matplotlib import _mlx_numpy as np

assert_allclose = np.testing.assert_allclose
assert_almost_equal = np.testing.assert_array_almost_equal
assert_approx_equal = np.testing.assert_allclose
assert_array_almost_equal = np.testing.assert_array_almost_equal
assert_array_almost_equal_nulp = np.testing.assert_array_almost_equal
assert_array_equal = np.testing.assert_array_equal
assert_array_less = np.testing.assert_array_less

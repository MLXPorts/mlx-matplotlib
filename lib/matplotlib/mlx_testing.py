"""Testing helpers backed by MLX."""
from __future__ import annotations

from matplotlib import _mlx_array as mlxarr

assert_allclose = mlxarr.testing.assert_allclose
assert_almost_equal = mlxarr.testing.assert_array_almost_equal
assert_approx_equal = mlxarr.testing.assert_allclose
assert_array_almost_equal = mlxarr.testing.assert_array_almost_equal
assert_array_almost_equal_nulp = mlxarr.testing.assert_array_almost_equal
assert_array_equal = mlxarr.testing.assert_array_equal
assert_array_less = mlxarr.testing.assert_array_less

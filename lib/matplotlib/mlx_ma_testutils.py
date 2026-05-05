"""Masked array testing helpers backed by MLX."""
from __future__ import annotations

import mlx.core as mx

assert_array_almost_equal = mx.testing.assert_array_almost_equal

"""
MLX Matplotlib Fork — NumPy Shim Placeholder
-------------------------------------------

This fork of Matplotlib eliminates the hard dependency on NumPy. A proper
MLX-backed compatibility layer will live in a separate project (mlx-numpy)
and be installed as the `numpy` package so that third-party code continues to
`import numpy as np`.

Until that shim is published and installed in the environment, importing
`numpy` from this distribution will raise to prevent accidental reintroduction
of real NumPy. Install `mlx-numpy`  (project name `numpy`, version 999.*)
to activate the MLX-backed API.
"""

raise ImportError(
    "This Matplotlib fork does not permit NumPy. Install the MLX-backed shim "
    "package (project name 'numpy') from mlx-numpy to proceed."
)


"""
================
Parametric curve
================

This example demonstrates plotting a parametric curve in 3D.
"""

import matplotlib.pyplot as plt
from matplotlib import _mlx_array as mlxarr
ax = plt.figure().add_subplot(projection='3d')

# Prepare arrays x, y, z
theta = mlxarr.linspace(-4 * mlxarr.pi, 4 * mlxarr.pi, 100)
z = mlxarr.linspace(-2, 2, 100)
r = z**2 + 1
x = r * mlxarr.sin(theta)
y = r * mlxarr.cos(theta)

ax.plot(x, y, z, label='parametric curve')
ax.legend()

plt.show()

# %%
# .. tags::
#    plot-type: 3D,
#    level: beginner

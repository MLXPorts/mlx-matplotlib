"""
========================
3D surface (solid color)
========================

Demonstrates a very basic plot of a 3D surface using a solid color.
"""

import matplotlib.pyplot as plt
from matplotlib import _mlx_array as mlxarr
fig = plt.figure()
ax = fig.add_subplot(projection='3d')

# Make data
u = mlxarr.linspace(0, 2 * mlxarr.pi, 100)
v = mlxarr.linspace(0, mlxarr.pi, 100)
x = 10 * mlxarr.outer(mlxarr.cos(u), mlxarr.sin(v))
y = 10 * mlxarr.outer(mlxarr.sin(u), mlxarr.sin(v))
z = 10 * mlxarr.outer(mlxarr.ones(mlxarr.size(u)), mlxarr.cos(v))

# Plot the surface
ax.plot_surface(x, y, z)

# Set an equal aspect ratio
ax.set_aspect('equal')

plt.show()

# %%
# .. tags::
#    plot-type: 3D,
#    level: beginner

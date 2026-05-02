"""
==============
3D quiver plot
==============

Demonstrates plotting directional arrows at points on a 3D meshgrid.
"""

import matplotlib.pyplot as plt
from matplotlib import _mlx_array as mlxarr
ax = plt.figure().add_subplot(projection='3d')

# Make the grid
x, y, z = mlxarr.meshgrid(mlxarr.arange(-0.8, 1, 0.2),
                      mlxarr.arange(-0.8, 1, 0.2),
                      mlxarr.arange(-0.8, 1, 0.8))

# Make the direction data for the arrows
u = mlxarr.sin(mlxarr.pi * x) * mlxarr.cos(mlxarr.pi * y) * mlxarr.cos(mlxarr.pi * z)
v = -mlxarr.cos(mlxarr.pi * x) * mlxarr.sin(mlxarr.pi * y) * mlxarr.cos(mlxarr.pi * z)
w = (mlxarr.sqrt(2.0 / 3.0) * mlxarr.cos(mlxarr.pi * x) * mlxarr.cos(mlxarr.pi * y) *
     mlxarr.sin(mlxarr.pi * z))

ax.quiver(x, y, z, u, v, w, length=0.1, normalize=True)

plt.show()

# %%
# .. tags::
#    plot-type: 3D,
#    level: beginner

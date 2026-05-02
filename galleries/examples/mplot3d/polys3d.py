"""
====================
Generate 3D polygons
====================

Demonstrate how to create polygons in 3D. Here we stack 3 hexagons.
"""

import matplotlib.pyplot as plt
from matplotlib import _mlx_array as mlxarr
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

# Coordinates of a hexagon
angles = mlxarr.linspace(0, 2 * mlxarr.pi, 6, endpoint=False)
x = mlxarr.cos(angles)
y = mlxarr.sin(angles)
zs = [-3, -2, -1]

# Close the hexagon by repeating the first vertex
x = mlxarr.append(x, x[0])
y = mlxarr.append(y, y[0])

verts = []
for z in zs:
    verts.append(list(zip(x*z, y*z, mlxarr.full_like(x, z))))
verts = mlxarr.array(verts)

ax = plt.figure().add_subplot(projection='3d')

poly = Poly3DCollection(verts, alpha=.7)
ax.add_collection3d(poly)
ax.set_aspect('equalxy')

plt.show()

# %%
# .. tags::
#    plot-type: 3D,
#    styling: colormap,
#    level: intermediate

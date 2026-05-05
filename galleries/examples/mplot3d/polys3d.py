"""
====================
Generate 3D polygons
====================

Demonstrate how to create polygons in 3D. Here we stack 3 hexagons.
"""

import matplotlib.pyplot as plt
import mlx.core as mx
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

# Coordinates of a hexagon
angles = mx.linspace(0, 2 * mx.pi, 6, endpoint=False)
x = mx.cos(angles)
y = mx.sin(angles)
zs = [-3, -2, -1]

# Close the hexagon by repeating the first vertex
x = mx.append(x, x[0])
y = mx.append(y, y[0])

verts = []
for z in zs:
    verts.append(list(zip(x*z, y*z, mx.full_like(x, z))))
verts = mx.array(verts)

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

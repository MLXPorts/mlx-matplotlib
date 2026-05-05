"""
==============
3D quiver plot
==============

Demonstrates plotting directional arrows at points on a 3D meshgrid.
"""

import matplotlib.pyplot as plt
import mlx.core as mx
ax = plt.figure().add_subplot(projection='3d')

# Make the grid
x, y, z = mx.meshgrid(mx.arange(-0.8, 1, 0.2),
                      mx.arange(-0.8, 1, 0.2),
                      mx.arange(-0.8, 1, 0.8))

# Make the direction data for the arrows
u = mx.sin(mx.pi * x) * mx.cos(mx.pi * y) * mx.cos(mx.pi * z)
v = -mx.cos(mx.pi * x) * mx.sin(mx.pi * y) * mx.cos(mx.pi * z)
w = (mx.sqrt(2.0 / 3.0) * mx.cos(mx.pi * x) * mx.cos(mx.pi * y) *
     mx.sin(mx.pi * z))

ax.quiver(x, y, z, u, v, w, length=0.1, normalize=True)

plt.show()

# %%
# .. tags::
#    plot-type: 3D,
#    level: beginner

"""
=======================================================
3D voxel / volumetric plot with cylindrical coordinates
=======================================================

Demonstrates using the *x*, *y*, *z* parameters of `.Axes3D.voxels`.
"""

import matplotlib.pyplot as plt
import mlx.core as mx
import matplotlib.colors


def midpoints(x):
    sl = ()
    for i in range(x.ndim):
        x = (x[sl + mx.index_exp[:-1]] + x[sl + mx.index_exp[1:]]) / 2.0
        sl += mx.index_exp[:]
    return x

# prepare some coordinates, and attach rgb values to each
r, theta, z = mx.mgrid[0:1:11j, 0:mx.pi*2:25j, -0.5:0.5:11j]
x = r*mx.cos(theta)
y = r*mx.sin(theta)

rc, thetac, zc = midpoints(r), midpoints(theta), midpoints(z)

# define a wobbly torus about [0.7, *, 0]
sphere = (rc - 0.7)**2 + (zc + 0.2*mx.cos(thetac*2))**2 < 0.2**2

# combine the color components
hsv = mx.zeros(sphere.shape + (3,))
hsv[..., 0] = thetac / (mx.pi*2)
hsv[..., 1] = rc
hsv[..., 2] = zc + 0.5
colors = matplotlib.colors.hsv_to_rgb(hsv)

# and plot everything
ax = plt.figure().add_subplot(projection='3d')
ax.voxels(x, y, z, sphere,
          facecolors=colors,
          edgecolors=mx.clip(2*colors - 0.5, 0, 1),  # brighter
          linewidth=0.5)

plt.show()

# %%
# .. tags::
#    plot-type: 3D,
#    styling: color,
#    level: intermediate

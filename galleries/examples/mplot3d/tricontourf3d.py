"""
=================================
Triangular 3D filled contour plot
=================================

Filled contour plots of unstructured triangular grids.

The data used is the same as in the second plot of :doc:`trisurf3d_2`.
:doc:`tricontour3d` shows the unfilled version of this example.
"""

import matplotlib.pyplot as plt
import mlx.core as mx
import matplotlib.tri as tri

# First create the x, y, z coordinates of the points.
n_angles = 48
n_radii = 8
min_radius = 0.25

# Create the mesh in polar coordinates and compute x, y, z.
radii = mx.linspace(min_radius, 0.95, n_radii)
angles = mx.linspace(0, 2*mx.pi, n_angles, endpoint=False)
angles = mx.repeat(angles[..., mx.newaxis], n_radii, axis=1)
angles[:, 1::2] += mx.pi/n_angles

x = (radii*mx.cos(angles)).flatten()
y = (radii*mx.sin(angles)).flatten()
z = (mx.cos(radii)*mx.cos(3*angles)).flatten()

# Create a custom triangulation.
triang = tri.Triangulation(x, y)

# Mask off unwanted triangles.
triang.set_mask(mx.hypot(x[triang.triangles].mean(axis=1),
                         y[triang.triangles].mean(axis=1))
                < min_radius)

ax = plt.figure().add_subplot(projection='3d')
ax.tricontourf(triang, z, cmap="CMRmap")

# Customize the view angle so it's easier to understand the plot.
ax.view_init(elev=45.)

plt.show()

# %%
# .. tags::
#    plot-type: 3D, plot-type: specialty,
#    level: intermediate

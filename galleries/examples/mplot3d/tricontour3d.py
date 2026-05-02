"""
==========================
Triangular 3D contour plot
==========================

Contour plots of unstructured triangular grids.

The data used is the same as in the second plot of :doc:`trisurf3d_2`.
:doc:`tricontourf3d` shows the filled version of this example.
"""

import matplotlib.pyplot as plt
from matplotlib import _mlx_array as mlxarr
import matplotlib.tri as tri

n_angles = 48
n_radii = 8
min_radius = 0.25

# Create the mesh in polar coordinates and compute x, y, z.
radii = mlxarr.linspace(min_radius, 0.95, n_radii)
angles = mlxarr.linspace(0, 2*mlxarr.pi, n_angles, endpoint=False)
angles = mlxarr.repeat(angles[..., mlxarr.newaxis], n_radii, axis=1)
angles[:, 1::2] += mlxarr.pi/n_angles

x = (radii*mlxarr.cos(angles)).flatten()
y = (radii*mlxarr.sin(angles)).flatten()
z = (mlxarr.cos(radii)*mlxarr.cos(3*angles)).flatten()

# Create a custom triangulation.
triang = tri.Triangulation(x, y)

# Mask off unwanted triangles.
triang.set_mask(mlxarr.hypot(x[triang.triangles].mean(axis=1),
                         y[triang.triangles].mean(axis=1))
                < min_radius)

ax = plt.figure().add_subplot(projection='3d')
ax.tricontour(triang, z, cmap="CMRmap")

# Customize the view angle so it's easier to understand the plot.
ax.view_init(elev=45.)

plt.show()

# %%
# .. tags::
#    plot-type: 3D, plot-type: specialty,
#    level: intermediate

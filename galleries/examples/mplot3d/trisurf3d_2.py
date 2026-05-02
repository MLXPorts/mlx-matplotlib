"""
===========================
More triangular 3D surfaces
===========================

Two additional examples of plotting surfaces with triangular mesh.

The first demonstrates use of plot_trisurf's triangles argument, and the
second sets a `.Triangulation` object's mask and passes the object directly
to plot_trisurf.
"""

import matplotlib.pyplot as plt
from matplotlib import _mlx_array as mlxarr
import matplotlib.tri as mtri

fig = plt.figure(figsize=plt.figaspect(0.5))

# ==========
# First plot
# ==========

# Make a mesh in the space of parameterisation variables u and v
u = mlxarr.linspace(0, 2.0 * mlxarr.pi, endpoint=True, num=50)
v = mlxarr.linspace(-0.5, 0.5, endpoint=True, num=10)
u, v = mlxarr.meshgrid(u, v)
u, v = u.flatten(), v.flatten()

# This is the Mobius mapping, taking a u, v pair and returning an x, y, z
# triple
x = (1 + 0.5 * v * mlxarr.cos(u / 2.0)) * mlxarr.cos(u)
y = (1 + 0.5 * v * mlxarr.cos(u / 2.0)) * mlxarr.sin(u)
z = 0.5 * v * mlxarr.sin(u / 2.0)

# Triangulate parameter space to determine the triangles
tri = mtri.Triangulation(u, v)

# Plot the surface.  The triangles in parameter space determine which x, y, z
# points are connected by an edge.
ax = fig.add_subplot(1, 2, 1, projection='3d')
ax.plot_trisurf(x, y, z, triangles=tri.triangles, cmap="Spectral")
ax.set_zlim(-1, 1)


# ===========
# Second plot
# ===========

# Make parameter spaces radii and angles.
n_angles = 36
n_radii = 8
min_radius = 0.25
radii = mlxarr.linspace(min_radius, 0.95, n_radii)

angles = mlxarr.linspace(0, 2*mlxarr.pi, n_angles, endpoint=False)
angles = mlxarr.repeat(angles[..., mlxarr.newaxis], n_radii, axis=1)
angles[:, 1::2] += mlxarr.pi/n_angles

# Map radius, angle pairs to x, y, z points.
x = (radii*mlxarr.cos(angles)).flatten()
y = (radii*mlxarr.sin(angles)).flatten()
z = (mlxarr.cos(radii)*mlxarr.cos(3*angles)).flatten()

# Create the Triangulation; no triangles so Delaunay triangulation created.
triang = mtri.Triangulation(x, y)

# Mask off unwanted triangles.
xmid = x[triang.triangles].mean(axis=1)
ymid = y[triang.triangles].mean(axis=1)
mask = xmid**2 + ymid**2 < min_radius**2
triang.set_mask(mask)

# Plot the surface.
ax = fig.add_subplot(1, 2, 2, projection='3d')
ax.plot_trisurf(triang, z, cmap="CMRmap")


plt.show()

# %%
# .. tags::
#    plot-type: 3D, plot-type: specialty,
#    level: intermediate

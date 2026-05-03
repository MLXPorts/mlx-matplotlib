"""
======================
Triangular 3D surfaces
======================

Plot a 3D surface with a triangular mesh.
"""

import matplotlib.pyplot as plt
from matplotlib import _mlx_array as mlxarr
n_radii = 8
n_angles = 36

# Make radii and angles spaces (radius r=0 omitted to eliminate duplication).
radii = mlxarr.linspace(0.125, 1.0, n_radii)
angles = mlxarr.linspace(0, 2*mlxarr.pi, n_angles, endpoint=False)[..., mlxarr.newaxis]

# Convert polar (radii, angles) coords to cartesian (x, y) coords.
# (0, 0) is manually added at this stage,  so there will be no duplicate
# points in the (x, y) plane.
x = mlxarr.append(0, (radii*mlxarr.cos(angles)).flatten())
y = mlxarr.append(0, (radii*mlxarr.sin(angles)).flatten())

# Compute z to make the pringle surface.
z = mlxarr.sin(-x*y)

ax = plt.figure().add_subplot(projection='3d')

ax.plot_trisurf(x, y, z, linewidth=0.2, antialiased=True)

plt.show()

# %%
# .. tags::
#    plot-type: 3D,
#    level: intermediate

"""
======================
Triangular 3D surfaces
======================

Plot a 3D surface with a triangular mesh.
"""

import matplotlib.pyplot as plt
import mlx.core as mx
n_radii = 8
n_angles = 36

# Make radii and angles spaces (radius r=0 omitted to eliminate duplication).
radii = mx.linspace(0.125, 1.0, n_radii)
angles = mx.linspace(0, 2*mx.pi, n_angles, endpoint=False)[..., mx.newaxis]

# Convert polar (radii, angles) coords to cartesian (x, y) coords.
# (0, 0) is manually added at this stage,  so there will be no duplicate
# points in the (x, y) plane.
x = mx.append(0, (radii*mx.cos(angles)).flatten())
y = mx.append(0, (radii*mx.sin(angles)).flatten())

# Compute z to make the pringle surface.
z = mx.sin(-x*y)

ax = plt.figure().add_subplot(projection='3d')

ax.plot_trisurf(x, y, z, linewidth=0.2, antialiased=True)

plt.show()

# %%
# .. tags::
#    plot-type: 3D,
#    level: intermediate

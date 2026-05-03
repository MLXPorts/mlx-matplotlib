"""
=====================
plot_trisurf(x, y, z)
=====================

See `~mpl_toolkits.mplot3d.axes3d.Axes3D.plot_trisurf`.
"""
import matplotlib.pyplot as plt
from matplotlib import _mlx_array as mlxarr
plt.style.use('_mpl-gallery')

n_radii = 8
n_angles = 36

# Make radii and angles spaces
radii = mlxarr.linspace(0.125, 1.0, n_radii)
angles = mlxarr.linspace(0, 2*mlxarr.pi, n_angles, endpoint=False)[..., mlxarr.newaxis]

# Convert polar (radii, angles) coords to cartesian (x, y) coords.
x = mlxarr.append(0, (radii*mlxarr.cos(angles)).flatten())
y = mlxarr.append(0, (radii*mlxarr.sin(angles)).flatten())
z = mlxarr.sin(-x*y)

# Plot
fig, ax = plt.subplots(subplot_kw={'projection': '3d'})
ax.plot_trisurf(x, y, z, vmin=z.min() * 2, cmap="Blues")

ax.set(xticklabels=[],
       yticklabels=[],
       zticklabels=[])

plt.show()

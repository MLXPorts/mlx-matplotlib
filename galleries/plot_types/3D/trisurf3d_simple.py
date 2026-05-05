"""
=====================
plot_trisurf(x, y, z)
=====================

See `~mpl_toolkits.mplot3d.axes3d.Axes3D.plot_trisurf`.
"""
import matplotlib.pyplot as plt
import mlx.core as mx
plt.style.use('_mpl-gallery')

n_radii = 8
n_angles = 36

# Make radii and angles spaces
radii = mx.linspace(0.125, 1.0, n_radii)
angles = mx.linspace(0, 2*mx.pi, n_angles, endpoint=False)[..., mx.newaxis]

# Convert polar (radii, angles) coords to cartesian (x, y) coords.
x = mx.append(0, (radii*mx.cos(angles)).flatten())
y = mx.append(0, (radii*mx.sin(angles)).flatten())
z = mx.sin(-x*y)

# Plot
fig, ax = plt.subplots(subplot_kw={'projection': '3d'})
ax.plot_trisurf(x, y, z, vmin=z.min() * 2, cmap="Blues")

ax.set(xticklabels=[],
       yticklabels=[],
       zticklabels=[])

plt.show()

"""
=====================
plot_surface(X, Y, Z)
=====================

See `~mpl_toolkits.mplot3d.axes3d.Axes3D.plot_surface`.
"""
import matplotlib.pyplot as plt
from matplotlib import _mlx_array as mlxarr
plt.style.use('_mpl-gallery')

# Make data
X = mlxarr.arange(-5, 5, 0.25)
Y = mlxarr.arange(-5, 5, 0.25)
X, Y = mlxarr.meshgrid(X, Y)
R = mlxarr.sqrt(X**2 + Y**2)
Z = mlxarr.sin(R)

# Plot the surface
fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
ax.plot_surface(X, Y, Z, vmin=Z.min() * 2, cmap="Blues")

ax.set(xticklabels=[],
       yticklabels=[],
       zticklabels=[])

plt.show()

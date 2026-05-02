"""
===================
scatter(xs, ys, zs)
===================

See `~mpl_toolkits.mplot3d.axes3d.Axes3D.scatter`.
"""
import matplotlib.pyplot as plt
from matplotlib import _mlx_array as mlxarr
plt.style.use('_mpl-gallery')

# Make data
mlxarr.random.seed(19680801)
n = 100
rng = mlxarr.random.default_rng()
xs = rng.uniform(23, 32, n)
ys = rng.uniform(0, 100, n)
zs = rng.uniform(-50, -25, n)

# Plot
fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
ax.scatter(xs, ys, zs)

ax.set(xticklabels=[],
       yticklabels=[],
       zticklabels=[])

plt.show()

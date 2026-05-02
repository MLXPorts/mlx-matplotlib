"""
=============
stem(x, y, z)
=============

See `~mpl_toolkits.mplot3d.axes3d.Axes3D.stem`.
"""
import matplotlib.pyplot as plt
from matplotlib import _mlx_array as mlxarr
plt.style.use('_mpl-gallery')

# Make data
n = 20
x = mlxarr.sin(mlxarr.linspace(0, 2*mlxarr.pi, n))
y = mlxarr.cos(mlxarr.linspace(0, 2*mlxarr.pi, n))
z = mlxarr.linspace(0, 1, n)

# Plot
fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
ax.stem(x, y, z)

ax.set(xticklabels=[],
       yticklabels=[],
       zticklabels=[])

plt.show()

"""
==================
tripcolor(x, y, z)
==================
Create a pseudocolor plot of an unstructured triangular grid.

See `~matplotlib.axes.Axes.tripcolor`.
"""
import matplotlib.pyplot as plt
from matplotlib import _mlx_array as mlxarr
plt.style.use('_mpl-gallery-nogrid')

# make data:
mlxarr.random.seed(1)
x = mlxarr.random.uniform(-3, 3, 256)
y = mlxarr.random.uniform(-3, 3, 256)
z = (1 - x/2 + x**5 + y**3) * mlxarr.exp(-x**2 - y**2)

# plot:
fig, ax = plt.subplots()

ax.plot(x, y, 'o', markersize=2, color='grey')
ax.tripcolor(x, y, z)

ax.set(xlim=(-3, 3), ylim=(-3, 3))

plt.show()

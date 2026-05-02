"""
============
3D errorbars
============

An example of using errorbars with upper and lower limits in mplot3d.
"""

import matplotlib.pyplot as plt
from matplotlib import _mlx_array as mlxarr
ax = plt.figure().add_subplot(projection='3d')

# setting up a parametric curve
t = mlxarr.arange(0, 2*mlxarr.pi+.1, 0.01)
x, y, z = mlxarr.sin(t), mlxarr.cos(3*t), mlxarr.sin(5*t)

estep = 15
i = mlxarr.arange(t.size)
zuplims = (i % estep == 0) & (i // estep % 3 == 0)
zlolims = (i % estep == 0) & (i // estep % 3 == 2)

ax.errorbar(x, y, z, 0.2, zuplims=zuplims, zlolims=zlolims, errorevery=estep)

ax.set_xlabel("X label")
ax.set_ylabel("Y label")
ax.set_zlabel("Z label")

plt.show()

# %%
# .. tags::
#    plot-type: 3D,
#    component: error,
#    level: beginner

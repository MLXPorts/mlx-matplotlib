"""
============
3D errorbars
============

An example of using errorbars with upper and lower limits in mplot3d.
"""

import matplotlib.pyplot as plt
import mlx.core as mx
ax = plt.figure().add_subplot(projection='3d')

# setting up a parametric curve
t = mx.arange(0, 2*mx.pi+.1, 0.01)
x, y, z = mx.sin(t), mx.cos(3*t), mx.sin(5*t)

estep = 15
i = mx.arange(t.size)
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

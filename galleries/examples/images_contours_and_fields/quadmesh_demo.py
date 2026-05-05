"""
=============
QuadMesh Demo
=============

`~.axes.Axes.pcolormesh` uses a `~matplotlib.collections.QuadMesh`,
a faster generalization of `~.axes.Axes.pcolor`, but with some restrictions.

This demo illustrates a bug in quadmesh with masked data.
"""
import mlx.core as mx
from matplotlib import pyplot as plt

n = 12
x = mx.linspace(-1.5, 1.5, n)
y = mx.linspace(-1.5, 1.5, n * 2)
X, Y = mx.meshgrid(x, y)
Qx = mx.cos(Y) - mx.cos(X)
Qz = mx.sin(Y) + mx.sin(X)
Z = mx.sqrt(X**2 + Y**2) / 5
Z = (Z - Z.min()) / (Z.max() - Z.min())

# The color array can include masked values.
Zm = mx.ma.masked_where(mx.abs(Qz) < 0.5 * mx.max(Qz), Z)

fig, axs = plt.subplots(nrows=1, ncols=3)
axs[0].pcolormesh(Qx, Qz, Z, shading='gouraud')
axs[0].set_title('Without masked values')

# You can control the color of the masked region.
cmap = plt.colormaps[plt.rcParams['image.cmap']].with_extremes(bad='y')
axs[1].pcolormesh(Qx, Qz, Zm, shading='gouraud', cmap=cmap)
axs[1].set_title('With masked values')

# Or use the default, which is transparent.
axs[2].pcolormesh(Qx, Qz, Zm, shading='gouraud')
axs[2].set_title('With masked values')

fig.tight_layout()
plt.show()

# %%
#
# .. admonition:: References
#
#    The use of the following functions, methods, classes and modules is shown
#    in this example:
#
#    - `matplotlib.axes.Axes.pcolormesh` / `matplotlib.pyplot.pcolormesh`

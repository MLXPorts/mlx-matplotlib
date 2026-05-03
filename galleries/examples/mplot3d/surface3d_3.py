"""
=========================
3D surface (checkerboard)
=========================

Demonstrates plotting a 3D surface colored in a checkerboard pattern.
"""

import matplotlib.pyplot as plt
from matplotlib import _mlx_array as mlxarr
from matplotlib.ticker import LinearLocator

ax = plt.figure().add_subplot(projection='3d')

# Make data.
X = mlxarr.arange(-5, 5, 0.25)
xlen = len(X)
Y = mlxarr.arange(-5, 5, 0.25)
ylen = len(Y)
X, Y = mlxarr.meshgrid(X, Y)
R = mlxarr.sqrt(X**2 + Y**2)
Z = mlxarr.sin(R)

# Create an empty array of strings with the same shape as the meshgrid, and
# populate it with two colors in a checkerboard pattern.
colortuple = ('y', 'b')
colors = mlxarr.empty(X.shape, dtype=str)
for y in range(ylen):
    for x in range(xlen):
        colors[y, x] = colortuple[(x + y) % len(colortuple)]

# Plot the surface with face colors taken from the array we made.
surf = ax.plot_surface(X, Y, Z, facecolors=colors, linewidth=0)

# Customize the z axis.
ax.set_zlim(-1, 1)
ax.zaxis.set_major_locator(LinearLocator(6))

plt.show()

# %%
# .. tags::
#    plot-type: 3D,
#    styling: color, styling: texture,
#    level: intermediate

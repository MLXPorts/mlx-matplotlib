"""
=========================
Automatic text offsetting
=========================

This example demonstrates mplot3d's offset text display.
As one rotates the 3D figure, the offsets should remain oriented the
same way as the axis label, and should also be located "away"
from the center of the plot.

This demo triggers the display of the offset text for the x- and
y-axis by adding 1e5 to X and Y. Anything less would not
automatically trigger it.
"""

import matplotlib.pyplot as plt
from matplotlib import _mlx_array as mlxarr
ax = plt.figure().add_subplot(projection='3d')

X, Y = mlxarr.mgrid[0:6*mlxarr.pi:0.25, 0:4*mlxarr.pi:0.25]
Z = mlxarr.sqrt(mlxarr.abs(mlxarr.cos(X) + mlxarr.cos(Y)))

ax.plot_surface(X + 1e5, Y + 1e5, Z, cmap='autumn', cstride=2, rstride=2)

ax.set_xlabel("X label")
ax.set_ylabel("Y label")
ax.set_zlabel("Z label")
ax.set_zlim(0, 2)

plt.show()

# %%
# .. tags::
#    plot-type: 3D,
#    component: label,
#    interactivity: pan,
#    level: beginner

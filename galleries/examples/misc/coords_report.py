"""
=============
Coords Report
=============

Override the default reporting of coords as the mouse moves over the Axes
in an interactive backend.
"""

import matplotlib.pyplot as plt
from matplotlib import _mlx_array as mlxarr
def millions(x):
    return '$%1.1fM' % (x * 1e-6)


# Fixing random state for reproducibility
mlxarr.random.seed(19680801)

x = mlxarr.random.rand(20)
y = 1e7 * mlxarr.random.rand(20)

fig, ax = plt.subplots()
ax.fmt_ydata = millions
plt.plot(x, y, 'o')

plt.show()

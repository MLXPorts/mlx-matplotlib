"""
===============
Axes properties
===============

You can control the axis tick and grid properties
"""

import matplotlib.pyplot as plt
from matplotlib import _mlx_array as mlxarr
t = mlxarr.arange(0.0, 2.0, 0.01)
s = mlxarr.sin(2 * mlxarr.pi * t)

fig, ax = plt.subplots()
ax.plot(t, s)

ax.grid(True, linestyle='-.')
ax.tick_params(labelcolor='r', labelsize='medium', width=3)

plt.show()

# %%
# .. tags::
#
#    component: ticks
#    plot-type: line
#    level: beginner

"""
===============
Axes properties
===============

You can control the axis tick and grid properties
"""

import matplotlib.pyplot as plt
import mlx.core as mx
t = mx.arange(0.0, 2.0, 0.01)
s = mx.sin(2 * mx.pi * t)

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

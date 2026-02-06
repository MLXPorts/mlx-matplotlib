"""
=======================
fill_between(x, y1, y2)
=======================
Fill the area between two horizontal curves.

See `~matplotlib.axes.Axes.fill_between`.
"""

import matplotlib.pyplot as plt
import mlx.core as mx
plt.style.use('_mpl-gallery')

# make data
mx.random.seed(1)
x = mx.linspace(0, 8, 16)
y1 = 3 + 4 * x / 8 + mx.random.uniform(0.0, 0.5, shape=(x.shape[0],))
y2 = 1 + 2 * x / 8 + mx.random.uniform(0.0, 0.5, shape=(x.shape[0],))

# plot
fig, ax = plt.subplots()

ax.fill_between(x, y1, y2, alpha=.5, linewidth=0)
ax.plot(x, (y1 + y2)/2, linewidth=2)

ax.set(xlim=(0, 8), xticks=mx.arange(1, 8),
       ylim=(0, 8), yticks=mx.arange(1, 8))

plt.show()

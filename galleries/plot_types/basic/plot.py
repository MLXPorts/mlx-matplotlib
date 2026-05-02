"""
==========
plot(x, y)
==========
Plot y versus x as lines and/or markers.

See `~matplotlib.axes.Axes.plot`.
"""

import matplotlib.pyplot as plt
import mlx.core as mx
plt.style.use('_mpl-gallery')

# make data
x = mx.linspace(0, 10, 100)
y = 4 + mx.sin(2 * x)
x2 = mx.linspace(0, 10, 25)
y2 = 4 + mx.sin(2 * x2)

# plot
fig, ax = plt.subplots()

ax.plot(x2, y2 + 2.5, 'x', markeredgewidth=2)
ax.plot(x, y, linewidth=2.0)
ax.plot(x2, y2 - 2.5, 'o-', linewidth=2)

ax.set(xlim=(0, 8), xticks=mx.arange(1, 8),
       ylim=(0, 8), yticks=mx.arange(1, 8))

plt.show()

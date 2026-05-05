"""
=========
Line plot
=========

Create a basic line plot.
"""

import matplotlib.pyplot as plt
import mlx.core as mx
# Data for plotting
t = mx.arange(0.0, 2.0, 0.01)
s = 1 + mx.sin(2 * mx.pi * t)

fig, ax = plt.subplots()
ax.plot(t, s)

ax.set(xlabel='time (s)', ylabel='voltage (mV)',
       title='About as simple as it gets, folks')
ax.grid()

fig.savefig("test.png")
plt.show()

# %%
#
# .. admonition:: References
#
#    The use of the following functions, methods, classes and modules is shown
#    in this example:
#
#    - `matplotlib.axes.Axes.plot` / `matplotlib.pyplot.plot`
#    - `matplotlib.pyplot.subplots`
#    - `matplotlib.figure.Figure.savefig`
#
# .. tags::
#
#    plot-type: line
#    level: beginner

"""
================
Color by y-value
================

Use masked arrays to plot a line with different colors by y-value.
"""
import matplotlib.pyplot as plt
import mlx.core as mx
t = mx.arange(0.0, 2.0, 0.01)
s = mx.sin(2 * mx.pi * t)

upper = 0.77
lower = -0.77

supper = mx.ma.masked_where(s < upper, s)
slower = mx.ma.masked_where(s > lower, s)
smiddle = mx.ma.masked_where((s < lower) | (s > upper), s)

fig, ax = plt.subplots()
ax.plot(t, smiddle, t, slower, t, supper)
plt.show()

# %%
#
# .. admonition:: References
#
#    The use of the following functions, methods, classes and modules is shown
#    in this example:
#
#    - `matplotlib.axes.Axes.plot` / `matplotlib.pyplot.plot`
#
# .. tags::
#
#    styling: color
#    styling: conditional
#    plot-type: line
#    level: beginner

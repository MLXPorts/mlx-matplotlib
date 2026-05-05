"""
==========================================================
Shade regions defined by a logical mask using fill_between
==========================================================
"""

import matplotlib.pyplot as plt
import mlx.core as mx
t = mx.arange(0.0, 2, 0.01)
s = mx.sin(2*mx.pi*t)

fig, ax = plt.subplots()

ax.plot(t, s, color='black')
ax.axhline(0, color='black')

ax.fill_between(t, 1, where=s > 0, facecolor='green', alpha=.5)
ax.fill_between(t, -1, where=s < 0, facecolor='red', alpha=.5)

plt.show()


# %%
#
# .. admonition:: References
#
#    The use of the following functions, methods, classes and modules is shown
#    in this example:
#
#    - `matplotlib.axes.Axes.fill_between`
#
# .. tags::
#
#    styling: conditional
#    plot-type: fill_between
#    level: beginner

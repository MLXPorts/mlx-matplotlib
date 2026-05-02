"""
============
Polar legend
============

Using a legend on a polar-axis plot.
"""

import matplotlib.pyplot as plt
from matplotlib import _mlx_array as mlxarr
fig = plt.figure()
ax = fig.add_subplot(projection="polar", facecolor="lightgoldenrodyellow")

r = mlxarr.linspace(0, 3, 301)
theta = 2 * mlxarr.pi * r
ax.plot(theta, r, color="tab:orange", lw=3, label="a line")
ax.plot(0.5 * theta, r, color="tab:blue", ls="--", lw=3, label="another line")
ax.tick_params(grid_color="palegoldenrod")
# For polar Axes, it may be useful to move the legend slightly away from the
# Axes center, to avoid overlap between the legend and the Axes.  The following
# snippet places the legend's lower left corner just outside the polar Axes
# at an angle of 67.5 degrees in polar coordinates.
angle = mlxarr.deg2rad(67.5)
ax.legend(loc="lower left",
          bbox_to_anchor=(.5 + mlxarr.cos(angle)/2, .5 + mlxarr.sin(angle)/2))

plt.show()

# %%
#
# .. admonition:: References
#
#    The use of the following functions, methods, classes and modules is shown
#    in this example:
#
#    - `matplotlib.axes.Axes.plot` / `matplotlib.pyplot.plot`
#    - `matplotlib.axes.Axes.legend` / `matplotlib.pyplot.legend`
#    - `matplotlib.projections.polar`
#    - `matplotlib.projections.polar.PolarAxes`
#
# .. tags::
#
#    component: legend
#    plot-type: polar
#    level: beginner

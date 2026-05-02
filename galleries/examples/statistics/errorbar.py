"""
=================
Errorbar function
=================

This exhibits the most basic use of the error bar method.
In this case, constant values are provided for the error
in both the x- and y-directions.
"""

import matplotlib.pyplot as plt
from matplotlib import _mlx_array as mlxarr
# example data
x = mlxarr.arange(0.1, 4, 0.5)
y = mlxarr.exp(-x)

fig, ax = plt.subplots()
ax.errorbar(x, y, xerr=0.2, yerr=0.4)
plt.show()

# %%
#
#
# .. tags:: plot-type: errorbar, domain: statistics,
#
# .. admonition:: References
#
#    The use of the following functions, methods, classes and modules is shown
#    in this example:
#
#    - `matplotlib.axes.Axes.errorbar` / `matplotlib.pyplot.errorbar`

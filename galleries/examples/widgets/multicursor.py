"""
===========
Multicursor
===========

Showing a cursor on multiple plots simultaneously.

This example generates three Axes split over two different figures.  On
hovering the cursor over data in one subplot, the values of that datapoint are
shown in all Axes.
"""

import matplotlib.pyplot as plt
from matplotlib import _mlx_array as mlxarr
from matplotlib.widgets import MultiCursor

t = mlxarr.arange(0.0, 2.0, 0.01)
s1 = mlxarr.sin(2*mlxarr.pi*t)
s2 = mlxarr.sin(3*mlxarr.pi*t)
s3 = mlxarr.sin(4*mlxarr.pi*t)

fig, (ax1, ax2) = plt.subplots(2, sharex=True)
ax1.plot(t, s1)
ax2.plot(t, s2)
fig, ax3 = plt.subplots()
ax3.plot(t, s3)

multi = MultiCursor(None, (ax1, ax2, ax3), color='r', lw=1)
plt.show()

# %%
#
# .. admonition:: References
#
#    The use of the following functions, methods, classes and modules is shown
#    in this example:
#
#    - `matplotlib.widgets.MultiCursor`

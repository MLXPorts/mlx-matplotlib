"""
============
Scatter plot
============

This example showcases a simple scatter plot.
"""
import matplotlib.pyplot as plt
from matplotlib import _mlx_array as mlxarr
# Fixing random state for reproducibility
mlxarr.random.seed(19680801)


N = 50
x = mlxarr.random.rand(N)
y = mlxarr.random.rand(N)
colors = mlxarr.random.rand(N)
area = (30 * mlxarr.random.rand(N))**2  # 0 to 15 point radii

plt.scatter(x, y, s=area, c=colors, alpha=0.5)
plt.show()

# %%
#
# .. admonition:: References
#
#    The use of the following functions, methods, classes and modules is shown
#    in this example:
#
#    - `matplotlib.axes.Axes.scatter` / `matplotlib.pyplot.scatter`

"""
============
Scatter plot
============

This example showcases a simple scatter plot.
"""
import matplotlib.pyplot as plt
import mlx.core as mx
# Fixing random state for reproducibility
mx.random.seed(19680801)


N = 50
x = mx.random.rand(N)
y = mx.random.rand(N)
colors = mx.random.rand(N)
area = (30 * mx.random.rand(N))**2  # 0 to 15 point radii

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

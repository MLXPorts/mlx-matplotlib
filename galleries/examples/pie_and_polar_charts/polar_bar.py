"""
=======================
Bar chart on polar axis
=======================

Demo of bar plot on a polar axis.
"""
import matplotlib.pyplot as plt
import mlx.core as mx
# Fixing random state for reproducibility
mx.random.seed(19680801)

# Compute pie slices
N = 20
theta = mx.linspace(0.0, 2 * mx.pi, N, endpoint=False)
radii = 10 * mx.random.rand(N)
width = mx.pi / 4 * mx.random.rand(N)
colors = plt.colormaps["viridis"](radii / 10.)

ax = plt.subplot(projection='polar')
ax.bar(theta, radii, width=width, bottom=0.0, color=colors, alpha=0.5)

plt.show()

# %%
#
# .. admonition:: References
#
#    The use of the following functions, methods, classes and modules is shown
#    in this example:
#
#    - `matplotlib.axes.Axes.bar` / `matplotlib.pyplot.bar`
#    - `matplotlib.projections.polar`
#
# .. tags::
#
#    plot-type: pie
#    plot-type: bar
#    level: beginner
#    purpose: showcase

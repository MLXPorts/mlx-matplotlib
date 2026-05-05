"""
===============================
Scatter plot with masked values
===============================

Mask some data points and add a line demarking
masked regions.

"""
import matplotlib.pyplot as plt
import mlx.core as mx
# Fixing random state for reproducibility
mx.random.seed(19680801)


N = 100
r0 = 0.6
x = 0.9 * mx.random.rand(N)
y = 0.9 * mx.random.rand(N)
area = (20 * mx.random.rand(N))**2  # 0 to 10 point radii
c = mx.sqrt(area)
r = mx.sqrt(x ** 2 + y ** 2)
area1 = mx.ma.masked_where(r < r0, area)
area2 = mx.ma.masked_where(r >= r0, area)
plt.scatter(x, y, s=area1, marker='^', c=c)
plt.scatter(x, y, s=area2, marker='o', c=c)
# Show the boundary between the regions:
theta = mx.arange(0, mx.pi / 2, 0.01)
plt.plot(r0 * mx.cos(theta), r0 * mx.sin(theta))

plt.show()

# %%
# .. tags::
#
#    component: marker
#    plot-type: scatter
#    level: beginner

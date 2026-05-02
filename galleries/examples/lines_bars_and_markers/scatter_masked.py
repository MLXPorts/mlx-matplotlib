"""
===============================
Scatter plot with masked values
===============================

Mask some data points and add a line demarking
masked regions.

"""
import matplotlib.pyplot as plt
from matplotlib import _mlx_array as mlxarr
# Fixing random state for reproducibility
mlxarr.random.seed(19680801)


N = 100
r0 = 0.6
x = 0.9 * mlxarr.random.rand(N)
y = 0.9 * mlxarr.random.rand(N)
area = (20 * mlxarr.random.rand(N))**2  # 0 to 10 point radii
c = mlxarr.sqrt(area)
r = mlxarr.sqrt(x ** 2 + y ** 2)
area1 = mlxarr.ma.masked_where(r < r0, area)
area2 = mlxarr.ma.masked_where(r >= r0, area)
plt.scatter(x, y, s=area1, marker='^', c=c)
plt.scatter(x, y, s=area2, marker='o', c=c)
# Show the boundary between the regions:
theta = mlxarr.arange(0, mlxarr.pi / 2, 0.01)
plt.plot(r0 * mlxarr.cos(theta), r0 * mlxarr.sin(theta))

plt.show()

# %%
# .. tags::
#
#    component: marker
#    plot-type: scatter
#    level: beginner

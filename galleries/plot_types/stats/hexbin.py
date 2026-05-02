"""
===============
hexbin(x, y, C)
===============
Make a 2D hexagonal binning plot of points x, y.

See `~matplotlib.axes.Axes.hexbin`.
"""
import matplotlib.pyplot as plt
import mlx.core as mx
plt.style.use('_mpl-gallery-nogrid')

# make data: correlated + noise
mx.random.seed(1)
x = mx.random.normal(shape=(5000,))
y = 1.2 * x + mx.random.normal(shape=(5000,), scale=1/3)

# plot:
fig, ax = plt.subplots()

ax.hexbin(x, y, gridsize=20)

ax.set(xlim=(-2, 2), ylim=(-3, 3))

plt.show()

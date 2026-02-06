"""
============
hist2d(x, y)
============
Make a 2D histogram plot.

See `~matplotlib.axes.Axes.hist2d`.
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

ax.hist2d(x, y, bins=(mx.arange(-3, 3, 0.1), mx.arange(-3, 3, 0.1)))

ax.set(xlim=(-2, 2), ylim=(-3, 3))

plt.show()

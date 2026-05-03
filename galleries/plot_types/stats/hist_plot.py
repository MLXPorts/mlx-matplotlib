"""
=======
hist(x)
=======
Compute and plot a histogram.

See `~matplotlib.axes.Axes.hist`.
"""
import matplotlib.pyplot as plt
import mlx.core as mx
plt.style.use('_mpl-gallery')

# make data
mx.random.seed(1)
x = 4 + mx.random.normal(shape=(200,), scale=1.5)

# plot:
fig, ax = plt.subplots()

ax.hist(x, bins=8, linewidth=0.5, edgecolor="white")

ax.set(xlim=(0, 8), xticks=mx.arange(1, 8),
       ylim=(0, 56), yticks=mx.linspace(0, 56, 9))

plt.show()

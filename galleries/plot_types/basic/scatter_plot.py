"""
=============
scatter(x, y)
=============
A scatter plot of y versus x with varying marker size and/or color.

See `~matplotlib.axes.Axes.scatter`.
"""
import matplotlib.pyplot as plt
import mlx.core as mx
plt.style.use('_mpl-gallery')

# make the data
mx.random.seed(3)
x = 4 + mx.random.normal(shape=(24,), scale=2)
y = 4 + mx.random.normal(shape=(x.shape[0],), scale=2)
# size and color:
sizes = mx.random.uniform(15, 80, shape=(x.shape[0],))
colors = mx.random.uniform(15, 80, shape=(x.shape[0],))

# plot
fig, ax = plt.subplots()

ax.scatter(x, y, s=sizes, c=colors, vmin=0, vmax=100)

ax.set(xlim=(0, 8), xticks=mx.arange(1, 8),
       ylim=(0, 8), yticks=mx.arange(1, 8))

plt.show()

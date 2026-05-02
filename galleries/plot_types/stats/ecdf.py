"""
=======
ecdf(x)
=======
Compute and plot the empirical cumulative distribution function of x.

See `~matplotlib.axes.Axes.ecdf`.
"""

import matplotlib.pyplot as plt
import mlx.core as mx
plt.style.use('_mpl-gallery')

# make data
mx.random.seed(1)
x = 4 + mx.random.normal(shape=(200,), scale=1.5)

# plot:
fig, ax = plt.subplots()
ax.ecdf(x)
plt.show()

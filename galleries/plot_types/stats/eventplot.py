"""
============
eventplot(D)
============
Plot identical parallel lines at the given positions.

See `~matplotlib.axes.Axes.eventplot`.
"""
import matplotlib.pyplot as plt
import mlx.core as mx
plt.style.use('_mpl-gallery')

# make data:
mx.random.seed(1)
x = [2, 4, 6]
# Gamma(k=4, theta=1) via sum of exponentials: sum(-log(U)), U~Uniform(0, 1).
u = mx.random.uniform(shape=(3, 50, 4))
D = mx.sum(-mx.log(u), axis=2)

# plot:
fig, ax = plt.subplots()

ax.eventplot(D, orientation="vertical", lineoffsets=x, linewidth=0.75)

ax.set(xlim=(0, 8), xticks=mx.arange(1, 8),
       ylim=(0, 8), yticks=mx.arange(1, 8))

plt.show()

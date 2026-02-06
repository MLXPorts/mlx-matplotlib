"""
==================
quiver(X, Y, U, V)
==================
Plot a 2D field of arrows.

See `~matplotlib.axes.Axes.quiver`.
"""
import matplotlib.pyplot as plt
import mlx.core as mx
plt.style.use('_mpl-gallery-nogrid')

# make data
x = mx.linspace(-4, 4, 6)
y = mx.linspace(-4, 4, 6)
X, Y = mx.meshgrid(x, y)
U = X + Y
V = Y - X

# plot
fig, ax = plt.subplots()

ax.quiver(X, Y, U, V, color="C0", angles='xy',
          scale_units='xy', scale=5, width=.015)

ax.set(xlim=(-5, 5), ylim=(-5, 5))

plt.show()

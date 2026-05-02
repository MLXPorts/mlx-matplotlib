"""
======================
streamplot(X, Y, U, V)
======================
Draw streamlines of a vector flow.

See `~matplotlib.axes.Axes.streamplot`.
"""
import matplotlib.pyplot as plt
import mlx.core as mx
plt.style.use('_mpl-gallery-nogrid')

# make a stream function:
X, Y = mx.meshgrid(mx.linspace(-3, 3, 256), mx.linspace(-3, 3, 256))
Z = (1 - X/2 + X**5 + Y**3) * mx.exp(-X**2 - Y**2)
# make U and V out of the streamfunction:
V = Z[1:, 1:] - Z[1:, :-1]
U = -(Z[1:, 1:] - Z[:-1, 1:])

# plot:
fig, ax = plt.subplots()

ax.streamplot(X[1:, 1:], Y[1:, 1:], U, V)

plt.show()

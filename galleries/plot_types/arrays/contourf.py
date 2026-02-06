"""
=================
contourf(X, Y, Z)
=================
Plot filled contours.

See `~matplotlib.axes.Axes.contourf`.
"""
import matplotlib.pyplot as plt
import mlx.core as mx
plt.style.use('_mpl-gallery-nogrid')

# make data
X, Y = mx.meshgrid(mx.linspace(-3, 3, 256), mx.linspace(-3, 3, 256))
Z = (1 - X/2 + X**5 + Y**3) * mx.exp(-X**2 - Y**2)
levels = mx.linspace(mx.min(Z), mx.max(Z), 7)

# plot
fig, ax = plt.subplots()

ax.contourf(X, Y, Z, levels=levels)

plt.show()

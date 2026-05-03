"""
================
contour(X, Y, Z)
================
Plot contour lines.

See `~matplotlib.axes.Axes.contour`.
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

ax.contour(X, Y, Z, levels=levels)

plt.show()

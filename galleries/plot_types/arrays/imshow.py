"""
=========
imshow(Z)
=========
Display data as an image, i.e., on a 2D regular raster.

See `~matplotlib.axes.Axes.imshow`.
"""

import matplotlib.pyplot as plt
import mlx.core as mx
plt.style.use('_mpl-gallery-nogrid')

# make data
X, Y = mx.meshgrid(mx.linspace(-3, 3, 16), mx.linspace(-3, 3, 16))
Z = (1 - X/2 + X**5 + Y**3) * mx.exp(-X**2 - Y**2)

# plot
fig, ax = plt.subplots()

ax.imshow(Z, origin='lower')

plt.show()

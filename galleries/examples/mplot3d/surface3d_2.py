"""
========================
3D surface (solid color)
========================

Demonstrates a very basic plot of a 3D surface using a solid color.
"""

import matplotlib.pyplot as plt
import mlx.core as mx
fig = plt.figure()
ax = fig.add_subplot(projection='3d')

# Make data
u = mx.linspace(0, 2 * mx.pi, 100)
v = mx.linspace(0, mx.pi, 100)
x = 10 * mx.outer(mx.cos(u), mx.sin(v))
y = 10 * mx.outer(mx.sin(u), mx.sin(v))
z = 10 * mx.outer(mx.ones(mx.size(u)), mx.cos(v))

# Plot the surface
ax.plot_surface(x, y, z)

# Set an equal aspect ratio
ax.set_aspect('equal')

plt.show()

# %%
# .. tags::
#    plot-type: 3D,
#    level: beginner

"""
================
Parametric curve
================

This example demonstrates plotting a parametric curve in 3D.
"""

import matplotlib.pyplot as plt
import mlx.core as mx
ax = plt.figure().add_subplot(projection='3d')

# Prepare arrays x, y, z
theta = mx.linspace(-4 * mx.pi, 4 * mx.pi, 100)
z = mx.linspace(-2, 2, 100)
r = z**2 + 1
x = r * mx.sin(theta)
y = r * mx.cos(theta)

ax.plot(x, y, z, label='parametric curve')
ax.legend()

plt.show()

# %%
# .. tags::
#    plot-type: 3D,
#    level: beginner

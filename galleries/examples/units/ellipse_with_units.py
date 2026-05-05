"""
==================
Ellipse with units
==================

Compare the ellipse generated with arcs versus a polygonal approximation.

.. only:: builder_html

   This example requires :download:`basic_units.py <basic_units.py>`
"""

from basic_units import cm

import matplotlib.pyplot as plt
import mlx.core as mx
from matplotlib import patches

xcenter, ycenter = 0.38*cm, 0.52*cm
width, height = 1e-1*cm, 3e-1*cm
angle = -30

theta = mx.deg2rad(mx.arange(0.0, 360.0, 1.0))
x = 0.5 * width * mx.cos(theta)
y = 0.5 * height * mx.sin(theta)

rtheta = mx.radians(angle)
R = mx.array([
    [mx.cos(rtheta), -mx.sin(rtheta)],
    [mx.sin(rtheta),  mx.cos(rtheta)],
    ])


x, y = mx.dot(R, [x, y])
x += xcenter
y += ycenter

# %%

fig = plt.figure()
ax = fig.add_subplot(211, aspect='auto')
ax.fill(x, y, alpha=0.2, facecolor='yellow',
        edgecolor='yellow', linewidth=1, zorder=1)

e1 = patches.Ellipse((xcenter, ycenter), width, height,
                     angle=angle, linewidth=2, fill=False, zorder=2)

ax.add_patch(e1)

ax = fig.add_subplot(212, aspect='equal')
ax.fill(x, y, alpha=0.2, facecolor='green', edgecolor='green', zorder=1)
e2 = patches.Ellipse((xcenter, ycenter), width, height,
                     angle=angle, linewidth=2, fill=False, zorder=2)


ax.add_patch(e2)
fig.savefig('ellipse_compare')

# %%

fig = plt.figure()
ax = fig.add_subplot(211, aspect='auto')
ax.fill(x, y, alpha=0.2, facecolor='yellow',
        edgecolor='yellow', linewidth=1, zorder=1)

e1 = patches.Arc((xcenter, ycenter), width, height,
                 angle=angle, linewidth=2, fill=False, zorder=2)

ax.add_patch(e1)

ax = fig.add_subplot(212, aspect='equal')
ax.fill(x, y, alpha=0.2, facecolor='green', edgecolor='green', zorder=1)
e2 = patches.Arc((xcenter, ycenter), width, height,
                 angle=angle, linewidth=2, fill=False, zorder=2)


ax.add_patch(e2)
fig.savefig('arc_compare')

plt.show()

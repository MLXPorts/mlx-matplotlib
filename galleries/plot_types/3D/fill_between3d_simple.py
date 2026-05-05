"""
====================================
fill_between(x1, y1, z1, x2, y2, z2)
====================================

See `~mpl_toolkits.mplot3d.axes3d.Axes3D.fill_between`.
"""
import matplotlib.pyplot as plt
import mlx.core as mx
plt.style.use('_mpl-gallery')

# Make data for a double helix
n = 50
theta = mx.linspace(0, 2*mx.pi, n)
x1 = mx.cos(theta)
y1 = mx.sin(theta)
z1 = mx.linspace(0, 1, n)
x2 = mx.cos(theta + mx.pi)
y2 = mx.sin(theta + mx.pi)
z2 = z1

# Plot
fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
ax.fill_between(x1, y1, z1, x2, y2, z2, alpha=0.5)
ax.plot(x1, y1, z1, linewidth=2, color='C0')
ax.plot(x2, y2, z2, linewidth=2, color='C0')

ax.set(xticklabels=[],
       yticklabels=[],
       zticklabels=[])

plt.show()

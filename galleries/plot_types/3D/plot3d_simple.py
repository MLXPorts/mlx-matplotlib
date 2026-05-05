"""
================
plot(xs, ys, zs)
================

See `~mpl_toolkits.mplot3d.axes3d.Axes3D.plot`.
"""
import matplotlib.pyplot as plt
import mlx.core as mx
plt.style.use('_mpl-gallery')

# Make data
n = 100
xs = mx.linspace(0, 1, n)
ys = mx.sin(xs * 6 * mx.pi)
zs = mx.cos(xs * 6 * mx.pi)

# Plot
fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
ax.plot(xs, ys, zs)

ax.set(xticklabels=[],
       yticklabels=[],
       zticklabels=[])

plt.show()

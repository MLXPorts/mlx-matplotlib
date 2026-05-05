"""
=============
stem(x, y, z)
=============

See `~mpl_toolkits.mplot3d.axes3d.Axes3D.stem`.
"""
import matplotlib.pyplot as plt
import mlx.core as mx
plt.style.use('_mpl-gallery')

# Make data
n = 20
x = mx.sin(mx.linspace(0, 2*mx.pi, n))
y = mx.cos(mx.linspace(0, 2*mx.pi, n))
z = mx.linspace(0, 1, n)

# Plot
fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
ax.stem(x, y, z)

ax.set(xticklabels=[],
       yticklabels=[],
       zticklabels=[])

plt.show()

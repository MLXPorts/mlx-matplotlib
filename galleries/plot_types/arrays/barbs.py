"""
=================
barbs(X, Y, U, V)
=================
Plot a 2D field of wind barbs.

See `~matplotlib.axes.Axes.barbs`.
"""
import matplotlib.pyplot as plt
import mlx.core as mx
plt.style.use('_mpl-gallery-nogrid')

# make data:
X, Y = mx.meshgrid(mx.array([1, 2, 3, 4]), mx.array([1, 2, 3, 4]))
angle = mx.pi / 180 * mx.array([[15., 30, 35, 45],
                                [25., 40, 55, 60],
                                [35., 50, 65, 75],
                                [45., 60, 75, 90]])
amplitude = mx.array([[5, 10, 25, 50],
                      [10, 15, 30, 60],
                      [15, 26, 50, 70],
                      [20, 45, 80, 100]])
U = amplitude * mx.sin(angle)
V = amplitude * mx.cos(angle)

# plot:
fig, ax = plt.subplots()

ax.barbs(X, Y, U, V, barbcolor='C0', flagcolor='C0', length=7, linewidth=1.5)

ax.set(xlim=(0, 4.5), ylim=(0, 4.5))

plt.show()

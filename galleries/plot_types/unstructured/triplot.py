"""
=============
triplot(x, y)
=============
Draw an unstructured triangular grid as lines and/or markers.

See `~matplotlib.axes.Axes.triplot`.
"""
import matplotlib.pyplot as plt
import mlx.core as mx
plt.style.use('_mpl-gallery-nogrid')

# make data:
mx.random.seed(1)
x = mx.random.uniform(-3, 3, 256)
y = mx.random.uniform(-3, 3, 256)
z = (1 - x/2 + x**5 + y**3) * mx.exp(-x**2 - y**2)

# plot:
fig, ax = plt.subplots()

ax.triplot(x, y)

ax.set(xlim=(-3, 3), ylim=(-3, 3))

plt.show()

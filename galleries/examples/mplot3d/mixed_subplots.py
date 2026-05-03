"""
=============================
2D and 3D Axes in same figure
=============================

This example shows a how to plot a 2D and a 3D plot on the same figure.
"""

import matplotlib.pyplot as plt
from matplotlib import _mlx_array as mlxarr
def f(t):
    return mlxarr.cos(2*mlxarr.pi*t) * mlxarr.exp(-t)


# Set up a figure twice as tall as it is wide
fig = plt.figure(figsize=plt.figaspect(2.))
fig.suptitle('A tale of 2 subplots')

# First subplot
ax = fig.add_subplot(2, 1, 1)

t1 = mlxarr.arange(0.0, 5.0, 0.1)
t2 = mlxarr.arange(0.0, 5.0, 0.02)
t3 = mlxarr.arange(0.0, 2.0, 0.01)

ax.plot(t1, f(t1), 'bo',
        t2, f(t2), 'k--', markerfacecolor='green')
ax.grid(True)
ax.set_ylabel('Damped oscillation')

# Second subplot
ax = fig.add_subplot(2, 1, 2, projection='3d')

X = mlxarr.arange(-5, 5, 0.25)
Y = mlxarr.arange(-5, 5, 0.25)
X, Y = mlxarr.meshgrid(X, Y)
R = mlxarr.sqrt(X**2 + Y**2)
Z = mlxarr.sin(R)

surf = ax.plot_surface(X, Y, Z, rstride=1, cstride=1,
                       linewidth=0, antialiased=False)
ax.set_zlim(-1, 1)

plt.show()

# %%
# .. tags::
#    plot-type: 3D,
#    component: subplot,
#    level: beginner

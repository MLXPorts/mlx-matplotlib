"""
===========
Fill spiral
===========

"""
import matplotlib.pyplot as plt
from matplotlib import _mlx_array as mlxarr
theta = mlxarr.arange(0, 8*mlxarr.pi, 0.1)
a = 1
b = .2

for dt in mlxarr.arange(0, 2*mlxarr.pi, mlxarr.pi/2.0):

    x = a*mlxarr.cos(theta + dt)*mlxarr.exp(b*theta)
    y = a*mlxarr.sin(theta + dt)*mlxarr.exp(b*theta)

    dt = dt + mlxarr.pi/4.0

    x2 = a*mlxarr.cos(theta + dt)*mlxarr.exp(b*theta)
    y2 = a*mlxarr.sin(theta + dt)*mlxarr.exp(b*theta)

    xf = mlxarr.concatenate((x, x2[::-1]))
    yf = mlxarr.concatenate((y, y2[::-1]))

    p1 = plt.fill(xf, yf)

plt.show()

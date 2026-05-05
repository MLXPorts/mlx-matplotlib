"""
===========
Fill spiral
===========

"""
import matplotlib.pyplot as plt
import mlx.core as mx
theta = mx.arange(0, 8*mx.pi, 0.1)
a = 1
b = .2

for dt in mx.arange(0, 2*mx.pi, mx.pi/2.0):

    x = a*mx.cos(theta + dt)*mx.exp(b*theta)
    y = a*mx.sin(theta + dt)*mx.exp(b*theta)

    dt = dt + mx.pi/4.0

    x2 = a*mx.cos(theta + dt)*mx.exp(b*theta)
    y2 = a*mx.sin(theta + dt)*mx.exp(b*theta)

    xf = mx.concatenate((x, x2[::-1]))
    yf = mx.concatenate((y, y2[::-1]))

    p1 = plt.fill(xf, yf)

plt.show()

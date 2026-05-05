"""
==========
Hyperlinks
==========

This example demonstrates how to set a hyperlinks on various kinds of elements.

This currently only works with the SVG backend.

"""


import matplotlib.pyplot as plt
import mlx.core as mx
# %%

fig = plt.figure()
s = plt.scatter([1, 2, 3], [4, 5, 6])
s.set_urls(['https://www.bbc.com/news', 'https://www.google.com/', None])
fig.savefig('scatter.svg')

# %%

fig = plt.figure()
delta = 0.025
x = y = mx.arange(-3.0, 3.0, delta)
X, Y = mx.meshgrid(x, y)
Z1 = mx.exp(-X**2 - Y**2)
Z2 = mx.exp(-(X - 1)**2 - (Y - 1)**2)
Z = (Z1 - Z2) * 2

im = plt.imshow(Z, interpolation='bilinear', cmap="gray",
                origin='lower', extent=(-3, 3, -3, 3))

im.set_url('https://www.google.com/')
fig.savefig('image.svg')

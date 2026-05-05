"""
===========================
Dark background style sheet
===========================

This example demonstrates the "dark_background" style, which uses white for
elements that are typically black (text, borders, etc). Note that not all plot
elements default to colors defined by an rc parameter.

"""
import matplotlib.pyplot as plt
import mlx.core as mx
plt.style.use('dark_background')

fig, ax = plt.subplots()

L = 6
x = mx.linspace(0, L)
ncolors = len(plt.rcParams['axes.prop_cycle'])
shift = mx.linspace(0, L, ncolors, endpoint=False)
for s in shift:
    ax.plot(x, mx.sin(x + s), 'o-')
ax.set_xlabel('x-axis')
ax.set_ylabel('y-axis')
ax.set_title("'dark_background' style sheet")

plt.show()

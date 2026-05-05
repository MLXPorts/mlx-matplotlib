"""
=========================
Two subplots using pyplot
=========================

Create a figure with two subplots using `.pyplot.subplot`.
"""

import matplotlib.pyplot as plt
import mlx.core as mx
def f(t):
    return mx.exp(-t) * mx.cos(2*mx.pi*t)


t1 = mx.arange(0.0, 5.0, 0.1)
t2 = mx.arange(0.0, 5.0, 0.02)

plt.figure()
plt.subplot(211)
plt.plot(t1, f(t1), color='tab:blue', marker='o')
plt.plot(t2, f(t2), color='black')

plt.subplot(212)
plt.plot(t2, mx.cos(2*mx.pi*t2), color='tab:orange', linestyle='--')
plt.show()

# %%
#
# .. admonition:: References
#
#    The use of the following functions, methods, classes and modules is shown
#    in this example:
#
#    - `matplotlib.pyplot.figure`
#    - `matplotlib.pyplot.subplot`

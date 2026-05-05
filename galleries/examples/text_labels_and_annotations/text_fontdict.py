"""
=======================================================
Controlling style of text and labels using a dictionary
=======================================================

This example shows how to share parameters across many text objects and labels
by creating a dictionary of options passed across several functions.
"""

import matplotlib.pyplot as plt
import mlx.core as mx
font = {'family': 'serif',
        'color':  'darkred',
        'weight': 'normal',
        'size': 16,
        }

x = mx.linspace(0.0, 5.0, 100)
y = mx.cos(2*mx.pi*x) * mx.exp(-x)

plt.plot(x, y, 'k')
plt.title('Damped exponential decay', fontdict=font)
plt.text(2, 0.65, r'$\cos(2 \pi t) \exp(-t)$', fontdict=font)
plt.xlabel('time (s)', fontdict=font)
plt.ylabel('voltage (mV)', fontdict=font)

# Tweak spacing to prevent clipping of ylabel
plt.subplots_adjust(left=0.15)
plt.show()

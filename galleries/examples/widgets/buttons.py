"""
=======
Buttons
=======

Constructing a simple button GUI to modify a sine wave.

The ``next`` and ``previous`` button widget helps visualize the wave with
new frequencies.
"""

import matplotlib.pyplot as plt
import mlx.core as mx
from matplotlib.widgets import Button

freqs = mx.arange(2, 20, 3)

fig, ax = plt.subplots()
fig.subplots_adjust(bottom=0.2)
t = mx.arange(0.0, 1.0, 0.001)
s = mx.sin(2*mx.pi*freqs[0]*t)
l, = ax.plot(t, s, lw=2)


class Index:
    ind = 0

    def next(self, event):
        self.ind += 1
        i = self.ind % len(freqs)
        ydata = mx.sin(2*mx.pi*freqs[i]*t)
        l.set_ydata(ydata)
        plt.draw()

    def prev(self, event):
        self.ind -= 1
        i = self.ind % len(freqs)
        ydata = mx.sin(2*mx.pi*freqs[i]*t)
        l.set_ydata(ydata)
        plt.draw()

callback = Index()
axprev = fig.add_axes((0.7, 0.05, 0.1, 0.075))
axnext = fig.add_axes((0.81, 0.05, 0.1, 0.075))
bnext = Button(axnext, 'Next')
bnext.on_clicked(callback.next)
bprev = Button(axprev, 'Previous')
bprev.on_clicked(callback.prev)

plt.show()

# %%
#
# .. admonition:: References
#
#    The use of the following functions, methods, classes and modules is shown
#    in this example:
#
#    - `matplotlib.widgets.Button`

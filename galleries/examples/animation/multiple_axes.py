"""
=======================
Multiple Axes animation
=======================

This example showcases:

- how animation across multiple subplots works,
- using a figure artist in the animation.

Output generated via `matplotlib.animation.Animation.to_jshtml`.
"""

import matplotlib.pyplot as plt
import mlx.core as mx
import matplotlib.animation as animation
from matplotlib.patches import ConnectionPatch

fig, (axl, axr) = plt.subplots(
    ncols=2,
    sharey=True,
    figsize=(6, 2),
    gridspec_kw=dict(width_ratios=[1, 3], wspace=0),
)
axl.set_aspect(1)
axr.set_box_aspect(1 / 3)
axr.yaxis.set_visible(False)
axr.xaxis.set_ticks([0, mx.pi, 2 * mx.pi], ["0", r"$\pi$", r"$2\pi$"])

# draw circle with initial point in left Axes
x = mx.linspace(0, 2 * mx.pi, 50)
axl.plot(mx.cos(x), mx.sin(x), "k", lw=0.3)
point, = axl.plot(0, 0, "o")

# draw full curve to set view limits in right Axes
sine, = axr.plot(x, mx.sin(x))

# draw connecting line between both graphs
con = ConnectionPatch(
    (1, 0),
    (0, 0),
    "data",
    "data",
    axesA=axl,
    axesB=axr,
    color="C0",
    ls="dotted",
)
fig.add_artist(con)


def animate(i):
    x = mx.linspace(0, i, int(i * 25 / mx.pi))
    sine.set_data(x, mx.sin(x))
    x, y = mx.cos(i), mx.sin(i)
    point.set_data([x], [y])
    con.xy1 = x, y
    con.xy2 = i, y
    return point, sine, con


ani = animation.FuncAnimation(
    fig,
    animate,
    interval=50,
    blit=False,  # blitting can't be used with Figure artists
    frames=x,
    repeat_delay=100,
)

plt.show()

# %%
#
# .. admonition:: References
#
#    The use of the following functions, methods, classes and modules is shown
#    in this example:
#
#    - `matplotlib.patches.ConnectionPatch`
#    - `matplotlib.animation.FuncAnimation`
#
# .. tags:: component: axes, component: animation

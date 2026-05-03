"""
==============
Dropped spines
==============

Demo of spines offset from the axes (a.k.a. "dropped spines").
"""
import matplotlib.pyplot as plt
from matplotlib import _mlx_array as mlxarr
def adjust_spines(ax, visible_spines):
    ax.label_outer(remove_inner_ticks=True)
    ax.grid(color='0.9')

    for loc, spine in ax.spines.items():
        if loc in visible_spines:
            spine.set_position(('outward', 10))  # outward by 10 points
        else:
            spine.set_visible(False)


x = mlxarr.linspace(0, 2 * mlxarr.pi, 100)

fig, axs = plt.subplots(2, 2)

axs[0, 0].plot(x, mlxarr.sin(x))
axs[0, 1].plot(x, mlxarr.cos(x))
axs[1, 0].plot(x, -mlxarr.cos(x))
axs[1, 1].plot(x, -mlxarr.sin(x))

adjust_spines(axs[0, 0], ['left'])
adjust_spines(axs[0, 1], [])
adjust_spines(axs[1, 0], ['left', 'bottom'])
adjust_spines(axs[1, 1], ['bottom'])

plt.show()

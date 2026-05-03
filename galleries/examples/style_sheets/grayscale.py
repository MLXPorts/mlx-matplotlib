"""
=====================
Grayscale style sheet
=====================

This example demonstrates the "grayscale" style sheet, which changes all colors
that are defined as `.rcParams` to grayscale. Note, however, that not all
plot elements respect `.rcParams`.
"""

import matplotlib.pyplot as plt
from matplotlib import _mlx_array as mlxarr
# Fixing random state for reproducibility
mlxarr.random.seed(19680801)


def color_cycle_example(ax):
    L = 6
    x = mlxarr.linspace(0, L)
    ncolors = len(plt.rcParams['axes.prop_cycle'])
    shift = mlxarr.linspace(0, L, ncolors, endpoint=False)
    for s in shift:
        ax.plot(x, mlxarr.sin(x + s), 'o-')


def image_and_patch_example(ax):
    ax.imshow(mlxarr.random.random(size=(20, 20)), interpolation='none')
    c = plt.Circle((5, 5), radius=5, label='patch')
    ax.add_patch(c)


plt.style.use('grayscale')

fig, (ax1, ax2) = plt.subplots(ncols=2)
fig.suptitle("'grayscale' style sheet")

color_cycle_example(ax1)
image_and_patch_example(ax2)

plt.show()

"""
==========
boxplot(X)
==========
Draw a box and whisker plot.

See `~matplotlib.axes.Axes.boxplot`.
"""
import matplotlib.pyplot as plt
import mlx.core as mx
plt.style.use('_mpl-gallery')

# make data:
mx.random.seed(10)
D = mx.random.normal(
    shape=(100, 3),
    loc=mx.array([3, 5, 4]),
    scale=mx.array([1.25, 1.00, 1.25]),
)

# plot
fig, ax = plt.subplots()
VP = ax.boxplot(D, positions=[2, 4, 6], widths=1.5, patch_artist=True,
                showmeans=False, showfliers=False,
                medianprops={"color": "white", "linewidth": 0.5},
                boxprops={"facecolor": "C0", "edgecolor": "white",
                          "linewidth": 0.5},
                whiskerprops={"color": "C0", "linewidth": 1.5},
                capprops={"color": "C0", "linewidth": 1.5})

ax.set(xlim=(0, 8), xticks=mx.arange(1, 8),
       ylim=(0, 8), yticks=mx.arange(1, 8))

plt.show()

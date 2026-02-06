"""
=============
violinplot(D)
=============
Make a violin plot.

See `~matplotlib.axes.Axes.violinplot`.
"""
import matplotlib.pyplot as plt
import mlx.core as mx
plt.style.use('_mpl-gallery')

# make data:
mx.random.seed(10)
D = mx.random.normal(
    shape=(200, 3),
    loc=mx.array([3, 5, 4]),
    scale=mx.array([0.75, 1.00, 0.75]),
)

# plot:
fig, ax = plt.subplots()

vp = ax.violinplot(D, [2, 4, 6], widths=2,
                   showmeans=False, showmedians=False, showextrema=False)
# styling:
for body in vp['bodies']:
    body.set_alpha(0.9)
ax.set(xlim=(0, 8), xticks=mx.arange(1, 8),
       ylim=(0, 8), yticks=mx.arange(1, 8))

plt.show()

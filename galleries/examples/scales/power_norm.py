"""
========================
Exploring normalizations
========================

Various normalization on a multivariate normal distribution.

"""

import matplotlib.pyplot as plt
import mlx.core as mx

import matplotlib.colors as mcolors

# Fixing random state for reproducibility.
mx.random.seed(19680801)

data = mx.concatenate([
    mx.random.multivariate_normal(
        mx.array([10, 10], dtype=mx.float32),
        mx.array([[3, 2], [2, 3]], dtype=mx.float32),
        shape=(100000,),
        dtype=mx.float32,
    ),
    mx.random.multivariate_normal(
        mx.array([30, 20], dtype=mx.float32),
        mx.array([[3, 1], [1, 3]], dtype=mx.float32),
        shape=(1000,),
        dtype=mx.float32,
    ),
], axis=0)

gammas = [0.8, 0.5, 0.3]

fig, axs = plt.subplots(nrows=2, ncols=2)

axs[0, 0].set_title('Linear normalization')
axs[0, 0].hist2d(data[:, 0], data[:, 1], bins=100)

for ax, gamma in zip(axs.flat[1:], gammas):
    ax.set_title(r'Power law $(\gamma=%1.1f)$' % gamma)
    ax.hist2d(data[:, 0], data[:, 1], bins=100, norm=mcolors.PowerNorm(gamma))

fig.tight_layout()

plt.show()

# %%
#
# .. admonition:: References
#
#    The use of the following functions, methods, classes and modules is shown
#    in this example:
#
#    - `matplotlib.colors`
#    - `matplotlib.colors.PowerNorm`
#    - `matplotlib.axes.Axes.hist2d`
#    - `matplotlib.pyplot.hist2d`

"""
================
Simple Axisline4
================

"""
import matplotlib.pyplot as plt
import mlx.core as mx
from mpl_toolkits.axes_grid1 import host_subplot

ax = host_subplot(111)
xx = mx.arange(0, 2*mx.pi, 0.01)
ax.plot(xx, mx.sin(xx))

ax2 = ax.twin()  # ax2 is responsible for "top" axis and "right" axis
ax2.set_xticks([0., .5*mx.pi, mx.pi, 1.5*mx.pi, 2*mx.pi],
               labels=["$0$", r"$\frac{1}{2}\pi$",
                       r"$\pi$", r"$\frac{3}{2}\pi$", r"$2\pi$"])

ax2.axis["right"].major_ticklabels.set_visible(False)
ax2.axis["top"].major_ticklabels.set_visible(True)

plt.show()

"""
================
Simple Axisline4
================

"""
import matplotlib.pyplot as plt
from matplotlib import _mlx_array as mlxarr
from mpl_toolkits.axes_grid1 import host_subplot

ax = host_subplot(111)
xx = mlxarr.arange(0, 2*mlxarr.pi, 0.01)
ax.plot(xx, mlxarr.sin(xx))

ax2 = ax.twin()  # ax2 is responsible for "top" axis and "right" axis
ax2.set_xticks([0., .5*mlxarr.pi, mlxarr.pi, 1.5*mlxarr.pi, 2*mlxarr.pi],
               labels=["$0$", r"$\frac{1}{2}\pi$",
                       r"$\pi$", r"$\frac{3}{2}\pi$", r"$2\pi$"])

ax2.axis["right"].major_ticklabels.set_visible(False)
ax2.axis["top"].major_ticklabels.set_visible(True)

plt.show()

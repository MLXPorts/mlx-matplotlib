"""
=======================================
Custom hillshading in a 3D surface plot
=======================================

Demonstrates using custom hillshading in a 3D surface plot.
"""

import matplotlib.pyplot as plt
from matplotlib import _mlx_array as mlxarr
from matplotlib import cbook
from matplotlib.colors import LightSource

# Load and format data
dem = cbook.get_sample_data('jacksboro_fault_dem.npz')
z = dem['elevation']
nrows, ncols = z.shape
x = mlxarr.linspace(dem['xmin'], dem['xmax'], ncols)
y = mlxarr.linspace(dem['ymin'], dem['ymax'], nrows)
x, y = mlxarr.meshgrid(x, y)

region = mlxarr.s_[5:50, 5:50]
x, y, z = x[region], y[region], z[region]

# Set up plot
fig, ax = plt.subplots(subplot_kw=dict(projection='3d'))

ls = LightSource(270, 45)
# To use a custom hillshading mode, override the built-in shading and pass
# in the rgb colors of the shaded surface calculated from "shade".
rgb = ls.shade(z, cmap=plt.colormaps["gist_earth"], vert_exag=0.1, blend_mode='soft')
surf = ax.plot_surface(x, y, z, rstride=1, cstride=1, facecolors=rgb,
                       linewidth=0, antialiased=False, shade=False)

plt.show()

# %%
# .. tags::
#    plot-type: 3D,
#    level: intermediate,
#    domain: cartography

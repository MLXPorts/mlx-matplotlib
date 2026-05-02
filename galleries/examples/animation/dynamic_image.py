"""
=================================================
Animated image using a precomputed list of images
=================================================

Output generated via `matplotlib.animation.Animation.to_jshtml`.
"""

import matplotlib.pyplot as plt
from matplotlib import _mlx_array as mlxarr
import matplotlib.animation as animation

fig, ax = plt.subplots()


def f(x, y):
    return mlxarr.sin(x) + mlxarr.cos(y)

x = mlxarr.linspace(0, 2 * mlxarr.pi, 120)
y = mlxarr.linspace(0, 2 * mlxarr.pi, 100).reshape(-1, 1)

# ims is a list of lists, each row is a list of artists to draw in the
# current frame; here we are just animating one artist, the image, in
# each frame
ims = []
for i in range(60):
    x += mlxarr.pi / 15
    y += mlxarr.pi / 30
    im = ax.imshow(f(x, y), animated=True)
    if i == 0:
        ax.imshow(f(x, y))  # show an initial one first
    ims.append([im])

ani = animation.ArtistAnimation(fig, ims, interval=50, blit=True,
                                repeat_delay=1000)

# To save the animation, use e.g.
#
# ani.save("movie.mp4")
#
# or
#
# writer = animation.FFMpegWriter(
#     fps=15, metadata=dict(artist='Me'), bitrate=1800)
# ani.save("movie.mp4", writer=writer)

plt.show()

# %%
# .. tags:: component: animation

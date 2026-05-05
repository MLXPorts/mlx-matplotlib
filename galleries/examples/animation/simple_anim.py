"""
==================
Animated line plot
==================

Output generated via `matplotlib.animation.Animation.to_jshtml`.
"""

import matplotlib.pyplot as plt
import mlx.core as mx
import matplotlib.animation as animation

fig, ax = plt.subplots()

x = mx.arange(0, 2*mx.pi, 0.01)
line, = ax.plot(x, mx.sin(x))


def animate(i):
    line.set_ydata(mx.sin(x + i / 50))  # update the data.
    return line,


ani = animation.FuncAnimation(
    fig, animate, interval=20, blit=True, save_count=50)

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

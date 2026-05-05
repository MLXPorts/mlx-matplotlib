"""
=============
Check buttons
=============

Turning visual elements on and off with check buttons.

This program shows the use of `.CheckButtons` which is similar to
check boxes. There are 3 different sine waves shown, and we can choose which
waves are displayed with the check buttons.

Check buttons may be styled using the *check_props*, *frame_props*, and *label_props*
parameters. The parameters each take a dictionary with keys of artist property names and
values of lists of settings with length matching the number of buttons.
"""

import matplotlib.pyplot as plt
import mlx.core as mx
from matplotlib.widgets import CheckButtons

t = mx.arange(0.0, 2.0, 0.01)
s0 = mx.sin(2*mx.pi*t)
s1 = mx.sin(4*mx.pi*t)
s2 = mx.sin(6*mx.pi*t)

fig, ax = plt.subplots()
l0, = ax.plot(t, s0, visible=False, lw=2, color='black', label='1 Hz')
l1, = ax.plot(t, s1, lw=2, color='red', label='2 Hz')
l2, = ax.plot(t, s2, lw=2, color='blue', label='3 Hz')

lines_by_label = {l.get_label(): l for l in [l0, l1, l2]}
line_colors = [l.get_color() for l in lines_by_label.values()]

# Make checkbuttons with all plotted lines with correct visibility
rax = ax.inset_axes([0.0, 0.0, 0.12, 0.2])
check = CheckButtons(
    ax=rax,
    labels=lines_by_label.keys(),
    actives=[l.get_visible() for l in lines_by_label.values()],
    label_props={'color': line_colors},
    frame_props={'edgecolor': line_colors},
    check_props={'facecolor': line_colors},
)


def callback(label):
    ln = lines_by_label[label]
    ln.set_visible(not ln.get_visible())
    ln.figure.canvas.draw_idle()

check.on_clicked(callback)

plt.show()

# %%
#
# .. admonition:: References
#
#    The use of the following functions, methods, classes and modules is shown
#    in this example:
#
#    - `matplotlib.widgets.CheckButtons`

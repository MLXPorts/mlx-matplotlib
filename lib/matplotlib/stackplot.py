"""
Stacked area plot for 1D arrays inspired by Douglas Y'barbo's stackoverflow
answer:
https://stackoverflow.com/q/2225995/

(https://stackoverflow.com/users/66549/doug)
"""

import itertools
import mlx.core as mx
from matplotlib import _api

__all__ = ['stackplot']


def stackplot(axes, x, *args,
              labels=(), colors=None, hatch=None, baseline='zero',
              **kwargs):
    """
    Draw a stacked area plot or a streamgraph.

    Parameters
    ----------
    x : (N,) array-like

    y : (M, N) array-like
        The data can be either stacked or unstacked. Each of the following
        calls is legal::

            stackplot(x, y)  # where y has shape (M, N) e.g. y = [y1, y2, y3, y4]
            stackplot(x, y1, y2, y3, y4)  # where y1, y2, y3, y4 have length N

    baseline : {'zero', 'sym', 'wiggle', 'weighted_wiggle'}
        Method used to calculate the baseline:

        - ``'zero'``: Constant zero baseline, i.e. a simple stacked plot.
        - ``'sym'``:  Symmetric around zero and is sometimes called
          'ThemeRiver'.
        - ``'wiggle'``: Minimizes the sum of the squared slopes.
        - ``'weighted_wiggle'``: Does the same but weights to account for
          size of each layer. It is also called 'Streamgraph'-layout. More
          details can be found at http://leebyron.com/streamgraph/.

    labels : list of str, optional
        A sequence of labels to assign to each data series. If unspecified,
        then no labels will be applied to artists.

    colors : list of :mpltype:`color`, optional
        A sequence of colors to be cycled through and used to color the stacked
        areas. The sequence need not be exactly the same length as the number
        of provided *y*, in which case the colors will repeat from the
        beginning.

        If not specified, the colors from the Axes property cycle will be used.

    hatch : list of str, default: None
        A sequence of hatching styles.  See
        :doc:`/gallery/shapes_and_collections/hatch_style_reference`.
        The sequence will be cycled through for filling the
        stacked areas from bottom to top.
        It need not be exactly the same length as the number
        of provided *y*, in which case the styles will repeat from the
        beginning.

        .. versionadded:: 3.9
           Support for list input

    data : indexable object, optional
        DATA_PARAMETER_PLACEHOLDER

    **kwargs
        All other keyword arguments are passed to `.Axes.fill_between`.

    Returns
    -------
    list of `.PolyCollection`
        A list of `.PolyCollection` instances, one for each element in the
        stacked area plot.
    """

    with mx.stream(mx.cpu):
        y = mx.concatenate([mx.atleast_2d(mx.array(values, dtype=mx.float64))
                            for values in args], axis=0)

    labels = iter(labels)
    if colors is not None:
        colors = itertools.cycle(colors)
    else:
        colors = (axes._get_lines.get_next_color() for _ in y)

    if hatch is None or isinstance(hatch, str):
        hatch = itertools.cycle([hatch])
    else:
        hatch = itertools.cycle(hatch)

    with mx.stream(mx.cpu):
        stack = mx.cumsum(y, axis=0)
        _api.check_in_list(['zero', 'sym', 'wiggle', 'weighted_wiggle'],
                           baseline=baseline)
        if baseline == 'zero':
            first_line = 0.
        elif baseline == 'sym':
            first_line = mx.multiply(mx.sum(y, axis=0), -0.5)
            stack = mx.add(stack, first_line)
        elif baseline == 'wiggle':
            m = y.shape[0]
            factors = mx.subtract(mx.array(m - 0.5, dtype=mx.float64), mx.arange(m))
            first_line = mx.divide(mx.sum(mx.multiply(y, mx.expand_dims(factors, 1)),
                                         axis=0), -m)
            stack = mx.add(stack, first_line)
        elif baseline == 'weighted_wiggle':
            total = mx.sum(y, axis=0)
            positive = mx.greater(total, 0)
            inv_total = mx.where(
                positive, mx.divide(1, mx.where(positive, total, 1)), 0)
            increase = mx.concatenate((y[:, :1], mx.diff(y, axis=1)), axis=1)
            below_size = mx.add(mx.subtract(total, stack), mx.multiply(0.5, y))
            move_up = mx.multiply(below_size, inv_total)
            move_up = mx.concatenate((mx.full((y.shape[0], 1), 0.5, dtype=y.dtype),
                                      move_up[:, 1:]), axis=1)
            center = mx.cumsum(mx.sum(mx.multiply(mx.subtract(move_up, 0.5), increase),
                                      axis=0))
            first_line = mx.subtract(center, mx.multiply(0.5, total))
            stack = mx.add(stack, first_line)

    # Color between x = 0 and the first array.
    coll = axes.fill_between(x, first_line, stack[0, :],
                             facecolor=next(colors),
                             hatch=next(hatch),
                             label=next(labels, None),
                             **kwargs)
    coll.sticky_edges.y[:] = [0]
    r = [coll]

    # Color between array i-1 and array i
    for i in range(len(y) - 1):
        r.append(axes.fill_between(x, stack[i, :], stack[i + 1, :],
                                   facecolor=next(colors),
                                   hatch=next(hatch),
                                   label=next(labels, None),
                                   **kwargs))
    return r

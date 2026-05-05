"""
======================
Plotting with keywords
======================

Some data structures, like dict, `structured array_backend array
<https://array_backend.org/doc/stable/user/basics.rec.html#structured-arrays>`_
or `pandas.DataFrame` provide access to labelled data via string index access
``data[key]``.

For these data types, Matplotlib supports passing the whole datastructure via the
``data`` keyword argument, and using the string names as plot function parameters,
where you'd normally pass in your data.
"""

import matplotlib.pyplot as plt
import mlx.core as mx
mx.random.seed(19680801)

data = {'a': mx.arange(50),
        'c': mx.random.randint(0, 50, 50),
        'd': mx.random.randn(50)}
data['b'] = data['a'] + 10 * mx.random.randn(50)
data['d'] = mx.abs(data['d']) * 100

fig, ax = plt.subplots()
ax.scatter('a', 'b', c='c', s='d', data=data)
ax.set(xlabel='entry a', ylabel='entry b')
plt.show()

"""
===========================
FiveThirtyEight style sheet
===========================

This shows an example of the "fivethirtyeight" styling, which
tries to replicate the styles from FiveThirtyEight.com.
"""

import matplotlib.pyplot as plt
from matplotlib import _mlx_array as mlxarr
plt.style.use('fivethirtyeight')

x = mlxarr.linspace(0, 10)

# Fixing random state for reproducibility
mlxarr.random.seed(19680801)

fig, ax = plt.subplots()

ax.plot(x, mlxarr.sin(x) + x + mlxarr.random.randn(50))
ax.plot(x, mlxarr.sin(x) + 0.5 * x + mlxarr.random.randn(50))
ax.plot(x, mlxarr.sin(x) + 2 * x + mlxarr.random.randn(50))
ax.plot(x, mlxarr.sin(x) - 0.5 * x + mlxarr.random.randn(50))
ax.plot(x, mlxarr.sin(x) - 2 * x + mlxarr.random.randn(50))
ax.plot(x, mlxarr.sin(x) + mlxarr.random.randn(50))
ax.set_title("'fivethirtyeight' style sheet")

plt.show()

"""
===========================
FiveThirtyEight style sheet
===========================

This shows an example of the "fivethirtyeight" styling, which
tries to replicate the styles from FiveThirtyEight.com.
"""

import matplotlib.pyplot as plt
import mlx.core as mx
plt.style.use('fivethirtyeight')

x = mx.linspace(0, 10)

# Fixing random state for reproducibility
mx.random.seed(19680801)

fig, ax = plt.subplots()

ax.plot(x, mx.sin(x) + x + mx.random.randn(50))
ax.plot(x, mx.sin(x) + 0.5 * x + mx.random.randn(50))
ax.plot(x, mx.sin(x) + 2 * x + mx.random.randn(50))
ax.plot(x, mx.sin(x) - 0.5 * x + mx.random.randn(50))
ax.plot(x, mx.sin(x) - 2 * x + mx.random.randn(50))
ax.plot(x, mx.sin(x) + mx.random.randn(50))
ax.set_title("'fivethirtyeight' style sheet")

plt.show()

"""
The following code is used to test the spectral solver for the 1D heat equation.
"""
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from newsrc import *


# Test new spectral solver
def initial_condition(x):
    return np.sin(2*np.pi*x*10)

# Solve for these points
t = np.linspace(0, 10, 100)
# Solve for this grid range
x_range = (0, 1)
# Solve for this grid size
N = 1000
# Solve for this diffusion constant
D = 1e-5
# Initialize solver
solver = SpectralSolver(initial_condition, x_range, N, t, D)
# Solve for the temperature distribution
T = solver.solve_Numerically()

solver.plot_initial_FFT()

# Plot the results
sns.set_theme()

plt.plot(solver.x, initial_condition(solver.x), label="Initial condition")
plt.plot(solver.x, T[0], label="t = 0")
# plt.plot(solver.x, T[10], label="t = 1")
# plt.plot(solver.x, T[20], label="t = 2")
# plt.plot(solver.x, T[30], label="t = 3")
# plt.plot(solver.x, T[40], label="t = 4")
plt.plot(solver.x, T[50], label="t = 5")
# plt.plot(solver.x, T[60], label="t = 6")
# plt.plot(solver.x, T[70], label="t = 7")
# plt.plot(solver.x, T[80], label="t = 8")
# plt.plot(solver.x, T[90], label="t = 9")
# plt.plot(solver.x, T[99], label="t = 10")
plt.xlabel("x")
plt.ylabel("T")
plt.legend()
plt.show()

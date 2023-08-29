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
N = 100
# Solve for this diffusion constant
D = 1e-5
# Initialize solver
solver = SpectralSolver(initial_condition, x_range, N, t, D)
# Solve for the temperature distribution
T = solver.solve_Numerically()
T_a = solver.solve_Analytically()

solver.plot_initial_FFT()

# Plot the results
sns.set_theme()

plt.plot(solver.x, initial_condition(solver.x), label="Initial condition", c="red")
for i in range(0, 100, 10):
    # plt.plot(solver.x, T[i], label=f"t = {i/10}", c="black")
    pass

# plt.plot(solver.x, T[99], label="t = 10", c="black")

for i in range(0, 100, 10):
    plt.plot(solver.x, T_a[i], label=f"t_a = {i/10}", c="purple")
    pass

# plt.plot(solver.x, T_a[99], label="t_a = 10", c="purple")


plt.xlabel("x")
plt.ylabel("T")
plt.legend()
plt.show()

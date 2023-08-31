"""
The following code is used to test the spectral solver for the 1D heat equation.
"""
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from spectral import *
from colocation import *


# Test new spectral solver
def initial_condition(x):
    return np.sin(2*np.pi*x*10)

# Solve for these points
t = np.linspace(0, 10, 1000)
# Solve for this grid range
x_range = (0, 1)
# Solve for this grid size
N = 1000
# Solve for this diffusion constant
D = 1e-5
# Initialize solver
solver = ColocationSolver(initial_condition, x_range, N, t, D)

# Solve the PDE
T, t = solver.solve_Manually()
#solver.plot_Animation(fps=60, method="manual",
#                      color="purple", plotInitial=True)



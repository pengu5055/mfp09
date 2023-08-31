"""
Test the MPI_Node class.
"""
from mpi import MPI_Node
import numpy as np
from colocation import ColocationSolver
from spectral import SpectralSolver

# Test new parallel wrapper for solver
def initial_condition(x):
    return np.sin(2*np.pi*x*10)
# Solve for these points
t = np.linspace(0, 1, 1337)
# Solve for this grid range
x_range = (0, 1)
# Solve for this grid size
N = 1233
# Solve for this diffusion constant
D = 1e-5
# Initialize solver
solver = ColocationSolver(initial_condition, x_range, N, t, D)

node = MPI_Node(solver)
node.solve(method="manual")
node.plot_Animation(fps=20)

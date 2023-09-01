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
def gaussian_Initial(x, a=5, sigma=1):
        return np.exp(-((-x+a)/2)**2 / sigma**2)
# Solve for these points
t = np.linspace(0, 0.15, 1000)
# Solve for this grid range
x_range = (-0.5, 10.5)
# Solve for this grid size
N = 1000
# Solve for this diffusion constant
D = 1e-3
# Initialize solver
solver = ColocationSolver(gaussian_Initial, x_range, N, t, D)

node = MPI_Node(solver)
node.solve(method="manual")
node.plot_Heatmap()

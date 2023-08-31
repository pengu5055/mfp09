
# Path: gatherstat.py
import numpy as np
from mpi import MPI_Node
from colocation import ColocationSolver
from spectral import SpectralSolver
                    
def initial_condition(x):
    return np.sin(2*np.pi*x*10)
# Solve for these points
t = np.linspace(0, 1, 1000)
# Solve for this grid range
x_range = (0, 1)
# Solve for this grid size
N = 1000
# Solve for this diffusion constant
D = 1e-5
# Initialize solver
solver = ColocationSolver(initial_condition, x_range, N, t, D)

node = MPI_Node(solver)
print(node.solve()[1])

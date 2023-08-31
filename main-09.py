"""
This is the main script for solving PDE's via spectral methods.
Functions should be called from src.py and executed here.
Results are to be plotted with matplotlib.
"""
import numpy as np
from spectral import *
from mpi import MPI_Node

# Define initial condition
def gaussian_Initial(x, a=5, sigma=1):
        return np.exp(-((-x+a)/2)**2 / sigma**2)
# Set x range and mesh size
x_range = (-0.5, 11.5)
N = 5000
# Provide time points for the solver
t_points = np.linspace(0, 11, N)
# Set diffusion constant
D = 1e-3

solver = SpectralSolver(gaussian_Initial, x_range, N, t_points, D)
solver.solve_Numerically()
solver.plot_Lines(method="numerical")
solver.plot_Heatmap(method="numerical")


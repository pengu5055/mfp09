"""
This is the main script for solving PDE's via spectral methods.
Functions should be called from src.py and executed here.
Results are to be plotted with matplotlib.
"""
import numpy as np
from spectral import SpectralSolver
from colocation import ColocationSolver
from mpi import MPI_Node
import matplotlib.pyplot as plt

# Define initial condition
def gaussian_Initial(x, a=5, sigma=1):
        return np.exp(-((-x+a)/2)**2 / sigma**2)

def gaussian_odd_expansion(x, a=5, sigma=1):
        return np.exp(-((x-a)/2)**2 / sigma**2) - np.exp(-((x+a)/2)**2 / sigma**2) 
def fm_modulated_sine_superposition_Initial(x):
         return np.sin(2 * np.pi * x * np.sin(2 * np.pi * x * 0.1)*gaussian_Initial(x) + np.sin(2 * np.pi * x * 0.03))
# Set x range and mesh size
x_range = (-10.5, 10.5)
N = 100#10000
# Provide time points for the solver
t_points = np.linspace(0, 5, N)
# Set diffusion constant
D = 1e-3

solver = SpectralSolver(fm_modulated_sine_superposition_Initial, x_range, N, t_points, D)
solver.solve_Analytically()
solver.plot_Lines(method="analytical")
solver.plot_Heatmap(method="analytical")


"""
This is the main script for solving PDE's via spectral methods.
Functions should be called from src.py and executed here.
Results are to be plotted with matplotlib.
"""
import numpy as np
import matplotlib.pyplot as plt
from newsrc import *

# Define initial condition
def gaussian_Initial(x, a=2, sigma=1):
        return np.exp(-((-x+a)/2)**2 / sigma**2)
# Set x range and mesh size
x_range = (-10, 10)
N = 1000
# Provide time points for the solver
t_points = np.linspace(0, 10, N)
# Set diffusion constant
D = 1e-5

solver = SpectralSolver(gaussian_Initial, x_range, N, t_points, D)

T, t1 = solver.solve_Numerically()

solver.plot_Animation(fps=60)

"""
The following code is used to test the spectral solver for the 1D heat equation.
"""
import numpy as np
import matplotlib.pyplot as plt
from src import *


t = np.linspace(0, 100, N)
x = np.linspace(0, 10, N)
init = gaussian(x, 5, 0.1)

solution = spectral_solver_Heat1D(init, t, debug=True)

plotAnimation(x, solution)

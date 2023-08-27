"""
The following code is used to test the spectral solver for the 1D heat equation.
"""
import numpy as np
import matplotlib.pyplot as plt
from src import *


t = np.linspace(0, 10e100, N)
x = np.linspace(0, 100, N)
# init = 237 * gaussian(x, 25, 1)
init = np.sin(2*np.pi*x*10)

solution = spectral_solver_Heat1D(init, t)

plotAnimation(x, solution, saveVideo=True)

"""
This is the main script for solving PDE's via spectral methods.
Functions should be called from src.py and executed here.
Results are to be plotted with matplotlib.
"""
import numpy as np
import matplotlib.pyplot as plt
from src import *


def ivp_1(y, t):
    """Test IVP to try out integrators.
    Has the analytical solution 2 * np.exp(-2*t) + np.exp(t)
    for y(0) = 5
    """
    return -2*y + 3 * np.exp(t)


if __name__ == "__main__":
    x = np.linspace(0, 10, N)
    t = np.linspace(0, 1, N)

    # Let's try an easier initial condition, perhaps a box function
    # init = np.zeros(N)
    # init[(N//2 - 100):(N//2 + 100)] = 1

    # Even simpler initial conditions. Just a sine wave. 
    freq = 2
    init = np.sin(2*np.pi*x*freq)

    evolve = spectral_solver_Heat1D(init, t)

    plot3D(x, t, evolve)

    
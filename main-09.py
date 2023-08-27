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

    # Even simpler initial conditions. Just a sine wave. 
    freq = 10
    # init = np.sin(2*np.pi*x*freq)
    # More difficult initial conditions. A superposition of sine waves.
    # init = np.sin(2*np.pi*x*freq) + np.sin(2*np.pi*x*freq*2) + np.sin(2*np.pi*x*freq*3)
    # Harder initial condition, perhaps a box function
    # init = np.zeros(N)
    # init[(N//2 - 100):(N//2 + 100)] = 1
    # Hard initial condition, a gaussian
    # init = gaussian(x, 5, 0.1)
    # Very hard initial conditions (not really), 
    # a superposition of gaussians modulated by a sine wave
    init = np.sin(2*np.pi*x*freq) * gaussian(x, 2, 0.2) + np.sin(2*np.pi*x*freq*2) * gaussian(x, 8, 0.3) + np.sin(2*np.pi*x*freq*3) * gaussian(x, 1, 0.1)

    hello_Rank(rank)

    # Split the temporal domain into equal chunks per rank
    t = np.linspace(0, 10, N)

    # Split the initial condition into equal chunks per rank
    init_assigned = splitter_distributer(init)

    # Solve the problem
    if rank != 0:
        solution = spectral_solver_Heat1D(init_assigned, t)
    elif rank == 0:
        solution = spectral_solver_Heat1D(init_assigned, t, debug=False)

    print(f"Solution shape {solution.shape}")

    # Gather the solutions from each rank
    if rank != 0:
        full_solution = solution_accumulator(solution)
    elif rank == 0:
        full_solution = solution_accumulator(solution, isRoot=True)
        print(full_solution.shape)

    if rank == 0:
        plot3D(x, t, full_solution)
        plt.show()

    
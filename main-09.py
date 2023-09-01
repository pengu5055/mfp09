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
import cmasher as cmr

# Define initial condition
def gaussian_Initial(x, a=5, sigma=1):
        return np.exp(-((-x+a)/2)**2 / sigma**2)
def gaussian_odd_expansion(x, a=5, sigma=1):
        return np.exp(-((x-a)/2)**2 / sigma**2) - np.exp(-((x+a)/2)**2 / sigma**2) 
def fm_modulated_sine_superposition_Initial(x):
         return np.sin(2 * np.pi * x * np.sin(2 * np.pi * x * 0.1)*gaussian_Initial(x) + np.sin(2 * np.pi * x * 0.03))
# Set x range and mesh size
x_range = (-10.5, 10.5)
N = 1000
# Provide time points for the solver
t_points = np.linspace(0, 2, N)
# Set diffusion constant
D = 1e-3

solverS = SpectralSolver(gaussian_odd_expansion, x_range, N, t_points, D)
solverC = ColocationSolver(gaussian_Initial, x_range, N, t_points, D)
sol1, t = solverS.solve_Analytically()
sol2, t2 = solverC.solve_Manually()

print(sol1)
sol1 = np.flip(sol1, axis=0)
sol2 = np.flip(sol2, axis=0)

# Plot the solution
plt.rcParams.update({'font.family': 'Verdana'})
fig, ax = plt.subplots(facecolor="#4d4c4c")

diff = np.abs(np.mean(sol2, axis=1) - np.mean(sol1, axis=1))
ax.plot(t_points, diff, c="#a8325e")



ax.set_xlabel(r"$t\>[arb. units]$")
ax.set_ylabel(r"abs. error")

ax.autoscale()
# Hardcoded limits for now. Just clear a little of the buffer due to edge divergence.
ax.set_xlim(0, 2)
ax.set_ylim(-0.1, 0.25)
# ax.set_ylim(np.min(self.solution), np.max(self.solution))
plt.suptitle("Absolute error between Spectral and Colocation mean solution", color="#dedede")
ax.set_facecolor("#bababa")
plt.grid(c="#d1d1d1", alpha=0.5)
ax.spines['bottom'].set_color("#dedede")
ax.spines['top'].set_color("#dedede")
ax.spines['right'].set_color("#dedede")
ax.spines['left'].set_color("#dedede")
ax.xaxis.label.set_color("#dedede")
ax.yaxis.label.set_color("#dedede")
ax.tick_params(axis="x", colors="#dedede")
ax.tick_params(axis="y", colors="#dedede")
ax.axhline(0, linestyle="--", color="#dedede")
#plt.subplots_adjust(right=1)
plt.show()
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
norm = plt.Normalize(vmin=-np.min(t_points), vmax=-np.max(t_points))
# norm = mpl.colors.LogNorm(vmin=0.01, vmax=12)
cm = cmr.ocean(np.linspace(0, 1, len(t_points)))
for i, sol in enumerate(sol1):
        ax.plot(solverS.x, sol, c=cm[i], alpha=0.3)
cm2 = cmr.sepia(np.linspace(0, 1, len(t_points)))
for i, sol in enumerate(sol2):
        ax.plot(solverS.x, sol, c=cm2[i], alpha=0.3)

scalar_Mappable = plt.cm.ScalarMappable(norm=norm, cmap=cmr.ocean)
scalar_Mappable2 = plt.cm.ScalarMappable(norm=norm, cmap=cmr.sepia)


ax.set_xlabel(r"$x\>[arb. units]$")
ax.set_ylabel(r"$T\>[arb. units]$")

ax.autoscale()
# Hardcoded limits for now. Just clear a little of the buffer due to edge divergence.
ax.set_xlim(0, 10)
ax.set_ylim(-0.25, 1.25)
# ax.set_ylim(np.min(self.solution), np.max(self.solution))

plt.suptitle("Comparison of Spectral and Colocation method", color="#dedede")

# Make it dark
ax.set_facecolor("#bababa")
cb = plt.colorbar(scalar_Mappable, ax=ax, label=r"$t\>[arb. units]$",
                orientation="vertical")
cb.set_label(r"$t\>[arb. units]$", color="#dedede")
cb.ax.xaxis.set_tick_params(color="#dedede")
cb.ax.yaxis.set_tick_params(color="#dedede")
ticks = -1*cb.ax.get_yticks()  # *-1 to get the correct values otherwise they are inverted
cb.ax.set_yticklabels(ticks)
cb.ax.tick_params(axis="x", colors="#dedede")
cb.ax.tick_params(axis="y", colors="#dedede")
cb2 = plt.colorbar(scalar_Mappable2, ax=ax, label=r"$t\>[arb. units]$",
                orientation="vertical")
cb2.set_label(r"$t\>[arb. units]$", color="#dedede")
cb2.ax.xaxis.set_tick_params(color="#dedede")
cb2.ax.yaxis.set_tick_params(color="#dedede")
ticks = -1*cb2.ax.get_yticks()  # *-1 to get the correct values otherwise they are inverted
cb2.ax.set_yticklabels(ticks)
cb2.ax.tick_params(axis="x", colors="#dedede")
cb2.ax.tick_params(axis="y", colors="#dedede")
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
plt.subplots_adjust(right=1)
plt.show()



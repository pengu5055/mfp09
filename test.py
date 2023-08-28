"""
The following code is used to test the spectral solver for the 1D heat equation.
"""
import numpy as np
import matplotlib.pyplot as plt
from src import *
from integrator import *


t = np.linspace(0, 10, N)
x = np.linspace(0, 10, N)

# Test MPI_Heat1D
# init = np.sin(2*np.pi*x*10)
# solution = spectral_solver_Heat1D(init, t)
# plotAnimation(x, solution, saveVideo=True)

# Test RK4_5() integrator
t = np.linspace(0, 10, 1000)
def ivp_2(t, y):
    """Test IVP to try out integrators.
    Has the analytical solution 2 * np.exp(-2*t) + np.exp(t)
    for y(0) = 5
    """
    return -2*y + 3 * np.exp(t)

eps = 1e-8
init = 5

override = {
            "divergenceTolerance": 10e12,
            "eps": 10e-6,
            "stepDownSafetyFactor": 0.8,
            "stepUpSafetyFactor": 1.05
        }

sol, err, steps = RK4_5(ivp_2, 0, init, 10, 1000, outputSteps=True, debug=False,
                         exitOnWarning=True, disableDivCheck=False)
analytical = 2 * np.exp(-2*t) + np.exp(t)

print(steps)

# TODO Potential issue identified in steps. After a few iterations, the step
# size becomes infinite and the solver fails but does not know it has failed.
# Add check for this. Investigate nan values in steps. 
# See if correct safety factors and eps are used.


plt.plot(t, analytical, label="Analytical")
plt.plot(steps, sol[0], label="RK8(9)")
plt.legend()
plt.show()

plt.plot(np.abs(analytical - sol[0]), label="Absolute error")
plt.plot(err[0], label="Estimated error")
plt.yscale("log")
plt.legend()
plt.show()

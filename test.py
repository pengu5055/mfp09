"""
The following code is used to test the spectral solver for the 1D heat equation.
"""
import numpy as np
import matplotlib.pyplot as plt
from src import *
from rk import *


t = np.linspace(0, 10, N)
x = np.linspace(0, 10, N)

# Test MPI_Heat1D
# init = np.sin(2*np.pi*x*10)
# solution = spectral_solver_Heat1D(init, t)
# plotAnimation(x, solution, saveVideo=True)

# Test RK8(9) integrator

def ivp_2(t, y):
    """Test IVP to try out integrators.
    Has the analytical solution 2 * np.exp(-2*t) + np.exp(t)
    for y(0) = 5
    """
    return -2*y + 3 * np.exp(t)

eps = 1e-8
init = np.full(N, 5)
sol, err, steps = RK8_9(ivp_2, 0, init, 10, 0.001, outputSteps=True, debug=False,
                         exitOnFail=True, disableDivCheck=False)
analytical = 2 * np.exp(-2*t) + np.exp(t)

plt.plot(t, analytical, label="Analytical")
plt.plot(t, sol[0], label="RK8(9)")
plt.legend()
plt.show()

plt.plot(np.abs(analytical - sol[0]), label="Absolute error")
plt.plot(err[0], label="Estimated error")
plt.yscale("log")
plt.legend()
plt.show()

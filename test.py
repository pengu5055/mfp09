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
eps = 1e-8
init = np.full(N, 5)
sol, err, steps = RK8_9(ivp_1, 0, init, 10, 0.001, outputSteps=True, debug=True, exitOnFail=True)
analytical = 2 * np.exp(-2*t) + np.exp(t)

print(np.abs(t[0] - steps[0, 0]) < eps) 

plt.plot(steps[0], sol[0], label="RK8(9)")
plt.plot(t, analytical, label="Analytical")
plt.legend()
plt.show()

plt.plot(np.abs(analytical - sol[0]), label="Absolute error")
plt.plot(err[0], label="Estimated error")
plt.yscale("log")
plt.legend()
plt.show()

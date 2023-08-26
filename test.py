"""
The following code is used to test the spectral solver for the 1D heat equation.
"""
import numpy as np
import matplotlib.pyplot as plt
from src import *


# Test the integrator

def ivp_1(y, t):
    """Test IVP to try out integrators.
    Has the analytical solution 2 * np.exp(-2*t) + np.exp(t)
    for y(0) = 5
    """
    return -2*y + 3 * np.exp(t)

t = np.linspace(0, 10, 1000)
init = 5
test = euler_integrator(ivp_1, init, t)
analytical = 2 * np.exp(-2*t) + np.exp(t)

plt.plot(t, test, label = "Euler")
plt.plot(t, analytical, label = "Analytical")
plt.legend()
plt.show()

import numpy as np
import matplotlib.pyplot as plt
from scipy.fftpack import fft, ifft

# Parameters
L = 10       # Domain length
T = 1        # Total time
N = 256      # Number of grid points
alpha = 0.01 # Diffusion coefficient

# Discretization
x = np.linspace(0, L, N, endpoint=False)
dx = x[1] - x[0]
dt = 0.001   # Time step
nt = int(T / dt)

# Initial condition
u0 = np.sin(np.pi * x / L)

# Initialize storage for the solution
u = u0.copy()

# Spectral wave numbers
k = 2 * np.pi * np.fft.fftfreq(N, dx)

for n in range(nt):
    u_hat = fft(u)
    u_hat_new = u_hat * np.exp(-alpha * k**2 * dt)
    u = np.real(ifft(u_hat_new))

# Plot the final solution
plt.plot(x, u)
plt.xlabel('x')
plt.ylabel('u(x, t=T)')
plt.title('1D Heat Equation')
plt.show()
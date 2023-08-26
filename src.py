"""
This is a source script for solving PDE's via spectral methods.
It should contain all the required functions to be able to solve for 
example heat diffusion in one dimension.
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from progressbar import ProgressBar
from matplotlib.animation import FuncAnimation, FFMpegWriter



# --- Simulation Parameters ---
LAMBDA = 100
RHO = 1000
SIGMA = 1
C = 200
N = 1000  # Samples
T = 1/100  # Sample frequency
a = N*T
# ---  Physical Parameters  ---
D_Al = 9.7e-5  # Thermal diffusivity of Aluminum
D_Cu = 1.1e-4  # Thermal diffusivity of Copper
# D = LAMBDA/(RHO * C)
D = D_Al
# -----------------------------

def V_euler_integrator(f, init_cond, t_points, *args):
    """Perform Euler integration for a given ODE but is vectorized.

    Parameters:
        f: The derivative function of the ODE.
        init_cond: The initial state of the system.
        time_points: A list of time points at which to compute
        the solution.

    Returns:
        values: A list of solutions corresponding to the given time points.
    """
    n = len(t_points)
    k = np.average(t_points[1]-t_points[0])

    values = np.zeros_like(init_cond)
    y0 = init_cond

    for t in tqdm(range(n - 1), total = n, desc ="Integration progress: "):
        yn = y0 + k * f(y0, t_points[t], *args)
        values = np.column_stack((values, yn))
        y0 = yn
    
    return values


def euler_integrator(f, y0, t_points, *args):
    """Perform Euler integration for a given ODE.

    Parameters:
        f: The derivative function of the ODE.
        y0: The initial state of the system.
        time_points: A list of time points at which to compute
        the solution.

    Returns:
        values: A list of solutions corresponding to the given time points.
    """
    # Step size parameter to be adjusted for accuracy
    k = 0.01
    print(f"Solving with step size {k}.")

    values = []
    y = y0

    with ProgressBar(marker='I', max_value=len(t_points)) as bar:
        for prog, t in enumerate(t_points):
            yn = y + k * f(y, t, *args)
            values.append(yn)
            y = yn
            bar.update(prog)

    # Delete first column in y
    # values = np.delete(values, 0, 1)
    
    return np.array(values)



def spectral_solver_Heat1D(init_cond, t_points, gaussian=False):
    freq = np.fft.fftfreq(N, T)
    
    # Get Fourier coefficients
    if gaussian:
        T_hat_k = np.fft.fft(init_cond) * gaussian_FFT_corr(freq)
    else:
        T_hat_k = np.fft.fft(init_cond) 

    # Get wavenumbers
    k = 2 * np.pi * np.fft.fftfreq(N, T)

    evolution = euler_integrator(T_evolve, T_hat_k, t_points, k)

    # DEBUG plot the fourier coefficients of the evolution
    plt.plot(freq, np.abs(evolution[:, 0]))
    plt.show()

    # Make evolution the same shape as output of V_euler_integrator (1000, 1000)
    # evolution = np.tile(T_hat_k, (N, 1))
    
    results = np.zeros_like(init_cond)

    # For sanity's sake lets do the inverse transform column by column
    for evolved in evolution:
        evolved = (np.fft.ifft(evolved)) # 1/N *
        results = np.column_stack((results, evolved))

    # Remove the first column of zeros
    results = np.delete(results, 0, 1)       

    # DEBUG plot the evolution and the initial condition
    plt.plot(init_cond, label="Initial condition")
    plt.plot(results[:, 0], label="Evolved")
    plt.legend()
    plt.show()

    # Return the transpose so x is the first axis and t is the second
    return results.T


def T_evolve(T_hat_k, t, k):
    """Evolve the temperature distribution in time.

    This is done by evolving the Fourier coefficients in time.

    Args:
        T_hat_k: The Fourier Coefficients.
        t: The time at which to evaluate the evolution.

    Return:
        Differential equation of the evolution.
    """
    return - D * k**2 * T_hat_k


def gaussian_FFT_corr(freq):
    return np.exp(2 * np.pi *1j * freq * T * (N/2))


def gaussian(x, a, sigma):
    return np.exp(-((-x+a)/2)**2 / sigma**2)


def plot3D(x_vector, t_vector, evolution):
    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
    x, t = np.meshgrid(x_vector, t_vector)

    surf = ax.plot_surface(x, t, evolution, cmap=cm.coolwarm,
                           linewidth=0, antialiased=False)

    fig.colorbar(surf, shrink=0.5, aspect=5)

    ax.set_xlabel(r'$x$')
    ax.set_ylabel(r'$t$')
    ax.set_zlabel(r'$T$')
    plt.show()

def plotAnimation(x, evolve):
    def update(frame):
        new_step = evolve[frame, :]
        line.set_data(x, new_step)

        return line,

    fig, ax = plt.subplots()
    line = ax.plot(x, evolve[0], label="Evolved")[0]
 
    ani = FuncAnimation(fig, update, frames=range(N), blit=True,)
    plt.rcParams['animation.ffmpeg_path'] ='C:\\Media\\ffmpeg\\bin\\ffmpeg.exe' 
    writervideo = FFMpegWriter(fps=20) 
    # ani.save("sweep.mp4", writer=writervideo)
    
    plt.show()


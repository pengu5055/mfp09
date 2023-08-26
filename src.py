"""
This is a source script for solving PDE's via spectral methods.
It should contain all the required functions to be able to solve for 
example heat diffusion in one dimension.
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from tqdm import tqdm
from matplotlib.animation import FuncAnimation, FFMpegWriter



# --- Simulation Parameters ---
LAMBDA = 100
RHO = 1000
SIGMA = 1
C = 200
N = 1000  # Samples
T = 1/1000  # Sample frequency
a = N*T
# ---  Physical Parameters  ---
D_Al = 9.7e-5  # Thermal diffusivity of Aluminum
D_Cu = 1.1e-4  # Thermal diffusivity of Copper
# -----------------------------

def euler_integrator(f, init_cond, t_points):
    """Perform Euler integration for a given ODE.

    Parameters:
        f: The derivative function of the ODE.
        init_cond: The initial state of the system.
        time_points: A list of time points at which to compute
        the solution.

    Returns:
        values: A list of solutions corresponding to the given time points.
    """
    n = len(t_points)
    k = t_points[1]-t_points[0]

    values = np.array([])
    y0 = init_cond

    for t in range(n):
        yn = y0 + k * f(y0, t_points[t])
        
        # values = np.column_stack((values, yn))
        values = np.append(values, yn)
        y0 = yn
    
    return values


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



def evolve_FFT_T(init_cond, time_points, suppressWarning=False):
    """Perform Euler integration for spectral method of solving PDE.
    Specifically in this case it solves heat diffusion.

    Args:
        init_cond: The initial state of the system.
        time_points: A list of time points at which to compute
        the solution.

    Parameters:
        suppressWarning: If True, will not warn about integration scheme stability.

    Returns:
        values: A list of solutions corresponding to the given time points.
    """

    values = np.asarray(init_cond)
    current_state = init_cond

    for id_t, t in enumerate(time_points[1:]):
        dt = t - time_points[id_t]
        step = dt * T_evol_step(current_state, t)
        current_state = current_state + step

        if not suppressWarning:
            # Stability check
            if np.abs(step.any()) < 1:
                values = np.column_stack((values, current_state))
            else:
                raise UserWarning(f"Euler unstable in step indexed: {id_t}\nValue: {np.abs(step)}")
        else:
            values = np.column_stack((values, current_state))

    
    return values


def spectral_solver_Heat1D(init_cond, t_points, gaussian=False):
    freq = np.fft.fftfreq(N, T)
    
    # Get Fourier coefficients
    if gaussian:
        T_hat_k = np.fft.fft(init_cond) * gaussian_FFT_corr(freq)
    else:
        T_hat_k = np.fft.fft(init_cond) 

    # Get wavenumbers
    k = 2 * np.pi * np.fft.fftfreq(N, T)

    # evolution = V_euler_integrator(T_evolve, T_hat_k, t_points, k)
    evolution = T_hat_k
    # Make evolution the same shape as output of V_euler_integrator (1000, 1000)
    evolution = np.tile(T_hat_k, (N, 1))
    
    results = np.zeros_like(init_cond)

    # For sanity's sake lets do the inverse transform column by column
    for evolved in evolution:
        evolved = (np.fft.ifft(evolved)) # 1/N *
        results = np.column_stack((results, evolved))

    # Remove the first column of zeros
    results = np.delete(results, 0, 1)       

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
    # D = LAMBDA/(RHO * C)
    D = D_Al
    return - D * k**2 * T_hat_k


def gaussian_FFT_corr(freq):
    return np.exp(2 * np.pi *1j * freq * T * (N/2))


def gaussian(x, a, sigma):
    return np.exp(-((-x+a)/2)**2 / sigma**2)

# Is this correct? I'm assuming that this could be the source of 
# the error. I'm not sure if the derivative is correct.
# def T_evol_step(x, t):
#     D = LAMBDA/(RHO * C)
#     f_k = k / a
#     return  D * (-4 * np.pi**2 * f_k**2) * x


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


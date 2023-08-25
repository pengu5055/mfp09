import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from scipy.integrate import RK45

# --- Simulation Parameters ---
LAMBDA = 100
RHO = 1000
A = 5
SIGMA = 1
C = 200
k = 0.01
N = 1000  # Samples
T = 1/100  # Sample frequency
a = N*T
# --- --------------------- ---

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


def V_euler_integrator(f, init_cond, t_points):
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

    values = np.array([])
    y0 = init_cond

    for t in range(n):
        yn = y0 + k * f(y0, t_points[t])
        
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


def spectral_PDE_solver(init_cond, suppressWarning=False):
    t = np.linspace(0, a, N)
    frequencies = np.fft.fftfreq(N, T)

    init_FFT = np.fft.fft(init_cond) * gaussian_FFT_corr(frequencies)

    # evolution = evolve_FFT_T(init_FFT_norm, t, suppressWarning=suppressWarning)
    evolution = RK45(T_evol_step, 0, init_FFT, 10)

    results = 1/N * np.fft.fftshift(np.fft.ifft(evolution))

    return results


def gaussian_FFT_corr(freq):
    return np.exp(2 * np.pi *1j * freq * T * (N/2))


def gaussian(x, a, sigma):
    return np.exp(-((-x+a)/2)**2 / sigma**2)


def T_init(x, t, a=0, sigma=1):
    return gaussian(x, a, sigma)


def T_evol_step(x, t):
    D = LAMBDA/(RHO * C)
    f_k = k / a
    return  D * (-4 * np.pi**2 * f_k**2) * x


def plot3D(x_vector, t_vector, evolution):
    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
    x, t = np.meshgrid(x_vector, t_vector)

    surf = ax.plot_surface(x, t, evolution, cmap=cm.coolwarm,
                           linewidth=0, antialiased=False)

    fig.colorbar(surf, shrink=0.5, aspect=5)

    ax.set_xlabel('X')
    ax.set_ylabel('t')
    ax.set_zlabel('T')
    plt.show()



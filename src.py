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
from mpi4py import MPI
import socket


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

# ---  MPI Parameters  ---
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()


# --- Functions ---
def hello_Rank(rank):
    """Says hi from the current rank and machine.

    Parameters:
        rank: The rank of the current process.
    """
    print(f"Hi! This is rank {rank} on {socket.gethostname()}. Ready to go to work...")


def splitter_distributer(points):
    """Divide and distribute given data into chunks per rank.

    Parameters:
        points: A list of data points which is to be
        divided and distributed.

    Returns:
        divided[rank]: A list of data points which is
        assigned to the current rank.
    """
    n = len(points)
    divided = []
    for i in range(size):
        divided.append(points[i * n // size:(i + 1) * n // size])

    return divided[rank]


def solution_accumulator(solution, isRoot=False):
    """
    Essentially a macro for MPI gather.

    Parameters:
        solution: The partial solution to be accumulated.

    Returns:
        accumulated_data: The accumulated data from all ranks.
    """
    accumulated_data = comm.gather(solution, root=0)

    if not isRoot:
        return accumulated_data  # comm.gather returns None on non-root ranks
    elif isRoot:
        # Transform the list of shape (5, 200, 200) into (1000, 1000)
        accumulated_data = np.concatenate(accumulated_data, axis=1)

        return accumulated_data


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
    k = 0.001
    print(f"Solving with step size {k}.")

    values = []
    y = y0

    print("Integration progress:")
    with ProgressBar(max_value=len(t_points)) as bar:
        for prog, t in enumerate(t_points):
            yn = y + k * f(y, t, *args)
            values.append(yn)
            y = yn
            bar.update(prog)
    
    return np.array(values)



def spectral_solver_Heat1D(init_cond, t_points, gaussian=False, debug=False, numericalSolve=True):
    """
    Solve the 1D heat equation using spectral methods.

    Args:
        init_cond: The initial condition of the system.
        t_points: A list of time points at which to compute
            the solution.

    Parameters:
        gaussian: Whether or not to use a gaussian correction for the FFT
            (semi-deprecated).
        debug: Whether or not to plot debug plots.
        numericalSolve: Whether or not to use the numerical solver. If False
            the analytical solution is used for evolution of the system.

    Raises:
        ValueError: If the rank is not 0 and debug is True.

    Returns:
        results: A list of solutions corresponding to the given time points.
    """
    if rank > 0 and debug:
        raise ValueError(f"Debugging is not supported on rank {rank}. Only rank 0 can plot.")
    # Number of samples since variable per rank
    n = len(init_cond)
    freq = np.fft.fftfreq(n, T)
    
    # Get Fourier coefficients
    if gaussian:
        T_hat_k = np.fft.fft(init_cond) * gaussian_FFT_corr(freq)
    else:
        T_hat_k = np.fft.fft(init_cond) 

    # Get wavenumbers
    k = 2 * np.pi * np.fft.fftfreq(n, T)
    
    if debug:
        # DEBUG plot the initial condition
        plt.plot(init_cond)
        plt.title("Initial condition")
        plt.show()

        # DEBUG plot the fourier coefficients of the initial condition
        plt.plot(freq, np.abs(T_hat_k))
        plt.title("Fourier coefficients before the evolution")
        plt.show()

    if numericalSolve:
        evolution = euler_integrator(T_evolve, T_hat_k, t_points, k)
    else:
        evolution = T_hat_k

    if debug:
        # DEBUG plot the fourier coefficients of the evolution
        plt.plot(freq, np.abs(evolution[0]))
        plt.title("Fourier coefficients of the evolution")
        plt.show()
    
    results = np.zeros_like(init_cond)

    # For sanity's sake lets do the inverse transform column by column
    print("Inverse Fourier transform progress:")
    with ProgressBar(max_value=len(t_points)) as bar:
        for prog, evolved in enumerate(evolution):
            evolved = (np.fft.ifft(evolved))
            results = np.column_stack((results, evolved))
            bar.update(prog)

    # Remove the first column of zeros
    results = np.delete(results, 0, 1)      

    if debug:
        # DEBUG plot the evolution and the initial condition
        plt.plot(init_cond, label="Initial condition")
        plt.plot(results[:, 0], label="Evolved")
        plt.title("Replication of the initial condition")
        plt.legend()
        plt.show()

    # Return the transpose so x is the first axis and t is the second
    return results.T

# TODO: WARNING an error has been identified in the way MPI_Heat1D solves.
# There is a discrepancy between the single threaded and multi threaded
# solutions. The single threaded solution is correct. The multi threaded
# solution is not. The error is likely in the way the solution is gathered
# from the different ranks. The error is likely in the solution_accumulator
# function.
def MPI_Heat1D(init_cond, t_points, gaussian=False, debug=False, numericalSolve=True):
    """
    Higher level function that contains all necessary function calls
    to solve the 1D heat equation using spectral method with MPI.

    Args:
        init_cond: The initial condition of the system.
        t_points: A list of time points at which to compute
            the solution.

    Parameters:
        gaussian: Whether or not to use a gaussian correction for the FFT
            (semi-deprecated).
        debug: Whether or not to plot debug plots. Only supported on rank 0.
        numericalSolve: Whether or not to use the numerical solver. If False
            the analytical solution is used for evolution of the system.

    Raises:
        ValueError: If the rank is not 0 and debug is True.

    Returns:
        full_solution: Full solution of the system.
    """
    # Split the initial condition into equal chunks per rank
    init_assigned = splitter_distributer(init_cond)

    # Solve the problem
    if rank != 0:
        solution = spectral_solver_Heat1D(init_assigned, t_points)
    elif rank == 0:
        solution = spectral_solver_Heat1D(init_assigned, t_points, debug=False)

    # Gather the solutions from each rank
    if rank != 0:
        full_solution = solution_accumulator(solution)
    elif rank == 0:
        full_solution = solution_accumulator(solution, isRoot=True)
    
    return full_solution


def T_analytical_evolve(T_hat_k, t, k):
    """Evolve the temperature distribution in time.

    This is done by evolving the Fourier coefficients in time
    using the analytical solution.

    Args:
        T_hat_k: The Fourier Coefficients.
        t: The time at which to evaluate the evolution.

    Return:
        Analytical solution of the evolution.
    """
    return np.exp(-D * k**2 * t) * T_hat_k


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


def ivp_1(y, t):
    """Test IVP to try out integrators.
    Has the analytical solution 2 * np.exp(-2*t) + np.exp(t)
    for y(0) = 5
    """
    return -2*y + 3 * np.exp(t)


def plot3D(x_vector, t_vector, evolution):
    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
    x, t = np.meshgrid(x_vector, t_vector)

    surf = ax.plot_surface(x, t, evolution, cmap=cm.coolwarm,
                           linewidth=0, antialiased=False)

    fig.colorbar(surf, pad=0.1)
    fig.tight_layout()
    ax.set_title("Time evolution of the temperature distribution")
    ax.set_xlabel(r'$x\>[\mathrm{arbitrary\>units}]$')
    ax.set_ylabel(r'$t\>[\mathrm{arbitrary\>units}]$')
    ax.set_zlabel(r'$T\>[\mathrm{arbitrary\>units}]$')
    ax.grid(color="#c1d0d4", alpha=0.3)
    fig.subplots_adjust(top=0.9)
    plt.show()

def plotAnimation(x, solution, saveVideo=False, videoName="video.mp4", fps=20):
    """Plot the time evolution of the temperature distribution.

    Arguments:
        x: The x-axis.
        solution: The solution to be plotted.

    Parameters:
        saveVideo: Whether or not to save the animation as a video.
        fps: Frames per second of the video.
    """
    def update(frame):
        new_step = solution[frame, :]
        line.set_data(x, new_step)

        return line,

    fig, ax = plt.subplots()
    line = ax.plot(x, solution[0], label="Evolved", color="#f79ec6")[0]
    
    ani = FuncAnimation(fig, update, frames=range(N), blit=True, interval=1/fps)
    plt.rcParams['animation.ffmpeg_path'] ='C:\\Media\\ffmpeg\\bin\\ffmpeg.exe' 
    writervideo = FFMpegWriter(fps=20) 

    ax.grid(color="#c1d0d4", alpha=0.3)
    ax.set_xlabel(r'$x$')
    ax.set_ylabel(r'$T$')
    ax.set_title("Time evolution of the temperature distribution")
    ax.legend()
    if saveVideo:
        ani.save(videoName, writer=writervideo, fps=fps)
    
    plt.show()

"""
This is a refresh of the old src.py file. This file will be used to
implement the spectral solver for the 1D heat equation. The spectral solver
will be implemented in a class called SpectralSolver. The class will have
methods for solving the heat equation and plotting the solution.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import fftpack
from scipy.integrate import odeint, solve_ivp
from typing import Callable, Tuple, Iterable

# Initialize class with grid size N and time steps t
# TODO: Add option to use MPI
# Store for later = ['#fff7f3','#fde0dd','#fcc5c0','#fa9fb5','#f768a1','#dd3497','#ae017e','#7a0177','#49006a']
class SpectralSolver:
    def __init__(self,
                 initial_condition: Callable[[float], float],
                 x_range: Tuple[float, float],
                 N: int,
                 t_points: Iterable[float],
                 D: float,
                ):
        """
        Initialize the solver with a grid size N and time steps t.
        """
        self.x_range = x_range
        self.N = N
        self.t_points = t_points
        self.x = np.linspace(x_range[0], x_range[1], N)
        self.dx = self.x[1] - self.x[0]
        self.dt = self.t_points[1] - self.t_points[0]
        self.freq = fftpack.fftshift(fftpack.fftfreq(N, d=self.dx))
        self.k = 2 * np.pi * self.freq
        self.initial_condition = initial_condition
        self.T_hat_k = fftpack.fftshift(fftpack.fft(initial_condition(self.x)))
        self.D = D
    
    def _internal_function_timer(func: Callable):
        def wrapper(*args, **kwargs):
            import time
            start = time.time()
            result = func(*args, **kwargs)
            end = time.time()
            print(f"Function {func.__name__} took {end - start} seconds to run.")
            return result, start - end
        return wrapper

    def _analytical_step(self, previous_coef: Iterable[float], t: Iterable[float] | float):
        """
        Evolve the temperature distribution in time.

        This is done by evolving the Fourier coefficients in time
        using the analytical solution. This function is inherently
        different from _evolve_step because it is vectorized. Thus
        there is no need for a loop over the wave numbers k.

        Parameters:
            previous_coef: The Fourier coefficients at which the previous
                time step was evaluated.
            t: The time or times at which to evaluate the evolution.

        Return:
            Analytical solution of the evolution.
        """
        return np.exp(-self.D * self.k**2 * t) * previous_coef
    
    def _evolve_step(self, coef: float, t: float, i_k: int):
        """
        Evolve the temperature distribution in time.

        This is done by evolving the Fourier coefficients in time.

        Parameters:
            coef: The Fourier coefficient at which to evaluate the evolution.
            t: The time at which to evaluate the evolution. This is essentially
                a dummy variable for odeint since there is no explicit time
                dependence.
            i_k: The index of the wave number k.

        Return:
            Differential equation of the evolution.
        """
        # print(f"Coef = {coef}, t = {t}, i_k = {i_k}, fixed = {self.T_hat_k[i_k]}")
        return - self.D * self.k[i_k]**2 * self.T_hat_k[i_k]
    
    def _evolve_step_vectorized(self, coef: float, t: float):
        """
        Evolve the temperature distribution in time.

        This is done by evolving the Fourier coefficients in time.

        Args:
            coef: The Fourier coefficient at which to evaluate the evolution.
            t: The time at which to evaluate the evolution. This is essentially
                a dummy variable for odeint since there is no explicit time
                dependence.

        Return:
            Differential equation of the evolution.
        """
        # print(f"Coef = {coef}, t = {t}, i_k = {i_k}, fixed = {self.T_hat_k[i_k]}")
        return - self.D * self.k**2 * self.T_hat_k

    def _gaussian_FFT_corr(self, freq: float):
        return np.exp(2 * np.pi *1j * freq * self.t_points * (self.N/2))
    
    def _gaussian(self, x: float, a: float, sigma: float):
        return np.exp(-((-x+a)/2)**2 / sigma**2)
    
    def _dirichlet_boundary(self, ya, yb):
        """
        Dirichlet boundary conditions. 
        """
        return np.zeros((2,))

    def solve_Analytically(self):
        """
        Solve the heat equation using the spectral solver and _analytical_step.
        """
        self.solution = np.zeros((len(self.t_points), self.N))
        self.solution[0] = self.initial_condition(self.x)
        # Absolutely need previous coefficients to be able to iterate and evolve.
        self.previous = np.copy(self.T_hat_k)
        for i, t in enumerate(self.t_points[1:]):
            self.previous = self._analytical_step(self.previous, t)        
            self.solution[i+1] = fftpack.ifft(fftpack.ifftshift(self.previous)).real
        
        return self.solution

    def solve_Numerically(self):
        """
        Solve the heat equation using the spectral solver _evolve_step.
        """
        self.solution = np.zeros((len(self.t_points), self.N))
        self.solution[0] = self.initial_condition(self.x)
        self.previous = np.copy(self.T_hat_k)

        for i, t in enumerate(self.t_points[1:]):
            self.previous = solve_ivp(self._evolve_step_vectorized, (self.t_points[0], self.t_points[-1]), self.previous,
                                        method="RK45", t_eval=[t], vectorized=True).y.flatten()
            self.solution[i+1] = fftpack.ifft(fftpack.ifftshift(self.previous)).real

        return self.solution
    
    def plot_initial_FFT(self):
        """
        Plot the initial condition in Fourier space.
        """
        # Plot real and imaginary parts of the initial condition
        fig, ax = plt.subplots(2, 1, figsize=(6, 8))
        ax[0].plot(self.freq, np.real(self.T_hat_k))
        ax[0].set_xlabel("frequency")
        ax[0].set_ylabel("Re(T_hat_k)")

        ax[1].plot(self.freq, np.imag(self.T_hat_k))
        ax[1].set_xlabel("frequency")
        ax[1].set_ylabel("Im(T_hat_k)")
        
        plt.suptitle("Initial condition in Fourier space")
        plt.show()
    
    
"""
This is a refresh of the old src.py file. This file will be used to
implement the spectral solver for the 1D heat equation. The spectral solver
will be implemented in a class called SpectralSolver. The class will have
methods for solving the heat equation and plotting the solution.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import fftpack
from scipy.integrate import odeint
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

    def _analytical_step(self, t: float, x: float):
        """
        Evolve the temperature distribution in time.

        This is done by evolving the Fourier coefficients in time
        using the analytical solution.

        Args:
            t: The time at which to evaluate the evolution.
            x: Dummy variable for odeint.

        Return:
            Analytical solution of the evolution.
        """
        return np.exp(-self.D * self.k**2 * t) * self.T_hat_k
    
    def _evolve_step(self, t: float, x: float):
        """
        Evolve the temperature distribution in time.

        This is done by evolving the Fourier coefficients in time.

        Args:
            t: The time at which to evaluate the evolution.
            x: Dummy variable for odeint.

        Return:
            Differential equation of the evolution.
        """
        return - self.D * self.k**2 * t
    
    def _gaussian_FFT_corr(self, freq: float):
        return np.exp(2 * np.pi *1j * freq * self.t_points * (self.N/2))
    
    def _gaussian(self, x: float, a: float, sigma: float):
        return np.exp(-((-x+a)/2)**2 / sigma**2)
    
    def solve_Analytically(self):
        """
        Solve the heat equation using the spectral solver and _analytical_step.
        """
        self.solution = np.zeros((len(self.t_points), self.N))
        self.solution[0] = self.initial_condition(self.x)
        for i, t in enumerate(self.t_points[1:]):
            # Second argument is actually a dummy variable for odeint
            self.T_hat_k = self._analytical_step(t, 0) 
            self.solution[i+1] = fftpack.ifft(fftpack.ifftshift(self.T_hat_k)).real
        
        return self.solution

    def solve_Numerically(self):
        """
        Solve the heat equation using the spectral solver _evolve_step.
        """
        self.solution = np.zeros((len(self.t_points), self.N))
        self.solution[0] = self.initial_condition(self.x)
        for i, t in enumerate(self.t_points[1:]):
            self.T_hat_k = odeint(self._evolve_step, np.real(self.T_hat_k), self.t_points, tfirst=True)[0]
            + 1j * odeint(self._evolve_step, np.imag(self.T_hat_k), self.t_points, tfirst=True)[0]
            self.solution[i+1] = fftpack.ifft(fftpack.ifftshift(self.T_hat_k)).real
        
        return self.solution
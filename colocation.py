"""
This file will contain the class for the colocation method. 
The colocation method is a numerical method for solving PDE's.
It is a spectral method, meaning that it uses a Fourier basis to approximate the solution.
We're going to need to solve matrix equations, so we'll use numpy.
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from scipy import fftpack
from typing import Tuple, Callable, List, Iterable
from scipy.interpolate import BSpline, splrep
from scipy.sparse import diags
from scipy.linalg import solve_banded
import seaborn as sns
import cmasher as cmr

class ColocationSolver:
    def __init__(self,
                 initial_condition: Callable[[float], float],
                 x_range: Tuple[float, float],
                 N: int,
                 t_points: Iterable[float],
                 D: float,
                 ) -> None:
        """
        Initialize the solver with a grid size N in the ranges set by x_range.
        Set the initial condition and the diffusion constant D. Prepare to solve
        the PDE at the time points given by t_points.
        """
        self.initial_condition = initial_condition
        self.x_range = x_range
        self.N = N
        self.t_points = t_points
        self.D = D
        self.x = np.linspace(self.x_range[0], self.x_range[1], N)
        self.dx = self.x[1] - self.x[0]
        self.dt = self.t_points[1] - self.t_points[0]
    
    def _internal_function_timer(func: Callable):
        def wrapper(*args, **kwargs):
            import time
            start = time.time()
            result = func(*args, **kwargs)
            end = time.time()
            print(f"Function {func.__name__} took {end - start} seconds to run.")
            return result, start - end
        return wrapper
    
    @_internal_function_timer
    def solve_Manually(self) -> Tuple[np.ndarray, np.ndarray]:
        pass
    
    @_internal_function_timer
    def solve_Properly(self) -> np.ndarray:
        """
        This method of solving the PDE uses bulit-in functions from scipy.
        """
        # Lets first try to fit a BSpline to the initial condition
        # using the scipy.interpolate.BSpline class and the splrep function.

        self.solution = np.zeros((len(self.t_points), self.N))
        f_init_vec = np.zeros(self.N)
        f_init_vec = self.initial_condition(self.x)
        c_init_vec = np.zeros(self.N)

        # Create tridiagonal matricies of coefficients
        # A = diags([1, 4, 1], [-1, 0, 1], shape=(self.N, self.N)).toarray()
        diagonals = [[1] * (self.N - 1), [4] * self.N, [1] * (self.N - 1)]
        A = np.diag(diagonals[0], -1) + np.diag(diagonals[1], 0) + np.diag(diagonals[2], 1)

        # Convert the tridiagonal matrix to a dense banded matrix
        A_banded = np.zeros((3, self.N))
        A_banded[0] = np.pad(diagonals[0], (1, 0), mode="constant")
        A_banded[1] = np.pad(diagonals[1], (0, 0), mode="constant")
        A_banded[2] = np.pad(diagonals[2], (0, 1), mode="constant")

        B = ((6*self.D)/self.dx**2) * diags([1, -2, 1], [-1, 0, 1], shape=(self.N, self.N)).toarray()

        # For initial condition it has to hold that
        # A * c_vec = f_init_vec
        eps = 1e-6

        c_init_vec = solve_banded((1, 1), A_banded, f_init_vec)
        c_init_vec2 = self._do_TDMA(A, f_init_vec)



        # DEBUG: plots to visualize happening
        t, c, k = splrep(self.x, self.solution[0], s=0, k=3)
        print(c)
        BS = BSpline(t, c, k, extrapolate=False)

        # plt.scatter(self.x, self.solution[0], label="Initial condition", c="red")
        # plt.plot(self.x, BS(self.x), label="BSpline", c="blue")
        # plt.show()

        return self.solution

    def BSpline(self):
        pass

    def _do_TDMA(self, A, f):
        """
        Do the Thomas algorithm for a tridiagonal matrix A and a vector f.
        """
        alpha = np.zeros(self.N)
        beta = np.zeros(self.N)
        c_new = np.zeros(self.N)

        alpha[1] = A[0, 1] / A[0, 0]
        beta[1] = f[0] / A[0, 0]

        for i in range(1, self.N-1):
            den = (A[i, i] - A[i, i-1] * alpha[i])
            alpha[i+1] = A[i, i+1] / den
            beta[i+1] = (f[i] - A[i, i-1] * beta[i]) / den
        
        c_new[self.N-1] = (f[self.N-1] - A[self.N-1, self.N-2] * beta[self.N-1]) / (A[self.N-1, self.N-1] - A[self.N-1, self.N-2] * alpha[self.N-1])
        for i in range(self.N-2, -1, -1):
            c_new[i] = beta[i+1] - alpha[i+1] * c_new[i+1]
        
        return c_new

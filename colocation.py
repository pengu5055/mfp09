"""
This file will contain the class for the colocation method. 
The colocation method is a numerical method for solving PDE's.
It is a spectral method, meaning that it uses a Fourier basis to approximate the solution.
We're going to need to solve matrix equations, so we'll use numpy.
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter
import matplotlib as mpl
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
    
    def _override_x(self, x):
        """
        Override the x grid with a new one. This is due to the fact that MPI
        will distribute the grid points to the nodes. The nodes will need to
        know the grid points they are responsible for but the wrapper class
        will take the already initialized solver as an argument. Therefore,
        the nodes will need to override the x grid.

        Parameters:
            x: The new x grid.
        """
        # TODO: Check if x is of correct size
        self.x = x
        self.x_range = (x[0], x[-1])
        self.dx = self.x[1] - self.x[0]
        self.N = len(x)

    @_internal_function_timer
    def solve_Manually(self) -> np.ndarray:
        """
        This method of solving the PDE uses the colocation method but 
        is implemented manually.

        It will gather all required data from the object itself and solve
        the PDE. The solution is stored in the object itself but is also
        returned as well as the time it took to solve the PDE.
        """
        self.solution_m = np.zeros((len(self.t_points), self.N))
        f_init_vec = np.zeros(self.N)
        f_init_vec = self.initial_condition(self.x)
        c_init_vec = np.zeros(self.N)

        # Create tridiagonal matricies of coefficients
        A = diags([1, 4, 1], [-1, 0, 1], shape=(self.N, self.N)).toarray()
        B = ((6*self.D)/self.dx**2) * diags([1, -2, 1], [-1, 0, 1], shape=(self.N, self.N)).toarray()

        A_inv = np.linalg.inv(A)

        # 1. Solve: A * c_init_vec = f_init_vec
        c_init_vec = self._do_TDMA(A, f_init_vec)

        # 2. Solve: A * dc/dt = B * c -> dc/dt = (c[i+1] - c[i]) / dt  <= Backward Euler
        c = np.zeros((len(self.t_points), self.N))
        c[0] = c_init_vec
        for i in range(1, len(self.t_points)):
            c[i] = A_inv*B*self.dt @ c[i-1] + c[i-1]
        
        # 3. Solve: A * c = f
        # self.solution_m = A @ c
        self.solution_m = (A @ c.T).T

        return self.solution_m
    
    @_internal_function_timer
    def solve_Properly(self) -> np.ndarray:
        """
        This method of solving the PDE uses bulit-in functions from scipy.

        It will gather all required data from the object itself and solve
        the PDE. The solution is stored in the object itself but is also
        returned as well as the time it took to solve the PDE.
        """
        self.solution = np.zeros((len(self.t_points), self.N))
        f_init_vec = np.zeros(self.N)
        f_init_vec = self.initial_condition(self.x)
        c_init_vec = np.zeros(self.N)

        # Create tridiagonal matricies of coefficients
        # A = diags([1, 4, 1], [-1, 0, 1], shape=(self.N, self.N)).toarray()
        diagonals = [[1] * (self.N - 1), [4] * self.N, [1] * (self.N - 1)]
        A = diags([1, 4, 1], [-1, 0, 1], shape=(self.N, self.N)).toarray()

        # Convert the tridiagonal matrix to a dense banded matrix
        A_banded = np.zeros((3, self.N))
        A_banded[0] = np.pad(diagonals[0], (1, 0), mode="constant")
        A_banded[1] = np.pad(diagonals[1], (0, 0), mode="constant")
        A_banded[2] = np.pad(diagonals[2], (0, 1), mode="constant")
        A_inv = np.linalg.inv(A)

        # (np.round(np.linalg.inv(A_inv))) <- Rounding may help stop propagation of errors due to 
        # floating point precision (non zero values in the inverse matrix)

        B = ((6*self.D)/self.dx**2) * diags([1, -2, 1], [-1, 0, 1], shape=(self.N, self.N)).toarray()

        # 1. Solve: A * c_init_vec = f_init_vec
        c_init_vec = solve_banded((1, 1), A_banded, f_init_vec)
        
        # 2. Solve: A * dc/dt = B * c -> dc/dt = (c[i+1] - c[i]) / dt  <= Forward Euler
        c = np.zeros((len(self.t_points), self.N))
        c[0] = c_init_vec
        for i in range(1, len(self.t_points)):
            c[i] = A_inv*B*self.dt @ c[i-1] + c[i-1]

        # 3. Solve: A * c = f
        self.solution = (A @ c.T).T

        return self.solution

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

    def plot_Animation(self, x: Iterable[float] | None = None, 
                       solution: Iterable[float] | None = None,
                       method: str = "proper",
                       color: str = "black",
                       saveVideo: bool = False, 
                       videoName: str = "animation.mp4", 
                       fps: int = 20,
                       plotInitial: bool = False,
                    ):
        """
        Plot the solution as an animation. Will try to get computed solution
        from solver itself. Can override if x, solution are not 'None'.

        The solution is plotted as an animation. The animation can be saved.

        Arguments:
            x: The grid points at which the solution is evaluated.
            solution: The solution to the heat equation.
            method: The method used to solve the heat equation. Can be either
                "proper" or "manual".
            color: The color of the plotted solution.
            saveVideo: Whether or not to save the animation as a video.
            videoName: The name of the video to save.
            fps: The frames per second of the video.
            plotInitial: Whether or not to plot the initial condition.
        
        Return:
            None
        """
        if np.all(x == None) and np.all(solution == None):
            try:
                x = self.x
                if method == "proper":
                    solution = self.solution
                elif method == "manual":
                    solution = self.solution_m
                else:
                    raise ValueError("Method must be either 'analytical' or 'numerical'!")
            except NameError:
                print("Call one of solving methods before trying to plot or supply data as function parameters!")


        def update(frame):
            line.set_ydata(solution[frame])
            line.set_color(color)
            L.get_texts()[0].set_text(f"t = {frame/fps:.2f} s")
            return line,

        fig, ax = plt.subplots()
        line, = ax.plot(x, solution[0], label="t = 0 s", c=color)
        if plotInitial:
            plt.plot(self.x, self.initial_condition(self.x), label="Initial condition", c="red", alpha=0.2)

        ax.set_xlabel("x")
        ax.set_ylabel("T")
        # ax.set_ylim(-1.5, 1.5)
        # ax.set_xlim(0, 1)
        plt.suptitle("Solution of the heat equation")
        L = plt.legend()

        ani = FuncAnimation(fig, update, frames=range(len(self.t_points)), blit=False, interval=1000/fps)
        plt.rcParams['animation.ffmpeg_path'] ='C:\\Media\\ffmpeg\\bin\\ffmpeg.exe' 
        if saveVideo:
            writervideo = FFMpegWriter(fps=fps)
            ani.save(videoName, writer=writervideo)

        plt.show()

    def plot_Lines(self, method: str = "proper"):
        """
        Plot the solution as lines.
        """
        plt.rcParams.update({'font.family': 'Verdana'})
        fig, ax = plt.subplots(facecolor="#4d4c4c")
        # for solution in self.solution:
        #     segments.append(np.asarray([np.column_stack((x, y)) for x, y in zip(self.x, solution)]).ravel())

        if method == "proper":
            data = np.copy(self.solution)
            data = np.flip(data, axis=0)
        elif method == "manual":
            data = np.copy(self.solution_m)
            data = np.flip(data, axis=0)
        else:
            raise ValueError("Method must be either 'analytical' or 'numerical'!") 

        x = self.x
        norm = plt.Normalize(vmin=-np.min(self.t_points), vmax=-np.max(self.t_points))
        cm = cmr.flamingo(np.linspace(0, 1, len(self.t_points)))
        for i, sol in enumerate(data):
            ax.plot(x, sol, c=cm[i], alpha=0.8)

        scalar_Mappable = plt.cm.ScalarMappable(norm=norm, cmap=cmr.flamingo)

        ax.set_xlabel(r"$x\>[arb. units]$")
        ax.set_ylabel(r"$T\>[arb. units]$")

        ax.autoscale()
        # Hardcoded limits for now. Just clear a little of the buffer due to edge divergence.
        ax.set_xlim(0, 10)
        ax.set_ylim(-0.25, 1.25)
        # ax.set_ylim(np.min(self.solution), np.max(self.solution))

        plt.suptitle("Evolution of the solution solved by Library Colocation method", color="#dedede")

        # Make it dark
        ax.set_facecolor("#bababa")
        cb = plt.colorbar(scalar_Mappable, ax=ax, label=r"$t\>[arb. units]$",
                      orientation="vertical")
        cb.set_label(r"$t\>[arb. units]$", color="#dedede")
        cb.ax.xaxis.set_tick_params(color="#dedede")
        cb.ax.yaxis.set_tick_params(color="#dedede")
        cb.ax.tick_params(axis="x", colors="#dedede")
        cb.ax.tick_params(axis="y", colors="#dedede")
        plt.grid(c="#d1d1d1", alpha=0.5)
        ax.spines['bottom'].set_color("#dedede")
        ax.spines['top'].set_color("#dedede")
        ax.spines['right'].set_color("#dedede")
        ax.spines['left'].set_color("#dedede")
        ax.xaxis.label.set_color("#dedede")
        ax.yaxis.label.set_color("#dedede")
        ax.tick_params(axis="x", colors="#dedede")
        ax.tick_params(axis="y", colors="#dedede")
        ax.axhline(0, linestyle="--", color="#dedede")
        plt.show()
    
    def plot_Heatmap(self, method: str = "proper"):
        """
        Plot the solution as a heatmap.
        """
        plt.rcParams.update({'font.family': 'Verdana'})
        fig, ax = plt.subplots(facecolor="#4d4c4c")
        if method == "proper":
            data = np.copy(self.solution)
            data = np.flip(data, axis=0)
        elif method == "manual":
            data = np.copy(self.solution_m)
            data = np.flip(data, axis=0)
        else:
            raise ValueError("Method must be either 'analytical' or 'numerical'!")

        plt.imshow(data, cmap=cmr.flamingo, aspect="auto", vmin=np.min(data), vmax=np.max(data),
                   extent=[self.x_range[0], self.x_range[1], self.t_points[0], self.t_points[-1]])

        x_ticks = np.linspace(self.x_range[0],self.x_range[1], 10)
        plt.xticks(x_ticks)
        y_ticks = np.linspace(self.t_points[0], self.t_points[-1], 10)
        plt.yticks(y_ticks)

        plt.xlabel(r"$x\>[arb. units]$")
        plt.ylabel(r"$t\>[arb. units]$")
        plt.suptitle("Heatmap of the solution solved by Library Colocation method", color="#dedede")
        norm = mpl.colors.Normalize(vmin=np.min(data), vmax=np.max(data))
        scalar_Mappable = plt.cm.ScalarMappable(norm=norm, cmap=cmr.flamingo)
        cb = plt.colorbar(scalar_Mappable, ax=ax, label=r"$T\>[arb. units]$",
                      orientation="vertical")
        cb.set_label(r"$t\>[arb. units]$", color="#dedede")
        cb.ax.xaxis.set_tick_params(color="#dedede")
        cb.ax.yaxis.set_tick_params(color="#dedede")
        cb.ax.tick_params(axis="x", colors="#dedede")
        cb.ax.tick_params(axis="y", colors="#dedede")
        ax.spines['bottom'].set_color("#dedede")
        ax.spines['top'].set_color("#dedede")
        ax.spines['right'].set_color("#dedede")
        ax.spines['left'].set_color("#dedede")
        ax.xaxis.label.set_color("#dedede")
        ax.yaxis.label.set_color("#dedede")
        ax.tick_params(axis="x", colors="#dedede")
        ax.tick_params(axis="y", colors="#dedede")

        plt.show()
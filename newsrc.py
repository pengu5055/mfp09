"""
This is a refresh of the old src.py file. This file will be used to
implement the spectral solver for the 1D heat equation. The spectral solver
will be implemented in a class called SpectralSolver. The class will have
methods for solving the heat equation and plotting the solution.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter
from matplotlib.patches import RegularPolygon
from matplotlib.collections import RegularPolyCollection
import seaborn as sns
from scipy import fftpack
from scipy.integrate import solve_ivp
from typing import Callable, Tuple, Iterable
import cmasher as cmr
import pandas as pd

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
    
    
    def _dirichlet_boundary(self, ya, yb):
        """
        Dirichlet boundary conditions. 
        """
        return np.zeros((2,))

    @_internal_function_timer
    def solve_Analytically(self):
        """
        Solve the heat equation using the spectral solver and _analytical_step.
        """
        self.solution_a = np.zeros((len(self.t_points), self.N))
        self.solution_a[0] = self.initial_condition(self.x)
        # Absolutely need previous coefficients to be able to iterate and evolve.
        self.previous = np.copy(self.T_hat_k)
        for i, t in enumerate(self.t_points[1:]):
            self.previous = self._analytical_step(self.previous, t)        
            self.solution_a[i+1] = fftpack.ifft(fftpack.ifftshift(self.previous)).real
        
        return self.solution_a

    @_internal_function_timer
    def solve_Numerically(self):
        """
        Solve the heat equation using the spectral solver _evolve_step.
        """
        self.solution = np.zeros((len(self.t_points), self.N))
        self.solution[0] = self.initial_condition(self.x)
        # Absolutely need previous coefficients to be able to iterate and evolve.
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
    
    def plot_Animation(self, x: Iterable[float] | None = None, 
                       solution: Iterable[float] | None = None,
                       method: str = "analytical",
                       color: str = "black",
                       saveVideo: bool = False, 
                       videoName: str = "animation.mp4", 
                       fps: int = 20,
                    ):
        """
        Plot the solution as an animation. Will try to get computed solution
        from solver itself. Can override if x, solution are not 'None'.

        The solution is plotted as an animation. The animation can be saved.

        Arguments:
            x: The grid points at which the solution is evaluated.
            solution: The solution to the heat equation.
            method: The method used to solve the heat equation. Can be either
                "analytical" or "numerical".
            color: The color of the plotted solution.
            saveVideo: Whether or not to save the animation as a video.
            videoName: The name of the video to save.
            fps: The frames per second of the video.
        
        Return:
            None
        """
        if np.all(x == None) and np.all(solution == None):
            try:
                x = self.x
                if method == "analytical":
                    solution = self.solution_a
                elif method == "numerical":
                    solution = self.solution
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
    
    def plot_Animation_both(self, color1 = "black",
                            color2 = "purple",
                            saveVideo: bool = False,
                            videoName: str = "animation.mp4",
                            fps: int = 20):
        """
        Plot the solution as an animation. Will try to get computed solution
        from solver itself. Can override if x, solution are not 'None'.
        

        The solution is plotted as an animation. The animation can be saved.
        """
        try:
            x = self.x
            solution = self.solution
            solution_a = self.solution_a

        except NameError: 
            print("Call BOTH of solving methods before trying to plot!")        

        def update(frame):
            line1.set_ydata(solution[frame])
            line1.set_label(f"Num t = {frame/fps:.2f} s")
            line1.set_color(color1)
            line2.set_ydata(solution_a[frame])
            line2.set_label(f"Ana t = {frame/fps:.2f} s")
            line2.set_color(color2)
            return line1,

        fig, ax = plt.subplots()
        line1, = ax.plot(x, solution[0], label="Num t = 0 s", c=color1)
        line2, = ax.plot(x, solution_a[0], label="Ana t = 0 s", c=color2)
        ax.set_xlabel("x")
        ax.set_ylabel("T")
        # ax.set_ylim(-1.5, 1.5)
        # ax.set_xlim(0, 1)
        plt.suptitle("Comparison of solutions to the heat equation")
        plt.legend()

        ani = FuncAnimation(fig, update, frames=range(len(self.t_points)), blit=True, interval=1000/fps)
        plt.rcParams['animation.ffmpeg_path'] ='C:\\Media\\ffmpeg\\bin\\ffmpeg.exe' 
        if saveVideo:
            writervideo = FFMpegWriter(fps=fps)
            ani.save(videoName, writer=writervideo)

        plt.show()

    def plot_Heatmap(self, method: str = "analytical"):
        """
        Plot the solution as a heatmap.
        """
        plt.rcParams.update({'font.family': 'Verdana'})
        fig, ax = plt.subplots()
        if method == "analytical":
            data = np.copy(self.solution_a)
            data = np.flip(data, axis=0)
        elif method == "numerical":
            data = np.copy(self.solution)
            data = np.flip(data, axis=0)
        else:
            raise ValueError("Method must be either 'analytical' or 'numerical'!")

        plt.imshow(data, cmap=cmr.flamingo, aspect="auto", vmin=np.min(data), vmax=np.max(data),
                   extent=[self.x_range[0], self.x_range[1], self.t_points[0], self.t_points[-1]])

        x_ticks = np.linspace(self.x_range[0],self.x_range[1], 10)
        plt.xticks(x_ticks)
        y_ticks = np.linspace(self.t_points[0], self.t_points[-1], 10)
        plt.yticks(y_ticks)

        plt.xlabel("x")
        plt.ylabel("t [s]")
        plt.title("Evolution of the solution to the heat equation")
        plt.show()
       
    def plot_Hexgrid(self, method: str = "analytical", size=(100, 100)):
        """
        Experimental function to plot the solution as a hexbin plot
        instead of a heatmap.
        """
        plt.rcParams.update({'font.family': 'Verdana'})
        if method == "analytical":
            data = np.copy(self.solution_a)
            data = np.flip(data, axis=0)
        elif method == "numerical":
            data = np.copy(self.solution)
            data = np.flip(data, axis=0)
        else:
            raise ValueError("Method must be either 'analytical' or 'numerical'!") 

        n_bins_x = size[0]
        n_bins_y = size[1]
        bin_size_x = (data.shape[0] - 1)/n_bins_x
        bin_size_y = (data.shape[1] - 1)/n_bins_y

        grid = np.zeros((n_bins_x, n_bins_y))
        
        for i in range(n_bins_x):
            for j in range(n_bins_y):
                start_x = i*int(bin_size_x)
                end_x = (i+1)*int(bin_size_x)
                start_y = j*int(bin_size_y)
                end_y = (j+1)*int(bin_size_y)

                bin_value = np.mean(data[start_x:end_x, start_y:end_y])
                grid[i, j] = bin_value

        x_centers = np.arange(0.5 * bin_size_x, data.shape[1], bin_size_x)
        y_centers = np.arange(0.5 * bin_size_y, data.shape[0], bin_size_y)

        hexagon_width = (x_centers[1] - x_centers[0]) / 2
        hexagon_height = (y_centers[1] - y_centers[0]) * 3**0.5 / 2

        x_store_center = []
        y_store_center = []
        for i in range(n_bins_x):
            x = x_centers[i]
            # Shift every other row
            shift_y = 0 if i % 2 == 0 else hexagon_width
            for j in range(n_bins_y):
                y = y_centers[j] + shift_y
                x_store_center.append(x)
                y_store_center.append(y)
            
        fig, ax = plt.subplots()
        norm = plt.Normalize(vmin=np.min(data), vmax=np.max(data))
        
        hexagon = RegularPolygon((0, 0), numVertices=6, radius=bin_size_x / 2,
                                  facecolor=cmr.flamingo(norm(0)), edgecolor='k')
        ax.add_patch(hexagon)

        x_vec = x_centers.repeat(n_bins_y)
        y_vec = np.tile(y_centers, n_bins_x)

        store_center = np.column_stack((x_store_center, y_store_center))

        facecolors = [cmr.flamingo(norm(value)) for value in grid.ravel()]
        collection = RegularPolyCollection(
            numsides=6, 
            rotation=60, 
            sizes=(100,),
            facecolors=facecolors,
            offsets=store_center,
            )
        ax.add_collection(collection) #, autolim=True)
        ax.set_aspect('equal')

        plt.xlabel("x")
        plt.ylabel("t [s]")
        plt.title("Evolution of the solution to the heat equation")
        # plt.xlim(self.x_range[0], self.x_range[1])
        # plt.ylim(self.t_points[0], self.t_points[-1])
        plt.show()

    def plot_Heat(self):
        """
        Plot the heat of the solution as a function of time.
        """
        plt.rcParams.update({'font.family': 'Verdana'})
        fig, ax = plt.subplots()
        ax.plot(self.t_points, np.sum(self.solution, axis=1))
        ax.set_xlabel("t [s]")
        ax.set_ylabel("Heat")
        ax.set_title("Heat of the solution to the heat equation")
        plt.show()

    def plot_Heatmap_sns(self, method: str = "analytical"):
        """
        Plot the solution as a heatmap using seaborn.
        """
        if method == "analytical":
            data = np.copy(self.solution_a)
            data = np.flip(data, axis=0)
        elif method == "numerical":
            data = np.copy(self.solution)
            data = np.flip(data, axis=0)
        else:
            raise ValueError("Method must be either 'analytical' or 'numerical'!") 

        data = pd.DataFrame(data)

        x_ticks = np.round(np.linspace(self.x_range[0],self.x_range[1], data.shape[0]))
        y_ticks = np.round(np.linspace(self.t_points[0], self.t_points[-1], data.shape[1]))

        data.columns = x_ticks
        data.index = y_ticks

        sns.set_theme()
        fig, ax = plt.subplots()
        sns.heatmap(data, cmap=cmr.flamingo, ax=ax)

        plt.xlabel("x")
        plt.ylabel("t [s]")
        plt.title("Evolution of the solution to the heat equation")
        plt.show()


        plt.show()
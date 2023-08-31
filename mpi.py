"""
This file should contain the wrapper class for enabling MPI parallelization.
The wrapper class should be able to take a solver as an argument and then
distribute the grid points to the nodes. The nodes should then be able to
solve the PDE in parallel and then gather the solution to the root node.
"""
from mpi4py import MPI
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter
from spectral import *
from colocation import *
from typing import Tuple, Callable, List, Iterable
import socket


class MPI_Node():
    def __init__(self, 
                 solver: ColocationSolver | SpectralSolver,
                 ) -> None:
        """
        Initialize the MPI node.
        Number of total points in the grid has to be divisible by the number of nodes.
        """
        # Init MPI
        self.comm = MPI.COMM_WORLD
        self.rank = self.comm.Get_rank()
        self.size = self.comm.Get_size()
        self.status = MPI.Status()
        self.name = MPI.Get_processor_name()
        self.hostname = socket.gethostname()
        print(f"Hi! This is rank {self.rank} on {self.hostname}. Ready to go to work...")

        # Init solver
        self.solver = solver
        self.initial_condition = solver.initial_condition
        self.x_range = solver.x_range
        self.N = solver.N
        self.t_points = solver.t_points
        self.D = solver.D
        self.x = np.linspace(self.x_range[0], self.x_range[1], self.N)
        if self.rank == 0:
            self.x_backup = np.copy(self.x)
        self.dx = self.x[1] - self.x[0]
        self.dt = self.t_points[1] - self.t_points[0]
        print(f"Rank {self.rank} has initialized the solver.")
    
    def _internal_function_timer(func: Callable):
        def wrapper(*args, **kwargs):
            import time
            start = time.time()
            result = func(*args, **kwargs)
            end = time.time()
            print(f"Function {func.__name__} took {end - start} seconds to run.")
            return result, end - start
        return wrapper
    
    def _distribute_data(self, data):
        """Divide and distribute given data into chunks per rank.

        Surprisingly, this uses no MPI functionality at all. It is
        just a simple list comprehension.
    
        Parameters:
            data: A list of data points which is to be
            divided and distributed.
    
        Returns:
            divided[rank]: A list of data points which is
            assigned to the current rank.
        """
        n = len(data)
        divided = []
        # One x point evolved in time is one data row
        total_data_rows = n
        rows_per_node = total_data_rows // self.size
        extra_rows = total_data_rows % self.size

        self.rows_distribution = [rows_per_node] * self.size

        # This will never go out of range, since the
        # extra rows cannot be more than the number of nodes.
        for i in range(extra_rows):
            self.rows_distribution[i] += 1

        # Distribute the data
        divided = [data[i:i+self.rows_distribution[self.rank]] for i 
                   in range(0, total_data_rows, self.rows_distribution[self.rank])]

        return divided[self.rank]
    
    def _gather_data(self, data, isRoot=False):
        """Gather data from all ranks.
    
        Parameters:
            data: The data to be gathered.
        
        Args:
            isRoot: Whether the current rank is the root rank.
    
        Returns:
            gathered_data: The gathered data.
        """
        gathered_data = self.comm.gather(data, root=0)

        if not isRoot:
            return gathered_data  # comm.gather returns None on non-root ranks
        elif isRoot:
            # ERROR IDENTIFIED: The gathered data is not in the correct order. Was broken by the
            # modifications made for the statistics gatherer.
            output = np.zeros((self.x_backup.shape[0], self.t_points.shape[0]))
            for i in range(self.size):
               output[i*self.rows_distribution[i]:(i+1)*self.rows_distribution[i], :] = gathered_data[i].T

            return output.T
            # gathered_data = np.asarray(gathered_data).transpose(1, 0, 2).reshape(self.x_backup.shape[0], self.t_points.shape[0]).T
        
            # return gathered_data

    @_internal_function_timer
    def solve(self, method=None, partialOutput=False):
        """
        Parallel solve the PDE but in chunks.

        Parameters:
            method: The method to use for solving the PDE.

        Args:
            partialOutput: Whether to return the partial 
                solution or not.

        Returns:
            solution: The partial solution of the PDE.
            time: The time it took to solve the PDE.
        """
        # Set defaults for method
        if method is None:
            if type(self.solver) == ColocationSolver:
                method = "proper"
            elif type(self.solver) == SpectralSolver:
                method = "analytical"

        x = self._distribute_data(self.x)
        self.solver._override_x(x)

        if type(self.solver) == ColocationSolver:
            if method == "manual":
                solution, time = self.solver.solve_Manually()
            elif method == "proper":
                solution, time = self.solver.solve_Properly()

        elif type(self.solver) == SpectralSolver:
            if method == "numerical":
                solution, time = self.solver.solve_Numerically()
            elif method == "analytical":
                solution, time = self.solver.solve_Analytically()

        else:
            raise TypeError("The solver must be of type ColocationSolver or SpectralSolver.")
        
        if partialOutput:
            # Copy of the solution pre-gather
            self.partialSolution = np.copy(solution)

        if self.rank != 0:
            self.solution = self._gather_data(solution)
        elif self.rank == 0:
            self.solution = self._gather_data(solution, isRoot=True)
        else:
            # Should never happen.
            raise ValueError("The rank must be an integer.")
        
        if partialOutput:
            return self.solution, self.partialSolution
        else:
            # Return the partial solution more as a proof of concept
            return solution
        
    def plot_Animation(self,
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
        # Only rank 0 should plot
        if self.rank != 0:
            return None
        try:
            x = self.x_backup
            solution = self.solution
        except NameError:
            print("Call solution method before trying to plot!")

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

    def plot_Lines(self):
        """
        Plot the solution as lines.
        """
        # Only rank 0 should plot
        if self.rank != 0:
            return None
        try:
            x = self.x_backup
            data = self.solution
        except NameError:
            print("Call solution method before trying to plot!")

        plt.rcParams.update({'font.family': 'Verdana'})
        fig, ax = plt.subplots(facecolor="#4d4c4c")

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

        ax.set_title("Evolution of the solution to the heat equation", color="#dedede")

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
        plt.show()
        

    


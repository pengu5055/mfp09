"""
This file contains a wrapper class for the MPI wrapper class which will
collect statistics from the MPI nodes and plot them. The statistics
gathered are the time it took to solve the PDE. 
"""
from mpi4py import MPI
from mpi import MPI_Node
import numpy as np
from colocation import ColocationSolver
from newsrc import SpectralSolver
import matplotlib.pyplot as plt
import socket
from typing import Tuple, Callable, List, Iterable


class GatherStatistics:
    def __init__(self,
                 solver: ColocationSolver | SpectralSolver,
                 rank_range: Tuple[int, int],
                 sample_size: int,
                 ) -> None:
        """
        Initialize the statistics gatherer.
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

        # Init statistics
        self.rank_range = rank_range
        self.sample_size = sample_size

    def _internal_function_timer(func: Callable):
        def wrapper(*args, **kwargs):
            import time
            start = time.time()
            result = func(*args, **kwargs)
            end = time.time()
            print(f"Function {func.__name__} took {end - start} seconds to run.")
            return result, start - end
        return wrapper

    def _distribute_data(self, data):
        """Divide and distribute given data into chunks per rank.

        Parameters:
            data: The data to be distributed.
        """
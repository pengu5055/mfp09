"""
This file will contain the class for the colocation method. 
The colocation method is a numerical method for solving PDE's.
It is a spectral method, meaning that it uses a Fourier basis to approximate the solution.
We're going to need to solve matrix equations, so we'll use numpy.
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

class ColocationSolver:
    def __init__(self) -> None:
        

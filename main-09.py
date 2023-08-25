import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter
from src import *


def update(frame):
    new_step = evolve[frame, :]
    line.set_data(x, new_step)
    return line,

def ivp_1(y, t):
    """Test IVP to try out integrators.
    Has the analytical solution 2 * np.exp(-2*t) + np.exp(t)
    for y(0) = 5
    """
    return -2*y + 3 * np.exp(t)


if __name__ == "__main__":
    x = np.linspace(0, 10, N)
    t = np.linspace(0, 10, N)
    init = gaussian(x, A, SIGMA)


    # evolve = N * spectral_PDE_solver(init, suppressWarning=True)

    # fig, ax = plt.subplots()
    # line = ax.plot(x, evolve[0], label="Evolved")[0]
# 
    # ani = FuncAnimation(fig, update, frames=range(N), blit=True,)
    # plt.rcParams['animation.ffmpeg_path'] ='C:\\Media\\ffmpeg\\bin\\ffmpeg.exe' 
    # writervideo = FFMpegWriter(fps=20) 
    # ani.save("sweep.mp4", writer=writervideo)

    sol = V_euler_integrator(ivp_1, init, t)
    true = 2 * np.exp(-2*t) + np.exp(t)

    plt.plot(t, sol[0], label="Calculated")
    plt.plot(t, true)
    plt.show()
    
    

    
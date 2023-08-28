"""
This source file contains the integrator for the spectral solver.
Essentially it contains an implementation of the RK4_5 integrator
with adaptive step size control.
"""
import numpy as np
from rich import print
from rich.progress import Progress
from typing import Callable, Tuple, Iterable

def RK4_5(
        f: Callable[[np.ndarray, float], np.ndarray],
        x0: float | Iterable[float],
        y0: float | Iterable[float],
        x_end: float,
        n_steps: int,
        outputSteps: bool = False,
        debug: bool = False,
        exitOnWarning: bool = False,
        disableDivCheck: bool = False,
        overrideInternalSettings: dict = None
):
    """
    """
    # Internal settings
    if overrideInternalSettings:
        INTERNAL = overrideInternalSettings
    else:
        INTERNAL = {
            "divergenceTolerance": 10e12,
            "tol": 1e-6,
            "safetyFactor": 0.9
        }
    
    # Cache variables that are going to be used in iterations
    divergenceTolerance = INTERNAL["divergenceTolerance"]
    tol = INTERNAL["tol"]
    safetyFactor = INTERNAL["safetyFactor"]


    if x0 > x_end:
        print(":warning: [bold red]Warning: x0 must be smaller than x_end.[/bold red]")
        if exitOnWarning:
            raise ValueError("x0 must be smaller than x_end.")
    
    if n_steps < 2:
        print(":warning: [bold red]Warning: n_steps must be larger than 1.[/bold red]")
        if exitOnWarning:
            raise ValueError("n_steps must be larger than 1.")
        
    solution = np.zeros_like(y0)
    errors = np.zeros_like(y0)
    x_steps = np.zeros_like(x_end)
    
    with Progress() as progress:
        task1 = progress.add_task("[red]Integrating..", total=n_steps)
        for i in range(n_steps):
            k1 = h * f(x0, y0)
            k2 = h * f(x0 + h/2, y0 + k1/2)
            k3 = h * f(x0 + h/2, y0 + k2/2)
            k4 = h * f(x0 + h, y0 + k3)
            k = (k1 + 2*k2 + 2*k3 + k4)/6

            yn = y0 + k
            yn_hat = y0 + RK5_step(f, x0, y0, h)
            error = np.abs(yn - yn_hat)
            if not disableDivCheck:
                if np.any(np.isnan(yn)) or np.any(yn > divergenceTolerance):
                    print(":warning: [bold red]Warning: Divergence detected.[/bold red]")
                    if exitOnWarning:
                        raise ValueError("Divergence detected.")
                    
            # Store results
            solution = np.column_stack((solution, yn))
            errors = np.column_stack((errors, np.abs(yn_hat - yn)))
            x_steps = np.column_stack((x_steps, x0))

            # Adaptive step size control
            error_ratio = tol/error
            suggested_h = safetyFactor * h * error_ratio**(1/5)

            # suggested_h is compared with h/2 and h*0.1 to avoid
            # the step being more than halved every time and from
            # becoming too small.
            h = np.min([suggested_h, np.max([h / 2, h * 0.1])])

            # Prepare for next iteration
            y0 = yn
            x0 += h
            progress.update(task1, i)
    
    # Remove first columns since they were generated empty
    solution = np.delete(solution, 0, 1)
    errors = np.delete(errors, 0, 1)
    x_steps = np.delete(x_steps, 0, 1)

    if outputSteps:
        return solution, errors, x_steps
    else:
        return solution, errors


def RK5_step(f, x, y, h):
    """
    
    """
    k1 = h * f(x, y)
    k2 = h * f(x + h/4, y + k1/4)
    k3 = h * f(x + 3*h/8, y + 3*k1/32 + 9*k2/32)
    k4 = h * f(x + 12*h/13, y + 1932*k1/2197 - 7200*k2/2197 + 7296*k3/2197)
    k5 = h * f(x + h, y + 439*k1/216 - 8*k2 + 3680*k3/513 - 845*k4/4104)
    k6 = h * f(x + h/2, y - 8*k1/27 + 2*k2 - 3544*k3/2565 + 1859*k4/4104 - 11*k5/40)
    
    y_next = y + 16*k1/135 + 6656*k3/12825 + 28561*k4/56430 - 9*k5/50 + 2*k6/55
    return y_next

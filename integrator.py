"""
This source file contains the integrator for the spectral solver.
Essentially it contains an implementation of the RK4_5 integrator
with adaptive step size control.
"""
import numpy as np
from rich import print
from rich.progress import Progress, TimeElapsedColumn, TextColumn, TimeRemainingColumn, SpinnerColumn, MofNCompleteColumn
from typing import Callable, Tuple, Iterable


def RK4_5_adapt(
        f: Callable[[np.ndarray, float], np.ndarray],
        x0: float | Iterable[float],
        y0: float | Iterable[float],
        x_end: float,
        h: float,
        outputSteps: bool = False,
        debug: bool = False,
        exitOnWarning: bool = False,
        disableDivCheck: bool = False,
        overrideInternalSettings: dict = None
) -> Tuple[np.ndarray, np.ndarray, np.ndarray] | Tuple[np.ndarray, np.ndarray]:
    """
    This is a non-working implementation of the RK4(5) integrator with
    adaptive step size control. It is not working because the step size
    is not updated correctly.

    Arguments:
        f: The function to integrate. Must be of the form f(x, y).
        x0: The initial x value.
        y0: The initial y value.
        x_end: The final x value.
        h: The initial step size.
    
    Parameters:
        outputSteps: If True the function will return the steps taken.
        debug: If True the function will print debug information.
        exitOnWarning: If True the function will exit on warning.
        disableDivCheck: If True the function will not check for divergence.
        overrideInternalSettings: If not None, the function will override
            the internal settings with the provided dictionary.

    Returns:
        If outputSteps is True, the function will return the solution,
        the estimated error and the steps taken. Otherwise it will return
        the solution and the estimated error.
    """
    # Internal settings
    if overrideInternalSettings:
        INTERNAL = overrideInternalSettings
    else:
        INTERNAL = {
            "divergenceTolerance": 10e12,
            "tol": 1e-6,
            "safetyFactor": 0.99,
            "maxIterations": 100000,
            "stepLowerLimit": 1e-12,
        }
    
    # Cache variables that are going to be used in iterations
    divergenceTolerance = INTERNAL["divergenceTolerance"]
    tol = INTERNAL["tol"]
    safetyFactor = INTERNAL["safetyFactor"]
    maxIterations = INTERNAL["maxIterations"]
    stepLowerLimit = INTERNAL["stepLowerLimit"]


    if x0 > x_end:
        print(":warning: [bold red]Warning: x0 is smaller than x_end![/bold red]")
        if exitOnWarning:
            raise ValueError("x0 must be smaller than x_end.")
    
    if h < tol:
        print(":warning: [bold red]Warning: Step below error tolerance![/bold red]")
        if exitOnWarning:
            raise ValueError("Step must be larger than error tolerance.")
        
    solution = np.zeros_like(y0)
    errors = np.zeros_like(y0)
    x_steps = np.zeros_like(x_end)
    
    with Progress(
        SpinnerColumn(spinner_name="pong"),
        TextColumn("[bold green]{task.description}"),
        TimeElapsedColumn(),
        TextColumn("Step size: [bold blue]{task.fields[step_size]}. ETA"),
        TimeRemainingColumn(),
    ) as progress:
        task = progress.add_task("[red]Integrating...", step_size=h, total=x_end)
        it = 0  # Iteration counter
        while x0 < x_end or it > maxIterations:
            yn = y0 + RK4_step(f, x0, y0, h)
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
            if debug:
                print(f"x: {x0}, y: {yn}, error: {error}")

            # Adaptive step size control
            if error != 0:
                error_ratio = tol / error
                suggested_h = safetyFactor * h * error_ratio**(1/8)

                # suggested_h is compared with h/2 and h*0.1 to avoid
                # the step being more than halved every time and from
                # becoming too small.
                h = np.min([suggested_h, np.max([h * 0.1, h * 0.9])])
                if h < stepLowerLimit:
                    print(":warning: [bold red]Warning: Step size below lower limit![/bold red]")
                    if exitOnWarning:
                        raise ValueError("Step size below lower limit.")

            # Prepare for next iteration
            y0 = yn
            x0 += h
            it += 1
            if debug:
                print(f"Step size: {h}, x: {x0}")
    
    # Remove first columns since they were generated empty
    solution = np.delete(solution, 0, 1)
    errors = np.delete(errors, 0, 1)
    x_steps = np.delete(x_steps, 0, 1)

    if outputSteps:
        return solution, errors, x_steps
    else:
        return solution, errors


def RK4_5(
        f: Callable[[np.ndarray, float], np.ndarray],
        x0: float | Iterable[float],
        y0: float | Iterable[float],
        x_end: float,
        n_steps: int,
        *args,
        outputSteps: bool = False,
        debug: bool = False,
        exitOnWarning: bool = False,
        disableDivCheck: bool = False,
        overrideInternalSettings: dict = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray] | Tuple[np.ndarray, np.ndarray]:
    """
    This is an implementation of the RK4(5) integrator with fixed step size.
    Thus making it simpler than the adaptive version. It is also more
    efficient and I suppose ready for MPI. 

    Arguments:
        f: The function to integrate. Must be of the form f(x, y).
        x0: The initial x value.
        y0: The initial y value.
        x_end: The final x value.
        n_steps: The number of steps to take.
    
    Parameters:
        outputSteps: If True the function will return the steps taken.
        debug: If True the function will print debug information.
        exitOnWarning: If True the function will exit on warning.
        disableDivCheck: If True the function will not check for divergence.
        overrideInternalSettings: If not None, the function will override
            the internal settings with the provided dictionary.
        args: Additional arguments to pass to the function f.
    
    Returns:
        If outputSteps is True, the function will return the solution,
        the estimated error and the steps taken. Otherwise it will return
        the solution and the estimated error.
    """
    # Internal settings
    if overrideInternalSettings:
        INTERNAL = overrideInternalSettings
    else:
        INTERNAL = {
            "divergenceTolerance": 10e12,
            "tol": 1e-6,
        }
    
    # Cache variables that are going to be used in iterations
    divergenceTolerance = INTERNAL["divergenceTolerance"]
    tol = INTERNAL["tol"]

    # Calculate step size
    h = (x_end - x0) / n_steps

    print(f"Integrating with step size: {h} and {n_steps} steps.")

    if x0 > x_end:
        print(":warning: [bold red]Warning: x0 is smaller than x_end![/bold red]")
        if exitOnWarning:
            raise ValueError("x0 must be smaller than x_end.")
    
    if h < tol:
        print(":warning: [bold red]Warning: Step below error tolerance![/bold red]")
        if exitOnWarning:
            raise ValueError("Step must be larger than error tolerance.")
        
    solution = np.zeros_like(y0)
    errors = np.zeros_like(y0)
    x_steps = np.zeros_like(x_end)
    
    with Progress(
        SpinnerColumn(spinner_name="pong"),
        TextColumn("[bold red]\[{task.description}][bold green][bold yellow] Elapsed: "),
        TimeElapsedColumn(),
        TextColumn("[bold blue]Step size: {task.fields[step_size]}.[bold green] ETA: "),
        TimeRemainingColumn(),
        TextColumn("[bold purple]Steps completed: "),
        MofNCompleteColumn(),
    ) as progress:
        task = progress.add_task("Integrating", step_size=h, total=n_steps)
        it = 0  # Iteration counter
        for i in range(n_steps):
            yn = y0 + RK4_step(f, x0, y0, h, *args)
            yn_hat = y0 + RK5_step(f, x0, y0, h, *args)
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
            if debug:
                print(f"x: {x0}, y: {yn}, error: {error}")


            # Prepare for next iteration
            y0 = yn
            x0 += h
            it += 1
            progress.update(task, advance=1)
    
    # Remove first columns since they were generated empty
    solution = np.delete(solution, 0, 1)
    errors = np.delete(errors, 0, 1)
    x_steps = np.delete(x_steps, 0, 1)

    if outputSteps:
        return solution, errors, x_steps
    else:
        return solution, errors



def RK5_step(f, x, y, h, *args):
    """
    This is the integration step of a 5th order Runge-Kutta method.
    It is used in the RK4(5) integrator for error estimation.
    
    Arguments:
        f: The function to integrate. Must be of the form f(x, y).
        x: The current x value.
        y: The current y value.
        h: The step size.
    
    Returns:
        The integration step.
    """
    k1 = h * f(x, y, *args)
    k2 = h * f(x + h/4, y + k1/4, *args)
    k3 = h * f(x + 3*h/8, y + 3*k1/32 + 9*k2/32, *args)
    k4 = h * f(x + 12*h/13, y + 1932*k1/2197 - 7200*k2/2197 + 7296*k3/2197, *args)
    k5 = h * f(x + h, y + 439*k1/216 - 8*k2 + 3680*k3/513 - 845*k4/4104, *args)
    k6 = h * f(x + h/2, y - 8*k1/27 + 2*k2 - 3544*k3/2565 + 1859*k4/4104 - 11*k5/40, *args)
    
    k = 16*k1/135 + 6656*k3/12825 + 28561*k4/56430 - 9*k5/50 + 2*k6/55
    return k

def RK4_step(f, x, y, h, *args):
    """
    This is the integration step of a 4th order Runge-Kutta method.
    
    Arguments:
        f: The function to integrate. Must be of the form f(x, y).
        x: The current x value.
        y: The current y value.
        h: The step size.
    
    Returns:
        The integration step.
    """
    k1 = h * f(x, y, *args)
    k2 = h * f(x + h/2, y + k1/2, *args)
    k3 = h * f(x + h/2, y + k2/2, *args)
    k4 = h * f(x + h, y + k3, *args)

    k = (k1 + 2*k2 + 2*k3 + k4)/6
    return k

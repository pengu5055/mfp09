"""
This source script contains my (pengu5055) attempt at implementing
the 8th order Runge-Kutta formula RK8(9). It is based on the paper
https://ntrs.nasa.gov/citations/19680027281 p. 76-87.
"""
import numpy as np
from rich import print
from rich.progress import Progress
from typing import Callable, Tuple, Iterable


def RK8_9(f: Callable,
          x0: Iterable[float] | float,
          y0: Iterable[float] | float,
          x: float,
          h: float, 
          outputSteps: bool = False, 
          debug: bool = False, 
          exitOnFail: bool = False, 
          disableDivCheck: bool = False,
          overrideInternalSettings: dict[str, float] = None
          ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    This function implements the 8th order Runge-Kutta formula RK8(9)
    with 9th order error estimation. It is an adaptive step size method
    for solving the initial value problem y' = f(x, y), y(x0) = y0.
    The function is based on the paper:
    https://ntrs.nasa.gov/citations/19680027281 p. 76-87.

    WARNING: This function is currently not working properly. It 
    diverges for even the simplest of initial value problems!

    It is an amateur-ish implementation by pengu5055.

    Arguments:
        f: The function f(x, y) in the initial value problem.
        x0: The initial value of x.
        y0: The initial value of y.
        x: The final value of x.
        h: The step size.

    Parameters:
        outputSteps: If True, the function will output the array of
            x coordinates which were used to compute the solution.
        debug: If True, the function will print debug information.
        exitOnFail: If True, the function will raise an error if
            the solution diverges or if the 1D initial conditions 
            generate different values during the steps.
        disableDivCheck: If True, the function will not check if
            the solution diverges.
        overrideInternalSettings: If not None, the function will
            override the internal settings with the given dictionary.
            The dictionary should have the following keys:
             - "divergenceTolerance": The tolerance for the divergence
                check.
             - "eps": The tolerance for the adaptive step size.  
             - "stepDownSafetyFactor": The safety factor for reducing
                the step size.
             - "stepUpSafetyFactor": The safety factor for increasing
                the step size.
    
    Returns:
        solution: An array where the columns are the solutions at 
        generated time points in x.
        errors: An array where the columns are the estimated errors
        at generated time points in x.
        x_steps: An array where the columns are the step points in x.
    """
    # Internal settings
    if overrideInternalSettings:
        INTERNAL = overrideInternalSettings
    else:
        INTERNAL = {
            "divergenceTolerance": 10e12,
            "eps": 10e-8,
            "stepDownSafetyFactor": 0.9,
            "stepUpSafetyFactor": 1.1
        }
    
    # Cache variables that are going to be used in iterations
    eps = INTERNAL["eps"]
    stepDownSafetyFactor = INTERNAL["stepDownSafetyFactor"]
    stepUpSafetyFactor = INTERNAL["stepUpSafetyFactor"]

    n = int((x - x0) / h)
    x = x0
    y = y0

    print(f"Running RK8_9() with {n} steps and a step size of {h}...")

    # The Butcher tableau for the 8th order Runge-Kutta formula
    alpha = np.zeros(16)
    beta = np.zeros((16, 16))
    c = np.zeros(16)
    c_hat = np.zeros(16)

    # Set the Butcher tableau
    alpha[1 - 1] = 0.44368940376498183109599404281370
    alpha[2 - 1] = 0.66553410564747274664399106422055
    alpha[3 - 1] = 0.99830115847120911996598659633083
    alpha[4 - 1] = 0.3155
    alpha[5 - 1] = 0.50544100948169068626516126737384
    alpha[6 - 1] = 0.17142857142857142857142857142857
    alpha[7 - 1] = 0.82857142857142857142857142857143
    alpha[8 - 1] = 0.66543966121011562534953769255586
    alpha[9 - 1] = 0.24878317968062652069722274560771
    alpha[10 - 1] = 0.109
    alpha[11 - 1] = 0.891
    alpha[12 - 1] = 0.3995
    alpha[13 - 1] = 0.6005
    alpha[14 - 1] = 1
    alpha[15 - 1] = 0
    alpha[16 - 1] = 1
    # To add to the confusion, the paper uses 1-based indexing, for
    # the first index of beta, but 0-based indexing for the second.
    beta[1 - 1, 0] = 0.44368940376198183109599404281370	
    beta[2 - 1, 0] = 0.16638352641186818666099776605514
    beta[2 - 1, 1] = 0.49915057923560455998299329816541
    beta[3 - 1, 0] = 0.24957528961780227999149664908271
    beta[3 - 1, 2] = 0.74872586885340683997448994724812
    beta[4 - 1, 0] = 0.20661891163400602426556710393185
    beta[4 - 1, 2] = 0.17707880377986347040380997288319
    beta[4 - 1, 3] = -0.68197715413869494669377076815048 * 10**-1
    beta[5 - 1, 0] = 0.10927823152666408227903890926157
    beta[5 - 1, 3] = 0.40215962642367995421990563690087 * 10**-2
    beta[5 - 1, 4] = 0.39214118169078980444392330174325
    beta[6 - 1, 0] = 0.98899281409164665304844765434355 * 10**-1
    beta[6 - 1, 3] = 0.35138370227963966951204487356703 * 10**-2
    beta[6 - 1, 4] = 0.12476099983160016621520625872489
    beta[6 - 1, 5] = -0.55745546834989799643742901466348 * 10**-1
    beta[7 - 1, 0] = -0.36806865286242203724153101080691
    beta[7 - 1, 4] = -0.22273897469476007645024020944166 * 10**1
    beta[7 - 1, 5] = 0.13742908256702910729565691245744 * 10**1
    beta[7 - 1, 6] = 0.20497390027111603002159354092206 * 10**1
    beta[8 - 1, 0] = 0.45467962641347150077351950603349 * 10**-1
    beta[8 - 1, 5] = 0.32542131701589147114677469648853
    beta[8 - 1, 6] = 0.28476660138527908888182420573687
    beta[8 - 1, 7] = 0.97837801675979152435868397271099 * 10**-2    
    beta[9 - 1, 0] = 0.60842071062622057051094145205182 * 10**-1
    beta[9 - 1, 5] = -0.21184565744037007526325275251206 * 10**-1
    beta[9 - 1, 6] = 0.19596557266170831957464490662983
    beta[9 - 1, 7] = -0.42742640364817603675144835342899 * 10**-2
    beta[9 - 1, 8] = 0.17434365736814911955323452558189 * 10**-1
    beta[10 - 1, 0] = 0.54059783296931917355785724111182 * 10**-1
    beta[10 - 1, 6] = 0.11029325597828926539283127648228
    beta[10 - 1, 7] = -0.12565008520072556414147763782250 * 10**-2
    beta[10 - 1, 8] = 0.36790043477581460136384043566339 * 10**-2
    beta[10 - 1, 9] = -0.57780542770372073940840628571866 * 10**-1
    beta[11 - 1, 0] = 0.12732477068657114546645181799160
    beta[11 - 1, 7] = 0.11448805006396105323658875721817
    beta[11 - 1, 8] = 0.28773020703697992776202201849198
    beta[11 - 1, 9] = 0.50945379459611363153735885079465
    beta[11 - 1, 10] = -0.14799682244372575900242144449610
    beta[12 - 1, 0] = -0.36526793876616740535848544394333 * 10**-2
    beta[12 - 1, 5] = 0.81629896012318919777819421247030 * 10**-1
    beta[12 - 1, 6] = -0.38607735635693506490517694313215
    beta[12 - 1, 7] = 0.30862242924605106450474166025206 * 10**-1
    beta[12 - 1, 8] = -0.58077254528320602815829374733518 * 10**-1
    beta[12 - 1, 9] = 0.33598659328884971493143451362322
    beta[12 - 1, 10] = 0.41066880401949958613549622786417
    beta[12 - 1, 11] = -0.11840245972355985520633156154536 * 10**-1
    beta[13 - 1, 0] = -0.12375357921245143254979096135669 * 10**1
    beta[13 - 1, 5] = -0.24430768551354785358734661366763 * 10**2
    beta[13 - 1, 6] = 0.54779568932778656050436528991173
    beta[13 - 1, 7] = -0.44413863533413246374959896569346 * 10**1
    beta[13 - 1, 8] = 0.10013104813713266094792617851022 * 10**2
    beta[13 - 1, 9] = -0.14995773102051758117170985072142 * 10**2
    beta[13 - 1, 10] = 0.58946948523217013620824539651427 * 10**1
    beta[13 - 1, 11] = 0.17380377503428984877616857440542 * 10**1
    beta[13 - 1, 12] = 0.27512330693166730263758622860276 * 10**2
    beta[14 - 1, 0] = -0.35260859388334522700502958875588
    beta[14 - 1, 5] = -0.18396103144848270375044198988231
    beta[14 - 1, 6] = -0.65570189449741645138006879985251
    beta[14 - 1, 7] = -0.39086144880439863435025520241310
    beta[14 - 1, 8] = 0.26794616712850022936584423271209
    beta[14 - 1, 9] = -0.10383022991382490865769858507427 * 10**1
    beta[14 - 1, 10] = 0.16672327324258671664727346168501 * 10**1
    beta[14 - 1, 11] = 0.49551925855315977067732967071411
    beta[14 - 1, 12] = 0.113940011323397063228586738141784 * 10**1
    beta[14 - 1, 13] = 0.51336696424658613688199097191534 * 10**-1
    beta[15 - 1, 0] = 0.10464847340614810391873002406755 * 10**-2
    beta[15 - 1, 8] = -0.67163886844990282237778446178020 * 10**-2
    beta[15 - 1, 9] = 0.81828762189425021265330065248999 * 10**-2
    beta[15 - 1, 10] = -0.42640342864483347277142138087561 * 10**-2
    beta[15 - 1, 11] = 0.28009029474168936545976331103703 * 10**-3
    beta[15 - 1, 12] = -0.87835333876238676639057813145633 * 10**-2
    beta[15 - 1, 13] = 0.10254505110825558084217769664009 * 10**-1
    beta[16 - 1, 0] = -0.13536550786174067080442168889966 * 10**1
    beta[16 - 1, 5] = -0.18396103144848270375044198988231
    beta[16 - 1, 6] = -0.65570189449741645138006879985251
    beta[16 - 1, 7] = -0.39086144880439863435025520241310
    beta[16 - 1, 8] = 0.27466285581299925758962207732989
    beta[16 - 1, 9] = -0.10464851753571915887035188572676 * 10**1
    beta[16 - 1, 10] = 0.16714967667123155012004488306588 * 10**1
    beta[16 - 1, 11] = 0.49523916825841808131186990740287
    beta[16 - 1, 12] = 0.11481836466273301905225795954930 * 10**1
    beta[16 - 1, 13] = 0.41082191313833055603981327527525 * 10**-1
    beta[16 - 1, 15] = 1
    # C has 0-based indexing
    c[0] = 0.32256083500216249913612900960247 * 10**-1
    c[8] = 0.25983725283715403018887023171963
    c[9] = 0.92847805996577027788063714302190 * 10**-1
    c[10] = 0.16452339514764342891647731842800
    c[11] = 0.17665951637860074367084298397547
    c[12] = 0.23920102320352759374108933320941	
    c[13] = 0.39484274604202853746752118829325 * 10**-2
    c[14] = 0.30726495475860640406368305522124 * 10**-1

    # Reduce ninth-order equations to eighth-order with these
    # assumptions.
    # Assume c is 0 indexed
    for ind in range(1, 7 + 1):
        c_hat[ind - 1] = c[ind] = 0

    
    c_hat[14 - 1] = 0
    c_hat[15 - 1] = c_hat[16 - 1] = c[14]
    for ind in range(8, 14 + 1):
        c_hat[ind - 1] = c[ind]

    solution = np.zeros_like(y0)
    errors = np.zeros_like(y0)
    x_steps = np.zeros_like(x)

    print("Integration in progress...")
    with Progress() as progress:
        task1 = progress.add_task("[red]Integration...", total=n)
        # The iterator i counts the number of steps taken
        for i in range(1, n+1):
            # Apply the 8th order Runge-Kutta formula
            f_vec = [f(x0, y0)] # First element already given

            for k in range(1, 16 + 1):
                value = f(x0 + alpha[k - 1]*h, y0 + 
                          h*np.sum([beta[k - 1][lam] * f_vec[lam] for lam in range(0, k)]))

                # TODO: Remove debug as it could slow down the program
                if debug:
                    # Check if all elements of value are the same
                    if not disableDivCheck:
                        if np.abs(value[0]) > INTERNAL["divergenceTolerance"]:
                            progress.console.print(f":warning: [bold red]Iteration step {i} on summation step {k} has diverged![/bold red]")
                            if exitOnFail:
                                progress.console.print(f"Last known value: {value[0]}")
                                raise ValueError("Divergence detected!")

                    if np.all(value == value[0]):
                        progress.console.print(f"Iteration step {i} on summation step {k} has value: {value[0]}")
                    else:
                        try:
                            if len(x0) > 0:
                                # Having a vector as initial value is allowed to 
                                # have different values
                                pass
                        except TypeError:
                            # Catching a type error means that x0 is a scalar and thus has no len()
                            progress.console.print(f":warning: [bold red]Iteration step {i} on summation step {k} has different values![/bold red]")
                            if exitOnFail:
                                raise ValueError("1D initial conditions should not generate different values during steps!")

                # END DEBUG

                f_vec.append(value)

            if debug:
                if len(f_vec) != 17:
                    progress.console.print(f":warning: [bold red]Iteration step {i} has {f_vec.size[0]} elements![/bold red]")
                    if exitOnFail:
                        raise ValueError("f_vec should have 17 elements!")

            # Apply the 8th order Runge-Kutta formula to estimate the solution
            # Give benefit of the doubt to the paper and assume that the
            # K can be indexed from 0.
            # y = y0 + h*np.sum([c[k - 1] * f_vec[k - 1] for k in range(1, 14 + 1)])
            y = y0 + h*np.sum([c[k] * f_vec[k] for k in range(0, 14)])
            
            # TODO: Remove debug as it could slow down the program
            if debug:
                # Check if all elements of y are the same
                if np.all(y== y[0]):
                    progress.console.print(f"On step {i} the value of y is {y[0]}")
                else:
                    try:
                        if len(x0) > 0:
                            # Having a vector as initial value is allowed to 
                            # have different values
                            pass
                    except TypeError:
                        # Catching a type error means that x0 is a scalar and thus has no len()
                        progress.console.print(f":warning: [bold red]Iteration step {i} on summation step {k} has different values![/bold red]")
                        if exitOnFail:
                            raise ValueError("1D initial conditions should not generate different values during steps!")
            
            # END DEBUG

            # Apply the 9th order Runge-Kutta formula to estimate the error
            # Give benefit of the doubt to the paper and assume that the
            # K can be indexed from 0.
            # y_hat = y0 + h*np.sum([c_hat[k - 1] * f_vec[k - 1] for k in range(1, 16 + 1)])
            y_hat = y0 + h*np.sum([c_hat[k] * f_vec[k] for k in range(0, 16)])

            solution = np.column_stack((solution, y))
            errors = np.column_stack((errors, y_hat - y))
            x_steps = np.column_stack((x_steps, x0))

            # Adaptive step size
            if np.any(np.abs(y_hat - y) > eps):
                # Error is too large so reduce step size
                h = h * stepDownSafetyFactor * (eps / np.abs(y_hat - y))**(1/8)
            else:
                # Error is small enough so increase step size
                h = h * stepUpSafetyFactor * (eps / np.abs(y_hat - y))**(1/8)

            if debug:
                print(f"Step size: {h}")

            # Prepare for the next iteration
            y0 = y
            x0 = (x0 + h)[0] # TODO TEMPORARY FIX: append only scalar values of x0

            # Sanity check if all x0 elements are the same
            if exitOnFail:
                if x0.size > 1:
                    if not np.all(x0 == x0[0]):
                        raise ValueError("x0 should not have different values during steps!")
            
            progress.update(task1, advance=1)

    # Remove first columns since they were generated empty
    solution = np.delete(solution, 0, 1)
    errors = np.delete(errors, 0, 1)
    x_steps = np.delete(x_steps, 0, 1)

    if outputSteps:
        return solution, errors, x_steps
    else:
        return solution, errors
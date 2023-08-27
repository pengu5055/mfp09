"""
This source script contains my (pengu5055) attempt at implementing
the 8th order Runge-Kutta formula RK8(9). It is based on the paper
https://ntrs.nasa.gov/citations/19680027281 p. 76-87.
"""
import numpy as np
from progressbar import ProgressBar


def RK8(f, x0, y0, x, h):
    """
    """
    n = int((x - x0) / h)
    x = x0
    y = y0

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
    beta[10 - 1, 0] = 0.54049783296931917355785724111182 * 10**-1
    beta[10 - 1, 6] = 0.11029325597828926539283127648228
    beta[10 - 1, 7] = -0.12565008520072556414147763782250 * 10**-2
    beta[10 - 1, 8] = 0.36790043477581460136384043566339 * 10**-2
    beta[10 - 1, 9] = -0.57780542770372073940840628571866 * 10**-1
    beta[11 - 1, 0] = 0.12732477068657114546645181799160
    beta[11 - 1, 7] = 0.11418303006395103323658875721817
    beta[11 - 1, 8] = 0.28773020703697992776202201849198
    beta[11 - 1, 9] = 0.50945379459611363153735885079465
    beta[11 - 1, 10] = -0.14799682244372575900242144449610
    beta[12 - 1, 0] = -0.36326793876616740535848544394333 * 10**-2
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
    beta[14 - 1, 12] = 0.11394001132397063228586738141784 * 10**1
    beta[14 - 1, 13] = 0.51336696424658613688199097191534 * 10**-1
    beta[15 - 1, 0] = 0.10464847340614810391873002406755 * 10**-2
    beta[15 - 1, 8] = -0.67163886844990282237778446178020 * 10**-2
    beta[15 - 1, 9] = 0.81828762189425021265330065248999 * 10**-2
    beta[15 - 1, 10] = -0.42640342864483347277142138087561 * 10**-2
    beta[15 - 1, 11] = 0.28009029474168936545976331103703 * 10**-3
    beta[15 - 1, 12] = -0.87835333876238676639057813145633 * 10**-2
    beta[15 - 1, 13] = 0.19976174238824432682773771616875 * 10**-1
    beta[16 - 1, 0] = -0,13536550786174067080442168889966 * 10**1
    beta[16 - 1, 5] = -0.18396103144848270375044198988231
    beta[16 - 1, 6] = -0.65570189449741645138006879985251
    beta[16 - 1, 7] = -0.39086144880439863435025520241310
    beta[16 - 1, 8] = 0.27466285581299925758962207732989
    beta[16 - 1, 9] = -0.10164851753571915887035188572676 * 10**1
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
    alpha[14 - 1] = alpha[15 - 1] = 1
    alpha[15 - 1] = 0
    for ind in range(1, 7 + 1):
        c_hat[ind - 1] = c[ind - 1] = 0

    c_hat[14 - 1] = 0
    c_hat[15 - 1] = c_hat[16 - 1] = c[14 - 1]
    for ind in range(1, 14 + 1):
        c_hat[ind - 1] = c[ind - 1]

    solution = np.zeros_like(y0)
    errors = np.zeros_like(y0)

    with ProgressBar(max_value=n) as bar:
        # The iterator i counts the number of steps taken
        for i in range(1, n+1):
            # Apply the 8th order Runge-Kutta formula
            f_vec = [f(x0, y0)] # First element already given

            for k in range(1, 16 + 1):
                f_vec.append(f(x0 + alpha[k]*h, y0 + h*np.sum([beta[k][lam] for lam in range(0, k - 1)])))

            y = y0 + h*np.sum(c[k]*f_vec[k] for k in range(1, 14 + 1))

            # Apply the 9th order Runge-Kutta formula to estimate the error
            y_hat = y0 + h*np.sum(c_hat[k]*f_vec[k] for k in range(1, 16 + 1))

            solution = np.column_stack((solution, y))
            errors = np.column_stack((errors, y_hat - y))

            # Prepare for the next iteration
            y0 = y
            x0 = x0 + h
            bar.update(i)

    # Remove first columns since they were generated empty
    solution = np.delete(solution, 0, 1)
    errors = np.delete(errors, 0, 1)

    
    return solution, errors
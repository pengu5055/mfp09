# DN 9. Spektralne metode za začetne probleme PDE (Difuzija toplote)

Naslednje tri naloge se ukvarjajo z različnimi metodami za reševanje parcialnih diferencialnih enačb, katere si lahko na nek način predstavljamo kot posplošitev diferencialnih enačb. Dopuščamo odvisnost od več parametrov. PDE so zelo pogoste v fiziki (pravzaprav, verjetno bolj pogoste kot ODE). Pojavijo se kadarkoli opisujemo količino, ki ima tako časovnno kot prostorsko odvisnost for example. Recimo primer Poissonove enačbe se pojavlja povsod (EMP, Gravitacija, Fluidomehanika, etc.).

## Navodila
Naloga želi, da rešiš enodimenzionalno difuzijo topote z začetni pogoj Gaussovsko porazdeljene temperature. To moraš narediti z Fourierovo metodo za periodični robni pogoj $T(0,\>t) = T(a,\>t)$, kjer je $a$ širina intervala na katerem rešujemo, in za homogen Dirichletov robni pogoj $T(0,\>t) = T(a,\>t) = 0$. Potem je potrebno rešiti nalogo še z Kolokacijsko metodo za homogen Dirichletov robni pogoj.

## Napotki
Priznam, da je kar nekaj časa minilo med tem ko sem oddal to nalogo in sedaj, ko jo končno objavljam. Zato se žal ne spomnim več vseh detajlov. Ko sem pisal to nalogo sem še vedno imel kar velik fokus na optimizirano kodo in možnost paralelnega izvajanja. Mogoče še najbolj smiseln napotek je, da se splača uporabiti tudi že obstoječe knjižnice za reševanje PDE. Recimo primer za reševanje z spektralnimi metodami je `dedalus-project`,
ki je med drugim tudi paraleliziran, če bi se slučajno kdo želel igrati s tem, kar je popolnoma nepotrebno sicer. 

Poleg tega vse kar se spomnim so splošni napotki v smislu, da se mi zdi koristno uporabljati razrede za numerične metode, ker so na tak način stvari bolj pregledne. Razred je kot nek kontejner, ki vsebuje vse potrebne metode in spremenljivke za reševanje nekega problema. Pridejo tudi zelo prav, če želi kdo narediti sprehod skozi parametre, kajti lahko hkrati narediš več instanc razreda z različnimi parametri. 

Več kot to, v tem trenutku ne morem povedati. Žal sem predolgo časa čakal, da sem objavil to nalogo.

## Kar sem jaz naredil
**Tu je verjetno tisto kar te najbolj zanima**. 

<details>
  <summary>Standard Disclaimer</summary>
  Objavljam tudi kodo. Ta je bila tokrat v svojem repozitoriju od začetka, ker sem teh zadnjih nekaj nalog opravljal med poletjem. Koda bi morala biti razmeroma pokomentirana, sploh v kasnejših nalogah. 
  
</details>

Vseeno pa priporočam, da si najprej sam poskusiš rešiti nalogo. As always za vprašanja sem na voljo.


* [**Poročilo DN9**](https://pengu5055.github.io/fmf-pdf/year3/mfp/Marko_Urban%C4%8D_09.pdf)
* [**Source repozitorij DN9**](https://github.com/pengu5055/mfp09)

Priznam, da zna biti source repozitorij nekoliko kaotičen. Over time sem se naučil boljše prakse. Zdi se mi, da je tole glavni `.py` file.

* [**main_09.py**](https://github.com/pengu5055/mfp09/blob/b7e996834bef0863eab74c2124b04b5ddb960a81/main-09.py)

## Citiranje
*Malo za šalo, malo za res*.. če želiš izpostaviti/omeniti/se sklicati ali pa karkoli že, na moje delo, potem ga lahko preprosto citiraš kot:

```bib
@misc{Urbanč_mfpDN9, 
  title={Spektralne metode za začetne probleme PDE}, 
  url={https://pengu5055.github.io/fmf-pages/year3/mfp/dn9.html}, 
  journal={Marko’s Chest}, 
  author={Urbanč, Marko}, 
  year={2023}, 
  month={Oct}
} 
```
To je veliko boljše kot prepisovanje.

# **Amateur-ish 8th order Runge-Kutta implementation**

The file `rk.py` contains the source code for a semi-functional implementation of an 8th order Runge-Kutta integrator
with 9th order error approximation and adaptive step capabilities. 

## **Installation**
The function has a few dependencies that can be installed with `pip`. Maybe in the future I will provide a `requirements.txt` file, but for now, you can install the dependencies with the following command:
```bash
pip install numpy rich
```
Rich is a package used for Progress Bars, pretty printing, and other nice things. It is not necessary for the function to work, but it is nice to have. Currently there is no way to disable it, but I will add that in the future I suppose.

## **Usage**
The function is used as follows:
```python
from rk import *

# Define the function to be integrated
def ivp(x, y):
    """Test IVP to try out integrators.
    Has the analytical solution 2 * np.exp(-2*t) + np.exp(t)
    for y(0) = 5
    """
    return -2*y + 3 * np.exp(x)

# Define the initial conditions
x0 = 0
y0 = 5
x = 10
stepSize = 0.0005
solution, errors= RK8_9(ivp, x0, y0, x, stepSize)
```
The function generally returns two arrays where the columns of the arrays are the values of the solution and the errors at each step. 

## **Parameters**
The function has optional parameters mostly pertaining to debugging and testing. They are as follows:

| Parameter | Type | Description |
| --- | --- | --- |
| `outputSteps` | `bool` | If `True`, the function also return a third array of the steps used for calculation. |
| `debug` | `bool` | If `True`, the function will print out the values of the solution and the errors at each step. |
| `exitOnFail` | `bool` | If `True`, the function will raise a ValueError on Failure/Warning. |
| `disableDivCheck` | `bool` | If `True`, the function will not check for divergence. |
| `overrideInternalSettings` | `dict[str, float]` | If not 'None', will use parameters in given dict. See below. |

## **Overriding Internal Settings**
The function has a few internal settings that can be overridden. They are as follows:

### divergenceToleance
The tolerance for divergence. If the error is greater than this value, the function will warn the user or exit, if so set by
`exitOnFail`. The default value is `1e10`.

### eps
The adaptive step size tolerance. If the error is less than this value, the step size will be increased and vice versa.
The default value is `1e-8`.

### stepDownSafetyFactor
The factor by which is used in the calculation of the decrease of the step size when its error is too large. The default value is `0.9`.

### stepUpSafetyFactor
The factor by which is used in the calculation of the increase of the step size when its error is too small. The default value is `1.1`.

#### **Example**
Below is an example of how to override the internal settings:
```python
override = {
    "divergenceTolerance": 1e12,
    "eps": 1e-6,
    "stepDownSafetyFactor": 0.8,
    "stepUpSafetyFactor": 1.2
}
```



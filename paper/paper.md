---
title: 'Kamodo: A functional API for space weather models and data'
tags:
  - Python
  - plasma physics
  - space weather
authors:
  - name: Asher Pembroke
    affiliation: 1
  - name: Darren DeZeeuw
    affiliation: 2, 3
  - name: Lutz Rastaetter
    affiliation: 2
  - name: Rebecca Ringuette
    affiliation: 4, 2
  - name: Oliver Gerland
    affiliation: 5
  - name: Dhruv Patel
    affiliation: 5
  - name: Michael Contreras
    affiliation: 5

affiliations:
 - name: Asher Pembroke, DBA
   index: 1
 - name: Community Coordinated Modeling Center, NASA GSFC
   index: 2
 - name: Catholic University of America
   index: 3
 - name: ADNET Systems Inc.
   index: 4
 - name: Ensemble Government Services
   index: 5
date: Sept 28, 2021
bibliography: paper.bib
---

# Summary

Kamodo is a functional application programing interface (API) for scientific models and data.
In Kamodo, all scientific resources are registered as symbolic fields which are mapped to model and data interpolators or algebraic expressions.
Kamodo performs function composition and employs a unit conversion system that mimics hand-written notation: units are declared in bracket notation and conversion factors are automatically inserted into user expressions.
Kamodo includes a LaTeX interface, automated plots, and a browser-based dashboard interface suitable for interactive data exploration.
Kamodo's json API provides context-dependent queries and allows compositions of models and data hosted in separate docker containers.
Kamodo is built primarily on sympy [@10.7717/peerj-cs.103] and plotly [@plotly].
While Kamodo was designed to solve the cross-disciplinary challenges of the space weather community, it is general enough to be applied in other fields of study.


# Statement of need

Space weather models and data employ a wide variety of specialized formats, data structures, and interfaces tailored for the needs of domain experts.
However, this specialization is also an impediment to cross-disciplinary research.
For example, data-model comparisons often require knowledge of multiple data structures and observational data formats.
Even when mature APIs are available, proficiency in programing languages such as Python is necessary before progress may be made.
This further complicates the transition from research to operations in space weather forecasting and mitigation, where many disparate data sources and models must be presented together in a clear and actionable manner.
Such complexity represents a high barrier to entry when introducing the field of space weather to newcomers at space weather workshops, where much of the student's time is spent installing and learning how to use prerequisite software.
Several attempts have been made to unify all existing space weather resources around common data standards, but have met with limited success.
In particular, introducing and leveraging a common data standard for space weather models was the primary goal of the Kameleon software suite, a predecessor to Kamodo developed between 1999-2011 at the Community Coordinated Modeling Center, NASA GSFC [@kameleon].
Kameleon consisted of a set of tools for converting raw simulation output into standardized HDF or CDF format with additional metadata specific to space weather modeling (scientific units, array structure, coordinate systems, and citation information) as well as interpolation APIs targeting several languages (C, C++, Fortran, Java, and Python).
Due to the complexity of space weather modeling techniques, these interpolators were tailored for specific models and had to be written by the Kameleon developers themselves.
This created a bottleneck in the time to onboard new simulations, and only a handful of models could be supported.
In addition, interpolation of observational data fell outside the scope of Kameleon's design requirements, and additional tooling was required for metrics and validation. Furthermore, the difficulty in installing the prerequisite libraries meant that only a few users could take advantage of Kameleon's powerful interpolation techniques. Often, scientific users either developed their own pipelines for analysis or simply relied on CCMC's static plots available over the web.
Our experience with Kameleon and its limitations were a strong motivating factor for Kamodo's functional design.


Kamodo all but eliminates the barrier to entry for accessing space weather resources by exposing all scientifically relevant parameters in a functional manner.
Kamodo is an ideal tool in the scientist's workflow, because many problems in space weather analysis, such as field line tracing, coordinate transformation, and interpolation, may be posed in terms of function compositions.
The underlying implementation of these functions are left to the model and data access libraries. This allows Kamodo to build on existing standards and APIs without requiring programing expertise on the part of the end user.
Kamodo is expressive enough to meet the needs of most scientists, educators, and space weather forecasters, and Kamodo containers enable a rapidly growing ecosystem of interoperable space weather resources. 

# Usage

## Kamodo Base Class

Kamodo's base class manages the registration of functionalized resources. As an example, here is how one might register the 500th-order approximation of the non-differentiable Weierstrass function [@weierstrass1872uber].

```python
from kamodo import Kamodo, kamodofy
import numpy as np

@kamodofy(
    equation=r"\sum_{n=0}^{500} (1/2)^n cos(3^n \pi x)",
    citation='Weierstrass, K. (1872). Uber continuirliche functionen eines '
      'reellen arguments, die fur keinen werth des letzteren einen '
      'bestimmten differentialquotienten besitzen, Akademievortrag. '
      'Math. Werke von Karl Weierstrass, Vol. 2, 71-74, Mayer & Mueller (1895).' 
    )
def weierstrass(x = np.linspace(-2, 2, 1000)):
    '''
    Weierstrass function (continuous and non-differentiable)

    https://en.wikipedia.org/wiki/Weierstrass_function
    '''
    nmax = 500
    n = np.arange(nmax)

    xx, nn = np.meshgrid(x, n)
    ww = (.5)**nn * np.cos(3**nn*np.pi*xx)
    return ww.sum(axis=0)

k = Kamodo(W=weierstrass)
```
When run in a jupyter notebook, the latex representation of the above function is shown: 

\begin{equation}W{\left(x \right)} = \sum_{n=0}^{500} (1/2)^n cos(3^n \pi x)\end{equation}

This function can be queried at any point within its domain:

```python
k.W(0.25)

# array([0.47140452])
```

Kamodo's plotting routines can automatically visualize this function at multiple zoom levels:

```python
k.plot('W')
```

The result of the above command is shown in \autoref{fig:weierstrass}. This exemplifies Kamodo's ability to work with highly resolved datasets through function inspection.

![Auto-generated plot of the Weierstrass function.\label{fig:weierstrass}](https://raw.githubusercontent.com/EnsembleGovServices/kamodo-core/master/docs/notebooks/weirstrass.png)

 
## Kamodo Subclasses

The Kamodo base class may be subclassed when third-packages are required. For example, the `pysatKamodo` subclass preregisters interpolating functions for Pysat [@pysat200] Instruments:

```python
from pysat_kamodo.nasa import Pysat_Kamodo

kcnofs = Pysat_Kamodo(
         # Pysat_Kamodo allows string dates
         '2009, 1, 1',
         # pysat mission name (C/NOFS)
         platform = 'cnofs',
         # pysat instrument suite (Vector Electric Field Investigation)
         name='vefi', 
         # pysat type of observation (here: DC magnetic fields)
         tag='dc_b',
         )
kcnofs['B'] = '(B_north**2+B_up**2+B_west**2)**.5' # a derived variable
```

Here is how the `kcnofs` instance appears in a jupyter notebook: 

\begin{equation}\operatorname{B_{north}}{\left(t \right)}[nT] = \lambda{\left(t \right)}\end{equation}
\begin{equation}\operatorname{B_{up}}{\left(t \right)}[nT] = \lambda{\left(t \right)}\end{equation}
\begin{equation}\operatorname{B_{west}}{\left(t \right)}[nT] = \lambda{\left(t \right)}\end{equation}
\begin{equation}\operatorname{B_{flag}}{\left(t \right)} = \lambda{\left(t \right)}\end{equation}
\begin{equation}\operatorname{B_{IGRF north}}{\left(t \right)}[nT] = \lambda{\left(t \right)}\end{equation}
\begin{equation}\operatorname{B_{IGRF up}}{\left(t \right)}[nT] = \lambda{\left(t \right)}\end{equation}
\begin{equation}\operatorname{B_{IGRF west}}{\left(t \right)}[nT] = \lambda{\left(t \right)}\end{equation}
\begin{equation}\operatorname{latitude}{\left(t \right)}[degrees] = \lambda{\left(t \right)}\end{equation}
\begin{equation}\operatorname{longitude}{\left(t \right)}[degrees] = \lambda{\left(t \right)}\end{equation}
\begin{equation}\operatorname{altitude}{\left(t \right)}[km] = \lambda{\left(t \right)}\end{equation}
\begin{equation}\operatorname{dB_{zon}}{\left(t \right)}[nT] = \lambda{\left(t \right)}\end{equation}
\begin{equation}\operatorname{dB_{mer}}{\left(t \right)}[nT] = \lambda{\left(t \right)}\end{equation}
\begin{equation}\operatorname{dB_{par}}{\left(t \right)}[nT] = \lambda{\left(t \right)}\end{equation}
\begin{equation}B{\left(t \right)}[nT] = \sqrt{\operatorname{B_{north}}^{2}{\left(t \right)} + \operatorname{B_{up}}^{2}{\left(t \right)} + \operatorname{B_{west}}^{2}{\left(t \right)}}\end{equation}

Units are explicitly shown on the left hand side, while the right hand side of these expressions represent interpolating functions ready for evaluation:

```python
kcnofs.B(pd.DatetimeIndex(['2009-01-01 00:00:03','2009-01-01 00:00:05']))
```
<!-- #region -->
```sh
2009-01-01 00:00:03    19023.052734
2009-01-01 00:00:05    19012.949219
dtype: float32
```
<!-- #endregion -->

Here, the function `B(t)` returns the result of a variable derived from preregistered variables as a pandas series object. However, kamodo itself does not require functions to utilize a specific data type, provided that the datatype supports algebraic operations.

Kamodo can auto-generate plots using function inspection:

```python
kcnofs.plot('B_up')
```

![Auto-generated plot of CNOFs Vefi instrument.\label{fig:cnofs}](https://github.com/pysat/pysatKamodo/raw/master/docs/cnofs_B_up.png)

The result of the above command is shown in \autoref{fig:cnofs}. To accomplish this, Kamodo analyzes the structure of inputs and outputs of `B_up` and selects an appropriate plot type from the Kamodo plotting module.

Citation information for the above plot may be generated from the `meta` property of the registered function:

```python
kcnofs.B_up.meta['citation']
```

which returns references for the C/NOFS platform [@cnofs] and VEFI instrument [@vefi].


# Related Projects

Kamodo is designed for compatibility with python-in-heliophysics [@ware_alexandria_2019_2537188] packages, such as PlasmaPy [@plasmapy_community_2020_4313063] and PySat [@Stoneback2018; @pysat200].
This is accomplished through Kamodo subclasses, which are responsible for registering each scientifically relevant variable with an interpolating function.
Metadata describing the function's units and other supporting documentation (citation, latex formatting, etc) may be provisioned by way of the `@kamodofy` decorator.

The PysatKamodo [@pysatKamodo] interface is made available in a separate git repository. Readers for various space weather models and data sources are under development by the Community Coordinated Modling Center and are hosted in their official NASA repository [@nasaKamodo].

Kamodo's unit system is built on SymPy [@10.7717/peerj-cs.103] and shares many of the unit conversion capabilities of `Astropy` [@astropy; @astropy2] with two key differences:
Kamodo uses an explicit unit conversion system, where units are declared during function registration and appropriate conversion factors are automatically inserted on the right-hand-side of final expressions, which permits back-of-the-envelope validation.
Second, units are treated as function metadata, so the types returned by functions need only support algebraic manipulation via libraries such as NumPy [@harris2020array] or Pandas [@reback2020pandas].
Output from kamodo-registered functions may still be cast into other unit systems that require a type, such as Astropy [@astropy; @astropy2] and Pint [@pint].

Kamodo can utilize some of the capabilities of raw data APIs such as HAPI, and a HAPI kamodo subclass is maintained in the ccmc readers repository [@nasaKamodo]. However, Kamodo also provides an API for purely functional data access, which allows users to specify positions or times for which interpolated values should be returned.
To that end, a prototype for functional REST api [@fielding2000rest] is available [@ensembleKamodo], as well as an RPC api [@nelson2020remote] for direct access from other programing languages.

Kamodo container services may be built on other containerized offerings.
Containerization allows dependency conflicts to be avoided through isolated install environments.
Kamodo extends the capabilities of space weather resource containers by allowing them to be composed together via the KamodoClient, which acts as a proxy for the containerized resource running the Kamodo RPC API.

# Acknowledgements

Development of Kamodo was initiated by the Community Coordinated Modeling Center, with funding provided by Catholic University of America under the NSF Division of Atmospheric and Geospace Sciences, Grant No 1503389.
Continued support for Kamodo is provided by Ensemble Government Services, LTD. via NASA Small Business Innovation Research (SBIR) Phase I/II, grant No 80NSSC20C0290, 80NSSC21C0585, resp.
Additional support is provided by NASA’s Heliophysics Data and Model Consortium.

The authors are thankful for the advice and support of Nicholas Gross, Katherine Garcia-Sage, and Richard Mullinex. 


# References

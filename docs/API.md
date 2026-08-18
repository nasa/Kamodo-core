# API documentation

## Kamodo

::: kamodo.Kamodo

### Initialization

::: kamodo.Kamodo
    options:
      members:
        - __init__

### Registering functions

::: kamodo.Kamodo
    options:
      members:
        - __setitem__

### Retrieving functions

Registered functions may be accessed via dictionary or attribute syntax.

::: kamodo.Kamodo
    options:
      members:
        - __getitem__
        - __getattr__

### Evaluation

Function evaluation may be performed either by keyword or attribute syntax:

```py
k = Kamodo(f='x^2-x-1')
assert k.f(3) == k['f'](3)

For closed-form expressions, kamodo uses the highly optimized [numexpr](https://numexpr.readthedocs.io/projects/NumExpr3/en/latest/intro.html) library if available and will fall back to numpy otherwise:

```py
x = np.linspace(-5,5,33000111)
k.f(x)
```

Programmatic evaluation is also possible:

::: kamodo.Kamodo
    :members: evaluate

### Plotting

#### single function plots

For plotting single variables, the `figure` method is most appropriate

::: kamodo.Kamodo
    :members: figure

#### multi-function plots

For multiple functions, the `plot` method is more convenient

::: kamodo.Kamodo
    :members: plot


### LaTeX rendering

The following methods allow Kamodo to integrate seemlessly with modern publication workflows. This includes support for LaTeX rendering within jupyter notebooks, LaTeX printing for manuscript preparation, and a high-level `detail` summary of registered functions.

::: kamodo.Kamodo
    :members: _repr_latex_ to_latex  detail

## Plotting

### Plot types

As described in the Visualization section, Kamodo automatically maps registered functions to certain plot types. All such functions expect the same input variables and return a triplet `[trace], chart_type, layout` where `[trace]` is a list of plotly trace objects.

::: kamodo.plotting.get_plot_types_df
    :docstring:

The available plot types may be imported thusly:

```python
from kamodo.plotting import plot_types
```

### Scatter plot

::: kamodo.plotting.scatter_plot
    :docstring:

### Line plot

::: kamodo.plotting.line_plot
    :docstring:

### Vector plot

::: kamodo.plotting.vector_plot
    :docstring:

### Contour plot

::: kamodo.plotting.contour_plot
    :docstring:

### 3D Plane

::: kamodo.plotting.plane
    :docstring:

### 3D Surface

::: kamodo.plotting.surface
    :docstring:

### Carpet plot

::: kamodo.plotting.carpet_plot
    :docstring:

### Triangulated Mesh plot

::: kamodo.plotting.tri_surface_plot
    :docstring:

### Image plot

::: kamodo.plotting.image
    :docstring:

## Decorators

These decorators may also be imported like this

```python
from kamodo import kamodofy
```

### kamodofy

::: kamodo.util.kamodofy
    :docstring:

### gridify

::: kamodo.util.gridify
    :docstring:

### pointlike

::: kamodo.util.pointlike
    :docstring:

### partial

::: kamodo.util.partial
    :docstring:


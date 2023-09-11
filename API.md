# API documentation

## Kamodo

::: kamodo.Kamodo
    :docstring:

### Initialization

::: kamodo.Kamodo
    :members: __init__


### Registering functions

::: kamodo.Kamodo
    :members: __setitem__


### Retrieving functions

Registered functions may be accessed via dictionary or attribute syntax.

::: kamodo.Kamodo
    :members: __getitem__ __getattr__


### Evaluation

Function evaluation may be performed either by keyword or attribute syntax:

```py
k = Kamodo(f='x^2-x-1')
assert k.f(3) == k['f'](3)
```

For closed-form expressions, kamodo uses the highly optimized [numexpr](https://numexpr.readthedocs.io/projects/NumExpr3/en/latest/intro.html) library if available and will fall back to numpy otherwise:

```py
x = np.linspace(-5,5,33000111)
k.f(x)
```

Programmatic evaluation is also possible:

::: kamodo.Kamodo
    :members: evaluate

### RPC server

Start a Kamodo asyncio server using `kamodo.serve`:

::: kamodo.Kamodo
    :members: serve

### RPC client

Start a Kamodo Client using the `KamodoClient` class:

::: kamodo.KamodoClient
    :members: __init__

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

As described in [Visualization](../notebooks/Visualization/), Kamodo automatically maps registered functions to certain plot types. All such functions expect the same input variables and return a triplet `[trace], chart_type, layout` where `[trace]` is a list of plotly trace objects.

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


## Test Suite

Kamodo features a full suite of tests run via pytest. We highlight a few of these tests below as further examples of Kamodo's expected use cases.

### Kamodo Tests

::: kamodo.test_kamodo.test_Kamodo_expr
    :docstring: 

::: kamodo.test_kamodo.test_Kamodo_latex
    :docstring: 

::: kamodo.test_kamodo.test_Kamodo_mismatched_symbols
    :docstring: 

::: kamodo.test_kamodo.test_Kamodo_reassignment
    :docstring: 

::: kamodo.test_kamodo.test_function_registry
    :docstring: 

::: kamodo.test_kamodo.test_unit_registry
    :docstring: 

::: kamodo.test_kamodo.test_komodofy_decorator
    :docstring: 

::: kamodo.test_kamodo.test_vectorize
    :docstring: 

::: kamodo.test_kamodo.test_jit_evaluate
    :docstring:  

::: kamodo.test_kamodo.test_multiple_traces
    :docstring:

### Plotting Tests

::: kamodo.test_plotting.test_scatter_plot
    :docstring:

::: kamodo.test_plotting.test_line_plot_line
    :docstring:

::: kamodo.test_plotting.test_line_plot_2d_line
    :docstring:

::: kamodo.test_plotting.test_line_plot_3d_line_pd
    :docstring:

::: kamodo.test_plotting.test_vector_plot_2d_vector
    :docstring:

::: kamodo.test_plotting.test_vector_plot_3d_vector
    :docstring:

::: kamodo.test_plotting.test_vector_plot_3d_vector
    :docstring:

::: kamodo.test_plotting.test_contour_plot_2d_grid
    :docstring:

::: kamodo.test_plotting.test_contour_plot_2d_skew
    :docstring:

::: kamodo.test_plotting.test_plane
    :docstring:

::: kamodo.test_plotting.test_surface_3d_surface
    :docstring:

::: kamodo.test_plotting.test_arg_shape_pd
    :docstring:

::: kamodo.test_plotting.test_image_plot
    :docstring:








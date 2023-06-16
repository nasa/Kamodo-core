```python
from kamodo import Kamodo, kamodofy
import numpy as np
```

```python

@kamodofy(
    equation=r"\sum_{n=0}^{500} (1/2)^n cos(3^n \pi x)",
    citation='Weierstrass, K. (1872). Uber continuirliche functionen eines reellen arguments, die fur keinen worth des letzteren einen bestimmten differentailqutienten besitzen, Akademievortrag. Math. Werke, 71-74.'
    )
def weierstrass(x = np.linspace(-2, 2, 1000)):
    '''
    Weierstrass  function
    A continuous non-differentiable 
    https://en.wikipedia.org/wiki/Weierstrass_function
    '''
    nmax = 500
    n = np.arange(nmax)

    xx, nn = np.meshgrid(x, n)
    ww = (.5)**nn * np.cos(3**nn*np.pi*xx)
    return ww.sum(axis=0)

k = Kamodo(W=weierstrass)
k
```

```python
k.to_latex()
```

```python
k.W(0.25)
```

```python
fig = k.plot('W')
fig
```

```python
fig.write_image('weirstrass.png', scale=2)
```

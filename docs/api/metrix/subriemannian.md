# Sub-Riemannian geometry

`HorizontalCometric` declares a rank-`r` frame distribution inside an
`n`-dimensional coordinate chart and constructs the positive semidefinite cometric
from that frame. It does not pretend to be an invertible Riemannian metric.

The horizontal gradient, normal Hamiltonian, and density-weighted sub-Laplacian are
defined directly from this cometric. `step_two_horizontal_rank` augments the frame by
all first Lie brackets and diagnoses step-two bracket generation at representative
points.

```python
import jax.numpy as jnp
import phydrax as phx

chart = phx.metrix.CoordinateChart("heisenberg", ("x", "y", "z"))
cometric = phx.metrix.HorizontalCometric(
    lambda q: jnp.array(
        [
            [1.0, 0.0],
            [0.0, 1.0],
            [-0.5 * q[1], 0.5 * q[0]],
        ]
    ),
    chart,
    2,
)

point = jnp.array([0.2, 0.3, 0.0])
assert phx.metrix.step_two_horizontal_rank(cometric, point) == 3
```

::: phydrax.metrix.HorizontalCometric

::: phydrax.metrix.horizontal_gradient

::: phydrax.metrix.horizontal_hamiltonian

::: phydrax.metrix.sub_laplacian

::: phydrax.metrix.step_two_horizontal_rank

::: phydrax.metrix.validate_horizontal_cometric

::: phydrax.operators.horizontal_grad

::: phydrax.operators.sub_laplacian

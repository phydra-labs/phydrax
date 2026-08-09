# Geometry-preserving solvers

Geometric solvers consume explicit state geometry rather than inferring constraints
from state shape. `RKMK` evaluates Runge--Kutta stages in local retraction coordinates.
`CommutatorFreeSolver` composes group-like retractions without explicitly forming Lie
brackets. `SRKMK` handles Stratonovich drift--diffusion systems.

For a separable canonical Hamiltonian `H(q, p) = V(q) + T(p)`, `StormerVerlet` applies
the second-order kick--drift--kick map. Its term must contain a
`SeparableHamiltonianVectorField`; arbitrary ODE terms are rejected rather than treated
as if they were separable Hamiltonian systems.

```python
import diffrax as dfx
import jax.numpy as jnp
import phydrax as phx

vector_field = phx.solver.SeparableHamiltonianVectorField(
    lambda t, q, args: q,
    lambda t, p, args: p,
    1,
)
solution = dfx.diffeqsolve(
    dfx.ODETerm(vector_field),
    phx.solver.StormerVerlet(1),
    t0=0.0,
    t1=10.0,
    dt0=0.1,
    y0=jnp.array([1.0, 0.0]),
    stepsize_controller=dfx.ConstantStepSize(),
)
```

::: phydrax.solver.AbstractGeometricSolver

::: phydrax.solver.GeometricEuler

::: phydrax.solver.RKMK

::: phydrax.solver.CommutatorFreeSolver

::: phydrax.solver.SRKMK

::: phydrax.solver.SeparableHamiltonianVectorField

::: phydrax.solver.StormerVerlet

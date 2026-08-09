# Lie groups

Lie-group structure is separate from Riemannian metric structure. `AbstractLieGroup`
provides identity, composition, inverse, exponential and logarithmic maps, adjoint
action, and Lie bracket. A group becomes an optimization manifold only when a positive
invariant metric is supplied through a manifold implementation.

The initial concrete groups are `SO(2)`, `SO(3)`, and `SE(2)`. General `SO(n)` retains
matrix-valued algebra coordinates; `hat` and `vee` are available where a compact
coordinate representation is defined.

```python
import jax.numpy as jnp
import phydrax as phx

group = phx.metrix.SpecialOrthogonalGroup(3)
algebra = group.hat(jnp.array([0.2, -0.1, 0.3]))
element = group.exp(algebra)
assert group.contains(element)
```

`LieGroupStateGeometry` delegates solver retractions and local coordinates to the group
operations, enabling RKMK and commutator-free solvers without duplicating group laws.

::: phydrax.metrix.AbstractLieGroup

::: phydrax.metrix.SpecialOrthogonalGroup

::: phydrax.metrix.SpecialEuclideanGroup

::: phydrax.metrix.LieGroupStateGeometry

::: phydrax.solver.RKMK

::: phydrax.solver.CommutatorFreeSolver

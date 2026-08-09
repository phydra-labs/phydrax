# Symplectic and Poisson geometry

`SymplecticForm` represents a closed, nondegenerate two-form. `PoissonStructure`
represents an antisymmetric bivector satisfying the Jacobi identity. A nondegenerate
symplectic form can be converted to its inverse Poisson structure; a general Poisson
structure may be degenerate and therefore has no canonical symplectic inverse.

```python
import jax.numpy as jnp
import phydrax as phx

chart = phx.metrix.CoordinateChart("phase", ("q", "p"))
symplectic = phx.metrix.canonical_symplectic_form(chart)
poisson = phx.metrix.symplectic_to_poisson(symplectic)

H = lambda z: 0.5 * jnp.dot(z, z)
X_H = phx.metrix.hamiltonian_vector_field(H, poisson, jnp.array([2.0, 3.0]))
assert jnp.allclose(X_H, jnp.array([3.0, -2.0]))
```

The general `DomainFunction` operators require an explicit Poisson or symplectic
structure and an ordered tuple of phase-variable labels. The older canonical operation
has the explicit name `canonical_poisson_bracket`; no global canonical default is
silently selected.

::: phydrax.metrix.SymplecticForm

::: phydrax.metrix.PoissonStructure

::: phydrax.metrix.canonical_symplectic_form

::: phydrax.metrix.symplectic_to_poisson

::: phydrax.metrix.hamiltonian_vector_field

::: phydrax.metrix.poisson_bracket

::: phydrax.metrix.casimir_residual

::: phydrax.metrix.validate_symplectic_form

::: phydrax.metrix.validate_poisson_structure

::: phydrax.operators.hamiltonian_vector_field

::: phydrax.operators.poisson_bracket

::: phydrax.operators.canonical_poisson_bracket

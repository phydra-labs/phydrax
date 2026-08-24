# Boundary-only harmonic fitting in four dimensions

This recipe fits a four-dimensional harmonic field without an interior PDE penalty.
The manufactured affine target belongs exactly to the degree-one harmonic basis, so a
fixed boundary realization lowers to one real linear least-squares problem.

```python
import jax.numpy as jnp
import jax.random as jr
import phydrax as phx

n = 4
domain = phx.domain.HyperRectangle(
    (-1.0,) * n,
    (1.0,) * n,
)

basis = phx.equations.HarmonicPolynomialBasis(n, 1)
model = phx.equations.LinearTrefftzField(basis)
u = domain.Model("x")(model)

coefficients = jnp.asarray([0.2, -0.4, 0.7, 0.1])
target = domain.Function("x")(
    lambda x: 0.3 + jnp.dot(coefficients, x)
)

boundary = domain.component({"x": phx.domain.Boundary()})
condition = phx.conditions.Dirichlet("u", boundary, target=target)
realization = phx.integration.materialize(
    phx.integration.mean_over(boundary),
    phx.domain.PointSampling(256),
    key=jr.key(0),
)
term = phx.terms.ResidualPenalty(
    condition,
    phx.integration.fixed(realization),
)
solver = phx.solver.FunctionalSolver(
    functions={"u": u},
    terms=(term,),
    enforcement=None,
)

fit = phx.solver.solve_linear_trial_space(solver, key=jr.key(1))
assert bool(fit.valid)
solver = fit.solver

interior = domain.component().sample(
    phx.domain.PointSampling(64),
    key=jr.key(2),
)
predicted = jnp.asarray(solver["u"](interior).data)
expected = jnp.asarray(target(interior).data)
assert jnp.allclose(predicted, expected, atol=1e-9, rtol=1e-9)

certificate = phx.equations.trial_space_certificate(solver["u"])
audit = phx.equations.audit_trial_space(solver["u"], interior)
assert certificate.equation_family == "laplace"
assert bool(audit.valid)
```

There is deliberately no `conditions.Residual` term for the Laplace equation. The basis
construction supplies that invariant. Boundary data still use normal Phydrax condition,
measure, term, and solver contracts.

Do not compile the boundary condition through `phx.enforcement`: its generic correction
need not remain harmonic, and Phydrax rejects enforcement on the certified field.

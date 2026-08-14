# Mechanics and Deep Ritz energy

This recipe minimizes the Poisson energy

$$
\mathcal J[u]=\int_0^1\left(\frac12(u')^2-u\right)\,dx,
\qquad u(0)=u(1)=0.
$$

Its Euler equation is $-u''=1$, with solution
$u^*(x)=\tfrac12x(1-x)$. The one-parameter ansatz below enforces both Dirichlet
conditions exactly. `IntegralFunctional` returns the signed energy; it does not square
the integrand.

```python
import jax.numpy as jnp
import jax.random as jr
import optax
import phydrax as phx

geom = phx.domain.Interval1d(0.0, 1.0)


@geom.Function("x")
def x_coordinate(x):
    return x[0]


# u_a(x) = a x(1-x) satisfies the boundary conditions for every a.
amplitude = geom.Parameter(0.0)
u = amplitude * x_coordinate * (1.0 - x_coordinate)


def energy_density(functions):
    field = functions["u"]
    du = phx.operators.grad(field, var="x")
    du_sq = phx.operators.einsum("...i,...i->...", du, du)
    return 0.5 * du_sq - field


energy = phx.terms.IntegralFunctional(
    target=phx.integration.over(geom.component()),
    plan=phx.integration.FixedQuadraturePlan(phx.integration.GaussLegendreRule(16)),
    integrand=energy_density,
    materialization_policy="fixed",
    label="poisson_energy",
)
solver = phx.solver.FunctionalSolver(
    functions={"u": u},
    terms=[energy],
)

trained = solver.solve(
    num_iter=16,
    optim=optax.sgd(1.5),
    seed=0,
    jit=True,
    keep_best=True,
    log_every=0,
)

final_energy = trained.loss(key=jr.key(1))
midpoint = trained["u"].func(jnp.asarray([0.5]))
assert jnp.allclose(final_energy, -1.0 / 24.0, atol=5e-5)
assert jnp.allclose(midpoint, 0.125, atol=5e-3)
```

The same solver can mix signed functionals with residual penalties:

$$
\text{loss}=\sum_i\ell_i+\sum_j\mathcal J_j+\sum_k r_k.
$$

Here $\ell_i$ are condition penalties, $\mathcal J_j$ are raw signed functional terms,
and $r_k$ are model-level losses. Essential conditions should normally be enforced
before evaluating a Ritz energy; otherwise the optimization problem is not the stated
variational problem.

For trajectory mechanics, construct `euler_lagrange(...)` or
`canonical_hamiltonian_residual(...)`, represent its zero requirement with
`phx.conditions.Residual`, select an explicit `phx.integration` source, and evaluate
it with `phx.terms.ResidualPenalty`. See
[Lagrangian and Hamiltonian mechanics](../guides_mechanics.md).

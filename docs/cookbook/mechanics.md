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
    optim=optax.sgd(3.0),
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

## Compressible finite-strain energy

For a conservative hyperelastic solid, represent internal and external work as
separate signed terms because they live on different measures:

```python
geom = phx.domain.GeometryDomain(
    phx.geometry.Square(center=(0.5, 0.5), side=1.0).compile()
)
full_boundary = geom.component({"x": phx.domain.Boundary()})
mu = 1.0
lambda_ = 4.0 - 2.0 * mu / 3.0


@geom.Function("x")
def traction(x):
    return jnp.where(
        x[0] > 1.0 - 1e-10,
        jnp.asarray((0.1, 0.0)),
        jnp.zeros(2),
    )


raw = geom.Model("x")(
    phx.nn.models.MLP(
        in_size=2,
        out_size=2,
        width_size=24,
        depth=3,
        key=jr.key(0),
    )
)
x0 = geom.Function("x")(lambda x: x[0])
u = x0 * raw  # Exact zero displacement on x[0] == 0.

internal = phx.terms.IntegralFunctional(
    target=phx.integration.over(geom.component()),
    plan=phx.integration.MonteCarloPlan(4096),
    integrand=lambda functions: phx.operators.neo_hookean_reference_energy(
        functions["u"],
        mu=mu,
        lambda_=lambda_,
    ),
    materialization_policy="fixed",
    fixed_key=jr.key(1),
    nonfinite_integrand="propagate",
)
external = phx.terms.IntegralFunctional(
    target=phx.integration.over(full_boundary),
    plan=phx.integration.MonteCarloPlan(1024),
    integrand=lambda functions: phx.operators.einsum(
        "...i,...i->...", traction, functions["u"]
    ),
    weight=-1.0,
    materialization_policy="fixed",
    fixed_key=jr.key(2),
    nonfinite_integrand="propagate",
)
trained = phx.solver.FunctionalSolver(
    functions={"u": u},
    terms=(internal, external),
).solve(
    num_iter=60,
    optim=optax.lbfgs(learning_rate=1.0),
    keep_best=False,
    log_every=0,
)
```

Here `lambda_` is Lamé's first parameter. The physical bulk modulus is
$K=\lambda+2\mu/3$. A two-component field is plane strain, while a
three-component field is fully three-dimensional. Plane stress and exact
incompressibility are separate constitutive problems.

The admissible material domain requires $J=\det(I+\nabla u)>0$. Propagating a
nonfinite integrand lets the line search reject an inverted trial; it does not
repair the trial. Qualify the returned field on independent quadrature and report
minimum $J$, strong equilibrium, reactions, and stress—not only the sampled
training potential.


For trajectory mechanics, construct `euler_lagrange(...)` or
`canonical_hamiltonian_residual(...)`, represent its zero requirement with
`phx.conditions.Residual`, select an explicit `phx.integration` source, and evaluate
it with `phx.terms.ResidualPenalty`. See
[Lagrangian and Hamiltonian mechanics](../guides_mechanics.md).

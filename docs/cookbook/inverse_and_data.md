# Inverse problems + hybrid physics–data

This recipe illustrates a common “SciML inverse” pattern: learn a state \(u\) and an unknown coefficient/parameter
using a PDE residual plus data terms.

## Problem (example)

On a spatial domain \(\Omega\), consider

$$
-\nabla\cdot\bigl(k(x)\nabla u(x)\bigr)=f(x)\quad\text{in }\Omega,\qquad
u=g\quad\text{on }\partial\Omega,
$$

where \(k\) is unknown (either a scalar parameter or a field), and you also have sparse observations of \(u\).

## Pattern: treat unknowns as additional fields

In Phydrax, you typically represent unknowns as additional `DomainFunction`s and couple them inside the residual
operator. Everything is still “minimize functionals over domains”.

## Example skeleton (learn \(u_\theta\) and \(k_\phi\)) {: data-toc-label="Example skeleton (learn u_θ and k_φ)"}

!!! example
    ```python
    import jax
    import jax.numpy as jnp
    import jax.random as jr
    import optax
    import phydrax as phx

    geom = phx.domain.GeometryDomain(
        phx.geometry.Square(center=(0.0, 0.0), side=2.0).compile()
    )  # Ω=[-1,1]^2


    # Known forcing and boundary value (toy choices)
    @geom.Function("x")
    def f(x):
        return 1.0


    @geom.Function("x")
    def g(x):
        return 0.0


    # State u(x) and unknown coefficient k(x) (positive via final activation)
    u_model = phx.nn.models.MLP(
        in_size=2, out_size="scalar", width_size=16, depth=2, key=jr.key(0)
    )
    k_model = phx.nn.models.MLP(
        in_size=2,
        out_size="scalar",
        width_size=16,
        depth=2,
        final_activation=jax.nn.softplus,
        key=jr.key(1),
    )

    u = geom.Model("x")(u_model)
    k = geom.Model("x")(k_model)

    layout = phx.domain.SampleLayout((("x",),))
    interior = geom.component()


    def pde_operator(u_f, k_f):
        grad_u = phx.operators.grad(u_f, var="x")  # ∇u (vector)
        flux = k_f * grad_u  # k∇u
        return -phx.operators.div(flux, var="x") - f  # -∇·(k∇u) - f


    pde_condition = phx.conditions.Residual(
        ("u", "k"),
        interior,
        pde_operator,
    )
    pde = phx.terms.ResidualPenalty(
        pde_condition,
        phx.integration.per_step(
            phx.integration.mean_over(pde_condition.on),
            phx.domain.PointSampling(128, layout=layout),
        ),
    )

    boundary = geom.component({"x": phx.domain.Boundary()})
    boundary_condition = phx.conditions.Dirichlet("u", boundary, target=g)
    bc = phx.terms.ResidualPenalty(
        boundary_condition,
        phx.integration.per_step(
            phx.integration.mean_over(boundary_condition.on),
            phx.domain.PointSampling(64, layout=layout),
        ),
        scale=10.0,
    )

    # Optional anchor data for u at scattered points
    anchors = jnp.array([[0.0, 0.0], [0.5, -0.25]])
    values = jnp.array([0.0, 0.1])


    @geom.Function("x")
    def observed_u(x):
        nearest = jnp.argmin(jnp.sum((anchors - x) ** 2, axis=-1))
        return values[nearest]


    observation = phx.conditions.Observation("u", interior, observed_u)
    observation_batch = interior.points({"x": anchors})
    data = phx.terms.ObservationPenalty(
        observation,
        phx.integration.fixed(
            phx.integration.from_samples(
                phx.integration.mean_over(observation.on),
                observation_batch,
            )
        ),
        scale=1.0,
    )

    solver = phx.solver.FunctionalSolver(
        functions={"u": u, "k": k},
        terms=(pde, bc, data),
    )
    solver = solver.solve(num_iter=20, optim=optax.adam(1e-3), seed=0)
    ```

## Notes

- For scalar unknown parameters (global coefficients), consider `domain.Parameter(...)` for a trainable constant.
- To enforce \(u=g\) exactly, replace the boundary penalty with an
  `phx.enforcement.EnforcementSpec` (see the Poisson recipe).
- For sensor tracks over time, construct their paired coordinates with
  `component.points(...)` and use a fixed `ObservationPenalty` source.
- When the forward model is an implicit index-one system rather than a learned
  residual field, use `DifferentialAlgebraicProblem` with a prepared fixed-grid
  `solve_dae`. Runtime parameter PyTrees and consistent initial values remain
  differentiable through the accepted BDF stages; solver status and work diagnostics
  remain explicit. See
  [Differential-algebraic equation integration](../api/solver/differential_algebraic.md).

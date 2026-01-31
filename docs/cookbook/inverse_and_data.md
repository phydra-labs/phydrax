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

    geom = phx.domain.Square(center=(0.0, 0.0), side=2.0)  # Ω=[-1,1]^2

    # Known forcing and boundary value (toy choices)
    @geom.Function("x")
    def f(x):
        return 1.0

    @geom.Function("x")
    def g(x):
        return 0.0

    # State u(x) and unknown coefficient k(x) (positive via final activation)
    u_model = phx.nn.MLP(in_size=2, out_size="scalar", width_size=16, depth=2, key=jr.key(0))
    k_model = phx.nn.MLP(
        in_size=2,
        out_size="scalar",
        width_size=16,
        depth=2,
        final_activation=jax.nn.softplus,
        key=jr.key(1),
    )

    u = geom.Model("x")(u_model)
    k = geom.Model("x")(k_model)

    structure = phx.domain.ProductStructure((("x",),))

    def pde_operator(u_f, k_f):
        grad_u = phx.operators.grad(u_f, var="x")      # ∇u (vector)
        flux = k_f * grad_u                            # k∇u
        return -phx.operators.div(flux, var="x") - f    # -∇·(k∇u) - f

    pde = phx.constraints.ContinuousPointwiseInteriorConstraint(
        ("u", "k"),
        geom,
        operator=pde_operator,
        num_points=128,
        structure=structure,
        reduction="mean",
    )

    boundary = geom.component({"x": phx.domain.Boundary()})
    bc = phx.constraints.ContinuousDirichletBoundaryConstraint(
        "u",
        boundary,
        target=g,
        num_points=64,
        structure=structure,
        weight=10.0,
    )

    # Optional anchor data for u at scattered points
    anchors = jnp.array([[0.0, 0.0], [0.5, -0.25]])
    values = jnp.array([0.0, 0.1])
    data = phx.constraints.DiscreteInteriorDataConstraint(
        "u",
        geom,
        points={"x": anchors},
        values=values,
        weight=1.0,
    )

    solver = phx.solver.FunctionalSolver(functions={"u": u, "k": k}, constraints=[pde, bc, data])
    solver = solver.solve(num_iter=20, optim=optax.adam(1e-3), seed=0)
    ```

## Notes

- For scalar unknown parameters (global coefficients), consider `domain.Parameter(...)` for a trainable constant.
- If you want to enforce \(u=g\) exactly, replace the boundary penalty with an enforced term (see the Poisson recipe).
- For sensor tracks over time, use `DiscreteInteriorDataConstraint(..., sensors=..., times=..., sensor_values=...)`.

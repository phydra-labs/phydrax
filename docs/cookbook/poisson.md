# Poisson on a square (field learning)

This recipe solves a simple elliptic PDE by learning a field \(u_\theta(x)\).

## Problem

Let \(\Omega=[-1,1]^2\). We solve

$$
\Delta u = 4 \quad \text{in }\Omega,\qquad
u = g \quad \text{on }\partial\Omega,
$$

with \(g(x,y)=x^2+y^2\). The exact solution is \(u^\star(x,y)=x^2+y^2\).

## Domain and components

- Domain: `Square(center=(0,0), side=2)` (label `"x"` with \(d=2\)).
- Interior component: `geom.component()` (default).
- Boundary component: `geom.component({"x": Boundary()})`.

## Soft PINN (PDE residual + boundary penalty)

!!! example
    ```python
    import jax.random as jr
    import optax
    import phydrax as phx

    geom = phx.domain.Square(center=(0.0, 0.0), side=2.0)

    @geom.Function("x")
    def g(x):
        return x[0] ** 2 + x[1] ** 2

    model = phx.nn.MLP(in_size=2, out_size="scalar", width_size=16, depth=2, key=jr.key(0))
    u = geom.Model("x")(model)

    structure = phx.domain.ProductStructure((("x",),))

    pde = phx.constraints.ContinuousPointwiseInteriorConstraint(
        "u",
        geom,
        operator=lambda f: phx.operators.laplacian(f, var="x") - 4.0,
        num_points=64,
        structure=structure,
        reduction="mean",
    )

    boundary = geom.component({"x": phx.domain.Boundary()})
    bc = phx.constraints.ContinuousDirichletBoundaryConstraint(
        "u",
        boundary,
        target=g,
        num_points=32,
        structure=structure,
        weight=10.0,
        reduction="mean",
    )

    solver = phx.solver.FunctionalSolver(functions={"u": u}, constraints=[pde, bc])
    solver = solver.solve(num_iter=20, optim=optax.adam(1e-3), seed=0)
    ```

## Enforced Dirichlet boundary (ansatz + interior residual)

Instead of penalizing boundary mismatch, enforce \(u=g\) by construction using
`enforce_dirichlet`, staged by an `EnforcedConstraintPipeline` (via `constraint_terms=...`).

!!! example
    ```python
    import jax.random as jr
    import optax
    import phydrax as phx

    geom = phx.domain.Square(center=(0.0, 0.0), side=2.0)

    @geom.Function("x")
    def g(x):
        return x[0] ** 2 + x[1] ** 2

    model = phx.nn.MLP(in_size=2, out_size="scalar", width_size=16, depth=2, key=jr.key(0))
    u = geom.Model("x")(model)

    structure = phx.domain.ProductStructure((("x",),))
    pde = phx.constraints.ContinuousPointwiseInteriorConstraint(
        "u",
        geom,
        operator=lambda f: phx.operators.laplacian(f, var="x") - 4.0,
        num_points=64,
        structure=structure,
    )

    boundary = geom.component({"x": phx.domain.Boundary()})
    term = phx.solver.SingleFieldEnforcedConstraint(
        "u",
        boundary,
        lambda f: phx.constraints.enforce_dirichlet(f, boundary, var="x", target=g),
    )

    solver = phx.solver.FunctionalSolver(
        functions={"u": u},
        constraints=[pde],
        constraint_terms=[term],
        boundary_weight_num_reference=128,
        boundary_weight_key=jr.key(1),
    )
    solver = solver.solve(num_iter=20, optim=optax.adam(1e-3), seed=0)
    ```

## Grid evaluation (coord-separable batches)

When you want axis-aware evaluation (e.g. spectral/basis derivatives, operator learning), sample coordinate axes and
evaluate on the implied Cartesian grid. For a 2D geometry label `"x"`, coord-separable sampling provides a tuple
\((x_{\text{axis}},y_{\text{axis}})\).

!!! example
    ```python
    import jax.random as jr
    import phydrax as phx

    # Basis/FD backends require the field to support structured inputs (a tuple of 1D axes).
    # Use a structured model like SeparableMLP/FNO/DeepONet for grid-native evaluation.
    geom = phx.domain.Square(center=(0.0, 0.0), side=2.0)
    model = phx.nn.SeparableMLP(
        in_size=2,
        out_size="scalar",
        latent_size=16,
        width_size=16,
        depth=2,
        key=jr.key(0),
    )
    u = geom.Model("x")(model)

    batch = geom.component().sample_coord_separable({"x": (32, 32)}, key=jr.key(1))

    # Evaluate a basis-aware Laplacian on the grid.
    du = phx.operators.laplacian(u, var="x", backend="basis", basis="poly")
    out = du(batch)
    ```

See [Guides → Domains and sampling](../guides_domain.md) and [Guides → Differential operators](../guides_differential.md).

# Heat equation (space–time)

This recipe shows a simple parabolic PDE on a space–time product domain, including an initial condition term and an
optional sensor/anchor data term.

## Problem

On \(\Omega=[0,1]\times[0,T]\), solve

$$
\partial_t u - \alpha\,\partial_{xx}u = 0,
$$

with boundary/initial conditions

$$
u(0,t)=u(1,t)=0,\qquad u(x,0)=u_0(x)=\sin(\pi x).
$$

## Domain and fields

Let \(x\in[0,1]\), \(t\in[0,T]\). In Phydrax:

- `Interval1d(0,1)` is a 1D geometry (label `"x"`).
- `TimeInterval(0,T)` is a scalar domain (label `"t"`).
- `domain = geom @ time` is the product.

## A basic training setup (soft BC + initial term)

!!! example
    ```python
    import jax.numpy as jnp
    import jax.random as jr
    import optax
    import phydrax as phx

    alpha = 0.1
    T = 1.0

    geom = phx.domain.Interval1d(0.0, 1.0)
    time = phx.domain.TimeInterval(0.0, T)
    domain = geom @ time

    def u0(x):
        return jnp.sin(jnp.pi * x[0])

    model = phx.nn.MLP(in_size=2, out_size="scalar", width_size=16, depth=2, key=jr.key(0))
    u = domain.Model("x", "t")(model)

    structure_xt = phx.domain.ProductStructure((("x", "t"),))
    structure_x = phx.domain.ProductStructure((("x",),))

    # PDE residual: u_t - alpha * u_xx = 0
    pde = phx.constraints.ContinuousPointwiseInteriorConstraint(
        "u",
        domain,
        operator=lambda f: phx.operators.dt(f, var="t") - alpha * phx.operators.laplacian(f, var="x"),
        num_points=128,
        structure=structure_xt,
        reduction="mean",
    )

    # Dirichlet boundary at x endpoints (soft)
    boundary = domain.component({"x": phx.domain.Boundary()})
    bc = phx.constraints.ContinuousDirichletBoundaryConstraint(
        "u",
        boundary,
        target=0.0,
        num_points=64,
        structure=structure_xt,
        weight=10.0,
        reduction="mean",
    )

    # Initial condition u(x,0) = u0(x)
    ic = phx.constraints.ContinuousInitialFunctionConstraint(
        "u",
        domain,
        func=u0,
        evolution_var="t",
        time_derivative_order=0,
        num_points=32,
        structure=structure_x,
        weight=10.0,
        reduction="mean",
    )

    solver = phx.solver.FunctionalSolver(functions={"u": u}, constraints=[pde, bc, ic])
    solver = solver.solve(num_iter=20, optim=optax.adam(1e-3), seed=0)
    ```

### Higher-order initial data

To constrain \(\partial_t^k u(\cdot,0)\) for \(k>0\), use `time_derivative_order=k`. For high-order time derivatives,
`time_derivative_backend="jet"` can be more direct than nested Jacobians.

## Adding sensors (time tracks) or anchors (scattered data)

For discrete measurements, add a data-fit constraint alongside the PDE terms. Phydrax supports:

- **Anchors**: scattered \((x_i,t_i)\mapsto y_i\).
- **Sensor tracks**: fixed sensors \(x_m\) with measurements over time \(y_m(t_j)\).

!!! example
    Sensor tracks via `DiscreteInteriorDataConstraint`:

    ```python
    import jax.numpy as jnp
    import phydrax as phx

    sensors = jnp.array([[0.25], [0.75]])     # M sensors in 1D
    times = jnp.linspace(0.0, T, 51)          # T time points
    sensor_values = jnp.zeros((2, 51))        # shape (M, T) for scalar u

    data = phx.constraints.DiscreteInteriorDataConstraint(
        "u",
        domain,
        sensors=sensors,
        times=times,
        sensor_values=sensor_values,
        num_points=256,
        structure=structure_xt,
        weight=1.0,
    )
    ```

See [API → Constraints → Discrete](../api/constraints/discrete.md).

## Compile the PDE IR to method-of-lines dynamics

For an array solver, the same equation can be represented once as PDE IR and
compiled against an existing spatial discretization. A sine basis encodes the
homogeneous Dirichlet boundary condition; the compiler therefore rejects a
periodic coordinate or a Neumann boundary condition paired with this basis.

!!! example
    ```python
    import jax
    import jax.numpy as jnp
    import phydrax as phx

    x = phx.equations.PDECoordinate(
        "x", "space", bounds=(0.0, 1.0), periodic=False
    )
    t = phx.equations.PDECoordinate("t", "time", bounds=(0.0, 1.0))
    field = phx.equations.PDEField("u", coordinates=("x", "t"))
    diffusivity = phx.equations.PDEParameter("alpha", value=0.1)
    u = phx.equations.PDEExpression.field("u")

    boundary_region = phx.equations.PDERegion(
        "x-boundary", "boundary", ("x",)
    )
    problem = phx.equations.PDEProblemIR(
        coordinates=(x, t),
        fields=(field,),
        parameters=(diffusivity,),
        equations=(
            phx.equations.PDEEquation(
                "heat",
                u.derivative("t"),
                phx.equations.PDEExpression.parameter("alpha")
                * u.laplacian("x"),
            ),
        ),
        conditions=(
            phx.equations.PDECondition(
                "homogeneous-dirichlet",
                "boundary",
                u,
                region="x-boundary",
                coordinate="x",
            ),
        ),
        regions=(boundary_region,),
    )

    axis = phx.domain.SineAxisSpec(64).materialize(0.0, 1.0)
    space = phx.solver.TensorGridDiscretization((axis,))
    dynamics = phx.equations.compile_semidiscrete_pde(problem, space)

    initial = jnp.sin(jnp.pi * axis.nodes)
    drift = jax.jit(
        lambda state, alpha: dynamics(0.0, state, {"alpha": alpha})
    )(initial, jnp.asarray(0.1))

    assert drift.shape == initial.shape
    assert dynamics.semilinear_drift is not None
    print(dynamics.compilation_id, dynamics.resolved_method)
    ```

`dynamics` has the solver-compatible signature `(time, state, args) -> state`.
Scalar single-field problems retain the spatial state shape; multiple or vector
fields use a static trailing-component packing described by
`dynamics.layout`. Runtime parameter mappings remain differentiable.

Nonhomogeneous Dirichlet or Neumann data require an explicit
`phx.equations.BoundaryLift`. The evolved state is then the homogeneous residual
and `dynamics.physical_state(time, state, args)` reconstructs the physical
field. This explicit split prevents a nonperiodic boundary from being silently
treated as periodic.

`TensorGridDiscretization` also exposes `partial_derivative`, `gradient`,
`divergence`, `curl`, and `integral` while preserving trailing field/component
axes. A direct vector field uses `divergence(vector)`. The gradient's
finite-difference and sine/cosine dual representation is explicit:
`divergence(gradient, dual=True)` gives the discrete Laplacian. PDE IR
`divergence(gradient(...))` tracks this representation automatically.
Functional parameters accept either component constants or arrays aligned with
the full spatial shape. Compilation IDs and semilinear operator IDs include
resolved parameter values and boundary-lift IDs, so cached artifacts cannot be
reused across different bound dynamics.

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


    model = phx.nn.models.MLP(
        in_size=2, out_size="scalar", width_size=16, depth=2, key=jr.key(0)
    )
    u = domain.Model("x", "t")(model)

    layout_xt = phx.domain.SampleLayout((("x", "t"),))
    layout_x = phx.domain.SampleLayout((("x",),))

    # PDE residual: u_t - alpha * u_xx = 0
    interior = domain.component()
    pde_condition = phx.conditions.Residual(
        "u",
        interior,
        lambda f: phx.operators.dt(f, var="t") - alpha * phx.operators.laplacian(f, var="x"),
    )
    pde = phx.terms.ResidualPenalty(
        pde_condition,
        phx.integration.per_step(
            phx.integration.mean_over(pde_condition.on),
            phx.domain.PointSampling(128, layout=layout_xt),
        ),
    )

    # Dirichlet boundary at x endpoints (soft)
    boundary = domain.component({"x": phx.domain.Boundary()})
    boundary_condition = phx.conditions.Dirichlet("u", boundary, target=0.0)
    bc = phx.terms.ResidualPenalty(
        boundary_condition,
        phx.integration.per_step(
            phx.integration.mean_over(boundary_condition.on),
            phx.domain.PointSampling(64, layout=layout_xt),
        ),
        scale=10.0,
    )

    # Initial condition u(x,0) = u0(x)
    initial_slice = domain.component({"t": phx.domain.FixedStart()})
    initial_condition = phx.conditions.Initial(
        "u",
        initial_slice,
        target=u0,
        evolution_var="t",
        order=0,
    )
    ic = phx.terms.ResidualPenalty(
        initial_condition,
        phx.integration.per_step(
            phx.integration.mean_over(initial_condition.on),
            phx.domain.PointSampling(32, layout=layout_x),
        ),
        scale=10.0,
    )

    solver = phx.solver.FunctionalSolver(functions={"u": u}, terms=(pde, bc, ic))
    solver = solver.solve(num_iter=20, optim=optax.adam(1e-3), seed=0)
    ```

### Higher-order initial data

To constrain \(\partial_t^k u(\cdot,0)\) for \(k>0\), set `order=k` on
`phx.conditions.Initial`. For high-order time derivatives, `backend="jet"` can
be more direct than nested Jacobians.

## Adding sensors (time tracks) or anchors (scattered data)

For discrete measurements, add an observation penalty alongside the PDE terms.
Phydrax supports:

- **Anchors**: scattered \((x_i,t_i)\mapsto y_i\).
- **Sensor tracks**: fixed sensors \(x_m\) with measurements over time \(y_m(t_j)\).

!!! example
    Sensor tracks become an explicit finite `PointBatch` and fixed integration
    source:

    ```python
    import jax.numpy as jnp
    import phydrax as phx

    sensors = jnp.array([[0.25], [0.75]])  # M sensors in 1D
    times = jnp.linspace(0.0, T, 51)  # T time points
    sensor_values = jnp.zeros((2, 51))  # shape (M, T) for scalar u


    @domain.Function("x", "t")
    def observed_temperature(x, t):
        sensor_index = jnp.argmin(jnp.sum((sensors - x) ** 2, axis=-1))
        time_index = jnp.argmin(jnp.abs(times - t))
        return sensor_values[sensor_index, time_index]


    track_x = jnp.repeat(sensors, times.size, axis=0)
    track_t = jnp.tile(times, sensors.shape[0])
    observation_index = jnp.arange(256) % track_t.size
    observation_batch = domain.component().points(
        {"x": track_x[observation_index], "t": track_t[observation_index]}
    )
    observation = phx.conditions.Observation("u", domain.component(), observed_temperature)
    observation_source = phx.integration.fixed(
        phx.integration.from_samples(
            phx.integration.mean_over(observation.on),
            observation_batch,
        )
    )
    data = phx.terms.ObservationPenalty(
        observation,
        observation_source,
        scale=1.0,
    )
    ```

See [Guide → Conditions, integration, terms, and enforcement](../guides_conditions.md).

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

    x = phx.equations.PDECoordinate("x", "space", bounds=(0.0, 1.0), periodic=False)
    t = phx.equations.PDECoordinate("t", "time", bounds=(0.0, 1.0))
    field = phx.equations.PDEField("u", coordinates=("x", "t"))
    diffusivity = phx.equations.PDEParameter("alpha", value=0.1)
    u = phx.equations.PDEExpression.field("u")

    boundary_region = phx.equations.PDERegion("x-boundary", "boundary", ("x",))
    problem = phx.equations.PDEProblemIR(
        coordinates=(x, t),
        fields=(field,),
        parameters=(diffusivity,),
        equations=(
            phx.equations.PDEEquation(
                "heat",
                u.derivative("t"),
                phx.equations.PDEExpression.parameter("alpha") * u.laplacian("x"),
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

    axis = phx.discretization.SineAxisSpec(64).materialize(0.0, 1.0)
    space = phx.discretization.TensorSpectralDiscretization.from_axes((axis,))
    dynamics = phx.equations.compile_semidiscrete_pde(problem, space)

    initial = jnp.sin(jnp.pi * axis.nodes)
    drift = jax.jit(lambda state, alpha: dynamics(0.0, state, {"alpha": alpha}))(
        initial, jnp.asarray(0.1)
    )

    assert drift.shape == initial.shape
    assert dynamics.semilinear_drift is not None
    print(dynamics.compilation_id, dynamics.resolved_method)
    ```

`dynamics` has the solver-compatible signature `(time, state, args) -> state`.
Without a spectral method argument, this bounded linear example selects the
point-value strong-form path, so scalar single-field problems retain the physical
spatial shape. Multiple or vector fields use a static trailing-component packing
described by `dynamics.layout`. Runtime parameter mappings remain differentiable.

Nonhomogeneous Dirichlet or Neumann data require an explicit
`phx.equations.BoundaryLift`. The evolved state is then the homogeneous residual
and `dynamics.physical_state(time, state, args)` reconstructs the physical
field. This explicit split prevents a nonperiodic boundary from being silently
treated as periodic.

`TensorSpectralDiscretization` exposes physical `partial_derivative`, `gradient`,
`divergence`, `curl`, and `integral` conveniences while its primary scientific state
space is modal. Pass an explicit `PseudospectralMethodPlan` to
`compile_semidiscrete_pde` for coefficient-resident compilation, then use
`project_state` and `reconstruct_state` at representation boundaries. Nonlinear
coefficient-resident compilation requires an explicit dealiasing policy.

Uniform finite-difference grids use the separate native FD compiler and explicit
stencil composition. Functional parameters accept either component constants or
arrays aligned with the full spatial shape. Compilation IDs and semilinear operator
IDs include resolved parameter values and boundary-lift IDs, so cached artifacts
cannot be reused across different bound dynamics.

Bounded PDE coordinates must match the materialized tensor-grid interval.
Semidiscrete volume integrals accept interior spatial regions; boundary, interface,
and component-specific boundary semantics are rejected rather than being approximated
as whole-domain operations. Sine and cosine derivative evaluators track primal and
dual extension parity through compatible expression trees. Ambiguous differentiated
composites are rejected instead of being assigned a silent boundary extension.

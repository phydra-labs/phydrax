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

#
#  Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import jax.numpy as jnp
import jax.random as jr
import optax

from phydrax.constraints import PointSetConstraint
from phydrax.domain import Interval1d
from phydrax.operators import build_mfd_cloud_plan, partial_n
from phydrax.solver import FunctionalSolver


def test_pointset_penalty_mean_and_sum():
    geom = Interval1d(0.0, 1.0)
    component = geom.component()

    points = {"x": jnp.array([[0.0], [0.5], [1.0]], dtype=float)}
    u = geom.Function()(0.0)

    def residual(functions):
        return functions["u"] - 1.0

    c_mean = PointSetConstraint.from_points(
        component=component,
        points=points,
        residual=residual,
        reduction="mean",
        weight=2.0,
    )
    loss_mean = c_mean.loss({"u": u}, key=jr.key(0))
    assert jnp.allclose(loss_mean, 2.0)

    c_sum = PointSetConstraint.from_points(
        component=component,
        points=points,
        residual=residual,
        reduction="sum",
        weight=2.0,
    )
    loss_sum = c_sum.loss({"u": u}, key=jr.key(0))
    assert jnp.allclose(loss_sum, 6.0)


def _x_values(x):
    x_arr = jnp.asarray(x, dtype=float)
    if x_arr.ndim == 0:
        return x_arr.reshape(())
    if x_arr.ndim == 1:
        if int(x_arr.shape[0]) == 1:
            return x_arr[0]
        return x_arr
    if x_arr.ndim == 2 and int(x_arr.shape[1]) == 1:
        return x_arr[:, 0]
    raise ValueError(f"Unsupported x shape {x_arr.shape}.")


def test_pointset_eval_kwargs_enable_mfd_cloud_mode():
    geom = Interval1d(0.0, 1.0)
    component = geom.component()
    coords = jnp.linspace(0.0, 1.0, 64)
    points = {"x": coords.reshape((-1, 1))}

    @geom.Function("x")
    def u(x):
        xx = _x_values(x)
        return xx**3

    @geom.Function("x")
    def dudx_exact(x):
        xx = _x_values(x)
        return 3.0 * xx**2

    plan = build_mfd_cloud_plan(points["x"], order=1, k=5)

    constraint = PointSetConstraint.from_points(
        component=component,
        points=points,
        residual=lambda fns: partial_n(fns["u"], var="x", order=1, backend="mfd")
        - dudx_exact,
        constraint_vars=("u",),
        reduction="mean",
        eval_kwargs={"mfd_mode": "cloud", "mfd_cloud_plan": plan},
    )
    loss = constraint.loss({"u": u}, key=jr.key(0))
    assert jnp.allclose(loss, 0.0, rtol=1e-7, atol=1e-7)


def test_pointset_from_operator_cloud_mode_sum_reduction():
    geom = Interval1d(0.0, 1.0)
    component = geom.component()
    coords = jnp.linspace(0.0, 1.0, 32)
    points = {"x": coords.reshape((-1, 1))}

    @geom.Function("x")
    def u(x):
        xx = _x_values(x)
        return xx**3

    @geom.Function("x")
    def dudx_exact(x):
        xx = _x_values(x)
        return 3.0 * xx**2

    plan = build_mfd_cloud_plan(points["x"], order=1, k=5)
    points_batch = PointSetConstraint.from_points(
        component=component,
        points=points,
        residual=lambda fns: fns["u"] - fns["u"],
    ).points
    constraint = PointSetConstraint.from_operator(
        points=points_batch,
        operator=lambda f: partial_n(f, var="x", order=1, backend="mfd") - dudx_exact,
        constraint_vars="u",
        reduction="sum",
        eval_kwargs={"mfd_mode": "cloud", "mfd_cloud_plan": plan},
    )
    loss = constraint.loss({"u": u}, key=jr.key(0))
    assert jnp.allclose(loss, 0.0, rtol=1e-7, atol=1e-7)


def test_pointset_cloud_mode_solver_jit_smoke():
    geom = Interval1d(0.0, 1.0)
    component = geom.component()
    coords = jnp.linspace(0.0, 1.0, 40)
    points = {"x": coords.reshape((-1, 1))}

    @geom.Function("x")
    def u(x):
        xx = _x_values(x)
        return xx**3

    @geom.Function("x")
    def dudx_exact(x):
        xx = _x_values(x)
        return 3.0 * xx**2

    plan = build_mfd_cloud_plan(points["x"], order=1, k=5)
    constraint = PointSetConstraint.from_points(
        component=component,
        points=points,
        residual=lambda fns: partial_n(fns["u"], var="x", order=1, backend="mfd")
        - dudx_exact,
        constraint_vars=("u",),
        reduction="mean",
        eval_kwargs={"mfd_mode": "cloud", "mfd_cloud_plan": plan},
    )
    solver = FunctionalSolver(functions={"u": u}, constraints=[constraint])
    init_loss = solver.loss(key=jr.key(0))
    trained = solver.solve(
        num_iter=2,
        optim=optax.adam(1e-3),
        seed=0,
        jit=True,
        keep_best=False,
        log_every=0,
    )
    final_loss = trained.loss(key=jr.key(1))
    assert jnp.isfinite(init_loss)
    assert jnp.isfinite(final_loss)

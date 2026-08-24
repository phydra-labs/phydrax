#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import jax
import jax.numpy as jnp
import pytest
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

import phydrax as phx


def _negative_entropy_geometry(dimension):
    primal_chart = phx.metrix.CoordinateChart(
        "functional-positive",
        tuple(f"x{index}" for index in range(dimension)),
    )
    dual_chart = phx.metrix.CoordinateChart(
        "functional-log",
        tuple(f"y{index}" for index in range(dimension)),
    )
    return phx.metrix.LegendreGeometry(
        phx.metrix.HessianGeometry(
            lambda point: jnp.sum(point * (jnp.log(point) - 1.0)),
            chart=primal_chart,
        ),
        jnp.exp,
        primal_support=phx.metrix.ChartSupport(
            primal_chart,
            lambda point: jnp.all(point > 0.0, axis=-1),
            support_id="functional-positive-support",
        ),
        dual_support=phx.metrix.ChartSupport(
            dual_chart,
            lambda point: jnp.all(jnp.isfinite(point), axis=-1),
            support_id="functional-log-support",
        ),
        geometry_id="functional-negative-entropy",
    )


def _solver_and_optimizer():
    domain = phx.domain.Interval1d(0.0, 1.0)
    target = jnp.asarray([0.8, 0.5, 1.7])
    positive = domain.Parameter(
        jnp.asarray([0.3, 1.4, 2.2]),
        transform=lambda value: jnp.sum(
            value * jnp.log(value / target) - value + target
        ),
    )
    offset = domain.Parameter(
        jnp.asarray(2.0),
        transform=lambda value: 0.5 * (value - 0.5) ** 2,
    )
    objective = phx.terms.IntegralFunctional(
        target=phx.integration.over(domain.component()),
        plan=phx.integration.FixedQuadraturePlan(phx.integration.GaussLegendreRule(4)),
        integrand=lambda functions: functions["positive"] + functions["offset"],
        materialization_policy="fixed",
    )
    solver = phx.solver.FunctionalSolver(
        functions={"positive": positive, "offset": offset},
        terms=(objective,),
    )
    parameters = solver.trainable_functions()
    positive_path = next(
        path
        for path in phx.optim.ParameterMirrorGeometry.array_leaf_paths(parameters)
        if "positive" in path
    )
    parameter_geometry = phx.optim.ParameterMirrorGeometry(
        parameters,
        {positive_path: _negative_entropy_geometry(3)},
    )
    return solver, phx.optim.mirror_descent(parameter_geometry, learning_rate=1.0)


def _trained_values(solver):
    leaves = tuple(
        jnp.asarray(leaf) for leaf in jax.tree.leaves(solver.trainable_functions())
    )
    positive = next(leaf for leaf in leaves if leaf.shape == (3,))
    offset = next(leaf for leaf in leaves if leaf.shape == ())
    return positive, offset


@pytest.mark.parametrize("jit", [False, True])
def test_functional_solver_trains_mixed_mirror_and_euclidean_parameters(jit):
    solver, optimizer = _solver_and_optimizer()
    initial_loss = solver.loss()
    trained = solver.solve(
        num_iter=1,
        optim=optimizer,
        keep_best=False,
        jit=jit,
        log_every=0,
    )
    positive, offset = _trained_values(trained)
    diagnostics = trained.training_diagnostics

    assert trained.loss() < initial_loss
    assert jnp.allclose(positive, jnp.asarray([0.8, 0.5, 1.7]), atol=1e-6)
    assert jnp.allclose(offset, 0.5, atol=1e-6)
    assert diagnostics["optimizer/mirror/num_legendre_leaves"] == 1
    assert diagnostics["optimizer/mirror/learning_rate"] == 1.0
    assert diagnostics["optimizer/mirror/bregman_step"] > 0.0
    assert diagnostics["optimizer/mirror/constraint_residual_max"] == 0.0
    assert not any(key.startswith("optimizer/riemannian/") for key in diagnostics)


def test_functional_solver_rejects_ambient_mirror_evaluation_transform():
    solver, optimizer = _solver_and_optimizer()
    with pytest.raises(ValueError, match="unsupported for mirror optimizers"):
        solver.solve(
            num_iter=1,
            optim=optimizer,
            evaluation_parameters=lambda state, parameters: parameters,
            log_every=0,
        )


def test_functional_solver_logs_mirror_tensorboard_metrics(tmp_path):
    solver, optimizer = _solver_and_optimizer()
    solver.solve(
        num_iter=1,
        optim=optimizer,
        keep_best=False,
        jit=True,
        log_every=0,
        tensorboard_log_dir=tmp_path,
        tensorboard_every=1,
    )
    accumulator = EventAccumulator(str(tmp_path))
    accumulator.Reload()
    tags = set(accumulator.Tags()["scalars"])

    assert {
        "optimizer/mirror/learning_rate",
        "optimizer/mirror/coordinate_gradient_norm",
        "optimizer/mirror/dual_displacement_norm",
        "optimizer/mirror/bregman_step",
        "optimizer/mirror/constraint_residual_max",
    } <= tags

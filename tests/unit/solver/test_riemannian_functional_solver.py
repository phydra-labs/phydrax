#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import jax
import jax.numpy as jnp
import pytest
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

import phydrax as phx


def _geometric_solver():
    domain = phx.domain.Interval1d(0.0, 1.0)
    target_direction = jnp.array([0.0, 1.0, 0.0])
    direction = domain.Parameter(
        jnp.array([1.0, 0.0, 0.0]),
        transform=lambda value: jnp.sum((value - target_direction) ** 2),
    )
    offset = domain.Parameter(2.0, transform=lambda value: (value - 0.5) ** 2)

    objective = phx.terms.IntegralFunctional(
        target=phx.integration.over(domain.component()),
        plan=phx.integration.FixedQuadraturePlan(phx.integration.GaussLegendreRule(6)),
        integrand=lambda functions: functions["direction"] + functions["offset"],
        materialization_policy="fixed",
    )
    return phx.solver.FunctionalSolver(
        functions={"direction": direction, "offset": offset},
        terms=(objective,),
    )


def _optimizer(solver, *, momentum=False, adaptive=False):
    parameters = solver.trainable_functions()
    direction_path = next(
        path
        for path in phx.optim.ParameterGeometry.array_leaf_paths(parameters)
        if "direction" in path
    )
    geometry = phx.optim.ParameterGeometry.from_leaf_paths(
        parameters,
        {direction_path: phx.metrix.SphereManifold(3)},
    )
    if adaptive:
        return phx.optim.riemannian_adam(
            geometry,
            learning_rate=lambda step: 0.12 / jnp.sqrt(step + 1.0),
            first_moment_decay=0.8,
            second_moment_decay=0.9,
            amsgrad=True,
        )
    if momentum:
        return phx.optim.riemannian_momentum(
            geometry,
            learning_rate=0.08,
            momentum=0.7,
        )
    return phx.optim.riemannian_sgd(geometry, learning_rate=0.12)


def _values(solver):
    leaves = tuple(
        jnp.asarray(leaf) for leaf in jax.tree.leaves(solver.trainable_functions())
    )
    direction = next(leaf for leaf in leaves if leaf.shape == (3,))
    offset = next(leaf for leaf in leaves if leaf.shape == ())
    return direction, offset


@pytest.mark.parametrize("jit", [False, True])
def test_functional_solver_trains_mixed_manifold_and_euclidean_parameters(jit):
    solver = _geometric_solver()
    initial_loss = solver.loss()
    trained = solver.solve(
        num_iter=35,
        optim=_optimizer(solver),
        keep_best=False,
        jit=jit,
        log_every=0,
    )
    direction, offset = _values(trained)

    assert trained.loss() < initial_loss
    assert jnp.allclose(jnp.linalg.norm(direction), 1.0, atol=1e-10)
    assert direction[1] > 0.999
    assert jnp.allclose(offset, 0.5, atol=2e-4)
    assert (
        trained.training_diagnostics["optimizer/riemannian/constraint_residual_max"]
        < 1e-10
    )
    assert (
        int(trained.training_diagnostics["optimizer/riemannian/num_manifold_leaves"]) == 1
    )


def test_functional_solver_accepts_transported_momentum():
    solver = _geometric_solver()
    trained = solver.solve(
        num_iter=45,
        optim=_optimizer(solver, momentum=True),
        keep_best=False,
        jit=True,
        log_every=0,
    )
    direction, offset = _values(trained)

    assert direction[1] > 0.995
    assert jnp.allclose(jnp.linalg.norm(direction), 1.0, atol=1e-10)
    assert jnp.abs(offset - 0.5) < 0.01
    assert trained.training_diagnostics["optimizer/riemannian/momentum_norm"] > 0.0


def test_functional_solver_accepts_intrinsic_adaptive_moments(tmp_path):
    solver = _geometric_solver()
    log_path = tmp_path / "training.log"
    initial_loss = solver.loss()
    trained = solver.solve(
        num_iter=100,
        optim=_optimizer(solver, adaptive=True),
        keep_best=False,
        jit=True,
        log_every=100,
        log_path=log_path,
    )
    direction, offset = _values(trained)
    diagnostics = trained.training_diagnostics

    assert trained.loss() < initial_loss
    assert direction[1] > 0.99
    assert jnp.allclose(jnp.linalg.norm(direction), 1.0, atol=1e-10)
    assert jnp.abs(offset - 0.5) < 0.15
    assert diagnostics["optimizer/riemannian/adaptive_denominator_minimum"] > 0.0
    assert (
        diagnostics["optimizer/riemannian/adaptive_denominator_maximum"]
        >= diagnostics["optimizer/riemannian/adaptive_denominator_minimum"]
    )
    assert "adaptive_denom=[" in log_path.read_text()


@pytest.mark.parametrize("optimizer_name", ("conjugate_gradient", "lbfgs"))
def test_functional_solver_supports_frozen_objective_line_search(optimizer_name):
    solver = _geometric_solver()
    geometry = _optimizer(solver).parameter_geometry
    if optimizer_name == "conjugate_gradient":
        optimizer = phx.optim.riemannian_conjugate_gradient(geometry)
    else:
        optimizer = phx.optim.riemannian_lbfgs(geometry, history_size=4)
    initial_loss = solver.loss()

    trained = solver.solve(
        num_iter=12,
        optim=optimizer,
        keep_best=False,
        jit=True,
        log_every=0,
    )
    direction, _ = _values(trained)
    diagnostics = trained.training_diagnostics

    assert trained.loss() < initial_loss
    assert jnp.allclose(jnp.linalg.norm(direction), 1.0, atol=1e-10)
    assert diagnostics["optimizer/riemannian/line_search_evaluations"] >= 1
    assert diagnostics["optimizer/riemannian/line_search_accepted"].dtype == jnp.bool_
    assert diagnostics["optimizer/riemannian/line_search_reduction"] >= 0.0
    assert diagnostics["optimizer/riemannian/restarted"].dtype == jnp.bool_
    assert diagnostics["optimizer/riemannian/pair_accepted"].dtype == jnp.bool_


def test_riemannian_solver_logging_and_tensorboard_diagnostics(tmp_path):
    solver = _geometric_solver()
    log_path = tmp_path / "training.log"
    tensorboard_dir = tmp_path / "tensorboard"
    solver.solve(
        num_iter=2,
        optim=_optimizer(solver),
        keep_best=False,
        jit=False,
        log_every=1,
        log_path=log_path,
        tensorboard_log_dir=tensorboard_dir,
        tensorboard_every=1,
    )

    text = log_path.read_text()
    assert "[phydrax][riemannian-sgd]" in text
    assert "rgrad=" in text
    assert "step_norm=" in text
    assert "constraint=" in text

    accumulator = EventAccumulator(str(tensorboard_dir))
    accumulator.Reload()
    scalar_tags = set(accumulator.Tags()["scalars"])
    assert "optimizer/riemannian/gradient_norm" in scalar_tags
    assert "optimizer/riemannian/tangent_step_norm" in scalar_tags
    assert "optimizer/riemannian/constraint_residual_max" in scalar_tags
    assert "optimizer/riemannian/tangent_residual" in scalar_tags
    assert "optimizer/riemannian/line_search_evaluations" in scalar_tags
    assert "optimizer/riemannian/line_search_reduction" in scalar_tags
    assert "optimizer/riemannian/restarted" in scalar_tags
    assert "optimizer/riemannian/pair_accepted" in scalar_tags
    assert "optimizer/riemannian/adaptive_denominator_minimum" in scalar_tags
    assert "optimizer/riemannian/adaptive_denominator_maximum" in scalar_tags


def test_riemannian_solver_rejects_ambient_evaluation_parameters():
    solver = _geometric_solver()
    with pytest.raises(ValueError, match="unsupported for Riemannian"):
        solver.solve(
            num_iter=1,
            optim=_optimizer(solver),
            evaluation_parameters=lambda _state, parameters: parameters,
            log_every=0,
        )


def test_riemannian_solver_rejects_geometry_bound_to_another_tree():
    solver = _geometric_solver()
    parameters = {"point": jnp.array([1.0, 0.0, 0.0])}
    geometry = phx.optim.ParameterGeometry.from_leaf_paths(
        parameters,
        {"['point']": phx.metrix.SphereManifold(3)},
    )
    optimizer = phx.optim.riemannian_sgd(geometry)

    with pytest.raises(ValueError, match="PyTree structure"):
        solver.solve(
            num_iter=1,
            optim=optimizer,
            jit=False,
            log_every=0,
        )


def test_zero_iteration_geometric_solve_returns_original_solver():
    solver = _geometric_solver()
    assert solver.solve(num_iter=0, optim=_optimizer(solver)) is solver

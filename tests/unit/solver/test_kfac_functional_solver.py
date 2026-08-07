#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import optax
import pytest

import phydrax as phx
import phydrax.solver._kfac_solver as kfac_solver
from phydrax.constraints import FunctionalConstraint
from phydrax.domain import PointSampling, SampleLayout
from phydrax.solver import FunctionalSolver
from phydrax.solver._kfac_solver import _quadratic_norm_and_clip


def _linear_solver(
    *,
    two_constraints=False,
    objectives=(),
    model_loss=False,
    collocation_policy=None,
):
    domain = phx.domain.Interval1d(0.0, 1.0)
    model = phx.nn.MLP(
        in_size=1,
        out_size="scalar",
        hidden_sizes=(),
        rwf=False,
        key=jr.key(0),
    )
    model = eqx.tree_at(
        lambda item: item.layers[0].weight,
        model,
        jnp.zeros_like(model.layers[0].weight),
    )
    model = eqx.tree_at(
        lambda item: item.layers[0].bias,
        model,
        jnp.zeros_like(model.layers[0].bias),
    )
    if model_loss:
        model = model.add_model_loss(
            lambda item, **kwargs: jnp.sum(jnp.square(item.layers[0].weight)),
            label="weight-penalty",
        )
    u = domain.Model("x")(model)
    sampling = PointSampling(8, layout=SampleLayout((("x",),)))
    first = FunctionalConstraint.from_operator(
        component=domain.component(),
        operator=lambda field: field - 1.0,
        constraint_vars="u",
        sampling=sampling,
        collocation_policy=collocation_policy,
        label="target-one",
    )
    constraints = [first]
    if two_constraints:
        constraints.append(
            FunctionalConstraint.from_operator(
                component=domain.component(),
                operator=lambda field: 0.5 * (field - 1.0),
                constraint_vars="u",
                sampling=sampling,
                label="target-one-scaled",
            )
        )
    return FunctionalSolver(
        functions={"u": u},
        constraints=constraints,
        objectives=objectives,
    )


def test_public_kfac_optimizer_decreases_frozen_functional_loss():
    solver = _linear_solver()
    initial = solver.loss(key=jr.key(20))
    trained = solver.solve(
        num_iter=2,
        optim=phx.optim.kfac(damping=1e-2),
        seed=21,
        jit=False,
        keep_best=False,
        log_every=0,
    )
    final = trained.loss(key=jr.key(20))

    assert final < initial
    assert trained.training_diagnostics["optimizer/kfac/factor_updates"] == 2
    assert trained.training_diagnostics["optimizer/kfac/step_size"] > 0.0
    assert trained.training_diagnostics["optimizer/kfac/num_affine_blocks"] == 1


def test_kfac_keep_best_includes_initial_parameters():
    solver = _linear_solver()
    trained = solver.solve(
        num_iter=1,
        optim=phx.optim.kfac(
            damping=1e-2,
            learning_rate=100.0,
            line_search=False,
        ),
        seed=21,
        jit=False,
        keep_best=True,
        log_every=0,
    )

    initial_model = solver.functions["u"].func.raw_model
    trained_model = trained.functions["u"].func.raw_model
    assert jnp.array_equal(trained_model.layers[0].weight, initial_model.layers[0].weight)
    assert jnp.array_equal(trained_model.layers[0].bias, initial_model.layers[0].bias)


def test_kfac_quadratic_norm_clip_enforces_requested_bound():
    clipped, norm = _quadratic_norm_and_clip(
        jnp.asarray([4.0]),
        jnp.asarray([4.0]),
        maximum=1.0,
    )

    assert jnp.allclose(clipped, jnp.asarray([1.0]))
    assert jnp.allclose(norm, 1.0)


def test_kfac_factor_update_period_and_constraint_subsampling_are_supported():
    trained = _linear_solver(two_constraints=True).solve(
        num_iter=3,
        optim=phx.optim.kfac(
            damping=1e-2,
            factor_update_period=2,
            line_search=False,
            learning_rate=0.05,
        ),
        seed=22,
        jit=False,
        keep_best=False,
        log_every=0,
        train_constraint_sample_size=1,
    )

    assert trained.training_diagnostics["optimizer/kfac/factor_updates"] == 2
    weight = trained.functions["u"].func.raw_model.layers[0].weight
    assert jnp.all(jnp.isfinite(weight))


def test_kfac_num_iter_zero_preserves_solver():
    solver = _linear_solver()
    assert solver.solve(num_iter=0, optim=phx.optim.kfac()) is solver


def test_kfac_rejects_negative_num_iter():
    with pytest.raises(ValueError, match="num_iter must be nonnegative"):
        _linear_solver().solve(num_iter=-1, optim=phx.optim.kfac())


def test_kfac_supports_implicit_constraint_vars_and_jax_iteration_scalar():
    domain = phx.domain.Interval1d(0.0, 1.0)
    model = phx.nn.MLP(
        in_size=1,
        out_size="scalar",
        hidden_sizes=(),
        rwf=False,
        key=jr.key(26),
    )

    @domain.Function(
        "x",
        binding=phx.domain.FunctionBinding(pass_iter=True),
    )
    def scheduled_weight(x, *, iter_):
        return jnp.ones_like(x[0]) + 0.0 * iter_.astype(float)

    constraint = FunctionalConstraint(
        component=domain.component(),
        residual=lambda functions: functions["u"] - 1.0,
        sampling=PointSampling(6, layout=SampleLayout((("x",),))),
        weight=scheduled_weight,
    )
    trained = FunctionalSolver(
        functions={"u": domain.Model("x")(model)},
        constraints=constraint,
    ).solve(
        num_iter=1,
        optim=phx.optim.kfac(damping=1e-2),
        keep_best=False,
        log_every=0,
    )
    assert jnp.isfinite(trained.loss(key=jr.key(27), iter_=jnp.asarray(1.0, dtype=float)))


def test_kfac_logs_train_and_eval_constraints_to_console_and_tensorboard(
    monkeypatch,
    tmp_path,
):
    scalar_tags = []

    class RecordingTensorBoard:
        def __init__(self, log_dir):
            del log_dir

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, traceback):
            del exc_type, exc, traceback

        def scalar(self, tag, value, step):
            del value, step
            scalar_tags.append(tag)

        def flush(self):
            return None

    monkeypatch.setattr(kfac_solver, "TensorBoardLogger", RecordingTensorBoard)
    base = _linear_solver()
    solver = FunctionalSolver(
        functions=base.functions,
        constraints=base.constraints,
        eval_constraints=base.constraints,
    )
    log_path = tmp_path / "kfac-diagnostics.log"
    solver.solve(
        num_iter=1,
        optim=phx.optim.kfac(damping=1e-2),
        keep_best=False,
        log_every=1,
        log_constraints=True,
        log_path=log_path,
        tensorboard_log_dir=tmp_path / "tensorboard",
        tensorboard_every=1,
    )

    log_text = log_path.read_text()
    assert "[train 0] target-one:" in log_text
    assert "[eval 0] target-one:" in log_text
    assert "train/constraints/000_target-one/loss" in scalar_tags
    assert "eval/constraints/000_target-one/loss" in scalar_tags


def test_kfac_honors_training_signal_stop(monkeypatch, tmp_path):
    class StopImmediately:
        signal_name = "SIGTERM"
        stop_requested = True

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, traceback):
            del exc_type, exc, traceback

    monkeypatch.setattr(kfac_solver, "_TrainingSignalGuard", StopImmediately)
    log_path = tmp_path / "kfac.log"
    trained = _linear_solver().solve(
        num_iter=5,
        optim=phx.optim.kfac(),
        log_every=0,
        log_path=log_path,
    )

    assert trained.training_diagnostics["optimizer/kfac/factor_updates"] == 0
    assert "received SIGTERM; exiting training loop after 0/5 iteration(s)" in (
        log_path.read_text()
    )


def test_kfac_rejects_standalone_objectives_without_curvature_roots():
    domain = phx.domain.Interval1d(0.0, 1.0)
    objective = phx.objectives.IntegralFunctional.from_operator(
        target=phx.integration.over(domain.component()),
        plan=phx.integration.FixedQuadraturePlan(phx.integration.GaussLegendreRule(4)),
        operator=lambda field: field,
        objective_vars="u",
    )

    with pytest.raises(ValueError, match="standalone objective"):
        _linear_solver(objectives=(objective,)).solve(
            num_iter=1,
            optim=phx.optim.kfac(),
            log_every=0,
        )


def test_kfac_rejects_attached_model_losses_without_curvature_roots():
    with pytest.raises(ValueError, match="attached model losses"):
        _linear_solver(model_loss=True).solve(
            num_iter=1,
            optim=phx.optim.kfac(),
            log_every=0,
        )


def test_kfac_replays_seed_across_eager_and_requested_jit_modes():
    solver = _linear_solver()
    eager = solver.solve(
        num_iter=2,
        optim=phx.optim.kfac(damping=1e-2, factor_chunk_size=2),
        seed=24,
        jit=False,
        keep_best=False,
        log_every=0,
    )
    requested_jit = solver.solve(
        num_iter=2,
        optim=phx.optim.kfac(damping=1e-2, factor_chunk_size=2),
        seed=24,
        jit=True,
        keep_best=False,
        log_every=0,
    )

    eager_params, _ = eager.partition_functions()
    jit_params, _ = requested_jit.partition_functions()
    for eager_leaf, jit_leaf in zip(
        jax.tree_util.tree_leaves(eager_params),
        jax.tree_util.tree_leaves(jit_params),
        strict=True,
    ):
        assert jnp.array_equal(eager_leaf, jit_leaf)
    assert not eager.training_diagnostics["optimizer/kfac/jit_requested"]
    assert requested_jit.training_diagnostics["optimizer/kfac/jit_requested"]


def test_kfac_refreshes_adaptive_collocation_before_frozen_step():
    solver = _linear_solver(
        collocation_policy=phx.constraints.PeriodicCollocation(
            refresh_every=1,
            sampler="uniform",
        )
    )
    initial = solver.collocation[0].batch.points["x"].data
    trained = solver.solve(
        num_iter=1,
        optim=phx.optim.kfac(damping=1e-2),
        seed=25,
        jit=False,
        keep_best=False,
        log_every=0,
    )
    refreshed = trained.collocation[0].batch.points["x"].data

    assert initial.shape == refreshed.shape
    assert not jnp.array_equal(initial, refreshed)


def test_existing_optax_dispatch_remains_unchanged():
    solver = _linear_solver()
    trained = solver.solve(
        num_iter=2,
        optim=optax.sgd(1e-2),
        seed=23,
        jit=False,
        keep_best=False,
        log_every=0,
        profile_adaptive=True,
    )

    assert "optimizer/kfac/factor_updates" not in trained.training_diagnostics
    assert trained.training_diagnostics["optimizer_first_step_wall_time_seconds"] > 0.0
    assert trained.training_diagnostics["optimizer_steady_step_wall_time_seconds"] > 0.0
    assert trained.loss(key=jr.key(20)) < solver.loss(key=jr.key(20))


def test_upstream_optax_lbfgs_decreases_deterministic_functional_loss():
    solver = _linear_solver()
    initial = solver.loss(key=jr.key(26))

    trained = solver.solve(
        num_iter=2,
        optim=optax.lbfgs(),
        seed=26,
        jit=True,
        keep_best=False,
        log_every=0,
    )

    assert trained.loss(key=jr.key(26)) < initial

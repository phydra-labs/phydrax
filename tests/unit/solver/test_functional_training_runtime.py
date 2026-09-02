import jax
import jax.numpy as jnp
import jax.random as jr
import optax
import pytest

import phydrax as phx
from phydrax._trainable import partition_trainable
from phydrax.solver._functional_checkpoint import load_functional_training_checkpoint
from phydrax.solver._functional_residual import prepare_functional_residual
from phydrax.solver._functional_surrogate import (
    prepare_functional_update,
    PreparedFunctionalUpdate,
)


def _fixed_interval_solver(*, blocks=None, evaluation=False):
    domain = phx.domain.Interval1d(0.0, 1.0)
    field = domain.Parameter(jnp.asarray([1.0, -2.0]))
    component = domain.component()
    condition = phx.conditions.Residual("u", component, lambda value: value)
    batch = component.sample(
        phx.domain.PointSampling(
            4,
            layout=phx.domain.SampleLayout((("x",),)),
        ),
        key=jr.key(0),
    )
    realization = phx.integration.from_samples(
        phx.integration.mean_over(component), batch
    )
    term = phx.terms.ResidualPenalty(
        condition,
        phx.integration.fixed(realization),
        blocks=blocks,
        label="state",
    )
    return phx.solver.FunctionalSolver(
        functions={"u": field},
        terms=(term,),
        evaluation_terms=(term,) if evaluation else (),
    )


def test_residual_block_layout_preserves_authored_loss_and_root_partition():
    layout = phx.terms.ResidualBlockLayout(("first", "second"))
    solver = _fixed_interval_solver(blocks=layout)
    params, fixed = partition_trainable(solver.functions)
    prepared = solver.objective.prepare_training(
        (0,),
        scale=1.0,
        evaluation_key=jr.key(1),
        sampling_key=jr.key(2),
        iteration=1,
    )
    residual = prepare_functional_residual(
        prepared, params, fixed, solver.enforcement
    )

    assert residual.layout.logical_blocks == ((0, "first"), (0, "second"))
    assert jnp.allclose(residual.loss(params), solver.loss(key=jr.key(3)))
    first = residual.layout.logical_indices(0, "first")
    second = residual.layout.logical_indices(0, "second")
    assert first.size == second.size == 4


def test_prepared_update_separates_equal_physical_and_untransformed_surrogate():
    solver = _fixed_interval_solver()
    params, fixed = partition_trainable(solver.functions)
    prepared = solver.objective.prepare_training(
        (0,),
        scale=1.0,
        evaluation_key=jr.key(4),
        sampling_key=jr.key(5),
        iteration=1,
    )
    update = prepare_functional_update(
        prepared, params, fixed, solver.enforcement
    )

    assert isinstance(update, PreparedFunctionalUpdate)
    physical = update.physical_values(solver.functions).total
    assert jnp.allclose(update.surrogate_loss(params, fixed), physical)


def test_functional_checkpoint_resume_matches_uninterrupted_steps(tmp_path):
    plan = phx.solver.FunctionalTrainingPlan(
        checkpoint=phx.solver.FunctionalCheckpointPolicy(
            tmp_path / "functional", every=1
        )
    )
    solver = _fixed_interval_solver()
    interrupted = solver.solve(
        num_iter=1,
        optim=optax.sgd(0.05),
        keep_best=False,
        log_every=0,
        training=plan,
    )
    assert interrupted.training_state is not None
    assert interrupted.training_state.progress.update_step == 1

    template = interrupted.training_state
    restored = load_functional_training_checkpoint(
        tmp_path / "functional", interrupted, template, plan
    )
    assert restored.state.progress.update_step == 1
    incompatible_plan = phx.solver.FunctionalTrainingPlan(
        checkpoint=phx.solver.FunctionalCheckpointPolicy(
            tmp_path / "functional",
            every=2,
        )
    )
    with pytest.raises(ValueError, match="training-plan identity"):
        load_functional_training_checkpoint(
            tmp_path / "functional",
            interrupted,
            template,
            incompatible_plan,
        )
    with pytest.raises(ValueError, match="In-memory functional training-plan"):
        interrupted.solve(
            num_iter=2,
            optim=optax.sgd(0.05),
            keep_best=False,
            log_every=0,
            training=incompatible_plan,
            resume=True,
        )

    disk_resumed = solver.solve(
        num_iter=2,
        optim=optax.sgd(0.05),
        keep_best=False,
        log_every=0,
        training=plan,
        resume=True,
    )

    resumed = interrupted.solve(
        num_iter=2,
        optim=optax.sgd(0.05),
        keep_best=False,
        log_every=0,
        training=plan,
        resume=True,
    )
    uninterrupted = solver.solve(
        num_iter=2,
        optim=optax.sgd(0.05),
        keep_best=False,
        log_every=0,
        training=phx.solver.FunctionalTrainingPlan(),
    )
    assert resumed.training_state.progress.update_step == 2
    assert jnp.allclose(
        resumed.training_state.current_functions["u"].func(),
        uninterrupted.training_state.current_functions["u"].func(),
    )
    assert jnp.allclose(
        disk_resumed.training_state.current_functions["u"].func(),
        uninterrupted.training_state.current_functions["u"].func(),
    )


def test_checkpoint_resume_replays_resampled_collocation(tmp_path):
    domain = phx.domain.Interval1d(0.0, 1.0)
    model = phx.nn.models.MLP(
        in_size=1,
        out_size="scalar",
        hidden_sizes=(),
        rwf=False,
        key=jr.key(20),
    )
    field = domain.Model("x")(model)
    component = domain.component()
    condition = phx.conditions.Residual("u", component, lambda value: value)
    term = phx.terms.ResidualPenalty(
        condition,
        phx.integration.per_step(
            phx.integration.mean_over(component),
            phx.integration.MonteCarloPlan(4),
        ),
    )
    solver = phx.solver.FunctionalSolver(functions={"u": field}, terms=(term,))
    plan = phx.solver.FunctionalTrainingPlan(
        checkpoint=phx.solver.FunctionalCheckpointPolicy(
            tmp_path / "resampled",
            every=1,
        )
    )
    solver.solve(
        num_iter=1,
        optim=optax.sgd(0.01),
        keep_best=False,
        log_every=0,
        seed=21,
        training=plan,
    )
    resumed = solver.solve(
        num_iter=2,
        optim=optax.sgd(0.01),
        keep_best=False,
        log_every=0,
        seed=21,
        training=plan,
        resume=True,
    )
    uninterrupted = solver.solve(
        num_iter=2,
        optim=optax.sgd(0.01),
        keep_best=False,
        log_every=0,
        seed=21,
        training=phx.solver.FunctionalTrainingPlan(),
    )

    resumed_leaves = jax.tree.leaves(resumed.trainable_functions())
    uninterrupted_leaves = jax.tree.leaves(uninterrupted.trainable_functions())
    assert all(
        jnp.allclose(left, right)
        for left, right in zip(
            resumed_leaves,
            uninterrupted_leaves,
            strict=True,
        )
    )


def test_kfac_checkpoint_resume_matches_uninterrupted_steps(tmp_path):
    domain = phx.domain.Interval1d(0.0, 1.0)
    model = phx.nn.models.MLP(
        in_size=1,
        out_size="scalar",
        hidden_sizes=(),
        rwf=False,
        key=jr.key(29),
    )
    field = domain.Model("x")(model)
    component = domain.component()
    condition = phx.conditions.Residual("u", component, lambda value: value)
    batch = component.points({"x": jnp.asarray([[0.1], [0.4], [0.7], [0.9]])})
    term = phx.terms.ResidualPenalty(
        condition,
        phx.integration.fixed(
            phx.integration.from_samples(
                phx.integration.mean_over(component),
                batch,
            )
        ),
    )
    solver = phx.solver.FunctionalSolver(functions={"u": field}, terms=(term,))
    plan = phx.solver.FunctionalTrainingPlan(
        checkpoint=phx.solver.FunctionalCheckpointPolicy(
            tmp_path / "kfac",
            every=1,
        )
    )
    solver.solve(
        num_iter=1,
        optim=phx.optim.kfac(damping=1e-2),
        keep_best=False,
        log_every=0,
        jit=False,
        seed=30,
        training=plan,
    )
    resumed = solver.solve(
        num_iter=2,
        optim=phx.optim.kfac(damping=1e-2),
        keep_best=False,
        log_every=0,
        jit=False,
        seed=30,
        training=plan,
        resume=True,
    )
    uninterrupted = solver.solve(
        num_iter=2,
        optim=phx.optim.kfac(damping=1e-2),
        keep_best=False,
        log_every=0,
        jit=False,
        seed=30,
        training=phx.solver.FunctionalTrainingPlan(),
    )

    resumed_leaves = jax.tree.leaves(resumed.trainable_functions())
    uninterrupted_leaves = jax.tree.leaves(uninterrupted.trainable_functions())
    assert all(
        jnp.allclose(left, right)
        for left, right in zip(
            resumed_leaves,
            uninterrupted_leaves,
            strict=True,
        )
    )


def test_fixed_evaluation_selection_is_recorded():
    solver = _fixed_interval_solver(evaluation=True)
    plan = phx.solver.FunctionalTrainingPlan(
        selection=phx.solver.FunctionalSelectionPolicy(every=1)
    )
    trained = solver.solve(
        num_iter=2,
        optim=optax.sgd(0.05),
        log_every=0,
        training=plan,
    )
    assert trained.training_state is not None
    assert trained.training_state.progress.best_value is not None
    assert trained.training_state.progress.best_step in (1, 2)


def test_exact_nonlinear_correction_freezes_base_and_restores_physical_scale():
    solver = _fixed_interval_solver()
    correction = solver.functions["u"].domain.Parameter(jnp.asarray([0.5, 1.0]))
    problem = phx.solver.prepare_functional_correction(
        solver,
        {"u": correction},
        epsilon=0.1,
    )

    correction_params, _ = partition_trainable(problem.training_solver.functions)
    expected = solver.functions["u"].func() + 0.1 * correction.func()
    scaled_loss = problem.training_solver.loss(key=jr.key(9))
    finalized = problem.finalize(problem.training_solver)

    assert len(jax.tree.leaves(correction_params)) == 1
    assert jnp.allclose(problem.training_solver.functions["u"].func(), expected)
    assert jnp.allclose(
        scaled_loss,
        finalized.loss(key=jr.key(9)) / 0.1**2,
    )


def test_training_policy_publishes_finite_ntk_diagnostics():
    trained = _fixed_interval_solver().solve(
        num_iter=1,
        optim=optax.sgd(0.01),
        keep_best=False,
        log_every=0,
        training=phx.solver.FunctionalTrainingPlan(
            diagnostics=phx.solver.FunctionalDiagnosticsPolicy(
                every=1,
                gradient_alignment=False,
                ntk=True,
                ntk_probes=2,
                ntk_eigenvalues=1,
            )
        ),
    )

    assert "ntk/trace" in trained.training_diagnostics
    assert bool(trained.training_diagnostics["ntk/finite"])
    assert trained.training_diagnostics["ntk/trace"] > 0.0


def test_frozen_correction_field_preserves_explicit_derivative_rules():
    domain = phx.domain.Interval1d(0.0, 1.0)
    field = domain.Parameter(2.0).with_derivative_rule(
        phx.domain.CallbackDerivativeRule(
            lambda **kwargs: domain.Parameter(7.0)
        )
    )
    frozen = phx.solver.freeze_domain_function(field)
    derivative = phx.operators.partial_n(frozen, var="x", order=1)

    assert jnp.allclose(derivative.func(), 7.0)

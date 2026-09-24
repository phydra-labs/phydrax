#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from typing import ClassVar

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import optax
import pytest

import phydrax as phx
from phydrax._training_kernel import TrainingAttemptOutcome, TrainingRejectionBudgetError


class _Closure(phx.AbstractComponentSlot):
    """Discretization slot: a learned damping rate of an explicit decay rollout."""

    component_authority: ClassVar = phx.ComponentAuthority.DISCRETIZATION
    slot_semantic_id: ClassVar[str] = "test.decay-closure"
    rate: jax.Array = phx.parameter_field()

    def __init__(self, rate):
        self.rate = jnp.asarray(rate)


class _Relaxation(phx.AbstractComponentSlot):
    """Accelerator slot: the relaxation factor of a fixed-work Richardson solve."""

    component_authority: ClassVar = phx.ComponentAuthority.ACCELERATOR
    slot_semantic_id: ClassVar[str] = "test.relaxation"
    log_omega: jax.Array = phx.parameter_field()

    def __init__(self, omega):
        self.log_omega = jnp.log(jnp.asarray(omega))


class _Rollout(eqx.Module):
    step: jax.Array
    closure: _Closure | None


class _Richardson(eqx.Module):
    matrix: jax.Array
    relaxation: _Relaxation | None


def _decay_rollout(owner, case):
    initial, observed = case
    state = jax.lax.fori_loop(
        0, 8, lambda _, u: u - owner.step * owner.closure.rate * u, initial
    )
    return phx.solver.SolverCaseResult(
        residual=state - observed, accepted=jnp.all(jnp.isfinite(state))
    )


def _richardson_work(owner, rhs):
    omega = jnp.exp(owner.relaxation.log_omega)
    initial = jnp.zeros_like(rhs)
    state = jax.lax.fori_loop(
        0, 4, lambda _, x: x + omega * (rhs - owner.matrix @ x), initial
    )
    return phx.solver.AlgorithmicWorkResult(
        initial_residual=rhs - owner.matrix @ initial,
        final_residual=rhs - owner.matrix @ state,
        iterations=jnp.asarray(4),
        accepted=jnp.all(jnp.isfinite(state)),
    )


def _rollout_objective(**options):
    initial = jnp.asarray([[1.0, 2.0], [0.5, -1.0]])
    observed = initial * (1.0 - 0.1 * 0.7) ** 8
    return phx.solver.RolloutObjective(
        _Rollout(jnp.asarray(0.1), None),
        lambda solve, closure: _Rollout(solve.step, closure),
        _decay_rollout,
        objective_id="decay-rollout",
        cases=(initial, observed),
        **options,
    )


def _work_objective(**options):
    return phx.solver.AlgorithmicWorkObjective(
        _Richardson(jnp.diag(jnp.asarray([1.0, 2.0, 3.0, 4.0])), None),
        lambda solve, relaxation: _Richardson(solve.matrix, relaxation),
        _richardson_work,
        work=4,
        objective_id="richardson-work",
        cases=jr.normal(jr.key(0), (5, 4)),
        **options,
    )


def test_mixed_closure_and_preconditioner_train_through_separate_objectives():
    tree = {"closure": _Closure(0.2), "relaxation": _Relaxation(0.1)}
    rollout = _rollout_objective(component=lambda value: value["closure"])
    work = _work_objective(component=lambda value: value["relaxation"])

    result = phx.solver.train_components(
        tree, (rollout, work), optimizer=optax.adam(5e-2), steps=60, key=jr.key(1)
    )

    assert result.selection == (("['closure'].rate",), ("['relaxation'].log_omega",))
    assert result.authorities == (
        ("['closure'].rate", "discretization"),
        ("['relaxation'].log_omega", "accelerator"),
    )
    assert result.accepted_updates == 60
    assert bool(jnp.all(result.outcomes == TrainingAttemptOutcome.ACCEPTED))
    first, last = result.objective_values[0], result.objective_values[-1]
    assert float(last[0]) < 1e-3 * float(first[0])
    assert float(last[1]) < float(first[1]) - 1.0
    assert float(result.tree["closure"].rate) == pytest.approx(0.7, rel=1e-2)

    with pytest.raises(ValueError, match="no admissible training signal for accelerator"):
        phx.solver.train_components(
            tree, (rollout,), optimizer=optax.adam(5e-2), steps=1, key=jr.key(1)
        )


def test_failed_case_under_reject_attempt_rolls_back_and_exhausts_the_budget():
    def diverging(owner, case):
        result = _decay_rollout(owner, case)
        return phx.solver.SolverCaseResult(
            residual=result.residual, accepted=owner.closure.rate < 0.5
        )

    objective = phx.solver.RolloutObjective(
        _Rollout(jnp.asarray(0.1), None),
        lambda solve, closure: _Rollout(solve.step, closure),
        diverging,
        objective_id="decay-rollout",
        cases=(jnp.ones((1, 2)), jnp.zeros((1, 2))),
    )
    tree = _Closure(0.6)

    with pytest.raises(TrainingRejectionBudgetError) as raised:
        phx.solver.train_components(
            tree, (objective,), optimizer=optax.sgd(1e-2), steps=3, key=jr.key(2)
        )
    assert raised.value.outcome is TrainingAttemptOutcome.NONFINITE
    assert raised.value.state.parameters.rate == tree.rate

    skipped = phx.solver.train_components(
        tree,
        (
            phx.solver.RolloutObjective(
                objective.solve.value,
                objective.bind,
                diverging,
                objective_id="decay-rollout",
                cases=(jnp.ones((1, 2)), jnp.zeros((1, 2))),
                accepted_results="reduce-support",
            ),
        ),
        optimizer=optax.sgd(1e-2),
        steps=3,
        key=jr.key(2),
    )
    assert skipped.accepted_updates == 0
    assert skipped.failed_cases.tolist() == [[1], [1], [1]]
    assert skipped.tree.rate == tree.rate


def test_checkpoint_resume_equals_the_uninterrupted_run(tmp_path):
    tree = _Relaxation(0.1)
    uninterrupted = phx.solver.train_components(
        tree, (_work_objective(),), optimizer=optax.adam(5e-2), steps=6, key=jr.key(3)
    )
    first = phx.solver.train_components(
        tree,
        (_work_objective(),),
        optimizer=optax.adam(5e-2),
        steps=3,
        key=jr.key(3),
        checkpoint=tmp_path / "run",
    )
    resumed = phx.solver.train_components(
        tree,
        (_work_objective(),),
        optimizer=optax.adam(5e-2),
        steps=6,
        key=jr.key(3),
        checkpoint=tmp_path / "run",
    )

    assert first.attempts == 3
    assert resumed.resumed_from_attempt == 3
    assert resumed.values.shape == (3,)
    assert resumed.parameter_revision == uninterrupted.parameter_revision
    assert jnp.array_equal(resumed.values, uninterrupted.values[3:])

    with pytest.raises(ValueError, match="does not match"):
        phx.solver.train_components(
            tree,
            (_work_objective(weight=2.0),),
            optimizer=optax.adam(5e-2),
            steps=6,
            key=jr.key(3),
            checkpoint=tmp_path / "run",
        )


class _OpaqueRate(phx.AbstractArrayModel):
    """Rate from a provider that declares no derivative route."""

    rate: jax.Array
    in_size: int = eqx.field(static=True)
    out_size: int = eqx.field(static=True)

    def __init__(self, rate):
        self.rate = jnp.asarray(rate)
        self.in_size = "scalar"
        self.out_size = "scalar"

    def __call__(self, x, /, *, key=None):
        return self.rate * x

    def model_execution_contract(self):
        return phx.ModelExecutionContract(
            derivative=phx.DerivativeContract(route=phx.DerivativeRoute.STOPPED),
            execution=phx.ExecutionCapabilities("native-jax"),
            randomness=phx.RandomnessContract("deterministic"),
        )


def test_non_differentiable_component_trains_only_through_an_explicit_derivative_free_optimizer():
    def rollout(owner, case):
        initial, observed = case
        state = jax.lax.fori_loop(
            0, 8, lambda _, u: u - owner.step * owner.closure.model(u), initial
        )
        return phx.solver.SolverCaseResult(residual=state - observed, accepted=True)

    initial = jnp.asarray([[1.0, 2.0]])
    objective = phx.solver.RolloutObjective(
        _Rollout(jnp.asarray(0.1), None),
        lambda solve, binding: _Rollout(solve.step, binding),
        rollout,
        objective_id="opaque-rollout",
        cases=(initial, initial * (1.0 - 0.1 * 0.7) ** 8),
    )
    tree = phx.bind_component(_OpaqueRate(0.2), phx.ComponentAuthority.MODEL)

    with pytest.raises(
        ValueError, match="declares no JAX derivative.*distribution-evolution"
    ):
        phx.solver.train_components(
            tree, (objective,), optimizer=optax.adam(1e-2), steps=1, key=jr.key(4)
        )
    result = phx.solver.train_components(
        tree,
        (objective,),
        optimizer=phx.optim.OpenEvolutionStrategy(
            16, initial_standard_deviation=0.1, learning_rate=0.05
        ),
        steps=30,
        key=jr.key(4),
    )
    assert result.accepted_updates == 30
    assert float(result.values[-1]) < 0.1 * float(result.values[0])
    assert bool(jnp.all(result.gradient_norms == 0.0))


def test_kfac_is_refused_with_its_reason():
    with pytest.raises(ValueError, match="KFAC forms its curvature"):
        phx.solver.train_components(
            _Relaxation(0.1),
            (_work_objective(),),
            optimizer=phx.optim.kfac(),
            steps=1,
            key=jr.key(0),
        )

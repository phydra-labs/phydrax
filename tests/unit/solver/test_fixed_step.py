#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import equinox as eqx
import jax
import jax.numpy as jnp
import pytest

import phydrax as phx


class OffsetTransform(phx.solver.AbstractAcceptedStepTransform):
    offset: float = eqx.field(static=True)
    transform_id: str = eqx.field(static=True)

    def __init__(self, offset):
        self.offset = float(offset)
        self.transform_id = f"transform:offset:{offset}"

    def apply(self, step_index, time, previous_state, candidate_state, args, /):
        del step_index, time, previous_state, args
        transformed = candidate_state + self.offset
        return phx.solver.AcceptedStepTransformResult(
            transformed,
            jnp.asarray(True),
            jnp.asarray(True),
            jnp.sqrt(jnp.sum((transformed - candidate_state) ** 2)),
        )


def test_fixed_step_ssprk_solves_and_saves_requested_stride():
    method = phx.solver.SSPRK33FixedStepMethod(lambda time, state, args: -state)
    problem = phx.solver.FixedStepProblem(
        method,
        jnp.asarray([1.0]),
        t0=0.0,
        t1=0.1,
        step_size=0.01,
    )
    solution = phx.solver.solve_fixed_step(problem, save_every=2)

    assert solution.successful
    assert solution.times.shape == (6,)
    assert solution.states.shape == (6, 1)
    assert jnp.all(solution.valid)
    assert jnp.allclose(solution.states[-1, 0], jnp.exp(-0.1), rtol=2e-6)


def test_fixed_step_composes_accepted_step_transforms():
    transform = phx.solver.CompositeAcceptedStepTransform(
        (OffsetTransform(0.1), OffsetTransform(-0.05))
    )
    method = phx.solver.SSPRK54FixedStepMethod(
        lambda time, state, args: jnp.zeros_like(state), transform=transform
    )
    solution = phx.solver.solve_fixed_step(
        phx.solver.FixedStepProblem(
            method,
            jnp.asarray([0.0]),
            t0=0.0,
            t1=0.03,
            step_size=0.01,
        )
    )

    assert solution.successful
    assert jnp.all(solution.transform_applied)
    assert jnp.allclose(solution.states[-1], jnp.asarray([0.15]))


def test_fixed_step_saves_and_atomically_freezes_structured_state():
    def step(step_index, time, state, step_size, args):
        del time, step_size, args
        candidate = {
            "position": state["position"] + 1.0,
            "route": state["route"] + 1,
            "active": ~state["active"],
        }
        successful = step_index < 1
        accepted = jax.tree.map(
            lambda proposed, current: jnp.where(successful, proposed, current),
            candidate,
            state,
        )
        return phx.solver.FixedStepResult(
            candidate,
            accepted,
            successful,
            jnp.zeros(()),
            jnp.asarray(1, dtype=jnp.int32),
            jnp.asarray(1, dtype=jnp.int32),
            jnp.asarray(False),
            jnp.zeros(()),
        )

    initial = {
        "position": jnp.asarray([0.0]),
        "route": jnp.asarray([4], dtype=jnp.int32),
        "active": jnp.asarray([True]),
    }
    problem = phx.solver.FixedStepProblem(
        phx.solver.CallableFixedStepMethod(step, "structured-test-step"),
        initial,
        t0=0.0,
        t1=0.03,
        step_size=0.01,
        state_geometry=phx.discretization.DEMStateGeometry("structured-test"),
    )

    solution = phx.solver.solve_fixed_step(problem)

    assert not solution.successful
    assert jnp.array_equal(solution.valid, jnp.asarray([True, True, False, False]))
    assert jnp.array_equal(
        solution.states["position"][:, 0], jnp.asarray([0.0, 1.0, 1.0, 1.0])
    )
    assert jnp.array_equal(solution.states["route"][:, 0], jnp.asarray([4, 5, 5, 5]))
    assert jnp.array_equal(
        solution.states["active"][:, 0],
        jnp.asarray([True, False, False, False]),
    )
    first = jax.tree.map(lambda leaf: leaf[1], solution.states)
    midpoint = problem.state_geometry.interpolate(initial, first, 0.5)
    assert jnp.array_equal(midpoint["position"], jnp.asarray([0.5]))
    assert jnp.array_equal(midpoint["route"], initial["route"])
    assert jnp.array_equal(midpoint["active"], initial["active"])


def _additive_problem(step_count=5, *, control=1.0):
    def step(step_index, time, state, step_size, args):
        del step_index, time, step_size
        candidate = state + args
        successful = jnp.asarray(True)
        return phx.solver.FixedStepResult(
            candidate,
            candidate,
            successful,
            jnp.abs(args),
            jnp.asarray(1, dtype=jnp.int32),
            jnp.asarray(1, dtype=jnp.int32),
            jnp.asarray(False),
            jnp.zeros((), dtype=state.dtype),
        )

    return phx.solver.FixedStepProblem(
        phx.solver.CallableFixedStepMethod(step, "additive-test-step"),
        jnp.asarray([0.0]),
        t0=0.0,
        t1=0.1 * step_count,
        step_size=0.1,
        args=jnp.asarray(control),
    )


def test_fixed_step_rollout_retention_preserves_exact_endpoints():
    problem = _additive_problem()
    legacy = phx.solver.solve_fixed_step(problem)
    final = phx.solver.FixedStepRolloutPlan(retention="final").rollout(problem)
    checkpoints = phx.solver.FixedStepRolloutPlan(
        retention="checkpoints", checkpoint_stride=2
    ).rollout(problem)
    trajectory = phx.solver.FixedStepRolloutPlan(retention="trajectory").rollout(problem)

    assert jnp.array_equal(final.times, jnp.asarray([0.5]))
    assert jnp.array_equal(final.states, legacy.states[-1:])
    assert jnp.array_equal(final.final_state, legacy.states[-1])
    assert jnp.array_equal(checkpoints.times, jnp.asarray([0.0, 0.2, 0.4, 0.5]))
    assert jnp.array_equal(checkpoints.states[:, 0], jnp.asarray([0.0, 2.0, 4.0, 5.0]))
    assert jnp.array_equal(trajectory.states, legacy.states)
    assert jnp.array_equal(trajectory.valid, legacy.valid)
    assert jnp.array_equal(trajectory.residuals, legacy.residuals)


def test_fixed_step_rollout_observes_fail_closed_endpoint_state():
    def step(step_index, time, state, step_size, args):
        del time, step_size, args
        candidate = state + 1.0
        successful = step_index < 1
        accepted = jnp.where(successful, candidate, state)
        return phx.solver.FixedStepResult(
            candidate,
            accepted,
            successful,
            jnp.zeros(()),
            jnp.asarray(1, dtype=jnp.int32),
            jnp.asarray(1, dtype=jnp.int32),
            jnp.asarray(False),
            jnp.zeros(()),
        )

    problem = phx.solver.FixedStepProblem(
        phx.solver.CallableFixedStepMethod(step, "failing-rollout-step"),
        jnp.asarray([0.0]),
        t0=0.0,
        t1=0.3,
        step_size=0.1,
    )
    plan = phx.solver.FixedStepRolloutPlan(
        retention="checkpoints",
        checkpoint_stride=2,
        diagnostics=lambda step_index, time, state, args: {
            "state": jnp.sum(state),
            "time": time,
        },
        diagnostics_id="accepted-endpoint",
    )
    result = plan.rollout(problem)

    assert not result.successful
    assert jnp.array_equal(result.states[:, 0], jnp.asarray([0.0, 1.0, 1.0]))
    assert jnp.array_equal(result.valid, jnp.asarray([True, False, False]))
    assert jnp.array_equal(result.diagnostics["state"], jnp.asarray([1.0, 1.0, 1.0]))
    assert jnp.allclose(result.diagnostics["time"], jnp.asarray([0.1, 0.2, 0.3]))


@pytest.mark.parametrize(
    "replay",
    (
        phx.solver.FixedStepReplayPolicy("full"),
        phx.solver.FixedStepReplayPolicy("step"),
        phx.solver.FixedStepReplayPolicy("block", block_size=2),
    ),
)
@pytest.mark.parametrize("retention", ("final", "checkpoints", "trajectory"))
def test_fixed_step_replay_preserves_primal_gradient_and_retention(replay, retention):
    keywords = (
        {"retention": retention, "checkpoint_stride": 2}
        if retention == "checkpoints"
        else {"retention": retention}
    )
    direct = phx.solver.FixedStepRolloutPlan(
        **keywords,
        replay=phx.solver.FixedStepReplayPolicy("full"),
    )
    candidate = phx.solver.FixedStepRolloutPlan(**keywords, replay=replay)

    def objective(control, plan):
        result = plan.rollout(_additive_problem(5, control=control))
        return jnp.sum(result.final_state**2), result

    control = jnp.asarray(0.25)
    direct_value, direct_gradient = jax.value_and_grad(
        lambda value: objective(value, direct)[0]
    )(control)
    candidate_value, candidate_gradient = jax.value_and_grad(
        lambda value: objective(value, candidate)[0]
    )(control)
    direct_result = objective(control, direct)[1]
    candidate_result = objective(control, candidate)[1]

    assert jnp.array_equal(candidate_result.times, direct_result.times)
    assert jnp.array_equal(candidate_result.states, direct_result.states)
    assert jnp.array_equal(candidate_result.valid, direct_result.valid)
    assert jnp.array_equal(candidate_result.residuals, direct_result.residuals)
    assert jnp.allclose(candidate_value, direct_value)
    assert jnp.allclose(candidate_gradient, direct_gradient)


@pytest.mark.parametrize(
    "replay",
    (
        phx.solver.FixedStepReplayPolicy("full"),
        phx.solver.FixedStepReplayPolicy("step"),
        phx.solver.FixedStepReplayPolicy("block", block_size=3),
    ),
)
def test_legacy_fixed_step_replay_preserves_save_stride(replay):
    direct = phx.solver.solve_fixed_step(_additive_problem(5), save_every=2)
    candidate = phx.solver.solve_fixed_step(
        _additive_problem(5),
        save_every=2,
        replay=replay,
    )

    assert jnp.array_equal(candidate.times, direct.times)
    assert jnp.array_equal(candidate.states, direct.states)
    assert jnp.array_equal(candidate.valid, direct.valid)

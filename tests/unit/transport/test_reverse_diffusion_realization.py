#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Key-addressed reverse-diffusion realizations, score context and compiled solves."""

from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import pytest

import phydrax as phx
from phydrax.transport.diffusion import (
    probability_flow_system,
    ReverseDiffusion,
    ScoreContext,
)


INITIAL_SCALE = 0.01
TERMINAL_SCALE = 1.0
LOG_RATIO = float(np.log(TERMINAL_SCALE / INITIAL_SCALE))


class _ShiftedGaussianScore(eqx.Module):
    """Exact VE marginal score of ``N(shift, s0^2)`` data (variance ``s0^2 e^{2 lambda t}``)."""

    def __call__(self, state: Any, time: Any, shift: Any) -> Any:
        variance = (INITIAL_SCALE * jnp.exp(LOG_RATIO * time)) ** 2
        return -(state - shift) / variance


class _FixedGaussianScore(eqx.Module):
    shift: jax.Array

    def __call__(self, state: Any, time: Any) -> Any:
        variance = (INITIAL_SCALE * jnp.exp(LOG_RATIO * time)) ** 2
        return -(state - self.shift) / variance


def _process() -> Any:
    return phx.stochastic.VarianceExplodingDiffusion(
        2, initial_scale=INITIAL_SCALE, terminal_scale=TERMINAL_SCALE
    )


def _domain(*, context: bool) -> Any:
    state = phx.domain.HyperRectangle(
        jnp.full((2,), -100.0), jnp.full((2,), 100.0), label="x"
    )
    domain = state @ phx.domain.TimeInterval(0.0, 1.0)
    if context:
        domain = domain @ phx.domain.HyperRectangle(
            jnp.full((2,), -1.0), jnp.full((2,), 1.0), label="shift"
        )
    return domain


def _conditional_score() -> Any:
    return _domain(context=True).Function("x", "t", "shift")(_ShiftedGaussianScore())


def _reverse() -> ReverseDiffusion:
    process = _process()
    return ReverseDiffusion(
        process,
        _conditional_score(),
        process.asymptotic_terminal_reference(),
        score_id="analytic-shifted-gaussian",
        dt0=0.01,
        wiener_tolerance=1e-4,
        context_labels=("shift",),
    )


def _context(shift: Any) -> ScoreContext:
    return ScoreContext({"shift": jnp.asarray(shift, dtype=jnp.float64)})


def test_traced_realization_is_key_addressed_and_replays_the_eager_one() -> None:
    reverse = _reverse()
    traces: list[None] = []

    @eqx.filter_jit
    def realize(transport: ReverseDiffusion, key: Any) -> Any:
        traces.append(None)
        return transport.realize(key, (4,))

    eager = reverse.realize(jr.key(3), (4,))
    staged = realize(reverse, jr.key(3))
    other = realize(reverse, jr.key(4))

    # A new key is dynamic data: no retrace, no host fingerprint in the trace.
    assert len(traces) == 1
    np.testing.assert_array_equal(staged.terminal_states, eager.terminal_states)
    np.testing.assert_array_equal(
        jr.key_data(staged.score_key), jr.key_data(eager.score_key)
    )
    np.testing.assert_array_equal(
        jr.key_data(staged.root_key), jr.key_data(eager.root_key)
    )
    assert staged.law_id == eager.law_id
    assert staged.realization_id == eager.realization_id
    assert other.realization_id != eager.realization_id
    assert not np.array_equal(other.terminal_states, eager.terminal_states)
    # The Wiener paths keep their static law and host-materialized digest.
    assert staged.wiener.law_id == eager.wiener.law_id
    assert staged.wiener.realization_id == eager.wiener.realization_id
    assert len(eager.wiener.realization_id) == 64
    assert other.wiener.realization_id != eager.wiener.realization_id
    assert eager.wiener.sample_shape == (4,)


def test_prepared_layout_is_validated_once_and_independent_of_the_key() -> None:
    reverse = _reverse()
    prepared = reverse.prepare((2,), save_times=np.asarray([0.5, 1.0]))
    first = prepared.realize(jr.key(0))
    second = prepared.realize(jr.key(1))
    context = _context([0.1, 0.2])
    left = prepared.solve(first, context=context)
    right = prepared.solve(second, context=context)

    assert prepared.save_times == (0.5, 1.0)
    assert first.law_id == second.law_id == prepared.law_id
    # The differential problem is the prepared layout; the realization binding
    # (root key, terminal states, Wiener paths) stays distinct from it.
    assert left.solution.problem_id == right.solution.problem_id == prepared.problem_id
    assert left.prepared_id == right.prepared_id == prepared.prepared_id
    assert left.realization_id != right.realization_id
    assert left.solution.states.shape == (2, 2, 2)
    with pytest.raises(ValueError, match="save_times"):
        reverse.prepare((2,), save_times=np.asarray([1.0, 0.5]))

    @eqx.filter_jit
    def traced_layout(transport: ReverseDiffusion, times: Any) -> Any:
        return transport.prepare((1,), save_times=times).prepared_id

    with pytest.raises(TypeError, match="save_times"):
        traced_layout(reverse, jnp.asarray([1.0]))


def test_traced_realization_has_no_host_identity() -> None:
    reverse = _reverse()

    @eqx.filter_jit
    def identity(transport: ReverseDiffusion, key: Any) -> Any:
        return transport.realize(key, (1,)).realization_id

    with pytest.raises(ValueError, match="host identity"):
        identity(reverse, jr.key(0))


def test_compiled_solve_matches_eager_solve_and_conditions_on_context() -> None:
    reverse = _reverse()
    realization = reverse.realize(jr.key(5), (64,))
    shift = jnp.asarray([0.4, -0.3])
    eager = reverse.solve(realization, context=_context(shift))

    @eqx.filter_jit
    def solve(transport: ReverseDiffusion, value: Any, context: ScoreContext) -> Any:
        return transport.solve(value, context=context).final_states

    staged = solve(reverse, realization, _context(shift))
    moved = solve(reverse, realization, _context(-shift))

    assert eager.successful
    assert eager.realization_id == realization.realization_id
    assert eager.realization_law_id == realization.law_id
    np.testing.assert_allclose(staged, eager.final_states, rtol=0.0, atol=1e-12)
    samples = np.asarray(eager.final_states)
    np.testing.assert_allclose(samples.mean(axis=0), shift, atol=0.02)
    assert float(np.max(samples.std(axis=0))) < 0.05
    np.testing.assert_allclose(np.asarray(moved).mean(axis=0), -shift, atol=0.02)
    replay = reverse.solve(realization, context=_context(shift))
    np.testing.assert_array_equal(replay.final_states, eager.final_states)


def test_score_context_must_match_the_declared_labels() -> None:
    reverse = _reverse()
    realization = reverse.realize(jr.key(1), (2,))

    with pytest.raises(ValueError, match="context"):
        reverse.solve(realization)
    with pytest.raises(ValueError, match="context"):
        reverse.solve(
            realization,
            context=ScoreContext({"shift": jnp.zeros((2,)), "other": jnp.zeros(())}),
        )
    with pytest.raises(ValueError, match="save_times"):
        reverse.solve(
            realization, context=_context([0.0, 0.0]), save_times=jnp.asarray([0.0, 1.0])
        )


def test_lane_mapped_solves_are_independent_of_batching() -> None:
    reverse = _reverse()
    keys = jr.split(jr.key(9), 3)
    shifts = jnp.asarray([[0.5, 0.5], [-0.5, 0.25], [0.0, -0.75]])

    @eqx.filter_jit
    def lanes(transport: ReverseDiffusion, lane_keys: Any, values: Any) -> Any:
        def one(key: Any, shift: Any) -> Any:
            realization = transport.realize(key, (1,))
            result = transport.solve(realization, context=ScoreContext({"shift": shift}))
            return result.final_states[0], result.valid[0]

        return jax.vmap(one)(lane_keys, values)

    batched, valid = lanes(reverse, keys, shifts)

    assert bool(jnp.all(valid))
    for index in range(3):
        single = reverse.solve(
            reverse.realize(keys[index], (1,)), context=_context(shifts[index])
        )
        np.testing.assert_allclose(
            batched[index], single.final_states[0], rtol=0.0, atol=1e-10
        )


def test_probability_flow_reads_score_key_and_context_through_args() -> None:
    process = _process()
    terminal = process.asymptotic_terminal_reference()
    shift = jnp.asarray([0.3, -0.6])
    conditional = probability_flow_system(
        process,
        _conditional_score(),
        state_layout=phx.dynamics.StateLayout((2,)),
        score_id="analytic-shifted-gaussian",
        context_labels=("shift",),
        score_inputs="args",
    )
    bound = probability_flow_system(
        process,
        _domain(context=False).Function("x", "t")(_FixedGaussianScore(shift)),
        state_layout=phx.dynamics.StateLayout((2,)),
        score_id="analytic-fixed-gaussian",
    )

    def transport(system: Any, args: Any) -> Any:
        return phx.transport.ContinuousTransport(
            terminal.law,
            phx.solver.DiffraxEvolution(system, rtol=1e-8, atol=1e-10),
            source_coordinate=0.0,
            target_coordinate=process.terminal_time,
            args=args,
        )

    conditioned = transport(
        conditional, (jr.key(0), _context(shift))
    ).sample_with_diagnostics(jr.key(2), (8,))
    reference = transport(bound, None).sample_with_diagnostics(jr.key(2), (8,))

    assert conditioned.successful and reference.successful
    np.testing.assert_allclose(
        conditioned.final_states, reference.final_states, rtol=0.0, atol=1e-9
    )
    np.testing.assert_allclose(
        np.asarray(conditioned.final_states).mean(axis=0), shift, atol=0.02
    )
    with pytest.raises(ValueError, match="args"):
        transport(conditional, None).sample_with_diagnostics(jr.key(2), (1,))
    with pytest.raises(ValueError, match="score_key"):
        probability_flow_system(
            process,
            _conditional_score(),
            state_layout=phx.dynamics.StateLayout((2,)),
            score_id="analytic-shifted-gaussian",
            score_key=jr.key(1),
            context_labels=("shift",),
            score_inputs="args",
        )

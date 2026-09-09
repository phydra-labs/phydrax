#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from phydrax.applications.skeletal_muscle.cellular import ShortenFastTwitchModel
from phydrax.applications.skeletal_muscle.fibers import (
    AbstractFiberReaction,
    PrescribedFiberStimulusSchedule,
    Shorten2007FiberReaction,
    StructuredFiberResponsePlan,
    StructuredFiberResponseState,
    StructuredFiberResponseStatus,
)


class _ManufacturedReaction(AbstractFiberReaction):
    """Integrate prescribed current and signed geometric stretch exactly."""

    def __init__(self):
        self.source_id = "manufactured-current-and-stretch-integrals"
        self.state_count = 2
        self.voltage_index = 0

    def initialize(self, batch_shape, /):
        return jnp.zeros(batch_shape + (2,))

    def rhs(self, time, values, current, stretch, rate, /):
        del time, values, stretch
        return jnp.stack((current, rate))

    def admissible(self, time, values, current, stretch, rate, /):
        del time
        return (
            jnp.all(jnp.isfinite(values))
            & jnp.isfinite(current)
            & (stretch > 0.0)
            & jnp.isfinite(rate)
        )


def _positions(x=(0.0, 0.4, 1.0, 2.0)):
    return jnp.zeros((1, len(x), 3)).at[0, :, 0].set(jnp.asarray(x))


def _runtime(*, theta=(0.0, 1.0), stimulus=None, diffusivity=0.3, **policy):
    positions = _positions()
    if stimulus is None:
        stimulus = PrescribedFiberStimulusSchedule(
            jnp.zeros((0,)),
            jnp.zeros((0,)),
            jnp.zeros((0,)),
            jnp.zeros((0, 1, 4), dtype=bool),
        )
    return StructuredFiberResponsePlan(
        ("fiber",),
        positions,
        stimulus,
        jnp.asarray(theta),
        geometry_source_id="manufactured-nonuniform-polyline-mm-world",
        **policy,
    ).prepare(_ManufacturedReaction(), jnp.full((1, 3), diffusivity))


def _state(runtime, voltage=(1.0, -0.5, 0.3, 0.0)):
    state = runtime.initialize()
    return eqx.tree_at(
        lambda s: s.values, state, state.values.at[0, :, 0].set(jnp.asarray(voltage))
    )


def _assert_same_state(actual, expected):
    assert actual.prepared_id == expected.prepared_id
    for a, b in zip(jax.tree.leaves(actual), jax.tree.leaves(expected), strict=True):
        np.testing.assert_array_equal(a, b)
        assert a.dtype == b.dtype


def _cn_oracle(voltage, positions, diffusivity, dt):
    lengths = np.sqrt(np.sum(np.diff(positions, axis=0) ** 2, axis=-1))
    mass = np.zeros(len(voltage))
    stiffness = np.zeros((len(voltage), len(voltage)))
    for index, length in enumerate(lengths):
        mass[index : index + 2] += 0.5 * length
        stiffness[index : index + 2, index : index + 2] += (
            diffusivity / length * np.array([[1.0, -1.0], [-1.0, 1.0]])
        )
    diagonal_mass = np.diag(mass)
    result = np.linalg.solve(
        diagonal_mass + 0.5 * dt * stiffness,
        (diagonal_mass - 0.5 * dt * stiffness) @ voltage,
    )
    return result, mass


def test_nonuniform_no_flux_response_conserves_weighted_voltage_and_matches_oracle():
    runtime = _runtime()
    state = _state(runtime)
    path = runtime.linear_geometry_path(state, state.node_positions_mm)
    candidate = eqx.filter_jit(runtime.candidate)(state, 0.04, path)
    assert bool(candidate.evidence.successful)
    accepted = runtime.commit(state, candidate)
    expected, mass = _cn_oracle(
        np.asarray(state.values[0, :, 0]), np.asarray(path[0, 0]), 0.3, 0.04
    )
    np.testing.assert_allclose(accepted.values[0, :, 0], expected, rtol=2e-10, atol=2e-10)
    np.testing.assert_allclose(
        mass @ accepted.values[0, :, 0], mass @ state.values[0, :, 0], atol=2e-12
    )
    assert int(accepted.accepted_steps) == 1


def test_moving_metric_changes_diffusion_and_integrates_signed_local_kinematics():
    runtime = _runtime()
    state = _state(runtime)
    path = runtime.linear_geometry_path(state, 2.0 * state.node_positions_mm)
    candidate = runtime.candidate(state, 0.04, path)
    assert bool(candidate.evidence.successful)
    accepted = runtime.commit(state, candidate)
    expected, _ = _cn_oracle(
        np.asarray(state.values[0, :, 0]),
        np.asarray(0.5 * (path[0, 0] + path[1, 0])),
        0.3,
        0.04,
    )
    np.testing.assert_allclose(accepted.values[0, :, 0], expected, atol=2e-10)
    np.testing.assert_allclose(accepted.values[..., 1], 1.0, atol=2e-10)
    np.testing.assert_allclose(candidate.geometry.segment_jacobian, 2.0)
    np.testing.assert_allclose(candidate.geometry.node_jacobian, 2.0)
    np.testing.assert_array_equal(accepted.node_positions_mm, path[-1])
    rotation = jnp.asarray([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    rotated = runtime.geometry(path[-1] @ rotation.T + jnp.asarray([2.0, -3.0, 1.0]))
    np.testing.assert_allclose(
        rotated.segment_conductance_mm_per_ms,
        candidate.geometry.segment_conductance_mm_per_ms,
    )


def test_event_aligned_substeps_integrate_pulse_without_endpoint_contamination():
    stimulus = PrescribedFiberStimulusSchedule(
        jnp.asarray([0.02]),
        jnp.asarray([0.02]),
        jnp.asarray([3.0]),
        jnp.ones((1, 1, 4), dtype=bool),
    )
    runtime = _runtime(theta=(0.0, 0.5, 1.0), stimulus=stimulus, diffusivity=0.0)
    state = runtime.initialize()
    path = runtime.linear_geometry_path(state, state.node_positions_mm)
    accepted = runtime.candidate(state, 0.04, path)
    assert bool(accepted.evidence.successful)
    np.testing.assert_allclose(accepted.commit(state).values[..., 0], 0.06, atol=2e-12)
    rejected = runtime.candidate(state, 0.03, path)
    assert int(rejected.evidence.status) & int(
        StructuredFiberResponseStatus.UNALIGNED_EVENT
    )
    _assert_same_state(rejected.commit(state), state)
    np.testing.assert_array_equal(rejected.evidence.reaction_solver_steps, 0)


def test_interior_segment_collapse_rejects_entire_geometry_cell_and_counter_transaction():
    runtime = _runtime()
    state = _state(runtime)
    # Both endpoints have nonzero lengths, but every segment collapses at theta=1/2.
    path = runtime.linear_geometry_path(state, -state.node_positions_mm)
    assert bool(runtime.geometry(path[-1]).valid)
    candidate = runtime.candidate(state, 0.04, path)
    assert int(candidate.evidence.status) & int(
        StructuredFiberResponseStatus.INVALID_GEOMETRY
    )
    _assert_same_state(candidate.commit(state), state)


def test_stale_candidate_keeps_current_state_and_foreign_numeric_binding_is_rejected():
    runtime = _runtime()
    state = _state(runtime)
    path = runtime.linear_geometry_path(state, state.node_positions_mm)
    candidate = runtime.candidate(state, 0.04, path)
    assert bool(candidate.evidence.successful)
    current = candidate.commit(state)
    _assert_same_state(candidate.commit(current), current)
    # A concurrent change with the same counter/time must also reject this source snapshot.
    changed = eqx.tree_at(
        lambda s: s.node_positions_mm, state, state.node_positions_mm + 1.0
    )
    _assert_same_state(candidate.commit(changed), changed)
    foreign = _runtime(diffusivity=0.4)
    assert foreign.semantic_id == runtime.semantic_id
    assert foreign.numeric_revision_id != runtime.numeric_revision_id
    with pytest.raises(ValueError, match="different prepared"):
        foreign.candidate(state, 0.04, path)


def test_reaction_parameter_changes_affect_prepared_numeric_not_source_identity():
    runtime = _runtime()
    model = ShortenFastTwitchModel()
    changed = ShortenFastTwitchModel(model.parameters.at[0].multiply(1.1))
    first = runtime.plan.prepare(Shorten2007FiberReaction(model), jnp.full((1, 3), 0.3))
    second = runtime.plan.prepare(
        Shorten2007FiberReaction(changed), jnp.full((1, 3), 0.3)
    )
    assert first.reaction.source_id == second.reaction.source_id
    assert first.semantic_id == second.semantic_id
    assert first.numeric_revision_id != second.numeric_revision_id
    with pytest.raises(ValueError, match="different prepared"):
        second.candidate(
            first.initialize(), 0.04, jnp.broadcast_to(_positions(), (2, 1, 4, 3))
        )


def test_geometry_response_jvp_vjp_agree_with_branch_local_finite_difference():
    runtime = _runtime(diffusivity=0.0)
    state = runtime.initialize()

    def response(scale):
        path = runtime.linear_geometry_path(state, scale * state.node_positions_mm)
        return jnp.sum(
            runtime.candidate(state, 0.04, path).candidate_state.values[..., 1]
        )

    scale = jnp.asarray(1.1)
    _, tangent = jax.jvp(response, (scale,), (jnp.asarray(1.0),))
    adjoint = jax.grad(response)(scale)
    difference = (response(scale + 1e-5) - response(scale - 1e-5)) / 2e-5
    np.testing.assert_allclose(tangent, 4.0, rtol=2e-8, atol=2e-8)
    np.testing.assert_allclose(adjoint, difference, rtol=2e-7, atol=2e-7)


def test_invalid_precision_geometry_and_unrepresentable_time_never_advance():
    runtime = _runtime()
    state = runtime.initialize()
    with pytest.raises(TypeError, match="real"):
        runtime.geometry(state.node_positions_mm.astype(complex) + 1j)
    with pytest.raises(ValueError, match="fixed substep"):
        runtime.candidate(state, 0.04, state.node_positions_mm)
    far_future = StructuredFiberResponseState(
        jnp.asarray(1e20, dtype=state.time_ms.dtype),
        state.values,
        state.node_positions_mm,
        state.accepted_steps,
        state.prepared_id,
    )
    path = runtime.linear_geometry_path(far_future, far_future.node_positions_mm)
    candidate = runtime.candidate(far_future, 0.04, path)
    assert int(candidate.evidence.status) & int(
        StructuredFiberResponseStatus.INVALID_STEP
    )
    _assert_same_state(candidate.commit(far_future), far_future)


def test_exhausted_local_reaction_and_invalid_dynamic_diffusivity_roll_back():
    runtime = _runtime(maximum_reaction_steps=1)
    state = _state(runtime)
    path = runtime.linear_geometry_path(state, 1.1 * state.node_positions_mm)
    candidate = runtime.candidate(state, 0.04, path)
    assert int(candidate.evidence.status) & int(
        StructuredFiberResponseStatus.REACTION_FAILURE
    )
    _assert_same_state(candidate.commit(state), state)
    invalid = eqx.tree_at(
        lambda prepared: prepared.diffusivity_mm2_per_ms,
        runtime,
        -runtime.diffusivity_mm2_per_ms,
    )
    rejected = invalid.candidate(state, 0.04, path)
    assert int(rejected.evidence.status) & int(StructuredFiberResponseStatus.INADMISSIBLE)
    np.testing.assert_array_equal(rejected.evidence.reaction_solver_steps, 0)
    _assert_same_state(rejected.commit(state), state)


def test_unrepresentable_reaction_half_step_rejects_before_local_solver():
    runtime = _runtime()
    state = runtime.initialize(float(2**48))
    path = runtime.linear_geometry_path(state, state.node_positions_mm)
    # The macro endpoints differ by one float64 ulp, but its midpoint is lost.
    candidate = runtime.candidate(state, 0.0625, path)
    assert int(candidate.evidence.status) & int(
        StructuredFiberResponseStatus.INVALID_STEP
    )
    np.testing.assert_array_equal(candidate.evidence.reaction_solver_steps, 0)
    _assert_same_state(candidate.commit(state), state)

#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import jax.numpy as jnp
import numpy as np

import phydrax as phx
from phydrax.applications.two_phase_flow._io import two_phase_inspection_frames
from phydrax.equations._flip_inspection import flip_inspection_frames


def _two_phase_case():
    grid = phx.discretization.TensorGridPlan(
        (
            phx.discretization.UniformCellAxisSpec(4, periodic=True),
            phx.discretization.UniformCellAxisSpec(4, periodic=True),
        ),
        axis_names=("x", "y"),
    ).prepare(jnp.asarray(((0.0, 0.0), (1.0, 1.0))))
    discretization = phx.discretization.FiniteVolumePlan(
        grid, component_names=("two-phase",)
    ).prepare()
    two_phase = phx.applications.two_phase_flow.IncompressibleTwoPhaseVOFPlan(
        discretization,
        phx.applications.two_phase_flow.TwoPhaseMaterialPlan(),
    ).prepare()
    velocity = tuple(jnp.zeros(layout.shape) for layout in discretization.face_layouts)
    accepted_state = two_phase.initial_state(
        jnp.zeros(discretization.cell_shape).at[1, 1].set(1.0), velocity
    )
    candidate_state = two_phase.initial_state(
        jnp.zeros(discretization.cell_shape).at[2, 1].set(1.0), velocity
    )
    method = phx.applications.two_phase_flow.IncompressibleTwoPhaseVOFMethod(two_phase)
    return (
        two_phase,
        method.initial_continuation(candidate_state),
        method.initial_continuation(accepted_state),
    )


def test_two_phase_host_inspection_keeps_candidate_and_rollback_distinct():
    two_phase, candidate_state, accepted_state = _two_phase_case()
    rejected = phx.solver.FixedStepResult(
        candidate_state,
        accepted_state,
        jnp.asarray(False),
        jnp.asarray(1.0),
        jnp.asarray(1, dtype=jnp.int32),
        jnp.asarray(1, dtype=jnp.int32),
        jnp.asarray(False),
        jnp.asarray(0.0),
    )

    candidate, accepted = two_phase_inspection_frames(
        two_phase,
        rejected,
        time=0.0,
        step=0,
        step_size=0.005,
        result_id="two-phase:attempt-1",
    )

    assert candidate.frame.state_kind == "candidate"
    assert accepted.frame.state_kind == "accepted"
    assert candidate.frame.result_id != accepted.frame.result_id
    assert np.isclose(candidate.frame.time, 0.005)
    assert candidate.frame.step == 1
    assert accepted.frame.time == 0.0
    assert accepted.frame.step == 0
    candidate_fields = {field.name: field for field in candidate.frame.fields}
    accepted_fields = {field.name: field for field in accepted.frame.fields}
    assert candidate_fields["velocity:x"].location == "face"
    assert (
        candidate_fields["velocity:x"].layout_id
        == two_phase.plan.discretization.face_layouts[0].layout_id
    )
    assert candidate_fields["alpha"].unit_id is None
    assert isinstance(candidate_fields["alpha"].values, np.ndarray)
    assert not candidate_fields["alpha"].values.flags.writeable
    np.testing.assert_array_equal(
        candidate_fields["geometry_epoch"].values,
        np.asarray(candidate_state.state.geometry_epoch),
    )
    assert "geometry=''" in candidate_fields["geometry_epoch"].provenance_id
    assert not np.array_equal(
        candidate_fields["alpha"].values, accepted_fields["alpha"].values
    )


def _flip_case():
    grid = phx.discretization.TensorGridPlan(
        tuple(phx.discretization.UniformCellAxisSpec(8) for _ in range(2)),
        axis_names=("x", "y"),
    ).prepare(jnp.asarray([[0.0, 0.0], [1.0, 1.0]]))
    finite_volume = phx.discretization.FiniteVolumePlan(grid).prepare()
    mac = phx.discretization.MACOperatorPlan(finite_volume).prepare()
    boundaries = phx.discretization.MACBoundaryPlan(mac).prepare()
    projection = phx.solver.MACFreeSurfaceProjectionPlan(
        mac, boundaries=boundaries, tolerance=1.0e-7
    )
    position = jnp.asarray([[0.25, 0.25], [0.40, 0.25], [0.25, 0.40], [0.40, 0.40]])
    active = jnp.asarray([True, False, True, True])
    particles = phx.discretization.ParticleSetPlan(
        jnp.arange(4),
        jnp.ones((4,)),
        ambient_dimension=2,
        active_mask=active,
    ).prepare()
    transfer = phx.discretization.flip.FLIPParticleTransferPlan(mac).prepare(particles)
    compiled = phx.equations.compile_flip_problem(
        phx.equations.FLIPProblemIR("inspection", 1.0, jnp.asarray([0.0, -0.1])),
        transfer,
        projection,
        phx.discretization.flip.FLIPMethodPlan(0.05, liquid_fraction_threshold=0.01),
    )
    state = compiled.initialize_state(position, jnp.zeros_like(position))
    return finite_volume, particles, active, compiled, state


def test_flip_host_inspection_preserves_capacity_masks_and_face_layouts():
    finite_volume, particles, active, compiled, state = _flip_case()
    result = compiled.step_detailed(state, jnp.asarray(1.0e-4))

    candidate, accepted = flip_inspection_frames(
        compiled, result, result_id="flip:attempt-1"
    )

    assert candidate.frame.state_kind == "candidate"
    assert accepted.frame.state_kind == "accepted"
    assert candidate.frame.result_id != accepted.frame.result_id
    fields = {field.name: field for field in candidate.frame.fields}
    position_field = fields["position"]
    assert position_field.values.shape == (particles.capacity, 2)
    np.testing.assert_array_equal(position_field.valid, np.asarray(active))
    np.testing.assert_array_equal(
        position_field.values,
        np.asarray(result.candidate_state.particles.position),
    )
    assert not position_field.values.flags.writeable
    assert position_field.unit_id is None
    assert fields["attempt_pre_grid_velocity:x"].location == "face"
    assert (
        fields["attempt_pre_grid_velocity:x"].layout_id
        == finite_volume.face_layouts[0].layout_id
    )

#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import jax.numpy as jnp
import pytest

import phydrax as phx


def _compile_periodic(control, radii):
    particles = phx.discretization.ParticleSetPlan(
        jnp.asarray([10, 20]), jnp.ones((2,)), ambient_dimension=2
    ).prepare()
    spheres = phx.discretization.RigidSphereSetPlan(
        jnp.asarray(radii), jnp.asarray([0, 0])
    )
    materials = phx.equations.DEMMaterialTable(
        jnp.asarray([2.0e5]),
        jnp.asarray([0.25]),
        jnp.asarray([[0.8]]),
        jnp.asarray([[0.4]]),
    )
    method = phx.discretization.SoftSphereDEMMethodPlan(
        phx.discretization.DEMContactModelPlan(
            phx.discretization.LinearSpringDashpotNormalPlan(1.0e4)
        ),
        periodic_cell_control=control,
        maximum_overlap_fraction=0.5,
    )
    problem = phx.equations.DiscreteElementProblemIR(
        "periodic-rheology", materials, gravity=jnp.zeros((2,))
    )
    cell = phx.discretization.ParticleCell(
        jnp.eye(2),
        periodic_axes=(True, True),
        maximum_condition_number=control.maximum_condition_number,
    )
    neighborhood = phx.discretization.DenseParticleNeighborhoodPlan(1, box=cell)
    return phx.equations.compile_discrete_element_problem(
        problem, particles, spheres, method, neighborhood=neighborhood
    )


def test_prescribed_deforming_cell_preserves_fractional_positions_and_work_ledger():
    control = phx.discretization.DEMPeriodicCellControlPlan(
        jnp.asarray([[0.0, 0.2], [0.0, 0.0]]),
        strain_rate_mask=jnp.asarray([[False, True], [False, False]]),
        maximum_condition_number=1.5,
    )
    compiled = _compile_periodic(control, (0.04, 0.04))
    state = compiled.initialize_state(
        0.0,
        jnp.asarray([[0.25, 0.25], [0.75, 0.75]]),
        jnp.zeros((2, 2)),
    )
    initial_fractional = compiled.dynamics.periodic_cell.fractional_with_vectors(
        state.kinematics.position, state.periodic_cell.vectors
    )
    detail = compiled.dynamics.step_detailed(
        jnp.asarray(0, dtype=jnp.int32),
        jnp.asarray(0.0),
        state,
        jnp.asarray(1.0e-3),
        None,
    )
    final_fractional = compiled.dynamics.periodic_cell.fractional_with_vectors(
        detail.accepted_state.kinematics.position,
        detail.accepted_state.periodic_cell.vectors,
    )

    assert detail.successful
    assert not jnp.allclose(
        detail.accepted_state.periodic_cell.vectors, state.periodic_cell.vectors
    )
    assert jnp.allclose(final_fractional, initial_fractional, atol=1.0e-12)
    assert jnp.isclose(
        detail.energy.cell_work, detail.accepted_state.periodic_cell.last_work
    )
    assert jnp.isclose(
        detail.accepted_state.energy.cumulative_cell_work,
        detail.accepted_state.periodic_cell.cumulative_work,
    )
    assert detail.evaluation.bulk_stress.successful


def test_mixed_stress_strain_control_uses_periodic_contact_stress():
    control = phx.discretization.DEMPeriodicCellControlPlan(
        jnp.asarray([[0.0, 0.1], [0.0, 0.0]]),
        strain_rate_mask=jnp.asarray([[False, True], [False, False]]),
        target_stress=jnp.zeros((2, 2)),
        stress_mask=jnp.asarray([[True, False], [False, False]]),
        stress_compliance=jnp.asarray([[1.0e-4, 0.0], [0.0, 0.0]]),
        maximum_condition_number=1.5,
    )
    compiled = _compile_periodic(control, (0.06, 0.06))
    state = compiled.initialize_state(
        0.0,
        jnp.asarray([[0.05, 0.5], [0.95, 0.5]]),
        jnp.zeros((2, 2)),
    )
    evaluation = compiled.dynamics.evaluate(
        jnp.asarray(0.0), state, jnp.asarray(1.0e-3), None
    )
    assert evaluation.particle_contact.active[0]
    assert evaluation.bulk_stress.contact_stress[0, 0] < 0.0

    detail = compiled.dynamics.step_detailed(
        jnp.asarray(0, dtype=jnp.int32),
        jnp.asarray(0.0),
        state,
        jnp.asarray(1.0e-3),
        None,
    )
    assert detail.successful
    assert detail.accepted_state.periodic_cell.vectors[0, 0] > 1.0
    assert detail.accepted_state.periodic_cell.vectors[1, 0] > 0.0
    assert jnp.isfinite(detail.energy.energy_residual)

    with pytest.raises(ValueError, match="disjoint"):
        phx.discretization.DEMPeriodicCellControlPlan(
            jnp.zeros((2, 2)),
            strain_rate_mask=jnp.eye(2, dtype=bool),
            stress_mask=jnp.eye(2, dtype=bool),
            stress_compliance=jnp.ones((2, 2)),
        )


def test_cell_control_failure_rolls_back_atomically():
    control = phx.discretization.DEMPeriodicCellControlPlan(
        jnp.asarray([[0.0, 100.0], [0.0, 0.0]]),
        strain_rate_mask=jnp.asarray([[False, True], [False, False]]),
        maximum_strain_increment=1.0e-4,
        maximum_condition_number=1.5,
    )
    compiled = _compile_periodic(control, (0.04, 0.04))
    state = compiled.initialize_state(
        0.0,
        jnp.asarray([[0.25, 0.25], [0.75, 0.75]]),
        jnp.zeros((2, 2)),
    )
    detail = compiled.dynamics.step_detailed(
        jnp.asarray(0, dtype=jnp.int32),
        jnp.asarray(0.0),
        state,
        jnp.asarray(1.0e-3),
        None,
    )
    assert not detail.successful
    assert detail.rejection_reasons & int(
        phx.discretization.DEMRejectionReason.CELL_CONTROL
    )
    assert jnp.array_equal(
        detail.accepted_state.periodic_cell.vectors, state.periodic_cell.vectors
    )

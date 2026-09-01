#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest

import phydrax as phx


def _compiled(transfer, schedule, *, fields=None):
    grid = phx.discretization.TensorGridPlan(
        tuple(
            phx.discretization.UniformAxisSpec(12, periodic=True, endpoint=False)
            for _ in range(2)
        ),
        axis_names=("x", "y"),
    ).prepare(jnp.asarray([[0.0, 0.0], [1.0, 1.0]]))
    position = jnp.asarray([[0.28, 0.31], [0.42, 0.36], [0.34, 0.49], [0.48, 0.52]])
    volume = jnp.full((4,), 0.01)
    particles = phx.discretization.ParticleSetPlan(
        jnp.arange(4), volume, ambient_dimension=2
    ).prepare()
    splat = phx.discretization.ParticleGridSplatPlan(
        grid, assignment=phx.discretization.TensorBSplineSplatAssignment(2)
    ).prepare(particles)
    method = phx.discretization.ExplicitMPMMethodPlan(
        transfer,
        schedule=schedule,
    )
    compiled = phx.equations.compile_material_point_problem(
        phx.equations.MaterialPointProblemIR(
            "transfer-family",
            phx.applications.solid_mechanics.NeoHookeanMPMConstitutivePlan(2),
        ),
        particles,
        splat,
        method,
        phx.discretization.MPMParticleDomainPlan(
            jnp.asarray([[0.0, 0.0], [1.0, 1.0]]),
            periodic=(True, True),
            support_margin=0.0,
        ),
        nodal_fields=fields,
    )
    arguments = phx.equations.MaterialPointArguments(
        phx.applications.solid_mechanics.NeoHookeanParameters.from_shear_bulk(2.0, 8.0)
    )
    velocity = jnp.broadcast_to(jnp.asarray((0.05, -0.02)), position.shape)
    state = compiled.initialize_state(position, velocity, volume, arguments)
    return compiled, arguments, state, velocity


@pytest.mark.parametrize(
    "transfer",
    [
        phx.discretization.PICTransferPlan(),
        phx.discretization.FLIPTransferPlan(),
        phx.discretization.PICFLIPTransferPlan(0.25),
        phx.discretization.APICTransferPlan(),
    ],
)
def test_transfer_family_preserves_constant_translation(transfer):
    compiled, arguments, state, velocity = _compiled(
        transfer, phx.discretization.USLMPMSchedule()
    )
    detail = compiled.dynamics.step_detailed(state, 0.001, arguments)
    assert bool(detail.successful)
    np.testing.assert_allclose(
        detail.accepted_state.particles.velocity, velocity, rtol=2e-10, atol=2e-10
    )
    if not transfer.requires_affine_state:
        np.testing.assert_allclose(
            detail.accepted_state.particles.affine_velocity, 0.0, atol=1e-14
        )


def test_advection_plans_are_independent_of_velocity_transfer():
    transferred = jnp.asarray([[2.0, 0.0]])
    pic = jnp.asarray([[1.0, 0.0]])
    previous = jnp.asarray([[0.0, 0.0]])
    np.testing.assert_array_equal(
        phx.discretization.PICAdvectionPlan().velocity(transferred, pic, previous), pic
    )
    np.testing.assert_array_equal(
        phx.discretization.TransferredVelocityAdvectionPlan().velocity(
            transferred, pic, previous
        ),
        transferred,
    )
    np.testing.assert_array_equal(
        phx.discretization.MidpointAdvectionPlan().velocity(transferred, pic, previous),
        [[1.0, 0.0]],
    )


@pytest.mark.parametrize(
    "schedule",
    [
        phx.discretization.AffineMUSLMPMSchedule(),
        phx.discretization.PostAdvectionMUSLMPMSchedule(),
        phx.discretization.PostAdvectionMUSLMPMSchedule(affine_transfer=True),
    ],
)
def test_remaining_musl_variants_rebuild_and_certify_second_transfer(schedule):
    compiled, arguments, state, _ = _compiled(
        phx.discretization.APICTransferPlan(), schedule
    )
    detail = compiled.dynamics.step_detailed(state, 0.001, arguments)
    assert bool(detail.successful)
    assert bool(detail.diagnostics.schedule.second_momentum_extrapolation)
    assert detail.diagnostics.schedule.second_transfer_mass_defect < 1e-10
    assert detail.diagnostics.schedule.second_transfer_momentum_defect < 1e-9


@pytest.mark.parametrize(
    "schedule",
    [
        phx.discretization.USFMPMSchedule(),
        phx.discretization.MUSLMPMSchedule(),
        phx.discretization.AffineMUSLMPMSchedule(),
        phx.discretization.PostAdvectionMUSLMPMSchedule(),
    ],
)
def test_multifield_schedules_reapply_simultaneous_constraints(schedule):
    slots = jnp.asarray((0, 0, 1, 1), dtype=jnp.int32)
    contact = phx.discretization.KWayMPMContactPlan(
        2,
        friction=phx.discretization.SharpCoulombMPMFrictionPlan(0.1),
        maximum_steps=40,
        tolerance=1e-8,
    )
    fields = phx.discretization.MPMNodalFieldPlan(
        ("left", "right"), slots, contact_plan=contact
    )
    compiled, arguments, state, _ = _compiled(
        phx.discretization.APICTransferPlan(), schedule, fields=fields
    )
    state = phx.discretization.MPMRuntimeState(
        state.particles,
        state.time,
        state.accepted_step,
        state.last_status,
        state.topology_generation,
        state.assignment_input,
        state.material_slots,
        slots,
        slots,
        state.storage_state,
    )
    detail = compiled.dynamics.step_detailed(state, 0.0005, arguments)
    assert bool(detail.successful)
    assert detail.grid.mass.shape[0] == 2
    assert bool(detail.diagnostics.transfer.field_contact_successful)

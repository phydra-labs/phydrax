#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest

import phydrax as phx


def _case(schedule, *, clamp_x=False):
    grid = phx.discretization.TensorGridPlan(
        tuple(
            phx.discretization.UniformAxisSpec(12, periodic=True, endpoint=False)
            for _ in range(2)
        ),
        axis_names=("x", "y"),
    ).prepare(jnp.asarray([[0.0, 0.0], [1.0, 1.0]]))
    particles = phx.discretization.ParticleSetPlan(
        jnp.arange(4), jnp.full((4,), 0.01), ambient_dimension=2
    ).prepare()
    splat = phx.discretization.ParticleGridSplatPlan(
        grid,
        assignment=phx.discretization.TensorBSplineSplatAssignment(2),
    ).prepare(particles)
    boundary = None
    if clamp_x:
        mask = jnp.zeros(grid.vertices().shape + (2,), dtype=bool)
        boundary = phx.discretization.PrescribedGridVelocityPlan(
            mask.at[..., 0].set(True)
        )
    compiled = phx.equations.compile_material_point_problem(
        phx.equations.MaterialPointProblemIR(
            "schedule-test",
            phx.applications.solid_mechanics.NeoHookeanMPMConstitutivePlan(2),
        ),
        particles,
        splat,
        phx.discretization.ExplicitMPMMethodPlan(schedule=schedule),
        phx.discretization.MPMParticleDomainPlan(
            jnp.asarray([[0.0, 0.0], [1.0, 1.0]]),
            periodic=(True, True),
            support_margin=0.0,
        ),
        boundary=boundary,
    )
    arguments = phx.equations.MaterialPointArguments(
        phx.applications.solid_mechanics.NeoHookeanParameters.from_shear_bulk(2.0, 8.0)
    )
    position = jnp.asarray([[0.28, 0.31], [0.42, 0.36], [0.34, 0.49], [0.48, 0.52]])
    return compiled, arguments, position


@pytest.mark.parametrize(
    ("schedule", "code", "stress_first", "second_transfer"),
    [
        (phx.discretization.USLMPMSchedule(), 0, False, False),
        (phx.discretization.USFMPMSchedule(), 1, True, False),
        (phx.discretization.MUSLMPMSchedule(), 2, False, True),
    ],
)
def test_explicit_schedule_phase_identity_and_translation(
    schedule, code, stress_first, second_transfer
):
    compiled, arguments, position = _case(schedule)
    velocity = jnp.broadcast_to(jnp.asarray((0.08, -0.03)), position.shape)
    state = compiled.initialize_state(position, velocity, jnp.full((4,), 0.01), arguments)
    detail = compiled.dynamics.step_detailed(state, 0.001, arguments)

    assert bool(detail.successful)
    assert int(detail.diagnostics.schedule.schedule_code) == code
    assert bool(detail.diagnostics.schedule.stress_updated_first) is stress_first
    assert (
        bool(detail.diagnostics.schedule.second_momentum_extrapolation) is second_transfer
    )
    assert bool(detail.diagnostics.schedule.successful)
    assert detail.grid.mass.shape == (1,) + compiled.dynamics.splat.target_shape
    np.testing.assert_allclose(
        detail.accepted_state.particles.position,
        position + 0.001 * velocity,
        rtol=2e-11,
        atol=2e-11,
    )
    np.testing.assert_allclose(
        detail.accepted_state.particles.velocity,
        velocity,
        rtol=2e-11,
        atol=2e-11,
    )
    if second_transfer:
        assert detail.diagnostics.schedule.second_transfer_mass_defect < 1e-12
        assert detail.diagnostics.schedule.second_transfer_momentum_defect < 1e-10


def test_musl_reapplies_prescribed_constraints_after_second_transfer():
    compiled, arguments, position = _case(
        phx.discretization.MUSLMPMSchedule(), clamp_x=True
    )
    velocity = jnp.broadcast_to(jnp.asarray((0.1, 0.0)), position.shape)
    state = compiled.initialize_state(position, velocity, jnp.full((4,), 0.01), arguments)
    detail = compiled.dynamics.step_detailed(state, 0.001, arguments)

    assert bool(detail.successful)
    np.testing.assert_allclose(
        detail.accepted_state.particles.velocity[:, 0], 0.0, atol=1e-12
    )
    assert detail.diagnostics.schedule.second_constraint_work <= 0.0
    assert detail.diagnostics.energy.boundary_work <= 0.0


def test_default_explicit_method_remains_usl_minus():
    method = phx.discretization.ExplicitMPMMethodPlan()
    assert isinstance(method.schedule, phx.discretization.USLMPMSchedule)
    assert method.schedule.common_name == "usl-minus"

#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

import phydrax as phx


def _compiled(field_plan=None):
    grid = phx.discretization.TensorGridPlan(
        tuple(
            phx.discretization.UniformAxisSpec(16, periodic=True, endpoint=False)
            for _ in range(2)
        ),
        axis_names=("x", "y"),
    ).prepare(jnp.asarray([[0.0, 0.0], [1.0, 1.0]]))
    position = jnp.asarray([[0.40, 0.47], [0.43, 0.53], [0.57, 0.47], [0.60, 0.53]])
    volume = jnp.full((4,), 0.01)
    particles = phx.discretization.ParticleSetPlan(
        jnp.arange(4), volume, ambient_dimension=2
    ).prepare()
    splat = phx.discretization.ParticleGridSplatPlan(
        grid, assignment=phx.discretization.TensorBSplineSplatAssignment(2)
    ).prepare(particles)
    compiled = phx.equations.compile_material_point_problem(
        phx.equations.MaterialPointProblemIR(
            "multifield",
            phx.applications.solid_mechanics.NeoHookeanMPMConstitutivePlan(2),
        ),
        particles,
        splat,
        phx.discretization.ExplicitMPMMethodPlan(),
        phx.discretization.MPMParticleDomainPlan(
            jnp.asarray([[0.0, 0.0], [1.0, 1.0]]),
            periodic=(True, True),
            support_margin=0.0,
        ),
        nodal_fields=field_plan,
    )
    arguments = phx.equations.MaterialPointArguments(
        phx.applications.solid_mechanics.NeoHookeanParameters.from_shear_bulk(2.0, 8.0)
    )
    return compiled, arguments, position, volume


def test_explicit_single_field_plan_is_exact_default_migration():
    default, arguments, position, volume = _compiled()
    explicit, _, _, _ = _compiled(
        phx.discretization.MPMNodalFieldPlan(
            ("material",), jnp.zeros((4,), dtype=jnp.int32)
        )
    )
    velocity = jnp.broadcast_to(jnp.asarray((0.04, -0.01)), position.shape)
    first = default.initialize_state(position, velocity, volume, arguments)
    second = explicit.initialize_state(position, velocity, volume, arguments)
    first_result = default.dynamics.step_detailed(first, 0.001, arguments)
    second_result = explicit.dynamics.step_detailed(second, 0.001, arguments)

    for left, right in zip(
        jax.tree.leaves(first_result.accepted_state.particles),
        jax.tree.leaves(second_result.accepted_state.particles),
        strict=True,
    ):
        np.testing.assert_array_equal(left, right)
    assert first_result.grid.mass.shape[0] == 1


def test_two_field_contact_preserves_action_reaction_and_separate_grid_fields():
    slots = jnp.asarray((0, 0, 1, 1), dtype=jnp.int32)
    fields = phx.discretization.MPMNodalFieldPlan(
        ("left", "right"),
        slots,
        contact_friction=phx.discretization.SharpCoulombMPMFrictionPlan(0.2),
    )
    compiled, arguments, position, volume = _compiled(fields)
    velocity = jnp.asarray([[0.12, 0.02], [0.12, 0.02], [-0.12, -0.01], [-0.12, -0.01]])
    state = compiled.initialize_state(
        position,
        velocity,
        volume,
        arguments,
        velocity_field_slots=slots,
        body_ids=slots,
    )
    detail = compiled.dynamics.step_detailed(state, 0.001, arguments)

    assert bool(detail.successful)
    assert detail.grid.mass.shape[0] == 2
    assert bool(detail.diagnostics.transfer.field_contact_successful)
    assert detail.diagnostics.transfer.field_action_reaction_defect < 1e-12
    assert detail.diagnostics.energy.contact_dissipation >= 0.0
    np.testing.assert_array_equal(detail.accepted_state.velocity_field_slots, slots)


def test_direct_two_field_projection_stops_approach_and_obeys_friction_cone():
    mass = jnp.asarray([[1.0], [2.0]])
    velocity = jnp.asarray([[[1.0, 0.4]], [[-0.5, 0.0]]])
    gradients = jnp.asarray([[[1.0, 0.0]], [[-1.0, 0.0]]])
    friction = phx.discretization.SharpCoulombMPMFrictionPlan(0.3)
    result = phx.discretization.project_two_field_contact(
        mass, velocity, gradients, friction=friction
    )

    assert bool(result.successful)
    assert bool(result.contact_mask[0])
    relative = result.velocity[0, 0] - result.velocity[1, 0]
    assert relative[0] >= -1e-12
    assert result.action_reaction_defect < 1e-12
    normal_impulse = abs(result.impulse[0, 0])
    tangential_impulse = abs(result.impulse[0, 1])
    assert tangential_impulse <= 0.3 * normal_impulse + 1e-12


def test_material_bank_accepts_disjoint_heterogeneous_histories():
    neo = phx.applications.solid_mechanics.NeoHookeanMPMConstitutivePlan(3)
    plastic = phx.applications.solid_mechanics.FiniteStrainJ2MPMConstitutivePlan()
    bank = phx.discretization.MPMMaterialBank(
        (
            phx.discretization.MPMMaterialBankEntry(
                neo, jnp.asarray((0, 2)), entry_id="elastic"
            ),
            phx.discretization.MPMMaterialBankEntry(
                plastic, jnp.asarray((1, 3)), entry_id="plastic"
            ),
        )
    )
    state = phx.discretization.MPMMaterialBankState(
        (
            neo.initialize_state((2,), jnp.float64),
            plastic.initialize_state((2,), jnp.float64),
        )
    )
    assert len(bank.entries) == 2
    assert state.histories[0].shape == (2, 0)
    assert state.histories[1].shape == (2, 10)

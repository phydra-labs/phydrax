#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest

import phydrax as phx


def _support(assignment):
    grid = phx.discretization.TensorGridPlan(
        tuple(
            phx.discretization.UniformAxisSpec(16, periodic=True, endpoint=False)
            for _ in range(2)
        ),
        axis_names=("x", "y"),
    ).prepare(jnp.asarray([[0.0, 0.0], [1.0, 1.0]]))
    particles = phx.discretization.ParticleSetPlan(
        jnp.arange(3), jnp.full((3,), 0.01), ambient_dimension=2
    ).prepare()
    splat = phx.discretization.ParticleGridSplatPlan(grid, assignment=assignment).prepare(
        particles
    )
    position = jnp.asarray([[0.27, 0.31], [0.43, 0.38], [0.36, 0.52]])
    deformation = jnp.asarray(
        [
            [[1.0, 0.0], [0.0, 1.0]],
            [[1.08, 0.04], [0.02, 0.94]],
            [[0.96, -0.03], [0.05, 1.06]],
        ]
    )
    assignment_input = assignment.update_input(position, deformation, None)
    state = splat.build(position, assignment_input=assignment_input)
    return grid, particles, splat, position, deformation, assignment_input, state


@pytest.mark.parametrize("evolving", [False, True])
def test_gimp_partition_gradient_moment_and_apic_compatibility(evolving):
    widths = jnp.full((3, 2), 0.025)
    assignment = phx.discretization.UniformGIMPSplatAssignment(
        widths, maximum_half_width_cells=0.75, evolving=evolving
    )
    _, _, _, _, _, assignment_input, state = _support(assignment)

    assert bool(state.successful)
    np.testing.assert_allclose(state.partition_sums, 1.0, atol=2e-12)
    np.testing.assert_allclose(state.gradient_sums, 0.0, atol=2e-11)
    np.testing.assert_allclose(state.first_moments, 0.0, atol=2e-11)
    assert jnp.all(jnp.linalg.det(state.second_moments) > 0.0)
    if evolving:
        assert isinstance(assignment_input, phx.discretization.GIMPAssignmentInput)
        assert not jnp.allclose(assignment_input.half_widths, widths)
    else:
        assert assignment_input is None


@pytest.mark.parametrize("kind", ["cpdi", "cpdi2"])
def test_cpdi_variants_are_complete_and_domain_state_is_transactional(kind):
    if kind == "cpdi":
        reference = jnp.broadcast_to(0.025 * jnp.eye(2), (3, 2, 2))
        assignment = phx.discretization.AffineCPDISplatAssignment(
            reference, maximum_extent_cells=1.5
        )
    else:
        signs = jnp.asarray(((-1.0, -1.0), (-1.0, 1.0), (1.0, -1.0), (1.0, 1.0)))
        reference = jnp.broadcast_to(0.025 * signs, (3, 4, 2))
        assignment = phx.discretization.CPDI2SplatAssignment(
            reference, maximum_extent_cells=1.5
        )
    _, _, _, position, deformation, assignment_input, state = _support(assignment)

    assert bool(state.successful)
    np.testing.assert_allclose(state.partition_sums, 1.0, atol=2e-12)
    np.testing.assert_allclose(state.gradient_sums, 0.0, atol=2e-10)
    np.testing.assert_allclose(state.first_moments, 0.0, atol=2e-10)
    assert jnp.all(jnp.linalg.det(state.second_moments) > 0.0)

    moved = position + jnp.asarray((0.01, -0.005))
    next_deformation = deformation.at[:, 0, 1].add(0.02)
    next_input = assignment.update_input(moved, next_deformation, assignment_input)
    assert type(next_input) is type(assignment_input)
    if kind == "cpdi2":
        assert not jnp.allclose(next_input.corners, assignment_input.corners)
        np.testing.assert_array_equal(assignment_input.center, position)


def test_cpdi_inversion_rejects_routes_and_mpm_attempt():
    reference = jnp.broadcast_to(0.025 * jnp.eye(2), (3, 2, 2))
    assignment = phx.discretization.AffineCPDISplatAssignment(reference)
    grid, particles, splat, position, _, _, _ = _support(assignment)
    inverted = jnp.broadcast_to(jnp.asarray([[-1.0, 0.0], [0.0, 1.0]]), (3, 2, 2))
    assignment_input = assignment.update_input(position, inverted, None)
    route = splat.build(position, assignment_input=assignment_input)
    assert not bool(route.successful)

    compiled = phx.equations.compile_material_point_problem(
        phx.equations.MaterialPointProblemIR(
            "cpdi-inversion",
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
    )
    arguments = phx.equations.MaterialPointArguments(
        phx.applications.solid_mechanics.NeoHookeanParameters.from_shear_bulk(2.0, 8.0)
    )
    with pytest.raises(Exception, match="Initial MPM state is inadmissible"):
        compiled.initialize_state(
            position,
            jnp.zeros_like(position),
            jnp.full((3,), 0.01),
            arguments,
            deformation_gradient=inverted,
        )

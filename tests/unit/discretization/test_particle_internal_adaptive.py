#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import jax.numpy as jnp
import numpy as np

import phydrax as phx


def _quad_grid(nx, ny):
    vertices = np.asarray(
        [(2.0 * i / nx, j / ny) for j in range(ny + 1) for i in range(nx + 1)]
    )
    cells = []
    for j in range(ny):
        for i in range(nx):
            lower_left = j * (nx + 1) + i
            lower_right = lower_left + 1
            upper_left = lower_left + nx + 1
            upper_right = upper_left + 1
            cells.append((lower_left, lower_right, upper_right, upper_left))
    return phx.discretization.UnstructuredFiniteVolumePlan(
        vertices,
        quadrilaterals=np.asarray(cells),
        cell_global_ids=np.arange(1000 + 100 * nx, 1000 + 100 * nx + len(cells)),
    ).prepare()


def _hierarchy(maximum_refined_cells=2):
    coarse = _quad_grid(2, 1)
    fine = _quad_grid(4, 2)
    parent = np.asarray((0, 0, 1, 1, 0, 0, 1, 1), dtype=np.int32)
    prolongation = phx.discretization.UnstructuredConservativeRemapPlan(
        coarse,
        fine,
        np.arange(fine.cell_count + 1, dtype=np.int32),
        parent,
        fine.cell_volumes,
        method="particle-nested-prolongation",
        provenance="analytic-particle-refinement",
    )
    restriction = phx.discretization.UnstructuredConservativeRemapPlan(
        fine,
        coarse,
        np.asarray((0, 4, 8), dtype=np.int32),
        np.asarray((0, 1, 4, 5, 2, 3, 6, 7), dtype=np.int32),
        np.asarray((0.25,) * 8),
        method="particle-nested-restriction",
        provenance="analytic-particle-refinement",
    )
    return phx.discretization.UnstructuredAMRHierarchyPlan(
        coarse,
        fine,
        prolongation,
        restriction,
        maximum_refined_cells=maximum_refined_cells,
    )


def _state(hierarchy):
    return phx.discretization.initialize_particle_internal_amr(
        hierarchy,
        jnp.asarray([[2.0, 4.0]]),
        jnp.asarray([[[1.0], [3.0]]]),
        jnp.asarray([[0.2, 0.4]]),
        jnp.asarray([[2.0, 3.0]]),
        jnp.asarray([[[0.25], [0.75]]]),
        jnp.asarray([1.0]),
        jnp.asarray([True]),
    )


def test_particle_internal_amr_refines_locally_and_conserves_extensive_state():
    hierarchy = _hierarchy()
    state = _state(hierarchy)
    result = phx.discretization.adapt_particle_internal_mesh(
        hierarchy,
        phx.discretization.ParticleInternalAdaptationPolicy(1.0, 0.5),
        state,
        jnp.asarray([[2.0, 0.0]]),
    )
    assert result.successful
    assert jnp.array_equal(result.accepted_state.coarse_refined, [[True, False]])
    assert jnp.sum(result.accepted_state.fine_active) == 4
    assert jnp.max(jnp.abs(result.evidence.energy_residual)) < 1.0e-12
    assert jnp.max(jnp.abs(result.evidence.species_residual)) < 1.0e-12
    assert jnp.max(jnp.abs(result.evidence.pore_volume_residual)) < 1.0e-12
    assert jnp.max(jnp.abs(result.evidence.surface_area_residual)) < 1.0e-12


def test_particle_internal_amr_coarsening_preserves_bounded_progress():
    hierarchy = _hierarchy()
    state = _state(hierarchy)
    policy = phx.discretization.ParticleInternalAdaptationPolicy(
        1.0, 0.5, minimum_dwell_windows=0
    )
    refined = phx.discretization.adapt_particle_internal_mesh(
        hierarchy, policy, state, jnp.asarray([[2.0, 0.0]])
    ).accepted_state
    coarsened = phx.discretization.adapt_particle_internal_mesh(
        hierarchy, policy, refined, jnp.zeros((1, 2))
    )
    assert coarsened.successful
    assert not jnp.any(coarsened.accepted_state.coarse_refined)
    assert jnp.all(
        (coarsened.accepted_state.coarse_reaction_progress >= 0.0)
        & (coarsened.accepted_state.coarse_reaction_progress <= 1.0)
    )


def test_particle_internal_amr_overflow_requests_growth_atomically():
    hierarchy = _hierarchy(maximum_refined_cells=1)
    state = _state(hierarchy)
    result = phx.discretization.adapt_particle_internal_mesh(
        hierarchy,
        phx.discretization.ParticleInternalAdaptationPolicy(1.0, 0.5),
        state,
        jnp.asarray([[2.0, 2.0]]),
    )
    assert not result.successful
    assert result.growth_required
    assert result.required_additional_cells == 1
    assert jnp.array_equal(result.accepted_state.coarse_refined, state.coarse_refined)


def test_coarse_fine_flux_correction_uses_extensive_register_once():
    hierarchy = _hierarchy()
    coarse = jnp.asarray([1.0, 2.0])
    register = phx.discretization.UnstructuredAMRFluxRegister(
        jnp.asarray([0.5, -0.5]),
        route_id=hierarchy.interface_route_id,
        layout_id=hierarchy.interface_layout_id,
    )
    corrected = phx.discretization.apply_particle_internal_flux_correction(
        hierarchy, coarse, register
    )
    expected = coarse + register.integrated_correction / hierarchy.coarse.cell_volumes
    assert jnp.allclose(corrected, expected)

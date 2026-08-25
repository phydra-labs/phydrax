#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import equinox as eqx
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


def _hierarchy():
    coarse = _quad_grid(2, 1)
    fine = _quad_grid(4, 2)
    parent = np.asarray((0, 0, 1, 1, 0, 0, 1, 1), dtype=np.int32)
    prolongation = phx.discretization.UnstructuredConservativeRemapPlan(
        coarse,
        fine,
        np.arange(fine.cell_count + 1, dtype=np.int32),
        parent,
        fine.cell_volumes,
        method="nested-constant-prolongation",
        provenance="analytic-2x-refinement",
    )
    restriction = phx.discretization.UnstructuredConservativeRemapPlan(
        fine,
        coarse,
        np.asarray((0, 4, 8), dtype=np.int32),
        np.asarray((0, 1, 4, 5, 2, 3, 6, 7), dtype=np.int32),
        np.asarray((0.25,) * 8),
        method="nested-volume-restriction",
        provenance="analytic-2x-refinement",
    )
    return phx.discretization.UnstructuredAMRHierarchyPlan(
        coarse,
        fine,
        prolongation,
        restriction,
        maximum_refined_cells=1,
    )


def test_unstructured_amr_selection_transfer_and_composite_integral():
    hierarchy = _hierarchy()
    selection = eqx.filter_jit(hierarchy.select)(
        jnp.asarray((2.0, 1.0)), jnp.asarray(0.0)
    )
    np.testing.assert_array_equal(selection.coarse_refined, (True, False))
    np.testing.assert_array_equal(
        selection.fine_active, (True, True, False, False, True, True, False, False)
    )
    assert selection.selected_count == 1
    assert selection.eligible_count == 2
    assert selection.capacity_overflow

    coarse = jnp.asarray(((1.0, 2.0), (3.0, 4.0)))
    fine = eqx.filter_jit(hierarchy.prolong)(coarse)
    np.testing.assert_allclose(
        fine,
        ((1.0, 2.0), (1.0, 2.0), (3.0, 4.0), (3.0, 4.0)) * 2,
    )
    np.testing.assert_allclose(hierarchy.restrict(fine), coarse)
    np.testing.assert_allclose(hierarchy.synchronize(coarse, fine, selection), coarse)
    coarse_integral = jnp.sum(hierarchy.coarse.cell_volumes[:, None] * coarse, axis=0)
    np.testing.assert_allclose(
        hierarchy.composite_integral(coarse, fine, selection), coarse_integral
    )


def test_unstructured_amr_preserves_bounded_volume_fraction_and_reflux_budget():
    hierarchy = _hierarchy()
    alpha = jnp.asarray((1.0, 0.2))
    fine_alpha = hierarchy.prolong(alpha)
    assert jnp.all((fine_alpha >= 0.0) & (fine_alpha <= 1.0))
    np.testing.assert_allclose(hierarchy.restrict(fine_alpha), alpha)
    coarse_phase_volume = jnp.sum(hierarchy.coarse.cell_volumes * alpha)
    fine_phase_volume = jnp.sum(hierarchy.fine.cell_volumes * fine_alpha)
    np.testing.assert_allclose(fine_phase_volume, coarse_phase_volume)

    coarse_state = jnp.asarray(((1.0, 2.0), (3.0, 4.0)))
    register = phx.discretization.UnstructuredAMRFluxRegister(
        jnp.asarray(((0.1, -0.2), (-0.1, 0.2)))
    )
    refluxed = hierarchy.reflux(coarse_state, register)
    old_integral = jnp.sum(hierarchy.coarse.cell_volumes[:, None] * coarse_state, axis=0)
    new_integral = jnp.sum(hierarchy.coarse.cell_volumes[:, None] * refluxed, axis=0)
    np.testing.assert_allclose(new_integral, old_integral)


def test_unstructured_amr_ties_are_deterministic_at_fixed_capacity():
    hierarchy = _hierarchy()
    selection = hierarchy.select(jnp.asarray((1.0, 1.0)), jnp.asarray(0.5))
    np.testing.assert_array_equal(selection.coarse_refined, (True, False))

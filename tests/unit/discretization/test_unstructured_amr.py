#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#


from typing import Any

import equinox as eqx
import jax.numpy as jnp
import numpy as np
import pytest

import phydrax as phx


def _quad_grid(nx: Any, ny: Any) -> Any:
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


def _hierarchy() -> Any:
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


def test_remap_transition_accepts_content_within_the_certified_coverage_bound() -> None:
    # Clipped intersections carry a 1e-12 relative roundoff excess, far inside the
    # plan's 1e-10 coverage certificate. Independent ledger: the fine field of
    # 1000 on total area 2 has content 2000; the coarse image carries the
    # covered measure 8 * 0.25 * (1 + 1e-12) times 1000, a 2e-9 excess.
    coarse, fine = _quad_grid(2, 1), _quad_grid(4, 2)
    excess = 1e-12
    plan = phx.discretization.UnstructuredConservativeRemapPlan(
        fine,
        coarse,
        np.asarray((0, 4, 8), dtype=np.int32),
        np.asarray((0, 1, 4, 5, 2, 3, 6, 7), dtype=np.int32),
        np.full((8,), 0.25 * (1.0 + excess)),
        method="clipped",
        provenance="geometric-intersection",
        tolerance=1e-10,
    )
    epoch = phx.discretization.TopologyEpoch
    source = epoch(0, plan.source_geometry_id, plan.source_topology_id, "p0")
    target = epoch(1, plan.target_geometry_id, plan.target_topology_id, "p0")
    transition = plan.epoch_transition(fine.cell_space, coarse.cell_space, source, target)
    values = jnp.full(fine.cell_space.vector_space.shape, 1000.0, dtype=jnp.float64)
    result = transition.apply(values)

    np.testing.assert_allclose(result.source_content, 2000.0, rtol=1e-14)
    np.testing.assert_allclose(result.conservation_residual, 2000.0 * excess, rtol=1e-3)
    assert bool(result.successful)
    # The admitted defect is the certified 1e-10 relative coverage, not more.
    assert float(result.content_tolerance) <= 1.01e-10 * 2000.0
    lc = phx.lifecycle
    staged = lc.CompositionEntry(
        result.values,
        entry_id="fluid/U",
        role="physical-state",
        owner_id="fluid",
        structure_id=target.epoch_id,
        revision_id="coarse",
        semantics_id="fluid:U",
    )
    held = lc.CompositionEntry(
        values,
        entry_id="fluid/U",
        role="physical-state",
        owner_id="fluid",
        structure_id=source.epoch_id,
        revision_id="fine",
        semantics_id="fluid:U",
    )
    receipt = lc.commit_composition_rebind(
        lc.CompositionRebind(
            lc.Composition((held,), boundary_id="accepted-window-1"),
            transports=(transition.composition_transport(held, staged),),
        ),
        accepted_boundary=True,
    )
    assert receipt.published and receipt.transport_accepted == (True,)


def test_unstructured_amr_contracts() -> None:
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
    with pytest.raises(Exception, match="must be finite"):
        phx.discretization.UnstructuredAMRFluxRegister(
            jnp.asarray(((jnp.nan, 0.0), (0.0, 0.0)))
        )

    with pytest.raises(ValueError, match="nonnegative int32"):
        phx.discretization.UnstructuredAMRFluxRegister(
            jnp.zeros((2, 2)),
            accepted_steps=np.asarray((-1,), dtype=np.int32),
        )
    hierarchy = _hierarchy()
    selection = hierarchy.select(jnp.asarray((1.0, 1.0)), jnp.asarray(0.5))
    np.testing.assert_array_equal(selection.coarse_refined, (True, False))

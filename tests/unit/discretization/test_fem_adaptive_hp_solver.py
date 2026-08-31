#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import jax.numpy as jnp
import numpy as np

from phydrax.discretization import CellBlock, CellMesh
from phydrax.discretization.fem import (
    coarsen_tensor_hp_cells,
    finite_element_hp_condensation_plan,
    finite_element_hp_transfer_plan,
    FiniteElementHPMultigridPlan,
    FiniteElementHPSkeletonPlan,
    FiniteElementHPSolverRefreshPlan,
    initial_finite_element_hp_topology,
    prepare_finite_element_hp_epoch,
    refine_tensor_hp_cells,
)


def _mesh():
    return CellMesh(
        jnp.asarray(((0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0))),
        (
            CellBlock(
                "quad",
                "quadrilateral",
                jnp.asarray(((0, 1, 2, 3),), dtype=jnp.int32),
                global_ids=jnp.asarray((10,), dtype=jnp.int64),
            ),
        ),
    )


def test_hp_condensation_skeleton_and_back_substitution_match_full_solve():
    topology, geometry = initial_finite_element_hp_topology(_mesh(), 3, 8)
    epoch = prepare_finite_element_hp_epoch(topology, geometry, "u")
    condensation = finite_element_hp_condensation_plan(epoch, "u")
    skeleton = FiniteElementHPSkeletonPlan(epoch, condensation)
    degree = condensation.bucket_degrees[0]
    local_size = condensation.eliminations[0].local_size
    matrix = jnp.asarray(
        np.eye(local_size)[None, :, :] * 3.0 + np.ones((1, local_size, local_size)) * 0.05
    )
    right_hand_side = jnp.linspace(0.2, 1.1, local_size)[None, :]
    condensed = condensation.condense(degree, matrix, right_hand_side)
    retained = jnp.asarray(
        np.linalg.solve(
            np.asarray(condensed.schur[0]),
            np.asarray(condensed.right_hand_side[0]),
        )
    )[None, :]
    reconstructed = condensation.reconstruct(degree, retained, condensed)
    expected = np.linalg.solve(np.asarray(matrix[0]), np.asarray(right_hand_side[0]))

    np.testing.assert_allclose(
        np.asarray(reconstructed[0]), np.asarray(expected), atol=2.0e-12
    )
    assert skeleton.epoch_id == epoch.epoch_id
    assert skeleton.retained_dofs_by_degree


def test_hp_multigrid_and_solver_refresh_reuse_degree_signatures():
    topology, geometry = initial_finite_element_hp_topology(_mesh(), 2, 12)
    fine = refine_tensor_hp_cells(topology, geometry, jnp.asarray((10,), dtype=jnp.int64))
    fine_epoch = prepare_finite_element_hp_epoch(
        fine.topology,
        fine.geometry,
        "u",
        conformity="L2",
    )
    coarse = coarsen_tensor_hp_cells(
        fine.topology,
        fine.geometry,
        jnp.asarray((10,), dtype=jnp.int64),
    )
    coarse_epoch = prepare_finite_element_hp_epoch(
        coarse.topology,
        coarse.geometry,
        "u",
        conformity="L2",
    )
    transfer = finite_element_hp_transfer_plan(
        fine_epoch,
        coarse_epoch,
        coarse.lineage,
        "u",
        "h-coarsening",
    )
    hierarchy = FiniteElementHPMultigridPlan((fine_epoch, coarse_epoch), (transfer,))
    fine_values = jnp.zeros((topology.capacity, transfer.primal.shape[2]))
    for slot, count in zip(
        np.asarray(transfer.source_slots),
        np.asarray(transfer.source_dof_count),
        strict=True,
    ):
        fine_values = fine_values.at[slot, :count].set(1.0)
    coarse_values = hierarchy.restrict(0, fine_values)
    parent = int(np.asarray(transfer.target_slots)[0])
    parent_count = int(np.asarray(transfer.target_dof_count)[0])
    np.testing.assert_allclose(
        np.asarray(coarse_values[parent, :parent_count]),
        1.0,
        atol=2.0e-12,
    )

    refresh = FiniteElementHPSolverRefreshPlan(coarse_epoch, fine_epoch)
    assert refresh.routes_changed
    assert refresh.metrics_changed
    assert refresh.skeleton_changed
    assert refresh.reused_signatures

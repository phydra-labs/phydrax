#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import jax.numpy as jnp
import numpy as np

from phydrax.discretization import CellBlock, CellMesh
from phydrax.discretization.fem import (
    FiniteElementHPPartitionPlan,
    inherit_finite_element_hp_ownership,
    initial_finite_element_hp_topology,
    prepare_finite_element_hp_epoch,
    refine_tensor_hp_cells,
)


def _mesh():
    return CellMesh(
        jnp.asarray(
            (
                (0.0, 0.0),
                (1.0, 0.0),
                (2.0, 0.0),
                (0.0, 1.0),
                (1.0, 1.0),
                (2.0, 1.0),
            )
        ),
        (
            CellBlock(
                "quads",
                "quadrilateral",
                jnp.asarray(((0, 1, 4, 3), (1, 2, 5, 4)), dtype=jnp.int32),
                global_ids=jnp.asarray((10, 20), dtype=jnp.int64),
            ),
        ),
    )


def test_children_inherit_owners_and_adaptive_halos_include_mortar_neighbours():
    topology, geometry = initial_finite_element_hp_topology(_mesh(), 2, 16)
    source_epoch = prepare_finite_element_hp_epoch(
        topology,
        geometry,
        "u",
        conformity="L2",
    )
    source_owners = np.full((topology.capacity,), -1, dtype=np.int32)
    source_owners[:2] = (0, 1)
    source_partition = FiniteElementHPPartitionPlan(source_epoch, source_owners, 2)

    refined = refine_tensor_hp_cells(topology, geometry, jnp.asarray((10,)))
    target_epoch = prepare_finite_element_hp_epoch(
        refined.topology,
        refined.geometry,
        "u",
        conformity="L2",
    )
    target_partition = inherit_finite_element_hp_ownership(
        source_partition,
        target_epoch,
        refined.lineage,
    )
    active = np.asarray(refined.topology.active)
    owners = np.asarray(target_partition.cell_owner_by_slot)
    assert owners[1] == 1
    np.testing.assert_array_equal(owners[2:6], np.zeros((4,), dtype=np.int32))
    assert np.all(owners[~active] == -1)
    assert bool(np.asarray(target_partition.mortar_dependencies)[0, 1])
    assert bool(np.asarray(target_partition.mortar_dependencies)[1, 0])

    owned = np.asarray(target_partition.worksets.owned_cells)
    owned_valid = np.asarray(target_partition.worksets.owned_valid)
    owned_cells = np.sort(owned[owned_valid])
    np.testing.assert_array_equal(
        owned_cells,
        np.arange(target_epoch.active_cell_slots.size),
    )
    halo_valid = np.asarray(target_partition.worksets.halo_valid)
    assert np.count_nonzero(halo_valid[0]) == 1
    assert np.count_nonzero(halo_valid[1]) == 2

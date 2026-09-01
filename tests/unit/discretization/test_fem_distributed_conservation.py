#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import jax.numpy as jnp
import numpy as np

from phydrax.discretization._cell_mesh import CellMesh
from phydrax.discretization.fem._distributed import (
    lower_distributed_finite_element_phases,
    partition_cells_cost_aware,
)
from phydrax.discretization.fem._generic import FiniteElementFieldSpec, FiniteElementPlan
from phydrax.discretization.fem._reference import discontinuous_element


def test_cost_partition_and_phases_are_exactly_once_and_conservative():
    mesh = CellMesh.from_triangles(
        np.asarray(
            (
                (0.0, 0.0),
                (1.0, 0.0),
                (1.0, 1.0),
                (0.0, 1.0),
            )
        ),
        np.asarray(((0, 1, 3), (1, 2, 3)), dtype=np.int32),
    )
    discretization = FiniteElementPlan(
        mesh,
        FiniteElementFieldSpec("u", discontinuous_element("triangle", 2)),
    ).prepare()
    cost_partition = partition_cells_cost_aware(discretization, 2)
    phases = lower_distributed_finite_element_phases(discretization, cost_partition)

    assert cost_partition.evidence.imbalance_ratio >= 1.0
    assert phases.phase_names == (
        "owned-local",
        "halo-update",
        "interface",
        "contribution-sum",
    )
    cell_values = jnp.asarray(((1.5, -0.25), (2.0, 0.75)))
    partition_total = sum(
        (
            phases.local_contribution(part, cell_values)
            for part in range(phases.partition.part_count)
        ),
        start=jnp.zeros((2,)),
    )
    np.testing.assert_allclose(partition_total, jnp.sum(cell_values, axis=0))

    masks = jnp.stack(
        tuple(phases.interface_mask(part) for part in range(phases.partition.part_count))
    )
    np.testing.assert_array_equal(jnp.sum(masks, axis=0), 1)
    flux = jnp.asarray(((0.4, -0.2),))
    serial = phases.facet_ownership.route_equal_opposite(flux)
    partitioned = sum(
        (
            phases.facet_ownership.route_partition(part, flux)
            for part in range(phases.partition.part_count)
        ),
        start=jnp.zeros_like(serial),
    )
    np.testing.assert_allclose(partitioned, serial)
    np.testing.assert_allclose(jnp.sum(partitioned, axis=0), 0.0)

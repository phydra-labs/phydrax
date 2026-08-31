#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import jax.numpy as jnp
import numpy as np

from phydrax.discretization import CellBlock, CellMesh
from phydrax.discretization.fem import (
    finite_element_hp_decision,
    FiniteElementHPErrorEstimate,
    initial_finite_element_hp_topology,
    prepare_finite_element_hp_epoch,
    refine_tensor_hp_cells,
    tensor_modal_decay_estimate,
)


def _two_cell_mesh():
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


def test_smooth_modal_tail_selects_p_and_reproduces_tensor_polynomial():
    nodes = np.linspace(-1.0, 1.0, 5)
    x, y = np.meshgrid(nodes, nodes, indexing="ij")
    values = 1.0 + x + y + x * y + x**2
    decay = tensor_modal_decay_estimate(
        values,
        (4, 4),
        nodes_by_axis=(nodes, nodes),
    )
    np.testing.assert_array_less(np.asarray(decay), 1.0e-22)

    topology, geometry = initial_finite_element_hp_topology(_two_cell_mesh(), 2, 24)
    smoothness = np.ones((topology.capacity, topology.dimension))
    smoothness[0] = np.asarray(decay)
    estimate = FiniteElementHPErrorEstimate(
        topology,
        jnp.asarray((2.0, 0.1) + (0.0,) * 22),
        smoothness=smoothness,
    )
    decision = finite_element_hp_decision(topology, estimate, maximum_degree=6)
    assert tuple(np.asarray(decision.target_degrees)[0]) == (3, 2)
    assert not bool(np.asarray(decision.refine)[0])
    epoch = prepare_finite_element_hp_epoch(topology, geometry, "u")
    assert epoch.discretization.dof_maps[0].global_dof_count > 0


def test_mixed_regular_and_singular_cells_choose_p_and_h_in_one_candidate():
    topology, geometry = initial_finite_element_hp_topology(_two_cell_mesh(), 2, 24)
    smoothness = np.zeros((topology.capacity, topology.dimension))
    smoothness[0] = (1.0e-9, 1.0e-3)
    smoothness[1] = (1.0, 1.0)
    estimate = FiniteElementHPErrorEstimate(
        topology,
        jnp.asarray((2.0, 1.5) + (0.0,) * 22),
        smoothness=smoothness,
    )
    decision = finite_element_hp_decision(
        topology,
        estimate,
        refine_fraction=0.5,
        maximum_active_cells=8,
    )
    assert tuple(np.asarray(decision.target_degrees)[0]) == (3, 2)
    assert bool(np.asarray(decision.refine)[1])

    marked = np.asarray(topology.cell_global_ids)[np.asarray(decision.refine)]
    refined = refine_tensor_hp_cells(
        topology,
        geometry,
        marked,
        target_degrees=np.asarray((decision.target_degrees[1],)),
    )
    assert refined.topology.active_count == 5
    candidate = prepare_finite_element_hp_epoch(
        refined.topology,
        refined.geometry,
        "u",
    )
    assert candidate.constraints

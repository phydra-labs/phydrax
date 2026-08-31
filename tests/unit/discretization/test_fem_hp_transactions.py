#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import equinox as eqx
import jax.numpy as jnp
import numpy as np

from phydrax.discretization.fem._hp import (
    finite_element_hp_workset_plan,
    FiniteElementHPLineage,
    FiniteElementHPTopology,
    FiniteElementHPTransferPlan,
)


def _topology(degrees):
    active = np.asarray((True, True, True, True, False, False))
    return FiniteElementHPTopology(
        "quadrilateral",
        "quad-mesh",
        np.asarray((30, 10, 20, 40, -1, -1), dtype=np.int64),
        active,
        active,
        np.asarray(degrees, dtype=np.int32),
    )


def test_hp_degree_buckets_are_fixed_capacity_and_deterministic():
    topology = _topology(((2, 1), (1, 1), (2, 1), (3, 2), (0, 0), (0, 0)))
    first = finite_element_hp_workset_plan(topology)
    second = finite_element_hp_workset_plan(topology)

    assert first.plan_id == second.plan_id
    assert first.cell_slots.shape == (topology.capacity, topology.capacity)
    np.testing.assert_array_equal(
        np.asarray(first.bucket_degrees[:3]),
        np.asarray(((1, 1), (2, 1), (3, 2)), dtype=np.int32),
    )
    np.testing.assert_array_equal(np.asarray(first.cell_slots[0, :1]), (1,))
    np.testing.assert_array_equal(np.asarray(first.cell_slots[1, :2]), (2, 0))
    np.testing.assert_array_equal(np.asarray(first.cell_bucket), (1, 0, 1, 2, -1, -1))

    gathered = eqx.filter_jit(first.gather)(jnp.arange(topology.capacity, dtype=float))
    np.testing.assert_allclose(np.asarray(gathered[1, :2]), (2.0, 0.0))
    np.testing.assert_allclose(np.asarray(gathered[3:]), 0.0)

    hex_topology = FiniteElementHPTopology(
        "hexahedron",
        "hex-mesh",
        np.asarray((90, -1), dtype=np.int64),
        np.asarray((True, False)),
        np.asarray((True, False)),
        np.asarray(((2, 3, 4), (0, 0, 0)), dtype=np.int32),
    )
    assert finite_element_hp_workset_plan(hex_topology).dimension == 3


def test_p_transfer_roles_remain_distinct():
    accepted = _topology(((2, 1), (1, 1), (2, 1), (3, 2), (0, 0), (0, 0)))
    candidate = _topology(((3, 2), (1, 1), (2, 2), (3, 2), (0, 0), (0, 0)))
    primal = np.asarray((((1.0, 0.0), (0.0, 1.0), (0.5, 0.5)),))
    pairing_adjoint = np.asarray((((1.0, 0.0, 0.25), (0.0, 1.0, 0.75)),))
    mass_projection = np.asarray((((1.0, 0.0), (0.0, 1.0), (0.25, 0.75)),))
    transfer = FiniteElementHPTransferPlan(
        "quad-mesh",
        "quad-mesh",
        "p",
        6,
        6,
        np.asarray((1,), dtype=np.int32),
        np.asarray((1,), dtype=np.int32),
        np.asarray((2,), dtype=np.int32),
        np.asarray((3,), dtype=np.int32),
        primal,
        source_plan_id=accepted.plan_id,
        target_plan_id=candidate.plan_id,
        pairing_adjoint=pairing_adjoint,
        mass_projection=mass_projection,
    )
    source = jnp.zeros((6, 2)).at[1].set(jnp.asarray((2.0, 4.0)))
    primal_value = eqx.filter_jit(transfer.apply_primal)(source)
    projected_value = eqx.filter_jit(transfer.apply_mass_projection)(source)

    np.testing.assert_allclose(np.asarray(primal_value[1]), (2.0, 4.0, 3.0))
    np.testing.assert_allclose(np.asarray(projected_value[1]), (2.0, 4.0, 3.5))
    target_dual = jnp.zeros((6, 3)).at[1].set(jnp.asarray((1.0, 2.0, 3.0)))
    np.testing.assert_allclose(
        np.asarray(transfer.pullback_raw(target_dual)[1]),
        primal[0].T @ np.asarray(target_dual[1]),
    )
    np.testing.assert_allclose(
        np.asarray(transfer.apply_pairing_adjoint(target_dual)[1]),
        pairing_adjoint[0] @ np.asarray(target_dual[1]),
    )


def test_refinement_and_coarsening_lineage_have_fixed_quad_child_capacity():
    coarse = FiniteElementHPTopology(
        "quadrilateral",
        "coarse-quad-mesh",
        np.asarray((100, 200, -1, -1, -1, -1, -1, -1)),
        np.asarray((True, True, False, False, False, False, False, False)),
        np.asarray((True, True, False, False, False, False, False, False)),
        np.asarray(((2, 2), (2, 2)) + ((0, 0),) * 6),
    )
    refined = FiniteElementHPTopology(
        "quadrilateral",
        "refined-quad-mesh",
        np.asarray((100, 200, 1000, 1001, 1002, 1003, -1, -1)),
        np.asarray((True, True, True, True, True, True, False, False)),
        np.asarray((False, True, True, True, True, True, False, False)),
        np.asarray(((0, 0), (2, 2), (2, 2), (2, 2), (2, 2), (2, 2), (0, 0), (0, 0))),
        root_cell_ids=np.asarray((100, 200, 100, 100, 100, 100, -1, -1)),
        path_codes=np.asarray((1, 1, 5, 6, 7, 8, -1, -1)),
        levels=np.asarray((0, 0, 1, 1, 1, 1, -1, -1)),
        parent_slots=np.asarray((-1, -1, 0, 0, 0, 0, -1, -1)),
        child_slots=np.asarray(((2, 3, 4, 5),) + ((-1, -1, -1, -1),) * 7),
        child_valid=np.asarray(
            ((True, True, True, True),) + ((False, False, False, False),) * 7
        ),
    )
    refinement = FiniteElementHPLineage(
        coarse.topology_id,
        refined.topology_id,
        8,
        8,
        np.asarray((0, 0, 0, 0, 1)),
        np.asarray((2, 3, 4, 5, 1)),
        ("refinement", "refinement", "refinement", "refinement", "unchanged"),
    )
    coarsening = FiniteElementHPLineage(
        refined.topology_id,
        coarse.topology_id,
        8,
        8,
        np.asarray((2, 3, 4, 5, 1)),
        np.asarray((0, 0, 0, 0, 1)),
        ("coarsening", "coarsening", "coarsening", "coarsening", "unchanged"),
    )

    assert np.count_nonzero(np.asarray(refinement.relation_mask("refinement"))) == 4
    assert np.count_nonzero(np.asarray(coarsening.relation_mask("coarsening"))) == 4
    assert refined.child_capacity == 4

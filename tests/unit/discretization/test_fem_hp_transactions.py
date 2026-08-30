#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from phydrax.discretization.fem._hp import (
    finite_element_hp_workset_plan,
    FiniteElementHPAcceptedPlan,
    FiniteElementHPLineage,
    FiniteElementHPTopology,
    FiniteElementHPTransaction,
    FiniteElementHPTransferPlan,
)


def _accepted_topology():
    return FiniteElementHPTopology(
        "quadrilateral",
        "quad-mesh",
        np.asarray([30, 10, 20, 40, -1, -1], dtype=np.int64),
        np.asarray([True, True, True, True, False, False]),
        np.asarray(
            [[2, 1], [1, 1], [2, 1], [3, 2], [0, 0], [0, 0]],
            dtype=np.int32,
        ),
    )


def _p_candidate_topology():
    return FiniteElementHPTopology(
        "quadrilateral",
        "quad-mesh",
        np.asarray([30, 10, 20, 40, -1, -1], dtype=np.int64),
        np.asarray([True, True, True, True, False, False]),
        np.asarray(
            [[3, 2], [1, 1], [2, 2], [3, 2], [0, 0], [0, 0]],
            dtype=np.int32,
        ),
    )


def test_hp_degree_buckets_are_fixed_capacity_and_deterministic():
    topology = _accepted_topology()
    first = finite_element_hp_workset_plan(topology)
    second = finite_element_hp_workset_plan(topology)

    assert first.plan_id == second.plan_id
    assert first.cell_slots.shape == (topology.capacity, topology.capacity)
    assert np.array_equal(
        np.asarray(first.bucket_degrees[:3]),
        np.asarray([[1, 1], [2, 1], [3, 2]], dtype=np.int32),
    )
    assert np.array_equal(np.asarray(first.cell_slots[0, :1]), [1])
    assert np.array_equal(np.asarray(first.cell_slots[1, :2]), [2, 0])
    assert np.array_equal(np.asarray(first.cell_bucket), [1, 0, 1, 2, -1, -1])

    gathered = eqx.filter_jit(first.gather)(jnp.arange(topology.capacity, dtype=float))
    assert jnp.allclose(gathered[1, :2], jnp.asarray([2.0, 0.0]))
    assert jnp.allclose(gathered[3:], 0.0)

    hex_topology = FiniteElementHPTopology(
        "hexahedron",
        "hex-mesh",
        np.asarray([90, -1], dtype=np.int64),
        np.asarray([True, False]),
        np.asarray([[2, 3, 4], [0, 0, 0]], dtype=np.int32),
    )
    assert finite_element_hp_workset_plan(hex_topology).dimension == 3


def test_p_transfer_roles_remain_distinct_and_candidate_promotion_is_rollback_safe():
    accepted_topology = _accepted_topology()
    candidate_topology = _p_candidate_topology()
    accepted = FiniteElementHPAcceptedPlan(accepted_topology)
    candidate = FiniteElementHPAcceptedPlan(candidate_topology)
    routes = np.asarray([0, 1, 2, 3], dtype=np.int32)
    lineage = FiniteElementHPLineage(
        accepted_topology.topology_id,
        candidate_topology.topology_id,
        accepted_topology.capacity,
        candidate_topology.capacity,
        routes,
        routes,
        ("unchanged",) * 4,
    )

    primal = np.asarray([[[1.0, 0.0], [0.0, 1.0], [0.5, 0.5]]])
    pairing_adjoint = np.asarray([[[1.0, 0.0, 0.25], [0.0, 1.0, 0.75]]])
    mass_projection = np.asarray([[[1.0, 0.0], [0.0, 1.0], [0.25, 0.75]]])
    transfer = FiniteElementHPTransferPlan(
        "quad-mesh",
        "quad-mesh",
        "p",
        6,
        6,
        np.asarray([1], dtype=np.int32),
        np.asarray([1], dtype=np.int32),
        np.asarray([2], dtype=np.int32),
        np.asarray([3], dtype=np.int32),
        primal,
        source_plan_id=accepted_topology.plan_id,
        target_plan_id=candidate_topology.plan_id,
        pairing_adjoint=pairing_adjoint,
        mass_projection=mass_projection,
    )
    transaction = FiniteElementHPTransaction(
        accepted,
        candidate,
        lineage,
        p_transfers=(transfer,),
    )

    source = jnp.zeros((6, 2)).at[1].set(jnp.asarray([2.0, 4.0]))
    primal_value = eqx.filter_jit(transfer.apply_primal)(source)
    projected_value = eqx.filter_jit(transfer.apply_mass_projection)(source)
    assert jnp.allclose(primal_value[1], jnp.asarray([2.0, 4.0, 3.0]))
    assert jnp.allclose(projected_value[1], jnp.asarray([2.0, 4.0, 3.5]))

    target_dual = jnp.zeros((6, 3)).at[1].set(jnp.asarray([1.0, 2.0, 3.0]))
    assert jnp.allclose(
        transfer.pullback_raw(target_dual)[1],
        primal[0].T @ np.asarray(target_dual[1]),
    )
    assert jnp.allclose(
        transfer.apply_pairing_adjoint(target_dual)[1],
        pairing_adjoint[0] @ np.asarray(target_dual[1]),
    )
    assert transaction.rollback().accepted_id == accepted.accepted_id
    assert transaction.promote(False).accepted_id == accepted.accepted_id
    assert transaction.promote(True).accepted_id == candidate.accepted_id
    assert accepted.topology.plan_id != candidate.topology.plan_id

    with pytest.raises(TypeError, match="explicit host decision"):
        jax.jit(lambda decision: transaction.promote(decision))(jnp.asarray(True))


def test_refinement_and_coarsening_lineage_have_fixed_quad_child_capacity():
    coarse = FiniteElementHPTopology(
        "quadrilateral",
        "coarse-quad-mesh",
        np.asarray([100, 200, -1, -1, -1, -1, -1, -1]),
        np.asarray([True, True, False, False, False, False, False, False]),
        np.asarray(
            [
                [2, 2],
                [2, 2],
                [0, 0],
                [0, 0],
                [0, 0],
                [0, 0],
                [0, 0],
                [0, 0],
            ]
        ),
    )
    refined = FiniteElementHPTopology(
        "quadrilateral",
        "refined-quad-mesh",
        np.asarray([-1, 200, 1000, 1001, 1002, 1003, -1, -1]),
        np.asarray([False, True, True, True, True, True, False, False]),
        np.asarray(
            [
                [0, 0],
                [2, 2],
                [2, 2],
                [2, 2],
                [2, 2],
                [2, 2],
                [0, 0],
                [0, 0],
            ]
        ),
    )
    refinement = FiniteElementHPLineage(
        coarse.topology_id,
        refined.topology_id,
        8,
        8,
        np.asarray([0, 0, 0, 0, 1]),
        np.asarray([2, 3, 4, 5, 1]),
        ("refinement", "refinement", "refinement", "refinement", "unchanged"),
    )
    coarsening = FiniteElementHPLineage(
        refined.topology_id,
        coarse.topology_id,
        8,
        8,
        np.asarray([2, 3, 4, 5, 1]),
        np.asarray([0, 0, 0, 0, 1]),
        ("coarsening", "coarsening", "coarsening", "coarsening", "unchanged"),
    )
    h_transfer = FiniteElementHPTransferPlan(
        coarse.topology_id,
        refined.topology_id,
        "h-refinement",
        8,
        8,
        np.asarray([0, 0, 0, 0, 1]),
        np.asarray([2, 3, 4, 5, 1]),
        np.ones((5,), dtype=np.int32),
        np.ones((5,), dtype=np.int32),
        np.ones((5, 1, 1)),
        source_plan_id=coarse.plan_id,
        target_plan_id=refined.plan_id,
    )
    transaction = FiniteElementHPTransaction(
        FiniteElementHPAcceptedPlan(coarse),
        FiniteElementHPAcceptedPlan(refined),
        refinement,
        h_transfers=(h_transfer,),
    )

    coarse_values = jnp.zeros((8, 1)).at[0, 0].set(3.0).at[1, 0].set(7.0)
    refined_values = eqx.filter_jit(h_transfer.apply_primal)(coarse_values)
    assert jnp.allclose(refined_values[jnp.asarray([2, 3, 4, 5]), 0], 3.0)
    assert jnp.allclose(refined_values[1, 0], 7.0)
    assert np.count_nonzero(np.asarray(refinement.relation_mask("refinement"))) == 4
    assert np.count_nonzero(np.asarray(coarsening.relation_mask("coarsening"))) == 4
    assert transaction.promote(True).topology.topology_id == "refined-quad-mesh"
    assert transaction.rollback().topology.topology_id == "coarse-quad-mesh"

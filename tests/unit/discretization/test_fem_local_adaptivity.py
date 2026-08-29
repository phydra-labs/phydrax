#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import jax.numpy as jnp

import phydrax as phx


def _mesh():
    vertices = jnp.asarray([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0], [0.5, 0.5]])
    cells = jnp.asarray([[0, 1, 4], [1, 2, 4], [2, 3, 4], [3, 0, 4]], dtype=jnp.int32)
    return phx.discretization.CellMesh.from_triangles(
        vertices,
        cells,
        vertex_global_ids=jnp.asarray([1, 2, 3, 4, 5]),
        cell_global_ids=jnp.asarray([10, 20, 30, 40]),
    )


def test_dorfler_local_refine_transfer_and_sibling_coarsen():
    mesh = _mesh()
    marked = phx.discretization.dorfler_mark(
        jnp.asarray([4.0, 1.0, 1.0, 1.0]),
        0.5,
        cell_global_ids=mesh.blocks[0].global_ids,
    )
    refined, adaptation, transfer = phx.discretization.refine_triangles_local(
        mesh, marked
    )
    constant = transfer.primal @ jnp.ones((5,))
    linear = transfer.primal @ mesh.coordinates[:, 0]
    children = adaptation.child_cell_ids[0][adaptation.child_valid[0]]
    restored = phx.discretization.coarsen_triangles_local(refined, adaptation, children)

    assert jnp.array_equal(marked, jnp.asarray([10]))
    assert refined.blocks[0].cell_count == 5
    assert jnp.allclose(constant, 1.0)
    assert jnp.allclose(linear, refined.coordinates[:, 0])
    assert restored.topology_id == mesh.topology_id


def test_transfer_dual_pairing_and_local_dwr_indicators():
    mesh = _mesh()
    refined, _adaptation, transfer = phx.discretization.refine_triangles_local(
        mesh, jnp.asarray([10])
    )
    primal = jnp.arange(5.0)
    target_dual = jnp.arange(float(refined.coordinates.shape[0]))
    left = jnp.vdot(transfer.primal @ primal, target_dual)
    right = jnp.vdot(primal, transfer.dual_pullback @ target_dual)
    dwr = phx.discretization.local_dual_weighted_residual(
        jnp.asarray([[1.0, -2.0], [3.0, 4.0]]),
        jnp.asarray([[0.5, 1.0], [2.0, -1.0]]),
    )

    assert jnp.allclose(left, right)
    assert jnp.allclose(dwr.signed, jnp.asarray([-1.5, 2.0]))
    assert jnp.allclose(dwr.absolute, jnp.asarray([1.5, 2.0]))


def test_rejected_topology_candidate_preserves_accepted_state_bitwise():
    mesh = _mesh()
    accepted = phx.solver.FiniteElementAcceptedState(
        (mesh.coordinates[:, 0],),
        0.0,
        0,
        mesh.topology_id,
        "prepared",
        "compiled",
    )
    transaction = phx.solver.FiniteElementTopologyTransaction(
        lambda candidate_mesh, fields, materials, adaptation, args: False
    )
    result = transaction.execute(accepted, mesh, jnp.asarray([10]))

    assert not bool(result.committed)
    assert result.state.accepted_id == accepted.accepted_id
    assert result.mesh.topology_id == mesh.topology_id
    assert jnp.array_equal(result.state.fields[0], accepted.fields[0])

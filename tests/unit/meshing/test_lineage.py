import jax.numpy as jnp
import numpy as np

import phydrax as phx


def _mesh():
    return phx.discretization.CellMesh.from_triangles(
        np.asarray(((0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0), (0.5, 0.5))),
        np.asarray(((0, 1, 4), (1, 2, 4), (2, 3, 4), (3, 0, 4)), dtype=np.int32),
        vertex_global_ids=np.asarray((1, 2, 3, 4, 5), dtype=np.int64),
        cell_global_ids=np.asarray((10, 20, 30, 40), dtype=np.int64),
    )


def test_triangle_transition_lineage_stencil_and_atomic_commit():
    mesh = _mesh()
    transition, transfer = phx.meshing.refine_triangle_mesh(
        mesh,
        np.asarray((10,), dtype=np.int64),
        phx.SpatialCoordinateContract.si(),
    )
    stencil = transition.vertex_stencil
    assert stencil is not None
    constant = stencil.apply(mesh.vertex_global_ids, jnp.ones((5,)))
    linear = stencil.apply(mesh.vertex_global_ids, mesh.coordinates[:, 0])

    assert jnp.allclose(constant, 1.0)
    assert jnp.allclose(linear, transition.target.mesh.coordinates[:, 0])
    assert transition.lineage.source_topology_id == mesh.topology_id
    assert transition.lineage.target_topology_id == transition.target.mesh.topology_id

    accepted = phx.solver.FiniteElementAcceptedState(
        (mesh.coordinates[:, 0],),
        0.0,
        0,
        mesh.topology_id,
        "prepared",
        "compiled",
    )
    transaction = phx.solver.FiniteElementTopologyTransaction(
        lambda candidate, fields, materials, lineage, args: (
            candidate.topology_id == lineage.target_topology_id
            and bool(jnp.all(jnp.isfinite(fields[0])))
        )
    )
    result = transaction.execute(accepted, mesh, transition, transfer)

    assert bool(result.committed)
    assert result.transition is transition
    assert result.mesh.topology_id == transition.target.mesh.topology_id
    assert jnp.allclose(result.state.fields[0], transition.target.mesh.coordinates[:, 0])

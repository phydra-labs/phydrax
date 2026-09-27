from typing import Any

import jax.numpy as jnp
import numpy as np

import phydrax as phx


def _source() -> Any:
    return phx.meshing.certify_cell_mesh(
        phx.discretization.CellMesh.from_triangles(
            np.asarray(((0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0), (0.5, 0.5))),
            np.asarray(((0, 1, 4), (1, 2, 4), (2, 3, 4), (3, 0, 4)), dtype=np.int32),
            vertex_global_ids=np.asarray((1, 2, 3, 4, 5), dtype=np.int64),
            cell_global_ids=np.asarray((10, 20, 30, 40), dtype=np.int64),
        ),
        phx.SpatialCoordinateContract.si(),
    )


def test_triangle_transition_lineage_stencil_and_atomic_commit() -> None:
    source = _source()
    mesh = source.mesh
    adaptation = phx.meshing.execute_mesh_adaptation(
        phx.meshing.prepare_mesh_adaptation(
            source,
            phx.meshing.MarkedMeshAdaptation(np.asarray((10,), dtype=np.int64)),
            policy=phx.meshing.MeshAdaptationPolicy(
                phx.meshing.MeshAdaptationRoute.NATIVE_BISECTION
            ),
        )
    )
    transition = adaptation.transition
    # ty: ignore[unresolved-attribute]
    stencil = transition.vertex_stencil
    assert stencil is not None
    constant = stencil.apply(mesh.vertex_global_ids, jnp.ones((5,)))
    linear = stencil.apply(mesh.vertex_global_ids, mesh.coordinates[:, 0])
    # ty: ignore[unresolved-attribute]
    cells = transition.lineage.entity_lineage(2)
    refined = np.asarray(cells.relation_kinds) == int(
        phx.meshing.EntityLineageKind.REFINED_FROM
    )

    assert jnp.allclose(constant, 1.0)
    # ty: ignore[unresolved-attribute]
    assert jnp.allclose(linear, transition.target.mesh.coordinates[:, 0])
    # ty: ignore[unresolved-attribute]
    assert transition.lineage.source_topology_id == mesh.topology_id
    # ty: ignore[unresolved-attribute]
    assert transition.lineage.target_topology_id == transition.target.mesh.topology_id
    np.testing.assert_array_equal(np.asarray(cells.source_global_ids)[refined], (10, 10))
    np.testing.assert_array_equal(np.asarray(cells.deleted_source_ids), ())

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
    result = transaction.execute(accepted, mesh, adaptation)

    assert bool(result.committed)
    assert result.adaptation is adaptation
    # ty: ignore[unresolved-attribute]
    assert result.mesh.topology_id == transition.target.mesh.topology_id
    # ty: ignore[unresolved-attribute]
    assert jnp.allclose(result.state.fields[0], transition.target.mesh.coordinates[:, 0])

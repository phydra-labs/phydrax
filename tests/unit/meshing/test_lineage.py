from typing import Any

import jax.numpy as jnp
import numpy as np
import pytest

import phydrax as phx
from phydrax.meshing._lineage import (
    EntityLineage,
    EntityLineageKind,
    inherit_mesh_attributes,
    MeshLineage,
)


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
        ),
        fields=phx.discretization.FiniteElementFieldSpec(
            "u", phx.discretization.lagrange_element("triangle", 1)
        ),
    )
    result = transaction.execute(accepted, mesh, adaptation)

    assert bool(result.committed)
    assert result.adaptation is adaptation
    # ty: ignore[unresolved-attribute]
    assert result.mesh.topology_id == transition.target.mesh.topology_id
    # ty: ignore[unresolved-attribute]
    assert jnp.allclose(result.state.fields[0], transition.target.mesh.coordinates[:, 0])
    assert result.receipt is not None and result.receipt.published
    assert result.state.transition_id == result.receipt.receipt_id


def test_attribute_lineage_preserves_exact_values_and_rejects_conflicting_merge() -> None:
    points = np.asarray(
        ((0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)), dtype=np.float64
    )
    mesh = phx.discretization.CellMesh.from_triangles(
        points,
        np.asarray(((0, 1, 2), (0, 2, 3)), dtype=np.int32),
        cell_global_ids=np.asarray((101, 205), dtype=np.int64),
    )
    entities = mesh.entity_set(2)
    scope = phx.meshing.MeshingScope(
        mesh.mesh_id,
        mesh.numeric_version,
        phx.meshing.MeshingEntityKind.MESH,
        2,
        entities.entity_set_id,
        np.asarray(entities.entity_ids, dtype=np.int64),
    )
    marker = phx.meshing.MeshAttribute(
        "material_marker",
        phx.meshing.MeshAttributeRole.MARKER,
        scope,
        np.asarray((19, 23), dtype=np.int64),
    )
    source = phx.meshing.certify_cell_mesh(
        mesh, phx.SpatialCoordinateContract.si(), attributes=(marker,)
    )
    target = phx.discretization.CellMesh.from_triangles(
        np.concatenate((points, np.asarray(((0.5, 0.5),), dtype=np.float64))),
        np.asarray(((0, 1, 4), (1, 2, 4), (0, 4, 3), (4, 2, 3)), dtype=np.int32),
        cell_global_ids=np.asarray((701, 702, 703, 704), dtype=np.int64),
    )
    children = np.asarray(target.entity_set(2).entity_ids, dtype=np.int64)
    refined = EntityLineage(
        2,
        entities.entity_set_id,
        target.entity_set(2).entity_set_id,
        np.repeat(np.asarray(entities.entity_ids, dtype=np.int64), 2),
        children,
        np.full((4,), int(EntityLineageKind.REFINED_FROM), dtype=np.int32),
    )
    attributes = inherit_mesh_attributes(
        source, target, MeshLineage(mesh.topology_id, target.topology_id, (refined,))
    )
    np.testing.assert_array_equal(
        np.asarray(attributes[0].global_values),
        np.asarray((19, 19, 23, 23), dtype=np.int64),
    )
    merged = EntityLineage(
        2,
        entities.entity_set_id,
        target.entity_set(2).entity_set_id,
        np.asarray(entities.entity_ids, dtype=np.int64),
        np.asarray((children[0], children[0]), dtype=np.int64),
        np.full((2,), int(EntityLineageKind.MERGED_INTO), dtype=np.int32),
    )
    with pytest.raises(ValueError, match="different scientific attribute values"):
        inherit_mesh_attributes(
            source, target, MeshLineage(mesh.topology_id, target.topology_id, (merged,))
        )

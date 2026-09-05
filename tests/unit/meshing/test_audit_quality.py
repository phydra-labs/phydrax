import jax
import jax.numpy as jnp
import numpy as np
import pytest

import phydrax as phx


def _mesh(cell_kind):
    topology = phx.discretization.reference_cell_topology(cell_kind)
    coordinates = np.asarray(topology.vertices, dtype=float)
    return phx.discretization.CellMesh(
        coordinates,
        (
            phx.discretization.CellBlock(
                "cells",
                cell_kind,
                np.arange(coordinates.shape[0], dtype=np.int32)[None, :],
            ),
        ),
    )


@pytest.mark.parametrize(
    "cell_kind",
    (
        "interval",
        "triangle",
        "quadrilateral",
        "tetrahedron",
        "prism",
        "pyramid",
        "hexahedron",
    ),
)
def test_reference_cells_have_positive_native_quality(cell_kind):
    mesh = _mesh(cell_kind)
    result = phx.meshing.certify_cell_mesh(mesh, phx.SpatialCoordinateContract.si())

    assert result.audit.passed
    assert result.quality.minimum_measure > 0.0
    assert result.quality.minimum_mean_ratio > 0.0
    assert result.quality.maximum_aspect_ratio >= 1.0


def test_triangle_quality_is_fixed_topology_differentiable_and_rejects_inversion():
    mesh = _mesh("triangle")

    def objective(coordinates):
        quality = phx.meshing.evaluate_cell_quality(mesh, coordinates)
        return quality.mean_ratios[0]

    coordinates = jnp.asarray(mesh.coordinates)
    gradient = jax.grad(objective)(coordinates)
    assert gradient.shape == coordinates.shape
    assert bool(jnp.all(jnp.isfinite(gradient)))

    swap = jnp.asarray([2, 1])
    inverted = coordinates.at[jnp.asarray([1, 2])].set(coordinates[swap])
    evaluation = phx.meshing.evaluate_cell_quality(mesh, inverted)
    assert not bool(evaluation.valid[0])
    with pytest.raises(phx.meshing.MeshingFailure):
        phx.meshing.certify_cell_mesh(
            mesh.with_coordinates(inverted, numeric_version="inverted"),
            phx.SpatialCoordinateContract.si(),
        )


def _two_cells(cell_kind="triangle"):
    points = np.asarray(phx.discretization.reference_cell_topology(cell_kind).vertices)
    count = len(points)
    mesh = phx.discretization.CellMesh(
        np.concatenate((points, points + 3.0)),
        (
            phx.discretization.CellBlock(
                "cells",
                cell_kind,
                np.arange(2 * count).reshape(2, count),
                global_ids=np.asarray([20, 10]),
            ),
        ),
        vertex_global_ids=np.arange(2 * count) + 100,
    )
    return phx.discretization.CellMesh(
        mesh.coordinates,
        mesh.blocks,
        vertex_global_ids=mesh.vertex_global_ids,
        entity_global_ids={
            dimension: np.arange(mesh.entity_set(dimension).count) + 1000 * dimension
            for dimension in range(1, mesh.topological_dimension)
        },
    )


def _entity_ids_by_vertices(mesh, dimension):
    if dimension == 0:
        return {int(value): int(value) for value in np.asarray(mesh.vertex_global_ids)}
    if dimension == mesh.topological_dimension:
        rows = np.concatenate([np.asarray(block.vertices) for block in mesh.blocks])
    elif dimension == 1:
        rows = np.asarray(mesh.connectivity.edges)
    elif isinstance(mesh.connectivity, phx.discretization.PolyhedralConnectivity):
        offsets = np.asarray(mesh.connectivity.face_vertex_offsets)
        vertices = np.asarray(mesh.connectivity.face_vertex_values)
        rows = [
            vertices[start:stop]
            for start, stop in zip(offsets[:-1], offsets[1:], strict=True)
        ]
    else:
        rows = np.asarray(mesh.connectivity.faces)
    vertex_ids = np.asarray(mesh.vertex_global_ids)
    return {
        frozenset(int(value) for value in vertex_ids[row]): int(identifier)
        for row, identifier in zip(
            rows, np.asarray(mesh.entity_set(dimension).entity_ids), strict=True
        )
    }


@pytest.mark.parametrize("cell_kind", ("triangle", "tetrahedron", "hexahedron", "prism"))
def test_canonicalization_preserves_persistent_ids_at_every_degree(cell_kind):
    mesh = _two_cells(cell_kind)
    canonical = phx.meshing.canonicalize_cell_mesh(mesh)

    assert np.array_equal(canonical.blocks[0].global_ids, [10, 20])
    for dimension in range(mesh.topological_dimension + 1):
        assert _entity_ids_by_vertices(canonical, dimension) == _entity_ids_by_vertices(
            mesh, dimension
        )
    assert phx.meshing.canonicalize_cell_mesh(canonical) is canonical


def _quadratic_geometry(mesh):
    element = phx.discretization.fem.lagrange_element("triangle", 2)
    nodes = np.asarray(element.reference_nodes)
    points = np.asarray(mesh.coordinates)
    local = points[np.asarray(mesh.blocks[0].vertices)]
    coordinates = (
        local[:, :1]
        + nodes[None, :, :1] * (local[:, 1:2] - local[:, :1])
        + nodes[None, :, 1:] * (local[:, 2:3] - local[:, :1])
    )
    vertex_nodes = np.asarray(
        phx.discretization.reference_cell_topology("triangle").vertices
    )
    edge_nodes = ~np.any(np.all(nodes[:, None] == vertex_nodes[None], axis=-1), axis=1)
    coordinates[:, edge_nodes, 1] += 0.05
    routes = np.arange(coordinates.shape[0] * coordinates.shape[1]).reshape(
        coordinates.shape[:2]
    )
    return phx.discretization.CellGeometrySpec(
        {"cells": element},
        {"cells": routes},
        coordinates.reshape(-1, coordinates.shape[-1]),
    )


def test_certification_rejects_reordering_supplied_curved_geometry():
    mesh = _two_cells()
    geometry = _quadratic_geometry(mesh)

    with pytest.raises(ValueError, match="reorder supplied geometry"):
        phx.meshing.certify_cell_mesh(
            mesh,
            phx.SpatialCoordinateContract.si(),
            geometry=geometry,
        )


def test_corner_quality_does_not_certify_high_order_geometry():
    mesh = phx.meshing.canonicalize_cell_mesh(_two_cells())
    geometry = _quadratic_geometry(mesh)
    audit = phx.meshing.audit_cell_mesh(
        mesh,
        geometry,
        phx.meshing.evaluate_cell_quality(mesh),
    )

    assert audit.quality_scope == "corner_cells"
    with pytest.raises(phx.meshing.MeshingFailure, match="high-order geometry"):
        phx.meshing.certify_cell_mesh(
            mesh,
            phx.SpatialCoordinateContract.si(),
            geometry=geometry,
        )


def _association(mesh, ids, *, entity_set_id=None, residual=0.0):
    return phx.meshing.GeometryAssociation(
        phx.meshing.GeometryAssociationKind.SURFACE,
        "source",
        "revision",
        mesh.entity_set(mesh.topological_dimension).entity_set_id
        if entity_set_id is None
        else entity_set_id,
        np.asarray(ids),
        tuple("source-face" for _ in ids),
        np.full(len(ids), residual),
    )


@pytest.mark.parametrize("bad_set", (False, True))
def test_resolved_association_cannot_hide_stale_target_binding(bad_set):
    mesh = _mesh("triangle")
    association = _association(
        mesh,
        [0] if bad_set else [999],
        entity_set_id="stale-entity-set" if bad_set else None,
    )
    assert association.complete
    with pytest.raises(phx.meshing.MeshingFailure, match="association_"):
        phx.meshing.certify_cell_mesh(
            mesh,
            phx.SpatialCoordinateContract.si(),
            associations=(association,),
        )


def test_association_coverage_is_checked_only_when_requested():
    mesh = phx.meshing.canonicalize_cell_mesh(_two_cells())
    first = _association(mesh, [10], residual=100.0)
    second = _association(mesh, [20], residual=100.0)
    contract = phx.SpatialCoordinateContract.si()
    phx.meshing.certify_cell_mesh(mesh, contract, associations=(first,))
    policy = phx.meshing.CellMeshAuditPolicy(require_complete_association=True)
    with pytest.raises(
        phx.meshing.MeshingFailure, match="incomplete_geometry_association"
    ):
        phx.meshing.certify_cell_mesh(
            mesh, contract, associations=(first,), audit_policy=policy
        )
    result = phx.meshing.certify_cell_mesh(
        mesh,
        contract,
        associations=(first, second),
        audit_policy=policy,
    )
    assert result.audit.passed


def test_association_rows_cannot_claim_two_unique_sources_for_one_target():
    mesh = _mesh("triangle")
    first = _association(mesh, [0])
    second = phx.meshing.GeometryAssociation(
        first.association_kind,
        first.source_id,
        first.source_revision,
        first.target_entity_set_id,
        first.target_global_ids,
        ("different-face",),
        [0.0],
    )
    with pytest.raises(
        phx.meshing.MeshingFailure, match="conflicting_geometry_association"
    ):
        phx.meshing.certify_cell_mesh(
            mesh,
            phx.SpatialCoordinateContract.si(),
            associations=(first, second),
        )


def _label(mesh, ids, *, source_id=None):
    entities = mesh.entity_set(mesh.topological_dimension)
    return phx.meshing.MeshLabel(
        "selection",
        phx.meshing.MeshingScope(
            mesh.mesh_id if source_id is None else source_id,
            mesh.numeric_version,
            phx.meshing.MeshingEntityKind.MESH,
            mesh.topological_dimension,
            entities.entity_set_id,
            np.asarray(ids),
        ),
    )


@pytest.mark.parametrize("stale_source", (False, True))
def test_certification_rejects_stale_organization(stale_source):
    mesh = _mesh("triangle")
    label = _label(
        mesh,
        [0] if stale_source else [999],
        source_id="old-mesh" if stale_source else None,
    )
    with pytest.raises(phx.meshing.MeshingFailure, match="organization_"):
        phx.meshing.certify_cell_mesh(
            mesh,
            phx.SpatialCoordinateContract.si(),
            labels=(label,),
        )


def test_canonicalization_never_silently_rebinds_old_organization():
    mesh = _two_cells()
    label = _label(mesh, [20])
    with pytest.raises(phx.meshing.MeshingFailure, match="organization_"):
        phx.meshing.certify_cell_mesh(
            mesh,
            phx.SpatialCoordinateContract.si(),
            labels=(label,),
        )


def test_audit_rejects_quality_from_other_coordinates():
    mesh = _mesh("triangle")
    quality = phx.meshing.evaluate_cell_quality(mesh, 2.0 * mesh.coordinates)
    audit = phx.meshing.audit_cell_mesh(
        mesh,
        phx.discretization.CellGeometrySpec.affine(mesh),
        quality,
    )
    assert not audit.passed
    assert "quality_binding" in audit.issues


def test_audit_detects_geometry_rows_bound_to_the_wrong_cells():
    mesh = phx.meshing.canonicalize_cell_mesh(_two_cells())
    geometry = phx.discretization.CellGeometrySpec(
        {"cells": phx.discretization.fem.lagrange_element("triangle", 1)},
        {"cells": np.asarray(mesh.blocks[0].vertices)[::-1]},
        mesh.coordinates,
    )
    audit = phx.meshing.audit_cell_mesh(
        mesh, geometry, phx.meshing.evaluate_cell_quality(mesh)
    )
    assert not audit.passed
    assert "geometry_corner_binding" in audit.issues


def _rebuild_result(result, **changes):
    return phx.meshing.CellMeshingResult(
        result.mesh,
        changes.pop("geometry", result.geometry),
        result.coordinate_contract,
        result.audit,
        result.quality,
        result.compliance,
        result.trace,
        result.provider,
        result.runtime,
        result.derivative_mode,
        result.provenance,
        **changes,
    )


def test_result_rejects_changed_geometry_with_the_same_layout():
    result = phx.meshing.certify_cell_mesh(
        _mesh("triangle"), phx.SpatialCoordinateContract.si()
    )
    geometry = phx.discretization.CellGeometrySpec(
        dict(zip(result.geometry.block_names, result.geometry.elements, strict=True)),
        dict(
            zip(result.geometry.block_names, result.geometry.geometry_dofs, strict=True)
        ),
        result.geometry.coordinates + 1.0,
    )
    assert geometry.geometry_layout_id == result.geometry.geometry_layout_id
    with pytest.raises(ValueError, match="geometry values"):
        _rebuild_result(result, geometry=geometry)


def test_result_rejects_valid_but_unaudited_semantic_evidence():
    result = phx.meshing.certify_cell_mesh(
        _mesh("triangle"), phx.SpatialCoordinateContract.si()
    )
    with pytest.raises(ValueError, match="audited evidence"):
        _rebuild_result(result, labels=(_label(result.mesh, [0]),))


def test_result_boundary_must_cover_the_actual_mesh_faces_and_coordinates():
    result = phx.meshing.certify_cell_mesh(
        _mesh("tetrahedron"), phx.SpatialCoordinateContract.si()
    )
    mesh = result.mesh
    faces = np.asarray(mesh.connectivity.faces)
    metadata = phx.geometry.surface.SurfaceMetadata(
        source_id="boundary",
        source_revision="r1",
        coordinate_contract=result.coordinate_contract,
        provenance=("qualification",),
    )

    def boundary(points, rows):
        return phx.geometry.surface.SurfaceModel.from_triangles(
            points,
            rows,
            metadata,
            vertex_global_ids=mesh.vertex_global_ids,
            repair_orientation=True,
        )

    valid = _rebuild_result(result, boundary=boundary(mesh.coordinates, faces))
    assert valid.boundary.mesh.blocks[0].cell_count == 4
    with pytest.raises(ValueError, match="boundary_coordinates"):
        _rebuild_result(
            result, boundary=boundary(np.asarray(mesh.coordinates) + 1.0, faces)
        )
    with pytest.raises(ValueError, match="boundary_coverage"):
        _rebuild_result(result, boundary=boundary(mesh.coordinates, faces[:-1]))

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import phydrax as phx
from phydrax.meshing._quality import _measure_rule


def _mesh(cell_kind):
    topology = phx.discretization.reference_cell_topology(cell_kind)
    coordinates = np.asarray(topology.vertices, dtype="float64")
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


@pytest.mark.parametrize("cell_kind", ("prism", "pyramid", "hexahedron"))
def test_solid_quality_compiles_on_a_cold_measure_rule_cache(cell_kind):
    # The first evaluation of a kind may happen under a trace; the cached host
    # quadrature rule must not capture tracers.
    _measure_rule.cache_clear()
    mesh = _mesh(cell_kind)

    compiled = jax.jit(
        lambda coordinates: phx.meshing.evaluate_cell_quality(mesh, coordinates)
    )(mesh.coordinates)

    eager = phx.meshing.evaluate_cell_quality(mesh)
    np.testing.assert_allclose(compiled.measures, eager.measures, rtol=1e-14)
    assert float(compiled.measures[0]) > 0.0


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
    assert not bool(evaluation.sampled_valid[0])
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


def test_curved_geometry_is_certified_beyond_corner_quality():
    mesh = phx.meshing.canonicalize_cell_mesh(_two_cells())
    geometry = _quadratic_geometry(mesh)
    audit = phx.meshing.audit_cell_mesh(mesh, geometry)

    assert audit.quality_scope == "corner_cells"
    assert audit.validity.all_certified
    assert audit.passed
    result = phx.meshing.certify_cell_mesh(
        mesh, phx.SpatialCoordinateContract.si(), geometry=geometry
    )
    assert result.audit.validity.certificate_id == audit.validity.certificate_id


def _single_triangle():
    return phx.discretization.CellMesh(
        np.asarray(((0.0, 0.0), (1.0, 0.0), (0.0, 1.0))),
        (phx.discretization.CellBlock("cells", "triangle", np.asarray(((0, 1, 2),))),),
    )


def _p2_triangle(edge_nodes):
    element = phx.discretization.fem.lagrange_element("triangle", 2)
    nodes = np.asarray(element.reference_nodes, dtype=np.float64).copy()
    nodes[3:] = edge_nodes
    return phx.discretization.CellGeometrySpec(
        {"cells": element}, {"cells": np.arange(6)[None, :]}, nodes
    )


def test_curved_p2_triangle_inverted_between_nodes_is_invalid():
    mesh = _single_triangle()
    # The Jacobian determinant is positive at all six Lagrange nodes and at the
    # corners, but negative inside the cell.
    geometry = _p2_triangle(((0.26, 0.23), (0.58, 0.77), (-0.1, 0.06)))
    certificate = phx.discretization.certify_cell_geometry_validity(geometry, mesh=mesh)

    assert bool(phx.meshing.evaluate_cell_quality(mesh).sampled_valid[0])
    assert int(certificate.status[0]) == phx.discretization.CellValidityStatus.INVALID
    assert float(certificate.determinant_lower[0]) < 0.0
    assert int(certificate.depth[0]) > 0
    with pytest.raises(phx.meshing.MeshingFailure, match="invalid_geometry"):
        phx.meshing.certify_cell_mesh(
            mesh, phx.SpatialCoordinateContract.si(), geometry=geometry
        )


def test_subdivision_budget_leaves_valid_curved_cell_unresolved():
    mesh = _single_triangle()
    # Valid everywhere, but one Bernstein edge coefficient is negative.
    geometry = _p2_triangle(((0.86, -0.17), (0.54, 0.29), (-0.06, 0.51)))
    tiny = phx.discretization.CellValidityPolicy(maximum_subdivision_depth=0)
    unresolved = phx.discretization.certify_cell_geometry_validity(
        geometry, mesh=mesh, policy=tiny
    )
    certified = phx.discretization.certify_cell_geometry_validity(geometry, mesh=mesh)

    assert int(unresolved.status[0]) == phx.discretization.CellValidityStatus.UNRESOLVED
    assert float(unresolved.determinant_lower[0]) < 0.0
    assert (
        int(certified.status[0]) == phx.discretization.CellValidityStatus.CERTIFIED_VALID
    )
    assert float(certified.determinant_lower[0]) > 0.0

    rejected = phx.meshing.audit_cell_mesh(
        mesh, geometry, policy=phx.meshing.CellMeshAuditPolicy(validity_policy=tiny)
    )
    assert "unresolved_geometry_validity" in rejected.issues
    assert "unresolved_geometry_validity" in rejected.evaluated_checks
    recorded = phx.meshing.audit_cell_mesh(
        mesh,
        geometry,
        policy=phx.meshing.CellMeshAuditPolicy(
            validity_policy=tiny, unresolved=phx.meshing.CellMeshAuditDisposition.RECORD
        ),
    )
    assert recorded.passed
    assert recorded.recorded == ("unresolved_geometry_validity",)
    assert "unresolved_geometry_validity" in recorded.evaluated_checks


def test_twisted_trilinear_hexahedron_is_invalid_despite_positive_corners():
    points = np.asarray(
        (
            (-0.62, 0.9, 0.43),
            (0.83, -0.37, -0.44),
            (1.06, 0.71, -0.34),
            (0.37, 1.16, -0.18),
            (0.33, 0.62, 0.51),
            (0.73, 0.42, 1.32),
            (1.1, 1.52, 0.51),
            (-0.67, 0.61, 1.06),
        )
    )
    mesh = phx.discretization.CellMesh(
        points,
        (phx.discretization.CellBlock("cells", "hexahedron", np.arange(8)[None, :]),),
    )
    quality = phx.meshing.evaluate_cell_quality(mesh)
    certificate = phx.discretization.certify_cell_geometry_validity(mesh)

    assert bool(quality.sampled_valid[0])
    assert float(quality.scaled_jacobian[0]) > 0.0
    assert int(certificate.status[0]) == phx.discretization.CellValidityStatus.INVALID
    assert float(certificate.determinant_lower[0]) < 0.0
    assert int(certificate.depth[0]) > 0


@pytest.mark.parametrize("reflex", range(4))
def test_dart_quadrilateral_reflex_corner_is_detected_at_every_vertex(reflex):
    points = np.asarray(
        phx.discretization.reference_cell_topology("quadrilateral").vertices
    )
    points[reflex] = 0.75 * points[(reflex + 2) % 4] + 0.25 * points[reflex]
    mesh = phx.discretization.CellMesh(
        points,
        (phx.discretization.CellBlock("cells", "quadrilateral", np.arange(4)[None, :]),),
    )
    quality = phx.meshing.evaluate_cell_quality(mesh)
    certificate = phx.discretization.certify_cell_geometry_validity(mesh)

    assert float(quality.measures[0]) > 0.0
    assert not bool(quality.sampled_valid[0])
    assert float(quality.scaled_jacobian[0]) < 0.0
    assert float(quality.maximum_angle[0]) > np.pi
    assert int(certificate.status[0]) == phx.discretization.CellValidityStatus.INVALID


def test_sliver_tetrahedron_has_degenerate_dihedral_angles():
    height = 1.0e-3
    points = np.asarray(
        ((0.0, 0.0, 0.0), (1.0, 1.0, 0.0), (0.0, 1.0, height), (1.0, 0.0, height))
    )
    mesh = phx.discretization.CellMesh(
        points,
        (phx.discretization.CellBlock("cells", "tetrahedron", np.arange(4)[None, :]),),
    )
    sliver = phx.meshing.evaluate_cell_quality(mesh)
    regular_points = np.asarray(
        (
            (0.0, 0.0, 0.0),
            (1.0, 0.0, 0.0),
            (0.5, 0.5 * np.sqrt(3.0), 0.0),
            (0.5, np.sqrt(3.0) / 6.0, np.sqrt(2.0 / 3.0)),
        )
    )
    regular = phx.meshing.evaluate_cell_quality(
        phx.discretization.CellMesh(
            regular_points,
            (
                phx.discretization.CellBlock(
                    "cells", "tetrahedron", np.arange(4)[None, :]
                ),
            ),
        )
    )

    assert bool(sliver.sampled_valid[0])
    assert float(sliver.aspect_ratios[0]) < 1.5
    assert float(sliver.minimum_angle[0]) < 0.01
    assert float(sliver.maximum_angle[0]) > np.pi - 0.01
    assert float(sliver.radius_ratios[0]) < 0.01
    assert float(sliver.sliver_measures[0]) < 0.01
    assert float(regular.minimum_angle[0]) == pytest.approx(np.arccos(1.0 / 3.0))
    assert float(regular.maximum_angle[0]) == pytest.approx(np.arccos(1.0 / 3.0))
    assert float(regular.radius_ratios[0]) == pytest.approx(1.0)
    assert float(regular.mean_ratios[0]) == pytest.approx(1.0)


def test_warped_quadrilateral_reports_warpage_but_planar_does_not():
    planar = np.asarray(
        ((0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (1.0, 1.0, 0.0), (0.0, 1.0, 0.0))
    )
    warped = planar.copy()
    warped[2, 2] = 0.3

    def evaluate(points):
        mesh = phx.discretization.CellMesh(
            points,
            (
                phx.discretization.CellBlock(
                    "cells", "quadrilateral", np.arange(4)[None, :]
                ),
            ),
        )
        return phx.meshing.evaluate_cell_quality(mesh)

    assert float(evaluate(planar).warpage[0]) == pytest.approx(0.0, abs=1.0e-12)
    quality = evaluate(warped)
    assert float(quality.warpage[0]) > 0.05
    assert bool(quality.sampled_valid[0])


def test_coincident_vertices_are_welded_and_disposed_by_policy():
    points = np.asarray(
        ((0.0, 0.0), (1.0, 0.0), (0.0, 1.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0))
    )
    mesh = phx.discretization.CellMesh(
        points,
        (
            phx.discretization.CellBlock(
                "cells", "triangle", np.asarray(((0, 1, 2), (3, 4, 5)))
            ),
        ),
    )
    geometry = phx.discretization.CellGeometrySpec.affine(mesh)
    rejected = phx.meshing.audit_cell_mesh(mesh, geometry)
    recorded = phx.meshing.audit_cell_mesh(
        mesh,
        geometry,
        policy=phx.meshing.CellMeshAuditPolicy(
            coincident_vertices=phx.meshing.CellMeshAuditDisposition.RECORD
        ),
    )

    assert rejected.issues == ("coincident_vertices",)
    assert dict(rejected.check_counts)["coincident_vertices"] == 4
    assert recorded.passed
    assert recorded.recorded == ("coincident_vertices",)
    # The welded edge is shared consistently, so no other topology finding.
    assert dict(recorded.check_counts)["inconsistent_orientation"] == 0
    assert dict(recorded.check_counts)["nonmanifold_vertices"] == 0


def test_pinched_vertex_is_non_manifold():
    points = np.asarray(((0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (-1.0, 0.0), (-1.0, -1.0)))
    mesh = phx.discretization.CellMesh(
        points,
        (
            phx.discretization.CellBlock(
                "cells", "triangle", np.asarray(((0, 1, 2), (0, 3, 4)))
            ),
        ),
    )
    audit = phx.meshing.audit_cell_mesh(
        mesh, phx.discretization.CellGeometrySpec.affine(mesh)
    )

    assert not audit.passed
    assert audit.issues == ("nonmanifold_vertices",)
    assert dict(audit.check_counts)["nonmanifold_vertices"] == 1


def test_default_audit_rejects_overlapping_cells_and_reports_skipped_checks():
    points = np.asarray(
        ((0.0, 0.0), (2.0, 0.0), (0.0, 2.0), (0.5, 0.5), (2.5, 0.5), (0.5, 2.5))
    )
    mesh = phx.discretization.CellMesh(
        points,
        (
            phx.discretization.CellBlock(
                "cells", "triangle", np.asarray(((0, 1, 2), (3, 4, 5)))
            ),
        ),
    )
    geometry = phx.discretization.CellGeometrySpec.affine(mesh)
    rejected = phx.meshing.audit_cell_mesh(mesh, geometry)
    skipped = phx.meshing.audit_cell_mesh(
        mesh,
        geometry,
        policy=phx.meshing.CellMeshAuditPolicy(
            self_intersection=phx.meshing.CellMeshAuditDisposition.SKIP
        ),
    )

    assert rejected.issues == ("self_intersection",)
    assert "self_intersection" in rejected.evaluated_checks
    assert rejected.skipped_checks == ("open_boundary",)
    assert skipped.passed
    assert skipped.skipped_checks == ("open_boundary", "self_intersection")
    assert "self_intersection" not in skipped.evaluated_checks
    assert "self_intersection" not in dict(skipped.check_counts)


def test_concave_polygon_cell_passes_the_self_intersection_audit():
    # Notched square: the vertex-zero fan folds over itself, the cell does not.
    points = np.asarray(((0.0, 0.0), (3.0, 0.0), (3.0, 3.0), (1.5, 1.0), (0.0, 3.0)))
    mesh = phx.discretization.CellMesh(
        points,
        (
            phx.discretization.CellBlock(
                "cells", "polygon", np.arange(5, dtype=np.int32)[None, :]
            ),
        ),
    )
    audit = phx.meshing.audit_cell_mesh(
        mesh, phx.discretization.CellGeometrySpec.affine(mesh)
    )

    assert audit.passed
    assert dict(audit.check_counts)["self_intersection"] == 0
    bounded = phx.meshing.audit_cell_mesh(
        mesh,
        phx.discretization.CellGeometrySpec.affine(mesh),
        policy=phx.meshing.CellMeshAuditPolicy(maximum_intersection_candidates=1),
    )
    assert not bounded.passed
    assert "self_intersection_capacity" in bounded.unresolved
    assert "unresolved_self_intersection_capacity" in bounded.issues


def test_metric_quality_measures_shape_in_metric_space():
    points = np.asarray(((0.0, 0.0), (10.0, 0.0), (5.0, 0.5 * np.sqrt(3.0))))
    mesh = phx.discretization.CellMesh(
        points,
        (phx.discretization.CellBlock("cells", "triangle", np.asarray(((0, 1, 2),))),),
    )
    scope = phx.meshing.MeshingScope(
        mesh.mesh_id,
        mesh.numeric_version,
        phx.meshing.MeshingEntityKind.MESH,
        0,
        mesh.entity_set(0).entity_set_id,
        np.asarray(mesh.vertex_global_ids),
    )

    def quality(tensor):
        metric = phx.meshing.MeshMetricField(
            scope,
            np.broadcast_to(tensor, (3, 2, 2)),
            minimum_size=0.01,
            maximum_size=100.0,
        )
        return phx.meshing.evaluate_cell_quality(mesh, metric=metric)

    euclidean = quality(np.eye(2))
    stretched = quality(np.diag((0.01, 1.0)))

    assert bool(jnp.isnan(phx.meshing.evaluate_cell_quality(mesh).metric_quality[0]))
    assert float(euclidean.metric_quality[0]) == pytest.approx(
        float(euclidean.mean_ratios[0])
    )
    assert float(stretched.mean_ratios[0]) < 0.25
    assert float(stretched.metric_quality[0]) == pytest.approx(1.0)


def test_finite_volume_quality_measures_skewness_and_non_orthogonality():
    def evaluate(shift):
        points = np.asarray(
            (
                (0.0, 0.0),
                (1.0, 0.0),
                (2.0, 0.0),
                (shift, 1.0),
                (1.0 + shift, 1.0),
                (2.0 + shift, 1.0),
            )
        )
        mesh = phx.discretization.CellMesh(
            points,
            (
                phx.discretization.CellBlock(
                    "cells",
                    "quadrilateral",
                    np.asarray(((0, 1, 4, 3), (1, 2, 5, 4))),
                    global_ids=np.asarray((7, 9)),
                ),
            ),
        )
        return phx.meshing.evaluate_finite_volume_quality(mesh)

    orthogonal = evaluate(0.0)
    sheared = evaluate(0.5)

    assert orthogonal.face_global_ids.shape == (1,)
    assert np.array_equal(orthogonal.owner_cell_global_ids, (7,))
    assert np.array_equal(orthogonal.neighbor_cell_global_ids, (9,))
    assert float(orthogonal.non_orthogonality[0]) == pytest.approx(0.0, abs=1.0e-12)
    assert float(orthogonal.skewness[0]) == pytest.approx(0.0, abs=1.0e-12)
    assert float(sheared.non_orthogonality[0]) == pytest.approx(np.arctan(0.5))
    assert float(sheared.skewness[0]) == pytest.approx(0.0, abs=1.0e-12)


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


def test_adjacent_zone_patch_is_audited_against_exact_face_incidence():
    mesh = phx.meshing.canonicalize_cell_mesh(
        phx.discretization.CellMesh(
            np.asarray(
                (
                    (0.0, 0.0, 0.0),
                    (1.0, 0.0, 0.0),
                    (0.0, 1.0, 0.0),
                    (0.0, 0.0, 1.0),
                    (0.0, 0.0, -1.0),
                )
            ),
            (
                phx.discretization.CellBlock(
                    "cells",
                    "tetrahedron",
                    np.asarray(((0, 1, 2, 3), (0, 2, 1, 4))),
                    global_ids=np.asarray((10, 20)),
                ),
            ),
        )
    )

    def scope(dimension, identifiers):
        entities = mesh.entity_set(dimension)
        return phx.meshing.MeshingScope(
            mesh.mesh_id,
            mesh.numeric_version,
            phx.meshing.MeshingEntityKind.MESH,
            dimension,
            entities.entity_set_id,
            np.asarray(identifiers),
        )

    fluid = phx.meshing.MeshZone(
        "fluid",
        phx.meshing.MeshZoneRole.REGION,
        scope(3, (10,)),
        material_id="water",
        region_role=phx.meshing.RegionRole.FLUID,
    )
    solid = phx.meshing.MeshZone(
        "solid",
        phx.meshing.MeshZoneRole.REGION,
        scope(3, (20,)),
        material_id="steel",
        region_role=phx.meshing.RegionRole.SOLID,
    )
    face_ids = np.asarray(mesh.entity_set(2).entity_ids)
    internal = face_ids[~np.asarray(mesh.connectivity.boundary_faces)]
    exterior = face_ids[np.asarray(mesh.connectivity.boundary_faces)]
    interface = phx.meshing.MeshPatch(
        "fluid-solid",
        scope(2, internal),
        adjacent_zone_ids=(solid.zone_id, fluid.zone_id),
    )
    result = phx.meshing.certify_cell_mesh(
        mesh,
        phx.SpatialCoordinateContract.si(),
        patches=(interface,),
        zones=(fluid, solid),
    )
    assert result.audit.passed

    wrong = phx.meshing.MeshPatch(
        "fluid-solid",
        scope(2, exterior[:1]),
        adjacent_zone_ids=(fluid.zone_id, solid.zone_id),
    )
    with pytest.raises(phx.meshing.MeshingFailure, match="patch_zone_adjacency"):
        phx.meshing.certify_cell_mesh(
            mesh,
            phx.SpatialCoordinateContract.si(),
            patches=(wrong,),
            zones=(fluid, solid),
        )


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

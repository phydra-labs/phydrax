from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import phydrax as phx
from phydrax.meshing._quality import _measure_rule


def _mesh(cell_kind: Any) -> Any:
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


def test_audit_quality_scenario_1() -> None:
    for cell_kind in (
        "interval",
        "triangle",
        "quadrilateral",
        "tetrahedron",
        "prism",
        "pyramid",
        "hexahedron",
    ):
        mesh = _mesh(cell_kind)
        result = phx.meshing.certify_cell_mesh(mesh, phx.SpatialCoordinateContract.si())

        assert result.audit.passed
        assert result.quality.minimum_measure > 0.0
        assert result.quality.minimum_mean_ratio > 0.0
        assert result.quality.maximum_aspect_ratio >= 1.0
    for cell_kind in ("prism", "pyramid", "hexahedron"):
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
    for cell_kind in ("triangle", "tetrahedron", "hexahedron", "prism"):
        mesh = _two_cells(cell_kind)
        canonical = phx.meshing.canonicalize_cell_mesh(mesh)

        assert np.array_equal(canonical.blocks[0].global_ids, [10, 20])
        for dimension in range(mesh.topological_dimension + 1):
            assert _entity_ids_by_vertices(
                canonical, dimension
            ) == _entity_ids_by_vertices(mesh, dimension)
        assert phx.meshing.canonicalize_cell_mesh(canonical) is canonical
    mesh = _two_cells()
    label = _label(mesh, [20])
    with pytest.raises(phx.meshing.MeshingFailure, match="organization_"):
        phx.meshing.certify_cell_mesh(
            mesh,
            phx.SpatialCoordinateContract.si(),
            labels=(label,),
        )


def test_triangle_quality_is_fixed_topology_differentiable_and_rejects_inversion() -> (
    None
):
    mesh = _mesh("triangle")

    def objective(coordinates: Any) -> Any:
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


def _two_cells(cell_kind: Any = "triangle") -> Any:
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


def _entity_ids_by_vertices(mesh: Any, dimension: Any) -> Any:
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


def _quadratic_geometry(mesh: Any) -> Any:
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


def test_audit_quality_scenario_2() -> None:
    mesh = _two_cells()
    geometry = _quadratic_geometry(mesh)

    with pytest.raises(ValueError, match="reorder supplied geometry"):
        phx.meshing.certify_cell_mesh(
            mesh,
            phx.SpatialCoordinateContract.si(),
            geometry=geometry,
        )
    for stale_source in (False, True):
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
    mesh = phx.meshing.canonicalize_cell_mesh(_two_cells())
    geometry = _quadratic_geometry(mesh)
    audit = phx.meshing.audit_cell_mesh(mesh, geometry)

    assert audit.validity.all_certified
    assert audit.passed
    result = phx.meshing.certify_cell_mesh(
        mesh, phx.SpatialCoordinateContract.si(), geometry=geometry
    )
    assert result.audit.validity.certificate_id == audit.validity.certificate_id
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


def _single_triangle() -> Any:
    return phx.discretization.CellMesh(
        np.asarray(((0.0, 0.0), (1.0, 0.0), (0.0, 1.0))),
        (phx.discretization.CellBlock("cells", "triangle", np.asarray(((0, 1, 2),))),),
    )


def _p2_triangle(edge_nodes: Any) -> Any:
    element = phx.discretization.fem.lagrange_element("triangle", 2)
    nodes = np.asarray(element.reference_nodes, dtype=np.float64).copy()
    nodes[3:] = edge_nodes
    return phx.discretization.CellGeometrySpec(
        {"cells": element}, {"cells": np.arange(6)[None, :]}, nodes
    )


def test_audit_quality_scenario_3() -> None:
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
    for reflex in range(4):
        points = np.asarray(
            phx.discretization.reference_cell_topology("quadrilateral").vertices
        )
        points[reflex] = 0.75 * points[(reflex + 2) % 4] + 0.25 * points[reflex]
        mesh = phx.discretization.CellMesh(
            points,
            (
                phx.discretization.CellBlock(
                    "cells", "quadrilateral", np.arange(4)[None, :]
                ),
            ),
        )
        quality = phx.meshing.evaluate_cell_quality(mesh)
        certificate = phx.discretization.certify_cell_geometry_validity(mesh)

        assert float(quality.measures[0]) > 0.0
        assert not bool(quality.sampled_valid[0])
        assert float(quality.scaled_jacobian[0]) < 0.0
        assert float(quality.maximum_angle[0]) > np.pi
        assert int(certificate.status[0]) == phx.discretization.CellValidityStatus.INVALID
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


def test_warped_quadrilateral_reports_warpage_but_planar_does_not() -> None:
    planar = np.asarray(
        ((0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (1.0, 1.0, 0.0), (0.0, 1.0, 0.0))
    )
    warped = planar.copy()
    warped[2, 2] = 0.3

    def evaluate(points: Any) -> Any:
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


def test_audit_quality_scenario_4() -> None:
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


@pytest.mark.parametrize(
    ("complete", "nonmanifold"),
    (((True, True), 1), ((False, True), 0)),
    ids=("complete-star-judged", "halo-truncated-star-skipped"),
)
def test_owner_local_manifold_check_judges_only_complete_stars(
    complete: tuple[bool, bool], nonmanifold: int
) -> None:
    from phydrax.discretization._cell_mesh import CellMeshStorage
    from phydrax.meshing._audit_topology import audit_welded_topology

    jax.config.update("jax_enable_x64", True)
    # A bow-tie: two triangles meeting only at vertex zero. A complete star there
    # is a genuine nonmanifold vertex; the same star with an incident cell cut at
    # an artificial halo boundary carries no manifold evidence either way.
    dense = phx.discretization.CellMesh(
        np.asarray(((0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (-1.0, 0.0), (-1.0, -1.0))),
        (
            phx.discretization.CellBlock(
                "cells", "triangle", np.asarray(((0, 1, 2), (0, 3, 4)), dtype=np.int32)
            ),
        ),
    )
    ids = tuple(
        np.asarray(entities.entity_ids) for entities in dense.topology.entity_sets
    )
    storage = CellMeshStorage(
        tuple(entities.count for entities in dense.topology.entity_sets),
        ids,
        tuple(np.zeros(value.shape, dtype=np.int32) for value in ids),
        partition_index=0,
        partition_count=2,
        logical_topology_id=dense.topology_id,
        logical_geometry_id=dense.geometry_id,
        evidence_id="a" * 64,
        logical_arrays=(("coordinates", dense.coordinates),),
        local_coordinates=dense.coordinates,
        local_blocks=dense.blocks,
        local_neighborhood_complete=np.asarray(complete, dtype=np.bool_),
        neighborhood_depth=1,
    )
    mesh = phx.discretization.CellMesh(dense.coordinates, dense.blocks, storage=storage)
    evidence = audit_welded_topology(
        mesh,
        np.asarray(mesh.coordinates),
        coincident_tolerance=None,
        candidate_capacity=64,
        intersection_candidate_capacity=64,
        check_manifold=True,
        check_watertight=False,
        check_self_intersection=False,
    )
    assert evidence.nonmanifold_vertices == nonmanifold


def test_audit_quality_scenario_5() -> None:
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
    for bad_set in (False, True):
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
    mesh = _mesh("triangle")
    first = _association(mesh, [0])
    second = phx.meshing.GeometryAssociation(
        first.association_kind,
        first.source_id,
        first.source_revision,
        first.target_entity_set_id,
        first.target_global_ids,
        ("different-face",),
        # ty: ignore[invalid-argument-type]
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


def test_metric_quality_measures_shape_in_metric_space() -> None:
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

    def quality(tensor: Any) -> Any:
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


def test_finite_volume_quality_measures_skewness_and_non_orthogonality() -> None:
    def evaluate(shift: Any) -> Any:
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


def _association(
    mesh: Any, ids: Any, *, entity_set_id: Any = None, residual: Any = 0.0
) -> Any:
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


def test_adjacent_zone_patch_is_audited_against_exact_face_incidence() -> None:
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

    def scope(dimension: Any, identifiers: Any) -> Any:
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
    # ty: ignore[unresolved-attribute]
    internal = face_ids[~np.asarray(mesh.connectivity.boundary_faces)]
    # ty: ignore[unresolved-attribute]
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


def _label(mesh: Any, ids: Any, *, source_id: Any = None) -> Any:
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


def test_audit_quality_scenario_6() -> None:
    mesh = _mesh("triangle")
    quality = phx.meshing.evaluate_cell_quality(mesh, 2.0 * mesh.coordinates)
    audit = phx.meshing.audit_cell_mesh(
        mesh,
        phx.discretization.CellGeometrySpec.affine(mesh),
        quality,
    )
    assert not audit.passed
    assert "quality_binding" in audit.issues
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
    result = phx.meshing.certify_cell_mesh(
        _mesh("triangle"), phx.SpatialCoordinateContract.si()
    )
    with pytest.raises(ValueError, match="audited evidence"):
        _rebuild_result(result, labels=(_label(result.mesh, [0]),))


def _rebuild_result(result: Any, **changes: Any) -> Any:
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


def test_result_boundary_must_cover_the_actual_mesh_faces_and_coordinates() -> None:
    result = phx.meshing.certify_cell_mesh(
        _mesh("tetrahedron"), phx.SpatialCoordinateContract.si()
    )
    mesh = result.mesh
    # ty: ignore[unresolved-attribute]
    faces = np.asarray(mesh.connectivity.faces)
    metadata = phx.geometry.surface.SurfaceMetadata(
        source_id="boundary",
        source_revision="r1",
        coordinate_contract=result.coordinate_contract,
        provenance=("qualification",),
    )

    def boundary(points: Any, rows: Any) -> Any:
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


def test_audit_separates_corner_and_mapped_evidence_with_failing_cells() -> None:
    mesh = _single_triangle()
    geometry = _p2_triangle(((0.26, 0.23), (0.58, 0.77), (-0.1, 0.06)))
    audit = phx.meshing.audit_cell_mesh(mesh, geometry)

    assert audit.topology_id == mesh.topology_id
    assert "invalid_geometry" in audit.mapped_checks
    assert "invalid_geometry" not in audit.corner_checks
    assert "self_intersection" in audit.corner_checks
    assert "sampled_quality" in audit.mapped_checks
    assert "sampled_quality" not in audit.corner_checks
    # Straight corner samples cannot establish the mapped-cell verdict.
    assert bool(phx.meshing.evaluate_cell_quality(mesh).sampled_valid[0])
    assert audit.failing_cells == (("invalid_geometry", (0,)),)


@pytest.mark.parametrize(
    ("invalid_geometry", "constructible"),
    [
        pytest.param(phx.meshing.CellMeshAuditDisposition.REJECT, False, id="reject"),
        pytest.param(phx.meshing.CellMeshAuditDisposition.RECORD, True, id="record"),
    ],
)
def test_recorded_unresolved_mandatory_check_blocks_a_successful_result(
    invalid_geometry: Any, constructible: bool
) -> None:
    mesh = _single_triangle()
    geometry = _p2_triangle(((0.86, -0.17), (0.54, 0.29), (-0.06, 0.51)))
    policy = phx.meshing.CellMeshAuditPolicy(
        validity_policy=phx.discretization.CellValidityPolicy(
            maximum_subdivision_depth=0
        ),
        unresolved=phx.meshing.CellMeshAuditDisposition.RECORD,
        invalid_geometry=invalid_geometry,
    )
    audit = phx.meshing.audit_cell_mesh(mesh, geometry, policy=policy)

    assert audit.passed
    assert audit.mandatory_unresolved == (() if constructible else ("geometry_validity",))
    if constructible:
        phx.meshing.certify_cell_mesh(
            mesh,
            phx.SpatialCoordinateContract.si(),
            geometry=geometry,
            audit_policy=policy,
        )
        return
    with pytest.raises(ValueError, match="mandatory checks unresolved"):
        phx.meshing.certify_cell_mesh(
            mesh,
            phx.SpatialCoordinateContract.si(),
            geometry=geometry,
            audit_policy=policy,
        )


def test_result_requires_a_passed_certification_bound_to_its_audit() -> None:
    square = np.asarray(((0, 0), (0.5, 0), (1, 0), (1, 1), (0.5, 1), (0, 1.0)))
    mesh = phx.discretization.CellMesh.from_triangles(
        square, np.asarray(((0, 1, 4), (0, 4, 5), (1, 2, 3), (1, 3, 4)))
    )
    result = phx.meshing.certify_cell_mesh(
        mesh,
        phx.SpatialCoordinateContract.si(),
        audit_policy=phx.meshing.CellMeshAuditPolicy(
            watertight_boundary=phx.meshing.CellMeshAuditDisposition.REJECT
        ),
    )
    domain = phx.geometry.PiecewiseLinearDomain(
        square,
        np.asarray(((0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 0), (1, 4))),
        np.asarray(((0, -1), (1, -1), (1, -1), (1, -1), (0, -1), (0, -1), (0, 1))),
        ("left", "right"),
        source_id="split-square",
    )
    points = np.asarray(result.mesh.coordinates)
    rows = np.asarray(result.mesh.blocks[0].vertices)
    left = np.mean(points[rows][:, :, 0], axis=1) < 0.5

    def report(regions: Any) -> Any:
        return phx.meshing.certify_meshing_acceptance(
            result.mesh,
            result.geometry,
            result.audit,
            schedule=phx.meshing.MeshCertificationSchedule("volume_plc"),
            domain=domain,
            cell_regions=regions,
        )

    accepted = _rebuild_result(
        result, certification=report(np.where(left, 0, 1).astype(np.int64))
    )
    assert accepted.certification is not None
    assert accepted.certification.passed
    assert accepted.result_id != result.result_id
    with pytest.raises(phx.meshing.MeshingFailure, match="domain_coverage"):
        _rebuild_result(result, certification=report(np.zeros(4, dtype=np.int64)))


@pytest.mark.parametrize("shifted_corner", [False, True], ids=["bound", "stale-corner"])
def test_restricted_quadrilateral_publication_checks_actual_corner_binding(
    shifted_corner: bool,
) -> None:
    from phydrax.discretization._cell_geometry import (
        coordinate_lagrange_element,
        RestrictedCellGeometryElement,
    )

    reference = np.asarray(
        phx.discretization.reference_cell_topology("quadrilateral").vertices,
        dtype=np.float64,
    )
    element = coordinate_lagrange_element("quadrilateral", 1)
    restricted = RestrictedCellGeometryElement(
        element,
        "quadrilateral",
        0.5 * np.eye(2, dtype=np.float64),
        np.zeros(2, dtype=np.float64),
    )
    corners = 0.5 * reference
    if shifted_corner:
        corners[0, 0] += 0.001
    mesh = phx.discretization.CellMesh(
        corners,
        (
            phx.discretization.CellBlock(
                "quad",
                "quadrilateral",
                np.arange(4, dtype=np.int32)[None],
            ),
        ),
    )
    geometry = phx.discretization.CellGeometrySpec(
        {"quad": restricted},
        {"quad": np.arange(element.local_dof_count, dtype=np.int32)[None]},
        np.asarray(element.reference_nodes, dtype=np.float64),
    )
    if shifted_corner:
        with pytest.raises(phx.meshing.MeshingFailure):
            phx.meshing.certify_cell_mesh(
                mesh,
                phx.SpatialCoordinateContract.si(),
                geometry=geometry,
            )
        return
    result = phx.meshing.certify_cell_mesh(
        mesh,
        phx.SpatialCoordinateContract.si(),
        geometry=geometry,
    )
    assert result.audit.validity.all_certified
    assert result.quality.minimum_measure == pytest.approx(0.25, abs=1.0e-14)
    accepted = result.geometry.elements[0]
    assert isinstance(accepted, RestrictedCellGeometryElement)
    values, _ = accepted.tabulate(jnp.asarray([[0.3, 0.7]], dtype=jnp.float64))
    route = np.asarray(result.geometry.geometry_dofs[0])[0]
    point = np.asarray(values) @ np.asarray(result.geometry.coordinates)[route]
    np.testing.assert_allclose(point, [[0.15, 0.35]], rtol=0.0, atol=1.0e-14)


def test_plc_association_preserves_wide_source_stratum_identity() -> None:
    index = 2**40 + 7
    association = phx.meshing.GeometryAssociation(
        phx.meshing.GeometryAssociationKind.PIECEWISE_LINEAR,
        "material-source",
        "epoch",
        "target-cells",
        np.asarray([10], dtype=np.int64),
        (f"epoch:region:{index}",),
        np.zeros(1, dtype=np.float64),
        exact=True,
        source_dimensions=np.asarray([3], dtype=np.int8),
        source_indices=np.asarray([index], dtype=np.int64),
        source_entity_roles=(phx.meshing.GeometrySourceEntityRole.REGION,),
    )
    represented_source = f"epoch:region:{int(association.source_indices[0])}"
    assert represented_source == association.source_entity_ids[0]
    assert association.complete


@pytest.mark.parametrize(
    "source_entity",
    ["other:region:7", "epoch:facet:7", "epoch:region:8"],
    ids=["stale-revision", "wrong-stratum", "wrong-definition"],
)
def test_plc_association_rejects_mismatched_explicit_source_identity(
    source_entity: str,
) -> None:
    with pytest.raises(ValueError):
        phx.meshing.GeometryAssociation(
            phx.meshing.GeometryAssociationKind.PIECEWISE_LINEAR,
            "material-source",
            "epoch",
            "target-cells",
            np.asarray([10], dtype=np.int64),
            (source_entity,),
            np.zeros(1, dtype=np.float64),
            exact=True,
            source_dimensions=np.asarray([3], dtype=np.int8),
            source_indices=np.asarray([7], dtype=np.int64),
            source_entity_roles=(phx.meshing.GeometrySourceEntityRole.REGION,),
        )


@pytest.mark.parametrize(
    ("dimension", "index", "entity"),
    [(3, 0, "implicit-zero-set"), (2, 1, "implicit-zero-set"), (2, 0, "other-surface")],
    ids=["bulk-is-not-zero-set", "undeclared-zero-set", "wrong-source-entity"],
)
def test_implicit_association_rejects_undeclared_source_strata(
    dimension: int,
    index: int,
    entity: str,
) -> None:
    with pytest.raises(ValueError):
        phx.meshing.GeometryAssociation(
            phx.meshing.GeometryAssociationKind.IMPLICIT,
            "field-source",
            "epoch",
            "target-faces",
            np.asarray([10], dtype=np.int64),
            (entity,),
            np.asarray([0.01], dtype=np.float64),
            source_dimensions=np.asarray([dimension], dtype=np.int32),
            source_indices=np.asarray([index], dtype=np.int32),
        )


def test_equal_source_dimensions_do_not_conflate_region_and_facet_identity() -> None:
    kind = phx.meshing.GeometryAssociationKind.PIECEWISE_LINEAR
    roles = phx.meshing.GeometrySourceEntityRole
    region = phx.meshing.GeometryAssociation(
        kind,
        "source",
        "epoch",
        "target",
        np.asarray([10], dtype=np.int64),
        ("epoch:region:42",),
        np.zeros(1, dtype=np.float64),
        exact=True,
        source_dimensions=np.asarray([2], dtype=np.int8),
        source_indices=np.asarray([42], dtype=np.int64),
        source_entity_roles=(roles.REGION,),
    )
    facet = phx.meshing.GeometryAssociation(
        kind,
        "source",
        "epoch",
        "target",
        np.asarray([10], dtype=np.int64),
        ("epoch:facet:42",),
        np.zeros(1, dtype=np.float64),
        exact=True,
        source_dimensions=np.asarray([2], dtype=np.int8),
        source_indices=np.asarray([42], dtype=np.int64),
        source_entity_roles=(roles.FACET,),
    )
    assert region.association_id != facet.association_id
    assert region.source_entity_ids != facet.source_entity_ids
    with pytest.raises(ValueError):
        phx.meshing.GeometryAssociation(
            kind,
            "source",
            "epoch",
            "target",
            np.asarray([10], dtype=np.int64),
            ("epoch:region:42",),
            np.zeros(1, dtype=np.float64),
            exact=True,
            source_dimensions=np.asarray([2], dtype=np.int8),
            source_indices=np.asarray([42], dtype=np.int64),
            source_entity_roles=(roles.FACET,),
        )


@pytest.mark.parametrize(
    ("dimension", "index"),
    [(1, 42), (2, 43)],
    ids=["different-source-stratum", "different-definition-index"],
)
def test_surface_association_identity_binds_explicit_source_metadata(
    dimension: int,
    index: int,
) -> None:
    arguments = (
        phx.meshing.GeometryAssociationKind.SURFACE,
        "authored-surface",
        "epoch",
        "target",
        np.asarray([10], dtype=np.int64),
        ("custom-source-entity",),
        np.zeros(1, dtype=np.float64),
    )
    original = phx.meshing.GeometryAssociation(
        *arguments,
        source_dimensions=np.asarray([2], dtype=np.int8),
        source_indices=np.asarray([42], dtype=np.int64),
        source_occurrence_paths=(("assembly", "instance"),),
    )
    changed = phx.meshing.GeometryAssociation(
        *arguments,
        source_dimensions=np.asarray([dimension], dtype=np.int8),
        source_indices=np.asarray([index], dtype=np.int64),
        source_occurrence_paths=(("assembly", "instance"),),
    )
    assert original.association_id != changed.association_id


def test_surface_association_refuses_volume_stratum_metadata() -> None:
    with pytest.raises(ValueError):
        phx.meshing.GeometryAssociation(
            phx.meshing.GeometryAssociationKind.SURFACE,
            "authored-surface",
            "epoch",
            "target",
            np.asarray([10], dtype=np.int64),
            ("custom-source-region",),
            np.zeros(1, dtype=np.float64),
            source_dimensions=np.asarray([3], dtype=np.int8),
            source_indices=np.asarray([42], dtype=np.int64),
        )


def _curved_chord_tetrahedron(
    sign: float,
) -> tuple[phx.discretization.CellMesh, phx.discretization.CellGeometrySpec]:
    from phydrax.discretization._cell_geometry import coordinate_lagrange_element

    element = coordinate_lagrange_element("tetrahedron", 2)

    def physical(reference: np.ndarray) -> np.ndarray:
        x = reference[:, 0] + 0.25 * reference[:, 2]
        y = reference[:, 1] + 0.25 * reference[:, 2]
        return np.column_stack((x, y, sign * (0.1 * reference[:, 2] + y * y)))

    corners = np.asarray(
        phx.discretization.reference_cell_topology("tetrahedron").vertices,
        dtype=np.float64,
    )
    mesh = phx.discretization.CellMesh(
        physical(corners),
        (
            phx.discretization.CellBlock(
                "curved",
                "tetrahedron",
                np.arange(4, dtype=np.int32)[None],
                global_ids=np.asarray([701], dtype=np.int64),
            ),
        ),
    )
    geometry = phx.discretization.CellGeometrySpec(
        {"curved": element},
        {"curved": np.arange(element.local_dof_count, dtype=np.int32)[None]},
        physical(np.asarray(element.reference_nodes, dtype=np.float64)),
    )
    return mesh, geometry


@pytest.mark.parametrize(
    "sign",
    (1.0, -1.0),
    ids=("positive-source-inverted-chord", "negative-source-positive-chord"),
)
def test_mapped_quality_audits_the_source_map_not_its_corner_chords(sign: float) -> None:
    mesh, geometry = _curved_chord_tetrahedron(sign)
    chord = phx.meshing.evaluate_cell_quality(mesh)
    mapped = phx.meshing.evaluate_cell_quality(mesh, geometry=geometry)
    assert float(chord.measures[0]) * sign < 0.0
    assert bool(chord.sampled_valid[0]) == (sign < 0.0)
    np.testing.assert_allclose(mapped.measures, [sign / 60.0], rtol=1e-12, atol=1e-14)
    assert bool(mapped.sampled_valid[0]) == (sign > 0.0)
    audit = phx.meshing.audit_cell_mesh(mesh, geometry, mapped)
    assert audit.passed == (sign > 0.0)
    assert audit.validity.all_certified == (sign > 0.0)
    if sign > 0.0:
        assert float(mapped.mean_ratios[0]) > 0.0
        assert float(mapped.scaled_jacobian[0]) > 0.0
        assert float(mapped.radius_ratios[0]) > 0.0
        assert float(mapped.sliver_measures[0]) > 0.0
        assert float(mapped.warpage[0]) > 0.0
        assert np.isfinite(float(mapped.condition_number[0]))
    else:
        assert "minimum_measure" in audit.issues
        assert "invalid_geometry" in audit.issues


def test_mapped_quality_keeps_dynamic_source_metric_and_volume_jvp() -> None:
    import equinox as eqx
    from jax import Array

    from phydrax.discretization.fem import FiniteElementSpec
    from phydrax.meshing._quality import CellQualityEvaluation

    mesh, geometry = _curved_chord_tetrahedron(1.0)
    scope = phx.meshing.MeshingScope(
        mesh.mesh_id,
        mesh.numeric_version,
        phx.meshing.MeshingEntityKind.MESH,
        0,
        mesh.entity_set(0).entity_set_id,
        mesh.vertex_global_ids,
    )
    metric = phx.meshing.MeshMetricField(
        scope,
        np.broadcast_to(np.eye(3, dtype=np.float64), (4, 3, 3)),
        minimum_size=0.01,
        maximum_size=100.0,
    )

    def evaluate(source: phx.discretization.CellGeometrySpec) -> CellQualityEvaluation:
        return phx.meshing.evaluate_cell_quality(mesh, geometry=source, metric=metric)

    eager = evaluate(geometry)
    assert phx.meshing.audit_cell_mesh(mesh, geometry, eager).passed
    compiled = eqx.filter_jit(evaluate)(geometry)
    np.testing.assert_allclose(compiled.measures, eager.measures, rtol=1e-12, atol=1e-14)
    np.testing.assert_allclose(
        compiled.mean_ratios, eager.mean_ratios, rtol=1e-12, atol=1e-14
    )
    np.testing.assert_allclose(
        eager.metric_quality, eager.mean_ratios, rtol=1e-12, atol=1e-14
    )
    reference = geometry.elements[0]
    if not isinstance(reference, FiniteElementSpec):
        raise AssertionError(
            "The independently authored source must retain its nodal tetrahedral basis."
        )
    nodes = np.asarray(reference.reference_nodes, dtype=np.float64)
    direction = jnp.zeros_like(geometry.coordinates).at[:, 2].set(nodes[:, 2])

    def volume(bank: Array) -> Array:
        return evaluate(
            eqx.tree_at(lambda source: source.coordinates, geometry, bank)
        ).measures[0]

    _, tangent = jax.jvp(volume, (geometry.coordinates,), (direction,))
    np.testing.assert_allclose(tangent, 1.0 / 6.0, rtol=1e-12, atol=1e-14)


def test_quality_refuses_two_competing_coordinate_authorities() -> None:
    mesh, geometry = _curved_chord_tetrahedron(1.0)
    with pytest.raises(ValueError, match="one owning geometry"):
        phx.meshing.evaluate_cell_quality(mesh, mesh.coordinates, geometry=geometry)


@pytest.mark.parametrize(
    "change", ("outputs", "scope"), ids=("forged-metric-output", "foreign-metric-owner")
)
def test_mapped_audit_refuses_forged_or_foreign_metric_quality(change: str) -> None:
    import equinox as eqx

    mesh, geometry = _curved_chord_tetrahedron(1.0)
    scope = phx.meshing.MeshingScope(
        mesh.mesh_id,
        mesh.numeric_version,
        phx.meshing.MeshingEntityKind.MESH,
        0,
        mesh.entity_set(0).entity_set_id,
        mesh.vertex_global_ids,
    )
    metric = phx.meshing.MeshMetricField(
        scope,
        np.broadcast_to(np.eye(3, dtype=np.float64), (4, 3, 3)),
        minimum_size=0.01,
        maximum_size=100.0,
    )
    quality = phx.meshing.evaluate_cell_quality(mesh, geometry=geometry, metric=metric)
    if change == "outputs":
        forged = eqx.tree_at(
            lambda value: value.metric_quality, quality, quality.metric_quality + 0.01
        )
        audit = phx.meshing.audit_cell_mesh(mesh, geometry, forged)
        assert not audit.passed and "quality_binding" in audit.issues
    else:
        foreign = phx.meshing.MeshingScope(
            "foreign-mesh-authority",
            scope.source_revision,
            scope.entity_kind,
            scope.entity_dimension,
            scope.entity_set_id,
            scope.entity_ids,
        )
        owner = phx.meshing.MeshMetricField(
            foreign,
            metric.values,
            minimum_size=metric.minimum_size,
            maximum_size=metric.maximum_size,
        )
        forged = eqx.tree_at(lambda value: value.metric, quality, owner)
        with pytest.raises(ValueError, match="exact owning mesh vertex scope"):
            phx.meshing.audit_cell_mesh(mesh, geometry, forged)

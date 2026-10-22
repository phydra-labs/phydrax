from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

import phydrax as phx
from phydrax.discretization._coordinate_budget import CoordinateEnclosureBudget
from phydrax.geometry.surface import SurfaceMetadata, SurfaceModel
from tests._support.audit_fixtures import (
    cell_label,
    quadratic_triangle_geometry,
    quadratic_triangle_with_edge_nodes,
    surface_cell_association,
    unordered_reference_cells,
)
from tests._support.mesh_certificate_fixtures import (
    coordinate_budget as coordinate_budget,
    reference_budget as reference_budget,
)
from tests._support.quality_fixtures import (
    curved_quality_tetrahedron,
    reference_quality_mesh,
)


# Empirical `test_restricted_quadrilateral_publication_checks_actual_corner_binding`: matched max |delta|=0; joint allowance fraction=0, margin=100%.
RESTRICTED_QUADRILATERAL_AREA_ATOL = 1e-14
# Empirical `test_restricted_quadrilateral_publication_checks_actual_corner_binding`: matched max |delta|=0; joint allowance fraction=0, margin=100%.
RESTRICTED_QUADRILATERAL_POINT_ATOL = 1e-14
# Empirical `test_mapped_quality_audits_the_source_map_not_its_corner_chords`: matched max |delta|=3.46945e-18; joint allowance fraction=0.000130104, margin=99.987%.
MAPPED_TETRAHEDRON_VOLUME_RTOL = 1e-12
# Empirical `test_mapped_quality_audits_the_source_map_not_its_corner_chords`: matched max |delta|=3.46945e-18; joint allowance fraction=0.000130104, margin=99.987%.
MAPPED_TETRAHEDRON_VOLUME_ATOL = 1e-14
# Empirical `test_mapped_quality_keeps_dynamic_source_metric_and_volume_jvp`: matched max |delta|=0; joint allowance fraction=0, margin=100%.
MAPPED_MEAN_RATIO_RTOL = 1e-12
# Empirical `test_mapped_quality_keeps_dynamic_source_metric_and_volume_jvp`: matched max |delta|=0; joint allowance fraction=0, margin=100%.
MAPPED_MEAN_RATIO_ATOL = 1e-14
# Empirical `test_mapped_quality_keeps_dynamic_source_metric_and_volume_jvp`: matched max |delta|=5.55112e-17; joint allowance fraction=0.000314214, margin=99.9686%.
MAPPED_VOLUME_JVP_RTOL = 1e-12
# Empirical `test_mapped_quality_keeps_dynamic_source_metric_and_volume_jvp`: matched max |delta|=5.55112e-17; joint allowance fraction=0.000314214, margin=99.9686%.
MAPPED_VOLUME_JVP_ATOL = 1e-14


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
def test_reference_cell_quality_certification_preserves_sampled_evidence(
    cell_kind: str,
) -> None:
    mesh = reference_quality_mesh(cell_kind)
    result = phx.meshing.certify_cell_mesh(mesh, phx.SpatialCoordinateContract.si())
    assert result.audit.passed
    assert result.quality.minimum_measure > 0.0
    assert result.quality.minimum_mean_ratio > 0.0
    assert result.quality.maximum_aspect_ratio >= 1.0


@pytest.mark.parametrize("case", ("reference", "skewed", "random"))
@pytest.mark.parametrize("kind", ("interval", "triangle", "tetrahedron"))
def test_affine_quality_preserves_metrics_with_admitted_differential_reuse(
    kind: str,
    case: str,
    reference_budget: CoordinateEnclosureBudget,
) -> None:
    import phydrax.meshing._quality as owner

    mesh = reference_quality_mesh(kind)
    if case == "random":
        corners = mesh.blocks[0].vertices.shape[1]
        points = np.random.default_rng(193).uniform(
            -2, 2, size=(17 * corners, mesh.ambient_dimension)
        )
        mesh = phx.discretization.CellMesh(
            points,
            (
                phx.discretization.CellBlock(
                    "cells", kind, np.arange(17 * corners).reshape(17, corners)
                ),
            ),
        )
    elif case == "skewed":
        dimension = mesh.ambient_dimension
        matrix = np.diag(1 + 0.17 * np.arange(1, dimension + 1))
        if dimension > 1:
            matrix[0, -1] += 0.23
        mesh = mesh.with_coordinates(
            np.asarray(mesh.coordinates) @ matrix.T + np.linspace(0.11, 0.31, dimension),
            numeric_version="skew-affine-test",
        )
    geometry = phx.discretization.CellGeometrySpec.affine(mesh)
    elements, routes, coordinates = geometry.resolve(mesh, budget=reference_budget)
    source = elements[0]
    values = coordinates[routes[0]]
    rule = owner._MappedQualityPlan(
        owner._mapped_quality_rule(kind, source),
        owner._boundary_rule(kind),
        owner._face_rules(kind, mesh.ambient_dimension),
    )
    before = reference_budget.ticket.geometry_queries
    actual = owner._mapped_block_quality(
        kind, source, values, None, budget=reference_budget, rule=rule
    )
    # One earned differential also feeds canonical P1 tetrahedral face probes;
    # their original normal arithmetic is retained without re-querying the map.
    expected_queries = values.shape[0] * (
        rule.query_count if rule.faces and kind != "tetrahedron" else 1
    )
    assert reference_budget.ticket.geometry_queries - before == expected_queries
    measured = owner._mapped_sample_worksets(kind, source, values, None, rule.volume)
    minimum, maximum = owner._mapped_boundary_angles(
        rule, source, values, budget=reference_budget
    )
    warpage = owner._mapped_face_warpage(
        rule, source, values, budget=reference_budget
    )
    expected = owner._finish_mapped_quality(measured, minimum, maximum, warpage)
    for observed, original in zip(
        jax.tree_util.tree_leaves(actual), jax.tree_util.tree_leaves(expected), strict=True
    ):
        np.testing.assert_array_equal(observed, original)


@pytest.mark.parametrize("cell_kind", ("triangle", "tetrahedron", "hexahedron", "prism"))
def test_canonical_cell_order_preserves_entity_identities(cell_kind: str) -> None:
    mesh = unordered_reference_cells(cell_kind)
    canonical = phx.meshing.canonicalize_cell_mesh(mesh)

    assert np.array_equal(canonical.blocks[0].global_ids, [10, 20])
    for dimension in range(mesh.topological_dimension + 1):
        assert _entity_ids_by_vertices(canonical, dimension) == _entity_ids_by_vertices(
            mesh, dimension
        )
    assert phx.meshing.canonicalize_cell_mesh(canonical) is canonical


def test_canonicalization_rejects_preexisting_organization_binding() -> None:
    mesh = unordered_reference_cells()
    label = cell_label(mesh, [20])
    with pytest.raises(phx.meshing.MeshingFailure, match="organization_"):
        phx.meshing.certify_cell_mesh(
            mesh,
            phx.SpatialCoordinateContract.si(),
            labels=(label,),
        )


def test_triangle_quality_is_fixed_topology_differentiable_and_rejects_inversion() -> (
    None
):
    mesh = reference_quality_mesh("triangle")

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


def test_canonicalization_rejects_reordering_supplied_geometry() -> None:
    mesh = unordered_reference_cells()
    geometry = quadratic_triangle_geometry(mesh)

    with pytest.raises(ValueError, match="reorder supplied geometry"):
        phx.meshing.certify_cell_mesh(
            mesh,
            phx.SpatialCoordinateContract.si(),
            geometry=geometry,
        )


@pytest.mark.parametrize(
    ("ids", "source_id"),
    [
        pytest.param([999], None, id="foreign-cell"),
        pytest.param([0], "old-mesh", id="stale-source"),
    ],
)
def test_audit_rejects_stale_cell_labels(ids: list[int], source_id: str | None) -> None:
    mesh = reference_quality_mesh("triangle")
    label = cell_label(mesh, ids, source_id=source_id)
    with pytest.raises(phx.meshing.MeshingFailure, match="organization_"):
        phx.meshing.certify_cell_mesh(
            mesh, phx.SpatialCoordinateContract.si(), labels=(label,)
        )


def test_mapped_audit_scope_and_certification_validity_agree() -> None:
    mesh = phx.meshing.canonicalize_cell_mesh(unordered_reference_cells())
    geometry = quadratic_triangle_geometry(mesh)
    audit = phx.meshing.audit_cell_mesh(mesh, geometry)

    # A mapped coordinate element is audited over its mapped cells: sampled
    # quality is mapped evidence, not corner evidence.
    assert audit.quality_scope == "mapped_coordinate_cells"
    assert "sampled_quality" in audit.mapped_checks
    assert "sampled_quality" not in audit.corner_checks
    assert audit.validity.all_certified
    assert audit.passed
    result = phx.meshing.certify_cell_mesh(
        mesh, phx.SpatialCoordinateContract.si(), geometry=geometry
    )
    assert result.audit.validity.certificate_id == audit.validity.certificate_id


def test_audit_rejects_quadratic_interior_inversion_with_valid_corners() -> None:
    mesh = reference_quality_mesh("triangle")
    # The Jacobian determinant is positive at all six Lagrange nodes and at the
    # corners, but negative inside the cell.
    geometry = quadratic_triangle_with_edge_nodes(
        ((0.26, 0.23), (0.58, 0.77), (-0.1, 0.06))
    )
    certificate = phx.discretization.certify_cell_geometry_validity(geometry, mesh=mesh)

    assert bool(phx.meshing.evaluate_cell_quality(mesh).sampled_valid[0])
    assert int(certificate.status[0]) == phx.discretization.CellValidityStatus.INVALID
    assert float(certificate.determinant_lower[0]) < 0.0
    assert int(certificate.depth[0]) > 0
    with pytest.raises(phx.meshing.MeshingFailure, match="invalid_geometry"):
        phx.meshing.certify_cell_mesh(
            mesh, phx.SpatialCoordinateContract.si(), geometry=geometry
        )


def test_audit_subdivision_capacity_obeys_unresolved_disposition() -> None:
    mesh = reference_quality_mesh("triangle")
    # Valid everywhere, but one Bernstein edge coefficient is negative.
    geometry = quadratic_triangle_with_edge_nodes(
        ((0.86, -0.17), (0.54, 0.29), (-0.06, 0.51))
    )
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


def test_welded_vertices_obey_reject_and_record_dispositions() -> None:
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


def test_audit_detects_nonmanifold_vertex_stars() -> None:
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


def test_audit_self_intersection_skip_excludes_evaluation_and_counts() -> None:
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
    # A bow-tie: two triangles meeting only at vertex zero. A fully incident star there
    # is a nonmanifold vertex; the same star with an incident cell cut at
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
    storage = phx.discretization.CellMeshStorage(
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
    evidence = phx.meshing.audit_cell_mesh(
        mesh,
        phx.discretization.CellGeometrySpec.affine(mesh),
        policy=phx.meshing.CellMeshAuditPolicy(
            coincident_vertices=phx.meshing.CellMeshAuditDisposition.SKIP,
            maximum_coincidence_candidates=64,
            maximum_intersection_candidates=64,
            self_intersection=phx.meshing.CellMeshAuditDisposition.SKIP,
        ),
    )
    assert dict(evidence.check_counts)["nonmanifold_vertices"] == nonmanifold


def test_notched_polygon_self_intersection_keeps_capacity_refusal() -> None:
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


@pytest.mark.parametrize(
    ("ids", "entity_set_id"),
    [
        pytest.param([999], None, id="foreign-cell"),
        pytest.param([0], "stale-entity-set", id="foreign-set"),
    ],
)
def test_audit_rejects_foreign_association_targets(
    ids: list[int], entity_set_id: str | None
) -> None:
    mesh = reference_quality_mesh("triangle")
    association = surface_cell_association(mesh, ids, entity_set_id=entity_set_id)
    assert association.complete
    with pytest.raises(phx.meshing.MeshingFailure, match="association_"):
        phx.meshing.certify_cell_mesh(
            mesh, phx.SpatialCoordinateContract.si(), associations=(association,)
        )


def test_complete_association_policy_requires_joint_cell_coverage() -> None:
    mesh = phx.meshing.canonicalize_cell_mesh(unordered_reference_cells())
    first = surface_cell_association(mesh, [10], residual=100.0)
    second = surface_cell_association(mesh, [20], residual=100.0)
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


def test_audit_rejects_conflicting_source_association_identity() -> None:
    mesh = reference_quality_mesh("triangle")
    first = surface_cell_association(mesh, [0])
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


def test_audit_rejects_quality_from_different_coordinates() -> None:
    mesh = reference_quality_mesh("triangle")
    quality = phx.meshing.evaluate_cell_quality(mesh, 2.0 * mesh.coordinates)
    audit = phx.meshing.audit_cell_mesh(
        mesh,
        phx.discretization.CellGeometrySpec.affine(mesh),
        quality,
    )
    assert not audit.passed
    assert "quality_binding" in audit.issues


def test_audit_rejects_geometry_with_reordered_corner_dofs() -> None:
    mesh = phx.meshing.canonicalize_cell_mesh(unordered_reference_cells())
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


def test_result_rejects_changed_geometry_values_with_the_same_layout() -> None:
    result = phx.meshing.certify_cell_mesh(
        reference_quality_mesh("triangle"), phx.SpatialCoordinateContract.si()
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


def test_result_rejects_organization_that_differs_from_audited_evidence() -> None:
    result = phx.meshing.certify_cell_mesh(
        reference_quality_mesh("triangle"), phx.SpatialCoordinateContract.si()
    )
    with pytest.raises(ValueError, match="audited evidence"):
        _rebuild_result(result, labels=(cell_label(result.mesh, [0]),))


def _rebuild_result(
    result: phx.meshing.CellMeshingResult, **changes: Any
) -> phx.meshing.CellMeshingResult:
    return phx.meshing.CellMeshingResult(
        changes.pop("mesh", result.mesh),
        changes.pop("geometry", result.geometry),
        changes.pop("coordinate_contract", result.coordinate_contract),
        changes.pop("audit", result.audit),
        changes.pop("quality", result.quality),
        changes.pop("compliance", result.compliance),
        changes.pop("trace", result.trace),
        changes.pop("provider", result.provider),
        changes.pop("runtime", result.runtime),
        changes.pop("derivative_mode", result.derivative_mode),
        changes.pop("provenance", result.provenance),
        **changes,
    )


def test_result_reconstruction_preserves_identity_and_shared_device_fields() -> None:
    result = phx.meshing.certify_cell_mesh(
        reference_quality_mesh("triangle"), phx.SpatialCoordinateContract.si()
    )
    rebuilt = _rebuild_result(result)

    assert rebuilt.result_id == result.result_id
    assert jax.tree_util.tree_structure(rebuilt) == jax.tree_util.tree_structure(result)
    assert rebuilt.mesh is result.mesh
    assert rebuilt.geometry is result.geometry
    assert rebuilt.audit is result.audit
    assert rebuilt.quality is result.quality
    assert rebuilt.compliance is result.compliance
    assert rebuilt.trace is result.trace
    assert rebuilt.runtime is result.runtime
    assert rebuilt.mesh.coordinates is result.mesh.coordinates
    assert rebuilt.mesh.coordinates.sharding == result.mesh.coordinates.sharding


def test_result_checked_boundary_refuses_an_invalid_audit_kind() -> None:
    result = phx.meshing.certify_cell_mesh(
        reference_quality_mesh("triangle"), phx.SpatialCoordinateContract.si()
    )
    with pytest.raises(TypeError, match="audit"):
        phx.meshing.CellMeshingResult(
            result.mesh,
            result.geometry,
            result.coordinate_contract,
            None,  # ty: ignore[invalid-argument-type]
            result.quality,
            result.compliance,
            result.trace,
            result.provider,
            result.runtime,
            result.derivative_mode,
            result.provenance,
        )


def test_result_refuses_failed_compliance_before_organization_validation() -> None:
    result = phx.meshing.certify_cell_mesh(
        reference_quality_mesh("triangle"), phx.SpatialCoordinateContract.si()
    )
    failed = phx.meshing.MeshingComplianceReport(
        result.compliance.specification_id, issues=("requested_quality",)
    )
    with pytest.raises(ValueError, match="result.compliance"):
        _rebuild_result(
            result,
            compliance=failed,
            labels=(cell_label(result.mesh, [0]),),
        )


@pytest.mark.parametrize(
    "field",
    (
        "quality",
        "compliance",
        "trace",
        "runtime",
        "certification",
        "collective_evidence",
        "execution_evidence",
        "surface_source",
    ),
)
def test_result_checked_boundary_refuses_invalid_evidence_kinds(field: str) -> None:
    result = phx.meshing.certify_cell_mesh(
        reference_quality_mesh("triangle"), phx.SpatialCoordinateContract.si()
    )
    with pytest.raises(TypeError, match=field):
        _rebuild_result(result, **{field: object()})


def test_result_refuses_an_invalid_derivative_selector_kind() -> None:
    result = phx.meshing.certify_cell_mesh(
        reference_quality_mesh("triangle"), phx.SpatialCoordinateContract.si()
    )
    with pytest.raises(TypeError, match="result.derivative_mode"):
        phx.meshing.CellMeshingResult(
            result.mesh,
            result.geometry,
            result.coordinate_contract,
            result.audit,
            result.quality,
            result.compliance,
            result.trace,
            result.provider,
            result.runtime,
            "fixed_topology",  # ty: ignore[invalid-argument-type]
            result.provenance,
        )


def test_stored_result_reconstruction_preserves_publication_and_device_identity() -> None:
    result = phx.meshing.certify_cell_mesh(
        reference_quality_mesh("triangle"), phx.SpatialCoordinateContract.si()
    )
    rebuilt = phx.meshing.CellMeshingResult.from_stored_fields(result)

    assert rebuilt.result_id == result.result_id
    assert jax.tree_util.tree_structure(rebuilt) == jax.tree_util.tree_structure(result)
    assert rebuilt.mesh is result.mesh
    assert rebuilt.geometry is result.geometry
    assert rebuilt.audit is result.audit
    assert rebuilt.quality is result.quality
    assert rebuilt.mesh.coordinates is result.mesh.coordinates
    assert rebuilt.mesh.coordinates.sharding == result.mesh.coordinates.sharding


def test_stored_result_reconstruction_refuses_a_changed_scientific_identity() -> None:
    result = phx.meshing.certify_cell_mesh(
        reference_quality_mesh("triangle"), phx.SpatialCoordinateContract.si()
    )
    changed = type(result.provenance)({"contract": "changed-publication-provenance"})
    retained = eqx.tree_at(lambda value: value.provenance, result, changed)

    assert retained.provenance.semantic_id != result.provenance.semantic_id
    assert retained.result_id == result.result_id
    with pytest.raises(ValueError, match="CellMeshingResult.result_id"):
        phx.meshing.CellMeshingResult.from_stored_fields(retained)


def test_result_boundary_must_cover_the_actual_mesh_faces_and_coordinates() -> None:
    result = phx.meshing.certify_cell_mesh(
        reference_quality_mesh("tetrahedron"), phx.SpatialCoordinateContract.si()
    )
    mesh = result.mesh
    # ty: ignore[unresolved-attribute]
    faces = np.asarray(mesh.connectivity.faces)
    metadata = SurfaceMetadata(
        source_id="boundary",
        source_revision="r1",
        coordinate_contract=result.coordinate_contract,
        provenance=("qualification",),
    )

    def boundary(points: Any, rows: Any) -> Any:
        return SurfaceModel.from_triangles(
            points,
            rows,
            metadata,
            vertex_global_ids=mesh.vertex_global_ids,
            repair_orientation=True,
        )

    valid = _rebuild_result(result, boundary=boundary(mesh.coordinates, faces))
    assert valid.boundary is not None
    assert valid.boundary.mesh.blocks[0].cell_count == 4
    with pytest.raises(ValueError, match="boundary_coordinates"):
        _rebuild_result(
            result, boundary=boundary(np.asarray(mesh.coordinates) + 1.0, faces)
        )
    with pytest.raises(ValueError, match="boundary_coverage"):
        _rebuild_result(result, boundary=boundary(mesh.coordinates, faces[:-1]))


def test_audit_separates_corner_and_mapped_evidence_with_failing_cells() -> None:
    mesh = reference_quality_mesh("triangle")
    geometry = quadratic_triangle_with_edge_nodes(
        ((0.26, 0.23), (0.58, 0.77), (-0.1, 0.06))
    )
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
    mesh = reference_quality_mesh("triangle")
    geometry = quadratic_triangle_with_edge_nodes(
        ((0.86, -0.17), (0.54, 0.29), (-0.06, 0.51))
    )
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
    assert result.quality.minimum_measure == pytest.approx(
        0.25, abs=RESTRICTED_QUADRILATERAL_AREA_ATOL
    )
    accepted = result.geometry.elements[0]
    assert isinstance(accepted, RestrictedCellGeometryElement)
    values, _ = accepted.tabulate(jnp.asarray([[0.3, 0.7]], dtype=jnp.float64))
    route = np.asarray(result.geometry.geometry_dofs[0])[0]
    point = np.asarray(values) @ np.asarray(result.geometry.coordinates)[route]
    np.testing.assert_allclose(
        point, [[0.15, 0.35]], rtol=0.0, atol=RESTRICTED_QUADRILATERAL_POINT_ATOL
    )


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
    assert represented_source == association.source_entities.entity_id(0)
    assert association.complete


@pytest.mark.parametrize(
    "source_entity",
    ["other:region:7", "epoch:facet:7", "epoch:region:8"],
    ids=["stale-revision", "wrong-stratum", "wrong-definition"],
)
def test_plc_association_rejects_mismatched_explicit_source_identity(
    source_entity: str,
) -> None:
    with pytest.raises(ValueError, match=r"GeometryAssociation\.source_entity_ids"):
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
    with pytest.raises(
        ValueError, match=r"GeometryAssociation\.(implicit_strata|source_entity_ids)"
    ):
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
    assert region.source_entities.entity_ids() != facet.source_entities.entity_ids()
    with pytest.raises(ValueError, match=r"GeometryAssociation\.source_entity_ids"):
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
    with pytest.raises(ValueError, match=r"GeometryAssociation\.surface_dimensions"):
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


@pytest.mark.parametrize(
    "sign",
    (1.0, -1.0),
    ids=("positive-source-inverted-chord", "negative-source-positive-chord"),
)
def test_mapped_quality_audits_the_source_map_not_its_corner_chords(sign: float) -> None:
    mesh, geometry = curved_quality_tetrahedron(sign)
    chord = phx.meshing.evaluate_cell_quality(mesh)
    mapped = phx.meshing.evaluate_cell_quality(mesh, geometry=geometry)
    assert float(chord.measures[0]) * sign < 0.0
    assert bool(chord.sampled_valid[0]) == (sign < 0.0)
    np.testing.assert_allclose(
        mapped.measures,
        [sign / 60.0],
        rtol=MAPPED_TETRAHEDRON_VOLUME_RTOL,
        atol=MAPPED_TETRAHEDRON_VOLUME_ATOL,
    )
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

    mesh, geometry = curved_quality_tetrahedron(1.0)
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
    np.testing.assert_allclose(
        compiled.measures,
        eager.measures,
        rtol=MAPPED_TETRAHEDRON_VOLUME_RTOL,
        atol=MAPPED_TETRAHEDRON_VOLUME_ATOL,
    )
    np.testing.assert_allclose(
        compiled.mean_ratios,
        eager.mean_ratios,
        rtol=MAPPED_MEAN_RATIO_RTOL,
        atol=MAPPED_MEAN_RATIO_ATOL,
    )
    np.testing.assert_allclose(
        eager.metric_quality,
        eager.mean_ratios,
        rtol=MAPPED_MEAN_RATIO_RTOL,
        atol=MAPPED_MEAN_RATIO_ATOL,
    )
    reference = geometry.elements[0]
    if not isinstance(reference, FiniteElementSpec):
        raise AssertionError(
            "fixture.reference must retain its nodal tetrahedral basis; got incompatible element"
        )
    nodes = np.asarray(reference.reference_nodes, dtype=np.float64)
    direction = jnp.zeros_like(geometry.coordinates).at[:, 2].set(nodes[:, 2])

    def volume(bank: Array) -> Array:
        return evaluate(
            eqx.tree_at(lambda source: source.coordinates, geometry, bank)
        ).measures[0]

    _, tangent = jax.jvp(volume, (geometry.coordinates,), (direction,))
    np.testing.assert_allclose(
        tangent, 1.0 / 6.0, rtol=MAPPED_VOLUME_JVP_RTOL, atol=MAPPED_VOLUME_JVP_ATOL
    )


def test_quality_refuses_two_competing_coordinate_authorities() -> None:
    mesh, geometry = curved_quality_tetrahedron(1.0)
    with pytest.raises(ValueError, match="quality.geometry"):
        phx.meshing.evaluate_cell_quality(mesh, mesh.coordinates, geometry=geometry)


@pytest.mark.parametrize(
    "change", ("outputs", "scope"), ids=("forged-metric-output", "foreign-metric-owner")
)
def test_mapped_audit_refuses_forged_or_foreign_metric_quality(change: str) -> None:
    import equinox as eqx

    mesh, geometry = curved_quality_tetrahedron(1.0)
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
        with pytest.raises(ValueError, match="quality.metric.scope"):
            phx.meshing.audit_cell_mesh(mesh, geometry, forged)


@pytest.mark.parametrize(
    "field",
    (
        "unused_entities",
        "coincident_vertices",
        "duplicate_cells",
        "nonmanifold",
        "inconsistent_orientation",
        "watertight_boundary",
        "self_intersection",
        "invalid_geometry",
        "unresolved",
    ),
)
def test_audit_policy_parses_each_disposition_without_changing_identity(
    field: str,
) -> None:
    disposition = phx.meshing.CellMeshAuditDisposition.RECORD
    arguments: dict[str, Any] = {field: disposition}
    policy = phx.meshing.CellMeshAuditPolicy(**arguments)
    assert getattr(policy, field) is disposition
    assert policy.policy_id == phx.meshing.CellMeshAuditPolicy(**arguments).policy_id
    assert policy.policy_id != phx.meshing.CellMeshAuditPolicy().policy_id
    invalid_arguments: dict[str, Any] = {field: disposition.value}
    with pytest.raises(TypeError, match=field):
        phx.meshing.CellMeshAuditPolicy(**invalid_arguments)


@pytest.mark.parametrize("field", ("policy", "prepared_validity", "boundary"))
def test_audit_checked_boundary_refuses_wrong_evidence_kind(field: str) -> None:
    mesh = reference_quality_mesh("triangle")
    geometry = phx.discretization.CellGeometrySpec.affine(mesh)
    arguments: dict[str, Any] = {field: object()}
    with pytest.raises(TypeError, match=field):
        phx.meshing.audit_cell_mesh(mesh, geometry, **arguments)


def test_audit_prepared_validity_reuses_the_bound_certificate_and_report_identity() -> (
    None
):
    mesh = reference_quality_mesh("triangle")
    geometry = phx.discretization.CellGeometrySpec.affine(mesh)
    cold = phx.meshing.audit_cell_mesh(mesh, geometry)
    prepared = phx.meshing.audit_cell_mesh(
        mesh, geometry, prepared_validity=cold.validity
    )
    assert prepared.validity is cold.validity
    assert prepared.report_id == cold.report_id
    assert prepared.check_counts == cold.check_counts
    assert prepared.evaluated_checks == cold.evaluated_checks


def test_audit_prepared_validity_refuses_a_different_policy() -> None:
    mesh = reference_quality_mesh("triangle")
    geometry = phx.discretization.CellGeometrySpec.affine(mesh)
    validity = phx.discretization.certify_cell_geometry_validity(geometry, mesh=mesh)
    policy = phx.meshing.CellMeshAuditPolicy(
        validity_policy=phx.discretization.CellValidityPolicy(maximum_subdivision_depth=0)
    )
    with pytest.raises(ValueError, match="prepared_validity"):
        phx.meshing.audit_cell_mesh(
            mesh, geometry, policy=policy, prepared_validity=validity
        )


def test_audit_prepared_validity_refuses_forged_positive_summary() -> None:
    mesh = reference_quality_mesh("triangle")
    geometry = phx.discretization.CellGeometrySpec.affine(mesh)
    validity = phx.discretization.certify_cell_geometry_validity(geometry, mesh=mesh)
    forged = eqx.tree_at(
        lambda certificate: certificate.status,
        validity,
        jnp.full_like(validity.status, phx.discretization.CellValidityStatus.INVALID),
    )
    assert forged.all_certified
    with pytest.raises(ValueError, match="prepared_validity"):
        phx.meshing.audit_cell_mesh(mesh, geometry, prepared_validity=forged)


@pytest.mark.parametrize(
    ("points", "first", "second", "expected_hit", "orientation_rows"),
    (
        (
            ((0, 0, 0), (1, 0, 0), (0, 1, 0), (0, 0, 1), (1, 1, 1)),
            (0, 1, 2),
            (0, 3, 4),
            False,
            2,
        ),
        (
            ((0, 0, 0), (1, 0, 0), (0, 1, 0), (0.2, 0.2, 1), (0.2, 0.2, -1)),
            (0, 1, 2),
            (0, 3, 4),
            True,
            # Two nonshared plane signs, the two needed plane sides of the
            # first triangle's free vertices and five opposite-edge signs.
            9,
        ),
        (
            ((0, 0, 0), (1, 0, 0), (0, 1, 0), (0, 0, 1), (1, 0, 1), (0, 1, 1)),
            (0, 1, 2),
            (3, 4, 5),
            False,
            3,
        ),
        (
            ((0, 0, 0), (1, 0, 0), (0, 1, 0), (0, -1, 0)),
            (0, 1, 2),
            (1, 0, 3),
            False,
            0,
        ),
        (
            ((0, 0, 0), (1, 0, 0), (0, 1, 0), (0.25, 0.25, 0)),
            (0, 1, 2),
            (0, 1, 3),
            True,
            0,
        ),
    ),
    ids=(
        "shared-vertex-separated",
        "shared-vertex-crossing",
        "disjoint-separated",
        "coordinate-plane-separated",
        "coordinate-plane-folded",
    ),
)
def test_triangle_intersection_skips_certified_irrelevant_predicates(
    points: tuple[tuple[float, float, float], ...],
    first: tuple[int, int, int],
    second: tuple[int, int, int],
    expected_hit: bool,
    orientation_rows: int,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import phydrax.meshing._audit_topology as owner
    from phydrax.discretization._coordinate_budget import CoordinateEnclosureBudget

    original = owner.orientation3d_signs
    queried_rows = 0

    def orientation(
        a: np.ndarray,
        b: np.ndarray,
        c: np.ndarray,
        d: np.ndarray,
        /,
        *,
        budget: CoordinateEnclosureBudget | None,
    ) -> tuple[np.ndarray, np.ndarray]:
        nonlocal queried_rows
        queried_rows += a.shape[0]
        return original(a, b, c, d, budget=budget)

    monkeypatch.setattr(owner, "orientation3d_signs", orientation)
    hit, certain = owner.triangle_pairs_intersect(
        np.asarray(points, dtype=np.float64),
        np.asarray((first,), dtype=np.int64),
        np.asarray((second,), dtype=np.int64),
    )
    np.testing.assert_array_equal(hit, (expected_hit,))
    np.testing.assert_array_equal(certain, (True,))
    assert queried_rows == orientation_rows


@pytest.mark.parametrize("exact_objects", (False, True))
@pytest.mark.parametrize("shared", (1, 2, 3))
def test_triangle_plane_predicates_never_receive_a_repeated_shared_index(
    shared: int, exact_objects: bool, monkeypatch: pytest.MonkeyPatch
) -> None:
    import phydrax.meshing._audit_topology as owner

    points = np.asarray(
        ((1, 2, 3), (2, 4, 7), (3, 1, 5), (2, 3, 8), (4, 2, 9)),
        dtype=np.float64,
    )
    if exact_objects:
        points = np.asarray(
            [[int(value) for value in row] for row in points], dtype=object
        )
    first = np.asarray(((0, 1, 2),), dtype=np.int64)
    second = np.asarray(
        ({1: (0, 3, 4), 2: (1, 0, 3), 3: (2, 0, 1)}[shared],),
        dtype=np.int64,
    )
    expected = owner.triangle_pairs_intersect(points, first, second)
    original = owner.orientation3d_signs

    def orientation(
        a: np.ndarray,
        b: np.ndarray,
        c: np.ndarray,
        d: np.ndarray,
        /,
        *,
        budget: CoordinateEnclosureBudget | None,
    ) -> tuple[np.ndarray, np.ndarray]:
        assert not np.any(np.all(a == d, axis=1))
        assert not np.any(np.all(b == d, axis=1))
        assert not np.any(np.all(c == d, axis=1))
        return original(a, b, c, d, budget=budget)

    monkeypatch.setattr(owner, "orientation3d_signs", orientation)
    actual = owner.triangle_pairs_intersect(points, first, second)
    np.testing.assert_array_equal(actual, expected)


@pytest.mark.parametrize(
    "kind", ("boolean", "complex", "object-float", "fraction", "large-integer")
)
def test_triangle_shared_indices_preserve_predicate_numeric_admission(kind: str) -> None:
    from fractions import Fraction

    from phydrax.meshing._audit_topology import triangle_pairs_intersect

    points = np.asarray(((1, 2, 3), (2, 4, 7), (3, 1, 5)), dtype=np.float64)
    if kind == "boolean":
        points = np.eye(3, dtype=np.bool_)
    elif kind == "complex":
        points = points.astype(np.complex128)
    elif kind == "object-float":
        points = points.astype(object)
    elif kind == "fraction":
        points = np.asarray(
            [[Fraction(value) for value in row] for row in points], dtype=object
        )
    else:
        points = points.astype(np.int64) * (1 << 54)
    with pytest.raises((TypeError, ValueError)):
        triangle_pairs_intersect(
            points,
            np.asarray(((0, 1, 2),), dtype=np.int64),
            np.asarray(((2, 0, 1),), dtype=np.int64),
        )


def _exact_orientation(*points: tuple[float, ...]) -> int:
    from fractions import Fraction

    a, b, c, d = (tuple(Fraction(value) for value in point) for point in points)
    u, v, w = (tuple(q - p for p, q in zip(a, point, strict=True)) for point in (b, c, d))
    determinant = (
        u[0] * (v[1] * w[2] - v[2] * w[1])
        - u[1] * (v[0] * w[2] - v[2] * w[0])
        + u[2] * (v[0] * w[1] - v[1] * w[0])
    )
    return (determinant > 0) - (determinant < 0)


def _reference_noncoplanar_contact(
    one: list[tuple[float, ...]], two: list[tuple[float, ...]], shared: int
) -> bool | None:
    """Per-direction segment/triangle route evaluating every orientation afresh."""
    plane = [_exact_orientation(*one, vertex) for vertex in two]
    if all(value == 0 for value in plane):
        return None
    free = [index != 0 for index in range(3)] if shared else [True] * 3
    if any(free) and (
        all(value > 0 or not keep for value, keep in zip(plane, free, strict=True))
        or all(value < 0 or not keep for value, keep in zip(plane, free, strict=True))
    ):
        return False
    contact = False
    for source, target in ((one, two), (two, one)):
        for index in range(3):
            if shared and index != 1:
                continue
            start, end = source[index], source[(index + 1) % 3]
            side_a = _exact_orientation(*target, start)
            side_b = _exact_orientation(*target, end)
            edges = [
                _exact_orientation(start, end, target[edge], target[(edge + 1) % 3])
                for edge in range(3)
            ]
            straddles = side_a * side_b <= 0 and not (side_a == 0 and side_b == 0)
            inside = all(value >= 0 for value in edges) or all(value <= 0 for value in edges)
            contact |= straddles and inside
    return contact


def test_triangle_crossing_reuses_alternating_orientations_exactly() -> None:
    import phydrax.meshing._audit_topology as owner

    generator = np.random.default_rng(20261020)
    points: list[tuple[float, ...]] = []
    first: list[tuple[int, int, int]] = []
    second: list[tuple[int, int, int]] = []
    expected: list[bool] = []
    while len(expected) < 400:
        corners = [tuple(float(v) for v in generator.integers(-2, 3, 3)) for _ in range(6)]
        shared = int(generator.integers(0, 2))
        if shared:
            corners[3] = corners[0]
        one, two = corners[:3], corners[3:]
        if any(
            all(_exact_orientation(*triangle, probe) == 0 for probe in corners)
            for triangle in (one, two)
        ):
            continue
        reference = _reference_noncoplanar_contact(one, two, shared)
        if reference is None:
            continue
        base = len(points)
        points.extend(one)
        first.append((base, base + 1, base + 2))
        if shared:
            points.extend(two[1:])
            second.append((base, base + 3, base + 4))
        else:
            points.extend(two)
            second.append((base + 3, base + 4, base + 5))
        expected.append(reference)
    hit, certain = owner.triangle_pairs_intersect(
        np.asarray(points, dtype=np.float64),
        np.asarray(first, dtype=np.int64),
        np.asarray(second, dtype=np.int64),
    )
    assert np.all(certain)
    assert any(expected) and not all(expected)
    np.testing.assert_array_equal(hit, expected)


def test_triangle_shared_vertex_contact_reduces_exact_native_queries() -> None:
    from phydrax._meshcore import (
        meshcore_available,
        NativeExecutionLimits,
        NativeExecutionScope,
    )
    from phydrax.meshing._audit_topology import triangle_pairs_intersect

    if not meshcore_available():
        pytest.skip("Native exact predicate evidence requires meshcore.")
    points = np.asarray(
        ((1, 2, 3), (2, 3, 5), (3, 2, 4), (1, 3, 5), (3, 3, 5)),
        dtype=np.float64,
    )
    with NativeExecutionScope(
        NativeExecutionLimits(100_000, 100_000, 0, 1 << 24, 120.0)
    ) as scope:
        hit, certain = triangle_pairs_intersect(
            points,
            np.asarray(((0, 1, 2),), dtype=np.int64),
            np.asarray(((0, 3, 4),), dtype=np.int64),
        )
    np.testing.assert_array_equal(hit, (True,))
    np.testing.assert_array_equal(certain, (True,))
    # The one filter-unresolved edge orientation is the even permutation of
    # an edge sign the filter already certified in the other direction.
    assert scope.completed_evidence().native_primitive_queries == 0


def test_triangle_contact_batch_reuses_reordered_subsets_and_binds_values(
    reference_budget: CoordinateEnclosureBudget,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import phydrax.meshing._audit_topology as owner

    points = np.asarray(
        ((0, 0, 0), (1, 0, 0), (0, 1, 0), (0, -1, 0), (0.25, 0.25, 0)),
        dtype=np.float64,
    )
    first = np.asarray(((0, 1, 2), (0, 1, 2)), dtype=np.int64)
    second = np.asarray(((1, 0, 3), (0, 1, 4)), dtype=np.int64)
    original = owner._triangle_pairs_intersect
    evaluated_rows = 0

    def evaluate(
        points: np.ndarray,
        first: np.ndarray,
        second: np.ndarray,
        /,
        *,
        budget: CoordinateEnclosureBudget | None,
    ) -> tuple[np.ndarray, np.ndarray]:
        nonlocal evaluated_rows
        evaluated_rows += first.shape[0]
        return original(points, first, second, budget=budget)

    monkeypatch.setattr(owner, "_triangle_pairs_intersect", evaluate)
    hit, certain = owner.triangle_pairs_intersect(
        points, first, second, budget=reference_budget
    )
    np.testing.assert_array_equal(hit, (False, True))
    np.testing.assert_array_equal(certain, (True, True))
    assert evaluated_rows == 2
    # Different batching, pair direction, and local winding consume the same
    # bound proof, including its negative (contact) verdict.
    hit, certain = owner.triangle_pairs_intersect(
        points, second[1:, ::-1], first[1:, ::-1], budget=reference_budget
    )
    np.testing.assert_array_equal(hit, (True,))
    np.testing.assert_array_equal(certain, (True,))
    assert evaluated_rows == 2
    points[4] = (0, -0.25, 0)
    hit, certain = owner.triangle_pairs_intersect(
        points, first[1:], second[1:], budget=reference_budget
    )
    np.testing.assert_array_equal(hit, (False,))
    np.testing.assert_array_equal(certain, (True,))
    assert evaluated_rows == 3


def test_triangle_contact_batch_binds_welded_vertex_identities(
    reference_budget: CoordinateEnclosureBudget,
) -> None:
    from phydrax.meshing._audit_topology import triangle_pairs_intersect

    points = np.asarray(
        ((0, 0, 0), (1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 0), (1, 0, 0)),
        dtype=np.float64,
    )
    first = np.asarray(((0, 1, 2),), dtype=np.int64)
    for second, expected in (((1, 0, 3), False), ((5, 4, 3), True)):
        hit, certain = triangle_pairs_intersect(
            points, first, np.asarray((second,), dtype=np.int64), budget=reference_budget
        )
        np.testing.assert_array_equal(hit, (expected,))
        np.testing.assert_array_equal(certain, (True,))


@pytest.mark.parametrize("shared_vertex", (False, True))
def test_filtered_triangle_separation_matches_complete_contact_predicate(
    shared_vertex: bool,
) -> None:
    import phydrax.meshing._audit_topology as owner

    rng = np.random.default_rng(519)
    one = rng.integers(-16, 17, size=(48, 3, 2)).astype(np.float64) / 8
    two = rng.integers(-16, 17, size=(48, 3, 2)).astype(np.float64) / 8
    shared = np.full(48, int(shared_vertex), dtype=np.int64)
    if shared_vertex:
        two[:, 0] = one[:, 0]
    separated = owner._coplanar_filtered_separation(one, two, shared, budget=None)
    expected_hit, expected_certain = owner._coplanar_projected_overlap(
        one, two, shared, budget=None
    )
    assert np.any(separated)
    assert np.all(expected_certain[separated] & ~expected_hit[separated])
    hit, certain = owner._coplanar_overlap(
        np.pad(one, ((0, 0), (0, 0), (0, 1))),
        np.pad(two, ((0, 0), (0, 0), (0, 1))),
        shared,
        budget=None,
    )
    np.testing.assert_array_equal(hit, expected_hit)
    np.testing.assert_array_equal(certain, expected_certain)


def _periodic_torus_fan_shifts(
    first: tuple[int, int], second: tuple[int, int]
) -> np.ndarray:
    a = np.asarray(first, dtype=np.int64)
    b = np.asarray(second, dtype=np.int64)
    zero = np.zeros(2, dtype=np.int64)
    return np.asarray(
        ((zero, a, zero), (a, a + b, zero), (a + b, b, zero), (b, zero, zero)),
        dtype=np.int64,
    )


def _periodic_vertex_audit(
    mesh: phx.discretization.CellMesh,
) -> phx.meshing.CellMeshAuditReport:
    skip = phx.meshing.CellMeshAuditDisposition.SKIP
    return phx.meshing.audit_cell_mesh(
        mesh,
        phx.discretization.CellGeometrySpec.affine(mesh),
        policy=phx.meshing.CellMeshAuditPolicy(
            coincident_vertices=skip,
            self_intersection=skip,
            watertight_boundary=skip,
        ),
    )


@pytest.mark.parametrize("chart_cut", (False, True), ids=("coherent-lift", "split-lift"))
def test_periodic_vertex_links_glue_chart_cuts_without_collapsing_windings(
    chart_cut: bool,
) -> None:
    from phydrax.meshing._periodic import publish_periodic_simplices

    shifts = _periodic_torus_fan_shifts((1, 0), (0, 1))
    if chart_cut:
        shifts[[1, 3]] += np.asarray((2, 0), dtype=np.int64)
    mesh = publish_periodic_simplices(
        np.asarray(((0.0, 0.0), (0.5, 0.5)), dtype=np.float64),
        np.tile(np.asarray((0, 0, 1), dtype=np.int64), (4, 1)),
        shifts,
        phx.discretization.PeriodicCell(np.eye(2, dtype=np.float64)),
    )
    audit = _periodic_vertex_audit(mesh)
    assert audit.passed
    assert dict(audit.check_counts)["nonmanifold_vertices"] == 0
    if chart_cut:
        unglued = phx.discretization.CellMesh(mesh.coordinates, mesh.blocks)
        rejected = _periodic_vertex_audit(unglued)
        assert rejected.issues == ("nonmanifold_vertices",)
        assert dict(rejected.check_counts)["nonmanifold_vertices"] == 2


def test_periodic_vertex_links_reject_two_tori_pinched_at_one_quotient_vertex() -> None:
    from phydrax.meshing._periodic import publish_periodic_simplices

    mesh = publish_periodic_simplices(
        np.asarray(((0.0, 0.0), (0.5, 0.5), (1.0, 1.5)), dtype=np.float64),
        np.concatenate(
            (
                np.tile(np.asarray((0, 0, 1), dtype=np.int64), (4, 1)),
                np.tile(np.asarray((0, 0, 2), dtype=np.int64), (4, 1)),
            )
        ),
        np.concatenate(
            (
                _periodic_torus_fan_shifts((1, 0), (0, 1)),
                _periodic_torus_fan_shifts((1, 1), (1, 2)),
            )
        ),
        phx.discretization.PeriodicCell(np.eye(2, dtype=np.float64)),
    )
    rejected = _periodic_vertex_audit(mesh)
    assert not rejected.passed
    assert rejected.issues == ("nonmanifold_vertices",)
    assert dict(rejected.check_counts)["nonmanifold_vertices"] == 1

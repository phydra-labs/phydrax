from fractions import Fraction

import numpy as np
import pytest

from phydrax._meshcore import meshcore_available
from phydrax._physical import SpatialCoordinateContract
from phydrax.discretization import CellBlock, CellGeometrySpec, CellMesh
from phydrax.geometry._domain_coverage import PiecewiseLinearDomain
from phydrax.geometry._mapped_reference_boundary import MappedReferenceBoundarySource
from phydrax.geometry._mapped_reference_domain import MappedReferenceDomain
from phydrax.meshing._association import (
    GeometryAssociation,
    GeometryAssociationKind,
    GeometrySourceEntityRole,
)
from phydrax.meshing._layer_core import _fixed_cap_edge_obstruction
from phydrax.meshing._layer_core_brep import _rebind_association
from phydrax.meshing._layer_core_mapped import mapped_source_lipschitz_bound
from phydrax.meshing._layer_core_regions import prepare_layer_region_identity
from phydrax.meshing._layer_core_subdivision import (
    _distance_increment_at_least,
    _growth_difference_sign,
    _nearest_wall_point,
    _PreparedWallTriangles,
    certify_nested_layer_columns,
    nested_prism_reference_partition,
    source_derived_nested_divisions,
    subdivide_nested_prism_columns,
    subdivide_parallel_prism_pairs,
)
from phydrax.meshing._layer_core_topology import (
    combined_domain,
    cycle,
    faces,
    prepare_identity,
    triangles,
)
from phydrax.meshing._plc_source import PiecewiseLinearComplex
from phydrax.meshing.providers._native_layer import _implicit_layer_surface_chart_witness
from tests._support.implicit_layer_curving_fixtures import certified_implicit_layer_wall
from tests._support.layer_curving_fixtures import _layer_mesh, _mixed_construction
from tests.unit.geometry.test_implicit_source_charts import _adaptive_source


@pytest.mark.parametrize("kind", ("tetrahedron", "hexahedron", "prism", "pyramid"))
def test_reference_subentity_predicates_match_exact_hull(kind: str) -> None:
    from phydrax.discretization._reference_cell import reference_cell_topology
    from phydrax.meshing._mixed_adaptation import _point_in_root_hull
    from phydrax.meshing._volume_entity_containment import (
        _point_key,
        _reference_subentity_contains,
    )

    topology = reference_cell_topology(kind)
    vertices = tuple(
        tuple(Fraction(float(value)) for value in point) for point in topology.vertices
    )
    points = (
        *vertices,
        (Fraction(1, 4), Fraction(1, 4), Fraction(1, 4)),
        (Fraction(1, 2), Fraction(1, 2), Fraction(0)),
        (Fraction(1, 2), Fraction(1, 2), Fraction(1, 2)),
        (Fraction(-1, 4), Fraction(0), Fraction(0)),
        (Fraction(2), Fraction(1, 2), Fraction(1, 2)),
    )
    for dimension in range(4):
        for index, route in enumerate(topology.entities[dimension]):
            hull = tuple(_point_key(vertices[vertex]) for vertex in route)
            for point in points:
                assert _reference_subentity_contains(kind, dimension, index, point) == (
                    _point_in_root_hull(_point_key(point), hull)
                )


def test_exact_root_corner_batch_matches_symbolic_nonbinary_maps() -> None:
    from phydrax.discretization._cell_geometry_transition_types import (
        _geometry_map_expressions,
    )
    from phydrax.discretization._coordinate_enclosure import evaluate_expression
    from phydrax.discretization._exact_plc_geometry import ExactPlcCellGeometrySource
    from phydrax.discretization._reference_cell import reference_cell_topology
    from phydrax.meshing._curving import _straight_geometry

    exact = []
    blocks = []
    for index, kind in enumerate(("tetrahedron", "prism")):
        vertices = reference_cell_topology(kind).vertices
        start = len(exact)
        exact.extend(
            (
                Fraction(index * 2) + Fraction(float(point[0])) / 3,
                Fraction(float(point[1])) / 3,
                Fraction(float(point[2])) / 3,
            )
            for point in vertices
        )
        blocks.append(
            CellBlock(
                kind,
                kind,
                np.asarray([tuple(range(start, len(exact)))]),
                global_ids=np.asarray([index + 10], dtype=np.int64),
            )
        )
    mesh = CellMesh(np.asarray(exact, dtype=np.float64), tuple(blocks))
    carrier = ExactPlcCellGeometrySource(
        "nonbinary-root",
        "revision",
        np.asarray(mesh.vertex_global_ids),
        tuple(exact),
        vertex_source_entity_ids=tuple(f"vertex-{index}" for index in range(len(exact))),
        vertex_occurrence_paths=((),) * len(exact),
    )
    base = CellGeometrySpec.power(mesh, carrier)
    geometry = _straight_geometry(mesh, 2, source_geometry=base)
    elements, routes, _ = geometry.resolve(mesh)
    maps = _geometry_map_expressions(mesh, geometry, (elements, routes))
    for block in mesh.blocks:
        corners = tuple(
            tuple(Fraction(float(value)) for value in point)
            for point in reference_cell_topology(block.cell_kind).vertices
        )
        ids = np.asarray(block.global_ids, dtype=np.int64)
        batches = geometry.evaluate_exact(mesh, ids, (corners,) * len(ids))
        for identifier, actual in zip(ids, batches, strict=True):
            expected = tuple(
                tuple(
                    evaluate_expression(expression, point)
                    for expression in maps.expressions(int(identifier))
                )
                for point in corners
            )
            assert actual == expected


def test_grouped_root_map_proof_checks_every_physical_cell() -> None:
    from phydrax.discretization._cell_geometry_transition_types import (
        _geometry_map_expressions,
        _require_exact_reference_group,
    )
    from phydrax.discretization._coordinate_enclosure import axes
    from phydrax.discretization._nested_reference_types import (
        exact_reference_cell_vertices,
        freeze_reference_chart_arguments,
        ReferenceCellMapWitnesses,
    )
    from phydrax.discretization._reference_cell import reference_cell_topology

    corners = np.asarray(
        reference_cell_topology("tetrahedron").vertices, dtype=np.float64
    )
    points = np.concatenate((corners, corners + np.asarray((2.0, 0.0, 0.0))))
    cells = np.asarray(((0, 1, 2, 3), (4, 5, 6, 7)), dtype=np.int64)
    source = CellMesh(
        points, (CellBlock("cells", "tetrahedron", cells, global_ids=[1, 2]),)
    )
    target = CellMesh(
        points, (CellBlock("cells", "tetrahedron", cells, global_ids=[101, 102]),)
    )
    source_geometry = CellGeometrySpec.affine(source)
    target_geometry = CellGeometrySpec.affine(target)
    source_elements, source_routes, _ = source_geometry.resolve(source)
    target_elements, target_routes, _ = target_geometry.resolve(target)
    source_maps = _geometry_map_expressions(
        source, source_geometry, (source_elements, source_routes)
    )
    target_maps = _geometry_map_expressions(
        target, target_geometry, (target_elements, target_routes)
    )
    exact = tuple(tuple(Fraction(float(value)) for value in point) for point in corners)
    witness = ReferenceCellMapWitnesses(
        np.asarray([101, 102]),
        np.asarray([1, 2]),
        "tetrahedron",
        "tetrahedron",
        np.asarray([corners, corners]),
        (exact, exact),
        freeze_reference_chart_arguments((axes(3), axes(3))),
    )
    _require_exact_reference_group(
        source_maps, target_maps, witness, exact_reference_cell_vertices(witness)
    )
    changed = np.asarray(target_geometry.coordinates, dtype=np.float64).copy()
    changed[7, 2] += 0.125
    forged = CellGeometrySpec(
        {"cells": target_elements[0]}, {"cells": target_routes[0]}, changed
    )
    forged_elements, forged_routes, _ = forged.resolve(target)
    forged_maps = _geometry_map_expressions(
        target, forged, (forged_elements, forged_routes)
    )
    with pytest.raises(ValueError, match="coordinate-map equality"):
        _require_exact_reference_group(
            source_maps, forged_maps, witness, exact_reference_cell_vertices(witness)
        )


def test_inward_wall_rebind_preserves_surface_source_columns() -> None:
    mesh = CellMesh(
        np.asarray(((0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0))),
        (CellBlock("wall", "triangle", np.asarray(((0, 1, 2),))),),
    )
    face_set = mesh.entity_set(2)
    parent = GeometryAssociation(
        GeometryAssociationKind.SURFACE,
        "source",
        "revision",
        face_set.entity_set_id,
        np.asarray(face_set.entity_ids),
        ("patch",),
        np.asarray((0.0,)),
        source_dimensions=np.asarray((2,)),
        source_indices=np.asarray((7,)),
        source_roles=(GeometrySourceEntityRole.FACET,),
        source_occurrence_paths=(("instance",),),
        source_region_sides=(("inside", None),),
        parameters=np.asarray(((0.25, 0.5),)),
        orientations=np.asarray((1,)),
    )
    result = _rebind_association(parent, mesh, 2, -1)
    np.testing.assert_array_equal(result.source_dimensions, (2,))
    np.testing.assert_array_equal(result.source_indices, (7,))
    np.testing.assert_array_equal(result.parameters, ((0.25, 0.5),))
    np.testing.assert_array_equal(result.orientations, (-1,))
    assert result.source_occurrence_paths == (("instance",),)
    assert result.source_region_sides == (("inside", None),)


def _tetrahedron_plc(region_id: str = "core") -> PiecewiseLinearComplex:
    vertices = np.asarray(
        ((0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)),
        dtype=np.float64,
    )
    polygons = (
        np.asarray((1, 2, 3), dtype=np.int64),
        np.asarray((0, 3, 2), dtype=np.int64),
        np.asarray((0, 1, 3), dtype=np.int64),
        np.asarray((0, 2, 1), dtype=np.int64),
    )
    return PiecewiseLinearComplex(
        vertices,
        polygons,
        np.arange(4, dtype=np.int64),
        np.asarray(((0, -1),) * 4, dtype=np.int64),
        (region_id,),
        vertex_ids=("v0", "v1", "v2", "v3"),
        vertex_occurrence_paths=((),) * 4,
        facet_ids=("f0", "f1", "f2", "f3"),
        facet_occurrence_paths=((),) * 4,
        segment_ids=(),
        segment_occurrence_paths=(),
        region_occurrence_paths=((),),
        boundary="fixed",
    )


def test_refined_column_wall_measurement_uses_exact_nearest_support() -> None:
    point = (Fraction(1, 4), Fraction(1, 4), Fraction(1, 10))
    triangle = (
        (Fraction(0), Fraction(0), Fraction(0)),
        (Fraction(1), Fraction(0), Fraction(0)),
        (Fraction(0), Fraction(1), Fraction(0)),
    )
    evidence = _nearest_wall_point(point, ((17, triangle),))
    assert evidence.squared_distance == Fraction(1, 100)
    assert evidence.supports == (
        (17, (Fraction(1, 2), Fraction(1, 4), Fraction(1, 4)), Fraction(1, 10)),
    )
    assert _distance_increment_at_least(Fraction(1, 100), Fraction(0), Fraction(1, 10))
    assert not _distance_increment_at_least(Fraction(1, 100), Fraction(0), Fraction(1, 9))
    assert (
        _growth_difference_sign(
            Fraction(4, 100), Fraction(1, 100), Fraction(0), Fraction(1)
        )
        == 0
    )
    assert (
        _growth_difference_sign(
            Fraction(9, 100), Fraction(1, 100), Fraction(0), Fraction(1)
        )
        > 0
    )


@pytest.mark.parametrize(
    ("origin", "scale"),
    (
        (Fraction(0), Fraction(1)),
        (Fraction(10**18), Fraction(1, 10**9)),
        (Fraction(-(10**100)), Fraction(10**90)),
        (Fraction(1, 10**300), Fraction(1, 10**310)),
    ),
)
def test_prepared_wall_bounds_preserve_exact_supports(
    origin: Fraction, scale: Fraction
) -> None:
    def point(
        x: int | Fraction, y: int | Fraction, z: int | Fraction
    ) -> tuple[Fraction, ...]:
        return (origin + x * scale, origin + y * scale, origin + z * scale)

    triangle = (point(0, 0, 0), point(1, 0, 0), point(0, 1, 0))
    far = (point(100, 100, 0), point(101, 100, 0), point(100, 101, 0))
    triangles = ((29, triangle), (7, triangle), (44, far))
    prepared = _PreparedWallTriangles(triangles)
    for query in (
        point(Fraction(1, 4), Fraction(1, 4), Fraction(1, 10)),
        point(0, 0, 0),
        point(Fraction(1, 2), Fraction(1, 2), 0),
        point(100, 100, 0),
    ):
        assert _nearest_wall_point(query, prepared.candidates(query)) == (
            _nearest_wall_point(query, triangles)
        )


def test_prepared_wall_rejects_far_degenerate_triangle() -> None:
    zero = Fraction(0)
    triangle = ((zero, zero, zero), (Fraction(1), zero, zero), (zero, Fraction(1), zero))
    far = (Fraction(100), zero, zero)
    with pytest.raises(ValueError, match="exactly nondegenerate"):
        _PreparedWallTriangles(((1, triangle), (2, (far, far, far))))


def test_transverse_prism_subdivision_preserves_columns_and_exact_stencils() -> None:
    points = np.asarray(
        (
            (0.0, 0.0, 0.0),
            (2.0, 0.0, 0.0),
            (0.0, 1.0, 0.0),
            (2.0, 1.0, 0.0),
            (0.0, 0.0, 0.01),
            (2.0, 0.0, 0.01),
            (0.0, 1.0, 0.02),
            (2.0, 1.0, 0.02),
            (0.0, 0.0, 0.02),
            (2.0, 0.0, 0.02),
            (0.0, 1.0, 0.04),
            (2.0, 1.0, 0.04),
        )
    )
    mesh = CellMesh(
        points,
        (
            CellBlock(
                "layers",
                "prism",
                np.asarray(
                    (
                        (0, 1, 3, 4, 5, 7),
                        (0, 3, 2, 4, 7, 6),
                        (4, 5, 7, 8, 9, 11),
                        (4, 7, 6, 8, 11, 10),
                    )
                ),
            ),
        ),
    )
    partition = subdivide_parallel_prism_pairs(
        mesh, np.asarray((0, 0, 1, 1)), np.zeros(4, dtype=np.int64), 2
    )
    assert partition.prism_vertices.shape == (8, 6)
    np.testing.assert_array_equal(
        np.unique(partition.source_pair_cell_ids, axis=0), ((0, 1), (2, 3))
    )
    np.testing.assert_array_equal(partition.coordinates[: len(points)], points)
    assert len(set(partition.stencils)) == len(partition.stencils)
    source_ids = np.asarray(mesh.vertex_global_ids)
    middle_edge = tuple(
        sorted(
            ((int(source_ids[4]), Fraction(1, 2)), (int(source_ids[5]), Fraction(1, 2)))
        )
    )
    middle_row = partition.stencils.index(middle_edge)
    assert np.count_nonzero(partition.prism_vertices[:4] == middle_row) > 0
    assert np.count_nonzero(partition.prism_vertices[4:] == middle_row) > 0
    assert partition.exact_coordinates[middle_row] == (
        Fraction(1),
        Fraction(0),
        Fraction.from_float(0.01),
    )


def test_directional_layer_subdivision_refuses_unproved_column_displacements() -> None:
    points = np.asarray(
        (
            (0.0, 0.0, 0.0),
            (2.0, 0.0, 0.0),
            (0.0, 1.0, 0.0),
            (2.0, 1.0, 0.0),
            (0.0, 0.0, 0.01),
            (2.0, 0.0, 0.02),
            (0.0, 1.0, 0.04),
            (2.0, 1.0, 0.08),
        )
    )
    mesh = CellMesh(
        points,
        (
            CellBlock(
                "layers",
                "prism",
                np.asarray(((0, 1, 3, 4, 5, 7), (0, 3, 2, 4, 7, 6))),
            ),
        ),
    )
    with pytest.raises(ValueError, match="equal-displacement"):
        subdivide_parallel_prism_pairs(
            mesh, np.zeros(2, dtype=np.int64), np.zeros(2, dtype=np.int64), 2
        )


def test_fixed_cap_edge_obstruction_uses_exact_represented_coordinates() -> None:
    z0 = float.fromhex("0x1.4d9cd7529b4ddp-6")
    z1 = float.fromhex("0x1.9c90fb05769cap-5")
    vertices = np.asarray(
        ((0.0, 0.0, z0), (1.0, 0.0, z0), (0.0, 1.0, z0), (1.0, 0.0, z1))
    )
    complex_ = PiecewiseLinearComplex(
        vertices,
        tuple(np.asarray(row) for row in ((0, 1, 2), (0, 3, 1), (1, 3, 2), (2, 3, 0))),
        np.arange(4),
        np.asarray(((0, -1),) * 4),
        ("core",),
        vertex_ids=("v0", "v1", "v2", "v3"),
        vertex_occurrence_paths=((),) * 4,
        facet_ids=("cap", "adjacent", "side1", "side2"),
        facet_occurrence_paths=((),) * 4,
        segment_ids=(),
        segment_occurrence_paths=(),
        region_occurrence_paths=((),),
        boundary="fixed",
    )
    result = _fixed_cap_edge_obstruction(complex_, np.asarray((0,)), 2.0)
    assert result is not None
    cap, face, edge, lower_squared = result
    height = Fraction.from_float(z1) - Fraction.from_float(z0)
    assert (cap, face, edge) == (0, 1, (0, 1))
    assert height == Fraction(8646911284551351, 288230376151711744)
    assert lower_squared == 1 / (16 * height * height)
    assert lower_squared > 4
    assert _fixed_cap_edge_obstruction(complex_, np.asarray((0,)), 9.0) is None


@pytest.mark.skipif(
    not meshcore_available(), reason="native boundary layers require meshcore"
)
def test_original_ridge_subdivision_reproves_authored_column_controls() -> None:
    from tools.meshing_qualification.scenarios.native_hybrid_profiles import (
        _archived_ridge_source,
        _periodic_layer_core_specification,
    )

    source, control, policy = _archived_ridge_source(cap_refinement="conforming")
    specification = _periodic_layer_core_specification(source)
    assert source_derived_nested_divisions(source) == 32
    partition = subdivide_nested_prism_columns(source.layers.mesh, 2)
    certificate = certify_nested_layer_columns(source, partition, specification)
    assert certificate.source_binding_id == source.binding_id
    assert certificate.specification_id == specification.specification_id
    assert (certificate.control_id, certificate.policy_id) == (
        control.control_id,
        policy.policy_id,
    )
    assert len(certificate.columns) == 15
    assert partition.prism_vertices.shape == (32, 6)
    np.testing.assert_array_equal(
        partition.coordinates[: source.layers.mesh.coordinates.shape[0]],
        np.asarray(source.layers.mesh.coordinates),
    )
    assert nested_prism_reference_partition(partition).maps[0].target_cell_ids.size == 32
    for column in certificate.columns:
        assert len(column.target_vertices) == 3
        assert column.wall_distances[0].squared_distance == 0
        assert all(distance.supports for distance in column.wall_distances[1:])

    mapping, cap_ids, polygons, points = prepare_identity(
        source.layers, source.complex, source.vertex_layer_ids, source.cap_polygon_ids
    )
    occurrences = (("wall-occurrence",), ("core-occurrence",))
    domain, _ = combined_domain(
        source.layers,
        source.complex,
        np.asarray(source.layer_regions),
        mapping,
        cap_ids,
        polygons,
        points,
        source.source_id,
        source.region_ids,
        np.asarray(source.core_region_map),
        occurrences,
    )
    assert domain.region_occurrence_paths == occurrences


@pytest.mark.skipif(
    not meshcore_available(), reason="native boundary layers require meshcore"
)
def test_noncap_opt_in_without_cad_atlas_retains_identity_layer_partition() -> None:
    from phydrax.meshing._layer_core_edge_policy import noncap_source_edge_classification
    from tools.meshing_qualification.scenarios.native_hybrid_profiles import (
        _archived_ridge_source,
    )

    fixed, _, _ = _archived_ridge_source()
    source, control, policy = _archived_ridge_source(noncap_refinement="conforming")
    assert source.cap_refinement == "fixed"
    assert source.conforming_execution and not fixed.conforming_execution
    assert source.binding_id != fixed.binding_id
    assert source.layer_control is not None
    assert source.layer_policy is not None
    assert source.layer_control.control_id == control.control_id
    assert source.layer_policy.policy_id == policy.policy_id
    source.validate_restored()
    # The periodic ridge has no CAD atlas; its opt-in alone proves no edge
    # removable and does not assert that a changed seam can be recovered.
    assert noncap_source_edge_classification(source) is None
    partition = subdivide_nested_prism_columns(source.layers.mesh, 1)
    original_cells = sorted(
        (int(identifier), tuple(int(vertex) for vertex in vertices))
        for block in source.layers.mesh.blocks
        for identifier, vertices in zip(block.global_ids, block.vertices, strict=True)
    )
    np.testing.assert_array_equal(partition.prism_ids, [row[0] for row in original_cells])
    np.testing.assert_array_equal(partition.parent_cell_ids, partition.prism_ids)
    np.testing.assert_array_equal(
        partition.prism_vertices, [row[1] for row in original_cells]
    )


@pytest.mark.skipif(
    not meshcore_available(), reason="native boundary layers require meshcore"
)
def test_original_ridge_fixed_cap_refuses_proven_quality_obstruction() -> None:
    from phydrax.meshing._contracts import MeshingFailure, MeshingFailureCategory
    from phydrax.meshing._layer_core import _generate_layer_core_unmapped
    from phydrax.meshing._volume_generation import NativeVolumeSchedule
    from tools.meshing_qualification.scenarios.native_hybrid_profiles import (
        _archived_ridge_source,
        _periodic_layer_core_specification,
    )

    source, _, _ = _archived_ridge_source()
    specification = _periodic_layer_core_specification(source)
    with pytest.raises(MeshingFailure) as caught:
        _generate_layer_core_unmapped(
            source,
            specification,
            specification,
            NativeVolumeSchedule(),
            frame_id="fixed-cap-regression",
        )
    failure = caught.value
    assert failure.category is MeshingFailureCategory.QUALITY_REJECTED
    assert failure.stage == "boundary_quality_preflight"
    assert failure.entity_ids
    finding = failure.evidence.logical_findings[0]
    assert "Fixed cap polygon 0, adjacent polygon 8" in finding
    assert "exact squared radius-edge lower bound" in finding


def test_fixed_plc_region_identity_retains_explicit_composite_materials() -> None:
    complex_ = _tetrahedron_plc("layer")

    names, core_map = prepare_layer_region_identity(
        complex_, np.asarray((0,), dtype=np.int64), ("core", "layer"), np.asarray((1,))
    )

    assert names == ("core", "layer")
    np.testing.assert_array_equal(core_map, np.asarray((1,), dtype=np.int64))
    assert not core_map.flags.writeable


def test_layer_core_region_identity_rejects_unrealized_composite_region() -> None:
    complex_ = _tetrahedron_plc()

    with pytest.raises(
        ValueError, match="realized layer or bounded core material incidence"
    ):
        prepare_layer_region_identity(
            complex_,
            np.asarray((0,), dtype=np.int64),
            ("core", "layer"),
            np.asarray((0,)),
        )


def test_quad_split_is_cyclically_stable_and_orientation_preserving() -> None:
    assert cycle((2, 3, 0, 1)) == (0, 1, 2, 3)
    assert triangles((0, 1, 2, 3)) == ((0, 1, 2), (0, 2, 3))
    assert triangles((0, 3, 2, 1)) == ((0, 3, 2), (0, 2, 1))


def test_combined_face_table_retains_opposite_transition_sides() -> None:
    points = np.asarray(
        (
            (0.0, 0.0, 0.0),
            (1.0, 0.0, 0.0),
            (0.0, 1.0, 0.0),
            (0.0, 0.0, 1.0),
            (0.0, 0.0, -1.0),
        ),
        dtype=np.float64,
    )
    mesh = CellMesh(
        points,
        (
            CellBlock(
                "transition-tetrahedra",
                "tetrahedron",
                np.asarray(((0, 1, 2, 3), (0, 2, 1, 4)), dtype=np.int64),
                global_ids=np.asarray((10, 20), dtype=np.int64),
            ),
        ),
        vertex_global_ids=np.asarray((41, 11, 31, 21, 51), dtype=np.int64),
    )

    table = faces(mesh, np.asarray((3, 7), dtype=np.int64))

    shared = table[(0, 1, 2)]
    assert len(shared) == 2
    assert tuple(face.region for face in shared) == (3, 7)
    assert cycle(shared[0].vertices) == cycle(tuple(reversed(shared[1].vertices)))


def test_layer_core_assembly_rejects_nonmanifold_core_transition() -> None:
    points = np.asarray(
        (
            (0.0, 0.0, 0.0),
            (1.0, 0.0, 0.0),
            (0.0, 1.0, 0.0),
            (0.0, 0.0, 1.0),
            (0.0, 0.0, -1.0),
            (0.0, 0.0, 2.0),
        ),
        dtype=np.float64,
    )
    # Three cells on one face never reach the combined face table: assembling
    # the composed mesh already refuses the non-manifold transition.
    with pytest.raises(ValueError, match="face-manifold"):
        CellMesh(
            points,
            (
                CellBlock(
                    "nonmanifold-tetrahedra",
                    "tetrahedron",
                    np.asarray(
                        ((0, 1, 2, 3), (0, 2, 1, 4), (0, 1, 2, 5)),
                        dtype=np.int64,
                    ),
                    global_ids=np.asarray((10, 20, 30), dtype=np.int64),
                ),
            ),
        )


def test_mapped_layer_core_sizing_uses_physical_root_jacobian_bound() -> None:
    vertices = np.asarray(
        ((0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)),
        dtype=np.float64,
    )
    facets = np.asarray(((1, 2, 3), (0, 3, 2), (0, 1, 3), (0, 2, 1)), dtype=np.int64)
    reference_domain = PiecewiseLinearDomain(
        vertices,
        facets,
        np.asarray(((0, -1),) * 4, dtype=np.int64),
        ("solid",),
        source_id="mapped-layer-root",
    )
    root_mesh = CellMesh(
        vertices,
        (CellBlock("root", "tetrahedron", np.asarray(((0, 1, 2, 3),), dtype=np.int32)),),
        vertex_global_ids=np.asarray((11, 13, 17, 19), dtype=np.int64),
    )
    affine = CellGeometrySpec.affine(root_mesh)
    coordinates = 2.0 * vertices + np.asarray((1.0, -2.0, 0.5), dtype=np.float64)
    source_geometry = CellGeometrySpec(
        dict(zip(affine.block_names, affine.elements, strict=True)),
        {
            name: np.asarray(route)
            for name, route in zip(affine.block_names, affine.geometry_dofs, strict=True)
        },
        coordinates,
    )
    domain = MappedReferenceDomain(
        reference_domain,
        root_mesh,
        source_geometry,
        np.asarray((0,), dtype=np.int64),
        source_id="mapped-layer-root",
        source_revision="revision-1",
        coordinate_contract=SpatialCoordinateContract.si(),
    )

    assert mapped_source_lipschitz_bound(domain) == pytest.approx(2.0)


def _assert_remeshed_implicit_layer_witness(*, smooth: bool) -> tuple[str, ...]:
    _surface, query, contract, _discovery_wall, _geometry, _ = _adaptive_source(
        smooth=smooth
    )
    # The discovery scaffold has sliver triangles, so the layer wall is the
    # certified zero-set remeshed carrier; its cells may share scaffold charts.
    layer_wall, root_witness = certified_implicit_layer_wall(query)
    assert dict(layer_wall.evidence)["carrier_remeshed"] == 1.0
    wall = layer_wall.wall
    layers = _layer_mesh(wall, None)
    source, construction = _mixed_construction(
        layers,
        query.source_id,
        query.source_revision,
        wall,
        contract.spatial_id,
        fidelity_source=query,
    )
    mesh = construction.mesh
    witness = _implicit_layer_surface_chart_witness(source, construction)

    assert witness.mesh_id == mesh.mesh_id
    assert witness.reference_dimension == 3
    assert witness.source_chart_ids == root_witness.source_chart_ids
    np.testing.assert_array_equal(
        witness.implicit_cell_keys, root_witness.implicit_cell_keys
    )
    np.testing.assert_array_equal(
        witness.implicit_root_edge_keys, root_witness.implicit_root_edge_keys
    )
    assert np.unique(witness.entity_global_ids).size == wall.blocks[0].cell_count
    return witness.source_chart_ids


def test_implicit_layer_witness_projects_to_output_volume_facets() -> None:
    if not meshcore_available():
        pytest.skip("native boundary-layer generation requires meshcore")
    _assert_remeshed_implicit_layer_witness(smooth=True)


def test_remeshed_sphere_layer_witness_repeats_seeding_scaffold_charts() -> None:
    if not meshcore_available():
        pytest.skip("native boundary-layer generation requires meshcore")
    chart_ids = _assert_remeshed_implicit_layer_witness(smooth=False)
    # Remeshed cells name their nearest scaffold triangle, so charts repeat.
    assert len(set(chart_ids)) < len(chart_ids)


def test_mapped_scoped_chart_witness_allows_unrelated_target_exterior_facets() -> None:
    from phydrax.geometry._source_boundary_records import SourceBoundaryChartWitness
    from phydrax.geometry._source_fidelity import validate_source_fidelity_witness
    from tests.unit.meshing.test_native_mapped_hex_routes import (
        _mapped_cube_source,
        _mapped_surface_chart_witness,
        _slice_mapped_chart_witness,
    )

    source, domain = _mapped_cube_source()
    geometry = CellGeometrySpec.affine(domain.reference_mesh)
    full_query, full_witness = _mapped_surface_chart_witness(source, domain, geometry)

    def scoped_chart_ids(query: MappedReferenceBoundarySource) -> set[str]:
        # A scoped query keeps the full chart table and selects its scope rows.
        return {query.face_chart_ids[row] for row in query._selected_rows(2)}

    # Scope every source triangle that charts one complete target hex facet;
    # the other five facets stay unrelated exterior facets.
    facets = tuple(
        zip(
            np.asarray(full_witness.cell_global_ids).tolist(),
            np.asarray(full_witness.local_entity_indices).tolist(),
            strict=True,
        )
    )
    rows = tuple(row for row, facet in enumerate(facets) if facet == facets[0])
    facet_chart_ids = {full_witness.source_chart_ids[row] for row in rows}
    selected_query = source.boundary_query(
        face_scope_keys=tuple(
            key
            for key in full_query.face_scope_keys
            if scoped_chart_ids(source.boundary_query(face_scope_keys=(key,)))
            & facet_chart_ids
        )
    )
    assert scoped_chart_ids(selected_query) == facet_chart_ids
    assert 1 < len(rows) < len(facets)
    sliced = _slice_mapped_chart_witness(full_witness, rows)
    # A scoped witness names the scoped query it was derived for.
    witness = SourceBoundaryChartWitness(
        mesh_id=sliced.mesh_id,
        geometry_id=sliced.geometry_id,
        topology_id=sliced.topology_id,
        source_id=sliced.source_id,
        source_revision=sliced.source_revision,
        source_query_id=selected_query.query_id,
        chart_dimension=sliced.chart_dimension,
        reference_dimension=sliced.reference_dimension,
        cell_global_ids=sliced.cell_global_ids,
        entity_global_ids=sliced.entity_global_ids,
        local_entity_indices=sliced.local_entity_indices,
        target_reference_vertices=sliced.target_reference_vertices,
        target_reference_rational_tokens=sliced.target_reference_rational_tokens,
        target_face_vertex_global_ids=sliced.target_face_vertex_global_ids,
        source_chart_ids=sliced.source_chart_ids,
        source_entity_ids=sliced.source_entity_ids,
        occurrence_paths=sliced.occurrence_paths,
        source_parameters=sliced.source_parameters,
        source_parameter_rational_tokens=sliced.source_parameter_rational_tokens,
        orientations=sliced.orientations,
    )

    validated = validate_source_fidelity_witness(
        domain.reference_mesh, geometry, selected_query, witness
    )
    with pytest.raises(AttributeError):
        setattr(validated, "witness", sliced)
    with pytest.raises(AttributeError):
        object.__setattr__(validated, "witness", sliced)
    assert (
        validate_source_fidelity_witness(
            domain.reference_mesh, geometry, selected_query, validated
        )
        is validated
    )
    from phydrax.meshing._certification_inputs import validate_fidelity_witnesses

    reused = validate_fidelity_witnesses(
        domain.reference_mesh,
        geometry,
        selected_query,
        witness,
        ((selected_query, witness, 0.0),),
        validated=(validated,),
    )
    assert reused[0] is validated and reused[1] is validated
    fresh = validate_fidelity_witnesses(
        domain.reference_mesh,
        geometry,
        selected_query,
        witness,
        ((selected_query, witness, 0.0),),
    )
    assert fresh[0] is fresh[1]

from __future__ import annotations

from fractions import Fraction
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pytest

from phydrax.discretization import CellBlock, CellMesh
from phydrax.geometry._coplanar_overlap import exact_coplanar_triangle_partitions
from phydrax.meshing._association import (
    GeometryAssociation,
    GeometryAssociationArchiveRecipe,
    GeometryAssociationKind,
    GeometrySourceEntityRole,
)
from phydrax.meshing._layer_core_strata_replay import (
    core_edge_association_tables,
    match_core_source_edges,
)
from phydrax.meshing._topology_edit import entity_keys
from phydrax.meshing.providers._native_layer_overlap import (
    exact_collinear_segment_partitions,
)


if TYPE_CHECKING:
    from phydrax.discretization import CellGeometrySpec
    from phydrax.geometry._domain_boundary import MeshingDomainBoundarySource
    from phydrax.geometry._source_boundary_records import SourceBoundaryFaceImageWitness


def _point(
    x: int | Fraction, y: int | Fraction, z: int | Fraction = 0
) -> tuple[Fraction, Fraction, Fraction]:
    return (Fraction(x), Fraction(y), Fraction(z))


def _interpolate(
    triangle: tuple[tuple[Fraction, ...], ...], uv: tuple[Fraction, ...]
) -> tuple[Fraction, ...]:
    return tuple(
        triangle[0][axis]
        + uv[0] * (triangle[1][axis] - triangle[0][axis])
        + uv[1] * (triangle[2][axis] - triangle[0][axis])
        for axis in range(3)
    )


def test_crossed_source_diagonal_retains_both_exact_chart_partitions() -> None:
    a, b, c, d = _point(0, 0), _point(1, 0), _point(1, 1), _point(0, 1)
    source = ((a, b, c), (a, c, d))
    target = ((a, b, d), (b, c, d))
    pieces = exact_coplanar_triangle_partitions(source, target)
    assert {(piece.source_row, piece.target_row) for piece in pieces} == {
        (0, 0),
        (0, 1),
        (1, 0),
        (1, 1),
    }
    for piece in pieces:
        assert tuple(
            _interpolate(source[piece.source_row], uv) for uv in piece.source_chart
        ) == tuple(
            _interpolate(target[piece.target_row], uv) for uv in piece.target_chart
        )
    assert any(Fraction(1, 2) in uv for piece in pieces for uv in piece.target_chart)


def test_triangle_partitions_reject_gaps_overlaps_and_non_coplanarity() -> None:
    a, b, c, d = _point(0, 0), _point(1, 0), _point(1, 1), _point(0, 1)
    source = ((a, b, c), (a, c, d))
    with pytest.raises(ValueError):
        exact_coplanar_triangle_partitions(source, ((a, b, d),))
    with pytest.raises(ValueError):
        exact_coplanar_triangle_partitions(source, ((a, b, d), (a, b, d), (b, c, d)))
    with pytest.raises(ValueError, match="overlap"):
        exact_coplanar_triangle_partitions(
            (source[0], source[0]),
            ((a, b, c),),
            require_source=False,
            require_target=False,
        )
    with pytest.raises(ValueError):
        exact_coplanar_triangle_partitions(source, ((_point(0, 0, 1), b, c), (a, c, d)))
    with pytest.raises(TypeError):
        exact_coplanar_triangle_partitions((((0.0, 0, 0), b, c),), ((a, b, c),))  # ty: ignore[invalid-argument-type]


def test_exact_curve_overlap_retains_reversed_source_intervals_and_protection() -> None:
    a, m, b = _point(0, 0), _point(Fraction(1, 2), 0), _point(1, 0)
    pieces = exact_collinear_segment_partitions(
        ((a, m), (b, m)), ((a, b),), require_source=True, require_target=True
    )
    assert [
        (piece.source_row, piece.source_interval, piece.target_interval)
        for piece in pieces
    ] == [
        (0, (Fraction(0), Fraction(1)), (Fraction(0), Fraction(1, 2))),
        (1, (Fraction(1), Fraction(0)), (Fraction(1, 2), Fraction(1))),
    ]
    with pytest.raises(ValueError, match="incomplete"):
        exact_collinear_segment_partitions(((a, b),), ((a, m),), require_source=True)
    # An unprotected diagonal may be absent from the execution edge set.
    assert exact_collinear_segment_partitions(((a, b),), ()) == ()
    with pytest.raises(ValueError, match="overlapping"):
        exact_collinear_segment_partitions(((a, b),), ((a, b), (a, b)))


def test_partial_edge_support_is_archived_and_identity_bound() -> None:
    half = Fraction(1, 2)

    def association(
        target_interval: tuple[Fraction, Fraction] | None,
    ) -> GeometryAssociation:
        return GeometryAssociation(
            GeometryAssociationKind.PIECEWISE_LINEAR,
            "source",
            "revision",
            "edges",
            np.asarray((7,), dtype=np.int64),
            ("curve",),
            np.asarray((0.0,), dtype=np.float64),
            exact=True,
            source_roles=(GeometrySourceEntityRole.EDGE,),
            source_parameter_intervals=((Fraction(0), Fraction(1)),),
            target_parameter_intervals=(target_interval,),
            orientations=np.asarray((1,), dtype=np.int8),
        )

    first = association((Fraction(0), half))
    second = association((half, Fraction(1)))
    whole = association(None)
    assert first.association_id != second.association_id != whole.association_id
    rebuilt = GeometryAssociationArchiveRecipe.from_association(first).restore()
    assert rebuilt.association_id == first.association_id
    assert rebuilt.target_parameter_intervals == ((Fraction(0), half),)
    assert association((Fraction(0), Fraction(1))).association_id == whole.association_id
    for invalid in ((Fraction(-1), half), (half, half), (half, Fraction(2))):
        with pytest.raises(ValueError, match="Target edge intervals"):
            association(invalid)


def test_face_piece_support_is_witness_bound_and_archived() -> None:
    tokens = (
        ((0, 1), (0, 1), (0, 1)),
        ((1, 1), (0, 1), (0, 1)),
        ((0, 1), (1, 1), (0, 1)),
    )
    pieces = (((9, 0, (), tokens),),)
    association = GeometryAssociation(
        GeometryAssociationKind.PIECEWISE_LINEAR,
        "source",
        "revision",
        "faces",
        np.asarray((7,), dtype=np.int64),
        ("facet",),
        np.asarray((0.0,), dtype=np.float64),
        exact=True,
        source_roles=(GeometrySourceEntityRole.FACET,),
        source_region_sides=(("material", None),),
        target_reference_pieces=pieces,
        source_boundary_witness_id="witness",
        orientations=np.asarray((1,), dtype=np.int8),
    )
    restored = GeometryAssociationArchiveRecipe.from_association(association).restore()
    assert restored.association_id == association.association_id
    assert restored.target_reference_pieces == pieces
    assert restored.source_boundary_witness_id == "witness"
    with pytest.raises(ValueError, match="witness"):
        GeometryAssociation(
            GeometryAssociationKind.PIECEWISE_LINEAR,
            "source",
            "revision",
            "faces",
            np.asarray((7,), dtype=np.int64),
            ("facet",),
            np.asarray((0.0,), dtype=np.float64),
            exact=True,
            source_roles=(GeometrySourceEntityRole.FACET,),
            source_region_sides=(("material", None),),
            target_reference_pieces=pieces,
        )


def test_real_source_edge_replay_keeps_partial_support_and_enforces_protection() -> None:
    from phydrax.discretization import CellGeometrySpec
    from phydrax.discretization._cell_geometry_transition_types import (
        _geometry_map_expressions,
    )
    from phydrax.discretization._coordinate_enclosure import (
        axes,
        compose_expression,
        constant,
    )
    from phydrax.discretization._exact_plc_geometry import ExactPlcCellGeometrySource
    from tests.unit.meshing.test_layer_source_association_replay import _source

    source = _source()
    complex_ = source.complex
    points = np.asarray(complex_.vertices, dtype=np.float64)
    row = 0
    first_row, last_row = complex_.plc_edge_vertices[row]
    first = tuple(Fraction.from_float(float(value)) for value in points[first_row])
    last = tuple(Fraction.from_float(float(value)) for value in points[last_row])
    start = tuple(2 * a - b for a, b in zip(first, last, strict=True))
    end = tuple(2 * b - a for a, b in zip(first, last, strict=True))
    axis = next(index for index in range(3) if end[index] != start[index])
    others = tuple(index for index in range(3) if index != axis)
    third, fourth = list(start), list(start)
    third[others[0]] += Fraction(1)
    fourth[others[1]] += Fraction(1)
    coordinate_source = ExactPlcCellGeometrySource(
        "partial-edge-target",
        "authored-exact-extension",
        np.arange(4, dtype=np.int64),
        (start, end, tuple(third), tuple(fourth)),
        vertex_source_entity_ids=(None,) * 4,
        vertex_occurrence_paths=((),) * 4,
    )
    mesh = CellMesh(
        coordinate_source.rounded_vertices,
        (
            CellBlock(
                "core",
                "tetrahedron",
                np.asarray(((0, 1, 2, 3),), dtype=np.int32),
                global_ids=np.asarray((8,), dtype=np.int64),
            ),
        ),
        vertex_global_ids=coordinate_source.vertex_global_ids,
    )
    geometry = CellGeometrySpec.power(mesh, coordinate_source)
    elements, routes, _ = geometry.resolve(mesh)
    maps = _geometry_map_expressions(mesh, geometry, (elements, routes))
    edge_id = next(
        int(identifier)
        for key, identifier in zip(
            entity_keys(mesh, 1), mesh.entity_set(1).entity_ids, strict=True
        )
        if tuple(int(value) for value in key if value >= 0) == (0, 1)
    )
    parameter = axes(1)[0]
    edge_chart = (parameter, constant(0, 1), constant(0, 1))
    image = tuple(compose_expression(value, edge_chart) for value in maps.expressions(8))
    matches = match_core_source_edges(
        source,
        {edge_id: ((8, image, None),)},
        {edge_id: frozenset(int(facet) for facet in complex_.plc_edge_facets[row])},
    )
    matching = next(item for item in matches[edge_id] if item[0] == row)
    assert matching[1:] == (
        (Fraction(0), Fraction(1)),
        (Fraction(1, 3), Fraction(2, 3)),
    )
    tables = core_edge_association_tables(
        source, mesh, matches, (row,), required_source_rows=()
    )
    assert tables[0].target_parameter_intervals == ((Fraction(1, 3), Fraction(2, 3)),)
    with pytest.raises(ValueError, match="only part"):
        core_edge_association_tables(
            source, mesh, matches, (row,), required_source_rows=(row,)
        )


def test_source_edge_replay_preserves_finite_and_whole_image_semantics() -> None:
    from phydrax.discretization._coordinate_enclosure import (
        add,
        constant,
        Expression,
        scale,
    )
    from phydrax.meshing._layer_core_strata_replay import source_line_interval
    from tests.unit.meshing.test_layer_source_association_replay import _source

    source = _source()
    complex_ = source.complex
    row = 0
    first_row, last_row = complex_.plc_edge_vertices[row]
    first, last = (
        tuple(Fraction.from_float(float(value)) for value in complex_.vertices[index])
        for index in (first_row, last_row)
    )

    def image(a: int, b: int, *, curved: bool = False) -> tuple[Expression, ...]:
        parameter: Expression = {
            (0,): Fraction(a),
            (2,) if curved else (1,): Fraction(b - a),
        }
        parameter = {power: value for power, value in parameter.items() if value}
        return tuple(
            add(constant(origin, 1), scale(parameter, end - origin))
            for origin, end in zip(first, last, strict=True)
        )

    images: dict[
        int, tuple[tuple[int, tuple[Expression, ...], tuple[Expression, ...] | None], ...]
    ] = {
        0: ((0, image(2, 3), None),),
        1: ((0, image(1, 2), None),),
        2: ((0, image(1, 0), None),),
        3: ((0, image(0, 1, curved=True), None),),
        4: ((0, image(0, 1), None),),
    }
    facets = frozenset(int(facet) for facet in complex_.plc_edge_facets[row])
    carriers = {edge: facets for edge in images}
    matches = match_core_source_edges(source, images, carriers)
    assert all(item[0] != row for edge in (0, 1, 3) for item in matches.get(edge, ()))
    assert next(item for item in matches[2] if item[0] == row)[1] == (
        Fraction(1),
        Fraction(0),
    )
    assert next(item for item in matches[4] if item[0] == row)[1] == (
        Fraction(0),
        Fraction(1),
    )
    assert match_core_source_edges(source, {4: images[4]}, {4: frozenset()}) == {}
    with pytest.raises(ValueError, match="distinct endpoints"):
        source_line_interval(image(0, 1), first, first)


def test_vertex_of_two_source_facets_is_located_on_their_shared_stratum() -> None:
    from phydrax.discretization._coordinate_enclosure import (
        add,
        constant,
        Expression,
        scale,
    )
    from phydrax.meshing._layer_core_association import _core_source_identity
    from phydrax.meshing._layer_core_strata_replay import (
        _exact_source_points,
        _shared_facet_vertex_stratum,
    )
    from tests.unit.meshing.test_layer_source_association_replay import _source

    source = _source()
    complex_ = source.complex
    points = _exact_source_points(source)
    row = next(
        index for index, facets in enumerate(complex_.plc_edge_facets) if len(facets) == 2
    )
    keys = frozenset(
        _core_source_identity(
            source,
            2,
            str(complex_.facet_ids[int(facet)]),
            tuple(complex_.facet_occurrence_paths[int(facet)]),
        )
        for facet in complex_.plc_edge_facets[row]
    )
    assert len(keys) == 2
    first_row, last_row = (int(value) for value in complex_.plc_edge_vertices[row])
    first, last = points[first_row], points[last_row]
    middle = tuple((a + b) / 2 for a, b in zip(first, last, strict=True))
    outside = tuple(value + 1 for value in middle)

    def chart(
        start: tuple[Fraction, ...], end: tuple[Fraction, ...]
    ) -> tuple[Expression, ...]:
        return tuple(
            add(constant(a, 1), scale({(1,): Fraction(1)}, b - a))
            for a, b in zip(start, end, strict=True)
        )

    def locate(*charts: tuple[Expression, ...]) -> tuple[int, Fraction | None] | None:
        images: dict[
            int,
            tuple[tuple[int, tuple[Expression, ...], tuple[Expression, ...] | None], ...],
        ] = {edge: ((0, value, None),) for edge, value in enumerate(charts)}
        return _shared_facet_vertex_stratum(
            source, keys, tuple((edge, 1) for edge in images), images, points
        )

    # A refinement vertex on an unrecovered shared edge is that edge's interior.
    assert locate(chart(outside, middle)) == (row, Fraction(1, 2))
    # A shared PLC vertex is the more specific stratum.
    assert locate(chart(outside, first)) == (-1 - first_row, None)
    # A point of neither closed facet pair intersection is not resolved.
    assert locate(chart(middle, outside)) is None
    with pytest.raises(ValueError, match="disagree"):
        locate(chart(outside, middle), chart(outside, first))


def test_public_audit_rejects_unbound_partial_edge_claims() -> None:
    from phydrax.discretization import CellGeometrySpec
    from phydrax.meshing import audit_cell_mesh, CellMeshAuditPolicy

    mesh = CellMesh(
        np.asarray(((0.0, 0.0), (1.0, 0.0), (0.0, 1.0))),
        (CellBlock("cell", "triangle", np.asarray(((0, 1, 2),))),),
    )
    edges = mesh.entity_set(1)
    edge = int(np.asarray(edges.entity_ids)[0])
    half = Fraction(1, 2)
    tables = tuple(
        GeometryAssociation(
            GeometryAssociationKind.PIECEWISE_LINEAR,
            "source",
            "revision",
            edges.entity_set_id,
            np.asarray((edge,), dtype=np.int64),
            (identity,),
            np.asarray((0.0,), dtype=np.float64),
            exact=True,
            source_roles=(GeometrySourceEntityRole.EDGE,),
            source_parameter_intervals=((Fraction(0), Fraction(1)),),
            target_parameter_intervals=(interval,),
            orientations=np.asarray((1,), dtype=np.int8),
        )
        for identity, interval in (
            ("first", (Fraction(0), half)),
            ("second", (half, Fraction(1))),
        )
    )
    report = audit_cell_mesh(
        mesh,
        CellGeometrySpec.affine(mesh),
        associations=tables,
        policy=CellMeshAuditPolicy(require_complete_association=True),
    )
    assert "association_source_witness" in report.issues
    assert "incomplete_geometry_association" in report.issues


def _cross_strata_face_image_fixture() -> tuple[
    CellMesh,
    CellGeometrySpec,
    MeshingDomainBoundarySource,
    SourceBoundaryFaceImageWitness,
]:
    from phydrax.discretization import CellGeometrySpec
    from phydrax.discretization._cell_geometry_validity import cell_geometry_id
    from phydrax.geometry._domain_boundary import MeshingDomainBoundarySource
    from phydrax.geometry._meshing_domain import (
        MeshingDomain,
        MeshingDomainCurve,
        MeshingDomainRegion,
        MeshingSurfacePatch,
        PatchCurveUse,
    )
    from phydrax.geometry._planar_primitives import signed_area2
    from phydrax.geometry._source_boundary_records import (
        rational_chart_tokens,
        SourceBoundaryFaceImageWitness,
    )
    from phydrax.geometry._source_fidelity import _volume_boundary_facet_lookup
    from phydrax.geometry.brep._patches import LineCurve
    from phydrax.geometry.reconstruction._core import _ReconstructedTrianglePatch

    coordinates = np.asarray(
        ((0, 0, 0), (1, 0, 0), (0, 1, 0), (0, 0, 1), (1.25, 0, 1), (0, 1, 1)),
        dtype=np.float64,
    )
    mesh = CellMesh(
        coordinates,
        (
            CellBlock(
                "prism",
                "prism",
                np.asarray(((0, 1, 2, 3, 4, 5),)),
                global_ids=np.asarray((7,)),
            ),
        ),
    )
    geometry = CellGeometrySpec.affine(mesh)
    a, b, c, d = coordinates[[0, 1, 4, 3]]
    extra = np.asarray(((3, 0, 0), (4, 0, 0), (3, 0, 1)), dtype=np.float64)
    corners = np.vstack((a, b, c, d, extra))
    curves = tuple(
        MeshingDomainCurve(first, last)
        for first, last in (
            (0, 1),
            (1, 3),
            (3, 0),
            (1, 2),
            (2, 3),
            (4, 5),
            (5, 6),
            (6, 4),
        )
    )
    triangle_uses = (
        (
            PatchCurveUse(0, LineCurve((0, 0), (1, 0)), 0, 1),
            PatchCurveUse(1, LineCurve((1, 0), (-1, 1)), 0, 1),
            PatchCurveUse(2, LineCurve((0, 1), (0, -1)), 0, 1),
        ),
        (
            PatchCurveUse(3, LineCurve((0, 0), (1, 0)), 0, 1),
            PatchCurveUse(4, LineCurve((1, 0), (-1, 1)), 0, 1),
            PatchCurveUse(1, LineCurve((0, 0), (0, 1)), 0, 1, sense=-1),
        ),
        (
            PatchCurveUse(5, LineCurve((0, 0), (1, 0)), 0, 1),
            PatchCurveUse(6, LineCurve((1, 0), (-1, 1)), 0, 1),
            PatchCurveUse(7, LineCurve((0, 1), (0, -1)), 0, 1),
        ),
    )
    domain = MeshingDomain(
        tuple(
            MeshingSurfacePatch(_ReconstructedTrianglePatch(corners[list(rows)]), (uses,))
            for rows, uses in zip(
                ((0, 1, 3), (1, 2, 3), (4, 5, 6)), triangle_uses, strict=True
            )
        ),
        curves,
        len(corners),
        source_id="face-image-source",
        source_revision="r1",
        regions=(MeshingDomainRegion("body", ((0, 1), (1, 1), (2, 1))),),
        corner_points=corners,
    )
    source = MeshingDomainBoundarySource(domain, (0, 1))
    source_triangles = tuple(
        tuple(
            tuple(Fraction.from_float(float(v)) for v in corners[index]) for index in rows
        )
        for rows in ((0, 1, 3), (1, 2, 3))
    )
    aa, bb, cc, dd = (
        tuple(Fraction.from_float(float(value)) for value in point)
        for point in (a, b, c, d)
    )
    target_triangles = ((aa, bb, cc), (aa, cc, dd))
    pieces = exact_coplanar_triangle_partitions(source_triangles, target_triangles)
    face_id, _ = _volume_boundary_facet_lookup(mesh)[(7, 2)]
    physical = tuple(
        tuple(
            _interpolate(source_triangles[piece.source_row], uv)
            for uv in piece.source_chart
        )
        for piece in pieces
    )
    ids = tuple(domain.entity_id(2, piece.source_row) for piece in pieces)
    witness = SourceBoundaryFaceImageWitness(
        mesh_id=mesh.mesh_id,
        geometry_id=cell_geometry_id(geometry),
        topology_id=mesh.topology_id,
        source_id=source.source_id,
        source_revision=source.source_revision,
        source_query_id=source.source_scope_id,
        cell_global_ids=np.full(len(pieces), 7),
        entity_global_ids=np.full(len(pieces), face_id),
        local_entity_indices=np.full(len(pieces), 2),
        source_chart_ids=ids,
        source_entity_ids=ids,
        occurrence_paths=tuple(
            domain.source_occurrences[2][piece.source_row] for piece in pieces
        ),
        source_region_sides=((domain.entity_id(3, 0), None),) * len(pieces),
        orientations=np.asarray(
            [1 if signed_area2(piece.source_chart) > 0 else -1 for piece in pieces]
        ),
        image_triangle_rational_tokens=rational_chart_tokens(physical),
    )
    return mesh, geometry, source, witness


def test_bilinear_face_image_retains_distinct_source_strata_and_rejects_forgery() -> None:
    from phydrax.geometry._source_boundary_records import SourceBoundaryFaceImageWitness
    from phydrax.geometry._source_fidelity import (
        certify_source_fidelity,
        validate_source_fidelity_witness,
    )

    mesh, geometry, source, witness = _cross_strata_face_image_fixture()
    assert len(set(witness.source_entity_ids)) == 2
    validated = validate_source_fidelity_witness(mesh, geometry, source, witness)
    certificate = certify_source_fidelity(
        mesh, geometry, source, tolerance=0.0, witness=validated
    )
    assert certificate.status == "certified"
    assert certificate.mesh_to_source_upper == certificate.source_to_mesh_upper == 0.0
    sides = list(witness.source_region_sides)
    sides[0] = (None, sides[0][0])
    forged = SourceBoundaryFaceImageWitness(
        mesh_id=witness.mesh_id,
        geometry_id=witness.geometry_id,
        topology_id=witness.topology_id,
        source_id=witness.source_id,
        source_revision=witness.source_revision,
        source_query_id=witness.source_query_id,
        cell_global_ids=witness.cell_global_ids,
        entity_global_ids=witness.entity_global_ids,
        local_entity_indices=witness.local_entity_indices,
        source_chart_ids=witness.source_chart_ids,
        source_entity_ids=witness.source_entity_ids,
        occurrence_paths=witness.occurrence_paths,
        source_region_sides=tuple(sides),
        orientations=witness.orientations,
        image_triangle_rational_tokens=witness.image_triangle_rational_tokens,
    )
    with pytest.raises(ValueError, match="material sides"):
        validate_source_fidelity_witness(mesh, geometry, source, forged)
    triangles = list(witness.image_triangle_rational_tokens)
    triangle = [list(point) for point in triangles[0]]
    triangle[0][0] = (triangle[0][0][0] + triangle[0][0][1], triangle[0][0][1])
    triangles[0] = tuple(tuple(point) for point in triangle)
    with pytest.raises(ValueError):
        forged = SourceBoundaryFaceImageWitness(
            mesh_id=witness.mesh_id,
            geometry_id=witness.geometry_id,
            topology_id=witness.topology_id,
            source_id=witness.source_id,
            source_revision=witness.source_revision,
            source_query_id=witness.source_query_id,
            cell_global_ids=witness.cell_global_ids,
            entity_global_ids=witness.entity_global_ids,
            local_entity_indices=witness.local_entity_indices,
            source_chart_ids=witness.source_chart_ids,
            source_entity_ids=witness.source_entity_ids,
            occurrence_paths=witness.occurrence_paths,
            source_region_sides=witness.source_region_sides,
            orientations=witness.orientations,
            image_triangle_rational_tokens=tuple(triangles),
        )
        validate_source_fidelity_witness(mesh, geometry, source, forged)


def test_image_face_support_reuses_exact_core_edges_and_checks_binding() -> None:
    import pickle
    from typing import Any, cast

    from phydrax.discretization import CellGeometrySpec
    from phydrax.geometry._source_fidelity import validate_source_fidelity_witness
    from phydrax.meshing._association import (
        _LayerImageSupportPreparation,
        _prepare_layer_image_support,
        _source_boundary_image_edge_intervals,
        _source_boundary_image_face_support,
        source_boundary_image_edge_intervals,
        source_boundary_image_face_support,
    )
    from phydrax.meshing._layer_core_boundary_replay import _image_support_for_audit
    from phydrax.meshing._layer_core_strata_replay import _source_face_support
    from tests.unit.meshing.test_layer_source_association_replay import _source

    mesh, geometry, source, witness = _cross_strata_face_image_fixture()
    validate_source_fidelity_witness(mesh, geometry, source, witness)
    expected = source_boundary_image_face_support(mesh, geometry, witness)
    layer_source = _source()
    prepared = _prepare_layer_image_support(layer_source, mesh, geometry, (7,))
    prepared.require(layer_source, mesh, geometry, (7,))
    with pytest.raises(ValueError, match="other source"):
        prepared.require(_source(), mesh, geometry, (7,))
    with pytest.raises(ValueError, match="core cells"):
        prepared.require(layer_source, mesh, geometry, ())
    with pytest.raises(ValueError, match="other source"):
        prepared.require(layer_source, mesh, geometry, (7,), root=cast(Any, object()))
    cache: list[_LayerImageSupportPreparation] = []
    first = _image_support_for_audit(
        layer_source, mesh, geometry, (7,), None, None, cache
    )
    second = _image_support_for_audit(
        layer_source, mesh, geometry, (7,), None, None, cache
    )
    assert first is second and len(cache) == 1
    other_geometry = CellGeometrySpec.affine(mesh)
    third = _image_support_for_audit(
        layer_source, mesh, other_geometry, (7,), None, None, cache
    )
    assert third is not first and len(cache) == 2
    with pytest.raises(ValueError, match="other source"):
        first.require(layer_source, mesh, other_geometry, (7,))
    with pytest.raises(TypeError, match="cannot be serialized"):
        pickle.dumps(prepared)
    assert _source_face_support(mesh, geometry, witness, prepared) == expected
    reused = _source_face_support(mesh, geometry, witness, prepared)
    for vertices, edges in reused.values():
        vertices.clear()
        edges.clear()
    assert _source_face_support(mesh, geometry, witness, prepared) == expected
    expected_intervals = source_boundary_image_edge_intervals(mesh, geometry, witness)
    intervals = _source_boundary_image_edge_intervals(mesh, geometry, witness, prepared)
    assert intervals == expected_intervals
    intervals.clear()
    assert (
        _source_boundary_image_edge_intervals(mesh, geometry, witness, prepared)
        == expected_intervals
    )
    assert prepared.edge_maps
    assert (
        _source_boundary_image_face_support(mesh, geometry, witness, prepared) == expected
    )
    endpoints = dict(prepared.endpoints)
    assert endpoints
    assert (
        _source_boundary_image_face_support(mesh, geometry, witness, prepared) == expected
    )
    assert prepared.endpoints == endpoints
    fallback = _prepare_layer_image_support(_source(), mesh, geometry, ())
    assert (
        _source_boundary_image_face_support(mesh, geometry, witness, fallback) == expected
    )
    with pytest.raises(ValueError, match="another mesh or geometry"):
        _source_boundary_image_face_support(
            mesh, CellGeometrySpec.affine(mesh), witness, prepared
        )


def test_face_image_receipt_reopens_and_reproves_in_fresh_process(tmp_path: Path) -> None:
    import subprocess
    import sys

    from phydrax.discretization._cell_geometry_validity import cell_geometry_id
    from phydrax.lifecycle import Composition, CompositionEntry
    from phydrax.meshing._model_codec import (
        model_archive_content_id,
        MODEL_ARCHIVE_LIMITS,
        write_model_artifact,
    )

    mesh, geometry, source, witness = _cross_strata_face_image_fixture()
    values = (
        ("mesh", mesh, mesh.topology_id, mesh.mesh_id),
        ("geometry", geometry, mesh.topology_id, cell_geometry_id(geometry)),
        ("source", source, source.source_id, source.source_scope_id),
        ("witness", witness, mesh.topology_id, witness.witness_id),
    )
    entries = tuple(
        CompositionEntry(
            value,
            entry_id=name,
            role="model-state",
            owner_id="phydrax.geometry",
            structure_id=structure,
            revision_id=revision,
            semantics_id=f"face-image-{name}",
        )
        for name, value, structure, revision in values
    )
    path = write_model_artifact(
        tmp_path / "face-image.phx",
        Composition(entries, boundary_id="face-image-cold"),
        limits=MODEL_ARCHIVE_LIMITS,
    )
    receipt = model_archive_content_id(path, limits=MODEL_ARCHIVE_LIMITS)
    code = """
import sys
from phydrax.geometry._source_fidelity import validate_source_fidelity_witness
from phydrax.geometry._surface_source_archive import _geometry
from phydrax.lifecycle import Composition
from phydrax.meshing._model_codec import MODEL_ARCHIVE_LIMITS, read_model_artifact, restore_model_archive_value

archive = read_model_artifact(
    sys.argv[1], expected_type=Composition, restore=restore_model_archive_value,
    limits=MODEL_ARCHIVE_LIMITS, expected_content_id=sys.argv[2],
)
mesh = archive.value("mesh")
geometry = _geometry(archive.value("geometry"), mesh)
source = archive.value("source")
witness = archive.value("witness")
assert len(set(witness.source_entity_ids)) == 2
proof = validate_source_fidelity_witness(mesh, geometry, source, witness)
assert proof.binds(mesh, geometry, source)
sides = list(witness.source_region_sides)
sides[0] = (None, sides[0][0])
forged = type(witness)(
    mesh_id=witness.mesh_id, geometry_id=witness.geometry_id,
    topology_id=witness.topology_id, source_id=witness.source_id,
    source_revision=witness.source_revision, source_query_id=witness.source_query_id,
    cell_global_ids=witness.cell_global_ids, entity_global_ids=witness.entity_global_ids,
    local_entity_indices=witness.local_entity_indices, source_chart_ids=witness.source_chart_ids,
    source_entity_ids=witness.source_entity_ids, occurrence_paths=witness.occurrence_paths,
    source_region_sides=tuple(sides), orientations=witness.orientations,
    image_triangle_rational_tokens=witness.image_triangle_rational_tokens,
)
try:
    validate_source_fidelity_witness(mesh, geometry, source, forged)
except ValueError:
    pass
else:
    raise AssertionError("Restored source accepted forged material sides.")
"""
    subprocess.run(
        [sys.executable, "-c", code, str(path), receipt],
        check=True,
        text=True,
        capture_output=True,
    )


def test_mapped_piecewise_edge_curve_uses_exact_original_parameters() -> None:
    from phydrax.discretization._coordinate_enclosure import Expression
    from phydrax.meshing._layer_core_strata_replay import source_curve_piecewise_interval

    half = Fraction(1, 2)
    image: tuple[Expression, ...] = (
        {(1,): Fraction(1)},
        {(0,): Fraction(0)},
        {(0,): Fraction(0)},
    )
    first: tuple[Expression, ...] = (
        {(1,): half},
        {(0,): Fraction(0)},
        {(0,): Fraction(0)},
    )
    second: tuple[Expression, ...] = (
        {(0,): half, (1,): half},
        {(0,): Fraction(0)},
        {(0,): Fraction(0)},
    )
    pieces = (((Fraction(0), half), first), ((half, Fraction(1)), second))
    assert source_curve_piecewise_interval(image, pieces, (Fraction(0), Fraction(1)))
    reversed_image: tuple[Expression, ...] = (
        {(0,): Fraction(1), (1,): Fraction(-1)},
        *image[1:],
    )
    assert source_curve_piecewise_interval(
        reversed_image, pieces, (Fraction(1), Fraction(0))
    )
    assert not source_curve_piecewise_interval(
        image, pieces[:1], (Fraction(0), Fraction(1))
    )
    assert not source_curve_piecewise_interval(
        image, (pieces[0], (pieces[1][0], first)), (Fraction(0), Fraction(1))
    )


def test_exact_cad_edge_classifier_distinguishes_overlap_and_intersection() -> None:
    from phydrax.meshing._layer_core_edge_policy import _segment_relation

    first, last = _point(0, 0), _point(1, 0)
    assert _segment_relation(first, last, _point(Fraction(1, 4), 0), _point(2, 0)) == (
        (Fraction(1, 4), Fraction(1)),
        False,
    )
    assert _segment_relation(first, last, _point(1, 0), _point(2, 0)) == (None, False)
    assert _segment_relation(
        first, last, _point(Fraction(1, 2), -1), _point(Fraction(1, 2), 1)
    ) == (None, True)
    assert _segment_relation(first, last, _point(1, -1), _point(1, 1)) == (None, False)
    assert _segment_relation(
        first, last, _point(Fraction(1, 2), -1, 1), _point(Fraction(1, 2), 1, 1)
    ) == (None, False)


def test_grouped_cube_facet_interior_diagonals_are_not_physical_edges() -> None:
    from phydrax.discretization._reference_cell import reference_cell_topology
    from phydrax.meshing._layer_core_edge_policy import conforming_physical_edge_rows
    from tests.unit.meshing.test_native_mapped_hex_routes import _mapped_cube_source

    source, _ = _mapped_cube_source(shared_facets=True)
    complex_ = source.complex
    physical = {
        tuple(sorted(complex_.plc_edge_vertices[row]))
        for row in conforming_physical_edge_rows(complex_)
    }
    expected = {
        tuple(sorted(edge)) for edge in reference_cell_topology("hexahedron").entities[1]
    }
    assert physical == expected

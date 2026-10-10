#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from fractions import Fraction
from pathlib import Path

import jax.numpy as jnp
import numpy as np
import pytest

from phydrax import interchange
from phydrax._physical import SpatialCoordinateContract
from phydrax.geometry.brep._constructors import (
    assemble_brep_model,
    brep_box,
    brep_extrusion,
    brep_planar_face,
    brep_revolution,
    BRepTessellationPolicy,
    PlanarProfile,
    ProfileLoop,
    ProfilePlane,
)
from phydrax.geometry.brep._intersection import (
    RootEndpoint,
    TrimIntersectionRoot,
    TrimRootEndpoint,
)
from phydrax.geometry.brep._intersection_curve import CurveTrimSegment, SurfaceRegion
from phydrax.geometry.brep._model import BRepCurve, BRepGeometry, BRepModel, BRepPCurve
from phydrax.geometry.brep._patches import (
    AbstractSurfacePatch,
    ConePatch,
    LineCurve,
    PlanePatch,
)
from phydrax.geometry.brep._query import prepare_brep_query
from phydrax.geometry.brep._root_bindings import (
    BRepCurveSurfaceLift,
    BRepRootSupport,
    BRepVertexRoot,
    certify_native_root_alias,
    surface_chart_regular,
)
from phydrax.geometry.brep._sewing import (
    BRepSewingFailure,
    BRepSewingPolicy,
    BRepSewingResult,
    sew_brep,
)
from phydrax.units import MILLIMETER


CONTRACT = SpatialCoordinateContract(MILLIMETER)

type _Point = tuple[float, float, float]


def test_identity_sewing_preserves_oriented_closed_shell() -> None:
    box = brep_box((0.0, 0.0, 0.0), (1.0, 2.0, 3.0), coordinate_contract=CONTRACT)
    geometry = box.geometry
    assert geometry is not None
    result = sew_brep(geometry, box.orientation, box.topology.solid_faces)
    assert result.geometry.edge_use_balance(box.orientation) == (0,) * len(
        geometry.edge_curves
    )
    assert result.geometry.euler_characteristic() == 2
    assert result.geometry.solid_shells == ((0,),)
    # The zero-repair route publishes the authored incidence and identity lineage.
    assert result.geometry.edge_curves == geometry.edge_curves
    assert result.geometry.face_loops == geometry.face_loops
    assert result.lineage.coedge_sources == tuple(range(len(geometry.coedge_edges)))
    assert all(
        image.source_edge == image.target_edge
        and image.scale == 1
        and not image.offset_rational
        and not image.offset_turns
        for image in result.lineage.edge_images
    )


def test_opposite_face_orientation_fails_without_mutating_geometry() -> None:
    box = brep_box((0.0, 0.0, 0.0), (1.0, 1.0, 1.0), coordinate_contract=CONTRACT)
    geometry = box.geometry
    assert geometry is not None
    original = geometry.geometry_id
    orientation = np.asarray(box.orientation).copy()
    orientation[0] *= -1.0
    with pytest.raises(BRepSewingFailure) as raised:
        sew_brep(geometry, orientation, box.topology.solid_faces)
    assert raised.value.edges
    assert geometry.geometry_id == original


def test_unclosed_region_ownership_is_not_repaired_by_sewing() -> None:
    box = brep_box((0.0, 0.0, 0.0), (1.0, 1.0, 1.0), coordinate_contract=CONTRACT)
    geometry = box.geometry
    assert geometry is not None
    with pytest.raises(BRepSewingFailure, match="open"):
        sew_brep(geometry, box.orientation, ((0,), tuple(range(1, 6))))


def test_sewing_requires_explicit_no_proximity_repair_policy() -> None:
    with pytest.raises(ValueError, match="proximity repair"):
        BRepSewingPolicy(tolerance=1e-6)


def test_coedge_work_budget_is_refused_before_candidate_publication() -> None:
    box = brep_box((0.0, 0.0, 0.0), (1.0, 1.0, 1.0), coordinate_contract=CONTRACT)
    geometry = box.geometry
    assert geometry is not None
    with pytest.raises(BRepSewingFailure, match="budget exhausted"):
        sew_brep(
            geometry,
            box.orientation,
            box.topology.solid_faces,
            policy=BRepSewingPolicy(maximum_coedges=1),
        )


def test_revolved_ring_tessellation_retains_complete_source_fidelity() -> None:
    policy = BRepTessellationPolicy()
    profile = PlanarProfile(
        ProfilePlane((0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 0.0, 1.0)),
        ProfileLoop.polygon(((2.0, 0.0), (3.0, 0.0), (3.0, 1.0), (2.0, 1.0))),
    )
    model = brep_revolution(
        profile,
        (0.0, 0.0),
        (0.0, 1.0),
        coordinate_contract=CONTRACT,
        tessellation=policy,
    )
    assert (
        np.max(np.asarray(model.tessellation_deviation_bounds))
        <= policy.linear_deflection
    )
    assert (
        np.max(np.asarray(model.tessellation_normal_bounds)) <= policy.angular_deflection
    )
    assert float(prepare_brep_query(model).measures.solid_volumes[0]) == pytest.approx(
        5.0 * np.pi,
        rel=1e-9,
    )


# ------------------------------------------------- independently authored faces


def _rectangle(
    origin: _Point, x_axis: _Point, y_axis: _Point, size: tuple[float, float]
) -> BRepModel:
    """One planar face whose own frame and edge parameterizations are authored."""
    width, height = size
    return brep_planar_face(
        PlanarProfile(
            ProfilePlane(origin, x_axis, y_axis),
            ProfileLoop.polygon(
                ((0.0, 0.0), (width, 0.0), (width, height), (0.0, height))
            ),
        ),
        coordinate_contract=CONTRACT,
    )


def _disc(
    origin: _Point,
    x_axis: _Point,
    y_axis: _Point,
    radius: float,
    center: tuple[float, float] = (0.0, 0.0),
) -> BRepModel:
    return brep_planar_face(
        PlanarProfile(
            ProfilePlane(origin, x_axis, y_axis), ProfileLoop.circle(center, radius)
        ),
        coordinate_contract=CONTRACT,
    )


def _cube(x: float) -> tuple[BRepModel, ...]:
    """Six separately authored outward-framed faces of ``[x, x+1] x [0, 1]^2``."""
    return (
        _rectangle((x, 0.0, 0.0), (0.0, 1.0, 0.0), (1.0, 0.0, 0.0), (1.0, 1.0)),
        _rectangle((x, 0.0, 1.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (1.0, 1.0)),
        _rectangle((x, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 0.0, 1.0), (1.0, 1.0)),
        _rectangle((x + 1.0, 1.0, 0.0), (-1.0, 0.0, 0.0), (0.0, 0.0, 1.0), (1.0, 1.0)),
        _rectangle((x, 0.0, 0.0), (0.0, 0.0, 1.0), (0.0, 1.0, 0.0), (1.0, 1.0)),
        _rectangle((x + 1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0), (1.0, 1.0)),
    )


def _split_top_slab(top_height: float = 1.0) -> tuple[BRepModel, ...]:
    """``[0, 2] x [0, 1]^2`` whose top is two coplanar faces in different frames."""
    return (
        _rectangle((0.0, 0.0, 0.0), (0.0, 1.0, 0.0), (1.0, 0.0, 0.0), (1.0, 2.0)),
        _rectangle((0.0, 0.0, top_height), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (1.0, 1.0)),
        _rectangle(
            (2.0, 1.0, top_height), (-1.0, 0.0, 0.0), (0.0, -1.0, 0.0), (1.0, 1.0)
        ),
        _rectangle((0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 0.0, 1.0), (2.0, 1.0)),
        _rectangle((2.0, 1.0, 0.0), (-1.0, 0.0, 0.0), (0.0, 0.0, 1.0), (2.0, 1.0)),
        _rectangle((0.0, 0.0, 0.0), (0.0, 0.0, 1.0), (0.0, 1.0, 0.0), (1.0, 1.0)),
        _rectangle((2.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0), (1.0, 1.0)),
    )


def _cylinder(top: BRepModel) -> tuple[tuple[BRepModel, int], ...]:
    """Extruded lateral face plus separately framed caps of a unit-radius, height-2 cylinder."""
    solid = brep_extrusion(
        PlanarProfile(ProfilePlane(), ProfileLoop.circle((0.0, 0.0), 1.0)),
        (0.0, 0.0, 2.0),
        coordinate_contract=CONTRACT,
    )
    # The bottom frame reverses the source circle; the top frame is a quarter turn.
    bottom = _disc((0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, -1.0, 0.0), 1.0)
    return (solid, solid.physical_tags.index("cylinder")), (bottom, 0), (top, 0)


@pytest.fixture
def quarter_turn_cap() -> BRepModel:
    return _disc((0.0, 0.0, 2.0), (0.0, 1.0, 0.0), (-1.0, 0.0, 0.0), 1.0)


def _face_soup(
    parts: tuple[tuple[BRepModel, int], ...],
) -> tuple[
    BRepGeometry,
    tuple[AbstractSurfacePatch, ...],
    np.ndarray,
    np.ndarray,
    tuple[str, ...],
]:
    """Disjoint union of authored faces: no vertex, edge or carrier is shared."""
    points: list[np.ndarray] = []
    vertex_roots: list[BRepVertexRoot | None] = []
    curves: list[BRepCurve] = []
    edge_curves: list[int] = []
    ranges: list[np.ndarray] = []
    edge_vertices: list[tuple[int, int]] = []
    edge_roots: list[tuple[RootEndpoint | None, RootEndpoint | None]] = []
    pcurves: list[BRepPCurve] = []
    coedge_edges: list[int] = []
    senses: list[int] = []
    coedge_roots: list[tuple[RootEndpoint | None, RootEndpoint | None]] = []
    face_loops: list[tuple[tuple[int, ...], ...]] = []
    for model, face in parts:
        geometry = model.geometry
        assert geometry is not None
        vertices: dict[int, int] = {}
        edges: dict[int, int] = {}
        carriers: dict[int, int] = {}
        loops: list[tuple[int, ...]] = []
        for loop in geometry.face_loops[face]:
            coedges: list[int] = []
            for coedge in loop:
                edge = geometry.coedge_edges[coedge]
                if edge not in edges:
                    for vertex in geometry.edge_vertices[edge]:
                        if vertex not in vertices:
                            vertices[vertex] = len(points)
                            points.append(np.asarray(geometry.vertex_points)[vertex])
                            vertex_roots.append(geometry.vertex_roots[vertex])
                    curve = geometry.edge_curves[edge]
                    if curve != -1 and curve not in carriers:
                        carriers[curve] = len(curves)
                        curves.append(geometry.curves[curve])
                    edges[edge] = len(edge_curves)
                    edge_curves.append(-1 if curve == -1 else carriers[curve])
                    ranges.append(np.asarray(geometry.edge_ranges)[edge])
                    start, end = geometry.edge_vertices[edge]
                    edge_vertices.append((vertices[start], vertices[end]))
                    edge_roots.append(geometry.edge_endpoint_roots[edge])
                coedges.append(len(coedge_edges))
                coedge_edges.append(edges[edge])
                senses.append(geometry.coedge_senses[coedge])
                pcurves.append(geometry.pcurves[coedge])
                coedge_roots.append(geometry.coedge_endpoint_roots[coedge])
            loops.append(tuple(coedges))
        face_loops.append(tuple(loops))
    soup = BRepGeometry(
        vertex_points=np.asarray(points),
        curves=tuple(curves),
        edge_curves=tuple(edge_curves),
        edge_ranges=np.asarray(ranges),
        edge_vertices=tuple(edge_vertices),
        pcurves=tuple(pcurves),
        coedge_edges=tuple(coedge_edges),
        coedge_senses=tuple(senses),
        face_loops=tuple(face_loops),
        shell_faces=(),
        shell_orientations=(),
        solid_shells=(),
        vertex_roots=tuple(vertex_roots),
        edge_endpoint_roots=tuple(edge_roots),
        coedge_endpoint_roots=tuple(coedge_roots),
    )
    patches = tuple(model.patches[face] for model, face in parts)
    bounds = np.stack([np.asarray(model.parameter_bounds)[face] for model, face in parts])
    orientation = np.asarray(
        [np.asarray(model.orientation)[face] for model, face in parts], dtype=np.float64
    )
    return (
        soup,
        patches,
        bounds,
        orientation,
        tuple(model.physical_tags[face] for model, face in parts),
    )


def _sewn_model(
    result: BRepSewingResult,
    patches: tuple[AbstractSurfacePatch, ...],
    bounds: np.ndarray,
    orientation: np.ndarray,
    tags: tuple[str, ...],
) -> BRepModel:
    return assemble_brep_model(
        result.geometry,
        patches,
        bounds,
        orientation,
        tags,
        coordinate_contract=CONTRACT,
        source_id=f"native-sewing:{result.certificate_id}",
        source_format="native-sewing",
        source_digest=result.certificate_id,
        import_policy_id=result.certificate_id,
    )


def _exact_images_agree(source: BRepGeometry, result: BRepSewingResult) -> None:
    """Independent oracle: each source carrier at ``s = scale*t + offset`` is the target carrier."""
    target = result.geometry
    for image in result.lineage.edge_images:
        if source.edge_curves[image.source_edge] == -1:
            continue
        first, last = np.asarray(target.edge_ranges)[image.target_edge]
        t = np.linspace(first, last, 7)
        offset = float(image.offset_rational) + float(image.offset_turns) * 2.0 * np.pi
        s = float(image.scale) * t + offset
        expected = np.asarray(
            target.curves[target.edge_curves[image.target_edge]].evaluate(jnp.asarray(t))
        )
        actual = np.asarray(
            source.curves[source.edge_curves[image.source_edge]].evaluate(jnp.asarray(s))
        )
        np.testing.assert_allclose(actual, expected, rtol=0.0, atol=1e-12)
        source_first, source_last = np.asarray(source.edge_ranges)[image.source_edge]
        assert (
            source_first - 1e-12 <= min(s[0], s[-1])
            and max(s[0], s[-1]) <= source_last + 1e-12
        )


def _covered_source_range(
    result: BRepSewingResult, edge: int
) -> list[tuple[float, float]]:
    target = np.asarray(result.geometry.edge_ranges)
    spans = []
    for image in result.lineage.source_images(edge):
        offset = float(image.offset_rational) + float(image.offset_turns) * 2.0 * np.pi
        ends = sorted(
            float(image.scale) * value + offset for value in target[image.target_edge]
        )
        spans.append((ends[0], ends[1]))
    return sorted(spans)


def test_coplanar_split_faces_fragment_t_junctions_into_closed_shell(
    tmp_path: Path,
) -> None:
    parts = tuple((model, 0) for model in _split_top_slab())
    soup, patches, bounds, orientation, tags = _face_soup(parts)
    original = soup.geometry_id
    result = sew_brep(soup, orientation, (tuple(range(len(parts))),))
    sewn = result.geometry
    assert soup.geometry_id == original
    assert sewn.solid_shells == ((0,),)
    assert sewn.edge_use_balance(orientation) == (0,) * len(sewn.edge_curves)
    # Independent incidence: corners plus two top T-junction vertices.
    assert (sewn.vertex_points.shape[0], len(sewn.edge_curves), len(sewn.face_loops)) == (
        10,
        15,
        7,
    )
    assert sewn.euler_characteristic() == 2
    uses = np.bincount(np.asarray(sewn.coedge_edges), minlength=len(sewn.edge_curves))
    assert np.all(uses == 2)
    assert (
        sorted(len(sources) for sources in result.lineage.vertex_sources)
        == [2, 2] + [3] * 8
    )
    # The long front/back top edges are fragmented at the authored split vertex.
    front_top = next(
        edge
        for edge in range(len(soup.edge_curves))
        if len(result.lineage.source_images(edge)) == 2
    )
    assert _covered_source_range(result, front_top) == [(0.0, 1.0), (1.0, 2.0)]
    assert (
        sum(
            len(result.lineage.source_images(edge)) == 2
            for edge in range(len(soup.edge_curves))
        )
        == 2
    )
    assert {image.orientation for image in result.lineage.edge_images} == {-1, 1}
    _exact_images_agree(soup, result)
    model = _sewn_model(result, patches, bounds, orientation, tags)
    measures = prepare_brep_query(model).measures
    assert float(measures.solid_volumes[0]) == pytest.approx(2.0, rel=1e-9)
    assert float(np.sum(measures.face_areas)) == pytest.approx(10.0, rel=1e-9)
    restored = interchange.load_brep_archive(
        interchange.save_brep_archive(model, tmp_path / "sewn.phx").path
    )
    assert restored.model_id == model.model_id
    assert restored.geometry is not None
    assert restored.geometry.geometry_id == sewn.geometry_id


def test_curved_faces_with_reflected_and_quarter_turn_frames_sew_exactly(
    quarter_turn_cap: BRepModel,
    tmp_path: Path,
) -> None:
    parts = _cylinder(quarter_turn_cap)
    soup, patches, bounds, orientation, tags = _face_soup(parts)
    result = sew_brep(soup, orientation, ((0, 1, 2),))
    sewn = result.geometry
    assert sewn.edge_use_balance(orientation) == (0,) * len(sewn.edge_curves)
    # Seam, one reflected bottom ring and the top ring cut at both authored seams.
    assert (sewn.vertex_points.shape[0], len(sewn.edge_curves)) == (3, 4)
    assert sewn.euler_characteristic() == 2
    top_cap_ring = soup.coedge_edges[soup.face_loops[2][0][0]]
    bottom_cap_ring = soup.coedge_edges[soup.face_loops[1][0][0]]
    bottom = result.lineage.source_images(bottom_cap_ring)
    assert len(bottom) == 1 and bottom[0].scale == -1 and bottom[0].offset_turns == 1
    top = result.lineage.source_images(top_cap_ring)
    assert sorted(
        (image.scale, image.offset_rational, image.offset_turns) for image in top
    ) == [
        (Fraction(1), Fraction(0), Fraction(-1, 4)),
        (Fraction(1), Fraction(0), Fraction(3, 4)),
    ]
    _exact_images_agree(soup, result)
    model = _sewn_model(result, patches, bounds, orientation, tags)
    measures = prepare_brep_query(model).measures
    reference = prepare_brep_query(
        brep_extrusion(
            PlanarProfile(ProfilePlane(), ProfileLoop.circle((0.0, 0.0), 1.0)),
            (0.0, 0.0, 2.0),
            coordinate_contract=CONTRACT,
        )
    ).measures
    assert float(measures.solid_volumes[0]) == pytest.approx(2.0 * np.pi, rel=1e-4)
    assert float(np.sum(measures.face_areas)) == pytest.approx(6.0 * np.pi, rel=1e-4)
    assert float(measures.solid_volumes[0]) == pytest.approx(
        float(reference.solid_volumes[0]), rel=1e-6
    )
    restored = interchange.load_brep_archive(
        interchange.save_brep_archive(model, tmp_path / "cylinder.phx").path
    )
    assert restored.model_id == model.model_id


def test_one_ulp_noncoincident_face_refuses_with_intersection_evidence() -> None:
    faces = _split_top_slab(top_height=float(np.nextafter(1.0, 2.0)))
    soup, _, _, orientation, _ = _face_soup(tuple((model, 0) for model in faces))
    original = soup.geometry_id
    with pytest.raises(BRepSewingFailure, match="after exact correspondence") as raised:
        sew_brep(soup, orientation, (tuple(range(len(faces))),))
    failure = raised.value
    assert soup.geometry_id == original
    assert failure.edges and failure.contacts
    assert any(contact.relations != ("separated",) for contact in failure.contacts)
    reported = {
        edge
        for contact in failure.contacts
        for edge in (contact.open_edge, contact.candidate_edge)
    }
    assert reported <= set(failure.edges)


def test_tangent_unequal_circles_refuse_with_tangent_evidence() -> None:
    tangent = _disc(
        (0.0, 0.0, 2.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0), 0.5, center=(0.5, 0.0)
    )
    soup, _, _, orientation, _ = _face_soup(_cylinder(tangent))
    original = soup.geometry_id
    with pytest.raises(BRepSewingFailure, match="after exact correspondence") as raised:
        sew_brep(soup, orientation, ((0, 1, 2),))
    assert any(
        {"tangent", "singular"} & set(contact.relations)
        for contact in raised.value.contacts
    )
    assert soup.geometry_id == original


def test_coincident_circle_support_without_exact_phase_is_refused() -> None:
    half = float(np.sqrt(0.5))
    rotated = _disc((0.0, 0.0, 2.0), (half, half, 0.0), (-half, half, 0.0), 1.0)
    soup, _, _, orientation, _ = _face_soup(_cylinder(rotated))
    original = soup.geometry_id
    with pytest.raises(
        BRepSewingFailure, match="no exactly representable parameter correspondence"
    ) as raised:
        sew_brep(soup, orientation, ((0, 1, 2),))
    assert raised.value.edges
    assert soup.geometry_id == original


def test_cross_region_edge_coincidence_without_shared_face_is_refused() -> None:
    faces = (*_cube(0.0), *_cube(1.0))
    soup, _, _, orientation, _ = _face_soup(tuple((model, 0) for model in faces))
    original = soup.geometry_id
    with pytest.raises(BRepSewingFailure, match="distinct regions") as raised:
        sew_brep(soup, orientation, (tuple(range(6)), tuple(range(6, 12))))
    assert len(raised.value.edges) >= 2
    assert soup.geometry_id == original


def test_correspondence_fragmentation_respects_coedge_budget() -> None:
    parts = tuple((model, 0) for model in _split_top_slab())
    soup, _, _, orientation, _ = _face_soup(parts)
    original = soup.geometry_id
    with pytest.raises(BRepSewingFailure, match="budget exhausted"):
        sew_brep(
            soup,
            orientation,
            (tuple(range(len(parts))),),
            policy=BRepSewingPolicy(maximum_coedges=len(soup.coedge_edges)),
        )
    assert soup.geometry_id == original


def test_native_face_roots_require_both_exact_spatial_carriers() -> None:
    primary_patch = PlanePatch(
        np.zeros(3), np.asarray((1.0, 0.0, 0.0)), np.asarray((0.0, 1.0, 0.0))
    )
    alias_patch = PlanePatch(
        np.zeros(3), np.asarray((0.0, 1.0, 0.0)), np.asarray((1.0, 0.0, 0.0))
    )
    first = LineCurve(np.zeros(2), np.asarray((1.0, 0.0)))
    second = LineCurve(np.zeros(2), np.asarray((0.0, 1.0)))
    first_trim = CurveTrimSegment(first, -1.0, 1.0)
    second_trim = CurveTrimSegment(second, -1.0, 1.0)
    primary = BRepRootSupport(
        primary_patch,
        TrimIntersectionRoot(
            first_trim,
            second_trim,
            parameter_lower=np.full(2, 0.3),
            parameter_upper=np.full(2, 0.7),
        ),
    )
    alias = BRepRootSupport(
        alias_patch,
        TrimIntersectionRoot(
            second_trim,
            first_trim,
            parameter_lower=np.full(2, 0.4),
            parameter_upper=np.full(2, 0.6),
        ),
    )
    x_edge = LineCurve(np.zeros(3), np.asarray((1.0, 0.0, 0.0)))
    y_edge = LineCurve(np.zeros(3), np.asarray((0.0, 1.0, 0.0)))
    bounds = np.asarray(((-1.0, -1.0), (1.0, 1.0)), dtype=np.float64)
    lifts = (
        BRepCurveSurfaceLift(
            x_edge, first, SurfaceRegion(primary_patch, bounds), -1.0, 1.0
        ),
        BRepCurveSurfaceLift(
            y_edge, second, SurfaceRegion(primary_patch, bounds), -1.0, 1.0
        ),
        BRepCurveSurfaceLift(
            x_edge, second, SurfaceRegion(alias_patch, bounds), -1.0, 1.0
        ),
        BRepCurveSurfaceLift(
            y_edge, first, SurfaceRegion(alias_patch, bounds), -1.0, 1.0
        ),
    )
    assert certify_native_root_alias(primary, alias, lifts)
    vertex = BRepVertexRoot(primary, aliases=(alias,), source_edge_lifts=lifts)
    assert vertex.supports_endpoint(TrimRootEndpoint(alias.root, "first"))
    assert not certify_native_root_alias(primary, alias, lifts[:-1])
    with pytest.raises(ValueError, match="common spatial"):
        BRepVertexRoot(primary, aliases=(alias,), source_edge_lifts=lifts[:-1])
    broad = BRepRootSupport(
        alias_patch,
        TrimIntersectionRoot(
            second_trim,
            first_trim,
            parameter_lower=np.full(2, 0.2),
            parameter_upper=np.full(2, 0.8),
        ),
    )
    assert not certify_native_root_alias(primary, broad, lifts)
    shifted = LineCurve(np.asarray((0.0, 0.01)), np.asarray((1.0, 0.0)))
    nearby = BRepRootSupport(
        alias_patch,
        TrimIntersectionRoot(
            second_trim,
            CurveTrimSegment(shifted, -1.0, 1.0),
            parameter_lower=np.full(2, 0.4),
            parameter_upper=np.full(2, 0.6),
        ),
    )
    shifted_edge = LineCurve(np.asarray((0.01, 0.0, 0.0)), np.asarray((0.0, 1.0, 0.0)))
    nearby_lifts = (
        *lifts[:3],
        BRepCurveSurfaceLift(
            shifted_edge,
            shifted,
            SurfaceRegion(alias_patch, bounds),
            -1.0,
            1.0,
        ),
    )
    # Both boxes overlap and one spatial edge is shared, but the second
    # carrier defines a distinct endpoint and cannot borrow the primary root.
    assert not certify_native_root_alias(primary, nearby, nearby_lifts)
    with pytest.raises(ValueError, match="common spatial"):
        BRepVertexRoot(primary, aliases=(nearby,), source_edge_lifts=nearby_lifts)


def test_collapsed_cone_apex_cannot_borrow_a_planar_root() -> None:
    first = LineCurve(np.zeros(2), np.asarray((1.0, 0.0)))
    second = LineCurve(np.zeros(2), np.asarray((0.0, 1.0)))
    primary = BRepRootSupport(
        PlanePatch(
            np.zeros(3),
            np.asarray((1.0, 0.0, 0.0)),
            np.asarray((0.0, 1.0, 0.0)),
        ),
        TrimIntersectionRoot(
            CurveTrimSegment(first, -1.0, 1.0),
            CurveTrimSegment(second, -1.0, 1.0),
            parameter_lower=np.full(2, 0.4),
            parameter_upper=np.full(2, 0.6),
        ),
    )
    apex = -1.0
    cone = ConePatch(
        np.zeros(3),
        np.asarray((1.0, 0.0, 0.0)),
        np.asarray((0.0, 1.0, 0.0)),
        np.asarray((0.0, 0.0, 1.0)),
        1.0,
        0.25 * np.pi,
    )
    first_at_apex = LineCurve(np.asarray((0.0, apex)), np.asarray((1.0, 0.0)))
    second_at_apex = LineCurve(np.asarray((0.0, apex)), np.asarray((0.0, 1.0)))
    collapsed = BRepRootSupport(
        cone,
        TrimIntersectionRoot(
            CurveTrimSegment(first_at_apex, -1.0, 1.0),
            CurveTrimSegment(second_at_apex, -1.0, 1.0),
            parameter_lower=np.full(2, 0.4),
            parameter_upper=np.full(2, 0.6),
        ),
    )
    assert not surface_chart_regular(cone, collapsed.root.point_enclosure())
    assert not certify_native_root_alias(primary, collapsed, ())

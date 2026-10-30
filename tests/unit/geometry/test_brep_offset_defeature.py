from __future__ import annotations

from fractions import Fraction

import numpy as np
import pytest

from phydrax._physical import SpatialCoordinateContract
from phydrax.geometry.brep._boolean import boolean_brep
from phydrax.geometry.brep._constructors import (
    _Builder,
    assemble_brep_model,
    brep_box,
    brep_cylinder,
    brep_extrusion,
    brep_offset,
    brep_revolution,
    brep_sphere,
    brep_torus,
    PlanarProfile,
    ProfileCurveUse,
    ProfileLoop,
    ProfilePlane,
)
from phydrax.geometry.brep._defeature import (
    _hole_wall_data,
    brep_defeature,
    BRepDefeatureResult,
)
from phydrax.geometry.brep._fillet import brep_chamfer, brep_fillet
from phydrax.geometry.brep._intersection import NativePeriodEndpoint
from phydrax.geometry.brep._model import BRepModel
from phydrax.geometry.brep._offset_result import (
    BRepOffsetResult,
    certify_revolution_offset_embedding,
)
from phydrax.geometry.brep._patches import (
    BSplineCurve,
    CircleCurve,
    CylinderPatch,
    EllipseCurve,
    ExtrusionSurface,
    LineCurve,
    OffsetSurface,
    RevolutionSurface,
    SurfaceIsoparametricCurve,
    SurfaceMappedCurve,
    TorusPatch,
)
from phydrax.geometry.brep._planar_offset import planar_carrier


_CONTRACT = SpatialCoordinateContract.si()


def _box() -> BRepModel:
    return brep_box((0.0, 0.0, 0.0), (2.0, 3.0, 4.0), coordinate_contract=_CONTRACT)


def _snapshot(model: BRepModel) -> tuple[object, ...]:
    geometry = model.geometry
    return (
        model.model_id,
        geometry.geometry_id,
        np.asarray(geometry.vertex_points).copy(),
        geometry.edge_vertices,
        geometry.face_loops,
    )


def _assert_unchanged(model: BRepModel, snapshot: tuple[object, ...]) -> None:
    current = _snapshot(model)
    assert current[:2] == snapshot[:2]
    np.testing.assert_array_equal(current[2], snapshot[2])
    assert current[3:] == snapshot[3:]


def _assert_offset_lineage(source: BRepModel, result: BRepOffsetResult) -> None:
    target = result.model
    assert target is not None
    assert result.source_model_id == source.model_id
    for kind in ("face", "edge", "vertex", "solid"):
        pairs = getattr(result, f"{kind}_images")
        assert all(source_id in getattr(source, f"{kind}_ids") for source_id, _ in pairs)
        assert all(target_id in getattr(target, f"{kind}_ids") for _, target_id in pairs)
        source_paths = {
            (source_id.index, source_id.occurrence_path) for source_id, _ in pairs
        }
        target_paths = {
            (target_id.index, target_id.occurrence_path) for _, target_id in pairs
        }
        assert source_paths == target_paths
        assert {index for index, _ in source_paths} == set(
            range(len(getattr(source, f"{kind}_ids")))
        )
    graph = result.association_graph
    assert graph.source_revision.revision_id == source.model_id
    assert graph.target_revision.revision_id == target.model_id
    assert graph.transaction.coverage.source_exhaustive
    assert graph.transaction.coverage.target_exhaustive
    assert not graph.transaction.unresolved_source_ids
    assert not graph.transaction.unresolved_target_ids


def _assert_defeature_lineage(source: BRepModel, result: BRepDefeatureResult) -> None:
    target = result.model
    assert target is not None
    assert result.source_model_id == source.model_id
    assert result.removed_face.kind == "face"
    assert result.removed_face in source.face_ids
    assert result.removed_face.source_revision == source.model_id
    for kind in ("face", "edge", "vertex", "solid"):
        for source_id, target_id in getattr(result, f"{kind}_images"):
            assert source_id.kind == kind
            assert target_id.kind == kind
            assert source_id in getattr(source, f"{kind}_ids")
            assert target_id in getattr(target, f"{kind}_ids")
            assert source_id.source_revision == source.model_id
            assert target_id.source_revision == target.model_id
            assert source_id.occurrence_path == target_id.occurrence_path
    mapped_targets = {
        target_id
        for kind in ("face", "edge", "vertex", "solid")
        for _, target_id in getattr(result, f"{kind}_images")
    }
    for target_id, support_ids in result.created_entity_sources:
        assert target_id not in mapped_targets
        assert target_id in getattr(target, f"{target_id.kind}_ids")
        assert support_ids
        assert all(source_id in source.face_ids for source_id in support_ids)
    graph = result.association_graph
    assert graph is not None
    assert graph.source_revision.revision_id == source.model_id
    assert graph.target_revision.revision_id == target.model_id
    assert graph.transaction.coverage.source_exhaustive
    assert graph.transaction.coverage.target_exhaustive
    assert not graph.transaction.unresolved_source_ids
    assert not graph.transaction.unresolved_target_ids


def _vertical_edge(model: BRepModel) -> int:
    geometry = model.geometry
    points = np.asarray(geometry.vertex_points)
    for edge, (first, last) in enumerate(geometry.edge_vertices):
        if np.array_equal(np.abs(points[last] - points[first]), (0.0, 0.0, 4.0)):
            return edge
    raise AssertionError("The box fixture has no vertical edge.")


def _plate_with_cylindrical_hole() -> BRepModel:
    profile = PlanarProfile(
        ProfilePlane((0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 0.0, 1.0)),
        ProfileLoop.polygon(((0.5, 0.0), (2.0, 0.0), (2.0, 1.0), (0.5, 1.0))),
    )
    return brep_revolution(profile, (0.0, 0.0), (0.0, 1.0), coordinate_contract=_CONTRACT)


def _source_with_conical_hole() -> BRepModel:
    profile = PlanarProfile(
        ProfilePlane((0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 0.0, 1.0)),
        ProfileLoop.polygon(((0.5, 0.0), (2.0, 0.0), (2.0, 1.0), (0.75, 1.0))),
    )
    return brep_revolution(profile, (0.0, 0.0), (0.0, 1.0), coordinate_contract=_CONTRACT)


def _rounded_l_with_hole() -> BRepModel:
    outer = brep_box((0.0, 0.0, 0.0), (3.0, 3.0, 2.0), coordinate_contract=_CONTRACT)
    notch = brep_box((1.0, 1.0, -1.0), (4.0, 4.0, 3.0), coordinate_contract=_CONTRACT)
    cutout = boolean_brep(outer, notch, "difference")
    assert not cutout.empty
    assert cutout.model is not None
    assert cutout.model.topology.num_solids == 1
    sharp = cutout.model
    geometry = sharp.geometry
    points = np.asarray(geometry.vertex_points)
    edge = next(
        index
        for index, (start, end) in enumerate(geometry.edge_vertices)
        if np.array_equal(points[start, :2], (1.0, 1.0))
        and np.array_equal(points[end, :2], (1.0, 1.0))
        and abs(points[end, 2] - points[start, 2]) == 2.0
    )
    blend = brep_fillet(sharp, (sharp.edge_ids[edge],), 0.25)
    assert blend.status == "certified", blend.reason
    assert blend.model is not None
    cutter = brep_cylinder(
        0.015625,
        4.0,
        base_center=(1.0, 1.125, -1.0),
        coordinate_contract=_CONTRACT,
    )
    cut = boolean_brep(blend.model, cutter, "difference")
    assert cut.model is not None
    assert not cut.empty
    assert cut.model.topology.num_solids == 1
    return cut.model


def _closed_bspline_revolution_source() -> BRepModel:
    meridian = BSplineCurve(
        np.asarray(
            (
                (2.5, 0.0, 0.0),
                (2.5, 0.0, 0.5),
                (2.2, 0.0, 0.7),
                (1.7, 0.0, 0.7),
                (1.4, 0.0, 0.0),
                (1.7, 0.0, -0.7),
                (2.2, 0.0, -0.7),
                (2.5, 0.0, -0.5),
                (2.5, 0.0, 0.0),
            ),
            dtype=np.float64,
        ),
        np.ones((9,), dtype=np.float64),
        np.asarray((0.0, 0.0, 0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 7.0, 7.0)),
        2,
    )
    patch = RevolutionSurface(meridian, (0.0, 0.0, 0.0), (0.0, 0.0, 1.0))
    builder = _Builder()
    vertex = builder.vertex(
        (2.5, 0.0, 0.0), exact_point=(Fraction(5, 2), Fraction(0), Fraction(0))
    )
    turn = builder.edge(
        CircleCurve((0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0), 2.5),
        0.0,
        float(2.0 * np.pi),
        vertex,
        vertex,
    )
    seam = builder.edge(meridian, 0.0, 7.0, vertex, vertex)
    coedges = (
        builder.coedge(turn, 1, LineCurve((0.0, 0.0), (1.0, 0.0))),
        builder.coedge(seam, 1, LineCurve((0.0, 0.0), (0.0, 1.0))),
        builder.coedge(turn, -1, LineCurve((0.0, 7.0), (1.0, 0.0))),
        builder.coedge(seam, -1, LineCurve((0.0, 0.0), (0.0, 1.0))),
    )
    builder.face(
        patch,
        ((0.0, 0.0), (float(2.0 * np.pi), 7.0)),
        1,
        "closed-bspline-revolution",
        [list(coedges)],
    )
    builder.native_phase(turn, patch, 0, coedge=coedges[0])
    builder.native_phase(turn, patch, 0, coedge=coedges[2])
    geometry, patches, bounds, orientations, tags, trims = builder.geometry(solid=True)
    return assemble_brep_model(
        geometry,
        patches,
        bounds,
        orientations,
        tags,
        coordinate_contract=_CONTRACT,
        source_id="closed-bspline-revolution-source",
        trim_domains=trims,
    )


def test_planar_box_offset_publishes_rounding_volume_and_exact_lineage() -> None:
    source = _box()
    before = _snapshot(source)
    distance = 0.1

    result = brep_offset(source, distance, include_parameter_range=True)

    _assert_unchanged(source, before)
    _assert_offset_lineage(source, result)
    assert result.parameter_range is not None
    assert result.parameter_range.parameter == "distance"
    assert result.parameter_range.interval[0] == -1.0
    assert result.parameter_range.end_status[0] == "attained_exactly"
    evidence = result.evidence
    assert evidence.face_rounding_bounds is not None
    assert evidence.vertex_rounding_bounds is not None
    assert evidence.volume_status == "certified"
    assert evidence.volume_enclosure is not None
    exact_volume = (
        (Fraction(2) + 2 * Fraction(distance))
        * (Fraction(3) + 2 * Fraction(distance))
        * (Fraction(4) + 2 * Fraction(distance))
    )
    assert Fraction(float(evidence.volume_enclosure[0])) <= exact_volume
    assert exact_volume <= Fraction(float(evidence.volume_enclosure[1]))
    geometry = result.model.geometry
    if geometry.edge_vertex_gap_bounds is not None:
        assert len(geometry.edge_vertex_gap_bounds) == len(geometry.edge_vertices)
    if geometry.edge_face_gap_bounds is not None:
        assert len(geometry.edge_face_gap_bounds) == len(geometry.coedge_edges)
    assert np.any(np.asarray(evidence.vertex_rounding_bounds) > 0.0)


def test_planar_concave_extrusion_offset_preserves_every_face_image() -> None:
    profile = PlanarProfile(
        ProfilePlane(),
        ProfileLoop.polygon(
            ((0.0, 0.0), (2.0, 0.0), (2.0, 1.0), (1.0, 1.0), (1.0, 2.0), (0.0, 2.0))
        ),
    )
    source = brep_extrusion(profile, (0.0, 0.0, 2.0), coordinate_contract=_CONTRACT)
    before = _snapshot(source)

    result = brep_offset(source, 0.125)

    _assert_unchanged(source, before)
    _assert_offset_lineage(source, result)
    assert result.evidence.volume_status == "certified"


def test_line_bspline_extrusion_offset_uses_analytic_plane_recognition() -> None:
    base = _box()
    patches = list(base.patches)
    face = next(
        index
        for index, patch in enumerate(patches)
        if isinstance(patch, ExtrusionSurface)
    )
    plane = planar_carrier(patches[face])
    assert plane is not None
    curve = BSplineCurve(
        np.stack(
            (
                np.asarray(plane.origin),
                np.asarray(plane.origin) + np.asarray(plane.first_axis),
            )
        ),
        np.ones((2,), dtype=np.float64),
        np.asarray((0.0, 0.0, 1.0, 1.0), dtype=np.float64),
        1,
    )
    patches[face] = ExtrusionSurface(curve, plane.second_axis)
    source = assemble_brep_model(
        base.geometry,
        tuple(patches),
        np.asarray(base.parameter_bounds),
        np.asarray(base.orientation),
        base.physical_tags,
        coordinate_contract=base.coordinate_contract,
        source_id="line-bspline-extrusion",
        trim_domains=base.trim_domains,
    )
    assert isinstance(source.patches[face], ExtrusionSurface)
    assert planar_carrier(source.patches[face]) is not None
    before = _snapshot(source)

    result = brep_offset(source, 0.125)

    _assert_unchanged(source, before)
    _assert_offset_lineage(source, result)


def test_single_chart_analytic_torus_offset_binds_its_source_entities() -> None:
    source = brep_torus(2.0, 0.5, coordinate_contract=_CONTRACT)
    before = _snapshot(source)

    result = brep_offset(source, 0.125)

    _assert_unchanged(source, before)
    _assert_offset_lineage(source, result)
    assert result.evidence.volume_status == "unresolved"
    assert result.evidence.volume_enclosure is None


def test_single_chart_analytic_sphere_offset_binds_pole_occurrences() -> None:
    source = brep_sphere(1.0, coordinate_contract=_CONTRACT)
    before = _snapshot(source)

    result = brep_offset(source, 0.125)

    _assert_unchanged(source, before)
    _assert_offset_lineage(source, result)
    target = result.model
    assert target is not None
    source_patch = source.patches[0]
    target_patch = target.patches[0]
    assert isinstance(source_patch, RevolutionSurface)
    assert isinstance(target_patch, OffsetSurface)
    assert target_patch.base is source_patch
    assert target_patch.native_period_symbols() == source_patch.native_period_symbols()
    np.testing.assert_array_equal(
        np.asarray(target.parameter_bounds), np.asarray(source.parameter_bounds)
    )
    edge_map = {
        source_id.index: target_id.index for source_id, target_id in result.edge_images
    }
    degenerate_edges = tuple(
        edge for edge, curve in enumerate(source.geometry.edge_curves) if curve < 0
    )
    assert len(degenerate_edges) == 2
    for edge in degenerate_edges:
        source_roots = source.geometry.edge_endpoint_roots[edge]
        target_roots = target.geometry.edge_endpoint_roots[edge_map[edge]]
        assert all(isinstance(root, NativePeriodEndpoint) for root in source_roots)
        assert all(isinstance(root, NativePeriodEndpoint) for root in target_roots)
        for source_root, target_root in zip(source_roots, target_roots, strict=True):
            assert isinstance(source_root, NativePeriodEndpoint)
            assert isinstance(target_root, NativePeriodEndpoint)
            assert source_root.axis == target_root.axis == 0
            assert source_root.rational == target_root.rational
            assert source_root.turns == target_root.turns
            assert source_root.patch is source_patch
            assert target_root.patch is target_patch
    assert result.evidence.volume_status == "unresolved"
    assert result.evidence.volume_enclosure is None


def _remove_blended_box_edge(operation: str) -> tuple[BRepModel, BRepModel, int]:
    box = _box()
    edge = _vertical_edge(box)
    if operation == "fillet":
        blend = brep_fillet(box, (box.edge_ids[edge],), 0.25)
    else:
        blend = brep_chamfer(box, (box.edge_ids[edge],), (0.25, 0.5))
    assert blend.status == "certified", blend.reason
    assert blend.model is not None
    source = blend.model
    if operation == "fillet":
        face = next(
            index
            for index, patch in enumerate(source.patches)
            if isinstance(patch, CylinderPatch)
        )
    else:
        face = next(
            index
            for index, patch in enumerate(source.patches)
            if (plane := planar_carrier(patch)) is not None
            and np.count_nonzero(
                np.abs(np.cross(plane.first_axis, plane.second_axis))[:2]
            )
            == 2
        )
    return box, source, face


def test_plane_pair_fillet_defeature_publishes_exact_lineage() -> None:
    base, source, face = _remove_blended_box_edge("fillet")
    base_before, source_before = _snapshot(base), _snapshot(source)

    result = brep_defeature(source, source.face_ids[face], operation="fillet")

    _assert_unchanged(base, base_before)
    _assert_unchanged(source, source_before)
    assert result.status == "certified", result.reason
    assert result.intersection_kind == "line"
    assert result.created_entity_sources
    _assert_defeature_lineage(source, result)


def test_plane_pair_chamfer_defeature_publishes_exact_lineage() -> None:
    base, source, face = _remove_blended_box_edge("chamfer")
    base_before, source_before = _snapshot(base), _snapshot(source)

    result = brep_defeature(source, source.face_ids[face], operation="chamfer")

    _assert_unchanged(base, base_before)
    _assert_unchanged(source, source_before)
    assert result.status == "certified", result.reason
    assert result.intersection_kind == "line"
    assert result.created_entity_sources
    _assert_defeature_lineage(source, result)


def test_cylindrical_through_hole_defeature_restores_the_capped_solid() -> None:
    source = _plate_with_cylindrical_hole()
    face = next(
        index
        for index, patch in enumerate(source.patches)
        if isinstance(patch, CylinderPatch) and float(patch.radius) == 0.5
    )
    before = _snapshot(source)

    result = brep_defeature(source, source.face_ids[face], operation="hole")

    _assert_unchanged(source, before)
    assert result.status == "certified", result.reason
    assert result.intersection_kind == "none"
    _assert_defeature_lineage(source, result)


def test_conical_through_hole_defeature_returns_enclosed_shape_change() -> None:
    source = _source_with_conical_hole()
    face = next(
        index
        for index, patch in enumerate(source.patches)
        if isinstance(patch, RevolutionSurface)
        and (wall := _hole_wall_data(source, index)) is not None
        and wall.family == "revolved-cone"
    )
    before = _snapshot(source)

    result = brep_defeature(source, source.face_ids[face], operation="hole")

    _assert_unchanged(source, before)
    assert result.status == "certified", result.reason
    assert result.volume_kind == "enclosed"
    assert result.volume_change_enclosure.shape == (2,)
    _assert_defeature_lineage(source, result)


def test_plane_cylinder_torus_blend_defeature_preserves_periodic_lineage() -> None:
    cylinder = brep_cylinder(2.0, 3.0, coordinate_contract=_CONTRACT)
    geometry = cylinder.geometry
    points = np.asarray(geometry.vertex_points)
    edge = next(
        index
        for index, (first, last) in enumerate(geometry.edge_vertices)
        if first == last and points[first, 2] == 3.0
    )
    blend = brep_fillet(cylinder, (cylinder.edge_ids[edge],), 0.25)
    assert blend.status == "certified", blend.reason
    assert blend.model is not None
    source = blend.model
    face = next(
        index
        for index, patch in enumerate(source.patches)
        if isinstance(patch, TorusPatch)
    )
    before = _snapshot(source)

    result = brep_defeature(source, source.face_ids[face], operation="fillet")

    _assert_unchanged(source, before)
    assert result.status == "certified", result.reason
    assert result.intersection_kind == "circle"
    assert result.created_entity_sources
    assert result.carrier_intent_points
    _assert_defeature_lineage(source, result)


def test_certified_collision_refusal_rolls_back_without_partial_output() -> None:
    source = _rounded_l_with_hole()
    face = next(
        index
        for index, patch in enumerate(source.patches)
        if isinstance(patch, CylinderPatch) and float(patch.radius) == 0.25
    )
    before = _snapshot(source)

    result = brep_defeature(source, source.face_ids[face], operation="fillet")

    _assert_unchanged(source, before)
    assert result.status == "collision", result.reason
    assert result.model is None
    assert result.association_graph is None
    assert result.volume_kind == "absent"


def test_unsupported_family_and_foreign_identity_leave_source_unchanged() -> None:
    sphere = brep_sphere(1.0, coordinate_contract=_CONTRACT)
    before = _snapshot(sphere)

    unsupported = brep_defeature(sphere, sphere.face_ids[0], operation="fillet")

    _assert_unchanged(sphere, before)
    assert unsupported.status == "unsupported"
    assert unsupported.model is None
    assert unsupported.association_graph is None

    sharp_cylinder = brep_cylinder(1.0, 2.0, coordinate_contract=_CONTRACT)
    cylinder_before = _snapshot(sharp_cylinder)
    with pytest.raises((NotImplementedError, ValueError), match="brep_offset"):
        brep_offset(sharp_cylinder, 0.1)
    _assert_unchanged(sharp_cylinder, cylinder_before)

    source = _box()
    foreign = brep_box((0.0, 0.0, 0.0), (3.0, 3.0, 4.0), coordinate_contract=_CONTRACT)
    source_before = _snapshot(source)
    with pytest.raises(ValueError, match="source revision"):
        brep_defeature(source, foreign.face_ids[0], operation="fillet")
    _assert_unchanged(source, source_before)


@pytest.mark.parametrize("distance", (0.2, -0.1), ids=("outward", "inward"))
def test_elliptic_revolution_offset_preserves_exact_surface_and_lineage(
    distance: float,
) -> None:
    ellipse = EllipseCurve((2.0, 0.0), (1.0, 0.0), (0.0, 1.0), 0.5, 0.25)
    profile = PlanarProfile(
        ProfilePlane(),
        ProfileLoop(
            ((2.5, 0.0),),
            (ProfileCurveUse(ellipse, 0.0, float(2.0 * np.pi)),),
        ),
    )
    source = brep_revolution(
        profile, (0.0, 0.0), (0.0, 1.0), coordinate_contract=_CONTRACT
    )
    before = _snapshot(source)

    result = brep_offset(source, distance, source_id="elliptic-revolution-offset")

    _assert_unchanged(source, before)
    _assert_offset_lineage(source, result)
    target = result.model
    assert target is not None
    assert target.source_id == "elliptic-revolution-offset"
    assert isinstance(target.patches[0], OffsetSurface)
    geometry = target.geometry
    assert all(
        isinstance(geometry.curves[curve], SurfaceIsoparametricCurve)
        for curve in geometry.edge_curves
        if curve >= 0
    )
    for coedge, edge in enumerate(geometry.coedge_edges):
        first, last = np.asarray(geometry.edge_ranges[edge], dtype=np.float64)
        parameters = np.linspace(first, last, 7)
        lifted = target.patches[0].evaluate(geometry.pcurves[coedge].evaluate(parameters))
        curve = geometry.curves[geometry.edge_curves[edge]]
        np.testing.assert_allclose(
            np.asarray(curve.evaluate(parameters)),
            np.asarray(lifted),
            rtol=0.0,
            atol=1e-12,
        )


def test_closed_bspline_revolution_offset_certifies_exact_normal_isotopy() -> None:
    source = _closed_bspline_revolution_source()
    patch = source.patches[0]
    assert isinstance(patch, RevolutionSurface)
    profile_domain = patch.curve.parameter_domain
    assert profile_domain is not None
    assert patch.parameter_domain == (
        (0.0, profile_domain[0]),
        (patch.periods[0], profile_domain[1]),
    )
    before = _snapshot(source)
    offset = OffsetSurface(source.patches[0], 0.05, frame_sign=1)
    proof = certify_revolution_offset_embedding(source, offset)
    assert proof.status == "certified", proof.reason

    result = brep_offset(source, 0.05, source_id="closed-bspline-revolution-offset")

    _assert_unchanged(source, before)
    _assert_offset_lineage(source, result)
    target = result.model
    assert target.source_id == "closed-bspline-revolution-offset"
    assert isinstance(target.patches[0], OffsetSurface)
    assert isinstance(target.patches[0].base, RevolutionSurface)
    assert isinstance(target.patches[0].base.curve, BSplineCurve)
    geometry = target.geometry
    assert any(
        isinstance(geometry.curves[curve], SurfaceMappedCurve)
        for curve in geometry.edge_curves
        if curve >= 0
    )
    assert any(
        isinstance(geometry.curves[curve], SurfaceIsoparametricCurve)
        for curve in geometry.edge_curves
        if curve >= 0
    )

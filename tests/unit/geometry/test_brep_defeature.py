"""Independent closed-form oracles for certified local feature removal."""

import math
from fractions import Fraction

import numpy as np
import pytest

import phydrax as phx
from phydrax.geometry.brep._coordinate_request import coordinate_source_request
from phydrax.geometry.brep._defeature import brep_defeature
from phydrax.geometry.brep._fillet import brep_chamfer, brep_fillet
from phydrax.geometry.brep._measure import brep_mass_properties
from phydrax.geometry.brep._model import BRepModel
from phydrax.geometry.brep._patches import CylinderPatch, PlanePatch, TorusPatch
from phydrax.geometry.brep._planar_offset import planar_carrier
from phydrax.geometry.design._naming import RecipeEntityName, RecipeEntityNames
from tests._support.brep_feature_sources import unnormalized_wedge


_CONTRACT = phx.SpatialCoordinateContract.si()
_LAZY = phx.geometry.BRepTessellationPolicy(realize=False)


def _box() -> BRepModel:
    return phx.geometry.brep_box(
        (0.0, 0.0, 0.0),
        (2.0, 3.0, 4.0),
        coordinate_contract=_CONTRACT,
        tessellation=_LAZY,
    )


def _vertical(model: BRepModel) -> int:
    geometry = model.require_geometry()
    return next(
        edge
        for edge, (a, b) in enumerate(geometry.edge_vertices)
        if np.array_equal(
            np.abs(np.asarray(geometry.vertex_points[b] - geometry.vertex_points[a])),
            (0.0, 0.0, 4.0),
        )
    )


def _contains(bounds: np.ndarray, expected: float) -> None:
    assert bounds[0] <= expected <= bounds[1], (bounds, expected)


def _mass(model: BRepModel, expected: float) -> None:
    with coordinate_source_request((model, 8)) as budget:
        result = brep_mass_properties(
            model, model.require_geometry(), subdivisions=8, coordinate_budget=budget
        )
    _contains(np.asarray(result.volume_enclosures)[0], expected)


def test_box_fillet_restores_volume_shape_and_untouched_names() -> None:
    box = _box()
    blend = brep_fillet(box, (box.edge_ids[_vertical(box)],), 0.25, tessellation=_LAZY)
    assert blend.status == "certified", blend.reason
    source = blend.model
    assert source is not None
    index = next(
        i for i, patch in enumerate(source.patches) if isinstance(patch, CylinderPatch)
    )
    names = RecipeEntityNames(
        source,
        tuple(
            RecipeEntityName("face", "boundary", f"face{i}", ("authored",))
            for i in range(len(source.face_ids))
        ),
        None,
    )
    result = brep_defeature(
        source,
        source.face_ids[index],
        operation="fillet",
        recipe_names=names,
        tessellation=_LAZY,
    )
    assert result.status == "certified", result.reason
    assert result.model is not None
    assert result.volume_kind == "exact"
    expected = (1 - math.pi / 4) * 0.25**2 * 4
    _contains(np.asarray(result.volume_change_enclosure), expected)
    _contains(np.asarray(result.hausdorff_enclosure), 0.25 * (math.sqrt(2) - 1))
    _mass(source, 24 - expected)
    _mass(result.model, 24)
    assert len(result.model.face_ids) == 6
    assert result.recipe_names is not None
    for old, new in result.face_images:
        assert names.name(old).key == result.recipe_names.name(new).key
    assert result.intersection_rounding_bounds.tolist() == [0.0, 0.0]
    assert result.model.require_geometry().edge_vertex_gap_bounds is None


def test_box_chamfer_restores_exact_volume() -> None:
    box = _box()
    blend = brep_chamfer(
        box, (box.edge_ids[_vertical(box)],), (0.25, 0.5), tessellation=_LAZY
    )
    assert blend.status == "certified", blend.reason
    source = blend.model
    assert source is not None
    index = next(
        i
        for i, patch in enumerate(source.patches)
        if (plane := planar_carrier(patch)) is not None
        and np.all(np.asarray(plane.first_axis)[:2] != 0)
    )
    result = brep_defeature(
        source, source.face_ids[index], operation="chamfer", tessellation=_LAZY
    )
    assert result.status == "certified", result.reason
    assert result.model is not None
    _contains(np.asarray(result.volume_change_enclosure), 0.25)
    _mass(result.model, 24)


def _plate() -> BRepModel:
    profile = phx.geometry.PlanarProfile(
        phx.geometry.ProfilePlane(),
        phx.geometry.ProfileLoop.polygon(
            ((-2.0, -2.0), (2.0, -2.0), (2.0, 2.0), (-2.0, 2.0))
        ),
        (phx.geometry.ProfileLoop.circle((0.0, 0.0), 0.5),),
    )
    return phx.geometry.brep_extrusion(
        profile, (0.0, 0.0, 1.0), coordinate_contract=_CONTRACT, tessellation=_LAZY
    )


def test_through_hole_restores_plate_volume_and_edge_names() -> None:
    source = _plate()
    index = next(
        i for i, patch in enumerate(source.patches) if isinstance(patch, CylinderPatch)
    )
    names = RecipeEntityNames(
        source,
        tuple(
            RecipeEntityName("face", "plate", f"face{i}", ("authored",))
            for i in range(len(source.face_ids))
        ),
        tuple(
            RecipeEntityName("edge", "plate", f"edge{i}", ("authored",))
            for i in range(len(source.edge_ids))
        ),
    )
    result = brep_defeature(
        source,
        source.face_ids[index],
        operation="hole",
        recipe_names=names,
        tessellation=_LAZY,
    )
    assert result.status == "certified", result.reason
    assert result.model is not None
    assert result.volume_pi == Fraction(1, 4)
    _contains(np.asarray(result.volume_change_enclosure), math.pi / 4)
    _mass(source, 16 - math.pi / 4)
    _mass(result.model, 16)
    assert result.recipe_names is not None
    for old, new in result.edge_images:
        assert result.recipe_names.name(new) == names.name(old)


def test_normal_cylindrical_boss_fillet_matches_pappus_change() -> None:
    cylinder = phx.geometry.brep_cylinder(
        2.0, 3.0, coordinate_contract=_CONTRACT, tessellation=_LAZY
    )
    geometry = cylinder.require_geometry()
    edge = next(
        i
        for i, (first, last) in enumerate(geometry.edge_vertices)
        if first == last and float(geometry.vertex_points[first, 2]) == 3.0
    )
    blend = brep_fillet(cylinder, (cylinder.edge_ids[edge],), 0.25, tessellation=_LAZY)
    assert blend.status == "certified", blend.reason
    source = blend.model
    assert source is not None
    index = next(
        i for i, patch in enumerate(source.patches) if isinstance(patch, TorusPatch)
    )
    result = brep_defeature(
        source, source.face_ids[index], operation="fillet", tessellation=_LAZY
    )
    assert result.status == "certified", result.reason
    assert result.model is not None
    # Revolving the square-minus-quarter-disk section about the cylinder axis.
    r, radius = 0.25, 2.0
    area = r * r * (1 - math.pi / 4)
    radial_first_moment = radius * area - r**3 * (5 / 6 - math.pi / 4)
    expected = 2 * math.pi * radial_first_moment
    _contains(np.asarray(result.volume_change_enclosure), expected)
    _mass(result.model, math.pi * radius**2 * 3)


def test_foreign_face_identity_refuses() -> None:
    box = _box()
    other = phx.geometry.brep_box(
        (0.0, 0.0, 0.0),
        (3.0, 3.0, 4.0),
        coordinate_contract=_CONTRACT,
        tessellation=_LAZY,
    )
    with pytest.raises(ValueError, match="source revision"):
        brep_defeature(box, other.face_ids[0], operation="fillet")


def test_three_four_five_wedge_fillet_contains_high_precision_oracle() -> None:
    import mpmath

    wedge = unnormalized_wedge()
    geometry = wedge.require_geometry()
    edge = next(
        i
        for i, (a, b) in enumerate(geometry.edge_vertices)
        if np.array_equal(np.asarray(geometry.vertex_points[a])[:2], (0.0, 0.0))
        and np.array_equal(np.asarray(geometry.vertex_points[b])[:2], (0.0, 0.0))
        and a != b
    )
    radius = 5 / 16
    blend = brep_fillet(wedge, (wedge.edge_ids[edge],), radius, tessellation=_LAZY)
    assert blend.status == "certified", blend.reason
    source = blend.model
    assert source is not None
    index = next(
        i for i, patch in enumerate(source.patches) if isinstance(patch, CylinderPatch)
    )
    result = brep_defeature(
        source,
        source.face_ids[index],
        operation="fillet",
        maximum_cells=1,
        tessellation=_LAZY,
    )
    assert result.status == "certified", result.reason
    assert result.model is not None
    with mpmath.workdps(80):
        alpha = mpmath.acos(mpmath.mpf(3) / 5)
        expected = float(mpmath.mpf(5 / 16) ** 2 * 4 * (2 - (mpmath.pi - alpha) / 2))
    _contains(np.asarray(result.volume_change_enclosure), expected)
    _mass(result.model, 56)


def _rounded_l_with_hole() -> BRepModel:
    vertices = (
        (0.0, 0.0),
        (3.0, 0.0),
        (3.0, 1.0),
        (1.25, 1.0),
        (1.0, 1.25),
        (1.0, 3.0),
        (0.0, 3.0),
    )
    segments = (
        phx.geometry.ProfileLine(),
        phx.geometry.ProfileLine(),
        phx.geometry.ProfileLine(),
        phx.geometry.ProfileArc((1.25, 1.25), counterclockwise=False),
        phx.geometry.ProfileLine(),
        phx.geometry.ProfileLine(),
        phx.geometry.ProfileLine(),
    )
    profile = phx.geometry.PlanarProfile(
        phx.geometry.ProfilePlane(),
        phx.geometry.ProfileLoop(vertices, segments),
        (phx.geometry.ProfileLoop.circle((1.0, 1.125), 0.015625),),
    )
    return phx.geometry.brep_extrusion(
        profile, (0.0, 0.0, 2.0), coordinate_contract=_CONTRACT, tessellation=_LAZY
    )


def test_extension_collision_refuses_without_publishing_a_model() -> None:
    source = _rounded_l_with_hole()
    index = next(
        i
        for i, patch in enumerate(source.patches)
        if isinstance(patch, CylinderPatch) and float(patch.radius) == 0.25
    )
    result = brep_defeature(
        source,
        source.face_ids[index],
        operation="fillet",
        maximum_cells=1,
        tessellation=_LAZY,
    )
    assert result.status == "collision", result.reason
    assert "intersection" in result.reason
    assert result.model is None
    assert result.volume_kind == "absent"


def _pocket(source: BRepModel) -> BRepModel:
    from phydrax.geometry.brep._closed_seam_blend import _copy_builder
    from phydrax.geometry.brep._constructors import finish
    from phydrax.geometry.brep._intersection import native_edge_pcurve
    from phydrax.geometry.brep._intersection_curve import SurfaceRegion
    from phydrax.geometry.brep._patches import LineCurve

    builder = _copy_builder(source)
    cap = next(
        i
        for i, patch in enumerate(source.patches)
        if isinstance(patch, PlanePatch)
        and np.array_equal(np.asarray(patch.origin)[2:], (4.0,))
        and np.array_equal(np.asarray(patch.first_axis)[2:], (0.0,))
        and np.array_equal(np.asarray(patch.second_axis)[2:], (0.0,))
    )
    polygon = ((0.75, 1.0), (1.25, 1.0), (1.25, 1.5), (0.75, 1.5))
    upper = [
        builder.vertex(np.asarray((x, y, 4.0), dtype=np.float64)) for x, y in polygon
    ]
    lower = [
        builder.vertex(np.asarray((x, y, 2.0), dtype=np.float64)) for x, y in polygon
    ]
    top, bottom, vertical = [], [], []
    for i in range(4):
        j = (i + 1) % 4
        for vertices, bank in ((upper, top), (lower, bottom)):
            a, b = vertices[i], vertices[j]
            bank.append(
                builder.edge(
                    LineCurve(
                        builder.vertices[a], builder.vertices[b] - builder.vertices[a]
                    ),
                    0.0,
                    1.0,
                    a,
                    b,
                )
            )
        vertical.append(
            builder.edge(
                LineCurve(builder.vertices[upper[i]], (0.0, 0.0, -2.0)),
                0.0,
                1.0,
                upper[i],
                lower[i],
            )
        )
    with coordinate_source_request((source, polygon)) as budget:
        region = SurfaceRegion(
            source.patches[cap], np.asarray(source.parameter_bounds[cap]), budget=budget
        )
        hole = []
        for edge in reversed(top):
            curve = builder.curves[builder.edge_curves[edge]]
            assert isinstance(curve, LineCurve)
            pcurve = native_edge_pcurve(curve, region)
            assert pcurve is not None
            hole.append(builder.coedge(edge, -1, pcurve, budget=budget))
        builder.faces[cap].loops.append(hole)
        floor = [
            builder.coedge(
                edge,
                1,
                LineCurve(
                    polygon[i], np.asarray(polygon[(i + 1) % 4]) - np.asarray(polygon[i])
                ),
                budget=budget,
            )
            for i, edge in enumerate(bottom)
        ]
        builder.face(
            PlanePatch((0.0, 0.0, 2.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0)),
            np.asarray(((0.75, 1.0), (1.25, 1.5)), dtype=np.float64),
            [floor],
            "pocket-floor",
            None,
            1,
        )
        for i in range(4):
            j = (i + 1) % 4
            uses = [
                builder.coedge(
                    top[i], 1, LineCurve((0.0, 0.0), (1.0, 0.0)), budget=budget
                ),
                builder.coedge(
                    vertical[j], 1, LineCurve((1.0, 0.0), (0.0, 1.0)), budget=budget
                ),
                builder.coedge(
                    bottom[i], -1, LineCurve((0.0, 1.0), (1.0, 0.0)), budget=budget
                ),
                builder.coedge(
                    vertical[i], -1, LineCurve((0.0, 0.0), (0.0, 1.0)), budget=budget
                ),
            ]
            builder.face(
                PlanePatch(
                    builder.vertices[upper[i]],
                    builder.vertices[upper[j]] - builder.vertices[upper[i]],
                    (0.0, 0.0, -2.0),
                ),
                np.asarray(((0.0, 0.0), (1.0, 1.0)), dtype=np.float64),
                [uses],
                "pocket-wall",
                None,
                1,
            )
        return finish(
            builder,
            solid=True,
            coordinate_contract=_CONTRACT,
            source_id="pocket-fixture",
            tessellation=_LAZY,
            budget=budget,
        )


def test_fillet_on_box_with_pocket_preserves_local_pocket_geometry() -> None:
    box = _box()
    blend = brep_fillet(box, (box.edge_ids[_vertical(box)],), 0.25, tessellation=_LAZY)
    assert blend.model is not None
    source = _pocket(blend.model)
    index = next(
        i for i, patch in enumerate(source.patches) if isinstance(patch, CylinderPatch)
    )
    result = brep_defeature(
        source,
        source.face_ids[index],
        operation="fillet",
        maximum_cells=1,
        tessellation=_LAZY,
    )
    assert result.status == "certified", result.reason
    assert result.model is not None
    _mass(result.model, 24 - 0.5 * 0.5 * 2)
    assert result.model.physical_tags.count("pocket-wall") == 4
    assert "pocket-floor" in result.model.physical_tags


def test_conical_through_hole_publishes_an_enclosed_volume_change() -> None:
    from phydrax.geometry.brep._patches import ConePatch

    profile = phx.geometry.PlanarProfile(
        phx.geometry.ProfilePlane((0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 0.0, 1.0)),
        phx.geometry.ProfileLoop.polygon(
            ((0.5, 0.0), (2.0, 0.0), (2.0, 1.0), (0.75, 1.0))
        ),
    )
    source = phx.geometry.brep_revolution(
        profile, (0.0, 0.0), (0.0, 1.0), coordinate_contract=_CONTRACT, tessellation=_LAZY
    )
    index = next(
        i for i, patch in enumerate(source.patches) if isinstance(patch, ConePatch)
    )
    result = brep_defeature(
        source, source.face_ids[index], operation="hole", tessellation=_LAZY
    )
    assert result.status == "certified", result.reason
    assert result.model is not None
    assert result.volume_kind == "enclosed"
    expected = math.pi * (0.5**2 + 0.5 * 0.75 + 0.75**2) / 3
    _contains(np.asarray(result.volume_change_enclosure), expected)
    _mass(result.model, 4 * math.pi)


def test_spherical_corner_family_refuses_without_shape_change_evidence() -> None:
    source = phx.geometry.brep_sphere(
        1.0, coordinate_contract=_CONTRACT, tessellation=_LAZY
    )
    result = brep_defeature(
        source, source.face_ids[0], operation="fillet", tessellation=_LAZY
    )
    assert result.status == "unsupported"
    assert result.model is None
    assert result.volume_change_enclosure.size == 0
    assert result.hausdorff_enclosure.size == 0


def test_request_reservation_refuses_before_feature_evaluation() -> None:
    from phydrax._resource_ticket import HostBatchTicket
    from phydrax.discretization._coordinate_budget import (
        CoordinateEnclosureBudget,
        CoordinateEnclosureResourceError,
    )

    source = _box()
    with HostBatchTicket(0, bytes_upper=0) as ticket:
        budget = CoordinateEnclosureBudget(
            ticket, maximum_work_units=0, maximum_memory_bytes=0
        )
        with pytest.raises(CoordinateEnclosureResourceError):
            brep_defeature(source, source.face_ids[0], operation="chamfer", budget=budget)

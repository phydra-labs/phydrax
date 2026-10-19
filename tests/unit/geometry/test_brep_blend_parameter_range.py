"""Independent adjacent-box-extent oracles for certified blend ranges."""

from fractions import Fraction

import numpy as np
import pytest

import phydrax as phx
from phydrax._resource_ticket import HostBatchTicket
from phydrax.discretization._coordinate_budget import (
    CoordinateEnclosureBudget,
    CoordinateEnclosureResourceError,
)
from phydrax.geometry.brep._fillet import (
    _blend_parameter_range,
    brep_chamfer,
    brep_fillet,
)
from phydrax.geometry.brep._parameter_range import refusal_parameter_range


_CONTRACT = phx.SpatialCoordinateContract.si()
_LAZY = phx.geometry.BRepTessellationPolicy(realize=False)
_EXTENTS = (2.0, 3.0, 4.0)


def _box_edge(axis: int) -> tuple[phx.geometry.BRepModel, int]:
    box = phx.geometry.brep_box(
        (0.0, 0.0, 0.0), _EXTENTS, coordinate_contract=_CONTRACT, tessellation=_LAZY
    )
    geometry = box.require_geometry()
    for index, (first, last) in enumerate(geometry.edge_vertices):
        direction = np.asarray(
            geometry.vertex_points[last] - geometry.vertex_points[first]
        )
        if abs(direction[axis]) == _EXTENTS[axis] and np.count_nonzero(direction) == 1:
            return box, index
    raise RuntimeError("Box range fixture has no edge on the requested axis")


@pytest.mark.parametrize("axis", (0, 1, 2))
def test_box_radius_limit_is_exact_adjacent_extent_and_straddles_construction(
    axis: int,
) -> None:
    box, edge = _box_edge(axis)
    # Independent box geometry: the quarter-circle consumes radius on both
    # adjacent faces, not on the face direction parallel to the selected edge.
    upper = min(value for index, value in enumerate(_EXTENTS) if index != axis)
    inside = brep_fillet(
        box,
        (box.edge_ids[edge],),
        upper - 2.0**-12,
        tessellation=_LAZY,
        include_parameter_range=True,
    )
    assert inside.status == "certified", inside.reason
    certificate = inside.parameter_range
    assert certificate is not None
    assert certificate.parameter == "radius"
    assert certificate.interval == (0.0, upper)
    assert certificate.end_status == ("attained_exactly", "attained_exactly")
    assert certificate.constraints == ("parameter_domain", "face_vanishes")
    assert certificate.entity_ids[0] == (box.edge_ids[edge],)
    assert certificate.entity_ids[1]
    outside = brep_fillet(
        box, (box.edge_ids[edge],), upper + 2.0**-12, tessellation=_LAZY
    )
    assert outside.status == "unsupported"
    assert (
        outside.reason
        == "planar blend refuses: setback reaches a neighboring edge or face vanishes"
    )
    assert outside.parameter_range is not None
    assert outside.parameter_range.interval == certificate.interval
    boundary = brep_fillet(box, (box.edge_ids[edge],), upper, tessellation=_LAZY)
    assert boundary.status == "unsupported"  # The certificate is an open interval.
    assert boundary.parameter_range is not None
    assert boundary.parameter_range.end_status[1] == "attained_exactly"


def test_success_certificate_is_optional_and_parameter_map_bounds_are_canonical() -> None:
    box, edge = _box_edge(2)
    ordinary = brep_fillet(box, (box.edge_ids[edge],), 0.25, tessellation=_LAZY)
    assert ordinary.status == "certified", ordinary.reason
    assert ordinary.parameter_range is None
    requested = brep_fillet(
        box,
        (box.edge_ids[edge],),
        0.25,
        tessellation=_LAZY,
        include_parameter_range=True,
    )
    assert requested.status == "certified", requested.reason
    assert requested.source is not None
    compiled = requested.source.compile(coordinate_unit=_CONTRACT.length_unit)
    assert compiled.schema.specs[0].bounds == (0.0, 2.0)
    assert ordinary.source is not None
    ordinary_compiled = ordinary.source.compile(coordinate_unit=_CONTRACT.length_unit)
    assert ordinary_compiled.schema.specs[0].bounds == compiled.schema.specs[0].bounds
    assert ordinary_compiled.schema.schema_id == compiled.schema.schema_id


def test_box_chamfer_scale_preserves_setback_ratio_and_straddles_limit() -> None:
    box, edge = _box_edge(2)
    # Equal setbacks remove their common length along both adjacent edges,
    # independent of the builder's predecessor/successor convention.
    setbacks = (0.25, 0.25)
    upper = min(_EXTENTS[:2]) / setbacks[0]
    original = brep_chamfer(
        box,
        (box.edge_ids[edge],),
        setbacks,
        tessellation=_LAZY,
        include_parameter_range=True,
    )
    assert original.status == "certified", original.reason
    certificate = original.parameter_range
    assert certificate is not None
    assert certificate.parameter == "setback_scale"
    assert certificate.interval == (0.0, upper)
    assert certificate.end_status == ("attained_exactly", "attained_exactly")
    for scale, expected in (
        (upper - 2.0**-10, "certified"),
        (upper + 2.0**-10, "unsupported"),
    ):
        result = brep_chamfer(
            box,
            (box.edge_ids[edge],),
            (setbacks[0] * scale, setbacks[1] * scale),
            tessellation=_LAZY,
            include_parameter_range=True,
        )
        assert result.status == expected, result.reason
        assert result.parameter_range is not None
        assert result.parameter_range.parameter == "setback_scale"


def test_nonbinary_chamfer_scale_endpoint_rounds_inward() -> None:
    box, edge = _box_edge(2)
    result = brep_chamfer(
        box,
        (box.edge_ids[edge],),
        (0.75, 0.75),
        tessellation=_LAZY,
        include_parameter_range=True,
    )
    assert result.status == "certified", result.reason
    certificate = result.parameter_range
    assert certificate is not None
    upper = certificate.interval[1]
    assert upper is not None
    exact = Fraction(8, 3)
    assert Fraction(upper) < exact < Fraction(float(np.nextafter(upper, np.inf)))
    assert certificate.end_status[1] == "certified_inner_bound"


def test_unsupported_corner_family_always_reports_unknown() -> None:
    box, edge = _box_edge(2)
    geometry = box.require_geometry()
    endpoints = set(geometry.edge_vertices[edge])
    other = next(
        index
        for index, pair in enumerate(geometry.edge_vertices)
        if index != edge and endpoints.intersection(pair)
    )
    result = brep_fillet(box, (box.edge_ids[edge], box.edge_ids[other]), 0.25)
    assert result.status == "unsupported"
    assert (
        result.reason
        == "vertex blend with multiple blended incident edges requires a corner patch"
    )
    assert result.parameter_range is not None
    assert result.parameter_range.interval == (None, None)
    assert result.parameter_range.end_status == ("unknown", "unknown")
    assert result.parameter_range.evidence == (result.reason,)


@pytest.mark.parametrize("radius", (0.0, -1.0, float("nan"), float("inf")))
def test_radius_validation_preserves_exception_and_adds_unknown_evidence(
    radius: float,
) -> None:
    box, edge = _box_edge(2)
    with pytest.raises(ValueError) as captured:
        brep_fillet(box, (box.edge_ids[edge],), radius)
    error = captured.value
    assert type(error) is ValueError
    assert error.args == (str(error),)
    assert "radius" in str(error)
    certificate = refusal_parameter_range(error)
    assert certificate.parameter == "radius"
    assert certificate.interval == (None, None)


def test_chamfer_validation_retains_original_message_and_unknown_evidence() -> None:
    box, edge = _box_edge(2)
    with pytest.raises(ValueError) as captured:
        brep_chamfer(box, (box.edge_ids[edge],), (0.0, 0.25))
    certificate = refusal_parameter_range(captured.value)
    assert certificate.parameter == "setback_scale"
    assert certificate.interval == (None, None)


def test_closed_cylinder_radius_and_chamfer_ranges_reuse_certified_chart_limits() -> None:
    cylinder = phx.geometry.brep_cylinder(
        3.0, 2.0, coordinate_contract=_CONTRACT, tessellation=_LAZY
    )
    geometry = cylinder.require_geometry()
    edge = next(
        index
        for index, (first, last) in enumerate(geometry.edge_vertices)
        if first == last and float(geometry.vertex_points[first, 2]) == 2.0
    )
    fillet = brep_fillet(
        cylinder,
        (cylinder.edge_ids[edge],),
        0.25,
        tessellation=_LAZY,
        include_parameter_range=True,
    )
    assert fillet.status == "certified", fillet.reason
    certificate = fillet.parameter_range
    assert certificate is not None
    upper = certificate.interval[1]
    # Independent meridian geometry: the offset center reaches the axis at
    # radius R=3 or the opposite cap at radius H=2, whichever comes first.
    assert upper is not None and 0.25 < upper < min(3.0, 2.0)
    assert min(3.0, 2.0) - upper < 16 * np.spacing(2.0)
    assert certificate.end_status == ("attained_exactly", "certified_inner_bound")
    chamfer = brep_chamfer(
        cylinder,
        (cylinder.edge_ids[edge],),
        (0.25, 0.5),
        tessellation=_LAZY,
        include_parameter_range=True,
    )
    assert chamfer.status == "certified", chamfer.reason
    certificate = chamfer.parameter_range
    assert certificate is not None and certificate.parameter == "setback_scale"
    upper = certificate.interval[1]
    oracle = min(3.0 / 0.25, 2.0 / 0.5)
    assert upper is not None and 1.0 < upper < oracle
    assert oracle - upper < 16 * np.spacing(oracle)


def test_parallel_cylinder_range_reports_certified_inner_epoch_not_exact_limit() -> None:
    source = phx.geometry.brep_extrusion(
        phx.geometry.PlanarProfile(
            phx.geometry.ProfilePlane(),
            phx.geometry.ProfileLoop(
                ((0.0, -4.0), (0.0, 4.0)),
                (
                    phx.geometry.ProfileArc((-3.0, 0.0)),
                    phx.geometry.ProfileArc((3.0, 0.0)),
                ),
            ),
        ),
        (0.0, 0.0, 2.0),
        coordinate_contract=_CONTRACT,
        tessellation=_LAZY,
    )
    geometry = source.require_geometry()
    edge = next(
        index
        for index, (first, last) in enumerate(geometry.edge_vertices)
        if first != last
        and float(geometry.vertex_points[first, 1]) == 4.0
        and float(geometry.vertex_points[last, 1]) == 4.0
    )
    radius = 0.25
    result = brep_fillet(
        source,
        (source.edge_ids[edge],),
        radius,
        tessellation=_LAZY,
        include_parameter_range=True,
    )
    assert result.status == "certified", result.reason
    certificate = result.parameter_range
    assert certificate is not None
    lower, upper = certificate.interval
    # The equal-radius circle-center construction has offset radius 5-r,
    # center separation 6, and a positive discriminant only while r<2.
    assert lower is not None and upper is not None
    assert 0.0 < lower < radius < upper < 2.0
    assert certificate.end_status == ("certified_inner_bound", "certified_inner_bound")
    assert certificate.evidence


@pytest.mark.parametrize("selection_count", (0, 2))
def test_invalid_edge_selection_preserves_refusal_message_and_unknown_range(
    selection_count: int,
) -> None:
    box, edge = _box_edge(2)
    with pytest.raises(ValueError) as captured:
        brep_fillet(box, (box.edge_ids[edge],) * selection_count, 0.25)
    assert str(captured.value) == (
        "brep_fillet.edges must contain distinct nonempty source identities; got an empty or repeated selection"
    )
    certificate = refusal_parameter_range(captured.value)
    assert certificate.parameter == "radius"
    assert certificate.interval == (None, None)


def test_parameter_range_dispatch_preserves_request_budget_refusal() -> None:
    box, edge = _box_edge(2)
    with HostBatchTicket(64, bytes_upper=4096) as ticket:
        budget = CoordinateEnclosureBudget(
            ticket, maximum_work_units=64, maximum_memory_bytes=4096
        )
        with budget.activate(), pytest.raises(CoordinateEnclosureResourceError):
            _blend_parameter_range(box, edge, "planes", 0.25, None, budget=budget)

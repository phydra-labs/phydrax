"""White-box contracts for certified intersection branch assembly."""

import numpy as np
import pytest

from phydrax.geometry.brep import PlanePatch, SurfaceRegion
from phydrax.geometry.brep._coordinate_request import coordinate_source_request
from phydrax.geometry.brep._intersection import (
    _BranchAssembly,
    _Budget,
    _Chart,
    _finish_surface_intersection,
    _link_segments,
    _PairContext,
    _Segment,
    CurveSurfaceSystem,
    prepare,
)
from phydrax.geometry.brep._intersection_curve import SurfacePairSystem
from phydrax.geometry.brep._patches import LineCurve
from phydrax.geometry.brep._request_preparations import TrimRequestPreparations


@pytest.fixture
def branch_regions() -> tuple[SurfaceRegion, SurfaceRegion]:
    first = SurfaceRegion(
        PlanePatch((0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0)),
        ((0.0, -1.0), (1.0, 1.0)),
    )
    second = SurfaceRegion(
        PlanePatch((0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 0.0, 1.0)),
        ((0.0, -1.0), (1.0, 1.0)),
    )
    return first, second


def _junction_segments(
    regions: tuple[SurfaceRegion, SurfaceRegion],
    offsets: tuple[float, ...],
    radius: float,
) -> tuple[list[_Segment], list[_PairContext]]:
    first, second = regions
    pair = prepare(SurfacePairSystem(first.patch, second.patch), 4, 1)
    contexts: list[_PairContext] = []
    segments: list[_Segment] = []
    for index, offset in enumerate(offsets):
        incoming = index == 0
        left, right = (0.0, 0.5) if incoming else (0.5, 1.0)
        low = np.asarray((left, -1.0, left, -1.0), dtype=np.float64)
        high = np.asarray((right, 1.0, right, 1.0), dtype=np.float64)
        contexts.append(_PairContext(index, pair, low, high, (0, 0)))
        start = np.asarray((left, offset, left, 0.0), dtype=np.float64)
        end = np.asarray((right, offset, right, 0.0), dtype=np.float64)
        chart = _Chart(
            axis=0,
            start=start,
            end=end,
            lower=np.asarray((left, offset - radius, left, -radius)),
            upper=np.asarray((right, offset + radius, right, radius)),
            preconditioner=np.eye(3, dtype=np.float64),
            contraction=0.25,
            certified=True,
        )
        segments.append(
            _Segment(
                index, [chart], ("boundary", (0, 0), start), ("boundary", (0, 1), end)
            )
        )
    return segments, contexts


def test_two_near_junction_partners_remain_unresolved(
    branch_regions: tuple[SurfaceRegion, SurfaceRegion],
) -> None:
    # All separations are inside the old 1e-8 coordinate tolerance. The binary
    # powers define the represented geometry/enclosures, not an assertion slack.
    segments, contexts = _junction_segments(
        branch_regions, (0.0, 2.0**-30, -(2.0**-30)), 2.0**-28
    )
    links = _link_segments(segments, contexts, *branch_regions)
    assert not links.pairs
    assert len(links.unresolved) == 3
    with coordinate_source_request(branch_regions) as coordinate_budget:
        result = _finish_surface_intersection(
            _BranchAssembly(curves=(), unresolved=links.unresolved),
            [],
            [],
            [],
            0.0,
            _Budget(10),
            *branch_regions,
            coordinate_budget=coordinate_budget,
        )
    assert not result.complete
    assert result.work.boxes_unresolved == 3


def test_unique_overlapping_partner_links(
    branch_regions: tuple[SurfaceRegion, SurfaceRegion],
) -> None:
    segments, contexts = _junction_segments(branch_regions, (0.0, 0.0), 2.0**-30)
    links = _link_segments(segments, contexts, *branch_regions)
    assert links.pairs == (((0, 1), (1, 0)),)
    assert not links.unresolved


def test_close_coordinates_without_overlapping_enclosures_do_not_link(
    branch_regions: tuple[SurfaceRegion, SurfaceRegion],
) -> None:
    segments, contexts = _junction_segments(branch_regions, (0.0, 2.0**-30), 2.0**-33)
    links = _link_segments(segments, contexts, *branch_regions)
    assert not links.pairs
    assert len(links.unresolved) == 2


def test_prepared_pair_programs_reuse_only_within_request(
    branch_regions: tuple[SurfaceRegion, SurfaceRegion],
) -> None:
    first, second = branch_regions
    system = SurfacePairSystem(first.patch, second.patch)
    request = TrimRequestPreparations()
    prepared = prepare(system, 4, 1, preparations=request)
    assert prepare(system, 4, 1, preparations=request) is prepared
    assert prepare(system, 4, 1, preparations=TrimRequestPreparations()) is not prepared


def test_curve_surface_programs_reuse_equal_sources_only_within_request() -> None:
    curve = LineCurve((0.0, 0.0, 0.0), (1.0, 0.0, 0.0))
    surface = PlanePatch((0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0))
    request = TrimRequestPreparations()
    prepared = prepare(CurveSurfaceSystem(curve, surface), 3, 1, preparations=request)
    equivalent = CurveSurfaceSystem(
        LineCurve((0.0, 0.0, 0.0), (1.0, 0.0, 0.0)),
        PlanePatch((0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0)),
    )
    assert prepare(equivalent, 3, 1, preparations=request) is prepared
    assert prepare(equivalent, 3, 2, preparations=request) is not prepared
    assert (
        prepare(equivalent, 3, 1, preparations=TrimRequestPreparations()) is not prepared
    )
    different = CurveSurfaceSystem(LineCurve((0.0, 0.0, 1.0), (1.0, 0.0, 0.0)), surface)
    assert prepare(different, 3, 1, preparations=request) is not prepared
    point = np.asarray(((0.25, 0.0, -0.25),), dtype=np.float64)
    np.testing.assert_array_equal(
        prepared.point_values(point), np.asarray(((0.25, 0.25, 0.0),))
    )

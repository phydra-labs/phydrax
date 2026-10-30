#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Exact overlap proofs between oriented source and candidate facet partitions."""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from typing import Literal

from ._domain_coverage import PiecewiseLinearDomain
from ._planar_primitives import (
    area2,
    intersection,
    plane_key,
    plane_orientation,
    project_polygon,
    projection_axes,
    segment_overlap,
)


type OverlapSemantics = Literal["exact", "candidate"]
type CoordinateSpace = Literal["physical", "reference"]
type RegionPair = tuple[int, int]


@dataclass(frozen=True, slots=True)
class OrientedFacet:
    facet_id: int
    region_pair: RegionPair
    vertices: tuple[tuple[Fraction, ...], ...]


@dataclass(frozen=True, slots=True)
class FacetOverlap:
    source_facet_id: int
    target_facet_id: int
    region_pair: RegionPair
    measure: Fraction
    semantics: OverlapSemantics
    coordinate_space: CoordinateSpace


def source_facets(domain: PiecewiseLinearDomain, /) -> tuple[OrientedFacet, ...]:
    points = tuple(
        tuple(Fraction(float(value)) for value in row) for row in domain.vertices
    )
    return tuple(
        OrientedFacet(
            facet_id,
            (
                int(domain.facet_regions[facet_id, 0]),
                int(domain.facet_regions[facet_id, 1]),
            ),
            tuple(points[int(index)] for index in domain.facets[facet_id, :]),
        )
        for facet_id in range(domain.facets.shape[0])
    )


def _facet_measure(facet: OrientedFacet, dimension: int, /) -> Fraction:
    if dimension == 2:
        first, second = facet.vertices
        delta = (second[0] - first[0], second[1] - first[1])
        axis = max(range(2), key=lambda value: abs(delta[value]))
        return abs(delta[axis])
    plane = plane_key(facet.vertices)
    return area2(project_polygon(facet.vertices, projection_axes(plane))) / 2


def _orientation_matches(
    first: OrientedFacet, second: OrientedFacet, dimension: int, /
) -> bool:
    if dimension == 2:
        first_direction = tuple(
            second_value - first_value
            for first_value, second_value in zip(
                first.vertices[0], first.vertices[1], strict=True
            )
        )
        second_direction = tuple(
            second_value - first_value
            for first_value, second_value in zip(
                second.vertices[0], second.vertices[1], strict=True
            )
        )
        return (
            sum(
                left * right
                for left, right in zip(first_direction, second_direction, strict=True)
            )
            > 0
        )
    return plane_orientation(first.vertices) == plane_orientation(second.vertices)


def _overlap(first: OrientedFacet, second: OrientedFacet, dimension: int, /) -> Fraction:
    if dimension == 2:
        first_start, first_end = ((x, y) for x, y in first.vertices)
        second_start, second_end = ((x, y) for x, y in second.vertices)
        return segment_overlap((first_start, first_end), (second_start, second_end))
    try:
        first_plane = plane_key(first.vertices)
        if first_plane != plane_key(second.vertices):
            return Fraction(0)
    except ValueError:
        return Fraction(0)
    axes = projection_axes(first_plane)
    first_polygon = project_polygon(first.vertices, axes)
    second_polygon = project_polygon(second.vertices, axes)
    return area2(intersection(first_polygon, second_polygon)) / 2


def partition_overlaps(
    domain: PiecewiseLinearDomain,
    targets: tuple[OrientedFacet, ...],
    /,
    *,
    semantics: OverlapSemantics = "exact",
    coordinate_space: CoordinateSpace = "physical",
) -> tuple[tuple[FacetOverlap, ...], tuple[tuple[str, int, int], ...]]:
    """Match both complete oriented facet chains; return exact witnesses/findings.

    Each accepted overlap has exact rational measure. Containment of each target
    facet and equality of every source-facet overlap sum, together with the
    positive embedding premise, proves source coverage and excludes missing or
    repeated pieces. No total-volume-only inference is made.
    """
    if not isinstance(domain, PiecewiseLinearDomain):
        raise TypeError("domain must be PiecewiseLinearDomain.")
    dimension = domain.ambient_dimension
    sources = source_facets(domain)
    overlaps: list[FacetOverlap] = []
    findings: list[tuple[str, int, int]] = []
    source_totals = [Fraction(0) for _ in sources]

    for index, first in enumerate(sources):
        for other_index in range(index + 1, len(sources)):
            second = sources[other_index]
            shared = _overlap(first, second, dimension)
            if shared:
                findings.append(
                    ("overlapping_source_facets", first.facet_id, second.facet_id)
                )

    for target in targets:
        target_measure = _facet_measure(target, dimension)
        if target_measure <= 0:
            findings.append(("degenerate_target_facet", target.facet_id, -1))
            continue
        matched = Fraction(0)
        reverse = Fraction(0)
        for source_index, source in enumerate(sources):
            overlap = _overlap(source, target, dimension)
            if overlap <= 0:
                continue
            if source.region_pair == target.region_pair and _orientation_matches(
                source, target, dimension
            ):
                matched += overlap
                source_totals[source_index] += overlap
                overlaps.append(
                    FacetOverlap(
                        source.facet_id,
                        target.facet_id,
                        target.region_pair,
                        overlap,
                        semantics,
                        coordinate_space,
                    )
                )
            elif source.region_pair == target.region_pair[::-1] or (
                source.region_pair == target.region_pair
                and not _orientation_matches(source, target, dimension)
            ):
                reverse += overlap
        if reverse:
            check = (
                "boundary_orientation"
                if -1 in target.region_pair
                else "interface_orientation"
            )
            findings.append((check, target.facet_id, -1))
        if matched != target_measure:
            findings.append(("unmatched_boundary_facet", target.facet_id, -1))

    for source_index, source in enumerate(sources):
        expected = _facet_measure(source, dimension)
        if source_totals[source_index] == 0:
            findings.append(("uncovered_boundary", source.facet_id, -1))
        elif source_totals[source_index] != expected:
            findings.append(("double_or_partial_source_coverage", source.facet_id, -1))

    return tuple(overlaps), tuple(findings)


__all__ = [
    "CoordinateSpace",
    "FacetOverlap",
    "OrientedFacet",
    "OverlapSemantics",
    "RegionPair",
    "partition_overlaps",
    "source_facets",
]

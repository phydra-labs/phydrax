#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#
"""Exact categorical-image boundary preparation for native PLC filling."""

from __future__ import annotations

import json
from dataclasses import dataclass
from fractions import Fraction
from typing import Final

import numpy as np

from .._fingerprint import canonical_fingerprint
from ..geometry._compartments import CompartmentMeshingSource
from ..geometry._source_identity import source_entity_id
from ..typing import checked
from ._plc_source import PiecewiseLinearComplex


_FACE_AXES: Final[tuple[tuple[int, int], ...]] = ((1, 2), (2, 0), (0, 1))


@dataclass(frozen=True, slots=True)
class CompartmentPlcPreparation:
    """Source-lattice PLC plus its semantic facet-to-region table."""

    source_id: str
    source_revision: str
    complex: PiecewiseLinearComplex
    lattice_vertex_keys: tuple[tuple[int, int, int], ...]
    facet_interface_ids: tuple[str | None, ...]
    facet_region_ids: tuple[tuple[str | None, str], ...]
    preparation_id: str


def _ordered_pair(first: str, second: str, /) -> tuple[str, str]:
    return (first, second) if first < second else (second, first)


def _image_face_corners(
    index: tuple[int, int, int], axis: int, direction: int, /
) -> tuple[tuple[int, int, int], ...]:
    first_axis, second_axis = _FACE_AXES[axis]
    center = [2 * value for value in index]
    center[axis] += direction
    corners = []
    for first_sign, second_sign in ((-1, -1), (1, -1), (1, 1), (-1, 1)):
        point = center.copy()
        point[first_axis] += first_sign
        point[second_axis] += second_sign
        corners.append((point[0], point[1], point[2]))
    return tuple(corners)


def _compartment_codes(
    source: CompartmentMeshingSource, /
) -> tuple[np.ndarray, tuple[str, ...], dict[tuple[str, str], tuple[str, str, str]]]:
    source.validate_source_integrity()
    labels = source.labels
    values = np.asarray(labels.asset.values)
    valid = np.asarray(labels.asset.valid_mask, dtype=np.bool_)
    if values.ndim != 3 or valid.shape != values.shape:
        raise ValueError("Compartment PLC preparation requires a valid 3D label volume.")

    ontology = {
        definition.label_id: definition.value for definition in labels.ontology.labels
    }
    region_ids = tuple(
        definition.compartment_id for definition in source.compartments.compartments
    )
    region_index = {name: index for index, name in enumerate(region_ids)}
    value_to_region: dict[int, int] = {}
    for definition in source.compartments.compartments:
        region = region_index[definition.compartment_id]
        for label_id in definition.label_ids:
            value = int(ontology[label_id])
            if value in value_to_region:
                raise ValueError(
                    "A categorical label cannot belong to multiple compartments."
                )
            value_to_region[value] = region

    code = np.full(values.shape, -1, dtype=np.int64)
    for value, region in value_to_region.items():
        code[valid & (values == value)] = region

    interfaces: dict[tuple[str, str], tuple[str, str, str]] = {}
    for interface_id, first, second, _role, _required in source.interface_definitions:
        pair = _ordered_pair(first, second)
        if pair in interfaces:
            raise ValueError("Compartment interface pairs must have one source identity.")
        interfaces[pair] = (interface_id, first, second)
    return code, region_ids, interfaces


_Point2 = tuple[Fraction, Fraction]


def _signed_area2(polygon: tuple[_Point2, ...], /) -> Fraction:
    return sum(
        (
            first[0] * second[1] - first[1] * second[0]
            for first, second in zip(polygon, (*polygon[1:], polygon[0]), strict=True)
        ),
        Fraction(0),
    )


def _polygon_area(polygon: tuple[_Point2, ...], /) -> Fraction:
    if len(polygon) < 3:
        return Fraction(0)
    return abs(_signed_area2(polygon)) / 2


def _clip_axis(
    polygon: tuple[_Point2, ...], axis: int, bound: int, keep_greater: bool, /
) -> tuple[_Point2, ...]:
    if not polygon:
        return ()

    def inside(point: _Point2, /) -> bool:
        return point[axis] >= bound if keep_greater else point[axis] <= bound

    output: list[_Point2] = []
    previous = polygon[-1]
    previous_inside = inside(previous)
    for current in polygon:
        current_inside = inside(current)
        if current_inside != previous_inside:
            denominator = current[axis] - previous[axis]
            if denominator == 0:
                raise ArithmeticError(
                    "Exact polygon clipping encountered a parallel crossing."
                )
            parameter = Fraction(bound - previous[axis], denominator)
            output.append(
                (
                    previous[0] + parameter * (current[0] - previous[0]),
                    previous[1] + parameter * (current[1] - previous[1]),
                )
            )
        if current_inside:
            output.append(current)
        previous = current
        previous_inside = current_inside
    return tuple(output)


def _clip_rectangle(
    polygon: tuple[_Point2, ...],
    bounds: tuple[int, int, int, int],
    /,
) -> tuple[_Point2, ...]:
    lower_u, upper_u, lower_v, upper_v = bounds
    result = _clip_axis(polygon, 0, lower_u, True)
    result = _clip_axis(result, 0, upper_u, False)
    result = _clip_axis(result, 1, lower_v, True)
    return _clip_axis(result, 1, upper_v, False)


def _clip_convex_polygon(
    subject: tuple[_Point2, ...], clipper: tuple[_Point2, ...], /
) -> tuple[_Point2, ...]:
    result = subject
    orientation = 1 if _signed_area2(clipper) > 0 else -1
    for first, second in zip(clipper, (*clipper[1:], clipper[0]), strict=True):
        if not result:
            return ()

        def side(point: _Point2, /) -> Fraction:
            return orientation * (
                (second[0] - first[0]) * (point[1] - first[1])
                - (second[1] - first[1]) * (point[0] - first[0])
            )

        output: list[_Point2] = []
        previous = result[-1]
        previous_side = side(previous)
        previous_inside = previous_side >= 0
        for current in result:
            current_side = side(current)
            current_inside = current_side >= 0
            if current_inside != previous_inside:
                denominator = previous_side - current_side
                if denominator == 0:
                    raise ArithmeticError(
                        "Exact convex clipping encountered a parallel crossing."
                    )
                parameter = previous_side / denominator
                output.append(
                    (
                        previous[0] + parameter * (current[0] - previous[0]),
                        previous[1] + parameter * (current[1] - previous[1]),
                    )
                )
            if current_inside:
                output.append(current)
            previous = current
            previous_side = current_side
            previous_inside = current_inside
        result = tuple(output)
    return result


def _verify_outer_surface_lattice(
    source: CompartmentMeshingSource,
    expected_faces: tuple[tuple[int, int, int, tuple[tuple[int, int, int], ...]], ...],
    /,
) -> None:
    """Prove exact planar coverage by the supplied, independently meshed boundary."""
    surface = source.outer_surface
    coordinates = np.asarray(surface.mesh.coordinates, dtype=np.float64)
    affine = source.labels.asset.spatial_affine
    doubled = 2.0 * np.asarray(affine.world_to_index(coordinates), dtype=np.float64)
    rounded = np.rint(doubled)
    if (
        not np.all(np.isfinite(doubled))
        or np.any(rounded < np.iinfo(np.int64).min)
        or np.any(rounded > np.iinfo(np.int64).max)
    ):
        raise ValueError("Outer surface vertices are outside the source lattice extent.")
    lattice = rounded.astype(np.int64)
    represented = np.asarray(affine.index_to_world(lattice.astype(np.float64) * 0.5))
    if not np.array_equal(represented, coordinates):
        raise ValueError(
            "Outer surface coordinates must be exact affine realizations of source lattice points."
        )

    expected_by_plane: dict[tuple[int, int, int], list[tuple[int, int, int, int]]] = {}
    for axis, plane, direction, loop in expected_faces:
        first_axis, second_axis = _FACE_AXES[axis]
        u = tuple(point[first_axis] for point in loop)
        v = tuple(point[second_axis] for point in loop)
        bounds = (min(u), max(u), min(v), max(v))
        if bounds[1] - bounds[0] != 2 or bounds[3] - bounds[2] != 2:
            raise ValueError("Source outer faces must be unit label-cell squares.")
        group = (axis, plane, direction)
        group_rectangles = expected_by_plane.setdefault(group, [])
        if bounds in group_rectangles:
            raise ValueError(
                "Categorical image boundary repeats one oriented lattice face."
            )
        group_rectangles.append(bounds)

    actual_by_plane: dict[tuple[int, int, int], list[tuple[_Point2, ...]]] = {}
    reflected = np.linalg.det(affine.matrix[:3, :3]) < 0.0
    for block in surface.mesh.blocks:
        if block.cell_kind != "triangle":
            raise ValueError("Outer compartment surface must use triangle cells.")
        for face in np.asarray(block.vertices, dtype=np.int64).tolist():
            keys = tuple(
                tuple(int(value) for value in lattice[int(vertex)]) for vertex in face
            )
            constant_axes = tuple(
                axis
                for axis in range(3)
                if keys[0][axis] == keys[1][axis] == keys[2][axis]
            )
            if len(constant_axes) != 1:
                raise ValueError(
                    "Every outer source triangle must lie in one image-cell face plane."
                )
            axis = constant_axes[0]
            plane = keys[0][axis]
            if plane % 2 == 0:
                raise ValueError(
                    "Outer source triangles must lie on voxel half-index planes."
                )
            first_axis, second_axis = _FACE_AXES[axis]
            triangle = tuple(
                (Fraction(point[first_axis]), Fraction(point[second_axis]))
                for point in keys
            )
            signed_area = _signed_area2(triangle)
            if signed_area == 0:
                raise ValueError("Outer source triangles must be nondegenerate.")
            direction = (1 if signed_area > 0 else -1) * (-1 if reflected else 1)
            actual_by_plane.setdefault((axis, plane, direction), []).append(triangle)

    if set(actual_by_plane) != set(expected_by_plane):
        raise ValueError(
            "Outer surface plane/orientation strata do not match the categorical image boundary."
        )
    for group, rectangles in expected_by_plane.items():
        coverage = {rectangle: Fraction(0) for rectangle in rectangles}
        clipped_by_rectangle: dict[
            tuple[int, int, int, int], list[tuple[_Point2, ...]]
        ] = {rectangle: [] for rectangle in rectangles}
        for triangle in actual_by_plane[group]:
            triangle_area = _polygon_area(triangle)
            if triangle_area <= 0:
                raise ValueError("Outer source triangles must have positive area.")
            triangle_bounds = (
                min(point[0] for point in triangle),
                max(point[0] for point in triangle),
                min(point[1] for point in triangle),
                max(point[1] for point in triangle),
            )
            covered_area = Fraction(0)
            for rectangle in rectangles:
                if (
                    triangle_bounds[1] <= rectangle[0]
                    or triangle_bounds[0] >= rectangle[1]
                    or triangle_bounds[3] <= rectangle[2]
                    or triangle_bounds[2] >= rectangle[3]
                ):
                    continue
                clipped = _clip_rectangle(triangle, rectangle)
                area = _polygon_area(clipped)
                if area:
                    covered_area += area
                    coverage[rectangle] += area
                    clipped_by_rectangle[rectangle].append(clipped)
            if covered_area != triangle_area:
                raise ValueError(
                    "An outer source triangle extends outside the categorical image boundary."
                )
        for rectangle, area in coverage.items():
            if area != 4:
                raise ValueError(
                    "Outer source triangles do not exactly cover each categorical boundary face."
                )
            polygons = clipped_by_rectangle[rectangle]
            for first in range(len(polygons)):
                for second in range(first + 1, len(polygons)):
                    if (
                        _polygon_area(
                            _clip_convex_polygon(polygons[first], polygons[second])
                        )
                        > 0
                    ):
                        raise ValueError(
                            "Outer source triangles overlap in the categorical image boundary."
                        )


@checked
def prepare_compartment_plc(
    source: CompartmentMeshingSource,
    /,
) -> CompartmentPlcPreparation:
    """Regenerate oriented exterior/material facets from categorical face incidence.

    Vertex identity comes from doubled source-index lattice keys. Region sides
    come from categorical voxel adjacency; no extracted mesh row or centroid
    determines source identity or material ownership.
    """
    if not isinstance(source, CompartmentMeshingSource):
        raise TypeError("source must be a CompartmentMeshingSource.")
    if source.outer_geometry is not None:
        raise ValueError(
            "A mapped outer geometry needs a source-to-lattice coverage proof before image PLC construction."
        )
    code, region_ids, interfaces = _compartment_codes(source)
    values: list[
        tuple[
            tuple[tuple[int, int, int], ...],
            tuple[str, tuple[str, ...]],
            tuple[int, int],
            str | None,
        ]
    ] = []
    region_index = {name: index for index, name in enumerate(region_ids)}
    reflected = np.linalg.det(source.labels.asset.spatial_affine.matrix[:3, :3]) < 0.0
    outer_faces: list[tuple[int, int, int, tuple[tuple[int, int, int], ...]]] = []
    for index in np.ndindex(code.shape):
        own = int(code[index])
        if own < 0:
            continue
        for axis in range(3):
            for direction in (-1, 1):
                neighbor = list(index)
                neighbor[axis] += direction
                other = (
                    int(code[tuple(neighbor)])
                    if 0 <= neighbor[axis] < code.shape[axis]
                    else -1
                )
                if other == own or (other >= 0 and direction < 0):
                    continue
                if other >= 0:
                    pair = _ordered_pair(region_ids[other], region_ids[own])
                    interface = interfaces.get(pair)
                    if interface is None:
                        first_region, second_region = pair
                        interface_id = None
                        interface_key = json.dumps(
                            ["undeclared-interface-pair", list(pair)],
                            separators=(",", ":"),
                        )
                        facet_interface_id = None
                    else:
                        interface_id, first_region, second_region = interface
                        interface_key = interface_id
                        facet_interface_id = interface_id
                    sides = (
                        region_index[second_region],
                        region_index[first_region],
                    )
                    reverse = (first_region == region_ids[own]) ^ reflected
                    facet_name = source_entity_id(
                        source.source_id,
                        source.source_revision,
                        "compartment-interface-facet",
                        interface_key,
                    )
                else:
                    sides = (-1, own)
                    reverse = (direction < 0) ^ reflected
                    outer_key = json.dumps(
                        [source.outer_surface.model_id, region_ids[own]],
                        separators=(",", ":"),
                    )
                    facet_name = source_entity_id(
                        source.source_id,
                        source.source_revision,
                        "compartment-outer-facet",
                        outer_key,
                    )
                    facet_interface_id = None
                loop = _image_face_corners(index, axis, direction)
                if reverse:
                    loop = tuple(reversed(loop))
                if other < 0:
                    outer_faces.append(
                        (axis, 2 * index[axis] + direction, direction, loop)
                    )
                values.append((loop, (facet_name, ()), sides, facet_interface_id))

    if not values:
        raise ValueError("Categorical source contains no occupied compartment cells.")
    _verify_outer_surface_lattice(source, tuple(outer_faces))

    facet_keys = tuple(sorted({facet_key for _, facet_key, _, _ in values}))
    facet_ids = tuple(name for name, _ in facet_keys)
    facet_occurrence_paths = tuple(path for _, path in facet_keys)
    facet_index = {key: row for row, key in enumerate(facet_keys)}
    facet_regions: list[tuple[int, int]] = [(-1, -1)] * len(facet_keys)
    facet_interface_ids: list[str | None] = [None] * len(facet_keys)
    facet_region_ids: list[tuple[str | None, str] | None] = [None] * len(facet_keys)
    polygon_facets: list[int] = []
    loops: list[tuple[tuple[int, int, int], ...]] = []
    vertex_keys = tuple(sorted({point for loop, _, _, _ in values for point in loop}))
    vertex_index = {key: row for row, key in enumerate(vertex_keys)}
    for loop, facet_key, sides, interface_id in values:
        facet = facet_index[facet_key]
        region_pair = (int(sides[0]), int(sides[1]))
        named_pair = (
            None if region_pair[0] < 0 else region_ids[region_pair[0]],
            region_ids[region_pair[1]],
        )
        previous = facet_region_ids[facet]
        if previous is not None and previous != named_pair:
            raise ValueError(
                "One compartment source facet cannot change region-side incidence."
            )
        facet_regions[facet] = region_pair
        facet_region_ids[facet] = named_pair
        facet_interface_ids[facet] = interface_id
        loops.append(loop)
        polygon_facets.append(facet)

    vertices = source.labels.asset.spatial_affine.index_to_world(
        np.asarray(vertex_keys, dtype=np.float64) * 0.5
    )
    vertex_ids = tuple(
        source_entity_id(
            source.source_id,
            source.source_revision,
            "lattice-vertex",
            json.dumps(key, separators=(",", ":")),
        )
        for key in vertex_keys
    )
    complex_ = PiecewiseLinearComplex(
        vertices,
        tuple(
            np.asarray([vertex_index[key] for key in loop], dtype=np.int64)
            for loop in loops
        ),
        np.asarray(polygon_facets, dtype=np.int64),
        np.asarray(facet_regions, dtype=np.int64),
        region_ids,
        segments=None,
        vertex_ids=vertex_ids,
        vertex_occurrence_paths=tuple(() for _ in vertex_ids),
        facet_ids=facet_ids,
        facet_occurrence_paths=facet_occurrence_paths,
        segment_ids=(),
        segment_occurrence_paths=(),
        region_occurrence_paths=tuple(() for _ in region_ids),
        boundary="conforming",
    )
    preparation_id = canonical_fingerprint(
        {
            "kind": "compartment-plc-preparation",
            "source": source.source_id,
            "revision": source.source_revision,
            "complex": complex_.complex_id,
            "lattice_vertices": vertex_keys,
            "facet_interfaces": tuple(facet_interface_ids),
            "facet_region_ids": tuple(facet_region_ids),
            "outer_surface": source.outer_surface.model_id,
        }
    )
    return CompartmentPlcPreparation(
        source.source_id,
        source.source_revision,
        complex_,
        vertex_keys,
        tuple(facet_interface_ids),
        tuple(value for value in facet_region_ids if value is not None),
        preparation_id,
    )


__all__ = ["CompartmentPlcPreparation", "prepare_compartment_plc"]

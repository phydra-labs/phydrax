#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Analysis-suitable bicubic T-spline basis substrate.

This module deliberately implements a narrow profile: rectilinear, analysis-suitable,
bicubic T-meshes over one rectangular parameter domain.  It provides local Bézier
extraction and sparse refinement routes; it never materializes a global extraction or
transfer matrix.  Extraordinary points and vendor-specific T-spline encodings are not
part of this profile.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from math import isfinite
from typing import Literal

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike
from scipy.interpolate import BSpline

from ..._fingerprint import array_tree_fingerprint, canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState


_BICUBIC_DEGREE = 3
_DEFAULT_TOLERANCE = 5.0e-11
_DIRECTIONS = ("left", "right", "down", "up")


def _identifier(name: str, value: str, /) -> str:
    identifier = str(value).strip()
    if not identifier:
        raise ValueError(f"{name} must be nonempty.")
    return identifier


def _coordinate(value: Sequence[float], /) -> tuple[float, float]:
    if len(value) != 2:
        raise ValueError("A T-mesh vertex coordinate must contain exactly two values.")
    point = (float(value[0]), float(value[1]))
    if not all(isfinite(component) for component in point):
        raise ValueError("T-mesh coordinates must be finite.")
    return point


def _bounds(
    value: Sequence[float] | Sequence[Sequence[float]], /
) -> tuple[float, float, float, float]:
    if len(value) == 2 and all(
        isinstance(axis, Sequence) and not isinstance(axis, (str, bytes))
        for axis in value
    ):
        axes = tuple(value)
        if len(axes[0]) != 2 or len(axes[1]) != 2:
            raise ValueError("Nested cell bounds must contain two two-value intervals.")
        result = (
            float(axes[0][0]),
            float(axes[0][1]),
            float(axes[1][0]),
            float(axes[1][1]),
        )
    else:
        if len(value) != 4:
            raise ValueError("Cell bounds must be (u0, u1, v0, v1).")
        result = tuple(float(component) for component in value)
    if not all(isfinite(component) for component in result):
        raise ValueError("Cell bounds must be finite.")
    if not result[0] < result[1] or not result[2] < result[3]:
        raise ValueError("T-mesh cells must have strictly positive parametric area.")
    return result


def _strict_breaks(name: str, values: Sequence[float], /) -> tuple[float, ...]:
    breaks = tuple(float(value) for value in values)
    if len(breaks) < 2 or not all(isfinite(value) for value in breaks):
        raise ValueError(f"{name} must contain at least two finite values.")
    if any(right <= left for left, right in zip(breaks, breaks[1:], strict=True)):
        raise ValueError(f"{name} must be strictly increasing.")
    return breaks


def _knot_tuple(name: str, values: Sequence[float], /) -> tuple[float, ...]:
    knots = tuple(float(value) for value in values)
    if len(knots) != _BICUBIC_DEGREE + 2:
        raise ValueError(f"{name} must contain exactly five knots for a cubic basis.")
    if not all(isfinite(value) for value in knots):
        raise ValueError(f"{name} must contain finite knots.")
    if any(right < left for left, right in zip(knots, knots[1:], strict=True)):
        raise ValueError(f"{name} must be nondecreasing.")
    if not knots[0] < knots[-1]:
        raise ValueError(f"{name} must have nonzero support.")
    return knots


def _array_digest(value: object, /) -> dict[str, object]:
    return array_tree_fingerprint(value)


class TVertex2D(StrictModule, NonTrainableState):
    """One immutable vertex of a rectilinear T-mesh."""

    vertex_id: int = eqx.field(static=True)
    parameter: tuple[float, float] = eqx.field(static=True)

    def __init__(self, vertex_id: int, parameter: Sequence[float], /):
        identifier = int(vertex_id)
        if identifier < 0:
            raise ValueError("vertex_id must be nonnegative.")
        self.vertex_id = identifier
        self.parameter = _coordinate(parameter)


class TEdge2D(StrictModule, NonTrainableState):
    """One atomic, axis-aligned T-mesh edge."""

    edge_id: int = eqx.field(static=True)
    vertex_ids: tuple[int, int] = eqx.field(static=True)
    cell_ids: tuple[int, ...] = eqx.field(static=True)
    axis: Literal["u", "v"] = eqx.field(static=True)

    def __init__(
        self,
        edge_id: int,
        vertex_ids: Sequence[int],
        cell_ids: Sequence[int],
        axis: Literal["u", "v"],
        /,
    ):
        identifier = int(edge_id)
        vertices = tuple(int(value) for value in vertex_ids)
        cells = tuple(sorted(int(value) for value in cell_ids))
        if identifier < 0 or len(vertices) != 2 or vertices[0] == vertices[1]:
            raise ValueError("A T-mesh edge requires an ID and two distinct vertices.")
        if axis not in ("u", "v"):
            raise ValueError("A T-mesh edge axis must be 'u' or 'v'.")
        if len(cells) not in (1, 2) or len(set(cells)) != len(cells):
            raise ValueError(
                "A manifold T-mesh edge must have one or two incident cells."
            )
        self.edge_id = identifier
        self.vertex_ids = vertices
        self.cell_ids = cells
        self.axis = axis


class TCell2D(StrictModule, NonTrainableState):
    """One immutable axis-aligned cell, including atomic boundary routes."""

    cell_id: int = eqx.field(static=True)
    parameter_bounds: tuple[float, float, float, float] = eqx.field(static=True)
    vertex_ids: tuple[int, ...] = eqx.field(static=True)
    edge_ids: tuple[int, ...] = eqx.field(static=True)
    level: int = eqx.field(static=True)

    def __init__(
        self,
        cell_id: int,
        parameter_bounds: Sequence[float] | Sequence[Sequence[float]],
        /,
        *,
        vertex_ids: Sequence[int] = (),
        edge_ids: Sequence[int] = (),
        level: int = 0,
    ):
        identifier = int(cell_id)
        level_ = int(level)
        if identifier < 0 or level_ < 0:
            raise ValueError("T-mesh cell IDs and levels must be nonnegative.")
        self.cell_id = identifier
        self.parameter_bounds = _bounds(parameter_bounds)
        self.vertex_ids = tuple(int(value) for value in vertex_ids)
        self.edge_ids = tuple(int(value) for value in edge_ids)
        self.level = level_

    @property
    def area(self) -> float:
        u0, u1, v0, v1 = self.parameter_bounds
        return (u1 - u0) * (v1 - v0)


class LocalKnotVector2D(StrictModule, NonTrainableState):
    """The two five-knot vectors defining one bicubic blending function."""

    u: tuple[float, ...] = eqx.field(static=True)
    v: tuple[float, ...] = eqx.field(static=True)

    def __init__(self, u: Sequence[float], v: Sequence[float], /):
        self.u = _knot_tuple("u knots", u)
        self.v = _knot_tuple("v knots", v)

    @property
    def support(self) -> tuple[float, float, float, float]:
        return (self.u[0], self.u[-1], self.v[0], self.v[-1])

    @property
    def greville_parameter(self) -> tuple[float, float]:
        return (
            sum(self.u[1:4]) / _BICUBIC_DEGREE,
            sum(self.v[1:4]) / _BICUBIC_DEGREE,
        )


class TAnchor2D(StrictModule, NonTrainableState):
    """One bicubic T-spline anchor and its complete local knot evidence."""

    anchor_id: int = eqx.field(static=True)
    parameter: tuple[float, float] = eqx.field(static=True)
    local_knots: LocalKnotVector2D

    def __init__(
        self,
        anchor_id: int,
        parameter: Sequence[float],
        local_knots: LocalKnotVector2D,
        /,
    ):
        identifier = int(anchor_id)
        if identifier < 0:
            raise ValueError("anchor_id must be nonnegative.")
        if not isinstance(local_knots, LocalKnotVector2D):
            raise TypeError("local_knots must be LocalKnotVector2D.")
        self.anchor_id = identifier
        self.parameter = _coordinate(parameter)
        self.local_knots = local_knots


class TJunctionExtension2D(StrictModule, NonTrainableState):
    """Odd-degree T-junction extension: one edge span back, two face spans forward."""

    junction_vertex_id: int = eqx.field(static=True)
    axis: Literal["u", "v"] = eqx.field(static=True)
    missing_direction: Literal["left", "right", "down", "up"] = eqx.field(static=True)
    segment: tuple[float, float, float, float] = eqx.field(static=True)
    backward_crossings: int = eqx.field(static=True)
    forward_crossings: int = eqx.field(static=True)

    def __init__(
        self,
        junction_vertex_id: int,
        axis: Literal["u", "v"],
        missing_direction: Literal["left", "right", "down", "up"],
        segment: Sequence[float],
        /,
        *,
        backward_crossings: int,
        forward_crossings: int,
    ):
        identifier = int(junction_vertex_id)
        segment_ = tuple(float(value) for value in segment)
        if identifier < 0 or axis not in ("u", "v"):
            raise ValueError("Invalid T-junction extension identity or axis.")
        if missing_direction not in _DIRECTIONS:
            raise ValueError("Invalid T-junction missing direction.")
        if len(segment_) != 4 or not all(isfinite(value) for value in segment_):
            raise ValueError("A T-junction extension segment needs four finite values.")
        if axis == "u" and not (segment_[2] == segment_[3] and segment_[0] < segment_[1]):
            raise ValueError("A u extension must be a nondegenerate horizontal segment.")
        if axis == "v" and not (segment_[0] == segment_[1] and segment_[2] < segment_[3]):
            raise ValueError("A v extension must be a nondegenerate vertical segment.")
        self.junction_vertex_id = identifier
        self.axis = axis
        self.missing_direction = missing_direction
        self.segment = segment_
        self.backward_crossings = int(backward_crossings)
        self.forward_crossings = int(forward_crossings)


class TMesh2D(StrictModule, NonTrainableState):
    """Immutable rectilinear bicubic T-mesh with explicit blending anchors."""

    vertices: tuple[TVertex2D, ...]
    edges: tuple[TEdge2D, ...]
    cells: tuple[TCell2D, ...]
    anchors: tuple[TAnchor2D, ...]
    t_junction_extensions: tuple[TJunctionExtension2D, ...]
    degree: tuple[int, int] = eqx.field(static=True)
    parameter_domain: tuple[float, float, float, float] = eqx.field(static=True)
    mesh_id: str = eqx.field(static=True)

    def __init__(
        self,
        cells: Sequence[TCell2D | Sequence[float] | Sequence[Sequence[float]]],
        anchors: Sequence[TAnchor2D],
        /,
        *,
        degree: Sequence[int] = (_BICUBIC_DEGREE, _BICUBIC_DEGREE),
        mesh_id: str | None = None,
        proprietary_format: str | None = None,
        extraordinary: bool = False,
    ):
        degrees = tuple(int(value) for value in degree)
        if degrees != (_BICUBIC_DEGREE, _BICUBIC_DEGREE):
            raise NotImplementedError("F7 supports bicubic ASTS2D bases only.")
        if proprietary_format is not None:
            raise NotImplementedError(
                "Proprietary T-spline encodings require an independently qualified adapter."
            )
        if bool(extraordinary):
            raise NotImplementedError(
                "Extraordinary vertices are outside the basis-only ASTS2D profile."
            )
        raw_cells = tuple(cells)
        if not raw_cells:
            raise ValueError("A T-mesh requires at least one active cell.")
        cell_ids: list[int] = []
        cell_bounds: list[tuple[float, float, float, float]] = []
        cell_levels: list[int] = []
        for index, cell in enumerate(raw_cells):
            if isinstance(cell, TCell2D):
                cell_ids.append(cell.cell_id)
                cell_bounds.append(cell.parameter_bounds)
                cell_levels.append(cell.level)
            else:
                cell_ids.append(index)
                cell_bounds.append(_bounds(cell))
                cell_levels.append(0)
        if len(set(cell_ids)) != len(cell_ids):
            raise ValueError("T-mesh cell IDs must be unique.")
        anchors_ = tuple(anchors)
        if not anchors_ or not all(isinstance(anchor, TAnchor2D) for anchor in anchors_):
            raise TypeError("anchors must be a nonempty sequence of TAnchor2D values.")
        anchor_ids = tuple(anchor.anchor_id for anchor in anchors_)
        if len(set(anchor_ids)) != len(anchor_ids):
            raise ValueError("T-spline anchor IDs must be unique.")
        vertices, edges, cells_ = _build_topology(cell_ids, cell_bounds, cell_levels)
        domain = (
            min(bounds[0] for bounds in cell_bounds),
            max(bounds[1] for bounds in cell_bounds),
            min(bounds[2] for bounds in cell_bounds),
            max(bounds[3] for bounds in cell_bounds),
        )
        extensions = _t_junction_extensions(vertices, edges, domain)
        self.vertices = vertices
        self.edges = edges
        self.cells = cells_
        self.anchors = tuple(sorted(anchors_, key=lambda anchor: anchor.anchor_id))
        self.t_junction_extensions = extensions
        self.degree = degrees
        self.parameter_domain = domain
        payload = {
            "kind": "asts2d-bicubic-tmesh",
            "degree": list(degrees),
            "cells": [
                {
                    "id": cell.cell_id,
                    "bounds": list(cell.parameter_bounds),
                    "level": cell.level,
                }
                for cell in cells_
            ],
            "anchors": [
                {
                    "id": anchor.anchor_id,
                    "parameter": list(anchor.parameter),
                    "u": list(anchor.local_knots.u),
                    "v": list(anchor.local_knots.v),
                }
                for anchor in self.anchors
            ],
        }
        self.mesh_id = (
            canonical_fingerprint(payload)
            if mesh_id is None
            else _identifier("mesh_id", mesh_id)
        )

    @classmethod
    def tensor_product(
        cls,
        u_breaks: Sequence[float],
        v_breaks: Sequence[float],
        /,
        *,
        mesh_id: str | None = None,
    ) -> TMesh2D:
        """Construct the canonical open bicubic tensor-product member of ASTS2D."""
        u = _strict_breaks("u_breaks", u_breaks)
        v = _strict_breaks("v_breaks", v_breaks)
        cells = [
            (u[i], u[i + 1], v[j], v[j + 1])
            for i in range(len(u) - 1)
            for j in range(len(v) - 1)
        ]
        u_knots = _open_knot_vector(u)
        v_knots = _open_knot_vector(v)
        local_u = tuple(
            tuple(u_knots[index : index + _BICUBIC_DEGREE + 2])
            for index in range(len(u_knots) - _BICUBIC_DEGREE - 1)
        )
        local_v = tuple(
            tuple(v_knots[index : index + _BICUBIC_DEGREE + 2])
            for index in range(len(v_knots) - _BICUBIC_DEGREE - 1)
        )
        anchors = []
        anchor_id = 0
        for u_local in local_u:
            for v_local in local_v:
                anchors.append(
                    TAnchor2D(
                        anchor_id,
                        (u_local[2], v_local[2]),
                        LocalKnotVector2D(u_local, v_local),
                    )
                )
                anchor_id += 1
        return cls(cells, anchors, mesh_id=mesh_id)

    @classmethod
    def from_cells(
        cls,
        cells: Sequence[TCell2D | Sequence[float] | Sequence[Sequence[float]]],
        anchors: Sequence[TAnchor2D],
        /,
        *,
        mesh_id: str | None = None,
    ) -> TMesh2D:
        """Construct a native ASTS mesh from rectangular cells and explicit anchors."""
        return cls(cells, anchors, mesh_id=mesh_id)

    @property
    def cell_count(self) -> int:
        return len(self.cells)

    @property
    def coefficient_count(self) -> int:
        return len(self.anchors)

    @property
    def t_junction_count(self) -> int:
        return len(self.t_junction_extensions)

    @property
    def axis_breaks(self) -> tuple[tuple[float, ...], tuple[float, ...]]:
        return (
            tuple(
                sorted(
                    {value for cell in self.cells for value in cell.parameter_bounds[:2]}
                )
            ),
            tuple(
                sorted(
                    {value for cell in self.cells for value in cell.parameter_bounds[2:]}
                )
            ),
        )

    def cell(self, cell_id: int, /) -> TCell2D:
        identifier = int(cell_id)
        for cell in self.cells:
            if cell.cell_id == identifier:
                return cell
        raise KeyError(f"Unknown T-mesh cell {identifier}.")

    def certify(self, *, tolerance: float = _DEFAULT_TOLERANCE) -> ASTSCertificate:
        return certify_asts(self, tolerance=tolerance)

    def realize(
        self, *, tolerance: float = _DEFAULT_TOLERANCE
    ) -> LocalExtractedBernsteinRealization:
        certificate = certify_asts(self, tolerance=tolerance)
        if not certificate.passed:
            reasons = ", ".join(certificate.failure_reasons)
            raise ValueError(f"T-mesh is not a certified ASTS2D basis: {reasons}.")
        return _local_realization(self, tolerance=tolerance)

    def refine(
        self,
        marked_cell_ids: Sequence[int],
        /,
        *,
        tolerance: float = _DEFAULT_TOLERANCE,
    ) -> ASTSRefinement:
        return analysis_suitable_refinement_closure(
            self, marked_cell_ids, tolerance=tolerance
        )


def _open_knot_vector(breaks: Sequence[float], /) -> tuple[float, ...]:
    return (
        (float(breaks[0]),) * (_BICUBIC_DEGREE + 1)
        + tuple(float(value) for value in breaks[1:-1])
        + (float(breaks[-1]),) * (_BICUBIC_DEGREE + 1)
    )


def _build_topology(
    cell_ids: Sequence[int],
    bounds: Sequence[tuple[float, float, float, float]],
    levels: Sequence[int],
    /,
) -> tuple[tuple[TVertex2D, ...], tuple[TEdge2D, ...], tuple[TCell2D, ...]]:
    coordinates = sorted(
        {
            coordinate
            for u0, u1, v0, v1 in bounds
            for coordinate in ((u0, v0), (u0, v1), (u1, v0), (u1, v1))
        }
    )
    vertices = tuple(
        TVertex2D(index, coordinate) for index, coordinate in enumerate(coordinates)
    )
    vertex_by_coordinate = {vertex.parameter: vertex.vertex_id for vertex in vertices}
    edge_cells: dict[tuple[int, int], set[int]] = {}
    cell_edge_keys: dict[int, list[tuple[int, int]]] = {
        cell_id: [] for cell_id in cell_ids
    }
    for cell_id, (u0, u1, v0, v1) in zip(cell_ids, bounds, strict=True):
        sides = (
            ("u", v0, u0, u1),
            ("u", v1, u0, u1),
            ("v", u0, v0, v1),
            ("v", u1, v0, v1),
        )
        for axis, fixed, lower, upper in sides:
            on_side = sorted(
                vertex.parameter[0] if axis == "u" else vertex.parameter[1]
                for vertex in vertices
                if (
                    (
                        vertex.parameter[1] == fixed
                        if axis == "u"
                        else vertex.parameter[0] == fixed
                    )
                    and lower
                    <= (vertex.parameter[0] if axis == "u" else vertex.parameter[1])
                    <= upper
                )
            )
            for first, second in zip(on_side, on_side[1:], strict=True):
                p0 = (first, fixed) if axis == "u" else (fixed, first)
                p1 = (second, fixed) if axis == "u" else (fixed, second)
                key = tuple(sorted((vertex_by_coordinate[p0], vertex_by_coordinate[p1])))
                edge_cells.setdefault(key, set()).add(cell_id)
                cell_edge_keys[cell_id].append(key)
    sorted_keys = sorted(edge_cells)
    edge_id_by_key = {key: index for index, key in enumerate(sorted_keys)}
    edges = []
    for edge_id, key in enumerate(sorted_keys):
        first = vertices[key[0]].parameter
        second = vertices[key[1]].parameter
        axis: Literal["u", "v"] = "u" if first[1] == second[1] else "v"
        edges.append(TEdge2D(edge_id, key, sorted(edge_cells[key]), axis))
    cells = []
    for cell_id, cell_bounds, level in zip(cell_ids, bounds, levels, strict=True):
        u0, u1, v0, v1 = cell_bounds
        vertex_ids = tuple(
            vertex.vertex_id
            for vertex in vertices
            if (vertex.parameter[0] in (u0, u1) and vertex.parameter[1] in (v0, v1))
        )
        edge_ids = tuple(sorted({edge_id_by_key[key] for key in cell_edge_keys[cell_id]}))
        cells.append(
            TCell2D(
                cell_id,
                cell_bounds,
                vertex_ids=vertex_ids,
                edge_ids=edge_ids,
                level=level,
            )
        )
    return vertices, tuple(edges), tuple(sorted(cells, key=lambda cell: cell.cell_id))


def _vertex_directions(
    vertex: TVertex2D,
    vertices: Sequence[TVertex2D],
    edges: Sequence[TEdge2D],
    /,
) -> frozenset[str]:
    directions: set[str] = set()
    u, v = vertex.parameter
    for edge in edges:
        if vertex.vertex_id not in edge.vertex_ids:
            continue
        other_id = (
            edge.vertex_ids[1]
            if edge.vertex_ids[0] == vertex.vertex_id
            else edge.vertex_ids[0]
        )
        other_u, other_v = vertices[other_id].parameter
        if other_u < u:
            directions.add("left")
        elif other_u > u:
            directions.add("right")
        elif other_v < v:
            directions.add("down")
        else:
            directions.add("up")
    return frozenset(directions)


def _t_junction_extensions(
    vertices: Sequence[TVertex2D],
    edges: Sequence[TEdge2D],
    domain: tuple[float, float, float, float],
    /,
) -> tuple[TJunctionExtension2D, ...]:
    u0, u1, v0, v1 = domain
    result = []
    for vertex in vertices:
        u, v = vertex.parameter
        if u in (u0, u1) or v in (v0, v1):
            continue
        directions = _vertex_directions(vertex, vertices, edges)
        if len(directions) != 3:
            continue
        missing = next(
            direction for direction in _DIRECTIONS if direction not in directions
        )
        axis: Literal["u", "v"] = "u" if missing in ("left", "right") else "v"
        if axis == "u":
            crossings = sorted(
                {
                    vertices[edge.vertex_ids[0]].parameter[0]
                    for edge in edges
                    if edge.axis == "v"
                    and min(
                        vertices[edge.vertex_ids[0]].parameter[1],
                        vertices[edge.vertex_ids[1]].parameter[1],
                    )
                    <= v
                    <= max(
                        vertices[edge.vertex_ids[0]].parameter[1],
                        vertices[edge.vertex_ids[1]].parameter[1],
                    )
                }
            )
            lower = [value for value in crossings if value < u]
            upper = [value for value in crossings if value > u]
            backward_values = lower if missing == "right" else upper
            forward_values = upper if missing == "right" else lower
            backward = (
                backward_values[-1]
                if missing == "right" and backward_values
                else backward_values[0]
                if backward_values
                else u
            )
            if missing == "right":
                forward = (
                    forward_values[min(1, len(forward_values) - 1)]
                    if forward_values
                    else u
                )
                segment = (backward, forward, v, v)
            else:
                forward = (
                    forward_values[max(0, len(forward_values) - 2)]
                    if forward_values
                    else u
                )
                segment = (forward, backward, v, v)
        else:
            crossings = sorted(
                {
                    vertices[edge.vertex_ids[0]].parameter[1]
                    for edge in edges
                    if edge.axis == "u"
                    and min(
                        vertices[edge.vertex_ids[0]].parameter[0],
                        vertices[edge.vertex_ids[1]].parameter[0],
                    )
                    <= u
                    <= max(
                        vertices[edge.vertex_ids[0]].parameter[0],
                        vertices[edge.vertex_ids[1]].parameter[0],
                    )
                }
            )
            lower = [value for value in crossings if value < v]
            upper = [value for value in crossings if value > v]
            backward_values = lower if missing == "up" else upper
            forward_values = upper if missing == "up" else lower
            backward = (
                backward_values[-1]
                if missing == "up" and backward_values
                else backward_values[0]
                if backward_values
                else v
            )
            if missing == "up":
                forward = (
                    forward_values[min(1, len(forward_values) - 1)]
                    if forward_values
                    else v
                )
                segment = (u, u, backward, forward)
            else:
                forward = (
                    forward_values[max(0, len(forward_values) - 2)]
                    if forward_values
                    else v
                )
                segment = (u, u, forward, backward)
        if (axis == "u" and segment[0] < segment[1]) or (
            axis == "v" and segment[2] < segment[3]
        ):
            result.append(
                TJunctionExtension2D(
                    vertex.vertex_id,
                    axis,
                    missing,
                    segment,
                    backward_crossings=min(1, len(backward_values)),
                    forward_crossings=min(2, len(forward_values)),
                )
            )
    return tuple(result)


def _extensions_are_analysis_suitable(
    extensions: Sequence[TJunctionExtension2D], /
) -> bool:
    horizontal = tuple(extension for extension in extensions if extension.axis == "u")
    vertical = tuple(extension for extension in extensions if extension.axis == "v")
    for first in horizontal:
        hu0, hu1, hv, _ = first.segment
        for second in vertical:
            vu, _, vv0, vv1 = second.segment
            if hu0 < vu < hu1 and vv0 < hv < vv1:
                return False
    return True


def _parameter_partition_is_valid(mesh: TMesh2D, tolerance: float, /) -> bool:
    cells = mesh.cells
    u0, u1, v0, v1 = mesh.parameter_domain
    domain_area = (u1 - u0) * (v1 - v0)
    scale = max(domain_area, 1.0)
    if abs(sum(cell.area for cell in cells) - domain_area) > tolerance * scale:
        return False
    for index, first in enumerate(cells):
        a0, a1, b0, b1 = first.parameter_bounds
        for second in cells[index + 1 :]:
            c0, c1, d0, d1 = second.parameter_bounds
            overlap_u = min(a1, c1) - max(a0, c0)
            overlap_v = min(b1, d1) - max(b0, d0)
            if overlap_u > tolerance and overlap_v > tolerance:
                return False
    return all(len(edge.cell_ids) in (1, 2) for edge in mesh.edges)


def _interior_multiplicity_is_supported(
    knots: Sequence[float], domain: tuple[float, float], /
) -> bool:
    lower, upper = domain
    values, counts = np.unique(np.asarray(knots), return_counts=True)
    return all(
        count == 1 or value in (lower, upper)
        for value, count in zip(values, counts, strict=True)
    )


def _knot_metadata_is_compatible(mesh: TMesh2D, tolerance: float, /) -> bool:
    u_breaks, v_breaks = mesh.axis_breaks
    u_set = set(u_breaks)
    v_set = set(v_breaks)
    u_domain = mesh.parameter_domain[:2]
    v_domain = mesh.parameter_domain[2:]
    signatures: set[tuple[tuple[float, ...], tuple[float, ...]]] = set()
    for anchor in mesh.anchors:
        knots = anchor.local_knots
        if not set(knots.u).issubset(u_set) or not set(knots.v).issubset(v_set):
            return False
        if knots.u[0] < u_domain[0] or knots.u[-1] > u_domain[1]:
            return False
        if knots.v[0] < v_domain[0] or knots.v[-1] > v_domain[1]:
            return False
        if not _interior_multiplicity_is_supported(knots.u, u_domain):
            return False
        if not _interior_multiplicity_is_supported(knots.v, v_domain):
            return False
        if (
            abs(anchor.parameter[0] - knots.u[2]) > tolerance
            or abs(anchor.parameter[1] - knots.v[2]) > tolerance
        ):
            return False
        signature = (knots.u, knots.v)
        if signature in signatures:
            return False
        signatures.add(signature)
    return True


def _bspline_value(knots: Sequence[float], degree: int, point: float, /) -> float:
    knots_ = tuple(float(value) for value in knots)
    if degree == 0:
        return float(knots_[0] <= point < knots_[1])
    value = 0.0
    left_denominator = knots_[degree] - knots_[0]
    if left_denominator > 0.0:
        value += (
            (point - knots_[0])
            / left_denominator
            * _bspline_value(knots_[:-1], degree - 1, point)
        )
    right_denominator = knots_[degree + 1] - knots_[1]
    if right_denominator > 0.0:
        value += (
            (knots_[degree + 1] - point)
            / right_denominator
            * _bspline_value(knots_[1:], degree - 1, point)
        )
    return value


def _bernstein_cubic(points: np.ndarray, derivative: int = 0) -> np.ndarray:
    t = np.asarray(points, dtype=float)
    if derivative == 0:
        return np.stack(
            ((1.0 - t) ** 3, 3.0 * t * (1.0 - t) ** 2, 3.0 * t**2 * (1.0 - t), t**3),
            axis=-1,
        )
    if derivative == 1:
        return np.stack(
            (
                -3.0 * (1.0 - t) ** 2,
                3.0 * (1.0 - t) * (1.0 - 3.0 * t),
                3.0 * t * (2.0 - 3.0 * t),
                3.0 * t**2,
            ),
            axis=-1,
        )
    if derivative == 2:
        return np.stack(
            (6.0 * (1.0 - t), 18.0 * t - 12.0, 6.0 - 18.0 * t, 6.0 * t),
            axis=-1,
        )
    if derivative == 3:
        return np.broadcast_to(np.asarray((-6.0, 18.0, -18.0, 6.0)), t.shape + (4,))
    if derivative > 3:
        return np.zeros(t.shape + (4,), dtype=float)
    raise ValueError("Bernstein derivative orders must be nonnegative.")


def _univariate_extraction(
    knots: Sequence[float], interval: tuple[float, float], /
) -> np.ndarray:
    lower, upper = interval
    if knots[0] >= upper or knots[-1] <= lower:
        return np.zeros((4,), dtype=float)
    if any(lower < knot < upper for knot in knots):
        raise ValueError("A local knot line cuts the interior of one extraction cell.")
    reference = np.asarray((0.125, 0.375, 0.625, 0.875))
    collocation = _bernstein_cubic(reference)
    points = lower + (upper - lower) * reference
    values = np.asarray(
        [_bspline_value(knots, _BICUBIC_DEGREE, float(point)) for point in points]
    )
    coefficients = np.linalg.solve(collocation, values)
    coefficients[np.abs(coefficients) < 32.0 * np.finfo(float).eps] = 0.0
    return coefficients


class ExtractedBernstein(StrictModule, NonTrainableState):
    """Cell-local cubic Bernstein extraction; rows are active T-spline anchors."""

    cell_id: int = eqx.field(static=True)
    parameter_bounds: tuple[float, float, float, float] = eqx.field(static=True)
    anchor_ids: Array
    extraction_operator: Array
    rank: int = eqx.field(static=True)
    condition_number: float = eqx.field(static=True)
    extraction_id: str = eqx.field(static=True)

    def __init__(
        self,
        cell_id: int,
        parameter_bounds: Sequence[float],
        anchor_ids: ArrayLike,
        extraction_operator: ArrayLike,
        /,
    ):
        bounds_ = _bounds(parameter_bounds)
        anchors = np.asarray(anchor_ids)
        operator = np.asarray(extraction_operator, dtype=float)
        if anchors.ndim != 1 or not np.issubdtype(anchors.dtype, np.integer):
            raise TypeError("anchor_ids must be one rank-1 integer array.")
        if operator.shape != (anchors.size, 16):
            raise ValueError("A bicubic extraction operator must have shape (local, 16).")
        rank = int(np.linalg.matrix_rank(operator))
        singular_values = np.linalg.svd(operator, compute_uv=False)
        condition = (
            float(singular_values[0] / singular_values[-1])
            if singular_values.size and singular_values[-1] > 0.0
            else float("inf")
        )
        self.cell_id = int(cell_id)
        self.parameter_bounds = bounds_
        self.anchor_ids = jnp.asarray(anchors, dtype=jnp.int32)
        self.extraction_operator = jnp.asarray(operator)
        self.rank = rank
        self.condition_number = condition
        self.extraction_id = canonical_fingerprint(
            {
                "kind": "asts2d-local-extraction",
                "cell": int(cell_id),
                "bounds": list(bounds_),
                "anchors": _array_digest(anchors.astype(np.int64, copy=False)),
                "operator": _array_digest(operator),
            }
        )

    @property
    def local_width(self) -> int:
        return int(self.anchor_ids.shape[0])

    def evaluate_reference(
        self,
        reference_points: ArrayLike,
        /,
        *,
        derivative: tuple[int, int] = (0, 0),
    ) -> Array:
        points = np.asarray(reference_points, dtype=float)
        if points.ndim != 2 or points.shape[1] != 2:
            raise ValueError("reference_points must have shape (point, 2).")
        du, dv = (int(derivative[0]), int(derivative[1]))
        if du < 0 or dv < 0:
            raise ValueError("Derivative orders must be nonnegative.")
        bu = _bernstein_cubic(points[:, 0], du)
        bv = _bernstein_cubic(points[:, 1], dv)
        tensor = np.asarray(
            [
                np.outer(first, second).reshape(-1)
                for first, second in zip(bu, bv, strict=True)
            ]
        )
        values = tensor @ np.asarray(self.extraction_operator).T
        return jnp.asarray(values)

    def evaluate(
        self,
        parameter_points: ArrayLike,
        /,
        *,
        derivative: tuple[int, int] = (0, 0),
    ) -> Array:
        points = np.asarray(parameter_points, dtype=float)
        if points.ndim != 2 or points.shape[1] != 2:
            raise ValueError("parameter_points must have shape (point, 2).")
        u0, u1, v0, v1 = self.parameter_bounds
        tolerance = 16.0 * np.finfo(float).eps * max(u1 - u0, v1 - v0, 1.0)
        if np.any(points[:, 0] < u0 - tolerance) or np.any(points[:, 0] > u1 + tolerance):
            raise ValueError("A parameter point lies outside the extraction cell.")
        if np.any(points[:, 1] < v0 - tolerance) or np.any(points[:, 1] > v1 + tolerance):
            raise ValueError("A parameter point lies outside the extraction cell.")
        reference = np.stack(
            ((points[:, 0] - u0) / (u1 - u0), (points[:, 1] - v0) / (v1 - v0)),
            axis=-1,
        )
        du, dv = derivative
        scale = (u1 - u0) ** (-int(du)) * (v1 - v0) ** (-int(dv))
        return scale * self.evaluate_reference(reference, derivative=derivative)


class LocalExtractedBernsteinRealization(StrictModule, NonTrainableState):
    """Local ASTS realization adapter with gathers and no global operator."""

    mesh: TMesh2D
    extractions: tuple[ExtractedBernstein, ...]
    realization_id: str = eqx.field(static=True)
    basis_id: str = eqx.field(static=True)
    global_coefficient_count: int = eqx.field(static=True)
    local_width: int = eqx.field(static=True)
    cell_ranks: Array
    cell_condition_numbers: Array

    def __init__(self, mesh: TMesh2D, extractions: Sequence[ExtractedBernstein], /):
        if not isinstance(mesh, TMesh2D):
            raise TypeError("mesh must be TMesh2D.")
        extractions_ = tuple(extractions)
        if len(extractions_) != mesh.cell_count:
            raise ValueError("Every T-mesh cell requires one extraction operator.")
        if {item.cell_id for item in extractions_} != {
            cell.cell_id for cell in mesh.cells
        }:
            raise ValueError("Extraction cell IDs must exactly cover the T-mesh cells.")
        self.mesh = mesh
        self.extractions = tuple(sorted(extractions_, key=lambda item: item.cell_id))
        self.basis_id = mesh.mesh_id
        self.global_coefficient_count = mesh.coefficient_count
        self.local_width = max(item.local_width for item in self.extractions)
        self.cell_ranks = jnp.asarray(
            [item.rank for item in self.extractions], dtype=jnp.int32
        )
        self.cell_condition_numbers = jnp.asarray(
            [item.condition_number for item in self.extractions]
        )
        self.realization_id = canonical_fingerprint(
            {
                "kind": "asts2d-extracted-bernstein-realization",
                "basis": mesh.mesh_id,
                "extractions": [item.extraction_id for item in self.extractions],
            }
        )

    def extraction(self, cell_id: int, /) -> ExtractedBernstein:
        identifier = int(cell_id)
        for extraction in self.extractions:
            if extraction.cell_id == identifier:
                return extraction
        raise KeyError(f"Unknown extraction cell {identifier}.")

    def gather(self, coefficients: ArrayLike, cell_id: int, /) -> Array:
        values = jnp.asarray(coefficients)
        if values.shape[0] != self.global_coefficient_count:
            raise ValueError("Coefficient leading extent does not match the ASTS basis.")
        return values[self.extraction(cell_id).anchor_ids]

    def gather_transpose(self, local_values: ArrayLike, cell_id: int, /) -> Array:
        extraction = self.extraction(cell_id)
        values = jnp.asarray(local_values)
        if values.shape[0] != extraction.local_width:
            raise ValueError("Local value leading extent does not match the cell gather.")
        result = jnp.zeros(
            (self.global_coefficient_count,) + values.shape[1:], dtype=values.dtype
        )
        return result.at[extraction.anchor_ids].add(values)

    def evaluate_cell(
        self,
        cell_id: int,
        parameter_points: ArrayLike,
        coefficients: ArrayLike,
        /,
        *,
        derivative: tuple[int, int] = (0, 0),
    ) -> Array:
        extraction = self.extraction(cell_id)
        basis = extraction.evaluate(parameter_points, derivative=derivative)
        local = self.gather(coefficients, cell_id)
        return jnp.tensordot(basis, local, axes=((1,), (0,)))

    def evaluate(
        self,
        parameter_points: ArrayLike,
        coefficients: ArrayLike,
        /,
        *,
        derivative: tuple[int, int] = (0, 0),
    ) -> Array:
        points = np.asarray(parameter_points, dtype=float)
        if points.ndim != 2 or points.shape[1] != 2:
            raise ValueError("parameter_points must have shape (point, 2).")
        coefficient_values = jnp.asarray(coefficients)
        outputs = []
        for point in points:
            candidates = [
                cell
                for cell in self.mesh.cells
                if cell.parameter_bounds[0] <= point[0] <= cell.parameter_bounds[1]
                and cell.parameter_bounds[2] <= point[1] <= cell.parameter_bounds[3]
            ]
            if not candidates:
                raise ValueError("A parameter point lies outside the T-mesh domain.")
            cell = min(candidates, key=lambda candidate: candidate.cell_id)
            outputs.append(
                self.evaluate_cell(
                    cell.cell_id,
                    point[None, :],
                    coefficient_values,
                    derivative=derivative,
                )[0]
            )
        return jnp.stack(outputs)


def _cell_extraction(mesh: TMesh2D, cell: TCell2D, /) -> ExtractedBernstein:
    u0, u1, v0, v1 = cell.parameter_bounds
    anchor_ids = []
    rows = []
    for anchor in mesh.anchors:
        knots = anchor.local_knots
        if knots.u[0] >= u1 or knots.u[-1] <= u0:
            continue
        if knots.v[0] >= v1 or knots.v[-1] <= v0:
            continue
        u_coefficients = _univariate_extraction(knots.u, (u0, u1))
        v_coefficients = _univariate_extraction(knots.v, (v0, v1))
        row = np.outer(u_coefficients, v_coefficients).reshape(-1)
        if np.max(np.abs(row)) > 0.0:
            anchor_ids.append(anchor.anchor_id)
            rows.append(row)
    operator = np.asarray(rows, dtype=float)
    if operator.size == 0:
        operator = np.empty((0, 16), dtype=float)
    return ExtractedBernstein(cell.cell_id, cell.parameter_bounds, anchor_ids, operator)


def _local_realization(
    mesh: TMesh2D, /, *, tolerance: float
) -> LocalExtractedBernsteinRealization:
    del tolerance
    return LocalExtractedBernsteinRealization(
        mesh, tuple(_cell_extraction(mesh, cell) for cell in mesh.cells)
    )


class ASTSCertificate(StrictModule, NonTrainableState):
    """Fail-closed evidence for the complete basis-only ASTS2D claim."""

    mesh_id: str = eqx.field(static=True)
    analysis_suitable: bool = eqx.field(static=True)
    dual_compatible: bool = eqx.field(static=True)
    partition_of_unity: bool = eqx.field(static=True)
    nonnegative: bool = eqx.field(static=True)
    linearly_independent: bool = eqx.field(static=True)
    extraction_full_rank: bool = eqx.field(static=True)
    geometry_valid: bool = eqx.field(static=True)
    maximum_partition_error: float = eqx.field(static=True)
    minimum_basis_value: float = eqx.field(static=True)
    minimum_extraction_rank: int = eqx.field(static=True)
    minimum_rank_margin: int = eqx.field(static=True)
    maximum_extraction_condition: float = eqx.field(static=True)
    tolerance: float = eqx.field(static=True)
    failure_reasons: tuple[str, ...] = eqx.field(static=True)
    certificate_id: str = eqx.field(static=True)

    @property
    def passed(self) -> bool:
        return (
            self.analysis_suitable
            and self.dual_compatible
            and self.partition_of_unity
            and self.nonnegative
            and self.linearly_independent
            and self.extraction_full_rank
            and self.geometry_valid
        )

    def qualification_payload(self) -> dict[str, object]:
        """Return JSON-ready qualification producer evidence without publishing it."""
        return {
            "profile": "F7-ASTS2D-bicubic-basis-only",
            "mesh_id": self.mesh_id,
            "certificate_id": self.certificate_id,
            "passed": self.passed,
            "checks": {
                "analysis_suitable": self.analysis_suitable,
                "dual_compatible": self.dual_compatible,
                "partition_of_unity": self.partition_of_unity,
                "nonnegative": self.nonnegative,
                "linearly_independent": self.linearly_independent,
                "extraction_full_rank": self.extraction_full_rank,
                "parametric_geometry_valid": self.geometry_valid,
            },
            "metrics": {
                "maximum_partition_error": self.maximum_partition_error,
                "minimum_basis_value": self.minimum_basis_value,
                "minimum_extraction_rank": self.minimum_extraction_rank,
                "minimum_rank_margin": self.minimum_rank_margin,
                "maximum_extraction_condition": self.maximum_extraction_condition,
                "tolerance": self.tolerance,
            },
            "failure_reasons": list(self.failure_reasons),
        }


def certify_asts(
    mesh: TMesh2D,
    /,
    *,
    tolerance: float = _DEFAULT_TOLERANCE,
) -> ASTSCertificate:
    """Certify local ASTS basis evidence without constructing a global matrix."""
    if not isinstance(mesh, TMesh2D):
        raise TypeError("mesh must be TMesh2D.")
    tolerance_ = float(tolerance)
    if not isfinite(tolerance_) or tolerance_ <= 0.0:
        raise ValueError("tolerance must be finite and positive.")
    geometry_valid = _parameter_partition_is_valid(mesh, tolerance_)
    analysis_suitable = _extensions_are_analysis_suitable(mesh.t_junction_extensions)
    knot_compatible = _knot_metadata_is_compatible(mesh, tolerance_)
    dual_compatible = analysis_suitable and knot_compatible
    extraction_valid = True
    extractions: list[ExtractedBernstein] = []
    for cell in mesh.cells:
        u0, u1, v0, v1 = cell.parameter_bounds
        polynomial = all(
            not any(u0 < knot < u1 for knot in anchor.local_knots.u)
            and not any(v0 < knot < v1 for knot in anchor.local_knots.v)
            for anchor in mesh.anchors
            if anchor.local_knots.u[0] < u1
            and anchor.local_knots.u[-1] > u0
            and anchor.local_knots.v[0] < v1
            and anchor.local_knots.v[-1] > v0
        )
        if not polynomial:
            extraction_valid = False
            continue
        extractions.append(_cell_extraction(mesh, cell))
    sample = np.asarray(
        (0.06943184420297371, 0.33000947820757187, 0.6699905217924281, 0.9305681557970262)
    )
    tensor_points = np.asarray([(u, v) for u in sample for v in sample])
    maximum_partition_error = float("inf")
    minimum_basis_value = float("-inf")
    minimum_rank = 0
    minimum_rank_margin = -1
    maximum_condition = float("inf")
    if extraction_valid and len(extractions) == mesh.cell_count:
        partition_errors = []
        minima = []
        ranks = []
        margins = []
        conditions = []
        for extraction in extractions:
            values = np.asarray(extraction.evaluate_reference(tensor_points))
            partition_errors.append(float(np.max(np.abs(np.sum(values, axis=1) - 1.0))))
            minima.append(float(np.min(values)))
            ranks.append(extraction.rank)
            margins.append(extraction.rank - extraction.local_width)
            conditions.append(extraction.condition_number)
        maximum_partition_error = max(partition_errors)
        minimum_basis_value = min(minima)
        minimum_rank = min(ranks)
        minimum_rank_margin = min(margins)
        maximum_condition = max(conditions)
    partition = extraction_valid and maximum_partition_error <= tolerance_
    nonnegative = extraction_valid and minimum_basis_value >= -tolerance_
    full_rank = (
        extraction_valid
        and bool(extractions)
        and all(extraction.rank == extraction.local_width for extraction in extractions)
    )
    unique_local_knots = (
        len({(anchor.local_knots.u, anchor.local_knots.v) for anchor in mesh.anchors})
        == mesh.coefficient_count
    )
    linearly_independent = (
        analysis_suitable and dual_compatible and unique_local_knots and full_rank
    )
    checks = (
        (analysis_suitable, "intersecting T-junction extensions"),
        (dual_compatible, "incompatible local knot vectors"),
        (partition, "partition-of-unity residual"),
        (nonnegative, "negative blending function"),
        (linearly_independent, "local independence evidence"),
        (full_rank, "rank-deficient local extraction"),
        (geometry_valid, "invalid parametric cell complex"),
    )
    failure_reasons = tuple(reason for passed, reason in checks if not passed)
    payload = {
        "kind": "asts2d-certificate",
        "mesh": mesh.mesh_id,
        "checks": [bool(value) for value, _ in checks],
        "metrics": {
            "partition": maximum_partition_error,
            "minimum_basis": minimum_basis_value,
            "minimum_rank": minimum_rank,
            "rank_margin": minimum_rank_margin,
            "maximum_condition": maximum_condition,
        },
        "tolerance": tolerance_,
    }
    return ASTSCertificate(
        mesh.mesh_id,
        analysis_suitable,
        dual_compatible,
        partition,
        nonnegative,
        linearly_independent,
        full_rank,
        geometry_valid,
        maximum_partition_error,
        minimum_basis_value,
        minimum_rank,
        minimum_rank_margin,
        maximum_condition,
        tolerance_,
        failure_reasons,
        canonical_fingerprint(payload),
    )


class ASTSTransferPlan(StrictModule, NonTrainableState):
    """Exact sparse coefficient routes for one certified nested ASTS refinement."""

    source_mesh_id: str = eqx.field(static=True)
    target_mesh_id: str = eqx.field(static=True)
    source_count: int = eqx.field(static=True)
    target_count: int = eqx.field(static=True)
    target_indices: Array
    source_indices: Array
    coefficients: Array
    maximum_reproduction_residual: float = eqx.field(static=True)
    constant_reproduction_residual: float = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        source_mesh_id: str,
        target_mesh_id: str,
        source_count: int,
        target_count: int,
        target_indices: ArrayLike,
        source_indices: ArrayLike,
        coefficients: ArrayLike,
        /,
        *,
        maximum_reproduction_residual: float,
        tolerance: float,
    ):
        targets = np.asarray(target_indices)
        sources = np.asarray(source_indices)
        weights = np.asarray(coefficients, dtype=float)
        if (
            targets.ndim != 1
            or sources.shape != targets.shape
            or weights.shape != targets.shape
        ):
            raise ValueError("Transfer routes must be aligned rank-1 arrays.")
        if not np.issubdtype(targets.dtype, np.integer) or not np.issubdtype(
            sources.dtype, np.integer
        ):
            raise TypeError("Transfer route indices must be integers.")
        source_count_ = int(source_count)
        target_count_ = int(target_count)
        if source_count_ <= 0 or target_count_ <= 0 or targets.size == 0:
            raise ValueError("A transfer plan requires nonempty source and target bases.")
        if np.any(targets < 0) or np.any(targets >= target_count_):
            raise ValueError("A target transfer route is out of range.")
        if np.any(sources < 0) or np.any(sources >= source_count_):
            raise ValueError("A source transfer route is out of range.")
        if not np.all(np.isfinite(weights)) or np.any(weights < -float(tolerance)):
            raise ValueError(
                "Exact ASTS transfer weights must be finite and nonnegative."
            )
        constant = np.zeros((target_count_,), dtype=float)
        np.add.at(constant, targets, weights)
        constant_residual = float(np.max(np.abs(constant - 1.0)))
        reproduction_residual = float(maximum_reproduction_residual)
        if constant_residual > float(tolerance) or reproduction_residual > float(
            tolerance
        ):
            raise ValueError(
                "Refinement routes do not exactly reproduce the nested ASTS basis."
            )
        order = np.lexsort((sources, targets))
        targets = targets[order].astype(np.int32, copy=False)
        sources = sources[order].astype(np.int32, copy=False)
        weights = weights[order]
        self.source_mesh_id = _identifier("source_mesh_id", source_mesh_id)
        self.target_mesh_id = _identifier("target_mesh_id", target_mesh_id)
        self.source_count = source_count_
        self.target_count = target_count_
        self.target_indices = jnp.asarray(targets)
        self.source_indices = jnp.asarray(sources)
        self.coefficients = jnp.asarray(weights)
        self.maximum_reproduction_residual = reproduction_residual
        self.constant_reproduction_residual = constant_residual
        self.plan_id = canonical_fingerprint(
            {
                "kind": "asts2d-exact-transfer-plan",
                "source": self.source_mesh_id,
                "target": self.target_mesh_id,
                "source_count": source_count_,
                "target_count": target_count_,
                "targets": _array_digest(targets),
                "sources": _array_digest(sources),
                "coefficients": _array_digest(weights),
            }
        )

    @property
    def route_count(self) -> int:
        return int(self.coefficients.shape[0])

    def apply(self, source_coefficients: ArrayLike, /) -> Array:
        values = jnp.asarray(source_coefficients)
        if values.shape[0] != self.source_count:
            raise ValueError("Source coefficient leading extent does not match the plan.")
        scale = self.coefficients.reshape((self.route_count,) + (1,) * (values.ndim - 1))
        routed = values[self.source_indices] * scale
        target = jnp.zeros((self.target_count,) + values.shape[1:], dtype=routed.dtype)
        return target.at[self.target_indices].add(routed)

    def transpose(self, target_cotangent: ArrayLike, /) -> Array:
        values = jnp.asarray(target_cotangent)
        if values.shape[0] != self.target_count:
            raise ValueError("Target cotangent leading extent does not match the plan.")
        scale = self.coefficients.reshape((self.route_count,) + (1,) * (values.ndim - 1))
        routed = values[self.target_indices] * scale
        source = jnp.zeros((self.source_count,) + values.shape[1:], dtype=routed.dtype)
        return source.at[self.source_indices].add(routed)


class ASTSRefinement(StrictModule, NonTrainableState):
    """Certified result of conservative full-line analysis-suitable closure."""

    source_mesh: TMesh2D
    refined_mesh: TMesh2D
    marked_cell_ids: tuple[int, ...] = eqx.field(static=True)
    closure_cell_ids: tuple[int, ...] = eqx.field(static=True)
    transfer_plan: ASTSTransferPlan
    certificate: ASTSCertificate
    refinement_id: str = eqx.field(static=True)

    def __init__(
        self,
        source_mesh: TMesh2D,
        refined_mesh: TMesh2D,
        marked_cell_ids: Sequence[int],
        closure_cell_ids: Sequence[int],
        transfer_plan: ASTSTransferPlan,
        certificate: ASTSCertificate,
        /,
    ):
        if transfer_plan.source_mesh_id != source_mesh.mesh_id:
            raise ValueError(
                "Transfer source identity does not match the refinement source."
            )
        if transfer_plan.target_mesh_id != refined_mesh.mesh_id:
            raise ValueError("Transfer target identity does not match the refined mesh.")
        if certificate.mesh_id != refined_mesh.mesh_id or not certificate.passed:
            raise ValueError("A refinement result requires a passing target certificate.")
        self.source_mesh = source_mesh
        self.refined_mesh = refined_mesh
        self.marked_cell_ids = tuple(sorted(int(value) for value in marked_cell_ids))
        self.closure_cell_ids = tuple(sorted(int(value) for value in closure_cell_ids))
        self.transfer_plan = transfer_plan
        self.certificate = certificate
        self.refinement_id = canonical_fingerprint(
            {
                "kind": "asts2d-analysis-suitable-refinement",
                "source": source_mesh.mesh_id,
                "target": refined_mesh.mesh_id,
                "marked": list(self.marked_cell_ids),
                "closure": list(self.closure_cell_ids),
                "transfer": transfer_plan.plan_id,
                "certificate": certificate.certificate_id,
            }
        )


def _knot_key(knots: Sequence[float], /) -> tuple[float, ...]:
    return tuple(round(float(value), 15) for value in knots)


def _refined_univariate_routes(
    local_knots: Sequence[float], target_knots: Sequence[float], /
) -> tuple[dict[int, float], float]:
    local = tuple(float(value) for value in local_knots)
    support_lower, support_upper = local[0], local[-1]
    augmented = (
        (support_lower,) * _BICUBIC_DEGREE + local + (support_upper,) * _BICUBIC_DEGREE
    )
    coefficients = np.zeros((len(augmented) - _BICUBIC_DEGREE - 1,), dtype=float)
    coefficients[_BICUBIC_DEGREE] = 1.0
    spline = BSpline(augmented, coefficients, _BICUBIC_DEGREE, extrapolate=False)
    target_values, target_counts = np.unique(np.asarray(target_knots), return_counts=True)
    for value, target_count in zip(target_values, target_counts, strict=True):
        if value < support_lower or value > support_upper:
            continue
        existing_count = int(np.count_nonzero(np.asarray(spline.t) == value))
        insertions = max(0, int(target_count) - existing_count)
        if insertions:
            spline = spline.insert_knot(float(value), insertions)
    target_windows = {
        _knot_key(target_knots[index : index + _BICUBIC_DEGREE + 2]): index
        for index in range(len(target_knots) - _BICUBIC_DEGREE - 1)
    }
    routes: dict[int, float] = {}
    for index, coefficient in enumerate(np.asarray(spline.c)):
        if abs(coefficient) <= 64.0 * np.finfo(float).eps:
            continue
        window = _knot_key(spline.t[index : index + _BICUBIC_DEGREE + 2])
        if window not in target_windows:
            raise ValueError(
                "Refined local knot vector is absent from the closure basis."
            )
        routes[target_windows[window]] = float(coefficient)
    samples = np.linspace(support_lower, support_upper, 41)[1:-1]
    original = np.asarray(
        [_bspline_value(local, _BICUBIC_DEGREE, float(point)) for point in samples]
    )
    reproduced = np.zeros_like(original)
    for target_index, coefficient in routes.items():
        window = target_knots[target_index : target_index + _BICUBIC_DEGREE + 2]
        reproduced += coefficient * np.asarray(
            [_bspline_value(window, _BICUBIC_DEGREE, float(point)) for point in samples]
        )
    residual = float(np.max(np.abs(original - reproduced)))
    return routes, residual


def _exact_transfer_plan(
    source: TMesh2D,
    target: TMesh2D,
    target_u_knots: Sequence[float],
    target_v_knots: Sequence[float],
    /,
    *,
    tolerance: float,
) -> ASTSTransferPlan:
    target_u_windows = tuple(
        _knot_key(target_u_knots[index : index + _BICUBIC_DEGREE + 2])
        for index in range(len(target_u_knots) - _BICUBIC_DEGREE - 1)
    )
    target_v_windows = tuple(
        _knot_key(target_v_knots[index : index + _BICUBIC_DEGREE + 2])
        for index in range(len(target_v_knots) - _BICUBIC_DEGREE - 1)
    )
    target_by_windows = {
        (
            _knot_key(anchor.local_knots.u),
            _knot_key(anchor.local_knots.v),
        ): anchor.anchor_id
        for anchor in target.anchors
    }
    target_indices = []
    source_indices = []
    weights = []
    maximum_residual = 0.0
    for anchor in source.anchors:
        u_routes, u_residual = _refined_univariate_routes(
            anchor.local_knots.u, target_u_knots
        )
        v_routes, v_residual = _refined_univariate_routes(
            anchor.local_knots.v, target_v_knots
        )
        maximum_residual = max(maximum_residual, u_residual, v_residual)
        for u_index, u_weight in u_routes.items():
            for v_index, v_weight in v_routes.items():
                key = (target_u_windows[u_index], target_v_windows[v_index])
                if key not in target_by_windows:
                    raise ValueError("Closure target lacks a required refined anchor.")
                target_indices.append(target_by_windows[key])
                source_indices.append(anchor.anchor_id)
                weights.append(u_weight * v_weight)
    return ASTSTransferPlan(
        source.mesh_id,
        target.mesh_id,
        source.coefficient_count,
        target.coefficient_count,
        target_indices,
        source_indices,
        weights,
        maximum_reproduction_residual=maximum_residual,
        tolerance=tolerance,
    )


def analysis_suitable_refinement_closure(
    mesh: TMesh2D,
    marked_cell_ids: Sequence[int],
    /,
    *,
    tolerance: float = _DEFAULT_TOLERANCE,
) -> ASTSRefinement:
    """Refine marked cells and close every inserted line across the parameter domain.

    Full-line closure is conservative rather than minimal, but it is deterministic,
    cannot create crossing T-junction extensions, and makes the exact nested transfer
    independently auditable from local knot-insertion routes.
    """
    if not isinstance(mesh, TMesh2D):
        raise TypeError("mesh must be TMesh2D.")
    tolerance_ = float(tolerance)
    if not isfinite(tolerance_) or tolerance_ <= 0.0:
        raise ValueError("tolerance must be finite and positive.")
    source_certificate = certify_asts(mesh, tolerance=tolerance_)
    if not source_certificate.passed:
        reasons = ", ".join(source_certificate.failure_reasons)
        raise ValueError(f"Refinement source is not certified ASTS2D: {reasons}.")
    marked = tuple(sorted({int(value) for value in marked_cell_ids}))
    if not marked:
        raise ValueError("At least one cell must be marked for refinement.")
    available = {cell.cell_id for cell in mesh.cells}
    if not set(marked).issubset(available):
        unknown = sorted(set(marked) - available)
        raise KeyError(f"Unknown marked T-mesh cells: {unknown}.")
    u_breaks, v_breaks = mesh.axis_breaks
    inserted_u = {
        0.5 * (cell.parameter_bounds[0] + cell.parameter_bounds[1])
        for cell in mesh.cells
        if cell.cell_id in marked
    }
    inserted_v = {
        0.5 * (cell.parameter_bounds[2] + cell.parameter_bounds[3])
        for cell in mesh.cells
        if cell.cell_id in marked
    }
    refined_u = tuple(sorted(set(u_breaks) | inserted_u))
    refined_v = tuple(sorted(set(v_breaks) | inserted_v))
    closure = tuple(
        cell.cell_id
        for cell in mesh.cells
        if any(
            cell.parameter_bounds[0] < value < cell.parameter_bounds[1]
            for value in inserted_u
        )
        or any(
            cell.parameter_bounds[2] < value < cell.parameter_bounds[3]
            for value in inserted_v
        )
    )
    refined = TMesh2D.tensor_product(refined_u, refined_v)
    target_certificate = certify_asts(refined, tolerance=tolerance_)
    if not target_certificate.passed:
        reasons = ", ".join(target_certificate.failure_reasons)
        raise RuntimeError(f"Analysis-suitable closure failed certification: {reasons}.")
    u_knots = _open_knot_vector(refined_u)
    v_knots = _open_knot_vector(refined_v)
    transfer = _exact_transfer_plan(
        mesh,
        refined,
        u_knots,
        v_knots,
        tolerance=tolerance_,
    )
    return ASTSRefinement(
        mesh,
        refined,
        marked,
        closure,
        transfer,
        target_certificate,
    )


class ASTSConsumerRequirement(StrictModule, NonTrainableState):
    """Declarative requirements a downstream surface or shell consumer must satisfy."""

    consumer: Literal["surface", "shell"] = eqx.field(static=True)
    intrinsic_dimension: int = eqx.field(static=True)
    ambient_dimensions: tuple[int, ...] = eqx.field(static=True)
    required_parameter_derivative_order: int = eqx.field(static=True)
    requires_positive_rational_weights: bool = eqx.field(static=True)
    requires_regular_geometry_map: bool = eqx.field(static=True)
    requires_consistent_orientation: bool = eqx.field(static=True)
    requires_cross_cell_c1: bool = eqx.field(static=True)
    basis_profile: str = eqx.field(static=True)


ASTS2D_CONSUMER_REQUIREMENTS: Mapping[str, ASTSConsumerRequirement] = {
    "surface": ASTSConsumerRequirement(
        "surface", 2, (2, 3), 1, True, True, True, False, "F7-ASTS2D-bicubic-basis-only"
    ),
    "shell": ASTSConsumerRequirement(
        "shell", 2, (3,), 2, True, True, True, True, "F7-ASTS2D-bicubic-basis-only"
    ),
}


def asts_consumer_requirement(
    consumer: Literal["surface", "shell"], /
) -> ASTSConsumerRequirement:
    """Return neutral requirement metadata; no mechanics are selected or implied."""
    if consumer not in ASTS2D_CONSUMER_REQUIREMENTS:
        raise ValueError("ASTS2D consumer must be 'surface' or 'shell'.")
    return ASTS2D_CONSUMER_REQUIREMENTS[consumer]


def asts_qualification_payload(certificate: ASTSCertificate, /) -> dict[str, object]:
    """Qualification producer hook retained outside the public capability registry."""
    if not isinstance(certificate, ASTSCertificate):
        raise TypeError("certificate must be ASTSCertificate.")
    return certificate.qualification_payload()


__all__ = [
    "ASTS2D_CONSUMER_REQUIREMENTS",
    "ASTSCertificate",
    "ASTSConsumerRequirement",
    "ASTSRefinement",
    "ASTSTransferPlan",
    "ExtractedBernstein",
    "LocalExtractedBernsteinRealization",
    "LocalKnotVector2D",
    "TAnchor2D",
    "TCell2D",
    "TEdge2D",
    "TJunctionExtension2D",
    "TMesh2D",
    "TVertex2D",
    "analysis_suitable_refinement_closure",
    "asts_consumer_requirement",
    "asts_qualification_payload",
    "certify_asts",
]

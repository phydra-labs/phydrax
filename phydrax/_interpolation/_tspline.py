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

from bisect import bisect_left, bisect_right
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from fractions import Fraction
from math import comb, isfinite, lcm
from typing import cast, Literal, overload

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jax import Array, core as jax_core
from jax.typing import ArrayLike

from .._fingerprint import array_tree_fingerprint, canonical_fingerprint
from .._strict import StrictModule
from .._trainable import NonTrainableState
from ..ein import contract
from ..typing import checked
from ._bspline_extraction import _insert_knot_once_exact


_BICUBIC_DEGREE = 3
_DEFAULT_TOLERANCE = 5.0e-11
_DIRECTIONS = ("left", "right", "down", "up")


def _identifier(name: str, value: str, /) -> str:
    identifier = str(value).strip()
    if not identifier:
        raise ValueError(f"{name} must be nonempty.")
    return identifier


def _fraction(value: Fraction | int | float, /) -> Fraction:
    if isinstance(value, Fraction):
        return value
    return Fraction(value)


def _fraction_payload(values: Sequence[Fraction], /) -> list[list[int]]:
    return [[value.numerator, value.denominator] for value in values]


def _coordinate(value: Sequence[Fraction | int | float], /) -> tuple[Fraction, Fraction]:
    if len(value) != 2:
        raise ValueError("A T-mesh vertex coordinate must contain exactly two values.")
    point = (_fraction(value[0]), _fraction(value[1]))
    if not all(isfinite(float(component)) for component in point):
        raise ValueError("T-mesh coordinates must be finite.")
    return point


def _bounds(
    value: Sequence[Fraction | int | float] | Sequence[Sequence[Fraction | int | float]],
    /,
) -> tuple[Fraction, Fraction, Fraction, Fraction]:
    nested = tuple(
        axis
        for axis in value
        if isinstance(axis, Sequence) and not isinstance(axis, (str, bytes))
    )
    if len(value) == 2 and len(nested) == 2:
        first, second = nested
        if len(first) != 2 or len(second) != 2:
            raise ValueError("Nested cell bounds must contain two two-value intervals.")
        result = (
            _fraction(first[0]),
            _fraction(first[1]),
            _fraction(second[0]),
            _fraction(second[1]),
        )
    else:
        if len(value) != 4:
            raise ValueError("Cell bounds must be (u0, u1, v0, v1).")
        flat = cast(Sequence[Fraction | int | float], value)
        flat_values = tuple(_fraction(item) for item in flat)
        result = (flat_values[0], flat_values[1], flat_values[2], flat_values[3])
    if not all(isfinite(float(component)) for component in result):
        raise ValueError("Cell bounds must be finite.")
    if not result[0] < result[1] or not result[2] < result[3]:
        raise ValueError("T-mesh cells must have strictly positive parametric area.")
    return result


def _strict_breaks(
    name: str, values: Sequence[Fraction | int | float], /
) -> tuple[Fraction, ...]:
    breaks = tuple(_fraction(value) for value in values)
    if len(breaks) < 2 or not all(isfinite(float(value)) for value in breaks):
        raise ValueError(f"{name} must contain at least two finite values.")
    if any(right <= left for left, right in zip(breaks[:-1], breaks[1:], strict=True)):
        raise ValueError(f"{name} must be strictly increasing.")
    return breaks


def _knot_tuple(
    name: str, values: Sequence[Fraction | int | float], /
) -> tuple[Fraction, ...]:
    knots = tuple(_fraction(value) for value in values)
    if len(knots) != _BICUBIC_DEGREE + 2:
        raise ValueError(f"{name} must contain exactly five knots for a cubic basis.")
    if not all(isfinite(float(value)) for value in knots):
        raise ValueError(f"{name} must contain finite knots.")
    if any(right < left for left, right in zip(knots[:-1], knots[1:], strict=True)):
        raise ValueError(f"{name} must be nondecreasing.")
    if not knots[0] < knots[-1]:
        raise ValueError(f"{name} must have nonzero support.")
    return knots


def _array_digest(value: object, /) -> dict[str, object]:
    return array_tree_fingerprint(value)


class TVertex2D(StrictModule, NonTrainableState):
    """One immutable vertex of a rectilinear T-mesh."""

    vertex_id: int = eqx.field(static=True)
    parameter: tuple[Fraction, Fraction] = eqx.field(static=True)

    def __init__(
        self, vertex_id: int, parameter: Sequence[Fraction | int | float], /
    ) -> None:
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
    ) -> None:
        identifier = int(edge_id)
        vertices = tuple(vertex_ids)
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
    parameter_bounds: tuple[Fraction, Fraction, Fraction, Fraction] = eqx.field(
        static=True
    )
    vertex_ids: tuple[int, ...] = eqx.field(static=True)
    edge_ids: tuple[int, ...] = eqx.field(static=True)
    level: int = eqx.field(static=True)
    parent_id: int | None = eqx.field(static=True)
    refinement_address: tuple[int, ...] = eqx.field(static=True)

    def __init__(
        self,
        cell_id: int,
        parameter_bounds: Sequence[Fraction | int | float]
        | Sequence[Sequence[Fraction | int | float]],
        /,
        *,
        vertex_ids: Sequence[int] = (),
        edge_ids: Sequence[int] = (),
        level: int = 0,
        parent_id: int | None = None,
        refinement_address: Sequence[int] = (),
    ) -> None:
        identifier = int(cell_id)
        level_ = int(level)
        if identifier < 0 or level_ < 0:
            raise ValueError("T-mesh cell IDs and levels must be nonnegative.")
        self.cell_id = identifier
        self.parameter_bounds = _bounds(parameter_bounds)
        self.vertex_ids = tuple(vertex_ids)
        self.edge_ids = tuple(edge_ids)
        self.level = level_
        parent = None if parent_id is None else int(parent_id)
        address = tuple(int(value) for value in refinement_address)
        if parent is not None and parent < 0:
            raise ValueError("T-mesh cell parent IDs must be nonnegative.")
        if any(value < 0 for value in address):
            raise ValueError("T-mesh refinement addresses must be nonnegative.")
        if (parent is None) != (not address):
            raise ValueError(
                "Refinement parent IDs and addresses must be supplied together."
            )
        self.parent_id = parent
        self.refinement_address = address

    @property
    def area(self) -> Fraction:
        u0, u1, v0, v1 = self.parameter_bounds
        return (u1 - u0) * (v1 - v0)


class LocalKnotVector2D(StrictModule, NonTrainableState):
    """The two five-knot vectors defining one bicubic blending function."""

    u: tuple[Fraction, ...] = eqx.field(static=True)
    v: tuple[Fraction, ...] = eqx.field(static=True)

    def __init__(
        self,
        u: Sequence[Fraction | int | float],
        v: Sequence[Fraction | int | float],
        /,
    ) -> None:
        self.u = _knot_tuple("u knots", u)
        self.v = _knot_tuple("v knots", v)

    @property
    def support(self) -> tuple[Fraction, Fraction, Fraction, Fraction]:
        return (self.u[0], self.u[-1], self.v[0], self.v[-1])

    @property
    def greville_parameter(self) -> tuple[Fraction, Fraction]:
        return (
            sum(self.u[1:4], Fraction(0)) / Fraction(_BICUBIC_DEGREE),
            sum(self.v[1:4], Fraction(0)) / Fraction(_BICUBIC_DEGREE),
        )


class TAnchor2D(StrictModule, NonTrainableState):
    """One bicubic T-spline anchor and its complete local knot evidence."""

    anchor_id: int = eqx.field(static=True)
    parameter: tuple[Fraction, Fraction] = eqx.field(static=True)
    local_knots: LocalKnotVector2D

    @checked
    def __init__(
        self,
        anchor_id: int,
        parameter: Sequence[Fraction | int | float],
        local_knots: LocalKnotVector2D,
        /,
    ) -> None:
        identifier = int(anchor_id)
        if identifier < 0 or identifier > np.iinfo(np.int64).max:
            raise OverflowError("anchor_id must fit the int64 identity representation.")
        self.anchor_id = identifier
        self.parameter = _coordinate(parameter)
        self.local_knots = local_knots


class TJunctionExtension2D(StrictModule, NonTrainableState):
    """Odd-degree T-junction extension: one edge span back, two face spans forward."""

    junction_vertex_id: int = eqx.field(static=True)
    axis: Literal["u", "v"] = eqx.field(static=True)
    missing_direction: Literal["left", "right", "down", "up"] = eqx.field(static=True)
    segment: tuple[Fraction, Fraction, Fraction, Fraction] = eqx.field(static=True)
    backward_crossings: int = eqx.field(static=True)
    forward_crossings: int = eqx.field(static=True)

    def __init__(
        self,
        junction_vertex_id: int,
        axis: Literal["u", "v"],
        missing_direction: Literal["left", "right", "down", "up"],
        segment: Sequence[Fraction | int | float],
        /,
        *,
        backward_crossings: int,
        forward_crossings: int,
    ) -> None:
        identifier = int(junction_vertex_id)
        segment_ = tuple(_fraction(value) for value in segment)
        if identifier < 0 or axis not in ("u", "v"):
            raise ValueError("Invalid T-junction extension identity or axis.")
        if missing_direction not in _DIRECTIONS:
            raise ValueError("Invalid T-junction missing direction.")
        if len(segment_) != 4 or not all(isfinite(float(value)) for value in segment_):
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
    parameter_domain: tuple[Fraction, Fraction, Fraction, Fraction] = eqx.field(
        static=True
    )
    cell_lineage: tuple[tuple[int, tuple[int, ...]], ...] = eqx.field(static=True)
    reserved_cell_ids: tuple[int, ...] = eqx.field(static=True)
    reserved_anchor_ids: tuple[int, ...] = eqx.field(static=True)
    mesh_id: str = eqx.field(static=True)

    def __init__(
        self,
        cells: Sequence[
            TCell2D
            | Sequence[Fraction | int | float]
            | Sequence[Sequence[Fraction | int | float]]
        ],
        anchors: Sequence[TAnchor2D] | None = None,
        /,
        *,
        degree: Sequence[int] = (_BICUBIC_DEGREE, _BICUBIC_DEGREE),
        mesh_id: str | None = None,
        cell_lineage: Sequence[tuple[int, Sequence[int]]] = (),
        reserved_cell_ids: Sequence[int] = (),
        reserved_anchor_ids: Sequence[int] = (),
        proprietary_format: str | None = None,
        extraordinary: bool = False,
    ) -> None:
        degrees = tuple(degree)
        if degrees != (_BICUBIC_DEGREE, _BICUBIC_DEGREE):
            raise NotImplementedError("F7 supports bicubic ASTS2D bases only.")
        degree_u, degree_v = degrees
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
        cell_bounds: list[tuple[Fraction, Fraction, Fraction, Fraction]] = []
        cell_levels: list[int] = []
        cell_parents: list[int | None] = []
        cell_addresses: list[tuple[int, ...]] = []
        for index, cell in enumerate(raw_cells):
            if isinstance(cell, TCell2D):
                cell_ids.append(cell.cell_id)
                cell_bounds.append(cell.parameter_bounds)
                cell_levels.append(cell.level)
                cell_parents.append(cell.parent_id)
                cell_addresses.append(cell.refinement_address)
            else:
                cell_ids.append(index)
                cell_bounds.append(_bounds(cell))
                cell_levels.append(0)
                cell_parents.append(None)
                cell_addresses.append(())
        if len(set(cell_ids)) != len(cell_ids):
            raise ValueError("T-mesh cell IDs must be unique.")
        anchors_ = None if anchors is None else tuple(anchors)
        vertices, edges, cells_ = _build_topology(
            cell_ids, cell_bounds, cell_levels, cell_parents, cell_addresses
        )
        domain = (
            min(bounds[0] for bounds in cell_bounds),
            max(bounds[1] for bounds in cell_bounds),
            min(bounds[2] for bounds in cell_bounds),
            max(bounds[3] for bounds in cell_bounds),
        )
        if anchors_ is None:
            anchors_ = _derive_anchors(vertices, edges, domain)
        if not anchors_ or not all(isinstance(anchor, TAnchor2D) for anchor in anchors_):
            raise TypeError("anchors must be a nonempty sequence of TAnchor2D values.")
        anchor_ids = tuple(anchor.anchor_id for anchor in anchors_)
        if len(set(anchor_ids)) != len(anchor_ids):
            raise ValueError("T-spline anchor IDs must be unique.")
        if not _anchors_match_topology(anchors_, vertices, edges, domain):
            raise ValueError(
                "T-spline local knot windows do not follow the T-mesh topology."
            )
        lineage_events = tuple(
            sorted(
                (
                    int(parent),
                    tuple(int(child) for child in children),
                )
                for parent, children in cell_lineage
            )
        )
        if len({parent for parent, _ in lineage_events}) != len(lineage_events) or any(
            parent < 0
            or not children
            or len(set(children)) != len(children)
            or any(child < 0 for child in children)
            for parent, children in lineage_events
        ):
            raise ValueError("T-mesh cell lineage must contain unique nonnegative IDs.")
        lineage_children = tuple(
            child for _, children in lineage_events for child in children
        )
        if len(set(lineage_children)) != len(lineage_children) or any(
            parent in children for parent, children in lineage_events
        ):
            raise ValueError("A T-mesh lineage child must have one unique parent.")
        lineage_ids = {
            identifier
            for parent, children in lineage_events
            for identifier in (parent, *children)
        }
        reserved_cells = tuple(
            sorted(
                {int(value) for value in reserved_cell_ids}
                | set(cell_ids)
                | {parent for parent in cell_parents if parent is not None}
                | lineage_ids
            )
        )
        reserved_anchors = tuple(
            sorted({int(value) for value in reserved_anchor_ids} | set(anchor_ids))
        )
        maximum_cell_id = np.iinfo(np.int64).max
        if any(value < 0 or value > maximum_cell_id for value in reserved_cells):
            raise OverflowError(
                "T-mesh cell IDs exceed the int64 identity representation."
            )
        if any(value < 0 or value > maximum_cell_id for value in reserved_anchors):
            raise OverflowError(
                "T-spline anchor IDs exceed the int64 identity representation."
            )
        extensions = _t_junction_extensions(vertices, edges, domain)
        self.vertices = vertices
        self.edges = edges
        self.cells = cells_
        self.anchors = tuple(sorted(anchors_, key=lambda anchor: anchor.anchor_id))
        self.t_junction_extensions = extensions
        self.cell_lineage = lineage_events
        self.reserved_cell_ids = reserved_cells
        self.reserved_anchor_ids = reserved_anchors
        self.degree = (degree_u, degree_v)
        self.parameter_domain = domain
        payload = {
            "kind": "asts2d-bicubic-tmesh",
            "degree": list(degrees),
            "cells": [
                {
                    "id": cell.cell_id,
                    "bounds": _fraction_payload(cell.parameter_bounds),
                    "level": cell.level,
                    "parent_id": cell.parent_id,
                    "refinement_address": list(cell.refinement_address),
                }
                for cell in cells_
            ],
            "anchors": [
                {
                    "id": anchor.anchor_id,
                    "parameter": _fraction_payload(anchor.parameter),
                    "u": _fraction_payload(anchor.local_knots.u),
                    "v": _fraction_payload(anchor.local_knots.v),
                }
                for anchor in self.anchors
            ],
            "cell_lineage": [
                [parent, list(children)] for parent, children in self.cell_lineage
            ],
            "reserved_cell_ids": list(self.reserved_cell_ids),
            "reserved_anchor_ids": list(self.reserved_anchor_ids),
        }
        self.mesh_id = (
            canonical_fingerprint(payload)
            if mesh_id is None
            else _identifier("mesh_id", mesh_id)
        )

    @classmethod
    def tensor_product(
        cls,
        u_breaks: Sequence[Fraction | int | float],
        v_breaks: Sequence[Fraction | int | float],
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
        cells: Sequence[
            TCell2D
            | Sequence[Fraction | int | float]
            | Sequence[Sequence[Fraction | int | float]]
        ],
        anchors: Sequence[TAnchor2D] | None = None,
        /,
        *,
        mesh_id: str | None = None,
        cell_lineage: Sequence[tuple[int, Sequence[int]]] = (),
        reserved_cell_ids: Sequence[int] = (),
        reserved_anchor_ids: Sequence[int] = (),
    ) -> TMesh2D:
        """Construct a native ASTS mesh from cells and optional exact anchors."""
        return cls(
            cells,
            anchors,
            mesh_id=mesh_id,
            cell_lineage=cell_lineage,
            reserved_cell_ids=reserved_cell_ids,
            reserved_anchor_ids=reserved_anchor_ids,
        )

    @property
    def cell_count(self) -> int:
        return len(self.cells)

    @property
    def coefficient_count(self) -> int:
        return len(self.anchors)

    def anchor_row(self, anchor_id: int, /) -> int:
        identifier = int(anchor_id)
        for row, anchor in enumerate(self.anchors):
            if anchor.anchor_id == identifier:
                return row
        raise KeyError(f"Unknown T-spline anchor ID {identifier}.")

    @property
    def t_junction_count(self) -> int:
        return len(self.t_junction_extensions)

    @property
    def axis_breaks(self) -> tuple[tuple[Fraction, ...], tuple[Fraction, ...]]:
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
        self,
        *,
        tolerance: float = _DEFAULT_TOLERANCE,
        certificate: ASTSCertificate | None = None,
    ) -> LocalExtractedBernsteinRealization:
        tolerance_ = float(tolerance)
        if not isfinite(tolerance_) or tolerance_ <= 0.0:
            raise ValueError("tolerance must be finite and positive.")
        if certificate is None:
            certificate_ = certify_asts(self, tolerance=tolerance_)
        elif (
            not isinstance(certificate, ASTSCertificate)
            or certificate.mesh_id != self.mesh_id
            or certificate.tolerance != tolerance_
        ):
            raise ValueError(
                "A retained ASTS certificate must match this mesh and tolerance."
            )
        else:
            certificate_ = _validated_asts_certificate(self, certificate)
        if not certificate_.passed:
            reasons = ", ".join(certificate_.failure_reasons)
            raise ValueError(f"T-mesh is not a certified ASTS2D basis: {reasons}.")
        return _local_realization(self, certificate_)

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


def _open_knot_vector(
    breaks: Sequence[Fraction | int | float], /
) -> tuple[Fraction, ...]:
    values = tuple(_fraction(value) for value in breaks)
    return (
        (values[0],) * (_BICUBIC_DEGREE + 1)
        + values[1:-1]
        + (values[-1],) * (_BICUBIC_DEGREE + 1)
    )


def _build_topology(
    cell_ids: Sequence[int],
    bounds: Sequence[tuple[Fraction, Fraction, Fraction, Fraction]],
    levels: Sequence[int],
    parent_ids: Sequence[int | None],
    refinement_addresses: Sequence[tuple[int, ...]],
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
            for first, second in zip(on_side[:-1], on_side[1:], strict=True):
                p0 = (first, fixed) if axis == "u" else (fixed, first)
                p1 = (second, fixed) if axis == "u" else (fixed, second)
                low, high = sorted((vertex_by_coordinate[p0], vertex_by_coordinate[p1]))
                key = (low, high)
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
    for (
        cell_id,
        cell_bounds,
        level,
        parent_id,
        refinement_address,
    ) in zip(cell_ids, bounds, levels, parent_ids, refinement_addresses, strict=True):
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
                parent_id=parent_id,
                refinement_address=refinement_address,
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
    domain: tuple[Fraction, Fraction, Fraction, Fraction],
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


def _line_knot_vector(
    axis: Literal["u", "v"],
    fixed_coordinate: Fraction,
    vertices: Sequence[TVertex2D],
    edges: Sequence[TEdge2D],
    domain: tuple[Fraction, Fraction, Fraction, Fraction],
    /,
) -> tuple[Fraction, ...]:
    if axis == "u":
        crossings = {
            vertex.parameter[0]
            for vertex in vertices
            if vertex.parameter[1] == fixed_coordinate
        }
        crossings.update(
            vertices[edge.vertex_ids[0]].parameter[0]
            for edge in edges
            if edge.axis == "v"
            and min(
                vertices[edge.vertex_ids[0]].parameter[1],
                vertices[edge.vertex_ids[1]].parameter[1],
            )
            <= fixed_coordinate
            <= max(
                vertices[edge.vertex_ids[0]].parameter[1],
                vertices[edge.vertex_ids[1]].parameter[1],
            )
        )
        lower, upper = domain[:2]
    else:
        crossings = {
            vertex.parameter[1]
            for vertex in vertices
            if vertex.parameter[0] == fixed_coordinate
        }
        crossings.update(
            vertices[edge.vertex_ids[0]].parameter[1]
            for edge in edges
            if edge.axis == "u"
            and min(
                vertices[edge.vertex_ids[0]].parameter[0],
                vertices[edge.vertex_ids[1]].parameter[0],
            )
            <= fixed_coordinate
            <= max(
                vertices[edge.vertex_ids[0]].parameter[0],
                vertices[edge.vertex_ids[1]].parameter[0],
            )
        )
        lower, upper = domain[2:]
    return _open_knot_vector(tuple(sorted(crossings | {lower, upper})))


def _line_knot_windows(
    axis: Literal["u", "v"],
    fixed_coordinate: Fraction,
    center_coordinate: Fraction,
    vertices: Sequence[TVertex2D],
    edges: Sequence[TEdge2D],
    domain: tuple[Fraction, Fraction, Fraction, Fraction],
    /,
) -> tuple[tuple[Fraction, ...], ...]:
    knots = _line_knot_vector(axis, fixed_coordinate, vertices, edges, domain)
    return tuple(
        tuple(knots[index : index + _BICUBIC_DEGREE + 2])
        for index in range(len(knots) - _BICUBIC_DEGREE - 1)
        if knots[index + 2] == center_coordinate
    )


def _derive_anchors(
    vertices: Sequence[TVertex2D],
    edges: Sequence[TEdge2D],
    domain: tuple[Fraction, Fraction, Fraction, Fraction],
    /,
) -> tuple[TAnchor2D, ...]:
    addresses: set[
        tuple[
            Fraction,
            Fraction,
            tuple[Fraction, ...],
            tuple[Fraction, ...],
        ]
    ] = set()
    for vertex in vertices:
        u, v = vertex.parameter
        u_windows = _line_knot_windows("u", v, u, vertices, edges, domain)
        v_windows = _line_knot_windows("v", u, v, vertices, edges, domain)
        addresses.update(
            (u, v, u_knots, v_knots) for u_knots in u_windows for v_knots in v_windows
        )
    ordered = sorted(addresses)
    return tuple(
        TAnchor2D(index, (u, v), LocalKnotVector2D(u_knots, v_knots))
        for index, (u, v, u_knots, v_knots) in enumerate(ordered)
    )


def _anchors_match_topology(
    anchors: Sequence[TAnchor2D],
    vertices: Sequence[TVertex2D],
    edges: Sequence[TEdge2D],
    domain: tuple[Fraction, Fraction, Fraction, Fraction],
    /,
) -> bool:
    addresses: set[
        tuple[Fraction, Fraction, tuple[Fraction, ...], tuple[Fraction, ...]]
    ] = set()
    vertex_parameters = {vertex.parameter for vertex in vertices}
    for anchor in anchors:
        u, v = anchor.parameter
        if (u, v) not in vertex_parameters:
            return False
        if anchor.local_knots.u not in _line_knot_windows(
            "u", v, u, vertices, edges, domain
        ):
            return False
        if anchor.local_knots.v not in _line_knot_windows(
            "v", u, v, vertices, edges, domain
        ):
            return False
        address = (u, v, anchor.local_knots.u, anchor.local_knots.v)
        if address in addresses:
            return False
        addresses.add(address)
    return True


def _extensions_are_analysis_suitable(
    extensions: Sequence[TJunctionExtension2D], /
) -> bool:
    """Require closed T-junction extension segments to be pairwise disjoint."""
    horizontal = tuple(extension for extension in extensions if extension.axis == "u")
    vertical = tuple(extension for extension in extensions if extension.axis == "v")
    for first in horizontal:
        hu0, hu1, hv, _ = first.segment
        for second in vertical:
            vu, _, vv0, vv1 = second.segment
            if hu0 <= vu <= hu1 and vv0 <= hv <= vv1:
                return False
    return True


def _parameter_partition_is_valid(mesh: TMesh2D, /) -> bool:
    cells = mesh.cells
    u0, u1, v0, v1 = mesh.parameter_domain
    domain_area = (u1 - u0) * (v1 - v0)
    if sum((cell.area for cell in cells), Fraction(0)) != domain_area:
        return False
    for index, first in enumerate(cells):
        a0, a1, b0, b1 = first.parameter_bounds
        for second in cells[index + 1 :]:
            c0, c1, d0, d1 = second.parameter_bounds
            if min(a1, c1) > max(a0, c0) and min(b1, d1) > max(b0, d0):
                return False
    return all(len(edge.cell_ids) in (1, 2) for edge in mesh.edges)


def _interior_multiplicity_is_supported(
    knots: Sequence[Fraction],
    domain: tuple[Fraction, Fraction],
    /,
) -> bool:
    lower, upper = domain
    multiplicities: dict[Fraction, int] = {}
    for knot in knots:
        multiplicities[knot] = multiplicities.get(knot, 0) + 1
    return all(
        count == 1 or knot in (lower, upper) for knot, count in multiplicities.items()
    )


def _knot_metadata_is_compatible(mesh: TMesh2D, /) -> bool:
    u_breaks, v_breaks = mesh.axis_breaks
    u_set = set(u_breaks)
    v_set = set(v_breaks)
    u_domain = mesh.parameter_domain[:2]
    v_domain = mesh.parameter_domain[2:]
    signatures: set[tuple[tuple[Fraction, ...], tuple[Fraction, ...]]] = set()
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
        if anchor.parameter != (knots.u[2], knots.v[2]):
            return False
        signature = (knots.u, knots.v)
        if signature in signatures:
            return False
        signatures.add(signature)
    return _anchors_match_topology(
        mesh.anchors, mesh.vertices, mesh.edges, mesh.parameter_domain
    )


def _window_index(knots: Sequence[Fraction], window: Sequence[Fraction], /) -> int:
    values = tuple(knots)
    local = tuple(window)
    matches = tuple(
        index
        for index in range(len(values) - _BICUBIC_DEGREE)
        if values[index : index + _BICUBIC_DEGREE + 2] == local
    )
    if len(matches) != 1:
        raise ValueError("A local T-spline knot window must identify one row.")
    return matches[0]


def _bernstein_cubic(points: ArrayLike, derivative: int = 0, /) -> Array:
    t = jnp.asarray(points, dtype=jnp.float64)
    if derivative == 0:
        return jnp.stack(
            ((1.0 - t) ** 3, 3.0 * t * (1.0 - t) ** 2, 3.0 * t**2 * (1.0 - t), t**3),
            axis=-1,
        )
    if derivative == 1:
        return jnp.stack(
            (
                -3.0 * (1.0 - t) ** 2,
                3.0 * (1.0 - t) * (1.0 - 3.0 * t),
                3.0 * t * (2.0 - 3.0 * t),
                3.0 * t**2,
            ),
            axis=-1,
        )
    if derivative == 2:
        return jnp.stack(
            (6.0 * (1.0 - t), 18.0 * t - 12.0, 6.0 - 18.0 * t, 6.0 * t),
            axis=-1,
        )
    if derivative == 3:
        return jnp.broadcast_to(
            jnp.asarray((-6.0, 18.0, -18.0, 6.0), dtype=jnp.float64),
            t.shape + (4,),
        )
    if derivative > 3:
        return jnp.zeros(t.shape + (4,), dtype=jnp.float64)
    raise ValueError("Bernstein derivative orders must be nonnegative.")


def _univariate_extraction(
    knots: Sequence[Fraction],
    basis_index: int,
    interval: tuple[Fraction, Fraction],
    /,
) -> tuple[Fraction, Fraction, Fraction, Fraction]:
    knot_vector = tuple(knots)
    lower, upper = interval
    if basis_index < 0 or basis_index >= len(knot_vector) - _BICUBIC_DEGREE - 1:
        raise IndexError("The local B-spline index lies outside its knot vector.")
    if (
        knot_vector[basis_index] >= upper
        or knot_vector[basis_index + _BICUBIC_DEGREE + 1] <= lower
    ):
        return Fraction(0), Fraction(0), Fraction(0), Fraction(0)
    if any(lower < knot < upper for knot in knot_vector):
        raise ValueError("A local knot line cuts the interior of one extraction cell.")
    count = len(knot_vector) - _BICUBIC_DEGREE - 1
    coefficients = tuple(
        tuple(Fraction(int(row == column)) for column in range(count))
        for row in range(count)
    )
    for endpoint in (lower, upper):
        while knot_vector.count(endpoint) < _BICUBIC_DEGREE + 1:
            knot_vector, coefficients = _insert_knot_once_exact(
                knot_vector, coefficients, _BICUBIC_DEGREE, endpoint
            )
    span = min(
        max(
            bisect_right(knot_vector, (lower + upper) / 2) - 1,
            _BICUBIC_DEGREE,
        ),
        len(knot_vector) - _BICUBIC_DEGREE - 2,
    )
    first = span - _BICUBIC_DEGREE
    extracted = tuple(coefficients[row][basis_index] for row in range(first, span + 1))
    if len(extracted) != 4:
        raise RuntimeError("Cubic B-spline extraction must return four coefficients.")
    return cast(
        tuple[Fraction, Fraction, Fraction, Fraction],
        (extracted[0], extracted[1], extracted[2], extracted[3]),
    )


class ExtractedBernstein(StrictModule, NonTrainableState):
    """Exact cell-local extraction with one prepared binary64 realization."""

    cell_id: int = eqx.field(static=True)
    piece_id: int = eqx.field(static=True)
    parameter_bounds: tuple[Fraction, Fraction, Fraction, Fraction] = eqx.field(
        static=True
    )
    anchor_ids: tuple[int, ...] = eqx.field(static=True)
    coefficient_row_ids: tuple[int, ...] = eqx.field(static=True)
    exact_extraction_operator: tuple[tuple[Fraction, ...], ...] = eqx.field(static=True)
    coefficient_rows: Array
    extraction_operator: Array
    maximum_conversion_error: float = eqx.field(static=True)
    rank: int = eqx.field(static=True)
    extraction_id: str = eqx.field(static=True)

    def __init__(
        self,
        cell_id: int,
        parameter_bounds: Sequence[Fraction | int | float],
        anchor_ids: Sequence[int],
        coefficient_rows: Sequence[int],
        exact_extraction_operator: Sequence[Sequence[Fraction]],
        /,
        *,
        piece_id: int | None = None,
    ) -> None:
        from ..linalg._exact import factor_exact_sparse

        cell_id_ = int(cell_id)
        piece_id_ = cell_id_ if piece_id is None else int(piece_id)
        if cell_id_ < 0 or piece_id_ < 0:
            raise ValueError("Extraction cell and piece IDs must be nonnegative.")
        bounds = _bounds(parameter_bounds)
        anchors = tuple(int(value) for value in anchor_ids)
        rows = tuple(int(value) for value in coefficient_rows)
        exact = tuple(
            tuple(_fraction(value) for value in row) for row in exact_extraction_operator
        )
        if (
            len(anchors) != len(rows)
            or len(set(anchors)) != len(anchors)
            or len(set(rows)) != len(rows)
            or any(row < 0 for row in rows)
            or len(exact) != len(rows)
            or any(len(row) != 16 for row in exact)
        ):
            raise ValueError(
                "Cell extraction IDs, rows, and exact coefficients must align."
            )
        operator = np.asarray(
            [[float(value) for value in row] for row in exact], dtype=np.float64
        )
        if not np.all(np.isfinite(operator)):
            raise ValueError(
                "Exact extraction coefficients have no finite binary64 realization."
            )
        if any(
            value != 0 and operator[row, column] == 0.0
            for row, coefficients in enumerate(exact)
            for column, value in enumerate(coefficients)
        ):
            raise ValueError(
                "A nonzero exact extraction coefficient underflows binary64."
            )
        exact_conversion_errors = tuple(
            abs(value - Fraction.from_float(float(operator[row, column])))
            for row, coefficients in enumerate(exact)
            for column, value in enumerate(coefficients)
        )
        maximum_conversion_error = max(exact_conversion_errors, default=Fraction(0))
        conversion_error = float(maximum_conversion_error)
        if Fraction.from_float(conversion_error) < maximum_conversion_error:
            conversion_error = float(np.nextafter(conversion_error, np.inf))
        rank = factor_exact_sparse(
            [
                {column: value for column, value in enumerate(coefficients) if value}
                for coefficients in exact
            ],
            16,
        ).evidence.rank
        self.cell_id = cell_id_
        self.piece_id = int(piece_id_)
        self.parameter_bounds = bounds
        self.anchor_ids = anchors
        self.exact_extraction_operator = exact
        self.coefficient_row_ids = rows
        self.coefficient_rows = jnp.asarray(rows, dtype=jnp.int32)
        self.extraction_operator = jnp.asarray(operator)
        self.maximum_conversion_error = conversion_error
        self.rank = rank
        self.extraction_id = canonical_fingerprint(
            {
                "kind": "asts2d-local-extraction",
                "piece": self.piece_id,
                "cell": self.cell_id,
                "bounds": _fraction_payload(bounds),
                "anchor_ids": list(anchors),
                "coefficient_rows": list(rows),
                "exact_operator": [
                    _fraction_payload(coefficients) for coefficients in exact
                ],
                "numeric_operator": _array_digest(operator),
                "maximum_conversion_error": conversion_error,
            }
        )

    @property
    def local_width(self) -> int:
        return len(self.anchor_ids)

    @property
    def condition_number(self) -> float:
        """Spectral condition number of the binary64 operator, a diagnostic only.

        LAPACK and BLAS kernels set its last bits, so it varies across platforms
        and is never part of an extraction, realization, or certificate identity.
        """
        singular_values = np.linalg.svd(
            np.asarray(self.extraction_operator), compute_uv=False
        )
        if singular_values.size and singular_values[-1] > 0.0:
            return float(singular_values[0] / singular_values[-1])
        return float("inf")

    def evaluate_reference(
        self,
        reference_points: ArrayLike,
        /,
        *,
        derivative: tuple[int, int] = (0, 0),
    ) -> Array:
        points = jnp.asarray(reference_points, dtype=jnp.float64)
        if points.ndim != 2 or points.shape[1] != 2:
            raise ValueError("reference_points must have shape (point, 2).")
        points = eqx.error_if(
            points,
            jnp.any(~jnp.isfinite(points)) | jnp.any((points < 0.0) | (points > 1.0)),
            "Reference points must be finite and inside the unit cell.",
        )
        du, dv = derivative
        if du < 0 or dv < 0 or du > 3 or dv > 3:
            raise ValueError("Derivative orders must lie between zero and degree three.")
        bu = _bernstein_cubic(points[:, 0], du)
        bv = _bernstein_cubic(points[:, 1], dv)
        tensor = (bu[:, :, None] * bv[:, None, :]).reshape((points.shape[0], 16))
        return tensor @ self.extraction_operator.T

    def evaluate(
        self,
        parameter_points: ArrayLike,
        /,
        *,
        derivative: tuple[int, int] = (0, 0),
    ) -> Array:
        points = jnp.asarray(parameter_points, dtype=jnp.float64)
        if points.ndim != 2 or points.shape[1] != 2:
            raise ValueError("parameter_points must have shape (point, 2).")
        u0, u1, v0, v1 = (float(value) for value in self.parameter_bounds)
        points = eqx.error_if(
            points,
            jnp.any(~jnp.isfinite(points))
            | jnp.any(
                (points[:, 0] < u0)
                | (points[:, 0] > u1)
                | (points[:, 1] < v0)
                | (points[:, 1] > v1)
            ),
            "Parameter points must be finite and lie inside the extraction cell.",
        )
        reference = jnp.stack(
            ((points[:, 0] - u0) / (u1 - u0), (points[:, 1] - v0) / (v1 - v0)),
            axis=-1,
        )
        du, dv = derivative
        scale = (u1 - u0) ** (-du) * (v1 - v0) ** (-dv)
        return scale * self.evaluate_reference(reference, derivative=derivative)


class LocalExtractedBernsteinRealization(StrictModule, NonTrainableState):
    """Fixed-route Bernstein pieces over a separate control T-mesh topology."""

    mesh: TMesh2D
    extractions: tuple[ExtractedBernstein, ...]
    basis_certificate: ASTSCertificate = eqx.field(static=True)
    cell_ids: Array
    cell_bounds: Array
    cell_coefficient_rows: Array
    cell_local_widths: Array
    cell_support_rows: tuple[tuple[int, ...], ...] = eqx.field(static=True)
    piece_ids: Array
    piece_cell_ids: Array
    piece_cell_rows: Array
    piece_bounds: Array
    piece_coefficient_rows: Array
    piece_extraction_operators: Array
    piece_local_widths: Array
    piece_ranks: Array
    realization_id: str = eqx.field(static=True)
    basis_id: str = eqx.field(static=True)
    global_coefficient_count: int = eqx.field(static=True)
    local_width: int = eqx.field(static=True)

    @checked
    def __init__(
        self,
        mesh: TMesh2D,
        extractions: Sequence[ExtractedBernstein],
        basis_certificate: ASTSCertificate,
        /,
    ) -> None:
        if (
            not isinstance(basis_certificate, ASTSCertificate)
            or basis_certificate.mesh_id != mesh.mesh_id
            or not basis_certificate.passed
        ):
            raise ValueError(
                "A realization requires passing ASTS evidence for its exact mesh."
            )
        expected_pieces = _extraction_pieces(mesh)
        extractions_ = tuple(sorted(extractions, key=lambda item: item.piece_id))
        expected_signature = tuple(
            (piece.piece_id, piece.cell_id, piece.parameter_bounds)
            for piece in expected_pieces
        )
        actual_signature = tuple(
            (item.piece_id, item.cell_id, item.parameter_bounds) for item in extractions_
        )
        if actual_signature != expected_signature:
            raise ValueError(
                "Exact Bernstein pieces must cover the derived extraction partition."
            )
        cells = tuple(mesh.cells)
        cell_rows_by_id = {cell.cell_id: row for row, cell in enumerate(cells)}
        cell_support_rows = tuple(
            tuple(
                sorted(
                    {
                        coefficient_row
                        for extraction in extractions_
                        if extraction.cell_id == cell.cell_id
                        for coefficient_row in extraction.coefficient_row_ids
                    }
                )
            )
            for cell in cells
        )
        cell_count = len(cells)
        piece_count = len(extractions_)
        local_width = max(len(rows) for rows in cell_support_rows)
        piece_width = max(item.local_width for item in extractions_)
        cell_rows = np.zeros((cell_count, local_width), dtype=np.int32)
        cell_local_widths = np.asarray(
            [len(rows) for rows in cell_support_rows], dtype=np.int32
        )
        for cell_row, rows in enumerate(cell_support_rows):
            cell_rows[cell_row, : len(rows)] = np.asarray(rows, dtype=np.int32)
        piece_rows = np.zeros((piece_count, piece_width), dtype=np.int32)
        piece_operators = np.zeros((piece_count, piece_width, 16), dtype=np.float64)
        piece_parent_rows = np.empty((piece_count,), dtype=np.int32)
        for piece_row, extraction in enumerate(extractions_):
            width = extraction.local_width
            piece_rows[piece_row, :width] = np.asarray(extraction.coefficient_rows)
            piece_operators[piece_row, :width] = np.asarray(
                extraction.extraction_operator
            )
            piece_parent_rows[piece_row] = cell_rows_by_id[extraction.cell_id]
        cell_bounds = np.asarray(
            [[float(value) for value in cell.parameter_bounds] for cell in cells],
            dtype=np.float64,
        )
        piece_bounds = np.asarray(
            [
                [float(value) for value in extraction.parameter_bounds]
                for extraction in extractions_
            ],
            dtype=np.float64,
        )
        self.mesh = mesh
        self.extractions = extractions_
        self.basis_certificate = basis_certificate
        self.cell_ids = jnp.asarray([cell.cell_id for cell in cells], dtype=jnp.int64)
        self.cell_bounds = jnp.asarray(cell_bounds)
        self.cell_coefficient_rows = jnp.asarray(cell_rows)
        self.cell_local_widths = jnp.asarray(cell_local_widths)
        self.cell_support_rows = cell_support_rows
        self.piece_ids = jnp.asarray(
            [item.piece_id for item in extractions_], dtype=jnp.int64
        )
        self.piece_cell_ids = jnp.asarray(
            [item.cell_id for item in extractions_], dtype=jnp.int64
        )
        self.piece_cell_rows = jnp.asarray(piece_parent_rows)
        self.piece_bounds = jnp.asarray(piece_bounds)
        self.piece_coefficient_rows = jnp.asarray(piece_rows)
        self.piece_extraction_operators = jnp.asarray(piece_operators)
        self.piece_local_widths = jnp.asarray(
            [item.local_width for item in extractions_], dtype=jnp.int32
        )
        self.piece_ranks = jnp.asarray(
            [item.rank for item in extractions_], dtype=jnp.int32
        )
        self.basis_id = mesh.mesh_id
        self.global_coefficient_count = mesh.coefficient_count
        self.local_width = local_width
        self.realization_id = canonical_fingerprint(
            {
                "kind": "asts2d-extracted-bernstein-realization",
                "basis": mesh.mesh_id,
                "basis_certificate": basis_certificate.certificate_id,
                "extractions": [item.extraction_id for item in extractions_],
            }
        )

    @property
    def piece_condition_numbers(self) -> Array:
        """Per-piece binary64 condition numbers; platform-dependent diagnostics."""
        return jnp.asarray(
            [item.condition_number for item in self.extractions], dtype=jnp.float64
        )

    def extraction_piece(self, piece_id: int, /) -> ExtractedBernstein:
        identifier = int(piece_id)
        for extraction in self.extractions:
            if extraction.piece_id == identifier:
                return extraction
        raise KeyError(f"Unknown T-spline extraction piece {identifier}.")

    def extraction_pieces_for_cell(
        self, cell_id: int, /
    ) -> tuple[ExtractedBernstein, ...]:
        identifier = int(cell_id)
        if identifier not in {cell.cell_id for cell in self.mesh.cells}:
            raise KeyError(f"Unknown T-mesh cell {identifier}.")
        return tuple(
            extraction
            for extraction in self.extractions
            if extraction.cell_id == identifier
        )

    def _cell_row(self, cell_id: int, /) -> int:
        identifier = int(cell_id)
        for row, cell in enumerate(self.mesh.cells):
            if cell.cell_id == identifier:
                return row
        raise KeyError(f"Unknown T-mesh cell {identifier}.")

    def gather(self, coefficients: ArrayLike, cell_id: int, /) -> Array:
        values = jnp.asarray(coefficients)
        if values.shape[0] != self.global_coefficient_count:
            raise ValueError("Coefficient leading extent does not match the ASTS basis.")
        rows = self.cell_support_rows[self._cell_row(cell_id)]
        return values[jnp.asarray(rows, dtype=jnp.int32)]

    def gather_transpose(self, local_values: ArrayLike, cell_id: int, /) -> Array:
        rows = self.cell_support_rows[self._cell_row(cell_id)]
        values = jnp.asarray(local_values)
        if values.shape[0] != len(rows):
            raise ValueError("Local value leading extent does not match the cell gather.")
        result = jnp.zeros(
            (self.global_coefficient_count,) + values.shape[1:], dtype=values.dtype
        )
        return result.at[jnp.asarray(rows, dtype=jnp.int32)].add(values)

    def evaluate_cell(
        self,
        cell_id: int,
        parameter_points: ArrayLike,
        coefficients: ArrayLike,
        /,
        *,
        derivative: tuple[int, int] = (0, 0),
    ) -> Array:
        points = jnp.asarray(parameter_points, dtype=jnp.float64)
        routes = jnp.full((points.shape[0],), int(cell_id), dtype=jnp.int64)
        return self.evaluate(points, coefficients, derivative=derivative, cell_ids=routes)

    def _route_cells(self, points: Array, cell_ids: ArrayLike | None, /) -> Array:
        piece_count = self.piece_bounds.shape[0]
        u, v = points[:, 0, None], points[:, 1, None]
        bounds = self.piece_bounds[None, :, :]
        contained = (
            (u >= bounds[:, :, 0])
            & (u <= bounds[:, :, 1])
            & (v >= bounds[:, :, 2])
            & (v <= bounds[:, :, 3])
        )
        if cell_ids is None:
            matches = contained
        else:
            requested = jnp.asarray(cell_ids)
            if requested.shape != points.shape[:1] or not jnp.issubdtype(
                requested.dtype, jnp.integer
            ):
                raise ValueError(
                    "cell_ids must contain one integer ID per parameter point."
                )
            matches = (self.piece_cell_ids[None, :] == requested[:, None]) & contained
        rows = jnp.argmin(
            jnp.where(
                matches,
                jnp.arange(piece_count, dtype=jnp.int32)[None, :],
                piece_count,
            ),
            axis=1,
        )
        rows = eqx.error_if(
            rows,
            jnp.any(rows == piece_count),
            "A parameter point lies outside its fixed T-mesh cell route.",
        )
        return jnp.minimum(rows, piece_count - 1)

    def evaluate(
        self,
        parameter_points: ArrayLike,
        coefficients: ArrayLike,
        /,
        *,
        derivative: tuple[int, int] = (0, 0),
        cell_ids: ArrayLike | None = None,
    ) -> Array:
        points = jnp.asarray(parameter_points, dtype=jnp.float64)
        points = eqx.error_if(
            points,
            jnp.any(~jnp.isfinite(points)),
            "Parameter points must be finite.",
        )
        if points.ndim != 2 or points.shape[1] != 2:
            raise ValueError("parameter_points must have shape (point, 2).")
        values = jnp.asarray(coefficients)
        if values.ndim == 0 or values.shape[0] != self.global_coefficient_count:
            raise ValueError("Coefficient leading extent does not match the ASTS basis.")
        piece_rows = self._route_cells(points, cell_ids)
        bounds = self.piece_bounds[piece_rows]
        reference = jnp.stack(
            (
                (points[:, 0] - bounds[:, 0]) / (bounds[:, 1] - bounds[:, 0]),
                (points[:, 1] - bounds[:, 2]) / (bounds[:, 3] - bounds[:, 2]),
            ),
            axis=-1,
        )
        bu = _bernstein_cubic(reference[:, 0], derivative[0])
        bv = _bernstein_cubic(reference[:, 1], derivative[1])
        tensor = (bu[:, :, None] * bv[:, None, :]).reshape((points.shape[0], 16))
        local_operator = self.piece_extraction_operators[piece_rows]
        local_basis = contract("np,nlp->nl", tensor, local_operator, backend="jax")
        width_scale = (bounds[:, 1] - bounds[:, 0]) ** (-derivative[0]) * (
            bounds[:, 3] - bounds[:, 2]
        ) ** (-derivative[1])
        local_basis = local_basis * width_scale[:, None]
        local_rows = self.piece_coefficient_rows[piece_rows]
        local_values = values[local_rows]
        return contract("nl,nl...->n...", local_basis, local_values, backend="jax")

    def exact_homogeneous_piece(
        self,
        piece_id: int,
        homogeneous_coefficients: Sequence[Sequence[Fraction]],
        /,
    ) -> tuple[tuple[Fraction, ...], ...]:
        values = tuple(
            tuple(_fraction(value) for value in row) for row in homogeneous_coefficients
        )
        if len(values) != self.global_coefficient_count or not values:
            raise ValueError(
                "Homogeneous coefficients must align with every T-spline anchor."
            )
        payload_width = len(values[0])
        if payload_width == 0 or any(len(row) != payload_width for row in values):
            raise ValueError(
                "Homogeneous coefficient rows must have one common positive width."
            )
        extraction = self.extraction_piece(piece_id)
        return tuple(
            tuple(
                sum(
                    (
                        extraction.exact_extraction_operator[local_row][control]
                        * values[coefficient_row][component]
                        for local_row, coefficient_row in enumerate(
                            extraction.coefficient_row_ids
                        )
                    ),
                    Fraction(0),
                )
                for component in range(payload_width)
            )
            for control in range(16)
        )

    def homogeneous_piece(
        self,
        piece_id: int,
        control_points: ArrayLike,
        weights: ArrayLike,
        /,
    ) -> Array:
        points = jnp.asarray(control_points)
        weights_ = jnp.asarray(weights)
        if points.ndim != 2 or points.shape[0] != self.global_coefficient_count:
            raise ValueError(
                "Control points must be aligned with the T-spline anchor rows."
            )
        if weights_.shape != (self.global_coefficient_count,):
            raise ValueError("Weights must contain one value per T-spline anchor row.")
        extraction = self.extraction_piece(piece_id)
        homogeneous = jnp.concatenate(
            (points * weights_[:, None], weights_[:, None]), axis=-1
        )
        local = homogeneous[extraction.coefficient_rows]
        return contract(
            "li,lk->ki",
            local,
            extraction.extraction_operator,
            backend="jax",
        )

    def evaluate_rational(
        self,
        parameter_points: ArrayLike,
        control_points: ArrayLike,
        weights: ArrayLike,
        /,
        *,
        cell_ids: ArrayLike | None = None,
        maximum_order: int = 2,
    ) -> TSplineRationalJet:
        points = jnp.asarray(parameter_points, dtype=jnp.float64)
        controls = jnp.asarray(control_points)
        weights_ = jnp.asarray(weights)
        if maximum_order not in (0, 1, 2):
            raise ValueError(
                "T-spline rational jets support parameter order zero through two."
            )
        if controls.ndim != 2 or controls.shape[0] != self.global_coefficient_count:
            raise ValueError(
                "Control points must have shape (anchor_count, ambient_dimension)."
            )
        if weights_.shape != (self.global_coefficient_count,):
            raise ValueError("Weights must contain one value per T-spline anchor row.")
        controls = eqx.error_if(
            controls,
            jnp.any(~jnp.isfinite(controls)),
            "T-spline control points must be finite.",
        )
        weights_ = eqx.error_if(
            weights_,
            jnp.any(~jnp.isfinite(weights_)) | jnp.any(weights_ <= 0.0),
            "T-spline weights must be finite and strictly positive.",
        )
        homogeneous_points = controls * weights_[:, None]
        denominator = self.evaluate(points, weights_, cell_ids=cell_ids)
        denominator = eqx.error_if(
            denominator,
            jnp.any(~jnp.isfinite(denominator)) | jnp.any(denominator <= 0.0),
            "T-spline rational denominator must be finite and positive.",
        )
        numerator = self.evaluate(points, homogeneous_points, cell_ids=cell_ids)
        values = numerator / denominator[:, None]
        gradients = None
        hessians = None
        if maximum_order >= 1:
            denominator_gradient = jnp.stack(
                (
                    self.evaluate(points, weights_, derivative=(1, 0), cell_ids=cell_ids),
                    self.evaluate(points, weights_, derivative=(0, 1), cell_ids=cell_ids),
                ),
                axis=-1,
            )
            numerator_gradient = jnp.stack(
                (
                    self.evaluate(
                        points, homogeneous_points, derivative=(1, 0), cell_ids=cell_ids
                    ),
                    self.evaluate(
                        points, homogeneous_points, derivative=(0, 1), cell_ids=cell_ids
                    ),
                ),
                axis=-1,
            )
            gradients = (
                numerator_gradient - values[:, :, None] * denominator_gradient[:, None, :]
            ) / denominator[:, None, None]
        if maximum_order >= 2:
            if gradients is None:
                raise RuntimeError("First-order T-spline jets are absent.")
            second_orders = (((2, 0), 0, 0), ((1, 1), 0, 1), ((0, 2), 1, 1))
            hessian = jnp.zeros(
                (points.shape[0], controls.shape[1], 2, 2), dtype=values.dtype
            )
            for derivative, first_axis, second_axis in second_orders:
                denominator_second = self.evaluate(
                    points, weights_, derivative=derivative, cell_ids=cell_ids
                )
                numerator_second = self.evaluate(
                    points, homogeneous_points, derivative=derivative, cell_ids=cell_ids
                )
                component = (
                    numerator_second
                    - values * denominator_second[:, None]
                    - gradients[:, :, first_axis]
                    * denominator_gradient[:, second_axis, None]
                    - gradients[:, :, second_axis]
                    * denominator_gradient[:, first_axis, None]
                ) / denominator[:, None]
                hessian = hessian.at[:, :, first_axis, second_axis].set(component)
                hessian = hessian.at[:, :, second_axis, first_axis].set(component)
            hessians = hessian
        return TSplineRationalJet(values, gradients, hessians, denominator)


class TSplineRationalJet(StrictModule):
    """Raw-weight rational map and fixed-topology parameter derivatives."""

    values: Array
    gradients: Array | None
    hessians: Array | None
    denominator: Array

    def __init__(
        self,
        values: Array,
        gradients: Array | None,
        hessians: Array | None,
        denominator: Array,
        /,
    ) -> None:
        self.values = values
        self.gradients = gradients
        self.hessians = hessians
        self.denominator = denominator


def _ordered_anchor_supports(mesh: TMesh2D, /) -> tuple[tuple[Fraction, int], ...]:
    return tuple(
        sorted((anchor.local_knots.u[0], row) for row, anchor in enumerate(mesh.anchors))
    )


def _cell_support_rows(
    mesh: TMesh2D,
    bounds: Sequence[Fraction],
    ordered_u_supports: Sequence[tuple[Fraction, int]],
    /,
) -> tuple[int, ...]:
    u0, u1, v0, v1 = bounds
    lower_knots = tuple(value for value, _ in ordered_u_supports)
    stop = bisect_left(lower_knots, u1)
    return tuple(
        row
        for _, row in ordered_u_supports[:stop]
        if mesh.anchors[row].local_knots.u[-1] > u0
        and mesh.anchors[row].local_knots.v[0] < v1
        and mesh.anchors[row].local_knots.v[-1] > v0
    )


def _cell_support_index(mesh: TMesh2D, /) -> dict[int, tuple[int, ...]]:
    ordered = _ordered_anchor_supports(mesh)
    return {
        cell.cell_id: _cell_support_rows(mesh, cell.parameter_bounds, ordered)
        for cell in mesh.cells
    }


@dataclass(frozen=True, slots=True)
class _ExtractionPiece:
    piece_id: int
    cell_id: int
    parameter_bounds: tuple[Fraction, Fraction, Fraction, Fraction]


def _extraction_pieces(
    mesh: TMesh2D,
    support_index: Mapping[int, Sequence[int]] | None = None,
    /,
) -> tuple[_ExtractionPiece, ...]:
    """Partition control cells only for exact polynomial extraction."""
    support = _cell_support_index(mesh) if support_index is None else support_index
    pieces: list[_ExtractionPiece] = []
    for cell in mesh.cells:
        u0, u1, v0, v1 = cell.parameter_bounds
        u_breaks = {u0, u1}
        v_breaks = {v0, v1}
        for row in support[cell.cell_id]:
            knots = mesh.anchors[row].local_knots
            u_breaks.update(knot for knot in knots.u if u0 < knot < u1)
            v_breaks.update(knot for knot in knots.v if v0 < knot < v1)
        for lower_u, upper_u in zip(sorted(u_breaks)[:-1], sorted(u_breaks)[1:]):
            for lower_v, upper_v in zip(sorted(v_breaks)[:-1], sorted(v_breaks)[1:]):
                pieces.append(
                    _ExtractionPiece(
                        len(pieces),
                        cell.cell_id,
                        (lower_u, upper_u, lower_v, upper_v),
                    )
                )
    return tuple(pieces)


def _common_extraction_piece_bounds(
    meshes: Sequence[TMesh2D],
    bounds: tuple[Fraction, Fraction, Fraction, Fraction],
    /,
) -> tuple[tuple[Fraction, Fraction, Fraction, Fraction], ...]:
    """Partition one target cell at every source or target local knot line."""
    u0, u1, v0, v1 = bounds
    u_breaks = {u0, u1}
    v_breaks = {v0, v1}
    for mesh in meshes:
        ordered = _ordered_anchor_supports(mesh)
        for row in _cell_support_rows(mesh, bounds, ordered):
            knots = mesh.anchors[row].local_knots
            u_breaks.update(knot for knot in knots.u if u0 < knot < u1)
            v_breaks.update(knot for knot in knots.v if v0 < knot < v1)
    u_values, v_values = tuple(sorted(u_breaks)), tuple(sorted(v_breaks))
    return tuple(
        (lower_u, upper_u, lower_v, upper_v)
        for lower_u, upper_u in zip(u_values[:-1], u_values[1:])
        for lower_v, upper_v in zip(v_values[:-1], v_values[1:])
    )


def _cell_extraction(
    mesh: TMesh2D, piece: _ExtractionPiece, anchor_rows: Sequence[int], /
) -> ExtractedBernstein:
    u0, u1, v0, v1 = piece.parameter_bounds
    anchor_ids: list[int] = []
    coefficient_rows: list[int] = []
    exact_rows: list[tuple[Fraction, ...]] = []
    for row_index in sorted(anchor_rows):
        anchor = mesh.anchors[row_index]
        local_knots = anchor.local_knots
        u_vector = _line_knot_vector(
            "u", anchor.parameter[1], mesh.vertices, mesh.edges, mesh.parameter_domain
        )
        v_vector = _line_knot_vector(
            "v", anchor.parameter[0], mesh.vertices, mesh.edges, mesh.parameter_domain
        )
        u_index = _window_index(u_vector, local_knots.u)
        v_index = _window_index(v_vector, local_knots.v)
        u_coefficients = _univariate_extraction(u_vector, u_index, (u0, u1))
        v_coefficients = _univariate_extraction(v_vector, v_index, (v0, v1))
        exact_row = tuple(
            first * second for first in u_coefficients for second in v_coefficients
        )
        if any(value != 0 for value in exact_row):
            anchor_ids.append(anchor.anchor_id)
            coefficient_rows.append(row_index)
            exact_rows.append(exact_row)
    return ExtractedBernstein(
        piece.cell_id,
        piece.parameter_bounds,
        anchor_ids,
        coefficient_rows,
        exact_rows,
        piece_id=piece.piece_id,
    )


def _exact_piece_coefficients(
    extraction: ExtractedBernstein,
    coefficients: Sequence[Sequence[Fraction]],
    /,
) -> tuple[tuple[Fraction, ...], ...]:
    values = tuple(tuple(_fraction(value) for value in row) for row in coefficients)
    if not values or any(len(row) != len(values[0]) for row in values):
        raise ValueError("Exact piece coefficients must have a common positive width.")
    if any(row < 0 or row >= len(values) for row in extraction.coefficient_row_ids):
        raise ValueError("Exact piece routes contain an out-of-range coefficient row.")
    return tuple(
        tuple(
            sum(
                (
                    extraction.exact_extraction_operator[local_row][control]
                    * values[coefficient_row][component]
                    for local_row, coefficient_row in enumerate(
                        extraction.coefficient_row_ids
                    )
                ),
                Fraction(0),
            )
            for component in range(len(values[0]))
        )
        for control in range(16)
    )


def _local_realization(
    mesh: TMesh2D, basis_certificate: ASTSCertificate, /
) -> LocalExtractedBernsteinRealization:
    support_index = _cell_support_index(mesh)
    ordered_supports = _ordered_anchor_supports(mesh)
    pieces = _extraction_pieces(mesh, support_index)
    extractions = tuple(
        _cell_extraction(
            mesh,
            piece,
            _cell_support_rows(mesh, piece.parameter_bounds, ordered_supports),
        )
        for piece in pieces
    )
    return LocalExtractedBernsteinRealization(mesh, extractions, basis_certificate)


def _exact_bernstein_values(
    coordinate: Fraction,
    degree: int = _BICUBIC_DEGREE,
    /,
) -> tuple[Fraction, ...]:
    return tuple(
        Fraction(comb(degree, index))
        * coordinate**index
        * (1 - coordinate) ** (degree - index)
        for index in range(degree + 1)
    )


def _differentiate_bernstein_net(
    net: tuple[tuple[Fraction, ...], ...],
    axis: int,
    /,
) -> tuple[tuple[Fraction, ...], ...]:
    if axis not in (0, 1):
        raise ValueError("Bernstein-net differentiation axis must be zero or one.")
    degree = len(net) - 1 if axis == 0 else len(net[0]) - 1
    if degree <= 0:
        return ((Fraction(0),),)
    if axis == 0:
        return tuple(
            tuple(
                degree * (net[row + 1][column] - net[row][column])
                for column in range(len(net[0]))
            )
            for row in range(degree)
        )
    return tuple(
        tuple(degree * (row[column + 1] - row[column]) for column in range(degree))
        for row in net
    )


def _exact_trace_value(
    extraction: ExtractedBernstein,
    basis_row: int | None,
    normal_axis: int,
    side: int,
    derivative_order: int,
    tangential_coordinate: Fraction,
    /,
) -> Fraction:
    if basis_row is None:
        return Fraction(0)
    try:
        local_row = extraction.coefficient_row_ids.index(basis_row)
    except ValueError:
        return Fraction(0)
    coefficients = extraction.exact_extraction_operator[local_row]
    net = tuple(tuple(coefficients[4 * u + v] for v in range(4)) for u in range(4))
    for _ in range(derivative_order):
        net = _differentiate_bernstein_net(net, normal_axis)
    u0, u1, v0, v1 = extraction.parameter_bounds
    normal_width = (u1 - u0) if normal_axis == 0 else (v1 - v0)
    net = tuple(
        tuple(value / normal_width**derivative_order for value in row) for row in net
    )
    normal_extent = len(net) if normal_axis == 0 else len(net[0])
    normal_index = 0 if side == 0 else normal_extent - 1
    line = (
        net[normal_index] if normal_axis == 0 else tuple(row[normal_index] for row in net)
    )
    tangential_bounds = (v0, v1) if normal_axis == 0 else (u0, u1)
    local_tangent = (tangential_coordinate - tangential_bounds[0]) / (
        tangential_bounds[1] - tangential_bounds[0]
    )
    basis = _exact_bernstein_values(local_tangent)
    return sum(
        (value * coefficient for value, coefficient in zip(line, basis, strict=True)),
        Fraction(0),
    )


def _traces_are_continuous(
    mesh: TMesh2D,
    extractions: Sequence[ExtractedBernstein],
    /,
    *,
    continuity_order: int = 2,
) -> bool:
    row_by_anchor = {anchor.anchor_id: row for row, anchor in enumerate(mesh.anchors)}
    for first_index, first in enumerate(extractions):
        a0, a1, b0, b1 = first.parameter_bounds
        for second in extractions[first_index + 1 :]:
            c0, c1, d0, d1 = second.parameter_bounds
            if a1 == c0 or c1 == a0:
                normal_axis = 0
                first_side = 1 if a1 == c0 else 0
                second_side = 0 if a1 == c0 else 1
                lower, upper = max(b0, d0), min(b1, d1)
            elif b1 == d0 or d1 == b0:
                normal_axis = 1
                first_side = 1 if b1 == d0 else 0
                second_side = 0 if b1 == d0 else 1
                lower, upper = max(a0, c0), min(a1, c1)
            else:
                continue
            if lower >= upper:
                continue
            for anchor_row in row_by_anchor.values():
                for derivative_order in range(continuity_order + 1):
                    for numerator in range(_BICUBIC_DEGREE + 1):
                        coordinate = lower + (upper - lower) * Fraction(
                            numerator, _BICUBIC_DEGREE
                        )
                        if _exact_trace_value(
                            first,
                            anchor_row,
                            normal_axis,
                            first_side,
                            derivative_order,
                            coordinate,
                        ) != _exact_trace_value(
                            second,
                            anchor_row,
                            normal_axis,
                            second_side,
                            derivative_order,
                            coordinate,
                        ):
                            return False
    return True


class ASTSCertificate(StrictModule, NonTrainableState):
    """Fail-closed evidence for the complete basis-only ASTS2D claim."""

    mesh_id: str = eqx.field(static=True)
    analysis_suitable: bool = eqx.field(static=True)
    dual_compatible: bool = eqx.field(static=True)
    partition_of_unity: bool = eqx.field(static=True)
    nonnegative: bool = eqx.field(static=True)
    linearly_independent: bool = eqx.field(static=True)
    extraction_full_rank: bool = eqx.field(static=True)
    parameter_partition_valid: bool = eqx.field(static=True)
    trace_continuous: bool = eqx.field(static=True)
    continuity_order: int = eqx.field(static=True)
    maximum_partition_error: float | None = eqx.field(static=True)
    minimum_basis_value: float | None = eqx.field(static=True)
    minimum_extraction_rank: int | None = eqx.field(static=True)
    minimum_rank_margin: int | None = eqx.field(static=True)
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
            and self.parameter_partition_valid
            and self.trace_continuous
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
                "extraction_full_rank_diagnostic": self.extraction_full_rank,
                "parameter_partition_valid": self.parameter_partition_valid,
                "trace_continuous": self.trace_continuous,
                "continuity_order": self.continuity_order,
            },
            "metrics": {
                "maximum_partition_error": self.maximum_partition_error,
                "minimum_basis_value": self.minimum_basis_value,
                "minimum_extraction_rank": self.minimum_extraction_rank,
                "minimum_rank_margin": self.minimum_rank_margin,
                "tolerance": self.tolerance,
            },
            "failure_reasons": list(self.failure_reasons),
        }


def _bernstein_product(
    left: tuple[tuple[Fraction, ...], ...],
    right: tuple[tuple[Fraction, ...], ...],
    /,
) -> tuple[tuple[Fraction, ...], ...]:
    left_u, left_v = len(left) - 1, len(left[0]) - 1
    right_u, right_v = len(right) - 1, len(right[0]) - 1
    if any(len(row) != left_v + 1 for row in left) or any(
        len(row) != right_v + 1 for row in right
    ):
        raise ValueError("Bernstein polynomial coefficient nets must be rectangular.")
    result = [
        [Fraction(0) for _ in range(left_v + right_v + 1)]
        for _ in range(left_u + right_u + 1)
    ]
    for left_i, left_row in enumerate(left):
        for left_j, left_value in enumerate(left_row):
            for right_i, right_row in enumerate(right):
                for right_j, right_value in enumerate(right_row):
                    index_u, index_v = left_i + right_i, left_j + right_j
                    factor_u = Fraction(
                        comb(left_u, left_i) * comb(right_u, right_i),
                        comb(left_u + right_u, index_u),
                    )
                    factor_v = Fraction(
                        comb(left_v, left_j) * comb(right_v, right_j),
                        comb(left_v + right_v, index_v),
                    )
                    result[index_u][index_v] += (
                        left_value * right_value * factor_u * factor_v
                    )
    return tuple(tuple(row) for row in result)


class TSplineGeometryCertificate(StrictModule, NonTrainableState):
    """Exact regularity and global embedding evidence for a rational ASTS map."""

    mesh_id: str = eqx.field(static=True)
    basis_certificate_id: str = eqx.field(static=True)
    numeric_state_id: str = eqx.field(static=True)
    denominator_positive: bool = eqx.field(static=True)
    regular: bool = eqx.field(static=True)
    embedded: bool = eqx.field(static=True)
    projection_axes: tuple[int, int] | None = eqx.field(static=True)
    monotonicity_sign: int = eqx.field(static=True)
    projection_jacobian_bounds: tuple[tuple[Fraction, Fraction], ...] = eqx.field(
        static=True
    )
    piece_denominator_lower_bounds: tuple[tuple[int, int, Fraction], ...] = eqx.field(
        static=True
    )
    minimum_monotonicity_margin: Fraction = eqx.field(static=True)
    failure_reasons: tuple[str, ...] = eqx.field(static=True)
    certificate_id: str = eqx.field(static=True)

    @property
    def passed(self) -> bool:
        return (
            self.denominator_positive
            and self.regular
            and self.embedded
            and self.projection_axes is not None
            and self.minimum_monotonicity_margin > 0
            and not self.failure_reasons
        )

    def qualification_payload(self) -> dict[str, object]:
        return {
            "mesh_id": self.mesh_id,
            "basis_certificate_id": self.basis_certificate_id,
            "numeric_state_id": self.numeric_state_id,
            "certificate_id": self.certificate_id,
            "passed": self.passed,
            "checks": {
                "denominator_positive": self.denominator_positive,
                "regular": self.regular,
                "embedded": self.embedded,
            },
            "proof": {
                "kind": "exact-rational-strong-monotonicity-projection",
                "projection_axes": (
                    None if self.projection_axes is None else list(self.projection_axes)
                ),
                "monotonicity_sign": self.monotonicity_sign,
                "minimum_monotonicity_margin": [
                    self.minimum_monotonicity_margin.numerator,
                    self.minimum_monotonicity_margin.denominator,
                ],
                "projection_jacobian_bounds": [
                    [
                        [lower.numerator, lower.denominator],
                        [upper.numerator, upper.denominator],
                    ]
                    for lower, upper in self.projection_jacobian_bounds
                ],
                "piece_denominator_lower_bounds": [
                    [
                        cell_id,
                        piece_id,
                        [lower.numerator, lower.denominator],
                    ]
                    for cell_id, piece_id, lower in self.piece_denominator_lower_bounds
                ],
            },
            "failure_reasons": list(self.failure_reasons),
        }


def certify_asts(
    mesh: TMesh2D,
    /,
    *,
    tolerance: float = _DEFAULT_TOLERANCE,
) -> ASTSCertificate:
    """Certify topology-derived ASTS basis identities without a global matrix."""
    if not isinstance(mesh, TMesh2D):
        raise TypeError("mesh must be TMesh2D.")
    tolerance_ = float(tolerance)
    if not isfinite(tolerance_) or tolerance_ <= 0.0:
        raise ValueError("tolerance must be finite and positive.")
    parameter_partition_valid = _parameter_partition_is_valid(mesh)
    analysis_suitable = _extensions_are_analysis_suitable(mesh.t_junction_extensions)
    knot_compatible = _knot_metadata_is_compatible(mesh)
    dual_compatible = analysis_suitable and knot_compatible
    support_index = _cell_support_index(mesh)
    ordered_supports = _ordered_anchor_supports(mesh)
    pieces = _extraction_pieces(mesh, support_index)
    extraction_valid = True
    active_support_complete = True
    extractions: list[ExtractedBernstein] = []
    for piece in pieces:
        active_rows = _cell_support_rows(mesh, piece.parameter_bounds, ordered_supports)
        if not active_rows:
            active_support_complete = False
            extraction_valid = False
            continue
        u0, u1, v0, v1 = piece.parameter_bounds
        if any(
            any(u0 < knot < u1 for knot in mesh.anchors[row].local_knots.u)
            or any(v0 < knot < v1 for knot in mesh.anchors[row].local_knots.v)
            for row in active_rows
        ):
            extraction_valid = False
            continue
        extraction = _cell_extraction(mesh, piece, active_rows)
        if extraction.local_width == 0:
            active_support_complete = False
            extraction_valid = False
            continue
        extractions.append(extraction)
    partition_errors: list[Fraction] = []
    extraction_complete = extraction_valid and len(extractions) == len(pieces)
    exact_partition = extraction_complete
    exact_nonnegative = extraction_complete
    # Exact tensor Bernstein values at the binary64 Gauss-Legendre nodes, scaled
    # by one common denominator; rows follow the extraction column order.
    univariate = tuple(
        _exact_bernstein_values(Fraction.from_float(coordinate))
        for coordinate in (
            0.06943184420297371,
            0.33000947820757187,
            0.6699905217924281,
            0.9305681557970262,
        )
    )
    tensor_points = tuple(
        tuple(first * second for first in values_u for second in values_v)
        for values_u in univariate
        for values_v in univariate
    )
    sample_denominator = lcm(
        *(value.denominator for point in tensor_points for value in point)
    )
    sample_numerators = tuple(
        tuple(
            value.numerator * (sample_denominator // value.denominator) for value in point
        )
        for point in tensor_points
    )
    minimum_basis_value: float | None = None
    minimum_rank: int | None = None
    minimum_rank_margin: int | None = None
    if extraction_complete:
        minima: list[Fraction] = []
        ranks: list[int] = []
        margins: list[int] = []
        for extraction in extractions:
            exact_rows = extraction.exact_extraction_operator
            sums = tuple(
                sum((row[column] for row in exact_rows), Fraction(0))
                for column in range(16)
            )
            partition_errors.append(
                max((abs(value - 1) for value in sums), default=Fraction(1))
            )
            exact_partition = exact_partition and all(value == 1 for value in sums)
            exact_nonnegative = exact_nonnegative and all(
                value >= 0 for row in exact_rows for value in row
            )
            for row in exact_rows:
                row_denominator = lcm(*(value.denominator for value in row))
                row_numerators = tuple(
                    value.numerator * (row_denominator // value.denominator)
                    for value in row
                )
                minima.append(
                    Fraction(
                        min(
                            sum(
                                coefficient * basis
                                for coefficient, basis in zip(
                                    row_numerators, point, strict=True
                                )
                            )
                            for point in sample_numerators
                        ),
                        row_denominator * sample_denominator,
                    )
                )
            ranks.append(extraction.rank)
            margins.append(extraction.rank - extraction.local_width)
        minimum_basis_value = float(min(minima))
        minimum_rank = min(ranks)
        minimum_rank_margin = min(margins)
    maximum_partition_error: float | None = None
    if extraction_complete:
        try:
            partition_error = float(max(partition_errors))
        except OverflowError:
            partition_error = float("inf")
        maximum_partition_error = partition_error if isfinite(partition_error) else None
    full_rank = (
        extraction_complete
        and bool(extractions)
        and all(extraction.rank == extraction.local_width for extraction in extractions)
    )
    continuity_order = 2
    trace_continuous = (
        bool(extractions)
        and len(extractions) == len(pieces)
        and _traces_are_continuous(mesh, extractions, continuity_order=continuity_order)
    )
    unique_anchors = (
        len({anchor.anchor_id for anchor in mesh.anchors}) == mesh.coefficient_count
    )
    linearly_independent = (
        analysis_suitable
        and dual_compatible
        and parameter_partition_valid
        and exact_partition
        and exact_nonnegative
        and unique_anchors
    )
    partition = exact_partition
    nonnegative = exact_nonnegative
    checks = (
        analysis_suitable,
        dual_compatible,
        partition,
        nonnegative,
        linearly_independent,
        parameter_partition_valid,
        active_support_complete,
        extraction_valid,
        trace_continuous,
    )
    failure_reasons = tuple(
        reason
        for passed, reason in zip(checks, _ASTS_CHECK_REASONS, strict=True)
        if not passed
    )
    return ASTSCertificate(
        mesh.mesh_id,
        analysis_suitable,
        dual_compatible,
        partition,
        nonnegative,
        linearly_independent,
        full_rank,
        parameter_partition_valid,
        trace_continuous,
        continuity_order,
        maximum_partition_error,
        minimum_basis_value,
        minimum_rank,
        minimum_rank_margin,
        tolerance_,
        failure_reasons,
        _asts_certificate_id(
            mesh.mesh_id,
            checks,
            maximum_partition_error,
            minimum_basis_value,
            minimum_rank,
            minimum_rank_margin,
            full_rank,
            continuity_order,
            tolerance_,
        ),
    )


_ASTS_CHECK_REASONS = (
    "intersecting T-junction extensions",
    "incompatible topology-derived knot windows",
    "exact partition-of-unity identity",
    "negative exact Bernstein coefficient",
    "analysis-suitable independence premises",
    "invalid exact parameter cell complex",
    "T-mesh cell has no active basis support",
    "non-polynomial T-mesh cell",
    "exact cross-cell C2 trace continuity",
)


def _asts_certificate_id(
    mesh_id: str,
    checks: Sequence[bool],
    maximum_partition_error: float | None,
    minimum_basis_value: float | None,
    minimum_rank: int | None,
    minimum_rank_margin: int | None,
    full_rank: bool,
    continuity_order: int,
    tolerance: float,
    /,
) -> str:
    """Identity `certify_asts` issues for one proved basis certificate."""
    return canonical_fingerprint(
        {
            "kind": "asts2d-certificate",
            "mesh": mesh_id,
            "checks": [bool(value) for value in checks],
            "diagnostics": {
                "maximum_partition_error": maximum_partition_error,
                "minimum_sampled_basis_value": minimum_basis_value,
                "minimum_extraction_rank": minimum_rank,
                "minimum_rank_margin": minimum_rank_margin,
                "local_full_rank_diagnostic": full_rank,
                "continuity_order": continuity_order,
            },
            "tolerance": tolerance,
        }
    )


def _validated_asts_certificate(
    mesh: TMesh2D, certificate: ASTSCertificate, /
) -> ASTSCertificate:
    """Check a retained basis certificate against the identity its proof issued.

    `certify_asts` proves once; its identity binds the mesh content identity,
    every check, the exact rank and sampled-value diagnostics, and the
    tolerance. Binary64 conditioning is not part of it. A certificate whose
    fields no longer reproduce that identity (an edited copy) is refused; an
    intact one is the proof for exactly this mesh and is not proved again.
    """
    if (
        not isinstance(certificate, ASTSCertificate)
        or certificate.mesh_id != mesh.mesh_id
    ):
        raise ValueError("A retained ASTS certificate must match its immutable mesh.")
    reasons = certificate.failure_reasons
    checks = (
        certificate.analysis_suitable,
        certificate.dual_compatible,
        certificate.partition_of_unity,
        certificate.nonnegative,
        certificate.linearly_independent,
        certificate.parameter_partition_valid,
        _ASTS_CHECK_REASONS[6] not in reasons,
        _ASTS_CHECK_REASONS[7] not in reasons,
        certificate.trace_continuous,
    )
    expected_reasons = tuple(
        reason
        for passed, reason in zip(checks, _ASTS_CHECK_REASONS, strict=True)
        if not passed
    )
    if reasons != expected_reasons or certificate.certificate_id != _asts_certificate_id(
        certificate.mesh_id,
        checks,
        certificate.maximum_partition_error,
        certificate.minimum_basis_value,
        certificate.minimum_extraction_rank,
        certificate.minimum_rank_margin,
        certificate.extraction_full_rank,
        certificate.continuity_order,
        certificate.tolerance,
    ):
        raise ValueError(
            "A retained ASTS certificate does not match recomputed evidence identity."
        )
    return certificate


def _exact_numeric_rows(
    values: ArrayLike, name: str, /
) -> tuple[tuple[Fraction, ...], ...]:
    try:
        array = np.asarray(values)
    except Exception as exc:
        raise ValueError(f"{name} must be a concrete real numeric bank.") from exc
    if array.ndim != 2 or np.iscomplexobj(array) or array.dtype == np.bool_:
        raise ValueError(f"{name} must be a rank-2 real numeric bank.")

    def scalar(value: object, /) -> Fraction:
        if isinstance(value, (bool, np.bool_)):
            raise TypeError("Boolean values are not exact geometry coordinates.")
        if isinstance(value, Fraction):
            return value
        if isinstance(value, (float, np.floating)):
            return Fraction.from_float(float(value))
        if isinstance(value, (int, np.integer)):
            return Fraction(int(value))
        raise TypeError("Geometry coordinates must be real numeric scalars.")

    try:
        return tuple(tuple(scalar(value) for value in row) for row in array)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name} contains a non-finite or non-real value.") from exc


def _subtract_bernstein_nets(
    left: tuple[tuple[Fraction, ...], ...],
    right: tuple[tuple[Fraction, ...], ...],
    /,
) -> tuple[tuple[Fraction, ...], ...]:
    if len(left) != len(right) or any(
        len(first) != len(second) for first, second in zip(left, right, strict=True)
    ):
        raise ValueError("Bernstein polynomial products have incompatible degrees.")
    return tuple(
        tuple(first - second for first, second in zip(a, b, strict=True))
        for a, b in zip(left, right, strict=True)
    )


def certify_tspline_geometry(
    realization: LocalExtractedBernsteinRealization,
    control_points: ArrayLike,
    weights: ArrayLike,
    /,
) -> TSplineGeometryCertificate:
    """Prove positive denominator and global embedding by exact interval monotonicity.

    The proof searches pairs of physical coordinate axes. A pair qualifies only if
    exact Bernstein bounds prove the projected Jacobian's symmetric part is uniformly
    positive or negative definite on every local polynomial piece. Strong monotonicity
    then proves projection injective on the rectangular parameter domain; this is a
    sufficient, deliberately fail-closed embedding criterion.
    """
    if not isinstance(realization, LocalExtractedBernsteinRealization):
        raise TypeError("realization must be LocalExtractedBernsteinRealization.")
    mesh = realization.mesh
    basis_certificate = realization.basis_certificate
    if basis_certificate.mesh_id != mesh.mesh_id or not basis_certificate.passed:
        raise ValueError(
            "T-spline geometry realization has no matching passing ASTS evidence."
        )
    controls = _exact_numeric_rows(control_points, "control_points")
    raw_weights = np.asarray(weights)
    if raw_weights.shape != (mesh.coefficient_count,):
        raise ValueError("Weights must contain one value per T-spline anchor.")
    weight_rows = _exact_numeric_rows(raw_weights.reshape((-1, 1)), "weights")
    weight_values = tuple(row[0] for row in weight_rows)
    if (
        len(controls) != mesh.coefficient_count
        or len(weight_values) != mesh.coefficient_count
    ):
        raise ValueError("Geometry banks must contain one row per T-spline anchor.")
    if (
        not controls
        or len(controls[0]) < 2
        or any(len(row) != len(controls[0]) for row in controls)
    ):
        raise ValueError(
            "T-spline geometry requires a common ambient dimension of at least two."
        )
    if any(weight <= 0 for weight in weight_values):
        raise ValueError("T-spline geometry weights must be strictly positive.")
    exact_geometry = tuple(
        tuple(weight * coordinate for coordinate in point) + (weight,)
        for point, weight in zip(controls, weight_values, strict=True)
    )
    numeric_state_id = canonical_fingerprint(
        {
            "kind": "asts2d-exact-binary-rational-geometry-bank",
            "control_points": [_fraction_payload(row) for row in controls],
            "weights": _fraction_payload(weight_values),
        }
    )
    reasons: list[str] = []
    if not basis_certificate.passed:
        reasons.append("ASTS basis certificate does not pass.")

    derivative_bounds: dict[tuple[int, int], list[tuple[Fraction, Fraction]]] = {}
    piece_denominator_lower_bounds: list[tuple[int, int, Fraction]] = []
    denominator_positive = basis_certificate.passed
    if basis_certificate.passed:
        for extraction in realization.extractions:
            homogeneous_piece = realization.exact_homogeneous_piece(
                extraction.piece_id, exact_geometry
            )
            ambient_dimension = len(controls[0])
            nets = tuple(
                tuple(
                    tuple(
                        homogeneous_piece[4 * u_index + v_index][component]
                        for v_index in range(4)
                    )
                    for u_index in range(4)
                )
                for component in range(ambient_dimension + 1)
            )
            denominator = nets[-1]
            denominator_squared = _bernstein_product(denominator, denominator)
            denominator_lower = min(value for row in denominator for value in row)
            piece_denominator_lower_bounds.append(
                (extraction.cell_id, extraction.piece_id, denominator_lower)
            )
            denominator_positive = denominator_positive and denominator_lower > 0
            if denominator_lower <= 0:
                continue
            squared_lower = min(value for row in denominator_squared for value in row)
            squared_upper = max(value for row in denominator_squared for value in row)
            if squared_lower <= 0:
                denominator_positive = False
                continue
            u0, u1, v0, v1 = extraction.parameter_bounds
            widths = (u1 - u0, v1 - v0)
            for physical_axis, numerator_net in enumerate(nets[:-1]):
                for parameter_axis in (0, 1):
                    derivative_numerator = _differentiate_bernstein_net(
                        numerator_net, parameter_axis
                    )
                    derivative_denominator = _differentiate_bernstein_net(
                        denominator, parameter_axis
                    )
                    width = widths[parameter_axis]
                    derivative_numerator = tuple(
                        tuple(value / width for value in row)
                        for row in derivative_numerator
                    )
                    derivative_denominator = tuple(
                        tuple(value / width for value in row)
                        for row in derivative_denominator
                    )
                    rational_numerator = _subtract_bernstein_nets(
                        _bernstein_product(derivative_numerator, denominator),
                        _bernstein_product(numerator_net, derivative_denominator),
                    )
                    numerator_lower = min(
                        value for row in rational_numerator for value in row
                    )
                    numerator_upper = max(
                        value for row in rational_numerator for value in row
                    )
                    quotients = (
                        numerator_lower / squared_lower,
                        numerator_lower / squared_upper,
                        numerator_upper / squared_lower,
                        numerator_upper / squared_upper,
                    )
                    derivative_bounds.setdefault(
                        (physical_axis, parameter_axis), []
                    ).append((min(quotients), max(quotients)))

    projection_axes: tuple[int, int] | None = None
    monotonicity_sign = 0
    projection_jacobian_bounds: tuple[tuple[Fraction, Fraction], ...] = ()
    minimum_margin = Fraction(0)
    ambient_dimension = len(controls[0])
    if basis_certificate.passed and denominator_positive:
        all_bounds = {
            key: (
                min(lower for lower, _ in ranges),
                max(upper for _, upper in ranges),
            )
            for key, ranges in derivative_bounds.items()
        }
        best: (
            tuple[Fraction, tuple[int, int], int, tuple[tuple[Fraction, Fraction], ...]]
            | None
        ) = None
        for first_axis in range(ambient_dimension):
            for second_axis in range(first_axis + 1, ambient_dimension):
                jacobian_bounds = (
                    all_bounds.get((first_axis, 0), (Fraction(0), Fraction(0))),
                    all_bounds.get((first_axis, 1), (Fraction(0), Fraction(0))),
                    all_bounds.get((second_axis, 0), (Fraction(0), Fraction(0))),
                    all_bounds.get((second_axis, 1), (Fraction(0), Fraction(0))),
                )
                off_diagonal = (
                    (jacobian_bounds[1][0] + jacobian_bounds[2][0]) / 2,
                    (jacobian_bounds[1][1] + jacobian_bounds[2][1]) / 2,
                )
                off_bound = max(abs(off_diagonal[0]), abs(off_diagonal[1]))
                for sign in (1, -1):
                    first_lower = (
                        jacobian_bounds[0][0] if sign > 0 else -jacobian_bounds[0][1]
                    )
                    second_lower = (
                        jacobian_bounds[3][0] if sign > 0 else -jacobian_bounds[3][1]
                    )
                    margin = min(first_lower - off_bound, second_lower - off_bound)
                    if margin > 0 and (best is None or margin > best[0]):
                        best = (
                            margin,
                            (first_axis, second_axis),
                            sign,
                            jacobian_bounds,
                        )
        if best is not None:
            (
                minimum_margin,
                projection_axes,
                monotonicity_sign,
                projection_jacobian_bounds,
            ) = best
    if not denominator_positive:
        reasons.append("Rational denominator lacks a positive exact Bernstein bound.")
    if projection_axes is None:
        reasons.append(
            "No physical coordinate projection has a strict global monotonicity proof."
        )
    regular = (
        basis_certificate.passed and denominator_positive and projection_axes is not None
    )
    embedded = regular
    payload = {
        "kind": "asts2d-physical-map-certificate",
        "mesh": mesh.mesh_id,
        "basis_certificate": basis_certificate.certificate_id,
        "numeric_state": numeric_state_id,
        "denominator_positive": denominator_positive,
        "regular": regular,
        "embedded": embedded,
        "projection_axes": projection_axes,
        "monotonicity_sign": monotonicity_sign,
        "projection_jacobian_bounds": [
            [
                [lower.numerator, lower.denominator],
                [upper.numerator, upper.denominator],
            ]
            for lower, upper in projection_jacobian_bounds
        ],
        "piece_denominator_lower_bounds": [
            [
                cell_id,
                piece_id,
                [lower.numerator, lower.denominator],
            ]
            for cell_id, piece_id, lower in piece_denominator_lower_bounds
        ],
        "minimum_monotonicity_margin": [
            minimum_margin.numerator,
            minimum_margin.denominator,
        ],
        "failure_reasons": reasons,
    }
    return TSplineGeometryCertificate(
        mesh.mesh_id,
        basis_certificate.certificate_id,
        numeric_state_id,
        denominator_positive,
        regular,
        embedded,
        projection_axes,
        monotonicity_sign,
        projection_jacobian_bounds,
        tuple(sorted(piece_denominator_lower_bounds)),
        minimum_margin,
        tuple(reasons),
        canonical_fingerprint(payload),
    )


class ASTSTransferPlan(StrictModule, NonTrainableState):
    """Exact sparse target-by-source prolongation for one nested ASTS basis."""

    source_mesh_id: str = eqx.field(static=True)
    target_mesh_id: str = eqx.field(static=True)
    source_count: int = eqx.field(static=True)
    target_count: int = eqx.field(static=True)
    route_targets: tuple[int, ...] = eqx.field(static=True)
    route_sources: tuple[int, ...] = eqx.field(static=True)
    exact_coefficients: tuple[Fraction, ...] = eqx.field(static=True)
    target_indices: Array
    source_indices: Array
    coefficients: Array
    maximum_coefficient_conversion_error: float = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        source_mesh_id: str,
        target_mesh_id: str,
        source_count: int,
        target_count: int,
        target_indices: Sequence[int],
        source_indices: Sequence[int],
        exact_coefficients: Sequence[Fraction],
        /,
    ) -> None:
        routes = tuple(
            sorted(
                (
                    int(target),
                    int(source),
                    _fraction(value),
                )
                for target, source, value in zip(
                    target_indices, source_indices, exact_coefficients, strict=True
                )
            )
        )
        source_count_ = int(source_count)
        target_count_ = int(target_count)
        if source_count_ <= 0 or target_count_ <= 0 or not routes:
            raise ValueError("A transfer plan requires nonempty source and target bases.")
        if len({(target, source) for target, source, _ in routes}) != len(routes):
            raise ValueError(
                "Exact transfer routes must have unique target/source pairs."
            )
        if any(
            target < 0
            or target >= target_count_
            or source < 0
            or source >= source_count_
            or value <= 0
            for target, source, value in routes
        ):
            raise ValueError(
                "Exact transfer routes require positive weights and valid rows."
            )
        numeric = np.asarray([float(value) for _, _, value in routes], dtype=np.float64)
        if not np.all(np.isfinite(numeric)):
            raise ValueError(
                "Exact transfer coefficients have no finite binary64 realization."
            )
        if np.any(numeric <= 0.0):
            raise ValueError("A nonzero exact transfer coefficient underflows binary64.")
        conversion_error = max(
            (
                float(
                    np.nextafter(
                        float(abs(value - Fraction(float(numeric[index])))), np.inf
                    )
                )
                if value != Fraction(float(numeric[index]))
                else 0.0
                for index, (_, _, value) in enumerate(routes)
            ),
            default=0.0,
        )
        targets = tuple(target for target, _, _ in routes)
        sources = tuple(source for _, source, _ in routes)
        exact = tuple(value for _, _, value in routes)
        source_identifier = _identifier("source_mesh_id", source_mesh_id)
        target_identifier = _identifier("target_mesh_id", target_mesh_id)
        self.source_mesh_id = source_identifier
        self.target_mesh_id = target_identifier
        self.source_count = source_count_
        self.target_count = target_count_
        self.route_targets = targets
        self.route_sources = sources
        self.exact_coefficients = exact
        self.target_indices = jnp.asarray(targets, dtype=jnp.int32)
        self.source_indices = jnp.asarray(sources, dtype=jnp.int32)
        self.coefficients = jnp.asarray(numeric)
        self.maximum_coefficient_conversion_error = conversion_error
        self.plan_id = canonical_fingerprint(
            {
                "kind": "asts2d-exact-transfer-plan",
                "source": source_identifier,
                "target": target_identifier,
                "source_count": source_count_,
                "target_count": target_count_,
                "routes": [
                    [
                        target,
                        source,
                        value.numerator,
                        value.denominator,
                    ]
                    for target, source, value in routes
                ],
                "numeric_coefficients": _array_digest(numeric),
                "maximum_coefficient_conversion_error": conversion_error,
            }
        )

    @property
    def route_count(self) -> int:
        return len(self.exact_coefficients)

    def require_current_identity(self, /) -> None:
        """Refuse a plan whose retained routes no longer reproduce its identity.

        Traced numeric leaves are checked by the concrete call that built them.
        """
        rebuilt = ASTSTransferPlan(
            self.source_mesh_id,
            self.target_mesh_id,
            self.source_count,
            self.target_count,
            self.route_targets,
            self.route_sources,
            self.exact_coefficients,
        )
        leaves = (self.target_indices, self.source_indices, self.coefficients)
        concrete = not any(isinstance(leaf, jax_core.Tracer) for leaf in leaves)
        if rebuilt.plan_id != self.plan_id or (
            concrete
            and not all(
                np.array_equal(np.asarray(expected), np.asarray(leaf))
                for expected, leaf in zip(
                    (
                        rebuilt.target_indices,
                        rebuilt.source_indices,
                        rebuilt.coefficients,
                    ),
                    leaves,
                    strict=True,
                )
            )
        ):
            raise ValueError(
                "A retained ASTS transfer plan does not reproduce its exact identity."
            )

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

    def apply_exact_scalars(
        self, source_coefficients: Sequence[Fraction], /
    ) -> tuple[Fraction, ...]:
        values = tuple(_fraction(value) for value in source_coefficients)
        if len(values) != self.source_count:
            raise ValueError("Exact source coefficients must match the source basis.")
        target = [Fraction(0) for _ in range(self.target_count)]
        for target_row, source_row, coefficient in zip(
            self.route_targets, self.route_sources, self.exact_coefficients, strict=True
        ):
            target[target_row] += coefficient * values[source_row]
        return tuple(target)

    @overload
    def apply_exact_payload(
        self, source_coefficients: Sequence[Fraction], /
    ) -> tuple[Fraction, ...]: ...

    @overload
    def apply_exact_payload(
        self, source_coefficients: Sequence[Sequence[Fraction]], /
    ) -> tuple[tuple[Fraction, ...], ...]: ...

    def apply_exact_payload(
        self,
        source_coefficients: Sequence[Sequence[Fraction] | Fraction],
        /,
    ) -> tuple[Fraction, ...] | tuple[tuple[Fraction, ...], ...]:
        if len(source_coefficients) == 0:
            raise ValueError("Exact source payloads must match the source basis.")
        scalar_payload = not isinstance(source_coefficients[0], Sequence)
        normalized: list[tuple[Fraction, ...]] = []
        for row in source_coefficients:
            if scalar_payload:
                if isinstance(row, Sequence):
                    raise ValueError(
                        "Exact source payload rows must have one common shape."
                    )
                normalized.append((_fraction(row),))
            else:
                if not isinstance(row, Sequence):
                    raise ValueError(
                        "Exact source payload rows must have one common shape."
                    )
                normalized.append(tuple(_fraction(value) for value in row))
        values = tuple(normalized)
        if len(values) != self.source_count:
            raise ValueError("Exact source payloads must match the source basis.")
        if not values[0] or any(len(row) != len(values[0]) for row in values):
            raise ValueError(
                "Exact source payload rows must have one common positive width."
            )
        width = len(values[0])
        target = [[Fraction(0) for _ in range(width)] for _ in range(self.target_count)]
        for target_row, source_row, coefficient in zip(
            self.route_targets, self.route_sources, self.exact_coefficients, strict=True
        ):
            for component in range(width):
                target[target_row][component] += (
                    coefficient * values[source_row][component]
                )
        if scalar_payload:
            return tuple(row[0] for row in target)
        return tuple(tuple(row) for row in target)

    def apply_exact_homogeneous_geometry(
        self,
        control_points: Sequence[Sequence[Fraction]],
        weights: Sequence[Fraction],
        /,
    ) -> tuple[tuple[Fraction, ...], ...]:
        """Prolong the authored rational geometry bank exactly in homogeneous form."""
        points = tuple(tuple(_fraction(value) for value in row) for row in control_points)
        weights_ = tuple(_fraction(value) for value in weights)
        if len(points) != self.source_count or len(weights_) != self.source_count:
            raise ValueError("Exact geometry values must match the source basis.")
        if (
            not points
            or not points[0]
            or any(len(row) != len(points[0]) for row in points)
        ):
            raise ValueError(
                "Exact control points require one common positive dimension."
            )
        if any(weight <= 0 for weight in weights_):
            raise ValueError("Exact geometry weights must be strictly positive.")
        homogeneous = tuple(
            tuple(weight * coordinate for coordinate in point) + (weight,)
            for point, weight in zip(points, weights_, strict=True)
        )
        return self.apply_exact_payload(homogeneous)

    def verify_exact_homogeneous_geometry_preservation(
        self,
        source_mesh: TMesh2D,
        target_mesh: TMesh2D,
        source_control_points: Sequence[Sequence[Fraction]],
        source_weights: Sequence[Fraction],
        target_homogeneous_coefficients: Sequence[Sequence[Fraction]],
        /,
        *,
        source_certificate: ASTSCertificate | None = None,
        target_certificate: ASTSCertificate | None = None,
    ) -> None:
        """Prove exact nested-basis and homogeneous geometry preservation.

        The target bank must be the exact transfer of the source bank. Exact
        Bernstein coefficient comparisons on common source/target polynomial
        pieces prove equality directly, without sampling error.
        """
        if (
            not isinstance(source_mesh, TMesh2D)
            or not isinstance(target_mesh, TMesh2D)
            or source_mesh.mesh_id != self.source_mesh_id
            or target_mesh.mesh_id != self.target_mesh_id
            or source_mesh.coefficient_count != self.source_count
            or target_mesh.coefficient_count != self.target_count
        ):
            raise ValueError("The exact transfer plan does not match both T-meshes.")
        source_certificate_ = (
            source_mesh.certify()
            if source_certificate is None
            else _validated_asts_certificate(source_mesh, source_certificate)
        )
        target_certificate_ = (
            target_mesh.certify()
            if target_certificate is None
            else _validated_asts_certificate(target_mesh, target_certificate)
        )
        if not source_certificate_.passed or not target_certificate_.passed:
            raise ValueError(
                "Exact geometry transfer requires passing ASTS certificates."
            )
        routes = tuple(
            zip(
                self.route_targets,
                self.route_sources,
                self.exact_coefficients,
                strict=True,
            )
        )
        if routes != _exact_basis_transfer_routes(source_mesh, target_mesh):
            raise ValueError(
                "The stored exact transfer routes do not match these meshes."
            )
        _verify_local_basis_inclusion(
            source_mesh, target_mesh, routes, target_certificate_
        )

        source_points = tuple(
            tuple(_fraction(value) for value in point) for point in source_control_points
        )
        source_weights_ = tuple(_fraction(value) for value in source_weights)
        if (
            len(source_points) != self.source_count
            or not source_points
            or not source_points[0]
            or any(len(point) != len(source_points[0]) for point in source_points)
            or len(source_weights_) != self.source_count
            or any(weight <= 0 for weight in source_weights_)
        ):
            raise ValueError(
                "Exact source geometry must align with the positive source basis."
            )
        source_homogeneous = tuple(
            tuple(weight * coordinate for coordinate in point) + (weight,)
            for point, weight in zip(source_points, source_weights_, strict=True)
        )
        expected_target = self.apply_exact_payload(source_homogeneous)
        persisted_target = tuple(
            tuple(_fraction(value) for value in row)
            for row in target_homogeneous_coefficients
        )
        if persisted_target != expected_target:
            raise ValueError(
                "Persisted exact target geometry is not the source prolongation."
            )

        source_supports = _ordered_anchor_supports(source_mesh)
        target_supports = _ordered_anchor_supports(target_mesh)
        source_cells = tuple(source_mesh.cells)
        for target_cell in target_mesh.cells:
            target_bounds = target_cell.parameter_bounds
            source_cell = next(
                (
                    cell
                    for cell in source_cells
                    if cell.parameter_bounds[0] <= target_bounds[0]
                    and target_bounds[1] <= cell.parameter_bounds[1]
                    and cell.parameter_bounds[2] <= target_bounds[2]
                    and target_bounds[3] <= cell.parameter_bounds[3]
                ),
                None,
            )
            if source_cell is None:
                raise ValueError("A refined cell is not contained in a source cell.")
            for bounds in _common_extraction_piece_bounds(
                (source_mesh, target_mesh), target_bounds
            ):
                piece = _ExtractionPiece(0, target_cell.cell_id, bounds)
                source_extraction = _cell_extraction(
                    source_mesh,
                    piece,
                    _cell_support_rows(source_mesh, bounds, source_supports),
                )
                target_extraction = _cell_extraction(
                    target_mesh,
                    piece,
                    _cell_support_rows(target_mesh, bounds, target_supports),
                )
                source_piece = _exact_piece_coefficients(
                    source_extraction, source_homogeneous
                )
                target_piece = _exact_piece_coefficients(
                    target_extraction, expected_target
                )
                if source_piece != target_piece:
                    raise ValueError(
                        "Exact homogeneous Bernstein geometry changed under refinement."
                    )

    @overload
    def apply_exact_rational_field(
        self,
        coefficients: Sequence[Fraction],
        weights: Sequence[Fraction],
        /,
    ) -> tuple[tuple[Fraction, ...], tuple[Fraction, ...]]: ...

    @overload
    def apply_exact_rational_field(
        self,
        coefficients: Sequence[Sequence[Fraction]],
        weights: Sequence[Fraction],
        /,
    ) -> tuple[tuple[tuple[Fraction, ...], ...], tuple[Fraction, ...]]: ...

    def apply_exact_rational_field(
        self,
        coefficients: Sequence[Sequence[Fraction] | Fraction],
        weights: Sequence[Fraction],
        /,
    ) -> tuple[
        tuple[Fraction, ...] | tuple[tuple[Fraction, ...], ...],
        tuple[Fraction, ...],
    ]:
        """Prolong exact scalar or vector fields in their weighted basis."""
        if len(coefficients) == 0:
            raise ValueError("Exact field coefficients must match the source basis.")
        scalar_field = not isinstance(coefficients[0], Sequence)
        normalized: list[tuple[Fraction, ...]] = []
        for row in coefficients:
            if scalar_field:
                if isinstance(row, Sequence):
                    raise ValueError("Exact field rows must have one common shape.")
                normalized.append((_fraction(row),))
            else:
                if not isinstance(row, Sequence):
                    raise ValueError("Exact field rows must have one common shape.")
                normalized.append(tuple(_fraction(value) for value in row))
        values = tuple(normalized)
        weights_ = tuple(_fraction(value) for value in weights)
        if len(values) != self.source_count or len(weights_) != self.source_count:
            raise ValueError(
                "Exact field values and weights must match the source basis."
            )
        if not values[0] or any(len(row) != len(values[0]) for row in values):
            raise ValueError("Exact field coefficients require a common positive width.")
        if any(weight <= 0 for weight in weights_):
            raise ValueError("Exact field weights must be strictly positive.")
        target_weights = self.apply_exact_scalars(weights_)
        if any(weight <= 0 for weight in target_weights):
            raise ValueError("Prolongated exact field weights must remain positive.")
        weighted_values = tuple(
            tuple(weight * value for value in row)
            for row, weight in zip(values, weights_, strict=True)
        )
        target_weighted = self.apply_exact_payload(weighted_values)
        target_values = tuple(
            tuple(value / weight for value in row)
            for row, weight in zip(target_weighted, target_weights, strict=True)
        )
        if scalar_field:
            return tuple(row[0] for row in target_values), target_weights
        return target_values, target_weights

    def apply_homogeneous_geometry(
        self,
        control_points: ArrayLike,
        weights: ArrayLike,
        /,
    ) -> tuple[Array, Array]:
        points = jnp.asarray(control_points)
        weights_ = jnp.asarray(weights)
        if points.ndim != 2 or points.shape[0] != self.source_count:
            raise ValueError("Control points must match the transfer source basis.")
        if weights_.shape != (self.source_count,):
            raise ValueError("Geometry weights must match the transfer source basis.")
        weights_ = eqx.error_if(
            weights_,
            jnp.any(~jnp.isfinite(weights_)) | jnp.any(weights_ <= 0.0),
            "Homogeneous geometry weights must be finite and positive.",
        )
        homogeneous = jnp.concatenate(
            (points * weights_[:, None], weights_[:, None]), axis=-1
        )
        target_homogeneous = self.apply(homogeneous)
        target_weights = eqx.error_if(
            target_homogeneous[:, -1],
            jnp.any(~jnp.isfinite(target_homogeneous[:, -1]))
            | jnp.any(target_homogeneous[:, -1] <= 0.0),
            "Prolongated homogeneous weights must remain finite and positive.",
        )
        target_points = target_homogeneous[:, :-1] / target_weights[:, None]
        return target_points, target_weights

    def apply_rational_field(
        self,
        coefficients: ArrayLike,
        weights: ArrayLike,
        /,
    ) -> tuple[Array, Array]:
        values = jnp.asarray(coefficients)
        weights_ = jnp.asarray(weights)
        if values.shape[0] != self.source_count:
            raise ValueError(
                "Rational field coefficients must match the transfer source basis."
            )
        if weights_.shape != (self.source_count,):
            raise ValueError(
                "Rational field weights must match the transfer source basis."
            )
        weights_ = eqx.error_if(
            weights_,
            jnp.any(~jnp.isfinite(weights_)) | jnp.any(weights_ <= 0.0),
            "Rational field weights must be finite and positive.",
        )
        payload_scale = weights_.reshape((self.source_count,) + (1,) * (values.ndim - 1))
        target_weights = self.apply(weights_)
        target_weights = eqx.error_if(
            target_weights,
            jnp.any(~jnp.isfinite(target_weights)) | jnp.any(target_weights <= 0.0),
            "Prolongated rational field weights must remain positive.",
        )
        target_values = self.apply(values * payload_scale)
        target_scale = target_weights.reshape(
            (self.target_count,) + (1,) * (values.ndim - 1)
        )
        return target_values / target_scale, target_weights


@dataclass(frozen=True, slots=True)
class _ProvedRefinementInclusion:
    """Issued only by `analysis_suitable_refinement_closure` after its proof.

    The closure has certified both bases, resolved the exact routes and proved
    complete local basis inclusion for exactly this plan; the refinement it
    constructs checks these identities instead of proving them again.
    """

    source_certificate: ASTSCertificate
    target_certificate_id: str
    plan_id: str


class ASTSRefinement(StrictModule, NonTrainableState):
    """Atomic local refinement with exact basis and entity lineage evidence."""

    source_mesh: TMesh2D
    refined_mesh: TMesh2D
    marked_cell_ids: tuple[int, ...] = eqx.field(static=True)
    closure_cell_ids: tuple[int, ...] = eqx.field(static=True)
    cell_lineage: tuple[tuple[int, tuple[int, ...]], ...] = eqx.field(static=True)
    transfer_plan: ASTSTransferPlan
    certificate: ASTSCertificate
    refinement_id: str = eqx.field(static=True)

    def __init__(
        self,
        source_mesh: TMesh2D,
        refined_mesh: TMesh2D,
        marked_cell_ids: Sequence[int],
        closure_cell_ids: Sequence[int],
        cell_lineage: Sequence[tuple[int, Sequence[int]]],
        transfer_plan: ASTSTransferPlan,
        certificate: ASTSCertificate,
        /,
        *,
        _proved_inclusion: _ProvedRefinementInclusion | None = None,
    ) -> None:
        if transfer_plan.source_mesh_id != source_mesh.mesh_id:
            raise ValueError(
                "Transfer source identity does not match the refinement source."
            )
        if transfer_plan.target_mesh_id != refined_mesh.mesh_id:
            raise ValueError("Transfer target identity does not match the refined mesh.")
        proof = (
            _proved_inclusion
            if isinstance(_proved_inclusion, _ProvedRefinementInclusion)
            and _proved_inclusion.plan_id == transfer_plan.plan_id
            and _proved_inclusion.target_certificate_id == certificate.certificate_id
            else None
        )
        proved = proof is not None
        source_certificate = (
            source_mesh.certify()
            if proof is None
            else _validated_asts_certificate(source_mesh, proof.source_certificate)
        )
        certificate_ = _validated_asts_certificate(refined_mesh, certificate)
        if not source_certificate.passed or not certificate_.passed:
            raise ValueError("A refinement result requires passing ASTS certificates.")
        if not proved:
            routes = tuple(
                zip(
                    transfer_plan.route_targets,
                    transfer_plan.route_sources,
                    transfer_plan.exact_coefficients,
                    strict=True,
                )
            )
            if routes != _exact_basis_transfer_routes(source_mesh, refined_mesh):
                raise ValueError(
                    "A refinement result requires exact source-to-target routes."
                )
            _verify_local_basis_inclusion(source_mesh, refined_mesh, routes, certificate_)
        marked = tuple(sorted(int(value) for value in marked_cell_ids))
        closure = tuple(sorted(int(value) for value in closure_cell_ids))
        lineage = tuple(
            sorted(
                (
                    int(parent),
                    tuple(int(child) for child in children),
                )
                for parent, children in cell_lineage
            )
        )
        if any(
            parent < 0 or not children or any(child < 0 for child in children)
            for parent, children in lineage
        ):
            raise ValueError("Refinement lineage must contain nonempty nonnegative IDs.")
        self.source_mesh = source_mesh
        self.refined_mesh = refined_mesh
        self.marked_cell_ids = marked
        self.closure_cell_ids = closure
        self.cell_lineage = lineage
        self.transfer_plan = transfer_plan
        self.certificate = certificate_
        self.refinement_id = canonical_fingerprint(
            {
                "kind": "asts2d-local-refinement",
                "source": source_mesh.mesh_id,
                "target": refined_mesh.mesh_id,
                "marked": list(marked),
                "closure": list(closure),
                "cell_lineage": [
                    [parent, list(children)] for parent, children in lineage
                ],
                "transfer": transfer_plan.plan_id,
                "certificate": certificate.certificate_id,
            }
        )


@dataclass(frozen=True, slots=True)
class _RefinementCell:
    root_id: int
    address: tuple[int, ...]
    bounds: tuple[Fraction, Fraction, Fraction, Fraction]
    level: int
    parent_key: tuple[int, tuple[int, ...]] | None
    original_cell: TCell2D | None

    @property
    def key(self) -> tuple[int, tuple[int, ...]]:
        return self.root_id, self.address


def _record_order(
    record: _RefinementCell, /
) -> tuple[int, tuple[int, ...], tuple[Fraction, ...]]:
    return record.root_id, record.address, record.bounds


def _temporary_candidate_mesh(records: Sequence[_RefinementCell], /) -> TMesh2D:
    ordered = tuple(sorted(records, key=_record_order))
    cells = tuple(
        TCell2D(index, record.bounds, level=record.level)
        for index, record in enumerate(ordered)
    )
    return TMesh2D(cells)


def _extension_graph_edges(
    horizontal: Sequence[TJunctionExtension2D],
    vertical: Sequence[TJunctionExtension2D],
    /,
) -> tuple[tuple[int, int, Fraction, Fraction], ...]:
    """Build graph edges for all intersections of closed extension segments."""
    crossings = []
    for horizontal_index, first in enumerate(horizontal):
        hu0, hu1, hv, _ = first.segment
        for vertical_index, second in enumerate(vertical):
            vu, _, vv0, vv1 = second.segment
            if hu0 <= vu <= hu1 and vv0 <= hv <= vv1:
                crossings.append((horizontal_index, vertical_index, vu, hv))
    return tuple(crossings)


def _face_extension_bounds(
    mesh: TMesh2D, extension: TJunctionExtension2D, /
) -> tuple[Literal["u", "v"], Fraction, Fraction, Fraction]:
    """Return extension axis, fixed coordinate, and face-extension interval."""
    u, v = mesh.vertices[extension.junction_vertex_id].parameter
    first, second, third, fourth = extension.segment
    if extension.axis == "u":
        if extension.missing_direction == "right":
            return "u", v, u, second
        return "u", v, first, u
    if extension.missing_direction == "up":
        return "v", u, v, fourth
    return "v", u, third, v


def _coupled_extension_graph(
    source: TMesh2D, target: TMesh2D, /
) -> tuple[tuple[int, int], ...]:
    """Return target-extension crossings and source-face-extension loops."""
    target_horizontal = tuple(
        extension for extension in target.t_junction_extensions if extension.axis == "u"
    )
    target_vertical = tuple(
        extension for extension in target.t_junction_extensions if extension.axis == "v"
    )
    edges = [
        (
            target_horizontal[first].junction_vertex_id,
            target_vertical[second].junction_vertex_id,
        )
        for first, second, _, _ in _extension_graph_edges(
            target_horizontal, target_vertical
        )
    ]
    source_by_axis = {
        axis: tuple(
            extension
            for extension in source.t_junction_extensions
            if extension.axis == axis
        )
        for axis in ("u", "v")
    }
    for extension in target.t_junction_extensions:
        axis, fixed, lower, upper = _face_extension_bounds(target, extension)
        endpoint = upper if extension.missing_direction in ("right", "up") else lower
        for source_extension in source_by_axis[axis]:
            source_axis, source_fixed, source_lower, source_upper = (
                _face_extension_bounds(source, source_extension)
            )
            if (
                source_axis == axis
                and source_fixed == fixed
                and source_lower < endpoint < source_upper
            ):
                vertex_id = extension.junction_vertex_id
                edges.append((vertex_id, vertex_id))
    return tuple(edges)


def _coupled_extension_weight(source: TMesh2D, target: TMesh2D, /) -> int:
    return len(_coupled_extension_graph(source, target))


def _atomic_occupancy(
    mesh: TMesh2D,
    u_coordinates: Sequence[Fraction],
    v_coordinates: Sequence[Fraction],
    /,
) -> frozenset[tuple[int, Fraction, Fraction, Fraction]]:
    u_breaks = tuple(sorted(set(u_coordinates)))
    v_breaks = tuple(sorted(set(v_coordinates)))
    occupied: set[tuple[int, Fraction, Fraction, Fraction]] = set()
    for fixed in v_breaks:
        for lower, upper in zip(u_breaks[:-1], u_breaks[1:], strict=True):
            if any(
                (cell.parameter_bounds[2] == fixed or cell.parameter_bounds[3] == fixed)
                and cell.parameter_bounds[0] <= lower
                and upper <= cell.parameter_bounds[1]
                for cell in mesh.cells
            ):
                occupied.add((0, fixed, lower, upper))
    for fixed in u_breaks:
        for lower, upper in zip(v_breaks[:-1], v_breaks[1:], strict=True):
            if any(
                (cell.parameter_bounds[0] == fixed or cell.parameter_bounds[1] == fixed)
                and cell.parameter_bounds[2] <= lower
                and upper <= cell.parameter_bounds[3]
                for cell in mesh.cells
            ):
                occupied.add((1, fixed, lower, upper))
    return frozenset(occupied)


def _four_children(parent: _RefinementCell, /) -> tuple[_RefinementCell, ...]:
    u0, u1, v0, v1 = parent.bounds
    u_mid = (u0 + u1) / 2
    v_mid = (v0 + v1) / 2
    bounds = (
        (u0, u_mid, v0, v_mid),
        (u_mid, u1, v0, v_mid),
        (u0, u_mid, v_mid, v1),
        (u_mid, u1, v_mid, v1),
    )
    return tuple(
        _RefinementCell(
            parent.root_id,
            parent.address + (index,),
            child_bounds,
            parent.level + 1,
            parent.key,
            None,
        )
        for index, child_bounds in enumerate(bounds)
    )


def _bisect_refinement_cell(
    parent: _RefinementCell,
    axis: int,
    coordinate: Fraction,
    /,
) -> tuple[_RefinementCell, _RefinementCell]:
    u0, u1, v0, v1 = parent.bounds
    if axis == 0:
        bounds = ((u0, coordinate, v0, v1), (coordinate, u1, v0, v1))
        address_codes = (4, 5)
    else:
        bounds = ((u0, u1, v0, coordinate), (u0, u1, coordinate, v1))
        address_codes = (6, 7)
    children = tuple(
        _RefinementCell(
            parent.root_id,
            parent.address + (address_code,),
            child_bounds,
            parent.level + 1,
            parent.key,
            None,
        )
        for child_bounds, address_code in zip(bounds, address_codes, strict=True)
    )
    return children[0], children[1]


def _legal_insertion_edges(
    records: Sequence[_RefinementCell],
    target: TMesh2D,
    weighted_junction_ids: set[int],
    u_coordinates: set[Fraction],
    v_coordinates: set[Fraction],
    /,
) -> tuple[tuple[tuple[int, tuple[int, ...]], int, Fraction], ...]:
    operations: set[tuple[tuple[int, tuple[int, ...]], int, Fraction]] = set()
    for extension in target.t_junction_extensions:
        if extension.junction_vertex_id not in weighted_junction_ids:
            continue
        axis, fixed, lower, upper = _face_extension_bounds(target, extension)
        if axis == "u":
            if fixed not in v_coordinates:
                continue
            for record in records:
                u0, u1, v0, v1 = record.bounds
                if v0 < fixed < v1 and max(u0, lower) < min(u1, upper):
                    operations.add((record.key, 1, fixed))
        else:
            if fixed not in u_coordinates:
                continue
            for record in records:
                u0, u1, v0, v1 = record.bounds
                if u0 < fixed < u1 and max(v0, lower) < min(v1, upper):
                    operations.add((record.key, 0, fixed))
    current_by_key = {record.key: record for record in records}
    return tuple(
        sorted(
            operations,
            key=lambda item: (
                item[1],
                item[2],
                current_by_key[item[0]].bounds,
                item[0][0],
                item[0][1],
            ),
        )
    )


def _candidate_records(
    records: Sequence[_RefinementCell],
    operation: tuple[tuple[int, tuple[int, ...]], int, Fraction],
    /,
) -> tuple[_RefinementCell, ...]:
    parent_key, axis, coordinate = operation
    result = []
    for record in records:
        if record.key == parent_key:
            result.extend(_bisect_refinement_cell(record, axis, coordinate))
        else:
            result.append(record)
    return tuple(sorted(result, key=_record_order))


def _exact_basis_transfer_routes(
    source: TMesh2D, target: TMesh2D, /
) -> tuple[tuple[int, int, Fraction], ...]:
    """Resolve nested basis routes from exact local Bernstein identities."""
    from ..linalg._exact import (
        ExactInconsistentSystemError,
        ExactRankDeficientSystemError,
        solve_exact_sparse,
    )

    source_supports = _ordered_anchor_supports(source)
    target_supports = _ordered_anchor_supports(target)
    source_anchor_by_row = {
        source.anchor_row(anchor.anchor_id): anchor for anchor in source.anchors
    }
    target_anchor_by_row = {
        target.anchor_row(anchor.anchor_id): anchor for anchor in target.anchors
    }
    piece_data: list[
        tuple[dict[int, tuple[Fraction, ...]], dict[int, tuple[Fraction, ...]]]
    ] = []
    target_rows_by_source: dict[int, set[int]] = {}
    for target_cell in target.cells:
        for bounds in _common_extraction_piece_bounds(
            (source, target), target_cell.parameter_bounds
        ):
            piece = _ExtractionPiece(len(piece_data), target_cell.cell_id, bounds)
            source_piece = _cell_extraction(
                source,
                piece,
                _cell_support_rows(source, bounds, source_supports),
            )
            target_piece = _cell_extraction(
                target,
                piece,
                _cell_support_rows(target, bounds, target_supports),
            )
            source_local = dict(
                zip(
                    source_piece.coefficient_row_ids,
                    source_piece.exact_extraction_operator,
                    strict=True,
                )
            )
            target_local = dict(
                zip(
                    target_piece.coefficient_row_ids,
                    target_piece.exact_extraction_operator,
                    strict=True,
                )
            )
            piece_data.append((source_local, target_local))
            for source_row in source_local:
                target_rows_by_source.setdefault(source_row, set()).update(target_local)

    source_groups: dict[tuple[int, ...], list[int]] = {}
    for source_row in source_anchor_by_row:
        target_rows = tuple(
            sorted(
                target_rows_by_source.get(source_row, ()),
                key=lambda row: target_anchor_by_row[row].anchor_id,
            )
        )
        if not target_rows:
            raise ValueError(
                "Exact target basis has no local support for a source anchor."
            )
        source_groups.setdefault(target_rows, []).append(source_row)

    zero_piece = (Fraction(0),) * 16
    routes: dict[tuple[int, int], Fraction] = {}
    for target_rows, unsorted_source_rows in sorted(
        source_groups.items(), key=lambda item: item[0]
    ):
        source_rows = tuple(
            sorted(
                unsorted_source_rows,
                key=lambda row: source_anchor_by_row[row].anchor_id,
            )
        )
        target_columns = {row: column for column, row in enumerate(target_rows)}
        matrix: list[dict[int, Fraction]] = []
        right: list[tuple[Fraction, ...]] = []
        equation_ids: list[int] = []
        for piece_index, (source_local, target_local) in enumerate(piece_data):
            active_target_rows = tuple(row for row in target_rows if row in target_local)
            for coefficient in range(16):
                operator_row = {
                    target_columns[row]: target_local[row][coefficient]
                    for row in active_target_rows
                    if target_local[row][coefficient]
                }
                right_row = tuple(
                    source_local.get(source_row, zero_piece)[coefficient]
                    for source_row in source_rows
                )
                if operator_row or any(right_row):
                    matrix.append(operator_row)
                    right.append(right_row)
                    equation_ids.append(piece_index * 16 + coefficient)
        try:
            solutions, evidence = solve_exact_sparse(
                matrix,
                right,
                len(target_rows),
                row_ids=equation_ids,
            )
        except (ExactInconsistentSystemError, ExactRankDeficientSystemError) as error:
            source_anchor_ids = tuple(
                source_anchor_by_row[row].anchor_id for row in source_rows
            )
            error.add_note(
                "Exact ASTS basis inclusion source anchor IDs: "
                f"{source_anchor_ids}; target anchor rows: {target_rows}."
            )
            raise
        expected_pivots = tuple(range(len(target_rows)))
        if evidence.rank != len(target_rows) or evidence.pivot_columns != expected_pivots:
            raise ValueError(
                "Exact local transfer factorization did not establish full target rank."
            )
        for target_column, target_row in enumerate(target_rows):
            for source_column, source_row in enumerate(source_rows):
                coefficient = solutions[target_column][source_column]
                if coefficient:
                    routes[(target_row, source_row)] = coefficient
    return tuple(
        (target_row, source_row, coefficient)
        for (target_row, source_row), coefficient in sorted(routes.items())
        if coefficient != 0
    )


def _verify_local_basis_inclusion(
    source: TMesh2D,
    target: TMesh2D,
    routes: Sequence[tuple[int, int, Fraction]],
    basis_certificate: ASTSCertificate,
    /,
) -> LocalExtractedBernsteinRealization:
    route_map = {
        (target_row, source_row): coefficient
        for target_row, source_row, coefficient in routes
    }
    source_supports = _ordered_anchor_supports(source)
    target_supports = _ordered_anchor_supports(target)
    target_realization = target.realize(
        tolerance=basis_certificate.tolerance,
        certificate=basis_certificate,
    )
    zero_piece = (Fraction(0),) * 16
    for target_cell in target.cells:
        for bounds in _common_extraction_piece_bounds(
            (source, target), target_cell.parameter_bounds
        ):
            piece = _ExtractionPiece(0, target_cell.cell_id, bounds)
            source_piece = _cell_extraction(
                source,
                piece,
                _cell_support_rows(source, bounds, source_supports),
            )
            target_piece = _cell_extraction(
                target,
                piece,
                _cell_support_rows(target, bounds, target_supports),
            )
            source_local = dict(
                zip(
                    source_piece.coefficient_row_ids,
                    source_piece.exact_extraction_operator,
                    strict=True,
                )
            )
            target_local = dict(
                zip(
                    target_piece.coefficient_row_ids,
                    target_piece.exact_extraction_operator,
                    strict=True,
                )
            )
            target_rows = tuple(target_local)
            source_candidates = set(source_local)
            source_candidates.update(
                source_row
                for target_row in target_rows
                for routed_target, source_row in route_map
                if routed_target == target_row
            )
            for source_row in source_candidates:
                coarse = source_local.get(source_row, zero_piece)
                fine = tuple(
                    sum(
                        (
                            route_map.get((target_row, source_row), Fraction(0))
                            * target_local[target_row][column]
                            for target_row in target_rows
                        ),
                        Fraction(0),
                    )
                    for column in range(16)
                )
                if coarse != fine:
                    raise ValueError(
                        "Exact sparse prolongation failed complete local basis inclusion."
                    )
    return target_realization


def analysis_suitable_refinement_closure(
    mesh: TMesh2D,
    marked_cell_ids: Sequence[int],
    /,
    *,
    tolerance: float = _DEFAULT_TOLERANCE,
) -> ASTSRefinement:
    """Apply Scott's coupled-extension-graph local refinement atomically."""
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
    source_by_id = {cell.cell_id: cell for cell in mesh.cells}
    unknown = sorted(set(marked) - set(source_by_id))
    if unknown:
        raise KeyError(f"Unknown marked T-mesh cells: {unknown}.")
    source_records = {
        cell_id: _RefinementCell(
            cell_id,
            cell.refinement_address,
            cell.parameter_bounds,
            cell.level,
            None,
            cell,
        )
        for cell_id, cell in source_by_id.items()
    }
    all_records = {record.key: record for record in source_records.values()}
    active_records: list[_RefinementCell] = []
    marked_u: set[Fraction] = set()
    marked_v: set[Fraction] = set()
    for cell in mesh.cells:
        record = source_records[cell.cell_id]
        if cell.cell_id not in marked:
            active_records.append(record)
            continue
        u0, u1, v0, v1 = cell.parameter_bounds
        marked_u.add((u0 + u1) / 2)
        marked_v.add((v0 + v1) / 2)
        children = _four_children(record)
        active_records.extend(children)
        all_records.update((child.key, child) for child in children)
    u_coordinates = tuple(sorted(set(mesh.axis_breaks[0]) | marked_u))
    v_coordinates = tuple(sorted(set(mesh.axis_breaks[1]) | marked_v))
    u_coordinate_set = set(u_coordinates)
    v_coordinate_set = set(v_coordinates)
    candidate_mesh = _temporary_candidate_mesh(active_records)
    occupied = _atomic_occupancy(candidate_mesh, u_coordinates, v_coordinates)
    source_occupied = _atomic_occupancy(mesh, u_coordinates, v_coordinates)
    if not source_occupied.issubset(occupied):
        raise RuntimeError(
            "Initial four-way refinement removed an occupied atomic skeleton."
        )
    closure_roots: set[int] = set()
    graph_edges = _coupled_extension_graph(mesh, candidate_mesh)
    while graph_edges:
        weighted_junction_ids = {vertex for edge in graph_edges for vertex in edge}
        current_by_key = {record.key: record for record in active_records}
        operations = set(
            _legal_insertion_edges(
                active_records,
                candidate_mesh,
                weighted_junction_ids,
                u_coordinate_set,
                v_coordinate_set,
            )
        )
        ordered_operations = tuple(
            sorted(
                operations,
                key=lambda item: (
                    item[1],
                    item[2],
                    current_by_key[item[0]].bounds,
                    item[0][0],
                    item[0][1],
                ),
            )
        )
        candidates = []
        for operation in ordered_operations:
            next_records = _candidate_records(active_records, operation)
            next_mesh = _temporary_candidate_mesh(next_records)
            next_occupied = _atomic_occupancy(next_mesh, u_coordinates, v_coordinates)
            if not occupied.issubset(next_occupied) or next_occupied == occupied:
                continue
            next_weight = _coupled_extension_weight(mesh, next_mesh)
            parent = current_by_key[operation[0]]
            order = (
                next_weight,
                operation[1],
                operation[2],
                parent.bounds,
                parent.root_id,
                parent.address,
            )
            candidates.append(
                (order, next_records, next_mesh, next_occupied, parent, operation)
            )
        if not candidates:
            raise RuntimeError(
                "Scott closure has no legal insertion that grows the finite atomic skeleton."
            )
        _, active_records, candidate_mesh, occupied, parent, operation = min(
            candidates, key=lambda candidate: candidate[0]
        )
        children = _bisect_refinement_cell(parent, operation[1], operation[2])
        all_records.update((child.key, child) for child in children)
        closure_roots.add(parent.root_id)
        graph_edges = _coupled_extension_graph(mesh, candidate_mesh)

    created_records = tuple(
        sorted(
            (record for record in all_records.values() if record.original_cell is None),
            key=_record_order,
        )
    )
    maximum_id = int(np.iinfo(np.int64).max)
    next_cell_id = max(mesh.reserved_cell_ids, default=-1) + 1
    if next_cell_id + len(created_records) - 1 > maximum_id:
        raise OverflowError("T-mesh cell identity allocation exceeds int64.")
    new_cell_ids = {
        record.key: next_cell_id + index for index, record in enumerate(created_records)
    }
    final_cells: list[TCell2D] = []
    for record in sorted(active_records, key=_record_order):
        if record.original_cell is not None:
            cell_id = record.original_cell.cell_id
            parent_id = record.original_cell.parent_id
            address = record.original_cell.refinement_address
        else:
            cell_id = new_cell_ids[record.key]
            if record.parent_key is None:
                raise RuntimeError("A refined T-mesh cell is missing its parent address.")
            parent_record = all_records[record.parent_key]
            parent_id = (
                parent_record.original_cell.cell_id
                if parent_record.original_cell is not None
                else new_cell_ids[parent_record.key]
            )
            address = record.address
        final_cells.append(
            TCell2D(
                cell_id,
                record.bounds,
                level=record.level,
                parent_id=parent_id,
                refinement_address=address,
            )
        )
    lineage_children: dict[int, list[_RefinementCell]] = {}
    for record in created_records:
        if record.parent_key is None:
            raise RuntimeError("A created T-mesh cell is missing its parent address.")
        parent_record = all_records[record.parent_key]
        parent_id = (
            parent_record.original_cell.cell_id
            if parent_record.original_cell is not None
            else new_cell_ids[parent_record.key]
        )
        lineage_children.setdefault(parent_id, []).append(record)
    lineage_events = tuple(
        (
            parent_id,
            tuple(
                new_cell_ids[child.key] for child in sorted(children, key=_record_order)
            ),
        )
        for parent_id, children in sorted(lineage_children.items())
    )
    cumulative_lineage = (*mesh.cell_lineage, *lineage_events)
    reserved_cell_ids = tuple(
        sorted(set(mesh.reserved_cell_ids) | set(new_cell_ids.values()))
    )
    target_structure = TMesh2D(
        final_cells,
        cell_lineage=cumulative_lineage,
        reserved_cell_ids=reserved_cell_ids,
        reserved_anchor_ids=mesh.reserved_anchor_ids,
    )
    source_anchor_ids = {
        (anchor.parameter, anchor.local_knots.u, anchor.local_knots.v): anchor.anchor_id
        for anchor in mesh.anchors
    }
    target_addresses = tuple(
        sorted(
            (anchor.parameter, anchor.local_knots.u, anchor.local_knots.v)
            for anchor in target_structure.anchors
        )
    )
    new_anchor_addresses = tuple(
        address for address in target_addresses if address not in source_anchor_ids
    )
    next_anchor_id = max(mesh.reserved_anchor_ids, default=-1) + 1
    anchor_ids = {
        address: source_anchor_ids[address]
        for address in target_addresses
        if address in source_anchor_ids
    }
    anchor_ids.update(
        (address, next_anchor_id + index)
        for index, address in enumerate(new_anchor_addresses)
    )
    refined_anchors = tuple(
        TAnchor2D(
            anchor_ids[(anchor.parameter, anchor.local_knots.u, anchor.local_knots.v)],
            anchor.parameter,
            anchor.local_knots,
        )
        for anchor in target_structure.anchors
    )
    reserved_anchor_ids = tuple(
        sorted(set(mesh.reserved_anchor_ids) | set(anchor_ids.values()))
    )
    refined = TMesh2D(
        final_cells,
        refined_anchors,
        cell_lineage=cumulative_lineage,
        reserved_cell_ids=reserved_cell_ids,
        reserved_anchor_ids=reserved_anchor_ids,
    )
    target_certificate = certify_asts(refined, tolerance=tolerance_)
    if not target_certificate.passed:
        reasons = ", ".join(target_certificate.failure_reasons)
        raise RuntimeError(
            f"Local analysis-suitable refinement failed certification: {reasons}."
        )
    routes = _exact_basis_transfer_routes(mesh, refined)
    _verify_local_basis_inclusion(mesh, refined, routes, target_certificate)
    transfer = ASTSTransferPlan(
        mesh.mesh_id,
        refined.mesh_id,
        mesh.coefficient_count,
        refined.coefficient_count,
        tuple(target for target, _, _ in routes),
        tuple(source for _, source, _ in routes),
        tuple(coefficient for _, _, coefficient in routes),
    )
    return ASTSRefinement(
        mesh,
        refined,
        marked,
        tuple(sorted(closure_roots)),
        lineage_events,
        transfer,
        target_certificate,
        _proved_inclusion=_ProvedRefinementInclusion(
            source_certificate,
            target_certificate.certificate_id,
            transfer.plan_id,
        ),
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

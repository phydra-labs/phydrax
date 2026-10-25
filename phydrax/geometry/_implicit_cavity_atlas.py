# Copyright © 2026 PHYDRA, Inc. All rights reserved.
"""Exact, fail-closed paired charts for one chronological projected cavity edit.

The caller supplies a chart and a shared anchor, not a heuristic acceptance
threshold. Unsupported non-graph patches and cavities without that common
strict kernel are refused before publication. All intersections belong to the
canonical planar/polyhedral owners; no physical-domain sliver is filled.
"""
from __future__ import annotations

import json
import weakref
from collections.abc import Sequence
from dataclasses import dataclass
from enum import StrEnum
from fractions import Fraction
from itertools import combinations
from typing import Never

import numpy as np

from .._fingerprint import canonical_fingerprint
from .._issued_proofs import IssuedProofs
from .._meshcore import MeshcoreError
from ..discretization._coordinate_budget import (
    CoordinateEnclosureBudget,
    CoordinateEnclosureResourceError,
)
from ._exact_polyhedral_geometry import (
    ExactOverlapRefusal,
    ExactPolyhedralRefusal,
    star_determinant,
    tetrahedra_disjoint,
    tetrahedron_overlap,
    TetrahedronPlanePreparation,
    triangulate_loop,
)
from ._planar_primitives import (
    intersection_polygon,
    plane_key,
    project,
    reserve_fraction_work,
    signed_measure,
    turn,
)


Point = tuple[Fraction, Fraction, Fraction]
Tet = tuple[Point, Point, Point, Point]
Box = tuple[Point, Point]
Face = tuple[Point, ...]
PlanarPoint = tuple[Fraction, ...]
PlanarTriangle = tuple[PlanarPoint, ...]
Patch = tuple[tuple[Face, str], ...]


class CavityAtlasStatus(StrEnum):
    INVALID_CAVITY = "invalid_cavity"
    INVALID_ANCESTRY = "invalid_ancestry"
    NOT_A_GRAPH = "not_a_graph"
    RIM_MISMATCH = "rim_mismatch"
    NO_COMMON_KERNEL = "no_common_kernel"
    COVERAGE_GAP = "coverage_gap"
    DOUBLE_COVERAGE = "double_coverage"


class CavityAtlasRefusal(ValueError):
    """Geometric refusal; canonical resource exceptions propagate unchanged."""

    def __init__(self, status: CavityAtlasStatus, message: str) -> None:
        super().__init__(message)
        self.status = status

    @property
    def recoverable(self) -> bool:
        """Only a different explicitly supplied chart/kernel may cure these."""
        return self.status in (
            CavityAtlasStatus.NOT_A_GRAPH, CavityAtlasStatus.NO_COMMON_KERNEL,
        )


@dataclass(frozen=True, slots=True)
class CavitySnapshot:
    """Actual cavity rows; face labels and rim ancestry are source-owned IDs.

    ``rim_ancestry`` contains ``(vertex_a, vertex_b, ancestry_id)`` for every
    boundary edge of the constrained patch, exactly once. Vertex indices are
    local to this snapshot; ``cell_ids`` retain original native row identity.
    """

    vertices: tuple[Point, ...]
    cells: tuple[tuple[int, int, int, int], ...]
    constrained_faces: tuple[tuple[int, int, int], ...]
    source_labels: tuple[str, ...]
    cell_ids: tuple[int, ...]
    rim_ancestry: tuple[tuple[int, int, str], ...]


@dataclass(frozen=True, slots=True)
class CavityFacetParent:
    """One actual oriented boundary facet, identified by its owning cell."""

    cell_id: int
    vertices: Face
    orientation: int


@dataclass(frozen=True, slots=True, weakref_slot=True)
class ImplicitCavityAtlas:
    old_tets: tuple[Tet, ...]
    new_tets: tuple[Tet, ...]
    chart_id: str
    source_cell_ids: tuple[int, ...]
    target_cell_ids: tuple[int, ...]
    source_id: str
    discovery_id: str
    chronology_index: int
    predecessor_id: str
    chart_axes: tuple[int, int]
    anchor: Point
    chart_direction: Point | None
    source_boundary_parents: tuple[CavityFacetParent, ...]
    target_boundary_parents: tuple[CavityFacetParent, ...]
    # Exact closed chart boxes locate inverse material traces without clipping.
    old_bounds: tuple[Box, ...]
    new_bounds: tuple[Box, ...]
    old_hull: Box
    new_hull: Box


def _refuse(status: CavityAtlasStatus, message: str) -> Never:
    raise CavityAtlasRefusal(status, message)


def _key(points: Sequence[Point]) -> Face:
    return tuple(sorted(points))


def _point(values: Sequence[Fraction]) -> Point:
    if len(values) != 3:
        _refuse(CavityAtlasStatus.INVALID_CAVITY, "point requires three exact coordinates")
    return values[0], values[1], values[2]


def _tet(values: Sequence[Point]) -> Tet:
    if len(values) != 4:
        _refuse(CavityAtlasStatus.INVALID_CAVITY, "tetrahedron requires four exact points")
    return values[0], values[1], values[2], values[3]


def _snapshot(snapshot: CavitySnapshot, budget: CoordinateEnclosureBudget) -> tuple[
    tuple[Tet, ...], Patch, set[Face], dict[Face, str],
]:
    budget.reserve(sum(len(p) for p in snapshot.vertices) + 16 * len(snapshot.cells),
                   256 * (len(snapshot.vertices) + len(snapshot.cells)))
    try:
        vertices = tuple(_point(tuple(Fraction(x) for x in p)) for p in snapshot.vertices)
    except (ValueError, TypeError, OverflowError) as error:
        raise CavityAtlasRefusal(CavityAtlasStatus.INVALID_CAVITY, "nonfinite or invalid cavity coordinates") from error
    if vertices:
        reserve_fraction_work(vertices, 12 * len(vertices), 3 * len(vertices), 1, budget=budget)
    if not vertices or any(len(p) != 3 for p in vertices) or len(set(vertices)) != len(vertices):
        _refuse(CavityAtlasStatus.INVALID_CAVITY, "invalid or duplicate cavity vertices")
    if not snapshot.cells or len(snapshot.cell_ids) != len(snapshot.cells) or len(set(snapshot.cell_ids)) != len(snapshot.cell_ids):
        _refuse(CavityAtlasStatus.INVALID_CAVITY, "missing or duplicate cavity cell IDs")
    faces = {}
    tets = []
    for row in snapshot.cells:
        if len(row) != 4 or len(set(row)) != 4 or any(i < 0 or i >= len(vertices) for i in row):
            _refuse(CavityAtlasStatus.INVALID_CAVITY, "invalid cavity connectivity")
        tet = _tet(tuple(vertices[i] for i in row))
        if star_determinant(tet, budget=budget) <= 0:
            _refuse(CavityAtlasStatus.INVALID_CAVITY, "nonpositive actual cavity tetrahedron")
        tets.append(tet)
        for face in combinations(tet, 3):
            key = _key(face)
            faces[key] = faces.get(key, 0) + 1
    if any(count > 2 for count in faces.values()):
        _refuse(CavityAtlasStatus.INVALID_CAVITY, "nonmanifold cavity face")
    boundary = {face for face, count in faces.items() if count == 1}
    if len(snapshot.source_labels) != len(snapshot.constrained_faces) or not snapshot.constrained_faces:
        _refuse(CavityAtlasStatus.INVALID_ANCESTRY, "missing constrained patch labels")
    patch = []
    for row, label in zip(snapshot.constrained_faces, snapshot.source_labels, strict=True):
        if len(row) != 3 or len(set(row)) != 3 or any(i < 0 or i >= len(vertices) for i in row) or not label:
            _refuse(CavityAtlasStatus.INVALID_ANCESTRY, "invalid constrained patch row")
        face = _key(tuple(vertices[i] for i in row))
        if face not in boundary or any(face == old for old, _ in patch):
            _refuse(CavityAtlasStatus.INVALID_CAVITY, "duplicate or nonboundary constrained face")
        patch.append((face, label))
    edges = {}
    for face, _ in patch:
        for edge in combinations(face, 2):
            key = _key(edge)
            edges[key] = edges.get(key, 0) + 1
    if any(n > 2 for n in edges.values()):
        _refuse(CavityAtlasStatus.NOT_A_GRAPH, "nonmanifold patch edge")
    rim = {edge for edge, n in edges.items() if n == 1}
    ancestry = {}
    for a, b, identity in snapshot.rim_ancestry:
        if a == b or min(a, b) < 0 or max(a, b) >= len(vertices) or not identity:
            _refuse(CavityAtlasStatus.INVALID_ANCESTRY, "invalid rim ancestry")
        edge = _key((vertices[a], vertices[b]))
        if edge in ancestry:
            _refuse(CavityAtlasStatus.INVALID_ANCESTRY, "duplicate rim ancestry")
        ancestry[edge] = identity
    if set(ancestry) != rim or not rim:
        _refuse(CavityAtlasStatus.INVALID_ANCESTRY, "rim ancestry must cover the entire patch rim")
    return tuple(tets), tuple(patch), boundary - {f for f, _ in patch}, ancestry


def _lift(point: PlanarPoint, face: Face, planar: PlanarTriangle,
          budget: CoordinateEnclosureBudget) -> Point:
    denominator = turn(planar[0], planar[1], planar[2], budget=budget)
    weights = (turn(point, planar[1], planar[2], budget=budget) / denominator,
               turn(planar[0], point, planar[2], budget=budget) / denominator)
    reserve_fraction_work((*face, weights), 24, 3, 16, budget=budget)
    return _point(tuple(weights[0] * face[0][i] + weights[1] * face[1][i]
                        + (1 - sum(weights)) * face[2][i] for i in range(3)))


def _chart_project(face: Face, axes: tuple[int, int], direction: Point | None,
                   budget: CoordinateEnclosureBudget) -> PlanarTriangle:
    if direction is None:
        return project(face, axes, budget=budget)
    dropped = next(axis for axis in range(3) if axis not in axes)
    reserve_fraction_work((*face, direction), 6 * len(face), 2 * len(face), 8, budget=budget)
    return tuple(tuple(direction[dropped] * point[axis] - direction[axis] * point[dropped]
                       for axis in axes) for point in face)


def _graph(patch: Patch, axes: tuple[int, int], direction: Point | None,
           budget: CoordinateEnclosureBudget) -> tuple[PlanarTriangle, ...]:
    projected = tuple(_chart_project(face, axes, direction, budget) for face, _ in patch)
    for triangle in projected:
        if signed_measure(triangle, budget=budget) == 0:
            _refuse(CavityAtlasStatus.NOT_A_GRAPH, "vertical or degenerate patch triangle")
    for i, j in combinations(range(len(patch)), 2):
        polygon = intersection_polygon(projected[i], projected[j], budget=budget)
        if signed_measure(polygon, budget=budget) != 0:
            _refuse(CavityAtlasStatus.NOT_A_GRAPH, "patch overlaps in the supplied planar chart")
        for p in polygon:
            if _lift(p, patch[i][0], projected[i], budget) != _lift(p, patch[j][0], projected[j], budget):
                _refuse(CavityAtlasStatus.NOT_A_GRAPH, "multiple patch heights at one chart coordinate")
    return projected


def _overlap(a: Tet, b: Tet, budget: CoordinateEnclosureBudget,
             preparation: TetrahedronPlanePreparation) -> Fraction:
    # Exact closed boxes can exclude positive volume, including face contact.
    # Keep the canonical overlap owner's prices for every surviving pair.
    if tetrahedra_disjoint(a, b):
        return Fraction(0)
    result = tetrahedron_overlap(a, b, budget=budget, preparation=preparation)
    if isinstance(result, ExactOverlapRefusal):
        raise ExactPolyhedralRefusal(result)
    return result.volume


def _oriented_cavity_boundary(
    actual: Sequence[Tet],
    cell_ids: Sequence[int],
    budget: CoordinateEnclosureBudget,
) -> dict[Face, CavityFacetParent]:
    """Authenticate the signed boundary of the actual positive tetrahedra."""
    if len(actual) != len(cell_ids) or len(set(cell_ids)) != len(cell_ids):
        _refuse(CavityAtlasStatus.INVALID_ANCESTRY, "actual cavity cell identities are incomplete")
    incidences: dict[Face, list[CavityFacetParent]] = {}
    for cell_id, tet in zip(cell_ids, actual, strict=True):
        if star_determinant(tet, budget=budget) <= 0:
            _refuse(CavityAtlasStatus.INVALID_CAVITY, "actual cavity requires positive tetrahedra")
        for indices in ((1, 2, 3), (0, 3, 2), (0, 1, 3), (0, 2, 1)):
            face = tuple(tet[index] for index in indices)
            reserve_fraction_work(face, 6, 0, 2, budget=budget)
            canonical = _key(face)
            inversions = sum(
                face[first] > face[second]
                for first, second in ((0, 1), (0, 2), (1, 2))
            )
            parent = CavityFacetParent(int(cell_id), canonical, -1 if inversions % 2 else 1)
            budget.reserve(1, 512)
            incidences.setdefault(canonical, []).append(parent)
    boundary = {}
    for face, parents in incidences.items():
        if len(parents) == 1:
            boundary[face] = parents[0]
        elif len(parents) != 2 or sum(parent.orientation for parent in parents) != 0:
            _refuse(CavityAtlasStatus.INVALID_CAVITY, "actual interior facet incidences must cancel with opposite signs")
    return boundary


def _add_planar_loop(
    chain: dict[tuple[int, Fraction, Fraction], dict[Fraction, int]],
    loop: PlanarTriangle,
    weight: int,
    budget: CoordinateEnclosureBudget,
) -> None:
    """Add an oriented triangle boundary to an exact planar 1-chain.

    Each segment is keyed by its exact supporting line and parameterized along
    it, so collinear subdivisions cancel exactly. A line's 1-chain vanishes
    iff every endpoint jump of its signed interval indicators vanishes.
    """
    for start, stop in zip(loop, (*loop[1:], loop[0]), strict=True):
        # Two differences, slope, intercept, and two endpoint comparisons.
        reserve_fraction_work((start, stop), 8, 3, 6, budget=budget)
        budget.reserve(2, 1024)
        dx, dy = stop[0] - start[0], stop[1] - start[1]
        if dx:
            slope = dy / dx
            key = (0, slope, start[1] - slope * start[0])
            first, last, sign = (
                (start[0], stop[0], weight) if dx > 0 else (stop[0], start[0], -weight)
            )
        else:
            key = (1, start[0], Fraction(0))
            first, last, sign = (
                (start[1], stop[1], weight) if dy > 0 else (stop[1], start[1], -weight)
            )
        jumps = chain.setdefault(key, {})
        jumps[first] = jumps.get(first, 0) + sign
        jumps[last] = jumps.get(last, 0) - sign


def _cover_coned_boundary(
    actual: Sequence[Tet],
    cell_ids: Sequence[int],
    virtual: Sequence[Tet],
    parents: Sequence[CavityFacetParent],
    anchor: Point,
    budget: CoordinateEnclosureBudget,
) -> None:
    """Prove equality of compact positive volume chains from their boundary.

    Actual tetrahedra are independently positive and disjoint. Their signed
    interior faces cancel. Each virtual tetrahedron is a positive cone from
    the same anchor over a child of one explicitly identified actual outward
    facet. Every child is exactly coplanar with its parent and carries the
    parent's outward orientation, and the planar boundary 1-chain of the
    children minus that of the parent cancels exactly. A compact planar 2-chain
    without boundary is zero, so the children's multiplicity equals the
    parent's indicator almost everywhere: containment, pairwise interior
    disjointness and a complete partition of every parent, including each
    unchanged face, establish equality of the *oriented* boundary 2-chains.

    For a closed boundary chain S, the cone identity gives boundary(cone S)=S;
    the radial faces cancel because the children's boundary is that of S.
    Thus the compact 3-chain virtual-actual has zero boundary in R3 and is zero.
    Positive virtual coefficients then give multiplicity one almost everywhere:
    there can be neither a gap nor positive-volume overlap. Global embedding
    still checks full chart interfaces when the virtual meshes are published.
    """
    if not actual or not virtual or len(virtual) != len(parents):
        _refuse(CavityAtlasStatus.INVALID_ANCESTRY, "every virtual cone requires its actual parent facet")
    preparation = TetrahedronPlanePreparation(budget)
    for first, second in combinations(actual, 2):
        if _overlap(first, second, budget, preparation):
            _refuse(CavityAtlasStatus.DOUBLE_COVERAGE, "actual cavity tetrahedra overlap")
    boundary = _oriented_cavity_boundary(actual, cell_ids, budget)
    children: dict[Face, list[Face]] = {face: [] for face in boundary}
    for tet, parent in zip(virtual, parents, strict=True):
        budget.reserve(4, 512)
        if boundary.get(parent.vertices) != parent:
            _refuse(CavityAtlasStatus.INVALID_ANCESTRY, "virtual cone parent must match an actual oriented boundary facet")
        if tet[0] != anchor or star_determinant(tet, budget=budget) <= 0:
            _refuse(CavityAtlasStatus.INVALID_CAVITY, "virtual cells must be positive cones from the common anchor")
        face = tet[1:]
        for point in face:
            if star_determinant(_tet((*parent.vertices, point)), budget=budget) != 0:
                _refuse(CavityAtlasStatus.COVERAGE_GAP, "virtual cone base must lie on its actual parent plane")
        children[parent.vertices].append(face)
    for face, parent in boundary.items():
        pieces = children[face]
        if not pieces:
            _refuse(CavityAtlasStatus.COVERAGE_GAP, "actual boundary facet has no virtual partition")
        plane = plane_key(face, budget=budget)
        if plane is None:
            _refuse(CavityAtlasStatus.INVALID_CAVITY, "actual boundary facet is degenerate")
        first_axis, second_axis = plane[1]
        axes = (first_axis, second_axis)
        projected = project(face, axes, budget=budget)
        area = signed_measure(projected, budget=budget)
        if not area:
            _refuse(CavityAtlasStatus.INVALID_CAVITY, "actual boundary facet projection is degenerate")
        # The coplanar projection is injective; orientation is its area sign.
        outward = (area > 0) == (parent.orientation > 0)
        with budget.temporary_scope():
            chain: dict[tuple[int, Fraction, Fraction], dict[Fraction, int]] = {}
            _add_planar_loop(
                chain, projected if parent.orientation > 0 else projected[::-1], -1, budget,
            )
            covered = Fraction(0)
            for child in pieces:
                triangle = project(child, axes, budget=budget)
                child_area = signed_measure(triangle, budget=budget)
                if not child_area or (child_area > 0) != outward:
                    _refuse(CavityAtlasStatus.INVALID_ANCESTRY, "virtual boundary orientation must match its actual outward parent")
                reserve_fraction_work(((covered, child_area),), 1, 1, 2, budget=budget)
                covered += abs(child_area)
                _add_planar_loop(chain, triangle, 1, budget)
            budget.reserve(sum(len(jumps) for jumps in chain.values()))
            if any(jump for jumps in chain.values() for jump in jumps.values()):
                # Equal measure without a partition still requires an overlap.
                _refuse(
                    CavityAtlasStatus.COVERAGE_GAP if covered < abs(area)
                    else CavityAtlasStatus.DOUBLE_COVERAGE,
                    "virtual children do not exactly partition their actual parent",
                )


def _kernel(tetrahedra: Sequence[Tet], faces: Sequence[Face], anchor: Point,
            budget: CoordinateEnclosureBudget) -> None:
    for face in faces:
        owner = next(tet for tet in tetrahedra if all(v in tet for v in face))
        opposite = next(v for v in owner if v not in face)
        expected = star_determinant(_tet((opposite, *face)), budget=budget)
        actual = star_determinant(_tet((anchor, *face)), budget=budget)
        if actual == 0 or (actual > 0) != (expected > 0):
            _refuse(CavityAtlasStatus.NO_COMMON_KERNEL, "anchor is not strictly inside every cavity boundary halfspace")


SNAPSHOT_NOTE_PREFIX = "implicit_cavity_snapshot="
CHART_NOTE_PREFIX = "implicit_cavity_chart="


def _replay_note(
    prefix: str, fields: dict[str, object], budget: CoordinateEnclosureBudget,
) -> str:
    """Prepay one exact JSON replay note, one work unit per emitted character.

    Rational values become exact decimal fractions; a ``b``-bit integer has at
    most ``b`` decimal digits. Encoder chunks and the joined text coexist.
    """
    characters = 2 + len(prefix)

    def encode(value: object) -> object:
        nonlocal characters
        if isinstance(value, CavitySnapshot):
            budget.reserve(6, 1024)
            characters += 128
            return {
                "vertices": encode(value.vertices),
                "cells": encode(value.cells),
                "cell_ids": encode(value.cell_ids),
                "constrained_faces": encode(value.constrained_faces),
                "source_labels": encode(value.source_labels),
                "rim_ancestry": encode(value.rim_ancestry),
            }
        if isinstance(value, (tuple, list)):
            budget.reserve(len(value), 128 + 32 * len(value))
            characters += 2 + len(value)
            return [encode(item) for item in value]
        if isinstance(value, Fraction):
            bits = value.numerator.bit_length() + value.denominator.bit_length() + 4
            budget.reserve(bits, 256 + 16 * bits)
            characters += bits + 2
            return str(value)
        if isinstance(value, int):
            bits = value.bit_length() + 2
            budget.reserve(bits, 128 + 16 * bits)
            characters += bits
            return value
        if isinstance(value, str):
            budget.reserve(len(value), 128 + 24 * len(value))
            characters += 6 * len(value) + 2
            return value
        if value is None:
            characters += 4
            return None
        raise TypeError("cavity snapshot diagnostics require exact rational inputs")

    budget.reserve(len(fields), 4096)
    characters += sum(len(name) + 4 for name in fields)
    payload = {name: encode(value) for name, value in fields.items()}
    budget.reserve(characters, 4096 + 32 * characters)
    return prefix + json.dumps(payload, separators=(",", ":"))


@dataclass(frozen=True, slots=True)
class ImplicitCavityEdit:
    """One chronological edit's exact inputs and their prepaid replay note.

    Every chart attempt of this edit attaches the same snapshot note; an
    attempt encodes only its own chart, anchor and resource limits.
    """

    before: CavitySnapshot
    after: CavitySnapshot
    source_id: str
    discovery_id: str
    chronology_index: int
    predecessor_id: str
    note: str


def prepare_implicit_cavity_edit(
    before: CavitySnapshot, after: CavitySnapshot, *,
    source_id: str, discovery_id: str, chronology_index: int,
    predecessor_id: str, budget: CoordinateEnclosureBudget,
) -> ImplicitCavityEdit:
    """Encode one edit's exact replay inputs once, before any chart attempt."""
    note = _replay_note(SNAPSHOT_NOTE_PREFIX, {
        "before": before, "after": after,
        "source_id": source_id, "discovery_id": discovery_id,
        "chronology_index": chronology_index, "predecessor_id": predecessor_id,
    }, budget)
    return ImplicitCavityEdit(
        before, after, source_id, discovery_id, chronology_index, predecessor_id, note,
    )


def _cavity_chart_note(
    *, chart_axes: tuple[int, int], anchor: tuple[Fraction, ...],
    chart_direction: tuple[Fraction, ...] | None,
    budget: CoordinateEnclosureBudget,
) -> str:
    """Prepay one attempt's exact chart note before its proof work."""
    return _replay_note(CHART_NOTE_PREFIX, {
        "chart_axes": chart_axes, "anchor": anchor, "chart_direction": chart_direction,
        "maximum_work_units": budget.maximum_work_units,
        "maximum_memory_bytes": budget.maximum_memory_bytes,
    }, budget)


def build_implicit_cavity_atlas(
    edit: ImplicitCavityEdit, *,
    chart_axes: tuple[int, int], anchor: tuple[Fraction, ...],
    budget: CoordinateEnclosureBudget,
    chart_direction: tuple[Fraction, ...] | None = None,
) -> ImplicitCavityAtlas:
    """Certify paired charts, retaining prepaid exact inputs on typed refusal.

    A refusal carries the edit's snapshot note followed by this attempt's
    chart note; together they replay the attempt exactly.
    """
    with budget.temporary_scope():
        chart = _cavity_chart_note(
            chart_axes=chart_axes, anchor=anchor,
            chart_direction=chart_direction, budget=budget,
        )
        try:
            return _build_implicit_cavity_atlas(
                edit, chart_axes=chart_axes, anchor=anchor,
                budget=budget, chart_direction=chart_direction,
            )
        except (
            CoordinateEnclosureResourceError,
            ExactPolyhedralRefusal,
            CavityAtlasRefusal,
            MeshcoreError,
        ) as error:
            error.add_note(edit.note)
            error.add_note(chart)
            raise


def _build_implicit_cavity_atlas(
    edit: ImplicitCavityEdit, *,
    chart_axes: tuple[int, int], anchor: tuple[Fraction, ...],
    budget: CoordinateEnclosureBudget,
    chart_direction: tuple[Fraction, ...] | None = None,
) -> ImplicitCavityAtlas:
    """Certify both complete cavities and return paired positive affine charts.

    Axis projection, or the explicitly supplied exact oblique direction, defines
    the local planar chart. Insertion callers supply projected minus original
    source-edge split position, preserving its exact material preimage. No
    alternative chart or anchor is tried after failure. Resource refusals retain
    their canonical types. Fractions in the result are authoritative.
    """
    before, after = edit.before, edit.after
    source_id, discovery_id = edit.source_id, edit.discovery_id
    chronology_index, predecessor_id = edit.chronology_index, edit.predecessor_id
    if len(chart_axes) != 2 or len(set(chart_axes)) != 2 or any(a not in (0, 1, 2) for a in chart_axes):
        _refuse(CavityAtlasStatus.NOT_A_GRAPH, "chart must select two distinct physical axes")
    if chart_direction is not None:
        direction_values = tuple(Fraction(value) for value in chart_direction)
        dropped = next(axis for axis in range(3) if axis not in chart_axes)
        if len(direction_values) != 3 or direction_values[dropped] == 0:
            _refuse(CavityAtlasStatus.NOT_A_GRAPH, "oblique chart requires a nonzero dropped direction component")
        chart_direction = _point(direction_values)
    if not source_id or not discovery_id or chronology_index < 0 or (chronology_index and not predecessor_id):
        _refuse(CavityAtlasStatus.INVALID_ANCESTRY, "missing original source/discovery or chronological predecessor")
    anchor_values = tuple(Fraction(x) for x in anchor)
    if len(anchor_values) != 3:
        _refuse(CavityAtlasStatus.NO_COMMON_KERNEL, "anchor must have three coordinates")
    anchor = _point(anchor_values)
    old, first, fixed_old, rim_old = _snapshot(before, budget)
    new, second, fixed_new, rim_new = _snapshot(after, budget)
    source_boundary = _oriented_cavity_boundary(old, before.cell_ids, budget)
    target_boundary = _oriented_cavity_boundary(new, after.cell_ids, budget)
    if fixed_old != fixed_new or rim_old != rim_new:
        _refuse(CavityAtlasStatus.RIM_MISMATCH, "edit changes the fixed cavity boundary or source rim")
    _kernel(old, (*fixed_old, *(face for face, _ in first)), anchor, budget)
    _kernel(new, (*fixed_new, *(face for face, _ in second)), anchor, budget)
    p = _graph(first, chart_axes, chart_direction, budget)
    q = _graph(second, chart_axes, chart_direction, budget)
    first_area = [Fraction(0)] * len(p)
    second_area = [Fraction(0)] * len(q)
    paired = []
    parent_pairs = []
    for i, (face, label) in enumerate(first):
        for j, (other, other_label) in enumerate(second):
            polygon = intersection_polygon(p[i], q[j], budget=budget)
            area = abs(signed_measure(polygon, budget=budget))
            if not area:
                continue
            if label != other_label:
                _refuse(CavityAtlasStatus.INVALID_ANCESTRY, "surface overlay crosses source labels")
            budget.reserve(2)
            first_area[i] += area
            second_area[j] += area
            # Both lifts are affine maps of this one planar polygon onto fixed
            # face planes, and the anchor lies strictly off both, so every
            # triangle of the polygon induces the same exact cone map. The
            # owner's exact triangulation uses only the polygon's own vertices,
            # keeps every collinear edge subdivision and never forms a
            # degenerate or averaged-center triangle. Clipping repeats a vertex
            # lying exactly on a clip line; each distinct vertex is lifted once.
            budget.reserve(2 * len(polygon), 512 * len(polygon))
            vertices = tuple(
                vertex for index, vertex in enumerate(polygon) if vertex != polygon[index - 1]
            )
            lifted = tuple(
                (_lift(vertex, face, p[i], budget), _lift(vertex, other, q[j], budget))
                for vertex in vertices
            )
            loop = np.empty((len(vertices), 3), dtype=object)
            for row, (point, _) in enumerate(lifted):
                loop[row] = point
            for triangle in triangulate_loop(loop, tuple(range(len(vertices))), budget=budget):
                paired.append((tuple(lifted[v][0] for v in triangle),
                               tuple(lifted[v][1] for v in triangle)))
                parent_pairs.append((source_boundary[face], target_boundary[other]))
    for triangles, areas in ((p, first_area), (q, second_area)):
        if any(area != abs(signed_measure(triangle, budget=budget)) for triangle, area in zip(triangles, areas, strict=True)):
            _refuse(CavityAtlasStatus.COVERAGE_GAP, "surface overlay does not cover both complete patches")
    paired.extend((face, face) for face in sorted(fixed_old))
    parent_pairs.extend(
        (source_boundary[face], target_boundary[face]) for face in sorted(fixed_old)
    )
    old_tets, new_tets = [], []
    for a, b in paired:
        first_tet, second_tet = _tet((anchor, *a)), _tet((anchor, *b))
        first_det = star_determinant(first_tet, budget=budget)
        second_det = star_determinant(second_tet, budget=budget)
        if first_det == 0 or second_det == 0 or (first_det > 0) != (second_det > 0):
            _refuse(CavityAtlasStatus.NO_COMMON_KERNEL, "paired cones lack a shared strict kernel orientation")
        if first_det < 0:
            first_tet = (anchor, a[0], a[2], a[1])
            second_tet = (anchor, b[0], b[2], b[1])
        old_tets.append(first_tet)
        new_tets.append(second_tet)
    source_parents = tuple(pair[0] for pair in parent_pairs)
    target_parents = tuple(pair[1] for pair in parent_pairs)
    _cover_coned_boundary(old, before.cell_ids, old_tets, source_parents, anchor, budget)
    _cover_coned_boundary(new, after.cell_ids, new_tets, target_parents, anchor, budget)
    reserve_fraction_work(tuple(v for tet in (*old_tets, *new_tets) for v in tet), 0,
                          12 * (len(old_tets) + len(new_tets)), 1, budget=budget)
    identity = canonical_fingerprint({
        "kind": "implicit-cavity-atlas", "source": source_id,
        "discovery": discovery_id, "chronology": chronology_index,
        "predecessor": predecessor_id, "axes": chart_axes,
        "direction": None if chart_direction is None else tuple(str(value) for value in chart_direction),
        "old": tuple(tuple(tuple(str(x) for x in v) for v in t) for t in old_tets),
        "new": tuple(tuple(tuple(str(x) for x in v) for v in t) for t in new_tets),
        "source_cells": before.cell_ids, "target_cells": after.cell_ids,
        "labels": (before.source_labels, after.source_labels),
        "rim": tuple((tuple(tuple(str(x) for x in v) for v in edge), label)
                     for edge, label in sorted(rim_old.items())),
        "source_boundary_parents": tuple(
            (parent.cell_id, tuple(tuple(str(x) for x in point) for point in parent.vertices), parent.orientation)
            for parent in source_parents
        ),
        "target_boundary_parents": tuple(
            (parent.cell_id, tuple(tuple(str(x) for x in point) for point in parent.vertices), parent.orientation)
            for parent in target_parents
        ),
    })
    old_bounds, old_hull = _chart_bounds(old_tets, budget)
    new_bounds, new_hull = _chart_bounds(new_tets, budget)
    atlas = ImplicitCavityAtlas(tuple(old_tets), tuple(new_tets), identity,
                                before.cell_ids, after.cell_ids, source_id,
                                discovery_id, chronology_index, predecessor_id,
                                chart_axes, anchor, chart_direction,
                                source_parents, target_parents,
                                old_bounds, new_bounds, old_hull, new_hull)
    # Both cone partitions above were proved against these actual cells.
    _issue_cavity_binding(atlas, old, before.cell_ids, new, after.cell_ids, budget)
    return atlas


def _chart_bounds(
    tets: Sequence[Tet], budget: CoordinateEnclosureBudget,
) -> tuple[tuple[Box, ...], Box]:
    """Exact closed boxes of every chart tetrahedron and of their union."""
    reserve_fraction_work(tuple(point for tet in tets for point in tet),
                          24 * len(tets), 0, 1, budget=budget)
    boxes = tuple(
        (_point(tuple(min(point[axis] for point in tet) for axis in range(3))),
         _point(tuple(max(point[axis] for point in tet) for axis in range(3))))
        for tet in tets
    )
    hull = (_point(tuple(min(box[0][axis] for box in boxes) for axis in range(3))),
            _point(tuple(max(box[1][axis] for box in boxes) for axis in range(3))))
    return boxes, hull


# Canonical exact payload: every rational is its (numerator, denominator)
# integer pair, so later mutation of a public Fraction cannot alias a value.
type _Payload = int | str | None | tuple[_Payload, ...]


def _canonical(value: object, /) -> _Payload:
    match value:
        case Fraction():
            return value.numerator, value.denominator
        case CavityFacetParent():
            return value.cell_id, _canonical(value.vertices), value.orientation
        case tuple():
            return tuple(_canonical(item) for item in value)
        case int() | str() | None:
            return value
        case _:
            raise TypeError(
                f"implicit cavity binding payload must be exact; got {type(value).__name__}"
            )


def _payload_cost(
    tets: int, parents: int, identities: int, /,
) -> tuple[int, int]:
    """Visits and transient bytes of one canonical payload: one visit per
    rational or integer leaf, and a pair tuple per rational."""
    rationals = 12 * tets + 9 * parents + 3
    return rationals + 2 * parents + identities + 1, 128 * (rationals + parents + tets) + 256


@dataclass(frozen=True, slots=True, weakref_slot=True)
class _IssuedCavityBinding:
    """Actual cavity cells against which one atlas's cone partitions were proved.

    ``atlas`` is the earning atlas object, held weakly. ``controls`` is the
    canonical integer payload of its proof inputs, ``old``/``new`` that of the
    actual cells by identity, sorted corners and orientation parity.
    """

    atlas: weakref.ReferenceType[ImplicitCavityAtlas]
    controls: _Payload
    old: _Payload
    new: _Payload


# Earned cavity bindings, weakly keyed by the request ledger that paid for
# them (see ``IssuedProofs``): another request, a restored atlas or a fresh
# process never reuses one and proves its cones again.
_ISSUED_CAVITY_BINDINGS: IssuedProofs[CoordinateEnclosureBudget, _IssuedCavityBinding] = (
    IssuedProofs()
)


def _atlas_controls(
    atlas: ImplicitCavityAtlas, budget: CoordinateEnclosureBudget,
) -> _Payload:
    """Canonical payload of every input of the atlas's two cone proofs."""
    parents = len(atlas.source_boundary_parents) + len(atlas.target_boundary_parents)
    visits, storage = _payload_cost(
        len(atlas.old_tets) + len(atlas.new_tets), parents,
        len(atlas.source_cell_ids) + len(atlas.target_cell_ids),
    )
    budget.reserve(visits, storage)
    return _canonical((
        atlas.old_tets, atlas.new_tets, atlas.source_boundary_parents,
        atlas.target_boundary_parents, atlas.anchor, atlas.source_cell_ids,
        atlas.target_cell_ids, atlas.chart_id,
    ))


def _binding_key(atlas: ImplicitCavityAtlas) -> str:
    return f"{atlas.chart_id}:{id(atlas)}"


def _oriented_cells(
    cells: Sequence[Tet], cell_ids: Sequence[int], budget: CoordinateEnclosureBudget,
) -> _Payload:
    """Cells as (identity, sorted corners, permutation parity): equal iff the
    same oriented tetrahedron, independent of its even corner rotation."""
    if len(cells) != len(cell_ids):
        _refuse(CavityAtlasStatus.INVALID_ANCESTRY, "actual cavity cell identities are incomplete")
    # Six exact pairwise corner comparisons per cell, then its payload.
    reserve_fraction_work(tuple(point for tet in cells for point in tet), 6 * len(cells), 0, 2,
                          budget=budget)
    visits, storage = _payload_cost(len(cells), 0, len(cells))
    budget.reserve(visits, storage)
    result = []
    for identity, tet in zip(cell_ids, cells, strict=True):
        inversions = sum(tet[a] > tet[b] for a, b in combinations(range(4), 2))
        result.append((int(identity), _canonical(tuple(sorted(tet))), inversions % 2))
    return tuple(sorted(result))


def _issue_cavity_binding(
    atlas: ImplicitCavityAtlas,
    old_cells: Sequence[Tet], old_ids: Sequence[int],
    new_cells: Sequence[Tet], new_ids: Sequence[int],
    budget: CoordinateEnclosureBudget,
) -> None:
    binding = _IssuedCavityBinding(
        weakref.ref(atlas),
        _atlas_controls(atlas, budget),
        _oriented_cells(old_cells, old_ids, budget),
        _oriented_cells(new_cells, new_ids, budget),
    )
    # The ledger owns the binding and its integer payloads for its lifetime;
    # the registry is weak.
    budget.retain_basis((binding, binding.controls, binding.old, binding.new))
    _ISSUED_CAVITY_BINDINGS.record(budget, _binding_key(atlas), binding)


def bind_implicit_cavity_cells(
    old_cells: Sequence[Tet], old_ids: Sequence[int],
    new_cells: Sequence[Tet], new_ids: Sequence[int],
    atlas: ImplicitCavityAtlas, budget: CoordinateEnclosureBudget,
) -> None:
    """Bind a paired atlas to the actual old and new cavity tetrahedra.

    The cells must be exactly the atlas's declared cavity identities. Each
    side's chart cones then prove, through their authenticated outward parent
    facets, equality with the actual cavity as oriented compact volume chains.

    If this request's ledger already earned that proof for this very atlas
    object, whose proof inputs still have the exact earned integer values, and
    the actual cells are exactly the earned oriented cells, the equal premises
    give the same conclusion; only those exact comparisons are repeated and
    charged. Any other atlas, changed value or cell set proves its cones again.
    """
    if (sorted(old_ids) != sorted(atlas.source_cell_ids)
            or sorted(new_ids) != sorted(atlas.target_cell_ids)):
        _refuse(CavityAtlasStatus.INVALID_CAVITY, "atlas cavity IDs are absent from actual endpoints")
    issued = _ISSUED_CAVITY_BINDINGS.get(budget, _binding_key(atlas))
    if (
        issued is not None
        and issued.atlas() is atlas
        and _atlas_controls(atlas, budget) == issued.controls
        and _oriented_cells(old_cells, old_ids, budget) == issued.old
        and _oriented_cells(new_cells, new_ids, budget) == issued.new
    ):
        return
    _cover_coned_boundary(
        tuple(old_cells), tuple(old_ids), atlas.old_tets, atlas.source_boundary_parents,
        atlas.anchor, budget,
    )
    _cover_coned_boundary(
        tuple(new_cells), tuple(new_ids), atlas.new_tets, atlas.target_boundary_parents,
        atlas.anchor, budget,
    )
    _issue_cavity_binding(atlas, old_cells, old_ids, new_cells, new_ids, budget)

# Copyright © 2026 PHYDRA, Inc. All rights reserved.
"""Rational geometric qualification of declared tagged-simplex bisection forests."""

from __future__ import annotations

from fractions import Fraction
from itertools import combinations, permutations
from math import acos, asin, factorial, isqrt, sqrt, ulp
from sys import float_info
from typing import NamedTuple, NoReturn

import numpy as np

from phydrax.discretization import CellMesh
from phydrax.meshing import BisectionHierarchy
from phydrax.typing import checked
from tools.meshing_qualification._runtime import qualification_failure


type Point = tuple[Fraction, ...]
type Simplex = tuple[Point, ...]
type AngleScore = tuple[int, Fraction]
type Interval = tuple[Fraction, Fraction]


class _Node(NamedTuple):
    vertices: tuple[int, ...]
    tag: int
    generation: int
    ancestor: int


class _Realization(NamedTuple):
    ideal: dict[int, Point]
    published: dict[int, Point]
    displacement_bounds: dict[int, Point]
    half_ulps: dict[int, Point]
    parent_edges: dict[int, tuple[int, int]]


def _realize_vertex(bank: _Realization, identity: int, low: int, high: int) -> None:
    ideal = _midpoint(bank.ideal[low], bank.ideal[high])
    published, half_ulps = _rounded_point(ideal)
    # Round the ideal descendant once, seeded from accepted epoch-zero floats.
    # Correct rounding bounds each coordinate displacement by half an ULP,
    # independently of how many rational midpoint generations preceded it.
    if identity in bank.ideal and (
        bank.ideal[identity] != ideal or bank.published[identity] != published
    ):
        _refuse("midpoint", identity)
    if any(
        abs(a - b) > limit
        for a, b, limit in zip(published, ideal, half_ulps, strict=True)
    ):
        _refuse("displacement_bound", identity)
    bank.ideal[identity], bank.published[identity] = ideal, published
    bank.displacement_bounds[identity], bank.half_ulps[identity] = half_ulps, half_ulps
    bank.parent_edges[identity] = (low, high)


def _refuse(field: str, observed: object) -> NoReturn:
    raise qualification_failure(
        "native_lifecycle.descent",
        f"native_lifecycle.descent.{field} must satisfy certified tagged bisection; got {observed!r}",
        {"field": field, "observed": repr(observed)},
    )


def _difference(a: Point, b: Point) -> Point:
    return tuple(x - y for x, y in zip(a, b, strict=True))


def _dot(a: Point, b: Point) -> Fraction:
    return sum((x * y for x, y in zip(a, b, strict=True)), Fraction(0))


def _cross(a: Point, b: Point) -> Point:
    return (
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    )


def _measure(simplex: Simplex) -> Fraction:
    edges = tuple(_difference(point, simplex[0]) for point in simplex[1:])
    determinant = (
        edges[0][0] * edges[1][1] - edges[0][1] * edges[1][0]
        if len(edges) == 2
        else _dot(edges[0], _cross(edges[1], edges[2]))
    )
    return abs(determinant) / factorial(len(edges))


def _children[T](
    vertices: tuple[T, ...], tag: int, midpoint: T
) -> tuple[tuple[T, ...], tuple[T, ...]]:
    return (
        (*vertices[:tag], midpoint, *vertices[tag + 1 :]),
        (*vertices[1 : tag + 1], midpoint, *vertices[tag + 1 :]),
    )


def _canonical(vertices: tuple[int, ...], tag: int) -> tuple[int, ...]:
    return min(vertices, (*vertices[: tag + 1][::-1], *vertices[tag + 1 :]))


def _midpoint(a: Point, b: Point) -> Point:
    return tuple((x + y) / 2 for x, y in zip(a, b, strict=True))


def _cosine_score(a: Point, b: Point) -> AngleScore:
    dot = _dot(a, b)
    denominator = _dot(a, a) * _dot(b, b)
    if denominator == 0:
        _refuse("angle", "degenerate vectors")
    sign = (dot > 0) - (dot < 0)
    return sign, sign * dot * dot / denominator


def _angle_vectors(simplex: Simplex) -> tuple[tuple[Point, Point], ...]:
    if len(simplex) == 3:
        return tuple(
            (_difference(simplex[j], simplex[i]), _difference(simplex[k], simplex[i]))
            for i, j, k in ((0, 1, 2), (1, 0, 2), (2, 0, 1))
        )
    normals = []
    for excluded in range(4):
        face = [simplex[i] for i in range(4) if i != excluded]
        normal = _cross(_difference(face[1], face[0]), _difference(face[2], face[0]))
        if _dot(normal, _difference(simplex[excluded], face[0])) > 0:
            normal = tuple(-x for x in normal)
        normals.append(normal)
    return tuple((a, tuple(-x for x in b)) for a, b in combinations(normals, 2))


def _minimum_angle_score(simplex: Simplex) -> AngleScore:
    return max(_cosine_score(a, b) for a, b in _angle_vectors(simplex))


def _angle_record(score: AngleScore) -> dict[str, object]:
    sign, squared = score
    return {
        "cosine_sign": sign,
        "squared_cosine": str(abs(squared)),
        "radians": acos(sign * sqrt(float(abs(squared)))),
    }


def check_angle_bound(simplex: Simplex, bound: AngleScore) -> AngleScore:
    """Refuse an angle below a bound using signed rational squared cosines."""
    observed = _minimum_angle_score(simplex)
    if observed > bound:
        _refuse("minimum_angle", {"bound": bound, "observed": observed})
    return observed


def _sqrt_interval(value: Fraction) -> Interval:
    """Enclose a rational square root in adjacent dyadics with 256 significand bits."""
    if value < 0:
        _refuse("square_root", value)
    if value == 0:
        return Fraction(0), Fraction(0)
    exponent = (value.numerator.bit_length() - value.denominator.bit_length()) // 2
    step = Fraction(2) ** (exponent - 256)
    scaled = value / (step * step)
    lower = isqrt(scaled.numerator // scaled.denominator) * step
    return lower, lower if lower * lower == value else lower + step


def _interval_product(a: Interval, b: Interval) -> Interval:
    products = tuple(x * y for x in a for y in b)
    return min(products), max(products)


def _cosine_interval(score: AngleScore) -> Interval:
    sign, squared = score
    lower, upper = _sqrt_interval(abs(squared))
    return (lower, upper) if sign >= 0 else (-upper, -lower)


def _perturbed_cosine_bound(
    bound: AngleScore, first: Fraction, second: Fraction
) -> Interval:
    """Enclose cos(bound angle - asin(first) - asin(second)) with rational radicals."""
    if not (0 <= first < 1 and 0 <= second < 1):
        _refuse("angle_perturbation", (first, second))
    a, b = _sqrt_interval(1 - first * first), _sqrt_interval(1 - second * second)
    product = _interval_product(a, b)
    cosine_delta = product[0] - first * second, product[1] - first * second
    sine_delta = first * b[0] + second * a[0], first * b[1] + second * a[1]
    cosine_bound = _cosine_interval(bound)
    sine_bound = _sqrt_interval(1 - abs(bound[1]))
    if cosine_delta[0] <= 0 or sine_delta[1] >= sine_bound[0]:
        _refuse(
            "angle_perturbation", "displacement cannot certify a positive angle floor"
        )
    c = _interval_product(cosine_bound, cosine_delta)
    s = _interval_product(sine_bound, sine_delta)
    return c[0] + s[0], c[1] + s[1]


def _direction_error(ideal: Point, displacement: Point) -> Fraction:
    norm = _sqrt_interval(_dot(ideal, ideal))[0]
    if norm == 0:
        _refuse("angle_perturbation", "degenerate ideal direction")
    return _sqrt_interval(_dot(displacement, displacement))[1] / norm


def _published_angles(
    ideal: Simplex, published: Simplex, bound: AngleScore
) -> tuple[AngleScore, list[dict[str, object]]]:
    """Certify published angles using displacement-induced direction changes.

    A vector of length l perturbed by e changes direction by at most asin(|e|/l)
    when |e| < l: the tangent from its origin to the radius-|e| ball gives this
    bound. A vertex angle changes by at most the sum of its two direction
    changes. For triangles, |e| <= |d_apex| + |d_other|; tetrahedral dihedrals use
    the displacement of their rational face normals. Cosine addition expresses
    the lowered class floor using rational square-root intervals, avoiding
    inverse-trigonometric decisions. Displayed radians are diagnostic only.
    """
    evidence: list[dict[str, object]] = []
    displacements = tuple(
        _difference(a, b) for a, b in zip(published, ideal, strict=True)
    )
    for index, ((a, b), (p, q)) in enumerate(
        zip(_angle_vectors(ideal), _angle_vectors(published), strict=True)
    ):
        if len(ideal) == 3:
            apex, one, two = ((0, 1, 2), (1, 0, 2), (2, 0, 1))[index]
            apex_norm = _sqrt_interval(_dot(displacements[apex], displacements[apex]))[1]
            first = (
                apex_norm
                + _sqrt_interval(_dot(displacements[one], displacements[one]))[1]
            ) / _sqrt_interval(_dot(a, a))[0]
            second = (
                apex_norm
                + _sqrt_interval(_dot(displacements[two], displacements[two]))[1]
            ) / _sqrt_interval(_dot(b, b))[0]
        else:
            first, second = (
                _direction_error(a, _difference(p, a)),
                _direction_error(b, _difference(q, b)),
            )
        if not (0 <= first < 1 and 0 <= second < 1):
            _refuse("angle_perturbation", (first, second))
        observed = _cosine_score(p, q)
        if observed > bound:
            allowance = _perturbed_cosine_bound(bound, first, second)
            if _cosine_interval(observed)[1] > allowance[0]:
                _refuse(
                    "published_minimum_angle",
                    {
                        "bound": bound,
                        "observed": observed,
                        "direction_errors": (first, second),
                    },
                )
        evidence.append(
            {
                "angle": index,
                "direction_error_bounds": (str(first), str(second)),
                "perturbation_radians": asin(float(first)) + asin(float(second)),
            }
        )
    return _minimum_angle_score(published), evidence


def _rounded_point(point: Point) -> tuple[Point, Point]:
    """Return correctly rounded normal float64 ideal coordinates and half-ULP bounds."""
    rounded, bounds = [], []
    for value in point:
        if abs(value) > Fraction(float_info.max):
            _refuse("realization_overflow", value)
        if 0 < abs(value) < Fraction(float_info.min):
            _refuse("realization_subnormal", value)
        realized = float(value)
        rounded.append(Fraction(realized))
        bounds.append(Fraction(ulp(realized)) / 2 if value else Fraction(0))
    return tuple(rounded), tuple(bounds)


def _distance_key(simplex: Simplex) -> tuple[Fraction, ...]:
    lengths = tuple(
        _dot(edge, edge)
        for edge in (_difference(a, b) for a, b in combinations(simplex, 2))
    )
    scale = max(lengths)
    if scale == 0 or _measure(simplex) == 0:
        _refuse("source_measure", "degenerate simplex")
    return tuple(length / scale for length in lengths)


def similarity_bound(simplex: Simplex, tag: int) -> tuple[AngleScore, int, int]:
    """Close the tagged similarity set, with four triangle or 192 tetrahedron shapes."""
    dimension = len(simplex) - 1
    shape_cap, tagged_cap = (4, 32) if dimension == 2 else (192, 1536)
    pending = [(simplex, tag)]
    seen: set[tuple[int, tuple[Fraction, ...]]] = set()
    shapes: set[tuple[Fraction, ...]] = set()
    bound: AngleScore = (-1, Fraction(-1))
    while pending:
        cell, current_tag = pending.pop()
        reversed_cell = (*cell[: current_tag + 1][::-1], *cell[current_tag + 1 :])
        key = current_tag, min(_distance_key(cell), _distance_key(reversed_cell))
        if key in seen:
            continue
        seen.add(key)
        shapes.add(min(_distance_key(tuple(order)) for order in permutations(cell)))
        if len(shapes) > shape_cap or len(seen) > tagged_cap:
            _refuse(
                "similarity_class_closure",
                {"shapes": len(shapes), "shape_cap": shape_cap, "tagged_cap": tagged_cap},
            )
        bound = max(bound, _minimum_angle_score(cell))
        midpoint = _midpoint(cell[0], cell[current_tag])
        child_tag = current_tag - 1 or dimension
        pending.extend(
            (child, child_tag) for child in _children(cell, current_tag, midpoint)
        )
    return bound, len(shapes), len(seen)


def _mesh_tables(mesh: CellMesh) -> tuple[dict[int, Point], dict[int, tuple[int, ...]]]:
    points = {
        int(identity): tuple(Fraction(float(value)) for value in point)
        for identity, point in zip(
            np.asarray(mesh.vertex_global_ids), np.asarray(mesh.coordinates), strict=True
        )
    }
    vertices = np.asarray(mesh.vertex_global_ids)
    cells = {
        int(identity): tuple(int(value) for value in vertices[row])
        for block in mesh.blocks
        for identity, row in zip(
            np.asarray(block.global_ids), np.asarray(block.vertices), strict=True
        )
    }
    return points, cells


def _replay_forest(
    pending: list[tuple[int, _Node]],
    bank: _Realization,
    hierarchy: BisectionHierarchy,
    records: dict[int, int],
) -> tuple[dict[int, _Node], set[int]]:
    """Construct the leaves from declared parent labels and rational binary halves."""
    parent_vertices = np.asarray(hierarchy.record_parent_vertices)
    parent_tags = np.asarray(hierarchy.record_parent_tags)
    child_ids = np.asarray(hierarchy.record_child_ids)
    midpoint_ids = np.asarray(hierarchy.record_vertex_ids)
    visited: set[int] = set()
    leaves: dict[int, _Node] = {}
    while pending:
        identity, node = pending.pop()
        if identity in visited:
            _refuse("forest", identity)
        visited.add(identity)
        if identity not in records:
            leaves[identity] = node
            continue
        row = records[identity]
        declared = tuple(map(int, parent_vertices[row]))
        if (
            int(parent_tags[row]) != node.tag
            or _canonical(declared, node.tag) != node.vertices
        ):
            _refuse("newest_vertex_edge", identity)
        midpoint_id = int(midpoint_ids[row])
        _realize_vertex(bank, midpoint_id, node.vertices[0], node.vertices[node.tag])
        tag = node.tag - 1 or hierarchy.dimension
        for child_id, vertices in zip(
            child_ids[row], _children(node.vertices, node.tag, midpoint_id), strict=True
        ):
            pending.append(
                (
                    int(child_id),
                    _Node(
                        _canonical(vertices, tag), tag, node.generation + 1, node.ancestor
                    ),
                )
            )
    return leaves, visited


def _prepare_roots(
    source_cells: dict[int, tuple[int, ...]],
    coordinates: dict[int, Point],
    hierarchy: BisectionHierarchy,
    records: dict[int, int],
    active: dict[int, int],
) -> tuple[
    list[tuple[int, _Node]],
    dict[int, AngleScore],
    list[dict[str, int]],
    dict[int, Fraction],
]:
    parent_vertices, parent_tags = (
        np.asarray(hierarchy.record_parent_vertices),
        np.asarray(hierarchy.record_parent_tags),
    )
    active_vertices, active_tags = (
        np.asarray(hierarchy.ordered_vertices),
        np.asarray(hierarchy.tags),
    )
    pending: list[tuple[int, _Node]] = []
    bounds: dict[int, AngleScore] = {}
    counts = []
    measures = {}
    for identity, vertices in source_cells.items():
        if identity in records:
            row = records[identity]
            ordered, tag = tuple(map(int, parent_vertices[row])), int(parent_tags[row])
        elif identity in active:
            row = active[identity]
            ordered, tag = tuple(map(int, active_vertices[row])), int(active_tags[row])
        else:
            _refuse("refinement_edge", identity)
        if set(ordered) != set(vertices) or tag not in range(1, hierarchy.dimension + 1):
            _refuse("source_label", (identity, ordered, tag))
        cell = tuple(coordinates[vertex] for vertex in ordered)
        bound, count, tagged_count = similarity_bound(cell, tag)
        bounds[identity] = bound
        counts.append(
            {"ancestor": identity, "classes": count, "tagged_classes": tagged_count}
        )
        measures[identity] = _measure(cell)
        pending.append((identity, _Node(_canonical(ordered, tag), tag, 0, identity)))
    return pending, bounds, counts, measures


def _verify_published_vertex(
    identity: int, points: dict[int, Point], bank: _Realization
) -> dict[str, object]:
    displacement = _difference(points[identity], bank.ideal[identity])
    limits = bank.displacement_bounds[identity]
    if any(abs(value) > limit for value, limit in zip(displacement, limits, strict=True)):
        _refuse("displacement_bound", identity)
    return {
        "vertex": identity,
        "displacement": tuple(map(str, displacement)),
        "derived_bound": tuple(map(str, limits)),
        "half_ulps": tuple(map(str, bank.half_ulps.get(identity, ()))),
    }


def _realization_evidence(
    identities: tuple[int, ...],
    points: dict[int, Point],
    bank: _Realization,
) -> list[dict[str, object]]:
    mismatches = [
        {
            "vertex": identity,
            "kind": "rounded-midpoint"
            if identity in bank.parent_edges
            else "epoch-zero-vertex",
            "parent_edge": bank.parent_edges.get(identity),
            "expected": tuple(map(str, bank.published[identity])),
            "observed": tuple(map(str, points.get(identity, ()))),
        }
        for identity in identities
        if identity not in points or points[identity] != bank.published[identity]
    ]
    if mismatches:
        _refuse("nested_vertex", {"mismatches": mismatches})
    return [_verify_published_vertex(identity, points, bank) for identity in identities]


def _verify_retained_vertices(source: CellMesh, target: CellMesh) -> None:
    """Require the binary64 carrier bits of every epoch-zero vertex to survive."""
    source_ids = np.asarray(source.vertex_global_ids)
    target_rows = {
        int(identity): row
        for row, identity in enumerate(np.asarray(target.vertex_global_ids))
    }
    rows = np.asarray(
        [target_rows.get(int(identity), -1) for identity in source_ids], dtype=np.int64
    )
    if np.any(rows < 0):
        _refuse(
            "retained_vertex",
            {"kind": "epoch-zero-vertex", "missing": source_ids[rows < 0].tolist()},
        )
    old = np.asarray(source.coordinates, dtype=np.float64).view(np.uint64)
    new = np.asarray(target.coordinates, dtype=np.float64).view(np.uint64)[rows]
    changed = np.flatnonzero(np.any(old != new, axis=1))
    if changed.size:
        _refuse(
            "retained_vertex",
            {
                "kind": "epoch-zero-vertex",
                "vertices": [
                    {
                        "vertex": int(source_ids[row]),
                        "expected_bits": old[row].tolist(),
                        "observed_bits": new[row].tolist(),
                    }
                    for row in changed
                ],
            },
        )


@checked
def certify_descent(
    source: CellMesh, target: CellMesh, hierarchy: BisectionHierarchy
) -> dict[str, object]:
    """Replay a declared forest from epoch zero with rational coordinates and measures."""
    source_points, source_cells = _mesh_tables(source)
    target_points, target_cells = _mesh_tables(target)
    dimension = hierarchy.dimension
    if any(
        len(point) != dimension
        for points in (source_points, target_points)
        for point in points.values()
    ):
        _refuse("dimension", dimension)
    _verify_retained_vertices(source, target)
    parents = np.asarray(hierarchy.record_parent_ids)
    records = {int(identity): row for row, identity in enumerate(parents)}
    active = {
        int(identity): row
        for row, identity in enumerate(np.asarray(hierarchy.cell_global_ids))
    }
    if set(active) != set(target_cells):
        _refuse("active_cells", sorted(active))
    active_vertices = np.asarray(hierarchy.ordered_vertices)
    active_tags = np.asarray(hierarchy.tags)
    active_generations = np.asarray(hierarchy.generations)
    bank = _Realization(
        dict(source_points),
        dict(source_points),
        {
            identity: tuple(Fraction(0) for _ in point)
            for identity, point in source_points.items()
        },
        {},
        {},
    )
    pending, bounds, class_counts, source_measures = _prepare_roots(
        source_cells,
        bank.ideal,
        hierarchy,
        records,
        active,
    )
    leaves, visited = _replay_forest(pending, bank, hierarchy, records)
    if set(leaves) != set(target_cells) or not set(records) <= visited:
        _refuse("forest_leaves", sorted(leaves))
    generations = []
    observed: AngleScore = (-1, Fraction(-1))
    totals = dict.fromkeys(source_cells, Fraction(0))
    vertex_evidence = _realization_evidence(
        tuple(sorted({vertex for node in leaves.values() for vertex in node.vertices})),
        target_points,
        bank,
    )
    angle_evidence = []
    ideal_observed: AngleScore = (-1, Fraction(-1))
    for identity, node in sorted(leaves.items()):
        row = active[identity]
        if (
            int(active_tags[row]) != node.tag
            or int(active_generations[row]) != node.generation
            or _canonical(tuple(map(int, active_vertices[row])), node.tag)
            != node.vertices
            or set(target_cells[identity]) != set(node.vertices)
        ):
            _refuse("target_label", identity)
        cell = tuple(target_points[vertex] for vertex in node.vertices)
        ideal = tuple(bank.ideal[vertex] for vertex in node.vertices)
        measure = _measure(ideal)
        if measure != source_measures[node.ancestor] / 2**node.generation:
            _refuse("halved_measure", identity)
        totals[node.ancestor] += measure
        ideal_observed = max(
            ideal_observed, check_angle_bound(ideal, bounds[node.ancestor])
        )
        published_score, perturbations = _published_angles(
            ideal, cell, bounds[node.ancestor]
        )
        observed = max(observed, published_score)
        angle_evidence.append({"cell": identity, "angles": perturbations})
        generations.append(
            {"cell": identity, "ancestor": node.ancestor, "generation": node.generation}
        )
    if totals != source_measures:
        _refuse("ancestor_partition", totals)
    return {
        "retained_vertices_bitwise_preserved": True,
        "status": "certified",
        "generations": generations,
        "class_counts": class_counts,
        "bound": _angle_record(max(bounds.values())),
        "observed_minimum": _angle_record(observed),
        "ideal_observed_minimum": _angle_record(ideal_observed),
        "published_angle_perturbations": angle_evidence,
        "realization": {
            "status": "certified",
            "rule": "round-to-nearest-even-ideal-vertex",
            "vertices": vertex_evidence,
        },
        "arithmetic": "rational-ideal-and-certified-float64-realization",
        "shape_class_cap": 4 if dimension == 2 else 192,
    }

# Copyright © 2026 PHYDRA, Inc. All rights reserved.
"""Exact tagged-simplex and finite whole-edge-star forest qualification."""

from __future__ import annotations

from fractions import Fraction
from itertools import combinations, permutations
from math import acos, asin, factorial, isqrt, sqrt, ulp
from sys import float_info
from typing import NamedTuple, NoReturn

import numpy as np

from phydrax.discretization import CellMesh
from phydrax.meshing import BisectionHierarchy, BisectionUniformRefinement
from phydrax.typing import checked
from tools.meshing_qualification._runtime import qualification_failure


type Point = tuple[Fraction, ...]
type Simplex = tuple[Point, ...]
type AngleScore = tuple[int, Fraction]
type Interval = tuple[Fraction, Fraction]
# Rational combination of products of square roots of nonnegative rationals:
# bit i of a key selects sqrt(radicands[i]) as a factor of that term.
type Surd = dict[int, Fraction]


class _Node(NamedTuple):
    vertices: tuple[int, ...]
    tag: int
    generation: int
    ancestor: int
    reference: int


class _References(NamedTuple):
    bounds: dict[int, AngleScore]
    counts: list[dict[str, int]]
    cache: dict[tuple[int, tuple[Fraction, ...]], tuple[AngleScore, int, int]]
    splits: list[dict[str, object]]


class _Realization(NamedTuple):
    ideal: dict[int, Point]
    published: dict[int, Point]
    displacement_bounds: dict[int, Point]
    half_ulps: dict[int, Point]
    parent_edges: dict[int, tuple[int, int]]
    midpoint_ids: dict[tuple[int, int], int]


def _realize_vertex(bank: _Realization, identity: int, low: int, high: int) -> None:
    edge = (min(low, high), max(low, high))
    if (
        identity in bank.ideal and bank.parent_edges.get(identity) != edge
    ) or bank.midpoint_ids.setdefault(edge, identity) != identity:
        _refuse("midpoint_edge", {"vertex": identity, "edge": edge})
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
    bank.parent_edges[identity] = edge


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


def _surd_product(a: Surd, b: Surd, radicands: tuple[Fraction, ...]) -> Surd:
    product: Surd = {}
    for left, x in a.items():
        for right, y in b.items():
            value = x * y
            for index, radicand in enumerate(radicands):
                if left & right & (1 << index):
                    value *= radicand
            product[left ^ right] = product.get(left ^ right, Fraction(0)) + value
    return {key: value for key, value in product.items() if value}


def _surd_sign(value: Surd, radicands: tuple[Fraction, ...]) -> int:
    """Exact sign of a surd by eliminating its highest square root.

    ``u + v sqrt(r)`` has the sign of ``u`` when ``u`` and ``v`` agree, and
    otherwise the sign of ``u`` times that of ``u*u - v*v*r``, a surd in fewer
    square roots. Radicands need not be independent or non-square.
    """
    if not value:
        return 0
    top = max(value).bit_length() - 1
    if top < 0:
        return (value[0] > 0) - (value[0] < 0)
    bit = 1 << top
    rational = {key: item for key, item in value.items() if not key & bit}
    radical = {key ^ bit: item for key, item in value.items() if key & bit}
    u, v = _surd_sign(rational, radicands), _surd_sign(radical, radicands)
    if v == 0 or radicands[top] == 0:
        return u
    if u == 0 or u == v:
        return v
    difference = _surd_product(rational, rational, radicands)
    for key, item in _surd_product(radical, radical, radicands).items():
        difference[key] = difference.get(key, Fraction(0)) - radicands[top] * item
    return u * _surd_sign(
        {key: item for key, item in difference.items() if item}, radicands
    )


def _within_perturbed_floor(
    bound: AngleScore, observed: AngleScore, first: Fraction, second: Fraction
) -> bool:
    """Decide cos(observed) <= cos(bound - asin(first) - asin(second)) exactly.

    The allowance expands by cosine addition into square roots of rational
    radicands; the decision is an exact surd sign, with equality accepted.
    """
    radicands = (
        abs(bound[1]),
        1 - abs(bound[1]),
        1 - first * first,
        1 - second * second,
        abs(observed[1]),
    )
    allowance: Surd = {
        0b01101: Fraction(bound[0]),
        0b00001: -bound[0] * first * second,
        0b01010: first,
        0b00110: second,
        0b10000: Fraction(-observed[0]),
    }
    return (
        _surd_sign({key: item for key, item in allowance.items() if item}, radicands)
        >= 0
    )


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
    inverse-trigonometric decisions; an interval overlap (an angle exactly at
    its lowered floor, as when an ideal angle at the class floor rounds along
    the tangent of its displacement ball) is decided by an exact surd sign.
    Displayed radians are diagnostic only.
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
            cosine = _cosine_interval(observed)
            if cosine[1] > allowance[0] and (
                cosine[0] > allowance[1]
                or not _within_perturbed_floor(bound, observed, first, second)
            ):
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


def _reference_floor(
    references: _References, identity: int, ancestor: int, simplex: Simplex, tag: int
) -> AngleScore:
    """Prove a complete tagged closure from ideal parent geometry, never target samples."""
    shape = (tag, _distance_key(simplex))
    reference = references.cache.get(shape)
    if reference is None:
        reference = similarity_bound(simplex, tag)
        references.cache[shape] = reference
    bound, count, tagged_count = reference
    references.bounds[identity] = bound
    references.counts.append(
        {"ancestor": ancestor, "classes": count, "tagged_classes": tagged_count}
    )
    return bound


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
    references: _References,
) -> tuple[dict[int, _Node], set[int]]:
    """Construct the leaves from declared parent labels and rational binary halves."""
    parent_vertices = np.asarray(hierarchy.record_parent_vertices)
    parent_rows = np.asarray(hierarchy.record_parent_rows)
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
        if hierarchy.refinement_rule == "whole_edge_star":
            if (
                len(set(declared)) != hierarchy.dimension + 1
                or set(declared) != set(node.vertices)
                or set(map(int, parent_rows[row])) != set(node.vertices)
                or int(parent_tags[row]) != 3
            ):
                _refuse("whole_edge_star_parent", identity)
            # Whole-star closure is allowed to announce a different edge from
            # the inherited tag. That relabeling starts a new finite tagged
            # closure; the original root's Maubach theorem no longer applies.
            node = _Node(
                _canonical(declared, 3), 3, node.generation, node.ancestor, identity
            )
            if identity not in references.bounds:
                _reference_floor(
                    references, identity, node.ancestor,
                    tuple(bank.ideal[vertex] for vertex in node.vertices), node.tag,
                )
            references.splits.append({
                "parent": identity,
                "ancestor": node.ancestor,
                "generation": node.generation,
                "edge": (node.vertices[0], node.vertices[node.tag]),
                "tag": node.tag,
                "child_tag": node.tag - 1,
                "children": tuple(map(int, child_ids[row])),
                "bound": _angle_record(references.bounds[identity]),
            })
        elif (
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
            if hierarchy.refinement_rule == "whole_edge_star":
                # Check each immediate child before a later announced edge can
                # replace its inherited reference. All geometry here is ideal.
                check_angle_bound(
                    tuple(bank.ideal[vertex] for vertex in vertices),
                    references.bounds[node.reference],
                )
            pending.append(
                (
                    int(child_id),
                    _Node(
                        _canonical(vertices, tag), tag, node.generation + 1,
                        node.ancestor, node.reference,
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
    _References,
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
    references = _References({}, [], {}, [])
    measures: dict[int, Fraction] = {}
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
        _reference_floor(references, identity, identity, cell, tag)
        measures[identity] = _measure(cell)
        pending.append(
            (identity, _Node(_canonical(ordered, tag), tag, 0, identity, identity))
        )
    return pending, references, measures


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


def _signed_measure(simplex: Simplex) -> Fraction:
    edges = tuple(_difference(point, simplex[0]) for point in simplex[1:])
    if len(edges) == 2:
        return edges[0][0] * edges[1][1] - edges[0][1] * edges[1][0]
    return _dot(edges[0], _cross(edges[1], edges[2]))


def _uniform_prefix(
    source: CellMesh,
    source_cells: dict[int, tuple[int, ...]],
    bank: _Realization,
    uniform: BisectionUniformRefinement,
    dimension: int,
) -> tuple[dict[int, tuple[int, ...]], dict[str, object]]:
    """Verify each epoch-zero cell's declared barycentric subdivision from scratch.

    Every parent must declare exactly ``(d+1)!`` distinct flag simplices
    ``(c_S1, ..., c_S(d+1))`` of strictly nested parent-vertex subsets: each
    corner's declared action is the rounded centroid weight ``1/|S|`` on ``S``
    and zero elsewhere. Ideal centroids are formed here from the epoch-zero
    rational vertices; a shared identity must name one subset and one ideal
    point in every parent, and is realized by correct rounding. Each child is
    nondegenerate with its parent's orientation, the rational child measures
    sum to the parent measure, and its block is the parent's. The children are
    returned as roots of the tagged forest.
    """
    arrays = uniform.host_arrays()
    width = dimension + 1
    arrangements = factorial(width)
    parent_ids = [int(value) for value in arrays["parent_ids"]]
    parent_rows, parent_blocks = arrays["parent_rows"], arrays["parent_blocks"]
    row_order = arrays["parent_row_order"]
    child_ids, child_vertices = arrays["child_ids"], arrays["child_vertices"]
    weights = arrays["barycentric_weights"]
    count = len(parent_ids)
    if (
        uniform.dimension != dimension
        or len(set(parent_ids)) != count
        or set(parent_ids) != set(source_cells)
        or parent_rows.shape != (count, width)
        or child_ids.shape != (count, arrangements)
        or child_vertices.shape != (count, arrangements, width)
        or weights.shape != (count, arrangements, width, width)
    ):
        _refuse("uniform_parents", sorted(parent_ids))
    source_rows = {
        int(identity): (index, row)
        for index, block in enumerate(source.blocks)
        for row, identity in enumerate(np.asarray(block.global_ids))
    }
    roots: dict[int, tuple[int, ...]] = {}
    supports: dict[int, frozenset[int]] = {}
    centroids = 0
    for parent, identity in enumerate(parent_ids):
        corners = tuple(int(value) for value in parent_rows[parent])
        block, row = source_rows[identity]
        if sorted(corners) != sorted(source_cells[identity]):
            _refuse("uniform_parent_vertices", identity)
        if int(parent_blocks[parent]) != block or int(row_order[parent]) != row:
            _refuse("uniform_parent_identity", identity)
        parent_simplex = tuple(bank.ideal[vertex] for vertex in corners)
        parent_sign = _signed_measure(parent_simplex)
        if parent_sign == 0:
            _refuse("uniform_parent_measure", identity)
        flags: set[tuple[frozenset[int], ...]] = set()
        total = Fraction(0)
        for child in range(arrangements):
            child_id = int(child_ids[parent, child])
            if child_id in roots or child_id in source_cells:
                _refuse("uniform_child", child_id)
            vertices = tuple(int(value) for value in child_vertices[parent, child])
            subsets = []
            for corner, vertex in enumerate(vertices):
                action = weights[parent, child, corner]
                support = frozenset(
                    corners[column] for column in range(width) if action[column] != 0.0
                )
                if not support or any(
                    action[column]
                    != (1.0 / len(support) if corners[column] in support else 0.0)
                    for column in range(width)
                ):
                    _refuse("uniform_centroid_weight", (child_id, corner))
                ideal = tuple(
                    sum((bank.ideal[member][axis] for member in support), Fraction(0))
                    / len(support)
                    for axis in range(dimension)
                )
                if len(support) == 1:
                    if vertex not in support:
                        _refuse("uniform_parent_corner", (child_id, vertex))
                elif supports.setdefault(vertex, support) != support or (
                    vertex in bank.ideal and bank.ideal[vertex] != ideal
                ):
                    _refuse("uniform_centroid", (child_id, vertex))
                elif vertex not in bank.ideal:
                    published, half_ulps = _rounded_point(ideal)
                    bank.ideal[vertex], bank.published[vertex] = ideal, published
                    bank.displacement_bounds[vertex] = half_ulps
                    bank.half_ulps[vertex] = half_ulps
                    centroids += 1
                subsets.append(support)
            chain = tuple(sorted(subsets, key=len))
            if [len(subset) for subset in chain] != list(range(1, width + 1)) or any(
                not first < second for first, second in zip(chain, chain[1:])
            ):
                _refuse("uniform_flag", child_id)
            if chain in flags:
                _refuse("uniform_duplicate_flag", child_id)
            flags.add(chain)
            signed = _signed_measure(tuple(bank.ideal[vertex] for vertex in vertices))
            if signed == 0 or (signed > 0) != (parent_sign > 0):
                _refuse("uniform_orientation", child_id)
            total += abs(signed) / factorial(dimension)
            roots[child_id] = vertices
        if total != abs(parent_sign) / factorial(dimension):
            _refuse("uniform_partition", identity)
    return roots, {
        "status": "certified",
        "rule": "barycentric-subdivision-flags",
        "parents": count,
        "children_per_parent": arrangements,
        "centroid_vertices": centroids,
    }


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
    if len(records) != len(parents):
        _refuse("forest", "duplicate parent identities")
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
        {},
    )
    roots = source_cells
    uniform_evidence: dict[str, object] | None = None
    if hierarchy.uniform_refinement is not None:
        roots, uniform_evidence = _uniform_prefix(
            source, source_cells, bank, hierarchy.uniform_refinement, dimension
        )
    pending, references, source_measures = _prepare_roots(
        roots,
        bank.ideal,
        hierarchy,
        records,
        active,
    )
    leaves, visited = _replay_forest(pending, bank, hierarchy, records, references)
    if set(leaves) != set(target_cells) or not set(records) <= visited:
        _refuse("forest_leaves", sorted(leaves))
    generations = []
    observed: AngleScore = (-1, Fraction(-1))
    totals = dict.fromkeys(roots, Fraction(0))
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
            int(active_generations[row]) != node.generation
            or set(target_cells[identity]) != set(node.vertices)
            or int(active_tags[row]) != node.tag
            or _canonical(tuple(map(int, active_vertices[row])), node.tag)
            != node.vertices
        ):
            _refuse("target_label", identity)
        cell = tuple(target_points[vertex] for vertex in node.vertices)
        ideal = tuple(bank.ideal[vertex] for vertex in node.vertices)
        measure = _measure(ideal)
        if measure != source_measures[node.ancestor] / 2**node.generation:
            _refuse("halved_measure", identity)
        totals[node.ancestor] += measure
        ideal_observed = max(
            ideal_observed, check_angle_bound(ideal, references.bounds[node.reference])
        )
        published_score, perturbations = _published_angles(
            ideal, cell, references.bounds[node.reference]
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
        "uniform_prefix": uniform_evidence,
        "refinement_rule": hierarchy.refinement_rule,
        "quality_floor_rule": (
            "finite-forest-announced-edge-tagged-reference-floor"
            if hierarchy.refinement_rule == "whole_edge_star"
            else "source-maubach-reference-floor"
        ),
        "generations": generations,
        "class_counts": references.counts,
        "split_bounds": references.splits,
        "bound": _angle_record(max(references.bounds.values())),
        "observed_minimum": _angle_record(observed),
        "ideal_observed_minimum": _angle_record(ideal_observed),
        "published_angle_perturbations": angle_evidence,
        "realization": {
            "status": "certified",
            "rule": "round-to-nearest-even-ideal-vertex",
            "vertices": vertex_evidence,
        },
        "arithmetic": "rational-ideal-and-certified-float64-realization",
        "shape_class_cap": (
            None if hierarchy.refinement_rule == "whole_edge_star"
            else 4 if dimension == 2 else 192
        ),
    }

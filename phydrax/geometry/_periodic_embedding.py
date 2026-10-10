#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#
"""Exact source-map contacts in a bounded scientific periodic group neighborhood.

Images are execution worksets, not new mesh entities. Lifted corner keys retain
both authored representative identity and relative winding. Bernstein hulls of
the entire source map, not its corner carrier, prove image sufficiency.
Finite linear cosets retain original generator windings; infinite-axis periods
are exact powers of those same generators, including screw translations.
"""

from __future__ import annotations

from fractions import Fraction
from itertools import product
from typing import TYPE_CHECKING

import numpy as np

from ..discretization._cell_geometry import (
    BarycentricCellGeometryElement,
    CellGeometryElement,
    CellVertexGeometryElement,
    coordinate_lagrange_element,
    LayerColumnCellGeometryElement,
    PolynomialComposedCellGeometryElement,
    RationalComposedCellGeometryElement,
    RestrictedCellGeometryElement,
)
from ..discretization._coordinate_enclosure import (
    _solve_exact,
    constant,
    coordinate_corner_images,
    coordinate_enclosure_budget,
    coordinate_expressions,
    coordinate_reference_chain,
    CoordinateEnclosureResourceError,
    CoordinateSourceBank,
    expression_bernstein_coefficients,
    expression_evaluate,
    expression_node_count,
    expression_scale,
    expression_sum,
    ExpressionComposition,
    prepared_coordinate_source_bank,
    RationalEnclosureError,
    restrict_chart_expressions,
)
from ..discretization._periodic_topology import (
    _exact_isometry_power,
    _exact_periodic_element,
    _exact_periodic_generators,
    PeriodicIsometryGroup,
    PeriodicIsometryIdentityError,
)
from ..discretization._reference_cell import reference_cell_topology
from ._mapped_embedding import (
    _bounds,
    _Cell,
    _common_atlas_extension,
    _domain,
    _injective,
    _pair_decision,
    _separating_plane,
    _tensor_gluing,
    _trace_equal,
    _vertices,
)


if TYPE_CHECKING:
    from ..discretization._cell_geometry import CellGeometrySpec
    from ..discretization._cell_mesh import CellMesh
    from ._mesh_certificates import (
        _EmbeddingState,
        MeshCertificateEntityKind,
        MeshCertificateLimits,
    )


def _transform(
    cell: _Cell, matrix: tuple[tuple[Fraction, ...], ...], vertices: tuple[int, ...]
) -> _Cell:
    coordinates = tuple(
        expression_sum(
            (
                *tuple(
                    expression_scale(value, weight)
                    for value, weight in zip(cell.coordinates, row[:-1], strict=True)
                ),
                constant(row[-1], cell.dimension),
            )
        )
        for row in matrix[:-1]
    )
    return _Cell(
        coordinates,
        cell.domain,
        cell.kind,
        cell.dimension,
        vertices,
        cell.reference_vertices,
    )


def _shared_affine_parent_traces(
    first: _Cell, second: _Cell, /
) -> tuple[tuple[int, ...], ...]:
    """Return exact entities or facet subdivisions shared by two cells."""
    first_topology = reference_cell_topology(first.kind)
    second_topology = reference_cell_topology(second.kind)
    second_vertices = set(second.vertices)
    result: dict[tuple[int, ...], None] = {}
    for entities in first_topology.entities[:-1]:
        for entity in entities:
            trace = tuple(first.vertices[index] for index in entity)
            if set(trace).issubset(second_vertices):
                result[trace] = None
    first_facets = first_topology.entities[first.dimension - 1]
    second_facets = second_topology.entities[second.dimension - 1]
    for first_facet in first_facets:
        first_trace = tuple(first.vertices[index] for index in first_facet)
        first_set = set(first_trace)
        for second_facet in second_facets:
            second_trace = tuple(second.vertices[index] for index in second_facet)
            second_set = set(second_trace)
            if second_set.issubset(first_set):
                support = tuple(
                    identifier for identifier in first_trace if identifier in second_set
                )
                if support:
                    result[support] = None
    return tuple(result)


def _affine_polytope_contact(
    first: _Cell, second: _Cell, maximum_work: int, /
) -> tuple[str, int] | None:
    """Use the exact convex-polytope separating axes of affine 3D cells."""
    from ..discretization._coordinate_enclosure import (
        _COORDINATE_BUDGET,
        RationalPolynomial,
    )

    if (
        first.kind not in ("tetrahedron", "prism")
        or second.kind not in ("tetrahedron", "prism")
        or first.dimension != 3
        or second.dimension != 3
    ):
        return None
    if any(
        isinstance(value, RationalPolynomial) or any(sum(index) > 1 for index in value)
        for cell in (first, second)
        for value in cell.coordinates
    ):
        return None
    if maximum_work <= 0:
        return "mapped_intersection_piece_budget", 0
    budget = _COORDINATE_BUDGET.get()
    if budget is not None:
        budget.reserve(2_048, 32_768)
    vertices = tuple(
        tuple(
            tuple(expression_evaluate(value, point) for value in cell.coordinates)
            for point in cell.reference_vertices
        )
        for cell in (first, second)
    )

    def subtract(
        left: tuple[Fraction, ...], right: tuple[Fraction, ...], /
    ) -> tuple[Fraction, ...]:
        return tuple(a - b for a, b in zip(left, right, strict=True))

    def cross(
        left: tuple[Fraction, ...], right: tuple[Fraction, ...], /
    ) -> tuple[Fraction, ...]:
        return (
            left[1] * right[2] - left[2] * right[1],
            left[2] * right[0] - left[0] * right[2],
            left[0] * right[1] - left[1] * right[0],
        )

    def canonical(axis: tuple[Fraction, ...], /) -> tuple[Fraction, ...] | None:
        pivot = next((value for value in axis if value), None)
        if pivot is None:
            return None
        return tuple(value / pivot for value in axis)

    axes: dict[tuple[Fraction, ...], None] = {}
    edge_vectors: list[tuple[tuple[Fraction, ...], ...]] = []
    for cell, points in zip((first, second), vertices, strict=True):
        topology = reference_cell_topology(cell.kind)
        edges = tuple(
            subtract(points[edge[1]], points[edge[0]]) for edge in topology.entities[1]
        )
        edge_vectors.append(edges)
        for facet in topology.entities[2]:
            origin = points[facet[0]]
            normal = next(
                (
                    value
                    for left, right in product(facet[1:], repeat=2)
                    if left < right
                    for value in (
                        cross(
                            subtract(points[left], origin),
                            subtract(points[right], origin),
                        ),
                    )
                    if any(value)
                ),
                None,
            )
            if normal is None:
                return None
            normalized = canonical(normal)
            if normalized is not None:
                axes[normalized] = None
    for first_edge in edge_vectors[0]:
        for second_edge in edge_vectors[1]:
            normalized = canonical(cross(first_edge, second_edge))
            if normalized is not None:
                axes[normalized] = None

    traces = tuple(set(trace) for trace in _shared_affine_parent_traces(first, second))
    work = 0
    for axis in axes:
        if work >= maximum_work:
            return "mapped_intersection_piece_budget", work
        work += 1
        if budget is not None:
            budget.reserve(256)
        projections = tuple(
            tuple(
                sum(
                    (weight * value for weight, value in zip(axis, point, strict=True)),
                    Fraction(0),
                )
                for point in points
            )
            for points in vertices
        )
        first_min, first_max = min(projections[0]), max(projections[0])
        second_min, second_max = min(projections[1]), max(projections[1])
        if first_max < second_min or second_max < first_min:
            return "separated", work
        first_support: tuple[int, ...] = ()
        if first_max == second_min:
            first_support = tuple(
                first.vertices[index]
                for index, value in enumerate(projections[0])
                if value == first_max
            )
        elif second_max == first_min:
            first_support = tuple(
                first.vertices[index]
                for index, value in enumerate(projections[0])
                if value == first_min
            )
        if first_support and any(set(first_support).issubset(trace) for trace in traces):
            return "separated", work
    return "overlap", work


def _affine_reference_contact(
    first: _Cell, second: _Cell, maximum_work: int, /
) -> tuple[str, int] | None:
    """Decide affine tetra/prism contact through canonical prism tetrahedra."""
    polytope = _affine_polytope_contact(first, second, maximum_work)
    if polytope is not None:
        return polytope
    from ._affine_mapped_contact import affine_simplex_contact

    tetrahedron = coordinate_lagrange_element("tetrahedron", 1)
    prism_tetrahedra = ((0, 1, 2, 3), (1, 2, 4, 3), (2, 4, 5, 3))

    def pieces(cell: _Cell) -> tuple[_Cell, ...] | None:
        if cell.kind == "tetrahedron":
            return (cell,)
        if cell.kind != "prism":
            return None
        corners = tuple(
            tuple(expression_evaluate(value, point) for value in cell.coordinates)
            for point in cell.reference_vertices
        )
        result = []
        for indices in prism_tetrahedra:
            coordinates = coordinate_expressions(
                tetrahedron, tuple(corners[index] for index in indices)
            )
            if coordinates is None:
                return None
            result.append(
                _Cell(
                    coordinates,
                    "simplex",
                    "tetrahedron",
                    cell.dimension,
                    tuple(cell.vertices[index] for index in indices),
                    _vertices("tetrahedron"),
                )
            )
        return tuple(result)

    left, right = pieces(first), pieces(second)
    if left is None or right is None:
        return None
    parent_traces = _shared_affine_parent_traces(first, second)
    work = 0
    for a in left:
        traces: dict[tuple[int, ...], None] = {}
        for parent_trace in parent_traces:
            parent_support = set(parent_trace)
            support = tuple(
                identifier for identifier in a.vertices if identifier in parent_support
            )
            if support and len(set(support)) <= a.dimension:
                traces[support] = None
        for b in right:
            decision = affine_simplex_contact(
                a,
                b,
                maximum_work - work,
                authoritative_first_traces=tuple(traces),
            )
            if decision is None:
                raise ValueError(
                    "Canonical affine reference tetrahedra lost their exact P1 source."
                )
            status, used = decision
            work += used
            if status != "separated":
                return status, work
    return "separated", work


def _layer_fiber_reference_cell(
    element: LayerColumnCellGeometryElement,
    source_local: CoordinateSourceBank,
    chart_coordinates: tuple | None,
    cell: _Cell,
    /,
) -> _Cell | None:
    """Pull one root or child chart back to its exact affine fiber carrier."""
    affine_coordinates = element.fiber_reference_expressions(source_local)
    if affine_coordinates is None:
        return None
    if chart_coordinates is None:
        reference_coordinates = affine_coordinates
    else:
        composition = ExpressionComposition(chart_coordinates)
        reference_coordinates = tuple(composition(value) for value in affine_coordinates)
    corner_values = tuple(
        tuple(expression_evaluate(value, point) for value in reference_coordinates)
        for point in cell.reference_vertices
    )
    canonical_reference = coordinate_expressions(
        coordinate_lagrange_element(cell.kind, 1), corner_values
    )
    if canonical_reference is None:
        raise ValueError("A layer fiber child lost its exact affine reference chart.")
    return _Cell(
        canonical_reference,
        cell.domain,
        cell.kind,
        cell.dimension,
        cell.vertices,
        cell.reference_vertices,
    )


def _source_contact_quantities(
    first: _Cell,
    second: _Cell,
    matrix: tuple[tuple[Fraction, ...], ...],
    exponents: tuple[int, ...],
    first_piece: int,
    second_piece: int,
    first_owner: int,
    second_owner: int,
    /,
) -> tuple[tuple[str, int], ...]:
    """Actual offending SCI action/incidence, never a reconstructed proxy."""
    quantities = [
        ("first_piece", first_piece),
        ("second_piece", second_piece),
        ("first_owner", first_owner),
        ("second_owner", second_owner),
        ("shared_vertex_count", len(set(first.vertices).intersection(second.vertices))),
    ]
    for axis, exponent in enumerate(exponents):
        quantities.extend(
            (
                (f"image_exponent_{axis}_positive", max(0, exponent)),
                (f"image_exponent_{axis}_negative", max(0, -exponent)),
            )
        )
    for row, values in enumerate(matrix):
        for column, value in enumerate(values):
            quantities.extend(
                (
                    (
                        f"action_{row}_{column}_numerator_positive",
                        max(0, value.numerator),
                    ),
                    (
                        f"action_{row}_{column}_numerator_negative",
                        max(0, -value.numerator),
                    ),
                    (f"action_{row}_{column}_denominator", value.denominator),
                )
            )
    for label, cell in (("first", first), ("second", second)):
        quantities.extend(
            (f"{label}_vertex_{corner}", identifier)
            for corner, identifier in enumerate(cell.vertices)
        )
    encoded = []
    for name, value in quantities:
        if value <= (1 << 53) - 1:
            encoded.append((name, value))
            continue
        # Public failure quantities are binary64. Preserve arbitrary exact SCI
        # integer bits as actual little-endian 32-bit limbs, never float(value).
        limb = 0
        while value:
            encoded.append((f"{name}_limb_{limb}", value & ((1 << 32) - 1)))
            value >>= 32
            limb += 1
        encoded.append((f"{name}_limb_count", limb))
    return tuple(encoded)


def _has_full_coefficient_source(element: CellGeometryElement) -> bool:
    """Nominal source-law dispatch; full W is not a Cartesian chart."""
    while isinstance(
        element,
        (
            RestrictedCellGeometryElement,
            PolynomialComposedCellGeometryElement,
            RationalComposedCellGeometryElement,
        ),
    ):
        element = element.source_element
    return isinstance(element, BarycentricCellGeometryElement)


def _closed_source_polygon_contains(
    source: tuple[tuple[Fraction, ...], ...],
    subject: tuple[tuple[Fraction, ...], ...],
) -> bool:
    """Contain a complete loop in one authored simple polygon, without a hull."""
    from ._planar_coverage import _reserve_fraction_work, turn

    def inside(point: tuple[Fraction, ...]) -> bool:
        winding = 0
        for a, b in zip(source, (*source[1:], source[0]), strict=True):
            side = turn(a, b, point)
            if not side and all(
                min(x, y) <= p <= max(x, y) for x, y, p in zip(a, b, point, strict=True)
            ):
                return True
            if a[1] <= point[1] < b[1] and side > 0:
                winding += 1
            elif b[1] <= point[1] < a[1] and side < 0:
                winding -= 1
        return bool(winding)

    if not source or not subject or not all(inside(point) for point in subject):
        return False
    for a, b in zip(subject, (*subject[1:], subject[0]), strict=True):
        if a == b:
            continue
        parameters = {Fraction(0), Fraction(1)}
        for c, d in zip(source, (*source[1:], source[0]), strict=True):
            _reserve_fraction_work((a, b, c, d), 48, 4, 16)
            r = tuple(y - x for x, y in zip(a, b, strict=True))
            s = tuple(y - x for x, y in zip(c, d, strict=True))
            q = tuple(y - x for x, y in zip(a, c, strict=True))
            divisor = r[0] * s[1] - r[1] * s[0]
            if divisor:
                t = (q[0] * s[1] - q[1] * s[0]) / divisor
                u = (q[0] * r[1] - q[1] * r[0]) / divisor
                if 0 <= t <= 1 and 0 <= u <= 1:
                    parameters.add(t)
            elif not turn(a, b, c):
                axis = next(index for index, value in enumerate(r) if value)
                for point in (c, d):
                    t = (point[axis] - a[axis]) / r[axis]
                    if 0 <= t <= 1:
                        parameters.add(t)
        ordered = sorted(parameters)
        for low, high in zip(ordered, ordered[1:]):
            _reserve_fraction_work((a, b, (low, high)), 8, 3, 8)
            middle = (low + high) / 2
            point = tuple(x + middle * (y - x) for x, y in zip(a, b, strict=True))
            if not inside(point):
                return False
    # A simple authored polygon has connected exterior. A contained closed
    # subject boundary cannot enclose an exterior component without crossing it.
    return True


def _source_polygon_plane(
    points: tuple[tuple[Fraction, ...], ...],
) -> tuple[tuple[Fraction, ...], tuple[int, ...], int, int] | None:
    from ._planar_coverage import _reserve_fraction_work, plane_key

    for a in range(1, len(points)):
        for b in range(a + 1, len(points)):
            plane = plane_key((points[0], points[a], points[b]))
            if plane is not None:
                key = plane[0]
                for point in points:
                    _reserve_fraction_work((key, point), 2 * len(point), 1, 4)
                    if sum(
                        (a * b for a, b in zip(key[:-1], point, strict=True)), key[-1]
                    ):
                        return None
                return plane
    return None


def _original_power_facet_authority(
    geometry: CellGeometrySpec,
    group: PeriodicIsometryGroup,
) -> tuple[
    tuple[int, tuple[Fraction, ...], tuple[int, ...], tuple[tuple[Fraction, ...], ...]],
    ...,
]:
    from ..discretization._coordinate_enclosure import _COORDINATE_BUDGET
    from ..discretization._exact_power_geometry import (
        _power_domain_binding,
        _power_preparation_binding,
        ExactPowerCellGeometryLinearActionSource,
        ExactPowerCellGeometryRestrictionSource,
        ExactPowerCellGeometrySource,
    )
    from ..meshing._controls import PeriodicConstraint
    from ..meshing._volume_generation import PiecewiseLinearComplex
    from ._planar_coverage import project, rational_points
    from ._triangulation import PeriodicPowerPreparation

    source = geometry.exact_source
    while isinstance(
        source,
        (
            ExactPowerCellGeometryLinearActionSource,
            ExactPowerCellGeometryRestrictionSource,
        ),
    ):
        source = source.parent
    if not isinstance(source, ExactPowerCellGeometrySource):
        return ()
    domain, constraints = source._domain_owner()
    if domain is None or not constraints:
        return ()
    if source.authority_binding != _power_domain_binding(domain, constraints):
        raise ValueError(
            "Original periodic source facet authority no longer matches "
            "its owning binding."
        )
    preparation = source._periodic_owner()
    if (
        not isinstance(preparation, PeriodicPowerPreparation)
        or not isinstance(preparation.periodic_group, PeriodicIsometryGroup)
        or preparation.periodic_group.group_id != group.group_id
    ):
        raise ValueError(
            "Original periodic source facet authority requires the actual original group."
        )
    if source.periodic_source_binding != _power_preparation_binding(preparation):
        raise ValueError(
            "Original periodic source facet authority requires the complete original source payload."
        )
    from .._fingerprint import array_tree_fingerprint

    if array_tree_fingerprint(preparation.periodic_group) != array_tree_fingerprint(
        group
    ):
        raise ValueError(
            "Original periodic source facet authority requires the complete original group leaves."
        )
    domain = source.domain_source
    if not isinstance(domain, PiecewiseLinearComplex):
        raise TypeError(
            "Original periodic facet authority requires its owning PLC source."
        )
    budget = _COORDINATE_BUDGET.get()
    if budget is not None:
        budget.reserve(
            domain.facet_regions.shape[0], 64 + 64 * domain.facet_regions.shape[0]
        )
    roots = list(range(domain.facet_regions.shape[0]))

    def root(value: int) -> int:
        while roots[value] != value:
            if budget is not None:
                budget.reserve(1)
            roots[value] = roots[roots[value]]
            value = roots[value]
        return value

    for constraint in source.periodic_constraints:
        if not isinstance(constraint, PeriodicConstraint):
            raise TypeError(
                "Original periodic facet authority requires original typed controls."
            )
        if (
            constraint.source_scope.entity_dimension != 2
            or constraint.target_scope.entity_dimension != 2
        ):
            continue
        targets = np.asarray(constraint.target_scope.entity_ids)
        if constraint.source_entity_ids is None:
            sources = np.asarray(constraint.source_scope.entity_ids)
            # A unique scope pair declares one facet correspondence. Larger
            # unpaired scopes do not identify same-side facets with each other.
            # Only the existing complete SCI trace proof may authorize them.
            pairs = (
                ((int(sources[0]), int(targets[0])),)
                if len(sources) == len(targets) == 1
                else ()
            )
        else:
            pairs = zip(np.asarray(constraint.source_entity_ids), targets, strict=True)
        for a, b in pairs:
            if budget is not None:
                budget.reserve(1, 32)
            a, b = int(a), int(b)
            if not 0 <= a < len(roots) or not 0 <= b < len(roots):
                raise ValueError(
                    "Original periodic source scopes index undeclared PLC facets."
                )
            ra, rb = root(a), root(b)
            roots[max(ra, rb)] = min(ra, rb)
    points = rational_points(np.asarray(domain.vertices))
    from ._exact_polyhedral_geometry import triangulate_loop, triangulation_charge

    if budget is not None:
        budget.reserve(0, 256 + 24 * len(points))
    point_array = np.asarray(points, dtype=object)
    result = []
    offsets, indices = (
        np.asarray(domain.polygon_offsets),
        np.asarray(domain.polygon_vertices),
    )
    for polygon, facet in enumerate(np.asarray(domain.polygon_facets)):
        row = tuple(map(int, indices[offsets[polygon] : offsets[polygon + 1]]))
        work, storage = triangulation_charge(point_array, row)
        if budget is not None:
            budget.admit_work_bound(work)
            budget.reserve(work, storage)
        triangulate_loop(point_array, row)
        loop = tuple(points[index] for index in row)
        plane = _source_polygon_plane(loop)
        if plane is None:
            raise ValueError("Original periodic source polygon has no supporting plane.")
        key, axes_, _, _ = plane
        if budget is not None:
            budget.reserve(1, 256)
        result.append((root(int(facet)), key, axes_, project(loop, axes_)))
    return tuple(result)


def _translation_ranges(
    boxes: list[tuple[np.ndarray, np.ndarray]], vectors: tuple[tuple[Fraction, ...], ...]
) -> tuple[range, ...]:
    if not vectors:
        return ()
    # Independent translation generators give a left inverse on their span.
    # Applying it to any difference T*n returns n, even for a partial lattice.
    gram = [
        [sum((a * b for a, b in zip(x, y, strict=True)), Fraction(0)) for y in vectors]
        for x in vectors
    ]
    inverse = _solve_exact(
        gram,
        [[Fraction(i == j) for j in range(len(vectors))] for i in range(len(vectors))],
    )
    dual = tuple(
        tuple(
            sum((inverse[i][k] * vectors[k][j] for k in range(len(vectors))), Fraction(0))
            for j in range(len(vectors[0]))
        )
        for i in range(len(vectors))
    )
    ranges = []
    for row in dual:
        minima, maxima = [], []
        for lower, upper in boxes:
            minima.append(
                sum(
                    (
                        weight * Fraction(float(lo if weight >= 0 else hi))
                        for weight, lo, hi in zip(row, lower, upper, strict=True)
                    ),
                    Fraction(0),
                )
            )
            maxima.append(
                sum(
                    (
                        weight * Fraction(float(hi if weight >= 0 else lo))
                        for weight, lo, hi in zip(row, lower, upper, strict=True)
                    ),
                    Fraction(0),
                )
            )
        diameter = max(maxima) - min(minima)
        limit = diameter.numerator // diameter.denominator
        ranges.append(range(-limit, limit + 1))
    return tuple(ranges)


def _planar_polyhedral_pieces(
    mesh: CellMesh,
    first_cell: int,
    count: int,
    vertices: np.ndarray,
    valid: np.ndarray,
    source: CoordinateSourceBank,
) -> tuple[np.ndarray, np.ndarray]:
    from .._geometry_predicates import PredicateMode
    from ._supermesh import _cone_decomposition, _polyhedral_face_triangles

    triangles, triangle_valid = _polyhedral_face_triangles(
        mesh.connectivity, first_cell, count
    )
    vertices = np.asarray(vertices, dtype=np.int64)
    ordered = np.sort(np.where(valid, vertices, np.iinfo(np.int64).max), axis=1)
    candidate_valid = ordered != np.iinfo(np.int64).max
    apex, _, invalid, uncertain = _cone_decomposition(
        np.asarray(mesh.coordinates, dtype=np.float64),
        triangles,
        triangle_valid,
        np.where(candidate_valid, ordered, 0),
        candidate_valid,
        PredicateMode.EXACT,
    )
    if np.any(invalid) or np.any(uncertain):
        raise RationalEnclosureError(
            "Packed planar polyhedral star decomposition is not certified."
        )
    from ._mesh_certificates import _det3

    owners = []
    pieces = []
    for cell in range(count):
        signs = set()
        cell_pieces = []
        anchor = int(apex[cell])
        for triangle in triangles[cell][triangle_valid[cell]]:
            # The numerical carrier proposes a star vertex only. Every cone
            # sign and every retained simplex is decided from actual source
            # coefficients, including ideal rational power vertices.
            vectors = tuple(
                tuple(
                    value - origin
                    for value, origin in zip(
                        source[int(vertex)], source[anchor], strict=True
                    )
                )
                for vertex in triangle
            )
            determinant = _det3(
                *tuple(np.asarray(vector, dtype=object) for vector in vectors)
            )
            if determinant:
                signs.add(1 if determinant > 0 else -1)
                cell_pieces.append((anchor, *triangle.tolist()))
        if len(signs) != 1:
            raise RationalEnclosureError(
                "The proposed polyhedral star is not valid in the exact source."
            )
        owners.extend([cell] * len(cell_pieces))
        pieces.extend(cell_pieces)
    return np.asarray(owners, dtype=np.int64), np.asarray(pieces, dtype=np.int64)


def _tensor_pair(
    first: _Cell, second: _Cell, state: _EmbeddingState, limits: MeshCertificateLimits
) -> bool:
    shared = np.asarray(
        sorted(set(first.vertices).intersection(second.vertices)), dtype=np.int64
    )
    gluing = _tensor_gluing(first, second, shared)
    if gluing is None:
        return False
    matrix, offset = gluing
    reference = np.asarray(
        reference_cell_topology(second.kind).vertices, dtype=np.float64
    )
    combined = np.concatenate(
        (
            np.asarray(reference_cell_topology(first.kind).vertices, dtype=np.float64),
            reference @ matrix.T + offset,
        )
    )
    lower = np.min(combined, axis=0)
    spans = np.max(combined, axis=0) - lower
    return _common_atlas_extension(
        [first, second],
        {
            0: (
                np.eye(first.dimension, dtype=np.float64),
                np.zeros(first.dimension, dtype=np.float64),
            ),
            1: (matrix, offset),
        },
        lower,
        spans,
        state,
        limits,
    )


def _root_action_pair(
    root: _Cell,
    first: _Cell,
    second: _Cell,
    physical_action: tuple[tuple[Fraction, ...], ...],
    root_corners: tuple[tuple[Fraction, ...], ...],
    state: _EmbeddingState,
    limits: MeshCertificateLimits,
) -> bool:
    """Prove a physical group action in one explicitly owned source reference.

    An affine corner chart proposes an action; exact equality of the complete
    root source expressions proves it. Corner agreement alone proves nothing.
    """
    from itertools import combinations

    dimension = root.dimension
    reference = tuple(
        tuple(Fraction(float(value)) for value in row)
        for row in reference_cell_topology(root.kind).vertices
    )
    basis = next(
        (
            indices
            for indices in combinations(range(1, len(reference)), dimension)
            if np.linalg.matrix_rank(
                np.asarray(
                    [
                        [float(reference[j][i] - reference[0][i]) for j in indices]
                        for i in range(dimension)
                    ],
                    dtype=np.float64,
                )
            )
            == dimension
        ),
        None,
    )
    if basis is None:
        return False
    reference_columns = [
        [reference[j][i] - reference[0][i] for i in range(dimension)] for j in basis
    ]
    physical_columns = [
        [root_corners[j][i] - root_corners[0][i] for i in range(dimension)] for j in basis
    ]
    coefficients = _solve_exact(reference_columns, physical_columns)
    affine = [[coefficients[j][i] for j in range(dimension)] for i in range(dimension)]
    if (
        np.linalg.matrix_rank(
            np.asarray(
                [[float(value) for value in row] for row in affine], dtype=np.float64
            )
        )
        != dimension
    ):
        return False
    center = tuple(
        root_corners[0][i]
        - sum((affine[i][j] * reference[0][j] for j in range(dimension)), Fraction(0))
        for i in range(dimension)
    )
    rotated = [
        [
            sum(
                (physical_action[i][k] * affine[k][j] for k in range(dimension)),
                Fraction(0),
            )
            for j in range(dimension)
        ]
        for i in range(dimension)
    ]
    displacement = [
        [
            sum(
                (physical_action[i][k] * center[k] for k in range(dimension)), Fraction(0)
            )
            + physical_action[i][-1]
            - center[i]
        ]
        for i in range(dimension)
    ]
    matrix_exact = _solve_exact(affine, rotated)
    offset_exact = tuple(row[0] for row in _solve_exact(affine, displacement))
    matrix = np.asarray(
        [[float(value) for value in row] for row in matrix_exact], dtype=np.float64
    )
    offset = np.asarray([float(value) for value in offset_exact], dtype=np.float64)
    if any(
        Fraction(float(value)) != exact
        for row, exact_row in zip(matrix, matrix_exact, strict=True)
        for value, exact in zip(row, exact_row, strict=True)
    ) or any(
        Fraction(float(value)) != exact
        for value, exact in zip(offset, offset_exact, strict=True)
    ):
        return False
    action = tuple(
        tuple((*row, start))
        for row, start in zip(matrix_exact, offset_exact, strict=True)
    ) + ((Fraction(0),) * dimension + (Fraction(1),),)
    image_chart = _transform(second, action, second.vertices)
    box_kind = "quadrilateral" if dimension == 2 else "hexahedron"
    identity = np.eye(dimension, dtype=np.float64)
    zero = np.zeros(dimension, dtype=np.float64)
    root_on_box = restrict_chart_expressions(
        root.coordinates, root.kind, box_kind, zero, identity
    )
    shifted = restrict_chart_expressions(
        root.coordinates, root.kind, box_kind, offset, matrix
    )
    transformed = _transform(
        _Cell(root_on_box, "box", box_kind, dimension, (), ()), physical_action, ()
    ).coordinates
    if shifted != transformed:
        return False
    boxes = (_bounds(first, zero, identity), _bounds(image_chart, zero, identity))
    lower = np.minimum(boxes[0][0], boxes[1][0])
    upper = np.maximum(boxes[0][1], boxes[1][1])
    spans = np.nextafter(upper - lower, np.inf)
    extension = _Cell(
        restrict_chart_expressions(
            root.coordinates, root.kind, box_kind, lower, np.diag(spans)
        ),
        "box",
        box_kind,
        dimension,
        (),
        (),
    )
    if any(
        expression_node_count(value, "box", dimension) > limits.maximum_bernstein_nodes
        for value in extension.coordinates
    ):
        return False
    if _injective(extension, limits, state) is not None:
        return False
    decision, work = _pair_decision(
        first,
        image_chart,
        limits,
        limits.maximum_subdivision_pieces - state.subdivision_pieces,
    )
    state.subdivision_pieces += work
    return decision == "separated"


def _inverse_equivalent_pair_key(
    first: int,
    second: int,
    exponents: tuple[int, ...],
    orders: tuple[int, ...],
    /,
) -> tuple[int, int, tuple[int, ...]]:
    """Identify the same contact after applying the inverse group action."""

    inverse = tuple(
        (-exponent) % order if order else -exponent
        for exponent, order in zip(exponents, orders, strict=True)
    )
    forward = (first, second, exponents)
    reverse = (second, first, inverse)
    return min(forward, reverse)


def certify_periodic_mapped_embedding(
    state: _EmbeddingState,
    mesh: CellMesh,
    geometry: CellGeometrySpec,
    cell_ids: np.ndarray,
    limits: MeshCertificateLimits,
    /,
) -> int:
    """Populate the canonical global certificate; return enumerated image count."""
    state.checks.extend(
        (
            "periodic_authored_group_identity",
            "periodic_full_source_hulls",
            "periodic_image_sufficiency",
            "periodic_mapped_trace_equivariance",
            "periodic_mapped_self_image_embedding",
        )
    )
    topology = mesh.periodic_topology
    if topology is None:
        raise ValueError("Periodic mapped embedding requires quotient topology.")
    maximum_images = limits.maximum_periodic_images
    image_count = 0
    budget = coordinate_enclosure_budget(
        limits.maximum_work_units, limits.maximum_scratch_bytes
    )
    resource_kind: MeshCertificateEntityKind = "mesh"
    resource_ids: tuple[int, ...] = ()
    try:
        with (
            budget.activate(),
            budget.bound_stage(limits.maximum_work_units, limits.maximum_scratch_bytes),
        ):
            matrices, orders = _exact_periodic_generators(topology.cell)
            if isinstance(topology.cell, PeriodicIsometryGroup):
                linear_orders = topology.cell.linear_orders
                periods = topology.cell.translation_periods
            else:
                linear_orders = tuple(1 for _ in orders)
                periods = tuple(1 if not order else 0 for order in orders)
            translations = tuple(
                tuple(row[-1] for row in period_matrix[:-1])
                for matrix, period in zip(matrices, periods, strict=True)
                if period
                for period_matrix in (_exact_isometry_power(matrix, period),)
            )
            finite_count = 1
            for order in linear_orders:
                finite_count *= order
            if finite_count > maximum_images:
                state.add("periodic_image_budget", "unresolved", "mesh")
                return finite_count
            finite = []
            for exponents in product(*(range(order) for order in linear_orders)):
                transform = _exact_periodic_element(matrices, orders, exponents)
                finite.append((exponents, transform))
            elements, routes, coordinates = geometry.resolve(mesh)
            points = np.asarray(coordinates, dtype=np.float64)
            source_bank = prepared_coordinate_source_bank(geometry)
            roots = np.asarray(topology.vertex_representatives, dtype=np.int64)
            shifts = np.asarray(topology.vertex_shifts, dtype=np.int64)
            fixed = np.asarray(topology.vertex_fixed_generators, dtype=np.bool_)
            keys: dict[tuple[int, ...], int] = {}

            def identifiers(
                vertices: np.ndarray, image: tuple[int, ...]
            ) -> tuple[int, ...]:
                result = []
                for vertex in vertices.tolist():
                    exponents = tuple(
                        0
                        if is_fixed
                        else (
                            (int(value) + shift) % order if order else int(value) + shift
                        )
                        for value, shift, order, is_fixed in zip(
                            shifts[vertex], image, orders, fixed[vertex], strict=True
                        )
                    )
                    key = (int(roots[vertex]), *exponents)
                    result.append(keys.setdefault(key, len(keys)))
                return tuple(result)

            cells: list[_Cell] = []
            owners: list[int] = []
            piece_vertices: list[np.ndarray] = []
            planar_polyhedral_owners: set[int] = set()
            source_roots: dict[
                int,
                tuple[
                    int,
                    _Cell,
                    _Cell,
                    tuple[tuple[Fraction, ...], ...],
                    str,
                    tuple[int, ...],
                ],
            ] = {}
            layer_reference_cells: dict[int, _Cell] = {}
            layer_column_owners: set[int] = set()
            record = geometry.restriction_source
            first_cell = 0
            for block, element, route in zip(mesh.blocks, elements, routes, strict=True):
                resource_kind = "cell"
                resource_ids = tuple(
                    int(value)
                    for value in cell_ids[first_cell : first_cell + block.cell_count]
                )
                if (
                    isinstance(element, CellVertexGeometryElement)
                    and block.cell_kind == "polyhedron"
                ):
                    if not np.array_equal(
                        np.asarray(route), np.asarray(block.vertices)
                    ) or not np.array_equal(points, np.asarray(mesh.coordinates)):
                        state.add(
                            "periodic_packed_polyhedral_source_identity",
                            "unresolved",
                            "cell",
                            cell_ids[first_cell : first_cell + block.cell_count],
                        )
                        return image_count
                    planar_polyhedral_owners.update(
                        range(first_cell, first_cell + block.cell_count)
                    )
                    budget.reserve(block.vertices.size, block.vertices.size * 64)
                    local_owners, rows = _planar_polyhedral_pieces(
                        mesh,
                        first_cell,
                        block.cell_count,
                        np.asarray(block.vertices),
                        np.asarray(block.vertex_valid),
                        source_bank,
                    )
                    source_rows = [
                        (
                            int(owner),
                            "tetrahedron",
                            coordinate_lagrange_element("tetrahedron", 1),
                            vertices,
                            vertices,
                        )
                        for owner, vertices in zip(local_owners, rows, strict=True)
                    ]
                else:
                    source_rows = [
                        (owner, block.cell_kind, element, row, vertices)
                        for owner, (row, vertices) in enumerate(
                            zip(
                                np.asarray(route), np.asarray(block.vertices), strict=True
                            )
                        )
                    ]
                for local_owner, kind, source_element, row, vertices in source_rows:
                    owner = first_cell + local_owner
                    resource_ids = (int(cell_ids[owner]),)
                    source_local = tuple(source_bank[int(index)] for index in row)
                    expressions = coordinate_expressions(source_element, source_local)
                    if expressions is None:
                        state.add(
                            "periodic_coordinate_source_expression",
                            "unresolved",
                            "cell",
                            cell_ids[owner : owner + 1],
                        )
                        return image_count
                    if any(
                        expression_node_count(
                            value, _domain(kind), mesh.topological_dimension
                        )
                        > limits.maximum_bernstein_nodes
                        for value in expressions
                    ):
                        state.add(
                            "periodic_bernstein_node_budget",
                            "unresolved",
                            "cell",
                            cell_ids[owner : owner + 1],
                        )
                        return image_count
                    cell = _Cell(
                        expressions,
                        _domain(kind),
                        kind,
                        mesh.topological_dimension,
                        identifiers(vertices, (0,) * len(orders)),
                        _vertices(kind),
                    )
                    root_domain_certified = False
                    layer_column_root = (
                        isinstance(source_element, LayerColumnCellGeometryElement)
                        and source_element.fiber_graph
                    )
                    if layer_column_root:
                        layer_column_owners.add(owner)
                        reference_cell = _layer_fiber_reference_cell(
                            source_element,
                            source_local,
                            None,
                            cell,
                        )
                        if reference_cell is None:
                            raise ValueError(
                                "A declared layer fiber graph lost its affine carrier."
                            )
                        layer_reference_cells[owner] = reference_cell
                    if (
                        record is not None
                        and not isinstance(source_element, CellVertexGeometryElement)
                        and not _has_full_coefficient_source(source_element)
                    ):
                        root_element, chart_coordinates = coordinate_reference_chain(
                            source_element
                        )
                        layer_column_root |= (
                            isinstance(root_element, LayerColumnCellGeometryElement)
                            and root_element.fiber_graph
                        )
                        root_coordinates = coordinate_expressions(
                            root_element, source_local
                        )
                        root_corners = coordinate_corner_images(
                            root_element, source_local
                        )
                        if (
                            isinstance(root_element, LayerColumnCellGeometryElement)
                            and root_element.fiber_graph
                        ):
                            layer_column_owners.add(owner)
                            reference_cell = _layer_fiber_reference_cell(
                                root_element,
                                source_local,
                                chart_coordinates,
                                cell,
                            )
                            if reference_cell is None:
                                raise ValueError(
                                    "A declared layer fiber graph lost its affine carrier."
                                )
                            layer_reference_cells[owner] = reference_cell
                        if root_coordinates is not None and root_corners is not None:
                            parent = int(
                                np.asarray(record.block_parent_cell_ids[block.name])[
                                    local_owner
                                ]
                            )
                            root = _Cell(
                                root_coordinates,
                                _domain(root_element.cell_kind),
                                root_element.cell_kind,
                                mesh.topological_dimension,
                                (),
                                _vertices(root_element.cell_kind),
                            )
                            chart = _Cell(
                                chart_coordinates,
                                cell.domain,
                                cell.kind,
                                cell.dimension,
                                cell.vertices,
                                cell.reference_vertices,
                            )
                            parent_vertices = tuple(
                                int(value)
                                for value in np.asarray(
                                    record.block_parent_vertex_ids[block.name]
                                )[local_owner]
                                if value >= 0
                            )
                            source_roots[owner] = (
                                parent,
                                root,
                                chart,
                                root_corners,
                                root_element.element_id,
                                parent_vertices,
                            )
                            from ._restricted_embedding import _reference_constraints

                            root_domain_certified = all(
                                min(
                                    expression_bernstein_coefficients(
                                        value, cell.domain, cell.dimension
                                    )
                                )
                                >= 0
                                for value in _reference_constraints(
                                    chart_coordinates, root.kind, cell.dimension
                                )
                            )
                    root_record = source_roots.get(owner)
                    if layer_column_root and "cell_validity" in state.checks:
                        # The owning exact Bernstein validity certificate has
                        # already proved this complete column map injective.
                        # Reuse that bound; the checks below still certify all
                        # quotient traces, images, and inter-cell contacts.
                        reason = None
                        if "periodic_layer_column_validity_reuse" not in state.checks:
                            state.checks.append("periodic_layer_column_validity_reuse")
                    elif root_record is None or not root_domain_certified:
                        reason = _injective(cell, limits, state)
                    else:
                        reason = _injective(root_record[1], limits, state) or _injective(
                            root_record[2], limits, state
                        )
                    if reason is not None:
                        state.add(
                            reason, "unresolved", "cell", cell_ids[owner : owner + 1]
                        )
                    cells.append(cell)
                    owners.append(owner)
                    piece_vertices.append(vertices)
                first_cell += block.cell_count
            if layer_column_owners:
                if layer_column_owners != set(layer_reference_cells):
                    raise ValueError(
                        "Layer fiber contact proof requires every declared child chart."
                    )
                state.checks.extend(
                    (
                        "periodic_layer_column_graph_source",
                        "periodic_layer_column_affine_reference_contact",
                    )
                )
            resource_kind, resource_ids = "mesh", ()
            from ._mesh_certificates import _facet_entity_ids, _mesh_facets

            facets = _mesh_facets(mesh)
            entity_ids = _facet_entity_ids(mesh, facets.rows)
            facet_rows = {
                int(value): row
                for row, value in enumerate(
                    np.asarray(mesh.entity_set(mesh.topological_dimension - 1).entity_ids)
                )
            }
            orbits, _, anchors = (
                np.asarray(value)
                for value in topology.orbits(mesh.topological_dimension - 1)
            )
            from ..discretization._periodic_topology import _lifted_loops

            edge_orbits = np.asarray(topology.orbits(1)[0])
            edge_rows = {}
            for rows_, corners_ in _lifted_loops(mesh, 1):
                edge_rows.update(
                    (frozenset(int(vertex) for vertex in corners), int(row_))
                    for row_, corners in zip(rows_, corners_, strict=True)
                )
            vertex_orbits = np.asarray(topology.orbits(0)[0])
            # Authored polyhedral facets, not the auxiliary tetrahedral fan.
            # Source keys give incidence; exact original coefficients separately
            # authenticate every coincident face point under the actual action.
            parent_faces = {}
            for occurrence, entity in enumerate(entity_ids.tolist()):
                owner = int(facets.cells[occurrence])
                if owner not in planar_polyhedral_owners:
                    continue
                row_vertices = facets.rows[occurrence]
                row_vertices = row_vertices[row_vertices >= 0]
                budget.reserve(len(row_vertices), 128 + 64 * len(row_vertices))
                face_keys = identifiers(row_vertices, (0,) * len(orders))
                parent_faces.setdefault(owner, []).append(
                    (
                        mesh.topological_dimension - 1,
                        int(orbits[facet_rows[entity]]),
                        tuple(int(vertex) for vertex in row_vertices),
                        face_keys,
                    )
                )
                # Close the AUTHORED source facet downward, retaining the
                # canonical edge/vertex orbit, not an auxiliary fan diagonal.
                budget.reserve(2 * len(row_vertices), 512 * len(row_vertices))
                for vertex in row_vertices:
                    vertex_ = int(vertex)
                    parent_faces[owner].append(
                        (
                            0,
                            int(vertex_orbits[vertex_]),
                            (vertex_,),
                            identifiers(
                                np.asarray((vertex_,), dtype=np.int64), (0,) * len(orders)
                            ),
                        )
                    )
                for position, vertex in enumerate(row_vertices):
                    edge = (
                        int(vertex),
                        int(row_vertices[(position + 1) % len(row_vertices)]),
                    )
                    edge_row = edge_rows[frozenset(edge)]
                    parent_faces[owner].append(
                        (
                            1,
                            int(edge_orbits[edge_row]),
                            edge,
                            identifiers(
                                np.asarray(edge, dtype=np.int64), (0,) * len(orders)
                            ),
                        )
                    )

            by_orbit = {}
            for owner, entities in parent_faces.items():
                unique = {
                    (degree, frozenset(vertices_)): (degree, orbit, vertices_, keys_)
                    for degree, orbit, vertices_, keys_ in entities
                }
                parent_faces[owner] = list(unique.values())
                lookup = {}
                for degree, orbit, vertices_, keys_ in parent_faces[owner]:
                    lookup.setdefault((degree, orbit), []).append((vertices_, keys_))
                by_orbit[owner] = lookup

            from ._planar_coverage import project

            original_facets = (
                _original_power_facet_authority(geometry, topology.cell)
                if isinstance(topology.cell, PeriodicIsometryGroup)
                else ()
            )
            source_faces = {}
            for owner, entities in parent_faces.items() if original_facets else ():
                for degree, _, vertices_, _ in entities:
                    if degree != mesh.topological_dimension - 1:
                        continue
                    points_ = tuple(source_bank[vertex] for vertex in vertices_)
                    plane = _source_polygon_plane(points_)
                    if plane is None:
                        continue
                    key = plane[0]
                    budget.reserve(1, 256)
                    authorities = set()
                    for facet_root, source_plane, axes_, polygon in original_facets:
                        budget.reserve(1)
                        if source_plane == key and _closed_source_polygon_contains(
                            polygon,
                            project(points_, axes_),
                        ):
                            if facet_root not in authorities:
                                budget.reserve(1, 128)
                            authorities.add(facet_root)
                    if authorities:
                        budget.reserve(len(authorities), 128 + 32 * len(authorities))
                        source_faces[(owner, vertices_)] = (key, authorities)

            def authored_traces(
                first: _Cell,
                owner_a: int,
                owner_b: int,
                action: tuple[tuple[Fraction, ...], ...],
                image: tuple[int, ...],
                first_source_vertices: np.ndarray,
            ) -> tuple[tuple[int, ...], ...]:
                if owner_a not in parent_faces or owner_b not in parent_faces:
                    return ()
                if (
                    identifiers(first_source_vertices, (0,) * len(orders))
                    != first.vertices
                ):
                    raise ValueError(
                        "Original periodic trace requires its complete first-piece SCI source bank."
                    )
                permitted = {}
                for degree_a, orbit_a, face_a, keys_a in parent_faces[owner_a]:
                    source_a = dict(
                        zip(
                            keys_a,
                            (source_bank[vertex] for vertex in face_a),
                            strict=True,
                        )
                    )
                    support = tuple(
                        vertex for vertex in first.vertices if vertex in source_a
                    )
                    if (
                        not support
                        or len(support) > first.dimension
                        or support in permitted
                    ):
                        continue
                    for face_b, _ in by_orbit[owner_b].get((degree_a, orbit_a), ()):
                        budget.reserve(len(face_b))
                        keys_b = identifiers(np.asarray(face_b, dtype=np.int64), image)
                        if set(keys_a) != set(keys_b):
                            continue
                        equal = True
                        for key, vertex in zip(keys_b, face_b, strict=True):
                            point = source_bank[vertex]
                            budget.reserve(len(point) * (len(point) + 1))
                            transformed = tuple(
                                row[-1]
                                + sum(
                                    (
                                        weight * value
                                        for weight, value in zip(
                                            row[:-1], point, strict=True
                                        )
                                    ),
                                    Fraction(0),
                                )
                                for row in action[:-1]
                            )
                            if transformed != source_a[key]:
                                equal = False
                                break
                        if equal:
                            budget.reserve(len(support), 128 + 64 * len(support))
                            permitted[support] = None
                            break
                for facet_root, source_plane, axes_, polygon in original_facets:
                    budget.reserve(0, 128 + 64 * len(first_source_vertices))
                    support_points = []
                    support_ids = []
                    for identifier, vertex in zip(
                        first.vertices, first_source_vertices, strict=True
                    ):
                        point = source_bank[int(vertex)]
                        budget.reserve(2 * len(point))
                        if sum(
                            (
                                a * b
                                for a, b in zip(source_plane[:-1], point, strict=True)
                            ),
                            source_plane[-1],
                        ):
                            continue
                        if _closed_source_polygon_contains(
                            polygon, project((point,), axes_)
                        ):
                            support_points.append(point)
                            support_ids.append(identifier)
                    support = tuple(support_ids)
                    if (
                        not support
                        or len(support) > first.dimension
                        or support in permitted
                    ):
                        continue
                    if not _closed_source_polygon_contains(
                        polygon, project(tuple(support_points), axes_)
                    ):
                        continue
                    for degree_b, _, face_b, _ in parent_faces[owner_b]:
                        if degree_b != mesh.topological_dimension - 1:
                            continue
                        authority_b = source_faces.get((owner_b, face_b))
                        if authority_b is None or facet_root not in authority_b[1]:
                            continue
                        budget.reserve(
                            len(face_b) * first.dimension * (first.dimension + 1)
                        )
                        transformed = tuple(
                            tuple(
                                row[-1]
                                + sum(
                                    (
                                        weight * value
                                        for weight, value in zip(
                                            row[:-1], source_bank[vertex], strict=True
                                        )
                                    ),
                                    Fraction(0),
                                )
                                for row in action[:-1]
                            )
                            for vertex in face_b
                        )
                        plane_b = _source_polygon_plane(transformed)
                        if plane_b is not None and plane_b[0] == source_plane:
                            budget.reserve(len(support), 128 + 64 * len(support))
                            permitted[support] = None
                            break
                return tuple(permitted)

            traces: dict[int, tuple[_Cell, ...]] = {}
            owner_pieces: dict[int, list[int]] = {}
            for piece, owner in enumerate(owners):
                owner_pieces.setdefault(owner, []).append(piece)
            for occurrence, entity in enumerate(entity_ids.tolist()):
                resource_kind, resource_ids = "facet", (int(entity),)
                row = facet_rows[entity]
                image = tuple(-int(value) for value in anchors[row])
                cell_row = int(facets.cells[occurrence])
                normalized = tuple(
                    _transform(
                        cells[piece],
                        _exact_periodic_element(matrices, orders, image),
                        identifiers(piece_vertices[piece], image),
                    )
                    for piece in owner_pieces[cell_row]
                )
                orbit = int(orbits[row])
                previous = traces.get(orbit)
                if previous is not None and any(
                    not _trace_equal(a, b) for a in previous for b in normalized
                ):
                    state.add(
                        "periodic_mapped_trace_mismatch", "violated", "facet", (entity,)
                    )
                traces[orbit] = normalized
            state.boundary_facets = int(
                np.count_nonzero(
                    np.asarray(
                        topology.quotient.entities(mesh.topological_dimension - 1)
                        .subset("boundary")
                        .mask
                    )
                )
            )
            resource_kind, resource_ids = "mesh", ()
            cell_ids = cell_ids[np.asarray(owners, dtype=np.int64)]
            identity = np.eye(mesh.topological_dimension, dtype=np.float64)
            origin = np.zeros(mesh.topological_dimension, dtype=np.float64)
            budget.reserve(
                finite_count * len(cells),
                finite_count * len(cells) * mesh.ambient_dimension * 16,
            )
            boxes = [
                _bounds(_transform(cell, matrix, cell.vertices), origin, identity)
                for _, matrix in finite
                for cell in cells
            ]
            if any(not np.all(np.isfinite(bound)) for box in boxes for bound in box):
                state.add("periodic_coordinate_bound_range", "unresolved", "mesh")
                return image_count
            ranges = _translation_ranges(boxes, translations)
            image_count = finite_count
            for interval in ranges:
                image_count *= len(interval)
            if image_count > maximum_images:
                state.add("periodic_image_budget", "unresolved", "mesh")
                return image_count
            base_boxes = [_bounds(cell, origin, identity) for cell in cells]
            vertices = piece_vertices
            inverse_pair_decisions: dict[tuple[int, int, tuple[int, ...]], str] = {}
            recorded_inverse_reuse = False
            for finite_exponents, _ in finite:
                for translation_exponents in product(*ranges):
                    iterator = iter(translation_exponents)
                    # Winding stays on the original authored generator axis:
                    # G^(r + L*k), never an independently authored translation.
                    exponents = tuple(
                        value + period * next(iterator) if period else value
                        for value, period in zip(finite_exponents, periods, strict=True)
                    )
                    transform_ = _exact_periodic_element(matrices, orders, exponents)
                    image_identity = not any(exponents)
                    for b, cell in enumerate(cells):
                        resource_kind, resource_ids = "cell", (int(cell_ids[b]),)
                        with budget.temporary_scope():
                            image = _transform(
                                cell, transform_, identifiers(vertices[b], exponents)
                            )
                            lower, upper = _bounds(image, origin, identity)
                            for a, first in enumerate(cells):
                                with budget.temporary_scope():
                                    if image_identity and (
                                        a >= b or owners[a] == owners[b]
                                    ):
                                        continue
                                    lo, hi = base_boxes[a]
                                    if np.any(hi < lower) or np.any(upper < lo):
                                        continue
                                    if (
                                        state.candidate_pairs
                                        >= limits.maximum_candidate_pairs
                                    ):
                                        state.add(
                                            "periodic_candidate_pair_budget",
                                            "unresolved",
                                            "cell",
                                            cell_ids[[a, b]],
                                        )
                                        return image_count
                                    state.candidate_pairs += 1
                                    resource_ids = (int(cell_ids[a]), int(cell_ids[b]))
                                    if not _trace_equal(first, image):
                                        state.add(
                                            "periodic_mapped_trace_mismatch",
                                            "violated",
                                            "cell",
                                            cell_ids[[a, b]],
                                        )
                                        continue
                                    if _tensor_pair(first, image, state, limits):
                                        continue
                                    root_a, root_b = (
                                        source_roots.get(owners[a]),
                                        source_roots.get(owners[b]),
                                    )
                                    if (
                                        root_a is not None
                                        and root_b is not None
                                        and root_a[0] == root_b[0]
                                        and root_a[4:] == root_b[4:]
                                        and root_a[1].coordinates == root_b[1].coordinates
                                    ):
                                        chart_b = _Cell(
                                            root_b[2].coordinates,
                                            root_b[2].domain,
                                            root_b[2].kind,
                                            root_b[2].dimension,
                                            image.vertices,
                                            root_b[2].reference_vertices,
                                        )
                                        if _root_action_pair(
                                            root_a[1],
                                            root_a[2],
                                            chart_b,
                                            transform_,
                                            root_a[3],
                                            state,
                                            limits,
                                        ):
                                            continue
                                    allowed_traces = authored_traces(
                                        first,
                                        owners[a],
                                        owners[b],
                                        transform_,
                                        exponents,
                                        piece_vertices[a],
                                    )
                                    reference_decision = None
                                    reference_first = layer_reference_cells.get(owners[a])
                                    reference_second = layer_reference_cells.get(
                                        owners[b]
                                    )
                                    if (
                                        reference_first is not None
                                        and reference_second is not None
                                    ):
                                        reference_image = _transform(
                                            reference_second, transform_, image.vertices
                                        )
                                        if _separating_plane(
                                            reference_first, reference_image
                                        ):
                                            continue
                                        reference_result = _affine_reference_contact(
                                            reference_first,
                                            reference_image,
                                            limits.maximum_subdivision_pieces
                                            - state.subdivision_pieces,
                                        )
                                        if reference_result is None:
                                            reference_decision, reference_work = (
                                                _pair_decision(
                                                    reference_first,
                                                    reference_image,
                                                    limits,
                                                    limits.maximum_subdivision_pieces
                                                    - state.subdivision_pieces,
                                                )
                                            )
                                        else:
                                            reference_decision, reference_work = (
                                                reference_result
                                            )
                                        state.subdivision_pieces += reference_work
                                        if reference_decision == "separated":
                                            continue
                                    key = (
                                        None
                                        if parent_faces
                                        else _inverse_equivalent_pair_key(
                                            a, b, exponents, orders
                                        )
                                    )
                                    inverse_decision = (
                                        None
                                        if key is None
                                        else inverse_pair_decisions.get(key)
                                    )
                                    reused_inverse = (
                                        reference_decision is None
                                        and inverse_decision is not None
                                    )
                                    decision = (
                                        reference_decision
                                        if reference_decision is not None
                                        else inverse_decision
                                    )
                                    if decision is None:
                                        if _separating_plane(first, image):
                                            decision, work = "separated", 0
                                        else:
                                            decision, work = _pair_decision(
                                                first,
                                                image,
                                                limits,
                                                limits.maximum_subdivision_pieces
                                                - state.subdivision_pieces,
                                                authoritative_first_traces=allowed_traces,
                                            )
                                        state.subdivision_pieces += work
                                        if key is not None and decision in (
                                            "overlap",
                                            "separated",
                                        ):
                                            inverse_pair_decisions[key] = decision
                                    elif reused_inverse and not recorded_inverse_reuse:
                                        state.checks.append(
                                            "periodic_inverse_pair_proof_reuse"
                                        )
                                        recorded_inverse_reuse = True
                                    if decision == "overlap":
                                        from ._mesh_certificates import (
                                            MeshCertificateFinding,
                                        )

                                        state.findings.append(
                                            MeshCertificateFinding(
                                                "periodic_mapped_cell_overlap",
                                                "violated",
                                                "cell",
                                                tuple(
                                                    int(value)
                                                    for value in cell_ids[[a, b]]
                                                ),
                                                observations=_source_contact_quantities(
                                                    first,
                                                    image,
                                                    transform_,
                                                    exponents,
                                                    a,
                                                    b,
                                                    owners[a],
                                                    owners[b],
                                                ),
                                            )
                                        )
                                    elif decision != "separated":
                                        state.add(
                                            decision,
                                            "unresolved",
                                            "cell",
                                            cell_ids[[a, b]],
                                        )
    except PeriodicIsometryIdentityError:
        state.add("periodic_authored_group_identity", "unresolved", "mesh")
    except RationalEnclosureError:
        state.add("periodic_rational_enclosure_premise", "unresolved", "mesh")
    except CoordinateEnclosureResourceError as error:
        state.add(
            "periodic_source_expression_resource_budget",
            "unresolved",
            resource_kind,
            resource_ids,
            resource_error=error,
            expression_budget=budget,
        )
    finally:
        state.source_expression_work_units = budget.work_units
        state.source_expression_peak_bytes = budget.peak_bytes_upper
    return image_count

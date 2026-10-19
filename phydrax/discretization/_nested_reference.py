#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#
"""Private preparation of scientific nested reference integration pairs."""

from __future__ import annotations

from collections.abc import Sequence
from fractions import Fraction
from typing import NamedTuple

import numpy as np
from jax.typing import ArrayLike
from numpy.typing import NDArray

import phydrax.discretization._cell_geometry_transition_types as transition_types_owner
import phydrax.discretization._mapped_geometry_measure as mapped_measure_owner
from phydrax.discretization._reference_cell_types import CellGeometryElement
from phydrax.linalg._exact import determinant_exact, solve_exact

from .._strict import StrictModule
from ..geometry._rational import rational_vector
from ..linalg import SmallLinearSolvePlan, solve_small_linear
from ..typing import checked
from . import (
    _cell_geometry as imported_cell_geometry,
    _coordinate_enclosure as imported_coordinate_enclosure,
    _nested_reference_types as reference_pair_types,
)
from ._cell_geometry import CellGeometrySpec
from ._cell_geometry_validity import cell_geometry_id
from ._cell_mesh import CellMesh
from ._coordinate_budget import CoordinateEnclosureBudget
from ._coordinate_enclosure import (
    affine_arguments,
    Polynomial,
    require_rational_reference_scalar,
)
from ._reference_cell import reference_cell_topology


class _ReferenceMap(NamedTuple):
    kind: str
    arguments: tuple[Polynomial, ...]


def _fraction_vertices(kind: str) -> tuple[tuple[Fraction, ...], ...]:
    return tuple(
        tuple(Fraction(value) for value in row)
        for row in reference_cell_topology(kind).vertices
    )


def _dot(first: Sequence[Fraction], second: Sequence[Fraction]) -> Fraction:
    return sum((a * b for a, b in zip(first, second, strict=True)), Fraction(0))


def _normal(
    points: Sequence[Sequence[Fraction]],
    center: Sequence[Fraction],
) -> tuple[Fraction, ...]:
    dimension = len(points[0])
    if dimension == 1:
        return (Fraction(1) if points[0][0] > center[0] else Fraction(-1),)
    first = tuple(b - a for a, b in zip(points[0], points[1], strict=True))
    if dimension == 2:
        normal: tuple[Fraction, ...] = (first[1], -first[0])
    else:
        second = tuple(b - a for a, b in zip(points[0], points[2], strict=True))
        normal = (
            first[1] * second[2] - first[2] * second[1],
            first[2] * second[0] - first[0] * second[2],
            first[0] * second[1] - first[1] * second[0],
        )
    outward = tuple(a - b for a, b in zip(points[0], center, strict=True))
    return normal if _dot(normal, outward) > 0 else tuple(-value for value in normal)


def _reference_measure(kind: str) -> Fraction:
    return {
        "interval": Fraction(1),
        "triangle": Fraction(1, 2),
        "quadrilateral": Fraction(1),
        "tetrahedron": Fraction(1, 6),
        "prism": Fraction(1, 2),
        "hexahedron": Fraction(1),
        "pyramid": Fraction(1, 3),
    }[kind]


def _reference_piece_measure(
    kind: str,
    arguments: tuple[Polynomial, ...],
    dimension: int,
    *,
    budget: CoordinateEnclosureBudget,
) -> Fraction:
    affine = all(sum(index) <= 1 for value in arguments for index in value)
    if affine:
        coefficients = tuple(
            tuple(
                value.get(
                    tuple(int(axis == column) for axis in range(dimension)),
                    Fraction(0),
                )
                for column in range(dimension)
            )
            for value in arguments
        )
        determinant = determinant_exact(coefficients)
        piece_measure = determinant * _reference_measure(kind)
    else:
        if (
            dimension != 2
            or kind != "quadrilateral"
            or any(
                any(exponent > 1 for exponent in index)
                for value in arguments
                for index in value
            )
        ):
            raise ValueError(
                "nested_reference.arguments must define bilinear quadrilateral charts for nonlinear partitions; got unsupported polynomial action"
            )
        jacobian = imported_coordinate_enclosure.determinant(
            tuple(
                tuple(
                    imported_coordinate_enclosure.derivative(value, axis, budget=budget)
                    for axis in range(2)
                )
                for value in arguments
            ),
            budget=budget,
        )
        determinant = min(
            imported_coordinate_enclosure.bernstein_coefficients(
                jacobian, "box", 2, budget=budget
            )
        )
        piece_measure = mapped_measure_owner.mapped_polynomial_integral_fraction(
            jacobian, kind, budget=budget
        )
    if determinant <= 0:
        raise ValueError(
            "nested_reference.jacobian must be positive on every fine cell; got nonpositive bound"
        )

    return require_rational_reference_scalar(piece_measure)


def _require_partition_facets(
    faces: dict[tuple[tuple[Fraction, ...], ...], list[tuple[Fraction, ...]]],
    planes: Sequence[tuple[tuple[Fraction, ...], Fraction]],
) -> None:
    for points, normals in faces.items():
        if len(normals) == 2:
            if _dot(normals[0], normals[1]) >= 0:
                raise ValueError(
                    "nested_reference.interior_facets must have opposite orientations; got inconsistent normals"
                )
        elif len(normals) == 1:
            if not any(
                all(_dot(normal, point) == bound for point in points)
                and _dot(normal, normals[0]) > 0
                for normal, bound in planes
            ):
                raise ValueError(
                    "nested_reference.boundary_facets must lie on the outward parent boundary; got unpaired interior facet"
                )
        else:
            raise ValueError(
                "nested_reference.facets must have manifold multiplicity; got overlapping facets"
            )


def _require_reference_partition(
    coarse_kind: str,
    fine_maps: Sequence[_ReferenceMap],
    *,
    budget: CoordinateEnclosureBudget,
) -> None:
    """Prove an oriented algebraic reference tiling, rather than sampling it.

    Positive fine cells are contained in the convex parent. Every interior
    polygon is paired with its opposite orientation, and unpaired polygons
    lie on the parent boundary. The closed outward boundary chain therefore
    has constant nonnegative integer multiplicity on that boundary; the algebraic
    reference-volume equality fixes it to one. Its winding number is one in
    the parent and zero outside, so positive cells cannot leave gaps/overlap.
    """
    vertices = _fraction_vertices(coarse_kind)
    dimension = len(vertices[0])
    center = tuple(
        sum((row[axis] for row in vertices), Fraction(0)) / len(vertices)
        for axis in range(dimension)
    )
    planes: list[tuple[tuple[Fraction, ...], Fraction]] = []
    for face in reference_cell_topology(coarse_kind).entities[dimension - 1]:
        points = tuple(vertices[index] for index in face)
        normal = _normal(points, center)
        planes.append((normal, _dot(normal, points[0])))
    faces: dict[tuple[tuple[Fraction, ...], ...], list[tuple[Fraction, ...]]] = {}
    measure = Fraction(0)

    for kind, arguments in fine_maps:
        if len(arguments) != dimension:
            raise ValueError(
                "nested_reference.dimension must agree for every fine map; got incompatible argument count"
            )
        piece_measure = _reference_piece_measure(
            kind, arguments, dimension, budget=budget
        )
        fine = tuple(
            tuple(
                require_rational_reference_scalar(
                    imported_coordinate_enclosure.evaluate(value, vertex, budget=budget)
                )
                for value in arguments
            )
            for vertex in _fraction_vertices(kind)
        )
        if any(_dot(normal, point) > bound for point in fine for normal, bound in planes):
            raise ValueError(
                "nested_reference.corners must lie inside the parent reference cell; got exterior corner"
            )
        fine_center = tuple(
            sum((row[axis] for row in fine), Fraction(0)) / len(fine)
            for axis in range(dimension)
        )
        for face in reference_cell_topology(kind).entities[dimension - 1]:
            points = tuple(fine[index] for index in face)
            faces.setdefault(tuple(sorted(points)), []).append(
                _normal(points, fine_center)
            )
        measure += piece_measure
    if measure != _reference_measure(coarse_kind):
        raise ValueError(
            "nested_reference.partition_measure must equal the parent measure; got unequal rational measures"
        )
    _require_partition_facets(faces, planes)


class _ReferenceRecord(NamedTuple):
    source: int
    target: int
    fine_is_source: bool
    corners: NDArray[np.float64] | None


class _ReferenceRecords(StrictModule):
    records: tuple[_ReferenceRecord, ...]
    source_kinds: tuple[str, ...]
    target_kinds: tuple[str, ...]


def _transition_reference_records(
    transition: transition_types_owner.CellGeometryTransition,
    source_index: dict[int, int],
    target_index: dict[int, int],
    *,
    nonlinear: bool,
) -> tuple[_ReferenceRecord, ...]:
    records: list[_ReferenceRecord] = []
    for target_id, source_id, corners in zip(
        np.asarray(transition.target_cell_ids),
        np.asarray(transition.parent_cell_ids),
        np.asarray(transition.parent_reference_vertices),
        strict=True,
    ):
        include = int(source_id) >= 0 if nonlinear else source_id >= 0
        if (not nonlinear or include) and int(target_id) not in target_index:
            raise ValueError(
                "nested_reference.target_cell must name a present scientific cell; got absent ID"
            )
        if include:
            if int(source_id) not in source_index:
                raise ValueError(
                    "nested_reference.source_cell must name a present scientific cell; got absent ID"
                )
            records.append(
                _ReferenceRecord(
                    source_index[int(source_id)],
                    target_index[int(target_id)],
                    False,
                    corners,
                )
            )
    for source_id, target_id, corners in zip(
        np.asarray(transition.coarsened_cell_ids),
        np.asarray(transition.coarsened_into_ids),
        np.asarray(transition.coarsened_reference_vertices),
        strict=True,
    ):
        if int(source_id) not in source_index or int(target_id) not in target_index:
            raise ValueError(
                "nested_reference.coarsening must name present scientific cells; got absent ID"
            )
        records.append(
            _ReferenceRecord(
                source_index[int(source_id)], target_index[int(target_id)], True, corners
            )
        )
    return tuple(records)


def _coarsening_reference_records(
    witnesses: transition_types_owner.NestedReferenceWitnesses,
    source_index: dict[int, int],
    target_index: dict[int, int],
    *,
    nonlinear: bool,
) -> tuple[_ReferenceRecord, ...]:
    fine_ids = (
        np.asarray(witnesses.fine_cell_ids)
        if nonlinear
        else np.asarray(witnesses.fine_cell_ids, dtype=np.int64)
    )
    if not nonlinear and np.unique(fine_ids).size != fine_ids.size:
        raise ValueError(
            "nested_reference.coarsening must name each source cell once; got repeated ID"
        )
    records: list[_ReferenceRecord] = []
    for source_id, target_id, corners in zip(
        fine_ids,
        np.asarray(witnesses.coarse_cell_ids),
        np.asarray(witnesses.fine_reference_vertices),
        strict=True,
    ):
        if int(source_id) not in source_index or int(target_id) not in target_index:
            raise ValueError(
                "nested_reference.coarsening must name present scientific cells; got absent ID"
            )
        records.append(
            _ReferenceRecord(
                source_index[int(source_id)],
                target_index[int(target_id)],
                True,
                np.asarray(corners),
            )
        )
    return tuple(records)


def _refinement_reference_records(
    parents_: ArrayLike | None,
    corners_: ArrayLike | None,
    source_count: int,
    target_count: int,
    dimension: int,
    *,
    nonlinear: bool,
) -> tuple[_ReferenceRecord, ...]:
    if parents_ is None or (not nonlinear and corners_ is None):
        raise ValueError(
            "nested_reference.refinement must provide parent cells and affine reference corners; got missing input"
        )
    parents = np.asarray(parents_)
    if parents.shape != (target_count,) or not np.issubdtype(parents.dtype, np.integer):
        raise ValueError(
            "nested_reference.parent_cells must contain one integer source parent per target cell; got incompatible array"
        )
    if np.any(parents < 0) or np.any(parents >= source_count):
        raise ValueError(
            "nested_reference.parent_cells must index present source cells; got out-of-range index"
        )
    corners = None if corners_ is None else np.asarray(corners_, dtype=np.float64)
    if corners is not None and (
        corners.ndim != 3
        or corners.shape[0] != target_count
        or corners.shape[2] != dimension
    ):
        raise ValueError(
            "nested_reference.parent_reference_vertices must align with target cells and reference dimension; got incompatible array"
        )
    return tuple(
        _ReferenceRecord(
            int(parent), index, False, None if corners is None else corners[index]
        )
        for index, parent in enumerate(parents)
    )


def _reference_records(
    source_mesh: CellMesh,
    source_geometry: CellGeometrySpec,
    target_mesh: CellMesh,
    target_geometry: CellGeometrySpec,
    parent_cells: ArrayLike | None,
    parent_reference_vertices: ArrayLike | None,
    geometry_transition: transition_types_owner.CellGeometryTransition | None,
    coarsening_witnesses: transition_types_owner.NestedReferenceWitnesses | None,
    *,
    nonlinear: bool,
) -> _ReferenceRecords:
    source_ids = np.concatenate(
        [np.asarray(block.global_ids, dtype=np.int64) for block in source_mesh.blocks]
    )
    target_ids = np.concatenate(
        [np.asarray(block.global_ids, dtype=np.int64) for block in target_mesh.blocks]
    )
    source_index = {int(value): index for index, value in enumerate(source_ids)}
    target_index = {int(value): index for index, value in enumerate(target_ids)}
    source_kinds = tuple(
        block.cell_kind for block in source_mesh.blocks for _ in range(block.cell_count)
    )
    target_kinds = tuple(
        block.cell_kind for block in target_mesh.blocks for _ in range(block.cell_count)
    )
    if source_mesh.topological_dimension != target_mesh.topological_dimension:
        raise ValueError(
            "nested_reference.dimension must agree at both endpoints; got different dimensions"
        )
    if geometry_transition is not None:
        if (
            geometry_transition.source_topology_id != source_mesh.topology_id
            or geometry_transition.target_topology_id != target_mesh.topology_id
            or geometry_transition.source_geometry_id != cell_geometry_id(source_geometry)
            or geometry_transition.target_geometry_id != cell_geometry_id(target_geometry)
        ):
            raise ValueError(
                "nested_reference.geometry_transition must bind both endpoint maps; got mismatched geometry or topology ID"
            )
        records = _transition_reference_records(
            geometry_transition, source_index, target_index, nonlinear=nonlinear
        )
    elif coarsening_witnesses is not None:
        records = _coarsening_reference_records(
            coarsening_witnesses, source_index, target_index, nonlinear=nonlinear
        )
    else:
        records = _refinement_reference_records(
            parent_cells,
            parent_reference_vertices,
            source_ids.size,
            target_ids.size,
            source_mesh.topological_dimension,
            nonlinear=nonlinear,
        )
    if not records and not nonlinear:
        raise ValueError(
            "nested_reference.records must contain integration pieces; got empty reference records"
        )
    if {record.source for record in records} != set(range(source_ids.size)) or {
        record.target for record in records
    } != set(range(target_ids.size)):
        raise ValueError(
            "nested_reference.records must cover source and target cells; got uncovered cells"
        )
    return _ReferenceRecords(records, source_kinds, target_kinds)


class _AffineReferenceGroup(StrictModule):
    indices: tuple[int, ...]
    pairs: tuple[reference_pair_types.NestedReferencePair, ...]


def _solve_affine_reference_group(
    kind: str,
    entries: list[tuple[int, NDArray[np.float64]]],
    records: tuple[_ReferenceRecord, ...],
    frame_indices: dict[str, tuple[int, ...]],
    dimension: int,
) -> _AffineReferenceGroup:
    pairs: list[reference_pair_types.NestedReferencePair] = []
    vertices = np.asarray(reference_cell_topology(kind).vertices, dtype=np.float64)
    corners = np.stack([row for _, row in entries])
    selected = list(frame_indices[kind])
    frame = (vertices[selected] - vertices[0]).T
    images = np.swapaxes(corners[:, selected] - corners[:, :1], -1, -2)
    solved = solve_small_linear(
        SmallLinearSolvePlan(dimension),
        np.broadcast_to(frame.T, (len(entries), dimension, dimension)),
        np.swapaxes(images, -1, -2),
    )
    if not np.all(np.asarray(solved.successful)):
        raise ValueError(
            "nested_reference.affine_frame must be nonsingular; got failed linear solve"
        )
    matrices = np.swapaxes(np.asarray(solved.value, dtype=np.float64), -1, -2)
    offsets = corners[:, 0] - matrices @ vertices[0]
    reconstructed = vertices[None] @ np.swapaxes(matrices, -1, -2) + offsets[:, None]
    if not np.array_equal(reconstructed, corners):
        raise ValueError(
            "nested_reference.corners must equal the represented affine map; got unequal reconstructed corners"
        )
    for row, (index, _) in enumerate(entries):
        source, target, fine_is_source, _ = records[index]
        pairs.append(
            reference_pair_types.NestedReferencePair(
                source, target, fine_is_source, matrices[row], offsets[row]
            )
        )

    return _AffineReferenceGroup(tuple(index for index, _ in entries), tuple(pairs))


def _affine_reference_pairs(
    prepared: _ReferenceRecords,
    dimension: int,
    *,
    budget: CoordinateEnclosureBudget,
) -> tuple[reference_pair_types.NestedReferencePair, ...]:
    records, source_kinds, target_kinds = (
        prepared.records,
        prepared.source_kinds,
        prepared.target_kinds,
    )
    frame_indices = {
        "interval": (1,),
        "triangle": (1, 2),
        "quadrilateral": (1, 3),
        "tetrahedron": (1, 2, 3),
        "prism": (1, 2, 3),
        "hexahedron": (1, 3, 4),
        "pyramid": (1, 3, 4),
    }
    built: list[reference_pair_types.NestedReferencePair | None] = [None] * len(records)
    groups: dict[str, list[tuple[int, NDArray[np.float64]]]] = {}
    for index, (source, target, fine_is_source, corners) in enumerate(records):
        kind = source_kinds[source] if fine_is_source else target_kinds[target]
        if corners is None:
            raise ValueError(
                "nested_reference.corners must provide an affine action; got missing corners"
            )
        if kind not in frame_indices or not np.all(np.isfinite(corners)):
            raise ValueError(
                "nested_reference.corners must define a supported finite map; got unsupported kind or nonfinite values"
            )
        arity = len(reference_cell_topology(kind).vertices)
        if corners.shape[0] < arity or corners.shape[1] != dimension:
            raise ValueError(
                "nested_reference.corners must contain every fine-cell corner; got incompatible array"
            )
        groups.setdefault(kind, []).append((index, np.asarray(corners[:arity])))
    for kind, entries in groups.items():
        solved = _solve_affine_reference_group(
            kind, entries, records, frame_indices, dimension
        )
        for index, pair in zip(solved.indices, solved.pairs, strict=True):
            built[index] = pair
    output: list[reference_pair_types.NestedReferencePair] = []
    for pair in built:
        if pair is None:
            raise ValueError(
                "nested_reference.affine_frame must produce every integration piece; got missing map"
            )
        output.append(pair)
    # Group numeric witnesses first; only one parent's algebraic proof bank is live.
    partitions: dict[
        tuple[bool, int], list[tuple[str, reference_pair_types.NestedReferencePair]]
    ] = {}
    for pair in output:
        coarse = (
            pair.fine_is_source,
            pair.target_cell if pair.fine_is_source else pair.source_cell,
        )
        fine_kind = (
            source_kinds[pair.source_cell]
            if pair.fine_is_source
            else target_kinds[pair.target_cell]
        )
        partitions.setdefault(coarse, []).append((fine_kind, pair))
    for (fine_is_source, index), witnesses in partitions.items():
        coarse_kind = target_kinds[index] if fine_is_source else source_kinds[index]
        with budget.temporary_scope():
            maps = [
                _ReferenceMap(
                    kind, affine_arguments(pair.offset, pair.matrix, budget=budget)
                )
                for kind, pair in witnesses
            ]
            _require_reference_partition(coarse_kind, maps, budget=budget)
    return tuple(output)


@checked
def nested_reference_pairs(
    source_mesh: CellMesh,
    source_geometry: CellGeometrySpec,
    target_mesh: CellMesh,
    target_geometry: CellGeometrySpec,
    /,
    *,
    parent_cells: ArrayLike | None = None,
    parent_reference_vertices: ArrayLike | None = None,
    geometry_transition: transition_types_owner.CellGeometryTransition | None = None,
    coarsening_witnesses: transition_types_owner.NestedReferenceWitnesses | None = None,
    budget: CoordinateEnclosureBudget,
) -> tuple[reference_pair_types.NestedReferencePair, ...]:
    """Bind immediate refinement/coarsening witnesses to geometry IDs.

    Each matrix/offset maps the finer reference cell into the coarser one.
    The field/volume owner additionally proves equality of the represented
    maps under that composition; this preparation never treats root-source
    coefficients as target nodal coordinates.
    """
    source_geometry.resolve(source_mesh, budget=budget)
    target_geometry.resolve(target_mesh, budget=budget)
    prepared = _reference_records(
        source_mesh,
        source_geometry,
        target_mesh,
        target_geometry,
        parent_cells,
        parent_reference_vertices,
        geometry_transition,
        coarsening_witnesses,
        nonlinear=False,
    )
    return _affine_reference_pairs(
        prepared, source_mesh.topological_dimension, budget=budget
    )


def _require_polynomial_reference_action(
    coarse_arguments: tuple[Polynomial, ...],
    fine_arguments: tuple[Polynomial, ...],
    arguments: tuple[Polynomial, ...],
    *,
    budget: CoordinateEnclosureBudget,
) -> None:
    """Prove equality of the reference laws coefficient by coefficient.

    Equal retained-root identities and coefficient banks then imply equal
    physical maps under this action, including for nonlinear coarse charts.
    The certified physical sup-norm difference is therefore zero without
    sampling or expanding the physical coordinate map.
    """
    if any(
        imported_coordinate_enclosure.compose(value, arguments, budget=budget) != expected
        for value, expected in zip(coarse_arguments, fine_arguments, strict=True)
    ):
        raise ValueError(
            "nested_reference.arguments must compose to the fine reference law; got unequal polynomial coefficients"
        )


def _polynomial_reference_pair(
    record: _ReferenceRecord,
    source_cells: tuple[
        tuple[CellGeometryElement, imported_coordinate_enclosure.CoordinateSourceBank],
        ...,
    ],
    target_cells: tuple[
        tuple[CellGeometryElement, imported_coordinate_enclosure.CoordinateSourceBank],
        ...,
    ],
    source_roots: tuple[tuple[str, str, int, tuple[int, ...]], ...],
    target_roots: tuple[tuple[str, str, int, tuple[int, ...]], ...],
    *,
    budget: CoordinateEnclosureBudget,
) -> reference_pair_types.PolynomialReferencePair:
    source_index, target_index, fine_is_source, witness = record
    source_element, source_bank = source_cells[source_index]
    target_element, target_bank = target_cells[target_index]
    if (
        source_roots[source_index] != target_roots[target_index]
        or source_bank != target_bank
    ):
        raise ValueError(
            "nested_reference.source_root must retain scientific identity and source coefficients; got stale roots or changed coefficient bank"
        )
    coarse, fine = (
        (target_element, source_element)
        if fine_is_source
        else (source_element, target_element)
    )
    arguments = _polynomial_reference_arguments(coarse, fine, budget=budget)
    vertices = reference_cell_topology(fine.cell_kind).vertices
    if witness is not None:
        exact = np.asarray(
            [
                imported_coordinate_enclosure.rounded_point(
                    tuple(
                        imported_coordinate_enclosure.evaluate(
                            value,
                            rational_vector(vertex),
                            budget=budget,
                        )
                        for value in arguments
                    )
                )
                for vertex in vertices
            ],
            dtype=np.float64,
        )
        if (
            witness.ndim != 2
            or witness.shape[0] < len(vertices)
            or witness.shape[1] != 2
            or not np.array_equal(exact, witness[: len(vertices)])
        ):
            raise ValueError(
                "nested_reference.corners must equal the rounded polynomial action; got incompatible reference record"
            )
    jacobian = imported_coordinate_enclosure.determinant(
        tuple(
            tuple(
                imported_coordinate_enclosure.derivative(value, axis, budget=budget)
                for axis in range(2)
            )
            for value in arguments
        ),
        budget=budget,
    )
    controls = imported_coordinate_enclosure.bernstein_coefficients(
        jacobian,
        "box" if fine.cell_kind == "quadrilateral" else "simplex",
        2,
        budget=budget,
    )
    return reference_pair_types.PolynomialReferencePair(
        source_index,
        target_index,
        fine_is_source,
        arguments,
        source_element.cell_kind,
        target_element.cell_kind,
        (
            require_rational_reference_scalar(min(controls)),
            require_rational_reference_scalar(max(controls)),
        ),
    )


def _polynomial_reference_arguments(
    coarse: CellGeometryElement,
    fine: CellGeometryElement,
    *,
    budget: CoordinateEnclosureBudget,
) -> tuple[Polynomial, ...]:

    coarse_root, coarse_expressions = (
        imported_coordinate_enclosure.coordinate_reference_chain(coarse, budget=budget)
    )
    fine_root, fine_expressions = (
        imported_coordinate_enclosure.coordinate_reference_chain(fine, budget=budget)
    )
    if any(
        isinstance(value, imported_coordinate_enclosure.RationalPolynomial)
        for value in (*coarse_expressions, *fine_expressions)
    ):
        raise ValueError(
            "nested_reference.arguments must be polynomial; got rational quotient"
        )
    coarse_arguments = tuple(
        value
        for value in coarse_expressions
        if not isinstance(value, imported_coordinate_enclosure.RationalPolynomial)
    )
    fine_arguments = tuple(
        value
        for value in fine_expressions
        if not isinstance(value, imported_coordinate_enclosure.RationalPolynomial)
    )
    if (
        coarse_root.element_id != fine_root.element_id
        or coarse_root.topological_dimension != 2
        or imported_coordinate_enclosure.coordinate_source_signature(coarse_root)
        != imported_coordinate_enclosure.coordinate_source_signature(fine_root)
    ):
        raise ValueError(
            "nested_reference.source_root must bind one two-dimensional coordinate source; got different root or source signature"
        )
    if any(sum(index) > 1 for value in coarse_arguments for index in value):
        _, expressions = imported_coordinate_enclosure.coordinate_reference_chain(
            fine, ancestor=coarse, budget=budget
        )
        if any(
            isinstance(value, imported_coordinate_enclosure.RationalPolynomial)
            for value in expressions
        ):
            raise ValueError(
                "nested_reference.arguments must be polynomial; got rational quotient"
            )
        arguments = tuple(
            value
            for value in expressions
            if not isinstance(value, imported_coordinate_enclosure.RationalPolynomial)
        )
        _require_polynomial_reference_action(
            coarse_arguments, fine_arguments, arguments, budget=budget
        )
        return arguments
    matrix = [
        [
            value.get(tuple(int(axis == column) for axis in range(2)), Fraction(0))
            for column in range(2)
        ]
        for value in coarse_arguments
    ]
    differences = tuple(
        imported_coordinate_enclosure.add(
            value,
            imported_coordinate_enclosure.constant(
                -old.get((0, 0), Fraction(0)), 2, budget=budget
            ),
            budget=budget,
        )
        for value, old in zip(fine_arguments, coarse_arguments, strict=True)
    )
    indices = sorted({index for value in differences for index in value})
    solved = solve_exact(
        matrix,
        [[value.get(index, Fraction(0)) for index in indices] for value in differences],
    )
    arguments: tuple[Polynomial, ...] = tuple(
        {
            index: coefficient
            for index, coefficient in zip(indices, row, strict=True)
            if coefficient
        }
        for row in solved
    )
    _require_polynomial_reference_action(
        coarse_arguments, fine_arguments, arguments, budget=budget
    )
    return arguments


def _has_nonlinear_reference_action(
    element: CellGeometryElement, *, budget: CoordinateEnclosureBudget
) -> bool:
    """Classify the full authored reference law, not its Python owner kind."""

    if imported_coordinate_enclosure.has_cartesian_reference_chain(element):
        # A chain of physical affine restrictions remains an affine reference
        # action. In particular, the collapsed-cube chart makes an
        # affine pyramid restriction look polynomial; that chart is the root
        # parameterization, not a nonlinear authored action.
        current = element
        while isinstance(current, imported_cell_geometry.RestrictedCellGeometryElement):
            current = current.source_element
        if not isinstance(
            current,
            (
                imported_cell_geometry.PolynomialComposedCellGeometryElement,
                imported_cell_geometry.RationalComposedCellGeometryElement,
            ),
        ):
            return False
        _, arguments = imported_coordinate_enclosure.coordinate_reference_chain(
            element, budget=budget
        )
    else:
        # Full coefficient actions are not Cartesian ancestry. Their existing
        # nested route is still checked against the physical maps.
        arguments = imported_coordinate_enclosure.source_expressions(
            element, budget=budget
        )
        if arguments is None:
            raise ValueError(
                "nested_reference.source_expressions must provide the scalar reference law; got absent source expressions"
            )
    for value in arguments:
        numerator, denominator = imported_coordinate_enclosure.expression_parts(
            value,
            reference_cell_topology(element.cell_kind).dimension,
            budget=budget,
        )
        if any(sum(index) != 0 for index in denominator) or not denominator:
            return True
        if any(sum(index) > 1 for index in numerator):
            return True
    return False


@checked
def rooted_nested_reference_pairs(
    source: CellMesh,
    source_geometry: CellGeometrySpec,
    target: CellMesh,
    target_geometry: CellGeometrySpec,
    *,
    parent_cells: ArrayLike | None = None,
    parent_reference_vertices: ArrayLike | None = None,
    geometry_transition: transition_types_owner.CellGeometryTransition | None = None,
    coarsening_witnesses: transition_types_owner.NestedReferenceWitnesses | None = None,
    budget: CoordinateEnclosureBudget,
) -> tuple[
    tuple[
        reference_pair_types.NestedReferencePair
        | reference_pair_types.PolynomialReferencePair,
        ...,
    ],
    float,
]:
    """Bind polynomial fine actions and their oriented reference tiling."""
    # Reference semantics belong to each block element; physical cell
    # coefficient banks are authenticated by the whole-map proof below.

    source_cells, target_cells = (
        mapped_measure_owner.mapped_geometry_cells(
            source, source_geometry, budget=budget
        ),
        mapped_measure_owner.mapped_geometry_cells(
            target, target_geometry, budget=budget
        ),
    )
    nonlinear = any(
        _has_nonlinear_reference_action(element, budget=budget)
        for element in target_geometry.elements
    ) or (
        (coarsening_witnesses is not None or geometry_transition is not None)
        and any(
            _has_nonlinear_reference_action(element, budget=budget)
            for element in source_geometry.elements
        )
    )
    if not nonlinear or (
        source.topology_id == target.topology_id
        and cell_geometry_id(source_geometry) == cell_geometry_id(target_geometry)
    ):
        pairs = nested_reference_pairs(
            source,
            source_geometry,
            target,
            target_geometry,
            parent_cells=parent_cells,
            parent_reference_vertices=parent_reference_vertices,
            geometry_transition=geometry_transition,
            coarsening_witnesses=coarsening_witnesses,
            budget=budget,
        )
        return pairs, mapped_measure_owner.certify_nested_geometry_pairs(
            source,
            source_geometry,
            target,
            target_geometry,
            pairs,
            budget=budget,
        )
    if (
        source.topological_dimension != 2
        or target.topological_dimension != 2
        or source.ambient_dimension != target.ambient_dimension
    ):
        raise ValueError(
            "nested_reference.dimension must define one common surface coordinate space; got incompatible dimensions"
        )
    prepared = _reference_records(
        source,
        source_geometry,
        target,
        target_geometry,
        parent_cells,
        parent_reference_vertices,
        geometry_transition,
        coarsening_witnesses,
        nonlinear=True,
    )
    records = prepared.records
    source_roots, target_roots = (
        mapped_measure_owner.mapped_root_identities(source, source_geometry),
        mapped_measure_owner.mapped_root_identities(target, target_geometry),
    )
    built: list[reference_pair_types.PolynomialReferencePair] = []
    partitions: dict[tuple[bool, int], list[_ReferenceMap]] = {}
    for source_index, target_index, fine_is_source, witness in records:
        pair = _polynomial_reference_pair(
            _ReferenceRecord(source_index, target_index, fine_is_source, witness),
            source_cells,
            target_cells,
            source_roots,
            target_roots,
            budget=budget,
        )
        built.append(pair)
        partitions.setdefault(
            (fine_is_source, target_index if fine_is_source else source_index), []
        ).append(
            _ReferenceMap(
                pair.source_kind if fine_is_source else pair.target_kind, pair.arguments
            )
        )
    for (fine_is_source, parent), maps in partitions.items():
        _require_reference_partition(
            (target_cells if fine_is_source else source_cells)[parent][0].cell_kind,
            maps,
            budget=budget,
        )
    # Every pair proves coarse_reference(action) == fine_reference over the
    # entire cell, with identical physical source coefficients and root IDs.
    # This algebraic equality certifies zero physical-map deviation.
    return tuple(built), 0.0
